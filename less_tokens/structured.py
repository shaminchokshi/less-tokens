"""
less_tokens.structured — zone-aware compression for prompts with parts that
must not be compressed (output formats, schemas) or must be compressed gently
(rules, constraints).

The core idea: a real prompt is not uniform. It has:

  * an instruction body that can be compressed freely,
  * rules / constraints that must keep their exact meaning (compress gently),
  * output-format specs, JSON schemas, examples that must survive verbatim.

:func:`compress_structured` lets you assign a compression *level* to each zone:

  * "free"      -> full aggressive compression (your chosen flags)
  * "careful"   -> only the safest, meaning-preserving techniques
  * "protected" -> returned byte-for-byte unchanged

Async variants :func:`acompress` and :func:`acompress_structured` run the
(CPU-bound, pure-Python) compression in a thread executor so they don't block
an asyncio event loop.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .compressor import compress, _coerce_flag


# ---------------------------------------------------------------------------
# Compression levels
# ---------------------------------------------------------------------------
# "careful" deliberately uses only techniques that cannot change meaning:
#   - filler phrase removal (hedging only)
#   - contractions ("do not" -> "don't")
#   - filler word removal ("basically", "really")
#   - whitespace normalisation
#   - named-entity protection ON
# It deliberately does NOT remove stopwords, prune function words, do POS-keep,
# lemmatize, or shorten synonyms, because each of those can subtly alter the
# meaning of a rule or constraint.

CAREFUL_FLAGS: Dict[str, int] = {
    "remove_filler_phrases":      1,
    "apply_abbreviations":        0,   # abbreviations can be ambiguous in rules
    "apply_contractions":         1,
    "remove_filler_words":        1,
    "remove_stopwords":           0,
    "remove_function_words":      0,
    "pos_keep_only":              0,
    "lemmatize":                  0,
    "shorten_synonyms":           0,
    "preserve_named_entities":    1,
    "normalize_whitespace_punct": 1,
}

# "free" default if the caller doesn't pass their own flags: a balanced preset.
FREE_DEFAULT_FLAGS: Dict[str, int] = {
    "remove_filler_phrases":      1,
    "apply_abbreviations":        1,
    "apply_contractions":         1,
    "remove_filler_words":        1,
    "remove_stopwords":           1,
    "remove_function_words":      0,
    "pos_keep_only":              0,
    "lemmatize":                  0,
    "shorten_synonyms":           0,
    "preserve_named_entities":    1,
    "normalize_whitespace_punct": 1,
}

VALID_LEVELS = ("free", "careful", "protected")


# ---------------------------------------------------------------------------
# A "zone" is a piece of the prompt with a compression level.
# ---------------------------------------------------------------------------
# Users can pass zones as dicts:
#     {"text": "...", "level": "free"}
# or as (text, level) tuples. We normalise both.

Zone = Union[Dict[str, str], Tuple[str, str]]


def _normalise_zone(zone: Zone) -> Tuple[str, str]:
    if isinstance(zone, dict):
        text = zone.get("text", "")
        level = zone.get("level", "free")
    elif isinstance(zone, (tuple, list)) and len(zone) == 2:
        text, level = zone
    else:
        raise TypeError(
            "Each zone must be a dict {'text':..., 'level':...} or a "
            f"(text, level) tuple, got {type(zone).__name__}"
        )
    if level not in VALID_LEVELS:
        raise ValueError(
            f"level must be one of {VALID_LEVELS}, got {level!r}"
        )
    if not isinstance(text, str):
        raise TypeError(f"zone text must be str, got {type(text).__name__}")
    return text, level


# ---------------------------------------------------------------------------
# Synchronous structured compression
# ---------------------------------------------------------------------------

def compress_structured(
    zones: Optional[Sequence[Zone]] = None,
    *,
    instruction: Optional[str] = None,
    rules: Optional[Union[str, Sequence[str]]] = None,
    output_format: Optional[str] = None,
    examples: Optional[Union[str, Sequence[str]]] = None,
    free_flags: Optional[Dict[str, int]] = None,
    separator: str = "\n\n",
    return_detail: bool = False,
    **free_flag_kwargs,
) -> Union[str, Dict]:
    """
    Compress a multi-part prompt, applying a different compression level to each
    part. Parts that hold an exact output format or schema are returned
    verbatim; rules are compressed gently; the instruction body is compressed
    freely.

    There are two ways to call this:

    **A) The convenience form** (most common). Pass named sections:

    >>> compress_structured(
    ...     instruction="I was wondering if you could analyse this customer "
    ...                 "review and tell me how the person is feeling.",
    ...     rules="Do not include any personal opinions. Never guess if unsure.",
    ...     output_format='{"sentiment": "positive|negative|neutral", '
    ...                   '"confidence": 0.0-1.0}',
    ...     remove_stopwords=1, remove_filler_phrases=1,
    ... )

    Here:
      * ``instruction`` is compressed FREELY using the flags you pass.
      * ``rules`` are compressed CAREFULLY (safe techniques only).
      * ``output_format`` is PROTECTED (returned exactly).
      * ``examples`` are PROTECTED (returned exactly).

    **B) The explicit form.** Pass an ordered list of zones, each tagged with a
    level, for full control over ordering and mixing:

    >>> compress_structured(zones=[
    ...     {"text": "Analyse the following review.", "level": "free"},
    ...     {"text": "Never fabricate information.",   "level": "careful"},
    ...     {"text": '{"sentiment": "..."}',           "level": "protected"},
    ... ])

    Parameters
    ----------
    zones
        Explicit ordered list of zones (form B). If given, the named-section
        arguments below are ignored.
    instruction
        The free-compression instruction body (form A).
    rules
        A rule string, or list of rule strings, compressed carefully (form A).
    output_format
        The output format / schema, returned verbatim (form A).
    examples
        Example(s), returned verbatim (form A).
    free_flags
        A dict of flags to use for "free" zones. If omitted, any
        ``**free_flag_kwargs`` you pass are used; if none are passed, a
        balanced default preset is used.
    separator
        String inserted between assembled zones. Default is a blank line.
    return_detail
        If True, return a dict with the assembled prompt plus per-zone detail.
    **free_flag_kwargs
        Compression flags (e.g. ``remove_stopwords=1``) applied to free zones.

    Returns
    -------
    str
        The reassembled, partially-compressed prompt (default).
    dict
        If ``return_detail=True``: ``{"compressed", "zones": [...]}`` with the
        per-zone before/after text and the level applied to each.

    Notes
    -----
    Rules are compressed with a deliberately conservative set of techniques
    (no stopword removal, no POS pruning, no synonym substitution) because
    those transforms can subtly change the meaning of a constraint. If you need
    a rule preserved *exactly*, mark it ``protected`` via the explicit zone
    form.
    """
    # Resolve the flags used for "free" zones.
    if free_flags is not None:
        resolved_free = {k: _coerce_flag(v) for k, v in free_flags.items()}
    elif free_flag_kwargs:
        resolved_free = {k: _coerce_flag(v) for k, v in free_flag_kwargs.items()}
    else:
        resolved_free = dict(FREE_DEFAULT_FLAGS)

    # Build the ordered list of (text, level) zones.
    resolved_zones: List[Tuple[str, str]] = []

    if zones is not None:
        # Explicit form B.
        for z in zones:
            resolved_zones.append(_normalise_zone(z))
    else:
        # Convenience form A: instruction -> rules -> examples -> output_format.
        if instruction:
            resolved_zones.append((instruction, "free"))
        if rules:
            if isinstance(rules, str):
                resolved_zones.append((rules, "careful"))
            else:
                # join multiple rules into one careful zone, one per line
                resolved_zones.append(("\n".join(rules), "careful"))
        if examples:
            if isinstance(examples, str):
                resolved_zones.append((examples, "protected"))
            else:
                resolved_zones.append(("\n".join(examples), "protected"))
        if output_format:
            # Prefix with a short, protected label so the model knows what it is.
            resolved_zones.append(("Output format:\n" + output_format, "protected"))

    if not resolved_zones:
        raise ValueError(
            "Nothing to compress. Pass either `zones=[...]` or at least one of "
            "`instruction`, `rules`, `output_format`, `examples`."
        )

    # Compress each zone according to its level.
    detail = []
    out_parts = []
    for text, level in resolved_zones:
        if level == "protected":
            compressed = text
        elif level == "careful":
            compressed = compress(text, **CAREFUL_FLAGS)
        else:  # "free"
            compressed = compress(text, **resolved_free)
        out_parts.append(compressed)
        detail.append({
            "level":     level,
            "original":  text,
            "compressed": compressed,
            "original_len":   len(text),
            "compressed_len": len(compressed),
        })

    assembled = separator.join(p for p in out_parts if p.strip())

    if return_detail:
        return {"compressed": assembled, "zones": detail}
    return assembled


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------
# Compression is CPU-bound and pure Python. To avoid blocking an asyncio event
# loop, we run the synchronous functions in the default thread-pool executor.

async def acompress(prompt: str, **flags) -> "str | Dict":
    """
    Async wrapper around :func:`less_tokens.compress`.

    Runs the (synchronous, CPU-bound) compressor in a thread executor so it
    doesn't block the event loop. Accepts exactly the same flags as
    :func:`compress`.

    Examples
    --------
    >>> import asyncio
    >>> from less_tokens import acompress
    >>> async def main():
    ...     return await acompress("I was wondering if you could help",
    ...                            remove_filler_phrases=1)
    >>> asyncio.run(main())
    'help'
    """
    loop = asyncio.get_event_loop()
    fn = functools.partial(compress, prompt, **flags)
    return await loop.run_in_executor(None, fn)


async def acompress_structured(
    zones: Optional[Sequence[Zone]] = None,
    **kwargs,
) -> "str | Dict":
    """
    Async wrapper around :func:`compress_structured`.

    Runs the structured compressor in a thread executor. Accepts exactly the
    same arguments as :func:`compress_structured`.

    Examples
    --------
    >>> import asyncio
    >>> from less_tokens import acompress_structured
    >>> async def main():
    ...     return await acompress_structured(
    ...         instruction="Please analyse this text in detail.",
    ...         output_format='{"result": "..."}',
    ...         remove_stopwords=1,
    ...     )
    >>> asyncio.run(main())
    """
    loop = asyncio.get_event_loop()
    fn = functools.partial(compress_structured, zones, **kwargs)
    return await loop.run_in_executor(None, fn)


__all__ = [
    "compress_structured",
    "acompress",
    "acompress_structured",
    "CAREFUL_FLAGS",
    "FREE_DEFAULT_FLAGS",
    "VALID_LEVELS",
]