"""
less_tokens.smart_compress — auto-detecting compression for a single message.

:func:`smart_compress` parses a message string (user input or LLM
response), identifies elements that must never be touched (code blocks, inline
code, tables, URLs, math, HTML), and compresses only the natural language prose
in between.

Apply it to each message in a conversation history individually:

    from less_tokens.smart_compress import smart_compress

    compressed_history = [
        smart_compress(msg, remove_filler_phrases=1, remove_stopwords=1)
        for msg in conversation
    ]
"""

from __future__ import annotations

import asyncio
import functools
import re
from typing import Dict, List, Tuple, Union

from less_tokens.compressor import compress, _coerce_flag


# ---------------------------------------------------------------------------
# Segment types
# ---------------------------------------------------------------------------
# Each segment is a (kind, text) tuple.
# kind = "prose"     -> compress this
# kind = "protected" -> return verbatim

_PROSE = "prose"
_PROTECTED = "protected"

Segment = Tuple[str, str]


# ---------------------------------------------------------------------------
# Regex patterns for protected zones
# ---------------------------------------------------------------------------
# Order matters: longer / more specific patterns are matched first.
# The outer group of each pattern is the entire token to protect.

_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # Fenced code blocks: ```lang\n...\n``` or ~~~...~~~
    ("fenced_code",  re.compile(r"(`{3,}|~{3,}).*?\1", re.DOTALL)),

    # Indented code blocks: 4-space or tab-indented lines (CommonMark)
    # Only matches a run of such lines (at least one).
    ("indented_code", re.compile(
        r"(?:(?:^|\n)(?: {4}|\t)[^\n]*)+", re.MULTILINE
    )),

    # Math blocks: $$...$$ (display) before inline $...$
    ("math_block",   re.compile(r"\$\$[\s\S]+?\$\$")),

    # Inline math: $...$  (non-greedy, no newline inside)
    ("math_inline",  re.compile(r"\$[^\n$]+?\$")),

    # HTML blocks / tags
    ("html",         re.compile(
        r"<(?:[a-zA-Z][^\n>]*?/?>|/[a-zA-Z][^\n>]*?>)", re.DOTALL
    )),

    # Markdown tables: one or more pipe-delimited lines including the separator row
    ("table",        re.compile(
        r"(?:^|\n)(\|[^\n]+\|[ \t]*\n)([ \t]*\|[-:| \t]+\|[ \t]*\n)"
        r"((?:\|[^\n]+\|[ \t]*\n?)*)",
        re.MULTILINE,
    )),

    # Markdown image links: ![alt](url) — before bare URL / inline link
    ("image_link",   re.compile(r"!\[.*?\]\(.*?\)")),

    # Markdown inline links: [text](url)
    ("md_link",      re.compile(r"\[.*?\]\(.*?\)")),

    # Bare URLs: http(s)://... or ftp://...
    ("url",          re.compile(r"(?:https?|ftp)://\S+")),

    # Inline code: `...`  (single or double backticks)
    ("inline_code",  re.compile(r"`{1,2}[^`\n]+?`{1,2}")),

    # JSON-like objects / arrays: { ... } or [ ... ] spanning multiple lines
    # (heuristic: only match if it starts at line start or after whitespace)
    ("json_block",   re.compile(
        r"(?:^|\n)\s*(?:\{[\s\S]*?\}|\[[\s\S]*?\])(?=\s*\n|$)",
        re.MULTILINE,
    )),
]


# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------

def _segment(text: str) -> List[Segment]:
    """
    Split *text* into an ordered list of (kind, content) segments where kind is
    either ``"prose"`` or ``"protected"``.

    The algorithm does a single left-to-right scan: at each position it tries
    every protected pattern and takes the earliest (leftmost) match. Everything
    before that match is prose; the match itself is protected; then scanning
    resumes after the match.
    """
    segments: List[Segment] = []
    pos = 0
    n = len(text)

    while pos < n:
        # Find the earliest protected match from the current position.
        best_start: int = n
        best_end: int = n
        best_text: str = ""

        for _name, pat in _PATTERNS:
            m = pat.search(text, pos)
            if m and m.start() < best_start:
                best_start = m.start()
                best_end = m.end()
                best_text = m.group(0)

        # No more protected zones from here on: everything left is prose.
        # Emit it exactly once and stop. (Previously this tail was appended by
        # BOTH the "prose before match" branch and this branch, which doubled
        # every prose-only message.)
        if best_start >= n:
            remaining = text[pos:]
            if remaining:
                segments.append((_PROSE, remaining))
            break

        # A protected match was found at best_start.
        # Everything from pos up to it is prose.
        if best_start > pos:
            segments.append((_PROSE, text[pos:best_start]))

        segments.append((_PROTECTED, best_text))
        pos = best_end

    return segments


# ---------------------------------------------------------------------------
# Prose splitter: compress heading text and list-item text separately
# ---------------------------------------------------------------------------
# Inside a prose chunk there can still be Markdown structure:
#   - Heading lines (# Title) — the "#" prefix is protected, text is compressed
#   - List item lines (- item / 1. item) — the marker is protected, text is compressed
#   - Plain paragraph text — compressed as-is

_HEADING_RE = re.compile(r"^(#{1,6}\s+)(.*)$", re.MULTILINE)
_UL_RE = re.compile(r"^([ \t]*[-*+]\s+)(.*)$", re.MULTILINE)
_OL_RE = re.compile(r"^([ \t]*\d+[.)]\s+)(.*)$", re.MULTILINE)


def _compress_prose(text: str, flags: Dict[str, bool]) -> str:
    """
    Compress a prose chunk, treating heading prefixes and list markers as
    protected sub-tokens (they are preserved exactly, only the following text
    is compressed).
    """
    if not text.strip():
        return text

    # Preserve leading/trailing newlines — these are structural separators
    # between prose and adjacent code blocks or other protected zones.
    leading_newlines  = len(text) - len(text.lstrip("\n"))
    trailing_newlines = len(text) - len(text.rstrip("\n"))
    lead_nl  = text[:leading_newlines]
    trail_nl = text[len(text) - trailing_newlines:] if trailing_newlines else ""
    inner    = text[leading_newlines: len(text) - trailing_newlines if trailing_newlines else len(text)]

    if not inner.strip():
        return text

    lines = inner.split("\n")
    out_lines: List[str] = []

    para_buf: List[str] = []

    def flush_para():
        nonlocal para_buf
        if para_buf:
            para_text = " ".join(para_buf)
            compressed = compress(para_text, **flags)
            out_lines.append(compressed)
            para_buf = []

    for line in lines:
        # Heading line?
        hm = _HEADING_RE.match(line)
        if hm:
            flush_para()
            prefix, body = hm.group(1), hm.group(2)
            compressed_body = compress(body, **flags) if body.strip() else body
            out_lines.append(prefix + compressed_body)
            continue

        # Unordered list item?
        um = _UL_RE.match(line)
        if um:
            flush_para()
            prefix, body = um.group(1), um.group(2)
            compressed_body = compress(body, **flags) if body.strip() else body
            out_lines.append(prefix + compressed_body)
            continue

        # Ordered list item?
        om = _OL_RE.match(line)
        if om:
            flush_para()
            prefix, body = om.group(1), om.group(2)
            compressed_body = compress(body, **flags) if body.strip() else body
            out_lines.append(prefix + compressed_body)
            continue

        # Blank line: flush the current paragraph buffer.
        if not line.strip():
            if para_buf:
                flush_para()
            out_lines.append("")
            continue

        # Ordinary prose line: accumulate into paragraph buffer.
        para_buf.append(line.strip())

    flush_para()
    return lead_nl + "\n".join(out_lines) + trail_nl


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def smart_compress(
    message: str,
    *,
    remove_filler_phrases: int = 0,
    apply_abbreviations: int = 0,
    apply_contractions: int = 0,
    remove_filler_words: int = 0,
    remove_stopwords: int = 0,
    remove_function_words: int = 0,
    pos_keep_only: int = 0,
    lemmatize: int = 0,
    shorten_synonyms: int = 0,
    preserve_named_entities: int = 1,
    normalize_whitespace_punct: int = 1,
    return_segments: bool = False,
) -> Union[str, Dict]:
    """
    Compress a single conversation message (user turn or LLM response) while
    leaving all non-prose elements completely untouched.

    Protected (never compressed):
      - Fenced and indented code blocks
      - Inline code (``backticks``)
      - Markdown tables
      - Bare URLs and Markdown links / image links
      - Math blocks (``$$...$$``) and inline math (``$...$``)
      - HTML tags
      - JSON / array blocks

    Compressed (natural language only):
      - Paragraph text
      - Heading text (the ``#`` prefix is preserved)
      - List-item text (the ``-`` / ``1.`` marker is preserved)

    Apply this to every message in a conversation history:

    >>> compressed_history = [
    ...     smart_compress(msg, remove_filler_phrases=1, remove_stopwords=1)
    ...     for msg in conversation
    ... ]

    Parameters
    ----------
    message
        A single message string (user input or LLM output).
    remove_filler_phrases, apply_abbreviations, apply_contractions,
    remove_filler_words, remove_stopwords, remove_function_words,
    pos_keep_only, lemmatize, shorten_synonyms, preserve_named_entities,
    normalize_whitespace_punct
        Compression flags — same semantics as :func:`less_tokens.compress`.
    return_segments
        If True, return a dict with ``"compressed"`` (the final string) and
        ``"segments"`` (list of ``{"kind", "original", "compressed"}`` dicts)
        for debugging.

    Returns
    -------
    str
        The compressed message (default).
    dict
        If ``return_segments=True``.

    Examples
    --------
    >>> from less_tokens.smart_compress import smart_compress
    >>> msg = '''
    ... I was wondering if you could explain the code below.
    ...
    ... ```python
    ... def hello():
    ...     print("hello world")
    ... ```
    ...
    ... Also, what does the URL https://example.com point to?
    ... '''
    >>> print(smart_compress(msg, remove_filler_phrases=1, remove_stopwords=1))
    explain code below.
    <BLANKLINE>
    ```python
    def hello():
        print("hello world")
    ```
    <BLANKLINE>
    Also, URL https://example.com point?
    """
    if not isinstance(message, str):
        raise TypeError(f"message must be str, got {type(message).__name__}")

    flags = {
        "remove_filler_phrases":      _coerce_flag(remove_filler_phrases),
        "apply_abbreviations":        _coerce_flag(apply_abbreviations),
        "apply_contractions":         _coerce_flag(apply_contractions),
        "remove_filler_words":        _coerce_flag(remove_filler_words),
        "remove_stopwords":           _coerce_flag(remove_stopwords),
        "remove_function_words":      _coerce_flag(remove_function_words),
        "pos_keep_only":              _coerce_flag(pos_keep_only),
        "lemmatize":                  _coerce_flag(lemmatize),
        "shorten_synonyms":           _coerce_flag(shorten_synonyms),
        "preserve_named_entities":    _coerce_flag(preserve_named_entities),
        "normalize_whitespace_punct": _coerce_flag(normalize_whitespace_punct),
    }

    segments = _segment(message)
    detail = []
    out_parts: List[str] = []

    for kind, text in segments:
        if kind == _PROTECTED:
            compressed = text
        else:
            compressed = _compress_prose(text, flags)
            # Preserve leading/trailing inline whitespace (spaces/tabs, not
            # newlines) so that protected tokens that sit mid-sentence — like
            # a URL — don't get glued to the surrounding compressed words.
            lead  = text[: len(text) - len(text.lstrip(" \t"))]
            trail = text[len(text.rstrip(" \t")):]
            inner = compressed.strip(" \t")
            compressed = lead + inner + trail
        out_parts.append(compressed)
        detail.append({"kind": kind, "original": text, "compressed": compressed})

    result = "".join(out_parts)

    if return_segments:
        return {"compressed": result, "segments": detail}
    return result


async def asmart_compress(
    message: str,
    **kwargs,
) -> "str | Dict":
    """
    Async wrapper around :func:`smart_compress`.

    Runs the (synchronous, CPU-bound) compressor in a thread executor so it
    doesn't block the event loop. Accepts exactly the same arguments as
    :func:`smart_compress`.

    Examples
    --------
    >>> import asyncio
    >>> from less_tokens.smart_compress import asmart_compress
    >>> async def main():
    ...     return await asmart_compress(
    ...         "I was wondering if you could explain this.",
    ...         remove_filler_phrases=1,
    ...     )
    >>> asyncio.run(main())
    """
    loop = asyncio.get_event_loop()
    fn = functools.partial(smart_compress, message, **kwargs)
    return await loop.run_in_executor(None, fn)


__all__ = ["smart_compress", "asmart_compress"]