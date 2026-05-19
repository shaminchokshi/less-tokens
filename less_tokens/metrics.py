"""
less_tokens.metrics — six similarity metrics + token/character reduction.

Public entry point is :func:`compare`. End users should normally call
:func:`less_tokens.compare` from the package root.
"""

from __future__ import annotations

import re
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Lazy singletons for expensive resources.
# ---------------------------------------------------------------------------
# Embedding model (SentenceBERT) and BERTScore model are loaded once and
# reused across calls.

_TOKENIZER = None
_EMBED_MODEL = None
_BERTSCORER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        import tiktoken
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    return _TOKENIZER


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBED_MODEL


def _get_bertscorer():
    global _BERTSCORER
    if _BERTSCORER is None:
        from bert_score import BERTScorer
        _BERTSCORER = BERTScorer(lang="en", rescale_with_baseline=False)
    return _BERTSCORER


# ---------------------------------------------------------------------------
# Compression metrics
# ---------------------------------------------------------------------------

def _token_counts(original: str, compressed: str) -> Dict:
    tok = _get_tokenizer()
    n0 = len(tok.encode(original or ""))
    n1 = len(tok.encode(compressed or ""))
    red = (1 - n1 / n0) * 100 if n0 else 0.0
    return {
        "original_tokens":     n0,
        "compressed_tokens":   n1,
        "token_reduction_pct": round(red, 2),
    }


def _char_counts(original: str, compressed: str) -> Dict:
    c0, c1 = len(original or ""), len(compressed or "")
    red = (1 - c1 / c0) * 100 if c0 else 0.0
    return {
        "original_chars":     c0,
        "compressed_chars":   c1,
        "char_reduction_pct": round(red, 2),
    }


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

def _cosine(a: str, b: str) -> float:
    import numpy as np
    m = _get_embed_model()
    embs = m.encode([a or " ", b or " "], normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


def _bleu(reference: str, candidate: str) -> float:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass

    ref = nltk.word_tokenize((reference or "").lower())
    cand = nltk.word_tokenize((candidate or "").lower())
    if not ref or not cand:
        return 0.0
    sf = SmoothingFunction().method1
    return float(sentence_bleu([ref], cand, smoothing_function=sf))


def _rouge(reference: str, candidate: str) -> Dict[str, float]:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference or "", candidate or "")
    return {
        "rouge1_f": round(scores["rouge1"].fmeasure, 4),
        "rouge2_f": round(scores["rouge2"].fmeasure, 4),
        "rougeL_f": round(scores["rougeL"].fmeasure, 4),
    }


def _bertscore(reference: str, candidate: str) -> Dict[str, float]:
    scorer = _get_bertscorer()
    P, R, F = scorer.score([candidate or " "], [reference or " "])
    return {
        "bertscore_p": round(float(P[0]), 4),
        "bertscore_r": round(float(R[0]), 4),
        "bertscore_f": round(float(F[0]), 4),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compare(
    original_prompt: str,
    compressed_prompt: str,
    original_output: str,
    compressed_output: str,
    *,
    bertscore: bool = True,
) -> Dict:
    """
    Compute all six similarity metrics between two LLM outputs, plus the
    compression statistics between the two prompts.

    The caller is responsible for making the actual LLM calls (with whichever
    provider they prefer) and passing the four strings here.

    Parameters
    ----------
    original_prompt
        The original, un-compressed prompt sent to the LLM.
    compressed_prompt
        The compressed prompt sent to the LLM.
    original_output
        The LLM's response to the original prompt.
    compressed_output
        The LLM's response to the compressed prompt.
    bertscore
        If True (default) compute BERTScore-F1. The first call downloads
        ~1 GB; subsequent calls reuse the loaded model. Set to False on
        constrained machines.

    Returns
    -------
    dict
        A dictionary with three sub-keys:

        - ``compression``: ``original_tokens``, ``compressed_tokens``,
          ``token_reduction_pct``, ``original_chars``, ``compressed_chars``,
          ``char_reduction_pct``.
        - ``prompt_similarity``: ``cosine`` between the two prompts.
        - ``output_similarity``: the six output-side metrics —
          ``cosine``, ``bleu``, ``rouge1_f``, ``rouge2_f``, ``rougeL_f``,
          and (if enabled) ``bertscore_p``, ``bertscore_r``, ``bertscore_f``.

    Examples
    --------
    >>> from less_tokens import compress, compare
    >>> orig = "Please explain to me how to brew a good cup of coffee."
    >>> comp = compress(orig, remove_filler_phrases=1, remove_stopwords=1)
    >>> # Make your own LLM calls here:
    >>> out_orig = my_llm.call(orig)
    >>> out_comp = my_llm.call(comp)
    >>> metrics = compare(orig, comp, out_orig, out_comp)
    >>> metrics["compression"]["token_reduction_pct"]
    33.3
    >>> metrics["output_similarity"]["bertscore_f"]
    0.91
    """
    for name, val in [("original_prompt", original_prompt),
                      ("compressed_prompt", compressed_prompt),
                      ("original_output", original_output),
                      ("compressed_output", compressed_output)]:
        if not isinstance(val, str):
            raise TypeError(f"{name} must be str, got {type(val).__name__}")

    compression = {**_token_counts(original_prompt, compressed_prompt),
                   **_char_counts(original_prompt, compressed_prompt)}

    prompt_sim = {"cosine": round(_cosine(original_prompt, compressed_prompt), 4)}

    out_sim = {"cosine": round(_cosine(original_output, compressed_output), 4),
               "bleu":   round(_bleu(original_output, compressed_output), 4)}
    out_sim.update(_rouge(original_output, compressed_output))
    if bertscore:
        out_sim.update(_bertscore(original_output, compressed_output))

    return {
        "compression":        compression,
        "prompt_similarity":  prompt_sim,
        "output_similarity":  out_sim,
    }


__all__ = ["compare"]
