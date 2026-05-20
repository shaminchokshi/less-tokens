"""Basic smoke tests for less_tokens. Run with: pytest tests/"""
import pytest
from less_tokens import compress, compare, TECHNIQUES


def test_compress_returns_str_by_default():
    out = compress("Hello world")
    assert isinstance(out, str)


def test_compress_no_flags_is_near_noop():
    """With only normalize_whitespace_punct (default on), output should be ~same."""
    p = "Hello world."
    assert compress(p).strip() == "Hello world."


def test_compress_filler_phrases_works():
    p = "I was wondering if you could explain something."
    out = compress(p, remove_filler_phrases=1)
    assert "wondering" not in out.lower()
    assert "explain" in out.lower()


def test_compress_stopwords_works():
    p = "The cat is on the mat."
    out = compress(p, remove_stopwords=1)
    assert "cat" in out.lower()
    assert "mat" in out.lower()
    # 'the' is a stopword, should be gone
    assert " the " not in f" {out.lower()} "


def test_compress_preserves_negations():
    """The compressor must never drop 'not'."""
    p = "Do not run this code."
    out = compress(p, remove_stopwords=1, remove_function_words=1)
    assert "not" in out.lower()


def test_compress_preserves_wh_words():
    p = "What is the capital of France?"
    out = compress(p, remove_stopwords=1)
    assert "what" in out.lower()


def test_compress_question_form_survives():
    p = "How do I run a python script?"
    out = compress(p, pos_keep_only=1)
    assert out.strip().endswith("?")


def test_compress_accepts_flag_aliases():
    """0/1 ints, bools, and 'on'/'off' strings should all work."""
    p = "I was wondering if you could help me."
    a = compress(p, remove_filler_phrases=1)
    b = compress(p, remove_filler_phrases=True)
    c = compress(p, remove_filler_phrases="on")
    assert a == b == c


def test_compress_with_trace():
    p = "I was wondering if you could explain this."
    result = compress(p, remove_filler_phrases=1, return_trace=True)
    assert "compressed" in result
    assert "trace" in result
    assert "flags" in result
    assert len(result["trace"]) >= 1


def test_compress_reduces_tokens_on_realistic_prompt():
    """End-to-end: a typical wordy prompt should shrink meaningfully."""
    p = ("I was wondering if you could please explain to me, in detail, how I "
         "can actually go about brewing a really good cup of coffee at home.")
    out = compress(
        p,
        remove_filler_phrases=1,
        apply_contractions=1,
        remove_filler_words=1,
        remove_stopwords=1,
    )
    assert len(out) < len(p) * 0.7   # at least 30% character reduction


def test_techniques_constant_exists():
    assert isinstance(TECHNIQUES, tuple)
    assert len(TECHNIQUES) == 11


def test_compare_returns_correct_shape():
    """Without making real LLM calls, just confirm metric shape."""
    metrics = compare(
        original_prompt="Hello world",
        compressed_prompt="Hello",
        original_output="The greeting is hello world.",
        compressed_output="The greeting is just hello.",
        bertscore=False,    # skip the 1GB download in unit tests
    )
    assert "compression" in metrics
    assert "prompt_similarity" in metrics
    assert "output_similarity" in metrics

    c = metrics["compression"]
    assert c["original_tokens"] >= c["compressed_tokens"]
    assert "token_reduction_pct" in c

    o = metrics["output_similarity"]
    for k in ("cosine", "bleu", "rouge1_f", "rouge2_f", "rougeL_f"):
        assert k in o
        assert 0.0 <= o[k] <= 1.0


def test_compare_rejects_non_string_inputs():
    with pytest.raises(TypeError):
        compare(123, "b", "c", "d")


# ---------------------------------------------------------------------------
# Structured compression tests
# ---------------------------------------------------------------------------
import asyncio
from less_tokens import (compress_structured, acompress, acompress_structured,
                         CAREFUL_FLAGS, VALID_LEVELS)


def test_structured_protected_zone_is_verbatim():
    """A protected zone must be returned byte-for-byte."""
    schema = '{"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0}'
    result = compress_structured(
        instruction="I was wondering if you could analyse this review.",
        output_format=schema,
        remove_filler_phrases=1,
        return_detail=True,
    )
    protected = [z for z in result["zones"] if z["level"] == "protected"][0]
    assert schema in protected["compressed"]
    assert protected["original_len"] == protected["compressed_len"] or schema in protected["compressed"]


def test_structured_careful_keeps_negations():
    """Careful zones must never drop negations."""
    result = compress_structured(
        zones=[("Do not fabricate. Never guess.", "careful")],
    )
    assert "not" in result.lower()
    assert "never" in result.lower()


def test_structured_free_zone_compresses():
    """Free zone with filler phrase should shrink."""
    result = compress_structured(
        zones=[("I was wondering if you could explain this", "free")],
        free_flags={"remove_filler_phrases": 1},
    )
    assert "wondering" not in result.lower()


def test_structured_explicit_zones_ordering():
    out = compress_structured(zones=[
        ("First part here.", "protected"),
        ("Second part here.", "protected"),
    ])
    assert out.index("First") < out.index("Second")


def test_structured_rejects_bad_level():
    import pytest
    with pytest.raises(ValueError):
        compress_structured(zones=[("text", "banana")])


def test_structured_requires_something():
    import pytest
    with pytest.raises(ValueError):
        compress_structured()


def test_acompress_runs():
    async def go():
        return await acompress("I was wondering if you could help",
                              remove_filler_phrases=1)
    result = asyncio.run(go())
    assert "wondering" not in result.lower()


def test_acompress_structured_runs():
    async def go():
        return await acompress_structured(
            instruction="I was wondering if you could analyse",
            output_format='{"x": 1}',
            free_flags={"remove_filler_phrases": 1},
        )
    result = asyncio.run(go())
    assert '{"x": 1}' in result


def test_acompress_concurrency():
    async def go():
        return await asyncio.gather(
            acompress("I was wondering if you could do A", remove_filler_phrases=1),
            acompress("I was wondering if you could do B", remove_filler_phrases=1),
        )
    results = asyncio.run(go())
    assert len(results) == 2
    assert all("wondering" not in r.lower() for r in results)


def test_valid_levels_constant():
    assert VALID_LEVELS == ("free", "careful", "protected")