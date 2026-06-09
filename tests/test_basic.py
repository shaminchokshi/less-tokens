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


# ---------------------------------------------------------------------------
# smart_compress tests
# ---------------------------------------------------------------------------
from less_tokens import smart_compress, asmart_compress


def test_smart_compress_returns_str_by_default():
    out = smart_compress("Hello world")
    assert isinstance(out, str)


def test_smart_compress_rejects_non_string():
    with pytest.raises(TypeError):
        smart_compress(123)


def test_smart_compress_fenced_code_block_is_verbatim():
    """Fenced code blocks must be returned byte-for-byte."""
    msg = (
        "I was wondering if you could explain what this does.\n\n"
        "```python\n"
        "def hello():\n"
        "    print('hello world')\n"
        "```\n\n"
        "Let me know."
    )
    out = smart_compress(msg, remove_filler_phrases=1, remove_stopwords=1)
    assert "def hello():" in out
    assert "print('hello world')" in out


def test_smart_compress_fenced_code_block_unchanged():
    """The code block content must be identical, not just present."""
    code = "```python\ndef add(a, b):\n    return a + b\n```"
    msg = f"Please explain this.\n\n{code}\n\nThank you."
    out = smart_compress(msg, remove_filler_phrases=1, remove_stopwords=1)
    assert code in out


def test_smart_compress_inline_code_is_verbatim():
    """Inline code tokens must not be compressed."""
    msg = "I was wondering if you could run the `pip install less-tokens` command."
    out = smart_compress(msg, remove_filler_phrases=1, remove_stopwords=1)
    assert "`pip install less-tokens`" in out


def test_smart_compress_url_is_verbatim():
    """Bare URLs must be returned untouched."""
    msg = "Check the docs at https://docs.python.org for more details."
    out = smart_compress(msg, remove_stopwords=1)
    assert "https://docs.python.org" in out


def test_smart_compress_markdown_link_is_verbatim():
    """Markdown links must be returned untouched."""
    msg = "See [the docs](https://docs.python.org) for more information."
    out = smart_compress(msg, remove_stopwords=1)
    assert "[the docs](https://docs.python.org)" in out


def test_smart_compress_table_is_verbatim():
    """Markdown tables must be returned byte-for-byte."""
    table = "| Name | Score |\n| --- | --- |\n| Alice | 95 |\n| Bob | 87 |"
    msg = f"Here is the summary.\n\n{table}\n\nLet me know what you think."
    out = smart_compress(msg, remove_stopwords=1)
    assert "| Alice | 95 |" in out
    assert "| Bob | 87 |" in out


def test_smart_compress_math_block_is_verbatim():
    """Display math blocks must not be touched."""
    msg = "The formula is $$E = mc^2$$ which is well known."
    out = smart_compress(msg, remove_stopwords=1)
    assert "$$E = mc^2$$" in out


def test_smart_compress_prose_is_compressed():
    """Natural language prose outside protected zones must be compressed."""
    msg = "I was wondering if you could explain this concept to me in detail."
    out = smart_compress(msg, remove_filler_phrases=1)
    assert "wondering" not in out.lower()
    assert "explain" in out.lower()


def test_smart_compress_heading_prefix_preserved():
    """The # prefix of a heading must survive; only the text is compressed."""
    msg = "## I was wondering if you could explain this section\n\nSome body text."
    out = smart_compress(msg, remove_filler_phrases=1)
    assert out.startswith("## ")
    assert "wondering" not in out.lower()


def test_smart_compress_list_marker_preserved():
    """List markers (- and 1.) must survive; only item text is compressed."""
    msg = (
        "- I was wondering if you could do task A\n"
        "- I was wondering if you could do task B"
    )
    out = smart_compress(msg, remove_filler_phrases=1)
    lines = [l for l in out.splitlines() if l.strip()]
    assert all(l.startswith("- ") for l in lines), f"markers lost: {lines}"
    assert "wondering" not in out.lower()


def test_smart_compress_mixed_message():
    """A realistic mixed message: prose + code + URL all handled correctly."""
    msg = (
        "I was wondering if you could explain what this function does.\n\n"
        "```python\n"
        "def greet(name):\n"
        "    return f'Hello, {name}'\n"
        "```\n\n"
        "Also see https://docs.python.org for more details."
    )
    out = smart_compress(msg, remove_filler_phrases=1, remove_stopwords=1)
    # Code untouched
    assert "def greet(name):" in out
    assert "return f'Hello, {name}'" in out
    # URL untouched
    assert "https://docs.python.org" in out
    # Filler phrase removed
    assert "wondering" not in out.lower()


def test_smart_compress_return_segments_shape():
    """return_segments=True must return a dict with the right keys."""
    msg = "Explain this.\n\n```python\npass\n```"
    result = smart_compress(msg, return_segments=True)
    assert isinstance(result, dict)
    assert "compressed" in result
    assert "segments" in result
    for seg in result["segments"]:
        assert "kind" in seg
        assert "original" in seg
        assert "compressed" in seg
        assert seg["kind"] in ("prose", "protected")


def test_smart_compress_return_segments_code_kind():
    """Code block segments must have kind='protected'."""
    code = "```python\npass\n```"
    msg = f"Some text.\n\n{code}\n\nMore text."
    result = smart_compress(msg, return_segments=True)
    protected = [s for s in result["segments"] if s["kind"] == "protected"]
    assert any("pass" in s["original"] for s in protected)


def test_smart_compress_no_flags_near_noop():
    """With default flags only, a plain prose message should be nearly unchanged."""
    msg = "Hello world. This is a test message."
    out = smart_compress(msg)
    assert "Hello" in out
    assert "test" in out


def test_smart_compress_plain_prose_only():
    """A message with no protected zones compresses the same as compress()."""
    msg = "I was wondering if you could explain this concept."
    out_smart = smart_compress(msg, remove_filler_phrases=1)
    out_plain = compress(msg, remove_filler_phrases=1)
    # They may differ slightly due to line-level batching, but both should
    # drop the filler phrase.
    assert "wondering" not in out_smart.lower()
    assert "wondering" not in out_plain.lower()


def test_smart_compress_empty_string():
    """Empty input should return empty string without error."""
    out = smart_compress("", remove_filler_phrases=1)
    assert out == ""


def test_smart_compress_code_only():
    """A message that is entirely a code block should be returned verbatim."""
    msg = "```python\ndef foo():\n    pass\n```"
    out = smart_compress(msg, remove_filler_phrases=1, remove_stopwords=1)
    assert out == msg


def test_asmart_compress_runs():
    """Async variant must run and return the same result as the sync version."""
    msg = "I was wondering if you could explain this."

    async def go():
        return await asmart_compress(msg, remove_filler_phrases=1)

    result = asyncio.run(go())
    assert isinstance(result, str)
    assert "wondering" not in result.lower()


def test_asmart_compress_concurrency():
    """Multiple async smart_compress calls must run concurrently without error."""
    messages = [
        "I was wondering if you could do task A.",
        "I was wondering if you could do task B.",
        "I was wondering if you could do task C.",
    ]

    async def go():
        return await asyncio.gather(
            *[asmart_compress(m, remove_filler_phrases=1) for m in messages]
        )

    results = asyncio.run(go())
    assert len(results) == 3
    assert all("wondering" not in r.lower() for r in results)


def test_asmart_compress_preserves_code():
    """Async variant must still protect code blocks."""
    msg = "Explain this.\n\n```python\npass\n```"

    async def go():
        return await asmart_compress(msg, remove_filler_phrases=1)

    result = asyncio.run(go())
    assert "pass" in result


def test_smart_compress_conversation_history_pattern():
    """
    Simulate the primary use case: compressing every message in a conversation.
    All protected zones across all messages must survive.
    """
    conversation = [
        "I was wondering if you could explain how `list.append()` works in Python.",
        (
            "Sure! `list.append(x)` adds item `x` to the end of the list.\n\n"
            "```python\n"
            "my_list = [1, 2, 3]\n"
            "my_list.append(4)\n"
            "print(my_list)  # [1, 2, 3, 4]\n"
            "```\n\n"
            "See https://docs.python.org/3/tutorial/datastructures.html for more."
        ),
        "I was wondering if you could show me how to remove an item instead.",
    ]

    compressed = [
        smart_compress(msg, remove_filler_phrases=1, remove_stopwords=1)
        for msg in conversation
    ]

    # Inline code in user message
    assert "`list.append()`" in compressed[0]
    # Code block in LLM response
    assert "my_list.append(4)" in compressed[1]
    # URL in LLM response
    assert "https://docs.python.org/3/tutorial/datastructures.html" in compressed[1]
    # Filler phrase removed from both user messages
    assert "wondering" not in compressed[0].lower()
    assert "wondering" not in compressed[2].lower()