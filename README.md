# less-tokens

**Deterministic, training-free lexical compression for LLM prompts.**

Reduces token count by 30–40% on typical prompts with no auxiliary model, no GPU, and no network call. Pure Python, runs in milliseconds on CPU.

## Why

Every token you send to an LLM API costs money and adds latency. Most prompts contain pleasantries, filler phrases, redundant articles, and bulky multi-word equivalents — all removable without changing the model's response. `less-tokens` strips these deterministically using a configurable lexical pipeline.

Unlike learned compressors (LLMLingua, Selective Context), `less-tokens`:

- Has **no neural model** — runs on a laptop CPU in milliseconds.
- Is **fully deterministic** — same input + same flags always produces the same output.
- Is **transparent** — every technique is a named function you can audit.

## Install

```bash
pip install less-tokens
```

On first use, NLTK resources (~30 MB) are downloaded automatically. BERTScore optionally downloads a ~1 GB model on first call — disable with `bertscore=False` if you don't need it.

## Two functions, that's it

### 1. `compress` — shrink a prompt

```python
from less_tokens import compress

prompt = ("I was wondering if you could please explain to me how "
          "I can actually run a Python script from the command line.")

# Flags: 1 = enabled, 0 = disabled (also accepts True/False, "on"/"off")
short = compress(
    prompt,
    remove_filler_phrases=1,    # drops "I was wondering if you could"
    apply_contractions=1,       # "do not" -> "don't"
    apply_abbreviations=1,      # "for example" -> "e.g."
    remove_filler_words=1,      # drops "actually", "basically", "really"
    remove_stopwords=1,         # drops "the", "is", "in" (keeps negations & wh-words)
    remove_function_words=0,    # drops articles + auxiliaries (aggressive)
    pos_keep_only=0,            # keeps only N/V/Adj/Num/Wh (very aggressive)
    lemmatize=0,                # "running" -> "run"
    shorten_synonyms=0,         # replaces long words with shorter WordNet synonyms
    preserve_named_entities=1,  # default: protects "New York", "Python", etc.
    normalize_whitespace_punct=1,  # default: collapse spaces, dedup punctuation
)

print(short)
# 'explain run Python script command line'
```

**The 11 techniques**, all independently toggleable:

| Flag | Effect | Token saving |
|------|--------|--------------|
| `remove_filler_phrases` | Drops hedges like "I was wondering if you could" | High on chatty prompts |
| `apply_abbreviations` | "for example" → "e.g." | Low–medium |
| `apply_contractions` | "do not" → "don't" | Low |
| `remove_filler_words` | Drops "basically", "actually", "really" | Low–medium |
| `remove_stopwords` | Drops NLTK stopwords (keeps negations & wh-words) | **High** |
| `remove_function_words` | Drops articles + auxiliaries | High |
| `pos_keep_only` | Keep only nouns/verbs/adjectives/numerals/wh/negations | **Highest** |
| `lemmatize` | "running" → "run", "studies" → "study" | Low |
| `shorten_synonyms` | WordNet substitution for shorter forms | Low (use with caution) |
| `preserve_named_entities` | Protects NE spans from pruning (modifier) | — |
| `normalize_whitespace_punct` | Collapses whitespace and duplicate punctuation | Tiny but always-on |

**Two cross-cutting safety rules:** negations (`not`, `no`, `never`) and wh-words (`what`, `why`, ...) are never dropped, even under maximum compression. Question form is re-asserted at the end if the original prompt was a question.

### 2. `compare` — evaluate compression quality

You make your own LLM calls with whichever provider you prefer, then pass the four strings to `compare`. It computes six similarity metrics and the compression statistics.

```python
from less_tokens import compress, compare
from openai import OpenAI                       # or anthropic, google, etc.

client = OpenAI()

def call_llm(prompt: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return r.choices[0].message.content

original  = "I was wondering if you could explain how do I run a python script."
compressed = compress(original, remove_filler_phrases=1, remove_stopwords=1)

out_original  = call_llm(original)
out_compressed = call_llm(compressed)

metrics = compare(original, compressed, out_original, out_compressed)
print(metrics)
```

**Output shape:**

```python
{
    "compression": {
        "original_tokens":     18,
        "compressed_tokens":   8,
        "token_reduction_pct": 55.56,
        "original_chars":      62,
        "compressed_chars":    27,
        "char_reduction_pct":  56.45,
    },
    "prompt_similarity": {
        "cosine": 0.842,                # MiniLM SentenceBERT cosine
    },
    "output_similarity": {              # the 6 metrics
        "cosine":      0.917,           # MiniLM SentenceBERT cosine
        "bleu":        0.412,           # BLEU-4 with smoothing
        "rouge1_f":    0.673,
        "rouge2_f":    0.418,
        "rougeL_f":    0.601,
        "bertscore_p": 0.923,
        "bertscore_r": 0.918,
        "bertscore_f": 0.920,           # the headline quality metric
    },
}
```

Skip BERTScore (and avoid the ~1 GB model download) by passing `bertscore=False`.

## Recommended configurations

These match the operating points characterised in the supporting paper:

| Goal | Flags | Typical reduction |
|------|-------|-------------------|
| **Safe** (≥0.96 BERTScore) | `remove_filler_phrases=1, remove_filler_words=1, apply_contractions=1` | ~2% |
| **Balanced** (≥0.91 BERTScore) | `remove_filler_phrases=1, apply_abbreviations=1, apply_contractions=1, remove_filler_words=1, remove_stopwords=1` | **~30%** |
| **Aggressive** (≥0.90 BERTScore) | `pos_keep_only=1, preserve_named_entities=1` | **~35%** |
| **Maximum** (~0.88 BERTScore) | all flags = 1 | ~40% |

## Caveats

- English only (NLTK stopwords + WordNet are English).
- Best on short and medium prompts (60–2000 chars). Long retrieval-augmented contexts are not the target.
- `shorten_synonyms` is the riskiest technique — WordNet sometimes picks topically narrower terms. Don't enable it without testing on your data.
- Quality is task-dependent. Commonsense reasoning (HellaSwag-style) degrades faster than other tasks under aggressive compression.

## License

MIT.
