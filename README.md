# less-tokens

[![PyPI version](https://img.shields.io/pypi/v/less-tokens.svg)](https://pypi.org/project/less-tokens/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Shrink your LLM prompts by 30 to 40 percent without changing what the model says back.

`less-tokens` is a small Python library that compresses prompts before you send them to an LLM. It works by stripping out filler words, redundant phrases, and grammatical scaffolding that the model doesn't actually need. The result is a shorter prompt that costs less and responds faster, while producing essentially the same answer.

No neural model, no GPU, no API key for the compression itself. It's classical lexical NLP, runs in milliseconds on a laptop CPU, and is fully deterministic.

```python
from less_tokens import compress

original = "I was wondering if you could please explain to me how I can run a Python script from the command line."
compressed = compress(original,
                      remove_filler_phrases=1,
                      remove_stopwords=1,
                      apply_contractions=1)

print(compressed)
# "explain run Python script command line"
```

## Why this exists

If you're calling OpenAI, Anthropic, or any other LLM API at meaningful volume, every token has a cost. And typical prompts carry a lot of fat:

- *"I was wondering if you could..."* is hedging the model ignores
- *"the"*, *"a"*, *"is"* are function words that rarely change meaning
- *"basically"*, *"actually"*, *"really"* are fillers
- *"for example"* is just a verbose way to write *"e.g."*

Strip these out and the model still gets your point, but you pay less. On a large benchmark we ran (1,242 prompts, 18,630 paired LLM completions), here's how the headline numbers came out:

| Setting | Token reduction | Output similarity (BERTScore F1) |
|---------|-----------------|----------------------------------|
| Conservative | about 2% | 0.96 |
| **Balanced** | **about 30%** | **0.91** |
| Aggressive | about 35% | 0.91 |
| Maximum | about 40% | 0.88 |

The balanced setting is the sweet spot for most production use. Aggressive gets you a bit more compression without much extra quality loss.

## Install

```bash
pip install less-tokens
```

On first use it downloads about 30 MB of NLTK data automatically. If you also call `compare()`, BERTScore will download an additional ~1 GB model the first time. You can skip that with `bertscore=False` if you don't need it.

Using a virtual environment is highly recommended:

```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# macOS or Linux
source .venv/bin/activate

pip install less-tokens
```

## How to use it

The library exposes two functions. That's it.

### `compress()` shrinks a prompt

Pass your prompt and any combination of eleven flags. Each flag is `1` to enable or `0` to disable. Bool and string aliases like `True` or `"on"` work too. Defaults are off for everything except whitespace cleanup, so you choose what runs.

```python
from less_tokens import compress

short = compress(
    "I was wondering if you could explain this to me.",
    remove_filler_phrases=1,
    remove_stopwords=1,
)
# "explain"
```

#### The eleven techniques

| Flag | What it does | Example |
|------|--------------|---------|
| `remove_filler_phrases` | Strips hedging phrases | "I was wondering if you could explain" becomes "explain" |
| `apply_abbreviations` | Replaces verbose forms | "for example" becomes "e.g." |
| `apply_contractions` | Combines into contractions | "do not" becomes "don't" |
| `remove_filler_words` | Drops single-word fillers | "this is basically really good" becomes "this is good" |
| `remove_stopwords` | Drops common stopwords | "the cat is on the mat" becomes "cat mat" |
| `remove_function_words` | Drops articles and auxiliaries | "the cat is running" becomes "cat running" |
| `pos_keep_only` | Keeps only content words | "I need to read the book quickly" becomes "need read book" |
| `lemmatize` | Reduces words to root forms | "running studies" becomes "run study" |
| `shorten_synonyms` | Substitutes shorter synonyms | "automobile" becomes "car" |
| `preserve_named_entities` | Protects names from pruning | "New York" stays intact (modifier flag) |
| `normalize_whitespace_punct` | Cleans up spacing | "hello   world!!!" becomes "hello world!" (always on) |

#### What never gets removed

Two categories of words are hard-coded as protected, even at the most aggressive setting.

First, negations. Words like `not`, `no`, `never`, `nothing`, `nor`, `nobody`, and `cannot`. Dropping these flips the meaning of a sentence, which would be catastrophic. "Do not run this code" becoming "Do run this code" is not a tradeoff anyone wants.

Second, question words. `What`, `why`, `how`, `when`, `where`, `which`. These carry the intent of a query.

Also, if your original prompt ended with a question mark, the compressed version will too. We re-assert question form at the end of the pipeline so it isn't lost during pruning.

#### Four presets you can copy

You don't have to figure out which flags to combine. Here are four named recipes for different aggression levels:

```python
# SAFE: barely shrinks anything, near-perfect quality preservation.
# Useful when you can't afford any quality risk.
compress(prompt,
         remove_filler_phrases=1,
         apply_contractions=1,
         remove_filler_words=1)
# about 2% reduction, 0.96 BERTScore

# BALANCED: the production default. Roughly 30% reduction with minimal
# quality loss. Start here.
compress(prompt,
         remove_filler_phrases=1,
         apply_abbreviations=1,
         apply_contractions=1,
         remove_filler_words=1,
         remove_stopwords=1)
# about 30% reduction, 0.91 BERTScore

# AGGRESSIVE: pure POS-based pruning. Slightly more reduction than balanced
# at very similar quality. Great for high-volume systems.
compress(prompt,
         pos_keep_only=1,
         preserve_named_entities=1)
# about 35% reduction, 0.91 BERTScore

# MAXIMUM: everything on. About 40% reduction at the cost of some output
# quality. Use when the cost savings really matter.
compress(prompt,
         remove_filler_phrases=1, apply_abbreviations=1, apply_contractions=1,
         remove_filler_words=1, remove_stopwords=1, remove_function_words=1,
         pos_keep_only=1, lemmatize=1, shorten_synonyms=1, preserve_named_entities=1)
# about 40% reduction, 0.88 BERTScore
```

### `compare()` measures the quality tradeoff

Compression is only useful if the LLM still produces the same answer. `compare()` quantifies that across six different similarity metrics so you can see exactly what compressing cost you.

You make the LLM calls yourself, with whichever provider you like. `compare()` only looks at the four strings: original prompt, compressed prompt, output from original, output from compressed.

```python
from less_tokens import compress, compare
from openai import OpenAI

client = OpenAI()

def call_llm(prompt: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return r.choices[0].message.content

original   = "I was wondering if you could explain how to brew good coffee at home."
compressed = compress(original, remove_filler_phrases=1, remove_stopwords=1)

out_original   = call_llm(original)
out_compressed = call_llm(compressed)

metrics = compare(original, compressed, out_original, out_compressed)
```

#### What you get back

```python
{
    "compression": {
        "original_tokens":     18,
        "compressed_tokens":   8,
        "token_reduction_pct": 55.56,    # you saved 55% of your tokens
        "original_chars":      72,
        "compressed_chars":    32,
        "char_reduction_pct":  55.56,
    },
    "prompt_similarity": {
        "cosine": 0.842,                 # the two prompts mean roughly the same thing
    },
    "output_similarity": {               # six metrics on the LLM outputs
        "cosine":      0.917,
        "bleu":        0.412,
        "rouge1_f":    0.673,
        "rouge2_f":    0.418,
        "rougeL_f":    0.601,
        "bertscore_p": 0.923,
        "bertscore_r": 0.918,
        "bertscore_f": 0.920,
    },
}
```

#### What each of the six metrics actually means

All six measure the same thing from different angles: how similar is the LLM's response to the compressed prompt, compared to its response to the original. Each one captures a different notion of "similar".

**1. `cosine`. Semantic similarity. Range 0.0 to 1.0.**

The plain-English question it answers: *do the two outputs mean the same thing?*

It works by embedding both outputs with SentenceBERT (MiniLM-L6-v2) and taking the cosine of the angle between them. This is the most forgiving metric in the set because it handles paraphrasing well.

Interpretation:
- 0.95 or above: essentially identical meaning
- 0.85 to 0.95: same meaning, different wording
- 0.70 to 0.85: related but starting to drift
- below 0.70: the meanings have meaningfully diverged

**2. `bleu`. Word-sequence overlap. Range 0.0 to 1.0.**

The plain-English question: *do the two outputs use the same exact words in the same order?*

BLEU-4 with smoothing, originally invented for machine translation (Papineni et al., 2002). This is very strict. It penalises rewording, even when the meaning is preserved perfectly.

Interpretation:
- 0.50 or above: near-identical phrasing
- 0.20 to 0.50: similar content but reworded
- below 0.20: very different word choices (which doesn't mean the answer is wrong, just that the LLM phrased it differently)

Don't panic if BLEU is low. That's expected when an LLM rephrases the same answer using different words.

**3. `rouge1_f`. Single-word overlap. Range 0.0 to 1.0.**

The plain-English question: *do the two outputs use the same words, regardless of order?*

ROUGE-1 F1 (Lin, 2004). Measures unigram overlap. Less strict than BLEU because word order doesn't matter.

Interpretation:
- 0.70 or above: strong vocabulary overlap
- 0.40 to 0.70: moderate overlap
- below 0.40: mostly different vocabulary

**4. `rouge2_f`. Two-word phrase overlap. Range 0.0 to 1.0.**

The plain-English question: *do the two outputs share the same two-word phrases?*

ROUGE-2 F1. Same idea as ROUGE-1 but measures bigrams (consecutive word pairs). Stricter than ROUGE-1 because the words have to appear in the same order locally.

Interpretation:
- 0.40 or above: strong phrasal similarity
- 0.15 to 0.40: some shared phrases
- below 0.15: mostly different phrasing

**5. `rougeL_f`. Longest matching subsequence. Range 0.0 to 1.0.**

The plain-English question: *what's the longest stretch of words that appear in both outputs in the same order?*

ROUGE-L F1. Measures the longest common subsequence: words that appear in both outputs in the same order, but allowing other words between them. Captures structural similarity better than BLEU does.

Interpretation:
- 0.60 or above: strong structural alignment
- 0.30 to 0.60: some shared structure
- below 0.30: mostly independent structure

**6. `bertscore_f`. Contextual semantic similarity. Range 0.0 to 1.0.**

The plain-English question: *do the two outputs convey the same ideas, accounting for context?*

BERTScore F1 (Zhang et al., 2020). Computes per-token cosine similarity in a BERT embedding space, matching each token in one output to its most similar token in the other. This is the headline quality metric and correlates better with human judgment than any of the metrics above.

Interpretation:
- 0.95 or above: essentially equivalent outputs
- 0.90 to 0.95: very close, with some phrasing differences
- 0.85 to 0.90: similar core content but noticeable rewording
- below 0.85: meaningful divergence

BERTScore also gives you `bertscore_p` for precision and `bertscore_r` for recall. F1 is the harmonic mean of both, and is the one you should focus on.

#### Which metric should you care about?

It depends what you're trying to measure:

| Use case | Look at this | Threshold to aim for |
|----------|--------------|----------------------|
| General quality check | `bertscore_f` | 0.90 or higher |
| You need exact specific words in the output | `bleu` | 0.40 or higher |
| You need the same vocabulary, word order flexible | `rouge1_f` | 0.60 or higher |
| Cheap sanity check without downloading BERT model | `cosine` | 0.85 or higher |

If you don't want the 1 GB BERTScore model downloaded, skip it:

```python
metrics = compare(original, compressed, out_original, out_compressed,
                  bertscore=False)
```

You still get the other five metrics, which together are very informative.

## A complete example

Here's the whole flow end to end:

```python
from less_tokens import compress, compare
from openai import OpenAI

client = OpenAI()

def ask_gpt(prompt: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return r.choices[0].message.content

original = ("I was wondering if you could please give me a step-by-step "
            "explanation of how to make a really good cup of pour-over "
            "coffee at home using a Hario V60.")

compressed = compress(
    original,
    remove_filler_phrases=1,
    apply_abbreviations=1,
    apply_contractions=1,
    remove_filler_words=1,
    remove_stopwords=1,
)

print(f"Original   ({len(original)} chars): {original}")
print(f"Compressed ({len(compressed)} chars): {compressed}")
print()

out_original   = ask_gpt(original)
out_compressed = ask_gpt(compressed)

metrics = compare(original, compressed, out_original, out_compressed)

print(f"Token reduction: {metrics['compression']['token_reduction_pct']}%")
print(f"BERTScore F1:    {metrics['output_similarity']['bertscore_f']}")
print(f"Cosine sim:      {metrics['output_similarity']['cosine']}")
```

Typical output looks like:

```
Original   (160 chars): I was wondering if you could please give me a step-by-step...
Compressed (95 chars):  step-by-step explanation pour-over coffee home Hario V60.

Token reduction: 40.6%
BERTScore F1:    0.918
Cosine sim:      0.911
```

You saved 40 percent of your tokens and the LLM still gave you essentially the same answer.

## What's happening under the hood

`less-tokens` is built on classical lexical NLP. These are the same techniques used in information retrieval and pre-neural NLP pipelines, just packaged together with sensible defaults and safety guarantees:

- **NLTK** (Loper and Bird, 2002) handles tokenisation, POS tagging, and named entity recognition
- **WordNet** (Miller, 1995) provides the synonym graph
- **tiktoken** counts tokens the same way GPT models do
- **sentence-transformers** computes cosine similarity
- **bert_score** computes BERTScore F1
- **rouge_score** computes ROUGE-1, ROUGE-2, and ROUGE-L
- **NLTK's BLEU** with method-1 smoothing

Every technique is a pure function. Same input plus same flags always produces the same output, byte for byte. Compression itself runs in well under 100 ms on a single CPU core.

## Limitations worth knowing about

A few honest caveats so you know what you're getting.

English only. NLTK stopwords and WordNet are English-language. Multilingual support is open work.

Best on short and medium prompts. Roughly 60 to 2000 characters. Very long retrieval-augmented contexts aren't the target use case. For those, look at learned compressors like [LLMLingua](https://github.com/microsoft/LLMLingua).

The `shorten_synonyms` flag is the riskiest. WordNet sometimes picks topically narrower terms. Don't enable it without testing on your own data first.

Quality is task-dependent. Open-ended Q&A and creative writing tolerate compression well. Commonsense reasoning (HellaSwag-style multiple choice) degrades faster.

`compare()` measures similarity, not correctness. If your original prompt produces a bad LLM output, a similar compressed output is still bad. Make sure your prompts work first, then compress.

## Contributing

Issues and pull requests are very welcome at [github.com/shaminchokshi/less-tokens](https://github.com/shaminchokshi/less-tokens).

To run the test suite locally:

```bash
git clone https://github.com/shaminchokshi/less-tokens.git
cd less-tokens
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT. See [LICENSE](LICENSE).

## Citations

If you're using `less-tokens` in research, the underlying techniques come from these foundational papers:

- **NLTK**: Loper and Bird (2002). *NLTK: The Natural Language Toolkit.* ACL Workshop.
- **WordNet**: Miller (1995). *WordNet: A Lexical Database for English.* CACM 38(11).
- **BERTScore**: Zhang et al. (2020). *BERTScore: Evaluating Text Generation with BERT.* ICLR.
- **BLEU**: Papineni et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation.* ACL.
- **ROUGE**: Lin (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.* ACL Workshop.
- **Sentence-BERT**: Reimers and Gurevych (2019). *Sentence-BERT.* EMNLP.

Related work on prompt compression you might want to compare against:

- **LLMLingua**: Jiang et al. (2023). EMNLP. Learned token pruning with an auxiliary LM, up to 20x compression.
- **Selective Context**: Li et al. (2023). EMNLP. Self-information-based pruning.