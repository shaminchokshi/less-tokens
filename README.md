<p align="center">
  <img src="https://raw.githubusercontent.com/shaminchokshi/less-tokens/main/logo.jpeg" alt="less-tokens logo" width="320">
</p>

<h1 align="center">less-tokens</h1>

<p align="center">
  <a href="https://pypi.org/project/less-tokens/"><img src="https://img.shields.io/pypi/v/less-tokens.svg" alt="PyPI version"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
</p>

<p align="center">
  <b>Shrink your LLM prompts by 30 to 40 percent without changing what the model says back.</b>
</p>

`less-tokens` is a small Python library for developers who are paying for tokens and want to stop paying for the ones that don't earn their place. It compresses prompts before you send them to an LLM, stripping out filler words, redundant phrases, and grammatical scaffolding the model doesn't actually need. The result is a shorter prompt that costs less and responds faster, while producing essentially the same answer.

No neural model, no GPU, no API key for the compression itself. It's classical lexical NLP, runs in milliseconds on a laptop CPU, and is fully deterministic — same input, same flags, same output, every time. That matters when you're putting something in a production pipeline.

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

## Contents

- [Why this exists](#why-this-exists)
- [Install](#install)
- [The functions at a glance](#the-functions-at-a-glance)
- [compress: shrink a prompt](#compress-shrink-a-prompt)
- [reduce_document: turn a file into clean markdown](#reduce_document-turn-a-file-into-clean-markdown)
- [reduce_image_ocr: pull text out of an image](#reduce_image_ocr-pull-text-out-of-an-image)
- [compress_structured: protect the parts that matter](#compress_structured-protect-the-parts-that-matter)
- [smart_compress: compress a conversation message](#smart_compress-compress-a-conversation-message)
- [compare: measure the quality tradeoff](#compare-measure-the-quality-tradeoff)
- [Async support](#async-support)
- [A complete example](#a-complete-example)
- [Under the hood](#under-the-hood)
- [Limitations](#limitations)

## Why this exists

If you're calling OpenAI, Anthropic, or any other LLM API at meaningful volume, every token is a line item on your bill. And the prompts your code sends carry a lot of fat that the model quietly ignores:

- *"I was wondering if you could..."* is hedging that adds nothing
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

For most production use, the balanced setting is the sweet spot. Aggressive gets you a bit more compression without much extra quality loss.

There's a second source of waste that bites you the moment your use case involves **files**. When your pipeline hands an LLM a raw PDF or Word document, you're shipping embedded fonts, positioning data, and office XML on top of the words you actually care about. If your use case only needs the *content* of the file, converting it to Markdown first cuts the token count enormously — that's what `reduce_document()` is for.

And when your input is an **image** that has text in it — a screenshot, a scanned page, a photo of a sign or receipt — you can't send it to a text-only model at all, and even a multimodal model charges you image tokens for pixels when all you wanted were the words. `reduce_image_ocr()` runs OCR and hands you back just the text.

## Install

```bash
pip install less-tokens
```

That single command pulls in everything: the compressor, the `compare()` metrics stack, the PDF/Word parsers used by `reduce_document()`, and the OCR engine used by `reduce_image_ocr()`. There are no optional extras to remember and nothing else to wire up.

On first use it downloads about 30 MB of NLTK data automatically. The first time you call `reduce_image_ocr()`, the OCR engine downloads its detection and recognition models (a few hundred MB, cached afterward). If you also call `compare()`, BERTScore will download an additional ~1 GB model the first time. You can skip that with `bertscore=False` if you don't need it.

Using a virtual environment is highly recommended:

```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# macOS or Linux
source .venv/bin/activate

pip install less-tokens
```

## The functions at a glance

The library gives you six functions. Pick the one that matches what your use case actually needs:

| Function | Reach for it when… |
|----------|--------------------|
| `compress()` | You have a prompt string and want it shorter |
| `compress_structured()` | Your prompt mixes free instructions with parts you can't touch, like a JSON output schema or strict rules |
| `reduce_document()` | Your input is a PDF or Word file and you only need its content as text, not a full file upload |
| `reduce_image_ocr()` | Your input is an image (PNG/JPG/JPEG) with text in it and you want the text out |
| `smart_compress()` | You have a single conversation message (user or LLM) that mixes prose and code/tables/URLs and you want only the prose compressed |
| `compare()` | You want to prove the compression didn't change the model's answer |
| `acompress()` / `acompress_structured()` / `areduce_document()` / `areduce_image_ocr()` / `asmart_compress()` | You're doing any of the above inside an async event loop |

The mental model: if you start with a **string**, use `compress()` (or `compress_structured()` if parts are sacred). If you start with a **file**, run `reduce_document()` first to get text, then optionally `compress()` that. If you start with an **image**, run `reduce_image_ocr()` to get text, then optionally `compress()` that. If you're compressing a **conversation history**, use `smart_compress()` on each message. When you want to know what it cost you in quality, run `compare()`.

## compress: shrink a prompt

This is the workhorse. Pass your prompt and any combination of eleven flags. Each flag is `1` to enable or `0` to disable. Bool and string aliases like `True` or `"on"` work too. Defaults are off for everything except whitespace cleanup, so you opt in to exactly the behavior your use case can tolerate.

```python
from less_tokens import compress

short = compress(
    "I was wondering if you could explain this to me.",
    remove_filler_phrases=1,
    remove_stopwords=1,
)
# "explain"
```

### The eleven techniques

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

### What never gets removed

Two categories of words are hard-coded as protected, even at the most aggressive setting, because dropping them would silently corrupt the instruction your code is sending.

First, negations. Words like `not`, `no`, `never`, `nothing`, `nor`, `nobody`, and `cannot`. Dropping these flips the meaning of a sentence, which would be catastrophic in a production prompt. "Do not run this code" becoming "Do run this code" is not a tradeoff anyone wants.

Second, question words. `What`, `why`, `how`, `when`, `where`, `which`. These carry the intent of a query.

Also, if your original prompt ended with a question mark, the compressed version will too. We re-assert question form at the end of the pipeline so it isn't lost during pruning.

### Four presets you can copy

You don't have to figure out which flags to combine. Here are four named recipes mapped to how much risk your use case can absorb:

```python
# SAFE: barely shrinks anything, near-perfect quality preservation.
# Use it when you can't afford any quality risk at all.
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
# at very similar quality. Great for high-volume systems where cost dominates.
compress(prompt,
         pos_keep_only=1,
         preserve_named_entities=1)
# about 35% reduction, 0.91 BERTScore

# MAXIMUM: everything on. About 40% reduction at the cost of some output
# quality. Use when the savings genuinely outweigh the quality hit.
compress(prompt,
         remove_filler_phrases=1, apply_abbreviations=1, apply_contractions=1,
         remove_filler_words=1, remove_stopwords=1, remove_function_words=1,
         pos_keep_only=1, lemmatize=1, shorten_synonyms=1, preserve_named_entities=1)
# about 40% reduction, 0.88 BERTScore
```

## reduce_document: turn a file into clean markdown

If your AI use case only requires the *content* of a PDF or a Word file — and not an entire multimodal text-plus-file upload — don't hand the raw file to the model. A raw `.pdf` or `.docx` is mostly *not content*: it's embedded fonts, per-glyph positioning, style definitions, office XML, page geometry. The model doesn't need any of that, but every byte of it costs you tokens.

Scrape the content instead. `reduce_document()` strips all that unnecessary info — the layout details, the metadata, the fonts and spacing — and keeps only the parts that actually carry meaning: titles, headings, bullet and numbered lists, tables. The result is far fewer tokens, and what you get back is clean Markdown the model reads happily.

And guess what: if your use case permits you to go even leaner, you can run that Markdown straight through `compress()` and shrink it again. File → clean Markdown → compressed text, each step cheaper than the last.

### Basic usage

```python
from less_tokens import reduce_document

markdown = reduce_document("quarterly_report.pdf")
print(markdown)
```

```
# Quarterly Report

## Summary

Revenue grew 18% quarter over quarter, driven mainly by the new
enterprise tier.

## Key figures

| Metric   | Q2    | Q3    |
| ---      | ---   | ---   |
| Revenue  | 4.1M  | 4.8M  |
| Churn    | 2.3%  | 1.9%  |

## Next steps

- Expand the sales team
- Launch in two new regions
```

Drop that Markdown straight into a prompt, store it, or compress it further. It's just text now.

### Parameters

| Parameter | What it does |
|-----------|--------------|
| `path` | Path to the document. PDF, Word, or any plain-text format. |
| `file_type` | Force a parser regardless of extension, e.g. `"pdf"` or `".docx"`. Handy when your pipeline receives files with missing or wrong extensions. |
| `include_tables` | `True` by default. Converts tables to Markdown tables. Set `False` to skip table detection entirely. |

### What it keeps and what it drops

| Kept (the content your model needs) | Dropped (the overhead you were paying for) |
|-------------------------------------|--------------------------------------------|
| Titles and headings (as `#`, `##`, ...) | Margins, indentation, page size |
| Paragraph text | Fonts, font sizes, colors |
| Bullet and numbered lists | Line and paragraph spacing |
| Tables (as Markdown tables) | Absolute positioning, page geometry |
| Bold and italic emphasis | Headers, footers, page numbers |
| Reading order | Office XML and style definitions |

### Supported file types

| Type | Extensions |
|------|------------|
| PDF | `.pdf` |
| Word | `.docx`, `.docm` |
| Plain text / Markdown | `.txt`, `.md`, `.rst`, ... |

All of these work out of the box with a plain `pip install less-tokens` — the PDF and Word parsers ship as part of the package.

### Pairing it with compress

This is the two-step move that gets your file-based use case to the smallest possible footprint: first strip the file down to its content, then compress that content lexically.

```python
from less_tokens import reduce_document, compress

# Step 1: file -> clean markdown (drops layout + metadata)
content = reduce_document("contract.docx")

# Step 2: markdown -> compressed text (drops filler + stopwords)
lean = compress(content,
                remove_filler_phrases=1,
                remove_stopwords=1,
                apply_contractions=1)

# `lean` is now a tiny fraction of the original file's token count.
```

One caution worth building into your code: if the document has **tables** you need intact, aggressive `compress()` flags (stopword removal, POS-keep) will chew up the cell text and pipe structure. Either keep `reduce_document(..., include_tables=False)` if you don't need them, or protect the table with `compress_structured()` (next section).

## reduce_image_ocr: pull text out of an image

When your input is an **image** with text in it — a screenshot, a scanned page exported as a PNG, a photo of a sign, a label, or a receipt — you have the same problem `reduce_document()` solves for PDFs, but worse. A text-only model can't read the image at all, and a multimodal model bills you image tokens for every pixel when all you actually wanted were the words.

`reduce_image_ocr()` runs OCR and hands you back just the text. It's the image-side companion to `reduce_document()`: same idea, same shape — something the model can't cheaply read goes in, clean text comes out.

It's built to be trivial to call. The simplest possible use is one line:

```python
from less_tokens import reduce_image_ocr

text = reduce_image_ocr("screenshot.png")
print(text)
```

```
Invoice #4821
Total due: $1,240.00
Payment terms: Net 30
```

That's it — pass an image, get text. English is the default; everything else is an optional keyword argument.

### What you can pass as the image

You're not locked into file paths. `reduce_image_ocr()` accepts whatever is most convenient in your code:

| Input type | Example |
|------------|---------|
| File path (str or `Path`) | `reduce_image_ocr("page.jpg")` |
| Raw bytes | `reduce_image_ocr(image_bytes)` |
| A file-like object | `reduce_image_ocr(open("p.png", "rb"))` — also a web-upload object or `io.BytesIO` |
| A `PIL.Image` | `reduce_image_ocr(Image.open("p.png"))` |
| A numpy array | `reduce_image_ocr(np_array)` |

PNG, JPG, and JPEG are the primary targets; BMP, TIFF, and WebP also work.

### Parameters

| Parameter | What it does |
|-----------|--------------|
| `image` | The image to read (any of the input types above). |
| `languages` | Language code or list of codes. Default `("en",)`. Latin-script languages combine freely; some non-Latin scripts (`"ch_sim"`, `"ja"`, `"ko"`, `"th"`, ...) may only be used alone or alongside `"en"`. |
| `gpu` | Use a CUDA GPU if available. Default `False` (CPU). Set `True` for a large speedup when you have the hardware and a CUDA-enabled GPU backend. |
| `min_confidence` | Drop detections below this confidence (0.0–1.0). Default `0.0` keeps everything. Ignored when `paragraph=True`. |
| `paragraph` | If `True`, group nearby detections into paragraph blocks for more natural reading order. Default `False`. |
| `separator` | String joining the detected pieces in the returned text. Default is a newline. |
| `detail` | If `True`, return a list of `{"text", "confidence", "bbox"}` dicts instead of a single string. |

### Getting per-detection detail

By default you get one clean string. When you need to filter or inspect what was found — say, to drop low-confidence noise — ask for detail:

```python
from less_tokens import reduce_image_ocr

detections = reduce_image_ocr("sign.png", min_confidence=0.5, detail=True)
for d in detections:
    print(d["confidence"], d["text"])
```

Each dict carries the recognised `text`, the `confidence` (a float, or `None` in `paragraph` mode), and the `bbox` polygon of where it was found on the image.

### Other languages

Pass one code or several. The default is English:

```python
# A single non-English language
reduce_image_ocr("menu.jpg", languages="fr")

# Several Latin-script languages together
reduce_image_ocr("flyer.png", languages=["en", "es", "de"])
```

A note on combining scripts: the OCR engine lets Latin-based languages mix freely, but several non-Latin scripts (Chinese, Japanese, Korean, Thai) can only be used on their own or paired with English. If you select two incompatible scripts you'll get an error from the engine, not from this function.

### Pairing it with compress

Same two-step move as documents — get the text out, then compress it:

```python
from less_tokens import reduce_image_ocr, compress

# Step 1: image -> text (OCR)
text = reduce_image_ocr("handwritten_note.jpg")

# Step 2: text -> compressed text
lean = compress(text,
                remove_filler_phrases=1,
                remove_stopwords=1,
                apply_contractions=1)
```

### A note on performance

The first call builds an OCR reader, which loads the detection and recognition models — slow the first time (and it downloads the weights once). After that the reader is cached per language set, so subsequent calls in the same process are fast. If you're processing a batch, reuse the same `languages` argument so you hit the cache, and reach for the async variant below to run several images concurrently.

## compress_structured: protect the parts that matter

The real prompts your application builds are rarely just instructions. They carry parts that must survive exactly — a JSON output schema, an example the model copies, or rules that break if a single word is dropped. Compressing those parts the same way you compress the instruction body will quietly corrupt your output contract.

`compress_structured()` solves this by letting you assign a compression *level* to each part of the prompt:

| Level | What happens | Use it for |
|-------|--------------|------------|
| `free` | Full compression using your chosen flags | The instruction body |
| `careful` | Only safe, meaning-preserving techniques (no stopword removal, no pruning, no synonyms) | Rules and constraints |
| `protected` | Returned byte-for-byte, untouched | JSON schemas, output formats, examples |

### The easy way: name your sections

The most common case in real code is an instruction, some rules, and an output format. Just pass them as named arguments. The compression flags you pass apply only to the instruction.

```python
from less_tokens import compress_structured

prompt = compress_structured(
    instruction="I was wondering if you could analyse this customer review and tell me how the person is feeling about the product.",
    rules="Do not include any personal opinions. Never guess if you are unsure.",
    output_format='{"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0}',
    remove_stopwords=1,
    remove_filler_phrases=1,
)

print(prompt)
```

Output:

```
analyse customer review tell person feeling product.

don't include any personal opinions. Never guess if you're unsure.

Output format:
{"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0}
```

Look at what happened to each part:

- The **instruction** got compressed hard. "I was wondering if you could" is gone, stopwords are gone.
- The **rules** were compressed gently. "Do not" became "don't" and "you are" became "you're", but the critical words "not" and "Never" survived intact. The meaning is identical.
- The **output format** is byte-for-byte unchanged. Your JSON schema is safe, so your parser downstream won't break.

### The flexible way: explicit zones

When you need full control over ordering, or you want to mix levels in a custom way, pass an explicit list of zones. Each zone is a dict with `text` and `level`, or a simple `(text, level)` tuple.

```python
from less_tokens import compress_structured

prompt = compress_structured(zones=[
    {"text": "I was wondering if you could summarize the following article.", "level": "free"},
    {"text": "Do not exceed 100 words. Never add facts not in the source.",   "level": "careful"},
    {"text": '{"summary": "...", "word_count": N}',                           "level": "protected"},
])
```

### Why "careful" mode exists

This is the most important design decision in the library, and the one that keeps it safe to drop into production. Rules carry meaning in their small words. If you ran full stopword removal on "Do not exceed 100 words" you might get "exceed 100 words", which is the exact opposite instruction. So careful mode disables every technique that could flip or blur meaning:

| Technique | free | careful | Why careful skips it |
|-----------|:----:|:-------:|----------------------|
| Filler phrase removal | yes | yes | Safe, only removes hedging |
| Contractions | yes | yes | Safe, "do not" to "don't" keeps meaning |
| Filler word removal | yes | yes | Safe, "basically" carries no logic |
| Stopword removal | yes | no | Can drop words that matter in a constraint |
| Function word pruning | yes | no | Can drop "not", "all", "only" type logic |
| POS-keep | yes | no | Too aggressive for precise rules |
| Lemmatize | yes | no | Can blur tense or number that matters |
| Synonym shortening | yes | no | Can pick a narrower or wrong synonym |

If even careful mode feels too risky for a specific rule, mark it `protected` and it won't be touched at all.

### Seeing what changed

Pass `return_detail=True` to get a breakdown of every zone — useful when you're debugging why an output contract broke:

```python
result = compress_structured(
    instruction="Please analyse this in detail.",
    output_format='{"x": 1}',
    remove_stopwords=1,
    return_detail=True,
)

print(result["compressed"])     # the assembled prompt
for zone in result["zones"]:
    print(zone["level"], zone["original_len"], "->", zone["compressed_len"])
```

## smart_compress: compress a conversation message

When you are working with a multi-turn conversation history — a list of user inputs and LLM responses — you cannot run `compress()` directly on each message. LLM responses routinely mix natural language prose with elements that must never be touched: fenced code blocks, inline code, Markdown tables, URLs, math expressions, and HTML. Compressing those would corrupt the code or break the output contract.

`smart_compress()` solves this. It parses each message, automatically detects every protected zone, and compresses only the natural language prose around them. Apply it to every message in your history:

```python
from less_tokens import smart_compress

compressed_history = [
    smart_compress(msg, remove_filler_phrases=1, remove_stopwords=1)
    for msg in conversation
]
```

### What is protected and what is compressed

| Protected (returned verbatim) | Compressed (natural language only) |
|-------------------------------|-------------------------------------|
| Fenced code blocks (` ``` `) | Paragraph prose |
| Indented code blocks | Heading text (the `#` prefix is kept) |
| Inline code (`` `backticks` ``) | List-item text (the `-` / `1.` marker is kept) |
| Markdown tables | |
| Bare URLs and Markdown links | |
| Math blocks (`$$...$$`) and inline math (`$...$`) | |
| HTML tags | |
| JSON / array blocks | |

### Debugging with return_segments

Pass `return_segments=True` to see exactly what was protected and what was compressed:

```python
result = smart_compress(msg, remove_filler_phrases=1, return_segments=True)
print(result["compressed"])
for seg in result["segments"]:
    print(seg["kind"], "->", seg["original"][:40])
```

## compare: measure the quality tradeoff

Compression is only worth shipping if the LLM still produces the answer your use case depends on. `compare()` quantifies that across six different similarity metrics, so you can decide based on numbers instead of vibes.

You make the LLM calls yourself, with whichever provider your stack uses. `compare()` only looks at the four strings: original prompt, compressed prompt, output from original, output from compressed.

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

### What you get back

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

### What each of the six metrics actually means

All six measure the same thing from different angles: how similar is the LLM's response to the compressed prompt, compared to its response to the original. Each one captures a different notion of "similar", and which one you care about depends on what your use case promises its users.

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

### Which metric should you care about?

It depends on what your use case is actually promising:

| Use case | Look at this | Threshold to aim for |
|----------|--------------|----------------------|
| General quality check | `bertscore_f` | 0.90 or higher |
| You need exact specific words in the output | `bleu` | 0.40 or higher |
| You need the same vocabulary, word order flexible | `rouge1_f` | 0.60 or higher |
| Cheap sanity check without downloading BERT model | `cosine` | 0.85 or higher |

If your environment can't afford the 1 GB BERTScore model download, skip it:

```python
metrics = compare(original, compressed, out_original, out_compressed,
                  bertscore=False)
```

You still get the other five metrics, which together are very informative.

## Async support

If your use case runs inside an async web server or processes prompts in large concurrent batches, the async functions run the (CPU-bound, pure-Python) work in a thread executor so they never block your event loop. They take exactly the same arguments as their synchronous counterparts.

| Sync | Async |
|------|-------|
| `compress()` | `acompress()` |
| `compress_structured()` | `acompress_structured()` |
| `reduce_document()` | `areduce_document()` |
| `reduce_image_ocr()` | `areduce_image_ocr()` |
| `smart_compress()` | `asmart_compress()` |

```python
import asyncio
from less_tokens import (acompress, acompress_structured,
                         areduce_document, areduce_image_ocr, asmart_compress)

async def main():
    # Async version of compress()
    short = await acompress(
        "I was wondering if you could help me with this",
        remove_filler_phrases=1, remove_stopwords=1,
    )

    # Async version of compress_structured()
    prompt = await acompress_structured(
        instruction="Please analyse this text in detail.",
        output_format='{"result": "..."}',
        remove_stopwords=1,
    )

    # Async version of reduce_document()
    content = await areduce_document("report.pdf")

    # Async version of reduce_image_ocr()
    caption = await areduce_image_ocr("screenshot.png")

    # Async version of smart_compress() — compress a full conversation history concurrently
    compressed_history = await asyncio.gather(
        *[asmart_compress(msg, remove_filler_phrases=1, remove_stopwords=1)
          for msg in conversation]
    )

    # Or reduce a batch of uploaded files / images at once
    docs = await asyncio.gather(
        areduce_document("a.pdf"),
        areduce_document("b.docx"),
        areduce_image_ocr("c.png"),
    )

asyncio.run(main())
```

This is what you want when you're compressing inside FastAPI or aiohttp, or reducing a batch of user-uploaded files and images concurrently.

## A complete example

Here's the whole flow end to end for a file-based use case: a user uploads a review as a PDF, you pull out just the content, compress the wordy instruction, protect the output schema, and verify with `compare()` that the model still returns the same structured answer your code depends on.

```python
from less_tokens import reduce_document, compress_structured, compare
from openai import OpenAI

client = OpenAI()

def ask_gpt(prompt: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return r.choices[0].message.content

# Step 0: a user uploaded a review as a PDF. Pull out just the content.
review = reduce_document("customer_review.pdf")

# Build the original prompt the long way
original = (
    "I was wondering if you could please analyse the following customer "
    f"review and tell me the overall sentiment.\n\n{review}\n\n"
    "Do not include any personal opinions. Never guess if you are unsure.\n\n"
    'Output format:\n{"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0}'
)

# Compress it, protecting the rules and output format
compressed = compress_structured(zones=[
    ("I was wondering if you could please analyse the following customer "
     f"review and tell me the overall sentiment.\n\n{review}", "free"),
    ("Do not include any personal opinions. Never guess if you are unsure.", "careful"),
    ('{"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0}', "protected"),
],
    remove_filler_phrases=1,
    remove_stopwords=1,
)

print(f"Original   ({len(original)} chars)")
print(f"Compressed ({len(compressed)} chars)")
print()

out_original   = ask_gpt(original)
out_compressed = ask_gpt(compressed)

metrics = compare(original, compressed, out_original, out_compressed)

print(f"Token reduction: {metrics['compression']['token_reduction_pct']}%")
print(f"BERTScore F1:    {metrics['output_similarity']['bertscore_f']}")
```

You pulled the content out of the file, shrank the wordy instruction, kept the rules safe, kept the JSON schema exact, and confirmed with `compare()` that the model still returns the same structured answer. That's the full library working together on one realistic use case.

If the user had uploaded a **screenshot** instead of a PDF, the only change is the first line — swap `reduce_document("customer_review.pdf")` for `reduce_image_ocr("customer_review.png")` and the rest of the pipeline is identical.

## Under the hood

`less-tokens` is built on classical lexical NLP — the same techniques used in information retrieval and pre-neural NLP pipelines, packaged together with sensible defaults and safety guarantees so you can drop them into real code:

- **NLTK** (Loper and Bird, 2002) handles tokenisation, POS tagging, and named entity recognition
- **WordNet** (Miller, 1995) provides the synonym graph
- **tiktoken** counts tokens the same way GPT models do
- **sentence-transformers** computes cosine similarity
- **bert_score** computes BERTScore F1
- **rouge_score** computes ROUGE-1, ROUGE-2, and ROUGE-L
- **NLTK's BLEU** with method-1 smoothing
- **PyMuPDF** gives us the raw text spans (with font size and bold/italic flags) and table regions of a PDF; `reduce_document()` reconstructs the Markdown from those primitives itself — headings from relative font size, emphasis from span flags, lists from leading glyphs, and reading order from on-page position
- **python-docx** reads Word documents in true reading order, which `reduce_document()` maps to Markdown headings, lists, and tables
- **OCR** powers `reduce_image_ocr()`; the reader is cached per language set so the (heavy) models load once per process and are reused on every subsequent call

Every compression technique is a pure function. Same input plus same flags always produces the same output, byte for byte — which is exactly what you want when the thing sits in a deterministic pipeline. Compression itself runs in well under 100 ms on a single CPU core, and document reduction is deterministic too: the same file always produces the same Markdown. (OCR is the one stage that depends on a learned model rather than pure lexical rules, so treat its output as best-effort recognition rather than a deterministic transform.)

## Limitations

A few honest caveats so you know whether this fits your use case before you build on it.

English only for the *lexical* techniques. NLTK stopwords and WordNet are English-language, so `compress()` is English-only. (OCR via `reduce_image_ocr()` supports many languages — that's a separate engine.) Multilingual compression is open work.

Best on short and medium prompts. Roughly 60 to 2000 characters. Very long retrieval-augmented contexts aren't the target use case. For those, look at learned compressors like [LLMLingua](https://github.com/microsoft/LLMLingua).

The `shorten_synonyms` flag is the riskiest. WordNet sometimes picks topically narrower terms. Don't enable it in production without testing on your own data first.

Quality is task-dependent. Open-ended Q&A and creative writing tolerate compression well. Commonsense reasoning (HellaSwag-style multiple choice) degrades faster.

`compare()` measures similarity, not correctness. If your original prompt produces a bad LLM output, a similar compressed output is still bad. Make sure your prompts work first, then compress.

`reduce_document()` reads text, not pixels. Scanned PDFs or image-only documents have no extractable text layer, so they come back empty — that's exactly what `reduce_image_ocr()` is for. `reduce_document()` also doesn't handle the old binary `.doc` format (convert to `.docx` first), and complex multi-column or heavily nested table layouts may not map cleanly onto Markdown.

`reduce_image_ocr()` is only as good as OCR. Recognition quality depends on image resolution, contrast, and how clean the text is; low-resolution, skewed, or noisy images yield weaker results, and stylised or handwritten text is harder than printed text. It is not deterministic in the way the lexical functions are, and the first call downloads the OCR models (a few hundred MB). For perfectly clean digital PDFs, prefer `reduce_document()` — OCR is for when the text only exists as pixels.

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

MIT. See [LICENSE](https://github.com/shaminchokshi/less-tokens/blob/main/LICENSE).

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