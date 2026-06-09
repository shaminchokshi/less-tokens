"""
less_tokens — deterministic, training-free lexical compression for LLM prompts.

Core functions
--------------
- compress(prompt, **flags) -> compressed prompt
- compare(original_prompt, compressed_prompt, original_output, compressed_output)
    -> dict of six similarity metrics + compression stats

Document reduction
------------------
For turning an uploaded PDF / Word / text file into compact, model-friendly
Markdown (content only — no layout, fonts, or metadata):

- reduce_document(path) -> markdown string

Structured (zone-aware) compression
------------------------------------
For prompts that mix free-text instructions with parts that must NOT be
compressed (output formats, JSON schemas) or must be compressed gently (rules):

- compress_structured(instruction=..., rules=..., output_format=..., **flags)
    -> a partially-compressed prompt that protects the parts you care about

Smart compression for conversation messages
-------------------------------------------
For compressing a single message from a multi-turn conversation history,
automatically detecting and protecting code blocks, tables, URLs, math,
and HTML while compressing only the natural language prose:

- smart_compress(message, **flags) -> compressed message

Async variants
--------------
- acompress(prompt, **flags)
- acompress_structured(...)
- asmart_compress(message, **flags)

Quick start
-----------
>>> from less_tokens import compress, compare
>>> p = "I was wondering if you could please explain how do I run python script"
>>> compress(p, remove_filler_phrases=1, remove_stopwords=1, apply_contractions=1)
'explain run python script'

>>> from less_tokens import reduce_document
>>> reduce_document("contract.pdf")[:40]
'# Master Services Agreement\\n## Term ...'

>>> from less_tokens import compress_structured
>>> compress_structured(
...     instruction="I was wondering if you could analyse this review.",
...     rules="Never fabricate. Do not include opinions.",
...     output_format='{"sentiment": "positive|negative|neutral"}',
...     remove_stopwords=1, remove_filler_phrases=1,
... )

>>> from less_tokens import smart_compress
>>> smart_compress(
...     "I was wondering if you could explain this.\\n\\n```python\\nprint('hi')\\n```",
...     remove_filler_phrases=1, remove_stopwords=1,
... )
"""

from .compressor import compress, TECHNIQUES
from .metrics import compare
from .document import reduce_document, areduce_document
from .structured import (
    compress_structured,
    acompress,
    acompress_structured,
    CAREFUL_FLAGS,
    FREE_DEFAULT_FLAGS,
    VALID_LEVELS,
)
from .smart_compress import smart_compress, asmart_compress

__version__ = "0.4.0"
__all__ = [
    "compress",
    "compare",
    "reduce_document",
    "areduce_document",
    "compress_structured",
    "acompress",
    "acompress_structured",
    "smart_compress",
    "asmart_compress",
    "TECHNIQUES",
    "CAREFUL_FLAGS",
    "FREE_DEFAULT_FLAGS",
    "VALID_LEVELS",
    "__version__",
]