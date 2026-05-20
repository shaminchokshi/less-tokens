"""
less_tokens — deterministic, training-free lexical compression for LLM prompts.

Core functions
--------------
- compress(prompt, **flags) -> compressed prompt
- compare(original_prompt, compressed_prompt, original_output, compressed_output)
    -> dict of six similarity metrics + compression stats

Structured (zone-aware) compression
------------------------------------
For prompts that mix free-text instructions with parts that must NOT be
compressed (output formats, JSON schemas) or must be compressed gently (rules):

- compress_structured(instruction=..., rules=..., output_format=..., **flags)
    -> a partially-compressed prompt that protects the parts you care about

Async variants
--------------
- acompress(prompt, **flags)
- acompress_structured(...)

Quick start
-----------
>>> from less_tokens import compress, compare
>>> p = "I was wondering if you could please explain how do I run python script"
>>> compress(p, remove_filler_phrases=1, remove_stopwords=1, apply_contractions=1)
'explain run python script'

>>> from less_tokens import compress_structured
>>> compress_structured(
...     instruction="I was wondering if you could analyse this review.",
...     rules="Never fabricate. Do not include opinions.",
...     output_format='{"sentiment": "positive|negative|neutral"}',
...     remove_stopwords=1, remove_filler_phrases=1,
... )
"""

from .compressor import compress, TECHNIQUES
from .metrics import compare
from .structured import (
    compress_structured,
    acompress,
    acompress_structured,
    CAREFUL_FLAGS,
    FREE_DEFAULT_FLAGS,
    VALID_LEVELS,
)

__version__ = "0.2.0"
__all__ = [
    "compress",
    "compare",
    "compress_structured",
    "acompress",
    "acompress_structured",
    "TECHNIQUES",
    "CAREFUL_FLAGS",
    "FREE_DEFAULT_FLAGS",
    "VALID_LEVELS",
    "__version__",
]