"""
less_tokens — deterministic, training-free lexical compression for LLM prompts.

Two public functions:

- :func:`compress(prompt, **flags)` -> compressed prompt
- :func:`compare(original_prompt, compressed_prompt, original_output, compressed_output)`
  -> dict of six similarity metrics + compression stats

Quick start
-----------
>>> from less_tokens import compress, compare
>>> p = "I was wondering if you could please explain how do I run python script"
>>> c = compress(p, remove_filler_phrases=1, remove_stopwords=1,
...              apply_contractions=1)
>>> c
'explain run python script'
>>>
>>> # Make your own LLM calls with whichever provider you like, then:
>>> # metrics = compare(p, c, out_original, out_compressed)
"""

from .compressor import compress, TECHNIQUES
from .metrics import compare

__version__ = "0.1.0"
__all__ = ["compress", "compare", "TECHNIQUES", "__version__"]
