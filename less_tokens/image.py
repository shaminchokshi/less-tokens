"""
less_tokens.image — OCR text extraction from images (PNG, JPG, JPEG, ...).

Turn an image that contains text — a screenshot, a scanned page saved as an
image, a photo of a sign, label, or receipt — into plain text using EasyOCR.

This is the image-side companion to :func:`reduce_document` (which handles PDF
and Word files): both take something an LLM can't cheaply or reliably read and
hand you back clean text you can drop into a prompt, store, or compress further
with :func:`less_tokens.compress`.

Design goal: trivial to call. The simplest possible use is::

    from less_tokens import reduce_image_ocr
    text = reduce_image_ocr("screenshot.png")

Everything else (languages, GPU, confidence filtering, per-detection detail) is
an optional keyword argument with a sensible default.

Public entry point is :func:`reduce_image_ocr`. End users should normally call
:func:`less_tokens.reduce_image_ocr` from the package root.
"""

from __future__ import annotations

import asyncio
import functools
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

PathLike = Union[str, "Path"]
# A developer might reasonably hand us any of these.
ImageInput = Union[str, "Path", bytes, bytearray, object]

# Image formats we explicitly advertise. (EasyOCR/Pillow read more than this,
# so we don't hard-reject other extensions — this set is informational.)
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

_OCR_HELP = (
    "Image OCR requires EasyOCR, which ships with less-tokens. If it's missing, "
    "reinstall the package:\n    pip install --force-reinstall less-tokens\n"
    "or install the OCR engine directly:\n    pip install easyocr"
)


# ---------------------------------------------------------------------------
# Reader cache
# ---------------------------------------------------------------------------
# Building an EasyOCR Reader loads the detection + recognition models, which is
# slow (seconds) and memory-heavy. We cache one per (languages, gpu) combo so
# repeated calls with the same settings reuse the loaded models. This mirrors
# the lazy-singleton pattern used for the embedding / BERTScore models in
# less_tokens.metrics.

_READERS: Dict[Tuple[Tuple[str, ...], bool], object] = {}


def _get_reader(languages: Tuple[str, ...], gpu: bool):
    key = (tuple(languages), bool(gpu))
    reader = _READERS.get(key)
    if reader is None:
        try:
            import easyocr
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(_OCR_HELP) from exc
        reader = easyocr.Reader(list(languages), gpu=gpu)
        _READERS[key] = reader
    return reader


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------

def _coerce_image(image: ImageInput):
    """Normalise whatever the developer passed into something EasyOCR can read.

    Accepts a file path (str / Path), raw image bytes, a file-like object
    (open file, ``io.BytesIO``, a Streamlit ``UploadedFile``), a ``PIL.Image``,
    or a numpy array. Returns a path string, bytes, or a numpy array — all of
    which ``Reader.readtext`` understands.
    """
    # Path on disk.
    if isinstance(image, (str, Path)):
        p = Path(image)
        if not p.exists():
            raise FileNotFoundError(f"No such file: {p}")
        return str(p)

    # Raw bytes.
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)

    # numpy array — pass straight through.
    try:
        import numpy as np
        if isinstance(image, np.ndarray):
            return image
    except ImportError:
        pass

    # PIL image — convert to a numpy array.
    try:
        from PIL import Image as _PILImage
        if isinstance(image, _PILImage.Image):
            import numpy as np
            return np.array(image.convert("RGB"))
    except ImportError:
        pass

    # File-like object (has .read()) — read the bytes out.
    if hasattr(image, "read"):
        data = image.read()
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(
                "file-like image must yield bytes from .read(), got "
                f"{type(data).__name__}"
            )
        return bytes(data)

    raise TypeError(
        "image must be a file path, bytes, a file-like object, a PIL.Image, or "
        f"a numpy array; got {type(image).__name__}"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def reduce_image_ocr(
    image: ImageInput,
    *,
    languages: Union[str, Sequence[str]] = ("en",),
    gpu: bool = False,
    min_confidence: float = 0.0,
    paragraph: bool = False,
    separator: str = "\n",
    detail: bool = False,
) -> Union[str, List[Dict]]:
    """
    Extract text from an image using OCR (EasyOCR under the hood).

    The image-side companion to :func:`reduce_document`: give it an image that
    contains text and get back clean, model-ready text.

    Parameters
    ----------
    image
        The image to read. Accepts a file path (``"page.png"`` / ``Path``),
        raw image ``bytes``, a file-like object (an open file, ``BytesIO``, or
        a web-upload object), a ``PIL.Image``, or a numpy array. PNG, JPG, and
        JPEG are the primary targets; BMP, TIFF, and WebP also work.
    languages
        Language code or list of codes to recognise. Default ``("en",)``.
        Latin-script languages can be combined freely; some non-Latin scripts
        (e.g. ``"ch_sim"``, ``"ja"``, ``"ko"``, ``"th"``) may only be used
        alone or alongside ``"en"``.
    gpu
        Use a CUDA GPU if available. Default ``False`` (CPU). Flip to ``True``
        for a large speedup when you have a GPU and a CUDA-enabled PyTorch.
    min_confidence
        Drop detections whose confidence is below this threshold (0.0–1.0).
        Default ``0.0`` keeps everything. Ignored when ``paragraph=True``
        (paragraph grouping does not expose per-line confidence).
    paragraph
        If ``True``, EasyOCR groups nearby detections into paragraph blocks,
        which reads more naturally for dense text. Default ``False``.
    separator
        String joining detected text pieces in the returned string. Default
        is a newline.
    detail
        If ``True``, return a list of per-detection dicts instead of a single
        string — ``{"text", "confidence", "bbox"}`` (``confidence`` is ``None``
        when ``paragraph=True``).

    Returns
    -------
    str
        The extracted text (default), with detections joined by ``separator``.
    list of dict
        If ``detail=True``: one dict per detection.

    Raises
    ------
    FileNotFoundError
        If ``image`` is a path that does not exist.
    ImportError
        If EasyOCR fails to import (it ships with the package, so this should
        only happen on a broken install).
    TypeError
        If ``image`` is not a supported type.

    Examples
    --------
    >>> from less_tokens import reduce_image_ocr
    >>> reduce_image_ocr("receipt.jpg")
    'TOTAL  $42.00\\nThank you for shopping'

    >>> # Pipe straight into the lexical compressor for even fewer tokens
    >>> from less_tokens import reduce_image_ocr, compress
    >>> text = reduce_image_ocr("notice.png")
    >>> lean = compress(text, remove_filler_phrases=1, remove_stopwords=1)

    >>> # Per-detection detail with a confidence floor
    >>> reduce_image_ocr("sign.png", min_confidence=0.5, detail=True)
    [{'text': 'EXIT', 'confidence': 0.99, 'bbox': [[..], [..], [..], [..]]}]
    """
    langs = (languages,) if isinstance(languages, str) else tuple(languages)
    if not langs:
        raise ValueError("languages must contain at least one language code.")

    reader = _get_reader(langs, gpu)
    src = _coerce_image(image)

    raw = reader.readtext(src, paragraph=paragraph)

    detections: List[Dict] = []
    for item in raw:
        if paragraph:
            # EasyOCR returns (bbox, text) when paragraph=True — no confidence.
            bbox, text = item
            conf = None
        else:
            bbox, text, conf = item
            if conf < min_confidence:
                continue
        detections.append({
            "text": (text or "").strip(),
            "confidence": (None if conf is None else float(conf)),
            "bbox": bbox,
        })

    if detail:
        return detections
    return separator.join(d["text"] for d in detections if d["text"])


__all__ = ["reduce_image_ocr", "areduce_image_ocr"]


# ---------------------------------------------------------------------------
# Async variant
# ---------------------------------------------------------------------------
# OCR is CPU-bound (or GPU-bound) and fully synchronous. To avoid blocking an
# asyncio event loop, we run it in the default thread-pool executor, mirroring
# acompress / acompress_structured / areduce_document.

async def areduce_image_ocr(
    image: ImageInput,
    **kwargs,
) -> Union[str, List[Dict]]:
    """
    Async wrapper around :func:`reduce_image_ocr`.

    Runs the (synchronous) OCR call in a thread executor so it doesn't block
    the event loop. Accepts exactly the same arguments as
    :func:`reduce_image_ocr`. Handy when OCR-ing many uploaded images
    concurrently inside an async web server.

    Examples
    --------
    >>> import asyncio
    >>> from less_tokens import areduce_image_ocr
    >>> async def main():
    ...     return await asyncio.gather(
    ...         areduce_image_ocr("a.png"),
    ...         areduce_image_ocr("b.jpg"),
    ...     )
    >>> asyncio.run(main())
    """
    loop = asyncio.get_event_loop()
    fn = functools.partial(reduce_image_ocr, image, **kwargs)
    return await loop.run_in_executor(None, fn)