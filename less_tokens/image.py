"""
less_tokens.image — OCR text extraction from images (PNG, JPG, JPEG, ...).

Turn an image that contains text — a screenshot, a scanned page saved as an
image, a photo of a sign, label, or receipt — into plain text using RapidOCR
(ONNXRuntime-based PaddleOCR models).

This is the image-side companion to :func:`reduce_document` (which handles PDF
and Word files): both take something an LLM can't cheaply or reliably read and
hand you back clean text you can drop into a prompt, store, or compress further
with :func:`less_tokens.compress`.

Design goal: trivial to call. The simplest possible use is::

    from less_tokens import reduce_image_ocr
    text = reduce_image_ocr("screenshot.png")

Everything else (GPU, confidence filtering, paragraph grouping, per-detection
detail) is an optional keyword argument with a sensible default.

Public entry point is :func:`reduce_image_ocr`. End users should normally call
:func:`less_tokens.reduce_image_ocr` from the package root.
"""

from __future__ import annotations

import asyncio
import functools
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

PathLike = Union[str, "Path"]
# A developer might reasonably hand us any of these.
ImageInput = Union[str, "Path", bytes, bytearray, object]

# Image formats we explicitly advertise. (RapidOCR/Pillow read more than this,
# so we don't hard-reject other extensions — this set is informational.)
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

_OCR_HELP = (
    "Image OCR requires RapidOCR, which ships with less-tokens. If it's missing, "
    "reinstall the package:\n    pip install --force-reinstall less-tokens\n"
    "or install the OCR engine directly:\n    pip install rapidocr-onnxruntime"
)


# ---------------------------------------------------------------------------
# Engine cache
# ---------------------------------------------------------------------------
# Building a RapidOCR engine loads the detection + recognition (+ angle-cls)
# ONNX models, which is slow (seconds) and memory-heavy. We cache one per `gpu`
# setting so repeated calls reuse the loaded models. Unlike EasyOCR, RapidOCR
# does not switch languages per call — its default models cover Latin scripts
# and Chinese, and other languages are selected via model files at install
# time — so the cache is keyed on `gpu` only. This mirrors the lazy-singleton
# pattern used for the embedding / BERTScore models in less_tokens.metrics.

_ENGINES: Dict[bool, object] = {}


def _import_rapidocr():
    """Import RapidOCR from whichever package flavour is installed.

    Returns ``(RapidOCR_class, flavour)`` where flavour is ``"onnxruntime"``
    (the classic ``rapidocr_onnxruntime`` package, list-style results) or
    ``"rapidocr"`` (the newer unified ``rapidocr`` package, object results).
    """
    try:
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR, "onnxruntime"
    except ImportError:
        pass
    try:
        from rapidocr import RapidOCR
        return RapidOCR, "rapidocr"
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_OCR_HELP) from exc


def _get_engine(gpu: bool):
    engine = _ENGINES.get(bool(gpu))
    if engine is None:
        RapidOCR, flavour = _import_rapidocr()
        if gpu and flavour == "onnxruntime":
            # CUDA execution needs onnxruntime-gpu; fall back to CPU if the
            # constructor doesn't accept the flags on this RapidOCR version.
            try:
                engine = RapidOCR(
                    det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True
                )
            except TypeError:
                engine = RapidOCR()
        else:
            engine = RapidOCR()
        _ENGINES[bool(gpu)] = engine
    return engine


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------

def _coerce_image(image: ImageInput):
    """Normalise whatever the developer passed into something RapidOCR can read.

    Accepts a file path (str / Path), raw image bytes, a file-like object
    (open file, ``io.BytesIO``, a Streamlit ``UploadedFile``), a ``PIL.Image``,
    or a numpy array. Returns a path string, bytes, or a numpy array — all of
    which RapidOCR's ``__call__`` understands.
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
# Result normalisation
# ---------------------------------------------------------------------------

def _normalise_result(raw) -> List[Tuple]:
    """Return a list of ``(bbox, text, score)`` from either RapidOCR flavour.

    * ``rapidocr_onnxruntime`` -> ``raw`` is a list of ``[box, text, score]``
      (already unpacked from the ``(result, elapse)`` tuple by the caller), or
      ``None`` when nothing was detected.
    * ``rapidocr`` (newer) -> ``raw`` is an output object exposing ``boxes``,
      ``txts``, and ``scores`` attributes.
    """
    if raw is None:
        return []

    # Newer unified `rapidocr` package: object with parallel attributes.
    if hasattr(raw, "txts") and hasattr(raw, "boxes"):
        boxes = list(raw.boxes) if raw.boxes is not None else []
        txts = list(raw.txts) if raw.txts is not None else []
        scores = list(raw.scores) if raw.scores is not None else []
        out = []
        for i, text in enumerate(txts):
            box = boxes[i] if i < len(boxes) else None
            score = scores[i] if i < len(scores) else None
            out.append((box, text, score))
        return out

    # Classic list-style result.
    out = []
    for item in raw:
        box = item[0] if len(item) > 0 else None
        text = item[1] if len(item) > 1 else ""
        score = item[2] if len(item) > 2 else None
        out.append((box, text, score))
    return out


def _bbox_to_list(box) -> Optional[List]:
    if box is None:
        return None
    if hasattr(box, "tolist"):
        return box.tolist()
    return [list(pt) for pt in box]


# ---------------------------------------------------------------------------
# Optional paragraph grouping
# ---------------------------------------------------------------------------
# RapidOCR returns line-level detections and has no native paragraph mode, so
# we provide a light vertical-gap heuristic: lines sorted top-to-bottom are
# merged into a block whenever the gap to the next line is small relative to
# the median line height. Confidence is set to None for merged blocks, matching
# the previous paragraph-mode contract.

def _bbox_bounds(bbox) -> Tuple[float, float, float, float]:
    xs = [float(p[0]) for p in bbox]
    ys = [float(p[1]) for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def _group_paragraphs(detections: List[Dict]) -> List[Dict]:
    if not detections:
        return []

    enriched = []
    for d in detections:
        if d["bbox"]:
            x0, y0, _x1, y1 = _bbox_bounds(d["bbox"])
        else:
            x0 = y0 = y1 = 0.0
        enriched.append((y0, y1, x0, d))
    enriched.sort(key=lambda e: (e[0], e[2]))

    heights = sorted(y1 - y0 for y0, y1, _x0, _d in enriched if y1 > y0)
    med_h = heights[len(heights) // 2] if heights else 0.0
    gap_thresh = 0.6 * med_h

    groups: List[List[Dict]] = []
    current: List[Dict] = []
    prev_bottom: Optional[float] = None
    for y0, y1, _x0, d in enriched:
        if current and prev_bottom is not None and (y0 - prev_bottom) > gap_thresh:
            groups.append(current)
            current = []
        current.append(d)
        prev_bottom = y1
    if current:
        groups.append(current)

    merged: List[Dict] = []
    for g in groups:
        text = " ".join(x["text"] for x in g if x["text"])
        pts = [p for x in g if x["bbox"] for p in x["bbox"]]
        if pts:
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            bbox = [[min(xs), min(ys)], [max(xs), min(ys)],
                    [max(xs), max(ys)], [min(xs), max(ys)]]
        else:
            bbox = None
        merged.append({"text": text, "confidence": None, "bbox": bbox})
    return merged


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
    Extract text from an image using OCR (RapidOCR under the hood).

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
        Accepted for API compatibility. RapidOCR does not switch languages per
        call — its default models recognise Latin scripts and Chinese, and
        other languages are selected via model files at install time — so this
        argument is validated but does not change the loaded models. Default
        ``("en",)``.
    gpu
        Use a CUDA GPU if available. Default ``False`` (CPU). Flip to ``True``
        for a large speedup when you have a GPU and ``onnxruntime-gpu``
        installed. Silently falls back to CPU if GPU execution isn't available.
    min_confidence
        Drop detections whose confidence is below this threshold (0.0–1.0).
        Default ``0.0`` keeps everything. Ignored when ``paragraph=True``
        (paragraph grouping does not expose per-line confidence).
    paragraph
        If ``True``, nearby line detections are merged into paragraph blocks
        (a vertical-gap heuristic) for more natural reading order. Default
        ``False``.
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
        If RapidOCR fails to import (it ships with the package, so this should
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

    engine = _get_engine(gpu)
    src = _coerce_image(image)

    result = engine(src)
    # rapidocr_onnxruntime returns (result, elapse); the newer rapidocr package
    # returns a single output object.
    if isinstance(result, tuple) and len(result) == 2:
        result = result[0]
    raw = _normalise_result(result)

    detections: List[Dict] = []
    for box, text, score in raw:
        text = (text or "").strip()
        conf = None if score is None else float(score)
        if not paragraph and conf is not None and conf < min_confidence:
            continue
        detections.append({
            "text": text,
            "confidence": conf,
            "bbox": _bbox_to_list(box),
        })

    if paragraph:
        detections = _group_paragraphs(detections)

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