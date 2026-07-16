"""
less_tokens.resize — shrink an image's pixel dimensions so it costs fewer
image tokens, without changing what it shows.

Companion to :func:`reduce_image_ocr` (pull the text out of an image) and
:func:`reduce_document` (pull the content out of a file). Here the input is an
image and the output is the *same* image, just smaller.

Rule (identical to the token-counter tool this was lifted from):
    if the long edge is larger than ``long_edge`` (default 512), scale the
    image down so its long edge is exactly ``long_edge``, preserving aspect
    ratio, then sharpen lightly to recover detail lost in the downscale.
    Images already within the limit are returned unchanged.

At 512px on the long edge, the four major providers (Claude, ChatGPT, Gemini,
Copilot) all hit their minimum image-token floor, so there's nothing to gain
from going larger.

Public entry point is :func:`reduce_image_resize`. End users should normally
call :func:`less_tokens.reduce_image_resize` from the package root.
"""

from __future__ import annotations

import asyncio
import functools
import io
from pathlib import Path
from typing import Union

PathLike = Union[str, "Path"]
# A caller might reasonably hand us any of these.
ImageInput = Union[str, "Path", bytes, bytearray, object]

#: Default long-edge target in pixels.
LONG_EDGE = 512


# ---------------------------------------------------------------------------
# Aspect-ratio box fit (exact logic from the original tool)
# ---------------------------------------------------------------------------

def _box(w: int, h: int, b: int):
    """Fit (w, h) inside a b×b box, preserving aspect ratio.

    Returns the original size unchanged if it already fits.
    """
    if max(w, h) <= b:
        return w, h
    return (b, max(1, round(h * b / w))) if w >= h else (max(1, round(w * b / h)), b)


# ---------------------------------------------------------------------------
# Input loading / normalisation (exact logic from the original tool's load())
# ---------------------------------------------------------------------------

def _load(image: ImageInput):
    """Open and normalise an image exactly the way the source app did.

    Accepts a file path (str / Path), raw ``bytes``, a file-like object (an
    open file / ``io.BytesIO`` / web-upload object), or an existing
    ``PIL.Image``. Honours EXIF orientation and coerces the colour mode so the
    resize/filter steps behave.
    """
    from PIL import Image, ImageOps

    if isinstance(image, Image.Image):
        im = image
    elif isinstance(image, (bytes, bytearray)):
        im = Image.open(io.BytesIO(bytes(image)))
    else:
        # path (str / Path) or any file-like object Image.open understands
        im = Image.open(image)

    try:
        im.seek(0)                      # first frame of animated / multi-page images
    except Exception:
        pass
    im = ImageOps.exif_transpose(im)    # honour EXIF orientation
    if im.mode in ("P", "LA", "PA"):
        im = im.convert("RGBA")
    elif im.mode not in ("RGB", "RGBA", "L"):
        im = im.convert("RGB")
    return im


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def reduce_image_resize(
    image: ImageInput,
    *,
    long_edge: int = LONG_EDGE,
):
    """
    Resize an image down so its long edge is at most ``long_edge`` pixels.

    Image in, image out. If the long edge already fits, the image is returned
    unchanged (aside from EXIF-orientation / colour-mode normalisation). If it
    doesn't, it's scaled down with LANCZOS resampling and lightly sharpened
    with an unsharp mask to recover detail lost in the downscale — the exact
    same treatment as the token-counter tool.

    Parameters
    ----------
    image
        The image to shrink. Accepts a file path (``"photo.png"`` / ``Path``),
        raw image ``bytes``, a file-like object, or a ``PIL.Image``.
    long_edge
        Target size for the longer side, in pixels. Default ``512``.

    Returns
    -------
    PIL.Image.Image
        The resized (or unchanged) image. Call ``.save("out.png")`` on it to
        write it to disk, or ``.size`` to read the new dimensions.

    Examples
    --------
    >>> from less_tokens import reduce_image_resize
    >>> small = reduce_image_resize("big_photo.jpg")
    >>> small.size
    (512, 384)
    >>> small.save("small_photo.png")
    """
    from PIL import Image, ImageFilter

    im = _load(image)
    w, h = im.size

    if max(w, h) > long_edge:
        nw, nh = _box(w, h, long_edge)
        im = im.resize((nw, nh), Image.LANCZOS).filter(
            ImageFilter.UnsharpMask(radius=1.0, percent=60, threshold=2)
        )
    return im


__all__ = ["reduce_image_resize", "areduce_image_resize"]


# ---------------------------------------------------------------------------
# Async variant
# ---------------------------------------------------------------------------
# Image decoding + resizing is CPU-bound and fully synchronous. To avoid
# blocking an asyncio event loop, we run it in the default thread-pool
# executor, mirroring acompress / areduce_document / areduce_image_ocr.

async def areduce_image_resize(
    image: ImageInput,
    **kwargs,
):
    """
    Async wrapper around :func:`reduce_image_resize`.

    Runs the (synchronous, CPU-bound) resize in a thread executor so it doesn't
    block the event loop. Accepts exactly the same arguments as
    :func:`reduce_image_resize`. Handy when resizing many uploaded images
    concurrently inside an async web server.

    Examples
    --------
    >>> import asyncio
    >>> from less_tokens import areduce_image_resize
    >>> async def main():
    ...     return await asyncio.gather(
    ...         areduce_image_resize("a.png"),
    ...         areduce_image_resize("b.jpg"),
    ...     )
    >>> asyncio.run(main())
    """
    loop = asyncio.get_event_loop()
    fn = functools.partial(reduce_image_resize, image, **kwargs)
    return await loop.run_in_executor(None, fn)