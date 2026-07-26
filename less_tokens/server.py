"""
less_tokens.server — a local compression backend that ships with the package.

This is the on-device counterpart to the hosted API. After ``pip install
less-tokens`` a ``less-tokens-serve`` command is available (wired up via
``[project.scripts]`` in pyproject.toml) that starts this FastAPI app on
127.0.0.1:8000. The VS Code / Cursor extension points at it instead of a
remote server, so a user's prompts and files never leave their machine.

It exposes ONLY the compression endpoints — no accounts, database, billing, or
email. Those belong to the hosted service, not to a local install.

Every endpoint is a thin wrapper over a function already exported from
``less_tokens``:

    POST /smart_compress_batch   -> smart_compress()   (one call per message)
    POST /compress               -> compress()
    POST /compress_structured    -> compress_structured()
    POST /reduce_document        -> reduce_document()
    POST /reduce_image           -> reduce_image_ocr()
    GET  /health                 -> liveness (the extension pings this)
    GET  /techniques             -> the eleven flag names
    GET  /warmup                 -> pre-load NLTK/WordNet so the first call is fast

Run it
------
    less-tokens-serve                 # 127.0.0.1:8000
    less-tokens-serve --port 9000
    python -m less_tokens.server      # same thing

Bind is loopback-only by design: only processes on this machine can reach it.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from less_tokens import (
    TECHNIQUES,
    compress,
    compress_structured,
    reduce_document,
    reduce_image_ocr,
    reduce_image_resize,
    smart_compress,
)

# tiktoken ships with less-tokens, so token counts are exact (GPT cl100k).
try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")

    def _ntok(s: str) -> int:
        return len(_enc.encode(s or ""))
except Exception:  # pragma: no cover - fall back to a rough word count
    def _ntok(s: str) -> int:
        return len((s or "").split())


# ---------------------------------------------------------------------------
# Flag handling
# ---------------------------------------------------------------------------
# Clients send a flat {flag_name: 0/1} object. Only pass through keys that are
# real techniques so an unexpected key can't raise a TypeError inside the lib.
_VALID = set(TECHNIQUES)


def _as_flags(raw: Optional[Dict[str, Any]]) -> Dict[str, int]:
    if not raw:
        return {}
    out: Dict[str, int] = {}
    for k, v in raw.items():
        if k in _VALID:
            out[k] = int(bool(v)) if isinstance(v, bool) else int(v)
    return out


# ---------------------------------------------------------------------------
# Warmup — load NLTK/WordNet once so the first real request isn't slow.
# ---------------------------------------------------------------------------
def _warm() -> None:
    try:
        smart_compress(
            "I was just wondering if you could warm up the models.",
            remove_filler_phrases=1, remove_stopwords=1,
            pos_keep_only=1, lemmatize=1, shorten_synonyms=1,
        )
    except Exception:  # warming is best-effort; never block startup on it
        pass


app = FastAPI(title="less-tokens local API", version="1.0.0")

# The extension runs in a webview whose origin isn't a normal http(s) origin.
# The server is bound to loopback (see main()), so it is only reachable from
# this machine; permissive CORS is safe here and keeps local clients simple.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    await run_in_threadpool(_warm)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: str = "user"
    content: Any = ""  # usually a string; multimodal arrays pass through


class BatchReq(BaseModel):
    messages: List[Message] = []
    flags: Dict[str, Any] = {}


class CompressReq(BaseModel):
    prompt: str = ""
    flags: Dict[str, Any] = {}


class Zone(BaseModel):
    text: str = ""
    level: str = "free"


class StructuredReq(BaseModel):
    zones: List[Zone] = []
    flags: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/techniques")
async def techniques() -> Dict[str, List[str]]:
    return {"techniques": list(TECHNIQUES)}


@app.get("/warmup")
async def warmup() -> Dict[str, str]:
    await run_in_threadpool(_warm)
    return {"status": "warm"}


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------
@app.post("/compress")
async def do_compress(req: CompressReq) -> Dict[str, Any]:
    """Compress a single prompt string."""
    flags = _as_flags(req.flags)
    out = await run_in_threadpool(lambda: compress(req.prompt, **flags))
    return {
        "compressed": out,
        "input_tokens": _ntok(req.prompt),
        "output_tokens": _ntok(out),
    }


@app.post("/smart_compress_batch")
async def do_smart_compress_batch(req: BatchReq) -> Dict[str, List[Any]]:
    """Compress a list of conversation messages, preserving order.

    Uses smart_compress() so code blocks, tables, URLs, math and HTML inside a
    message survive verbatim — only prose is compressed. Non-string content
    (multimodal image/file parts) passes through untouched.
    """
    flags = _as_flags(req.flags)

    def _run() -> List[Any]:
        out: List[Any] = []
        for m in req.messages:
            c = m.content
            if isinstance(c, str) and c.strip():
                out.append(smart_compress(c, **flags))
            else:
                out.append(c)
        return out

    return {"messages": await run_in_threadpool(_run)}


@app.post("/compress_structured")
async def do_compress_structured(req: StructuredReq) -> Dict[str, Any]:
    """Zone-aware compression.

        free       full compression with the chosen flags   (instruction body)
        careful    safe, meaning-preserving techniques only  (rules & constraints)
        protected  returned byte-for-byte, untouched         (JSON schemas, formats)

    Flags only affect ``free`` zones. Returns the assembled prompt plus per-zone
    detail (``return_detail=True``).
    """
    if not req.zones:
        raise HTTPException(status_code=422, detail="Provide at least one zone.")

    zones = [{"text": z.text, "level": z.level} for z in req.zones]
    flags = _as_flags(req.flags)

    result = await run_in_threadpool(
        lambda: compress_structured(zones=zones, return_detail=True, **flags)
    )
    return result  # {"compressed": ..., "zones": [...]}


# ---------------------------------------------------------------------------
# Document / image reduction (write to a temp file, parse, delete)
# ---------------------------------------------------------------------------
async def _save_temp(file: UploadFile) -> str:
    suffix = os.path.splitext(file.filename or "upload")[1] or ""
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name


@app.post("/reduce_document")
async def do_reduce_document(
    file: UploadFile = File(...),
    include_tables: str = Form("true"),
) -> Dict[str, Any]:
    """PDF / Word / text file -> clean Markdown (content only)."""
    keep_tables = include_tables.lower() not in ("false", "0", "no", "")
    path = await _save_temp(file)
    try:
        markdown = await run_in_threadpool(
            lambda: reduce_document(path, include_tables=keep_tables)
        )
    except Exception as exc:  # surface a clean error to the client
        raise HTTPException(
            status_code=422,
            detail=f"Could not read '{file.filename}': {exc}",
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)

    return {
        "filename": file.filename,
        "markdown": markdown,
        "markdown_tokens": _ntok(markdown),
        "markdown_chars": len(markdown),
    }


@app.post("/reduce_image")
async def do_reduce_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Image -> OCR'd text (RapidOCR ships as a dependency)."""
    path = await _save_temp(file)
    try:
        markdown = await run_in_threadpool(lambda: reduce_image_ocr(path))
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not OCR '{file.filename}': {exc}",
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)

    return {
        "filename": file.filename,
        "markdown": markdown,
        "markdown_tokens": _ntok(markdown),
        "markdown_chars": len(markdown),
    }

@app.post("/reduce_image_resize")
async def do_reduce_image_resize(
    file: UploadFile = File(...),
    long_edge: int = Form(512),
    fmt: str = Form("PNG"),            # PNG | JPEG | WEBP
    as_base64: str = Form("false"),    # "true" -> JSON with base64 instead of raw bytes
):
    """Image -> same image, long edge capped at `long_edge` px."""
    import base64
    import io

    from fastapi.responses import JSONResponse, Response

    data = await file.read()           # reduce_image_resize accepts raw bytes; no temp file
    if not data:
        raise HTTPException(status_code=422, detail="Empty upload.")

    try:
        im = await run_in_threadpool(
            lambda: reduce_image_resize(data, long_edge=int(long_edge))
        )

        out_fmt = (fmt or "PNG").upper()
        if out_fmt in ("JPG", "JPEG"):
            out_fmt = "JPEG"
        if out_fmt not in ("PNG", "JPEG", "WEBP"):
            out_fmt = "PNG"
        if out_fmt == "JPEG" and im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGB")     # JPEG has no alpha channel

        buf = io.BytesIO()
        im.save(buf, format=out_fmt)
        raw = buf.getvalue()
        w, h = im.size
    except Exception as exc:
        import traceback
        traceback.print_exc()          # full trace in the server console
        raise HTTPException(
            status_code=422,
            detail=f"Could not resize '{file.filename}': {type(exc).__name__}: {exc}",
        )

    media = "image/jpeg" if out_fmt == "JPEG" else f"image/{out_fmt.lower()}"

    if str(as_base64).lower() in ("true", "1", "yes"):
        return JSONResponse({
            "filename": file.filename,
            "width": w,
            "height": h,
            "bytes": len(raw),
            "media_type": media,
            "image_base64": base64.b64encode(raw).decode("ascii"),
        })

    return Response(
        content=raw,
        media_type=media,
        headers={
            "X-Image-Width": str(w),
            "X-Image-Height": str(h),
            "X-Image-Bytes": str(len(raw)),
            "Content-Disposition": f'inline; filename="resized.{out_fmt.lower()}"',
        },
    )

# ---------------------------------------------------------------------------
# Entry point (console script: less-tokens-serve)
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="less-tokens-serve",
                                     description="Run the local less-tokens backend.")
    parser.add_argument("--host", default=os.environ.get("LESS_TOKENS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("LESS_TOKENS_PORT", "8000")))
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()