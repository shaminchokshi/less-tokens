"""
less_tokens.document — turn an uploaded document (PDF, Word, or plain text)
into clean Markdown, keeping the *content* (titles, headings, lists, tables)
and dropping everything that only describes *layout* (margins, fonts, line
spacing, page geometry, absolute positioning, XML scaffolding).

Why this exists
---------------
Uploading a raw ``.pdf`` or ``.docx`` to an LLM ships a lot of bytes that are
not the actual content: embedded fonts, positioning data, style definitions,
office XML. If all you want is *what the document says*, converting it to
Markdown first is dramatically cheaper in tokens while keeping the structure
the model actually needs — headings, bullet points, numbered lists, tables.

This is the only part of ``less_tokens`` that touches binary files. Its parsers
(PyMuPDF for PDF, python-docx for Word) ship as part of the package, so a plain
``pip install less-tokens`` is all that's needed.

Public entry point is :func:`reduce_document`. End users should normally call
:func:`less_tokens.reduce_document` from the package root.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import io
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

PathLike = Union[str, "Path"]

# ---------------------------------------------------------------------------
# Recognised extensions
# ---------------------------------------------------------------------------

_PDF_EXTS = {".pdf"}
_DOCX_EXTS = {".docx", ".docm"}
# Anything already textual is passed through with light Markdown cleanup.
_TEXT_EXTS = {".md", ".markdown", ".txt", ".text", ".rst", ".rtf", ".log", ""}

_NUMBERED_RE = re.compile(r"^\d+\.\s")

_PDF_HELP = (
    "Reading PDF files requires PyMuPDF, which ships with less-tokens. If it's "
    "missing, reinstall the package:\n    pip install --force-reinstall less-tokens\n"
    "or install the parser directly:\n    pip install pymupdf"
)
_DOCX_HELP = (
    "Reading Word files requires python-docx, which ships with less-tokens. If "
    "it's missing, reinstall the package:\n"
    "    pip install --force-reinstall less-tokens\n"
    "or install the parser directly:\n    pip install python-docx"
)


# ---------------------------------------------------------------------------
# Shared Markdown cleanup
# ---------------------------------------------------------------------------

def _normalize_markdown(text: str) -> str:
    """Trim trailing spaces per line and collapse runs of blank lines."""
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


# ---------------------------------------------------------------------------
# PDF -> Markdown
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PDF -> Markdown  (built from scratch on top of PyMuPDF's raw text spans)
# ---------------------------------------------------------------------------
# We do NOT delegate to a high-level "to_markdown" helper. We pull the raw
# layout out of the PDF ourselves and reconstruct structure:
#   * headings    -> inferred from font size relative to the body text size
#   * bold/italic -> inferred from span font flags / font names
#   * lists       -> inferred from the leading glyph of a line ("•", "1.", ...)
#   * tables      -> located with PyMuPDF's table finder, emitted as md tables
#   * reading order -> elements are sorted by their vertical position on a page
# All of the Markdown assembly below is our own.

# PyMuPDF span "flags" bitfield: bit 1 (=2) italic, bit 4 (=16) bold.
_FLAG_ITALIC = 1 << 1
_FLAG_BOLD = 1 << 4

# Glyphs that commonly start an unordered list item in a PDF.
_BULLET_GLYPHS = "•‣◦▪·●○∙◆-*–—"
_BULLET_RE = re.compile(rf"^\s*[{re.escape(_BULLET_GLYPHS)}]\s+(.*)$")
_ORDERED_NUM_RE = re.compile(r"^\s*\d{1,3}[.)]\s+(.*)$")
_ORDERED_ALPHA_RE = re.compile(r"^\s*[a-z][)]\s+(.*)$")


def _span_is_bold(span) -> bool:
    if span.get("flags", 0) & _FLAG_BOLD:
        return True
    name = (span.get("font") or "").lower()
    return "bold" in name or "black" in name or "heavy" in name or "semibold" in name


def _span_is_italic(span) -> bool:
    if span.get("flags", 0) & _FLAG_ITALIC:
        return True
    name = (span.get("font") or "").lower()
    return "italic" in name or "oblique" in name


def _line_plaintext(spans) -> str:
    return "".join(s.get("text", "") for s in spans)


def _line_size(spans) -> float:
    """Representative font size of a line = size of its longest text span."""
    real = [s for s in spans if s.get("text", "").strip()]
    if not real:
        return 0.0
    longest = max(real, key=lambda s: len(s["text"]))
    return float(longest.get("size", 0.0))


def _line_markdown(spans) -> str:
    """Reconstruct a line's text, wrapping bold/italic runs in Markdown marks.

    Adjacent spans that share the same style are merged first so we don't emit
    broken sequences like ``**a****b**``.
    """
    segs: List[Tuple[str, bool, bool]] = []
    for s in spans:
        txt = s.get("text", "")
        if txt == "":
            continue
        bold, ital = _span_is_bold(s), _span_is_italic(s)
        if segs and segs[-1][1] == bold and segs[-1][2] == ital:
            segs[-1] = (segs[-1][0] + txt, bold, ital)
        else:
            segs.append((txt, bold, ital))

    out: List[str] = []
    for txt, bold, ital in segs:
        core = txt.strip()
        if core and (bold or ital):
            lead = txt[: len(txt) - len(txt.lstrip())]
            trail = txt[len(txt.rstrip()):]
            mark = "***" if (bold and ital) else ("**" if bold else "*")
            out.append(f"{lead}{mark}{core}{mark}{trail}")
        else:
            out.append(txt)
    return "".join(out).strip()


def _bullet_split(md_text: str):
    """Return ("ul"|"ol", content) if the line looks like a list item, else None."""
    m = _BULLET_RE.match(md_text)
    if m:
        return "ul", m.group(1).strip()
    m = _ORDERED_NUM_RE.match(md_text)
    if m:
        return "ol", m.group(1).strip()
    m = _ORDERED_ALPHA_RE.match(md_text)
    if m:
        return "ol", m.group(1).strip()
    return None


def _build_heading_map(doc) -> dict:
    """Map font sizes that are clearly larger than the body to heading levels.

    Body size = the size covering the most characters across the document.
    Any distinct size meaningfully larger than that becomes a heading; the
    largest is ``#``, the next ``##``, and so on (capped at 6).
    """
    char_by_size: dict = {}
    for page in doc:
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "")
                    n = len(txt.strip())
                    if n:
                        size = round(float(span.get("size", 0.0)), 1)
                        char_by_size[size] = char_by_size.get(size, 0) + n
    if not char_by_size:
        return {}
    body = max(char_by_size, key=char_by_size.get)
    heading_sizes = sorted(
        (s for s in char_by_size
         if s >= body + 0.5 and s >= body * 1.10),
        reverse=True,
    )
    return {size: min(i + 1, 6) for i, size in enumerate(heading_sizes)}


def _clean_cell(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).replace("|", "\\|")


def _pdf_table_to_markdown(table) -> str:
    try:
        data = table.extract()
    except Exception:
        return ""
    rows = [[_clean_cell(c) for c in row] for row in data]
    rows = [r for r in rows if any(cell for cell in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, *body = rows
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * width) + " |"]
    for r in body:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _rect_mostly_inside(inner, outer, threshold: float = 0.6) -> bool:
    """True if `inner` (x0,y0,x1,y1) overlaps `outer` by >= threshold of its area."""
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    ox = max(0.0, min(ix1, ox1) - max(ix0, ox0))
    oy = max(0.0, min(iy1, oy1) - max(iy0, oy0))
    overlap = ox * oy
    area = max((ix1 - ix0) * (iy1 - iy0), 1e-6)
    return overlap / area >= threshold


def _process_text_block(block, heading_map) -> List[Tuple[float, Tuple]]:
    """Turn one text block into ordered (y, element) tuples.

    element is one of:
        ("heading", (level, text))
        ("list",    ("ul"|"ol", text))
        ("para",    text)
    Wrapped lines of an ordinary paragraph are merged into a single para.
    """
    results: List[Tuple[float, Tuple]] = []
    buf: List[str] = []
    buf_y: Optional[float] = None

    def flush():
        nonlocal buf, buf_y
        if buf:
            para = " ".join(buf).strip()
            if para:
                results.append((buf_y, ("para", para)))
        buf, buf_y = [], None

    for line in block.get("lines", []):
        spans = line.get("spans", [])
        md = _line_markdown(spans)
        if not md.strip():
            continue
        y0 = line["bbox"][1]
        size = round(_line_size(spans), 1)
        level = heading_map.get(size)
        bullet = _bullet_split(md)

        if level:
            flush()
            results.append((y0, ("heading", (level, _line_plaintext(spans).strip()))))
        elif bullet is not None:
            flush()
            results.append((y0, ("list", bullet)))
        else:
            if buf_y is None:
                buf_y = y0
            buf.append(md)
    flush()
    return results


def _assemble_elements(elements: List[Tuple]) -> str:
    """Render ordered elements to Markdown, keeping consecutive list items tight."""
    pieces: List[str] = []
    prev_list = False
    for _, _, el in elements:
        kind = el[0]
        if kind == "heading":
            level, text = el[1]
            md, is_list = "#" * level + " " + text, False
        elif kind == "list":
            ltype, text = el[1]
            md, is_list = ("1. " if ltype == "ol" else "- ") + text, True
        else:  # "para" or "table"
            md, is_list = el[1], False
        if pieces:
            pieces.append("\n" if (is_list and prev_list) else "\n\n")
        pieces.append(md)
        prev_list = is_list
    return "".join(pieces)


def _pdf_to_markdown(path: Path, include_tables: bool = True) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_PDF_HELP) from exc

    doc = fitz.open(str(path))
    try:
        heading_map = _build_heading_map(doc)
        page_chunks: List[str] = []

        for page in doc:
            elements: List[Tuple[float, float, Tuple]] = []
            table_boxes: List[Tuple[float, float, float, float]] = []

            # Tables first: locate them and emit positioned md-table elements.
            if include_tables:
                try:
                    # find_tables() prints an informational nag to stdout; keep
                    # it out of the user's stream.
                    with contextlib.redirect_stdout(io.StringIO()):
                        finder = page.find_tables()
                        found = list(getattr(finder, "tables", finder))
                except Exception:
                    found = []
                for tbl in found:
                    md = _pdf_table_to_markdown(tbl)
                    if md:
                        bbox = tuple(tbl.bbox)
                        table_boxes.append(bbox)
                        elements.append((bbox[1], bbox[0], ("table", md)))

            # Text blocks: skip any block that sits inside a detected table.
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                if block.get("type", 0) != 0:  # skip images / non-text
                    continue
                bbox = block.get("bbox", (0, 0, 0, 0))
                if any(_rect_mostly_inside(bbox, tb) for tb in table_boxes):
                    continue
                x0 = bbox[0]
                for y0, el in _process_text_block(block, heading_map):
                    elements.append((y0, x0, el))

            # Reading order: top-to-bottom, then left-to-right.
            elements.sort(key=lambda e: (round(e[0], 1), e[1]))
            chunk = _assemble_elements(elements)
            if chunk.strip():
                page_chunks.append(chunk)
    finally:
        doc.close()

    return _normalize_markdown("\n\n".join(page_chunks))


# ---------------------------------------------------------------------------
# Word (.docx) -> Markdown
# ---------------------------------------------------------------------------

def _iter_block_items(document):
    """Yield Paragraph and Table objects in true document order."""
    from docx.document import Document as _Document
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = document.element.body if isinstance(document, _Document) else document._element
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def _heading_prefix(style_name: Optional[str]) -> Optional[str]:
    if not style_name:
        return None
    name = style_name.lower()
    if name == "title":
        return "#"
    if name == "subtitle":
        return "##"
    if name.startswith("heading"):
        digits = "".join(ch for ch in name if ch.isdigit())
        level = int(digits) if digits else 1
        return "#" * max(1, min(level, 6))
    return None


def _list_prefix(style_name: Optional[str]) -> Optional[str]:
    if not style_name:
        return None
    name = style_name.lower()
    if "number" in name:
        return "1. "
    if name.startswith("list") or "bullet" in name:
        return "- "
    return None


def _runs_to_markdown(paragraph) -> str:
    """Inline formatting: wrap bold/italic runs in Markdown markers."""
    parts: List[str] = []
    for run in paragraph.runs:
        txt = run.text
        if not txt:
            continue
        if txt.strip():
            if run.bold and run.italic:
                txt = f"***{txt.strip()}***"
            elif run.bold:
                txt = f"**{txt.strip()}**"
            elif run.italic:
                txt = f"*{txt.strip()}*"
        parts.append(txt)
    text = "".join(parts) if parts else paragraph.text
    return text.strip()


def _paragraph_to_markdown(paragraph) -> Tuple[bool, str]:
    """Return (is_list_item, markdown_text)."""
    text = _runs_to_markdown(paragraph)
    if not text:
        return False, ""
    style = paragraph.style.name if paragraph.style is not None else ""
    heading = _heading_prefix(style)
    if heading:
        return False, f"{heading} {text}"
    lst = _list_prefix(style)
    if lst:
        return True, f"{lst}{text}"
    return False, text


def _docx_table_to_markdown(table) -> str:
    rows: List[List[str]] = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            # Flatten internal newlines and escape pipes so the table is valid.
            cell_text = " ".join(cell.text.split()).replace("|", "\\|")
            cells.append(cell_text)
        rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, *body = rows
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * width) + " |"]
    for r in body:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _docx_to_markdown(path: Path, include_tables: bool = True) -> str:
    try:
        import docx  # python-docx
    except ImportError as exc:
        raise ImportError(_DOCX_HELP) from exc
    from docx.text.paragraph import Paragraph
    from docx.table import Table

    document = docx.Document(str(path))
    blocks: List[Tuple[bool, str]] = []  # (is_list_item, text)

    for item in _iter_block_items(document):
        if isinstance(item, Paragraph):
            is_list, md = _paragraph_to_markdown(item)
            if md:
                blocks.append((is_list, md))
        elif isinstance(item, Table) and include_tables:
            md = _docx_table_to_markdown(item)
            if md:
                blocks.append((False, md))

    # Assemble: consecutive list items stay tight (single newline); everything
    # else is separated by a blank line.
    pieces: List[str] = []
    prev_list = False
    for is_list, md in blocks:
        if pieces:
            pieces.append("\n" if (is_list and prev_list) else "\n\n")
        pieces.append(md)
        prev_list = is_list
    return _normalize_markdown("".join(pieces))


# ---------------------------------------------------------------------------
# Plain text / already-Markdown -> Markdown
# ---------------------------------------------------------------------------

def _text_to_markdown(path: Path) -> str:
    raw = path.read_bytes()
    if b"\x00" in raw[:2048]:
        raise ValueError("File appears to be binary, not text.")
    return _normalize_markdown(raw.decode("utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def reduce_document(
    path: PathLike,
    *,
    file_type: Optional[str] = None,
    include_tables: bool = True,
) -> str:
    """
    Extract the *content* of a document into clean Markdown, discarding layout
    and metadata (margins, fonts, spacing, page geometry, office XML).

    This is a preprocessing step, not a compressor: it turns an expensive
    binary upload into compact, model-friendly text. Pipe the result through
    :func:`less_tokens.compress` if you also want lexical compression.

    Parameters
    ----------
    path
        Path to the document. PDF (``.pdf``), Word (``.docx`` / ``.docm``), and
        plain-text formats (``.txt``, ``.md``, ``.rst``, ...) are supported.
    file_type
        Force a parser regardless of the file's extension, e.g. ``"pdf"`` or
        ``".docx"``. Useful for files with missing or misleading extensions.
    include_tables
        If True (default) tables are converted to Markdown tables. Set False to
        skip table detection entirely (faster, and avoids noisy tables in some
        PDFs).

    Returns
    -------
    str
        The document content as Markdown — headings as ``#``/``##``, bullet and
        numbered lists, and tables — with all layout/metadata removed.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ImportError
        If a parser fails to import (it ships with the package, so this should
        only happen on a broken install).
    ValueError
        If the file type is unsupported or the file is binary but not a
        recognised document format.

    Examples
    --------
    >>> from less_tokens import reduce_document, compress
    >>> md = reduce_document("report.pdf")
    >>> print(md[:60])
    # Quarterly Report
    ## Summary
    Revenue grew ...
    >>> # optionally compress the extracted text further
    >>> smaller = compress(md, remove_filler_phrases=1, remove_stopwords=1)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")

    ext = (file_type or p.suffix).lower()
    if ext and not ext.startswith("."):
        ext = "." + ext

    if ext in _PDF_EXTS:
        return _pdf_to_markdown(p, include_tables=include_tables)
    if ext in _DOCX_EXTS:
        return _docx_to_markdown(p, include_tables=include_tables)
    if ext in _TEXT_EXTS:
        return _text_to_markdown(p)

    # Unknown extension: best-effort read as text, otherwise give up clearly.
    try:
        return _text_to_markdown(p)
    except (UnicodeDecodeError, ValueError):
        raise ValueError(
            f"Unsupported file type {ext!r}. Supported: .pdf, .docx, and "
            "plain-text formats (.txt, .md, .rst, ...). Pass file_type=... to "
            "force a specific parser."
        )


__all__ = ["reduce_document", "areduce_document"]


# ---------------------------------------------------------------------------
# Async variant
# ---------------------------------------------------------------------------
# Document parsing is I/O- plus CPU-bound and fully synchronous. To avoid
# blocking an asyncio event loop, we run it in the default thread-pool
# executor, mirroring acompress / acompress_structured.

async def areduce_document(
    path: PathLike,
    *,
    file_type: Optional[str] = None,
    include_tables: bool = True,
) -> str:
    """
    Async wrapper around :func:`reduce_document`.

    Runs the (synchronous) document parser in a thread executor so it doesn't
    block the event loop. Accepts exactly the same arguments as
    :func:`reduce_document`. Handy when reducing many uploaded files
    concurrently inside an async web server.

    Examples
    --------
    >>> import asyncio
    >>> from less_tokens import areduce_document
    >>> async def main():
    ...     return await asyncio.gather(
    ...         areduce_document("a.pdf"),
    ...         areduce_document("b.docx"),
    ...     )
    >>> asyncio.run(main())
    """
    loop = asyncio.get_event_loop()
    fn = functools.partial(
        reduce_document, path, file_type=file_type, include_tables=include_tables
    )
    return await loop.run_in_executor(None, fn)