"""Prepare PDF pages as text (the model reads text only)."""

import base64
import hashlib
import math
import threading
import time
from collections import OrderedDict
from contextlib import closing
from dataclasses import asdict, dataclass

if __package__:
    from .errors import APIError
else:
    from errors import APIError

MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PAGES = 20
MAX_TEXT_CHARACTERS = 1_000_000
MAX_RENDERED_BYTES = 32 * 1024 * 1024
MAX_REQUEST_DOCUMENT_BYTES = 64 * 1024 * 1024
CACHE_BYTES = 64 * 1024 * 1024
CACHE_ENTRIES = 16


@dataclass(slots=True)
class DocumentBudget:
    deadline: float | None = None
    remaining_bytes: int = MAX_REQUEST_DOCUMENT_BYTES

    def remaining_time(self):
        if self.deadline is None:
            return -1
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise APIError(504, "request timed out", "request_timeout")
        return remaining

    def charge(self, size):
        self.remaining_time()
        if size > self.remaining_bytes:
            raise APIError(400, "documents exceed the request size limit")
        self.remaining_bytes -= size


@dataclass(frozen=True, slots=True)
class Page:
    text: str

    @property
    def size(self):
        return len(self.text) * 4


@dataclass(frozen=True, slots=True)
class RenderLimits:
    pages: int
    text_characters: int
    rendered_bytes: int


def _render_limits():
    return RenderLimits(MAX_PAGES, MAX_TEXT_CHARACTERS, MAX_RENDERED_BYTES)


# Serialize cache misses so only one bounded PDF worker is active at a time.
_pdf_lock = threading.Lock()
_cache: OrderedDict[bytes, tuple[Page, ...]] = OrderedDict()
_cache_bytes = 0


def _render(payload, budget):
    if __package__:
        from .document_worker import render
    else:
        from document_worker import render

    values = render(payload, asdict(_render_limits()), budget.remaining_time())
    pages = tuple(Page(**value) for value in values)
    budget.charge(sum(page.size for page in pages))
    return pages


def render_pages(payload, budget, limits=None):
    import pypdfium2 as pdfium

    limits = limits or _render_limits()
    try:
        budget.remaining_time()
        with pdfium.PdfDocument(payload) as document:
            document.init_forms()
            if not 1 <= len(document) <= limits.pages:
                raise APIError(
                    400, f"PDF documents must contain 1–{limits.pages} pages"
                )
            pages = []
            total_characters = total_bytes = 0
            for index in range(len(document)):
                budget.remaining_time()
                with closing(document[index]) as page:
                    width, height = page.get_size()
                    if (
                        not all(
                            math.isfinite(side) and side > 0 for side in (width, height)
                        )
                        or max(width, height) / min(width, height) > 200
                    ):
                        raise APIError(400, "PDF page dimensions are invalid")
                    with closing(page.get_textpage()) as text_page:
                        total_characters += text_page.count_chars()
                        if total_characters > limits.text_characters:
                            raise APIError(400, "PDF text exceeds the size limit")
                        text = text_page.get_text_bounded().strip()
                    prepared = Page(f"PDF page {index + 1}:\n{text}\n")
                    total_bytes += prepared.size
                    if total_bytes > limits.rendered_bytes:
                        raise APIError(400, "rendered PDF exceeds the size limit")
                    budget.charge(prepared.size)
                    pages.append(prepared)
            return tuple(pages)
    except APIError:
        raise
    except pdfium.PdfiumError as error:
        if error.err_code == pdfium.raw.FPDF_ERR_PASSWORD:
            raise APIError(
                400, "PDF documents requiring an opening password are not supported"
            ) from error
        if error.err_code == pdfium.raw.FPDF_ERR_SECURITY:
            raise APIError(400, "PDF security handler is not supported") from error
        raise APIError(400, "PDF document could not be decoded") from error
    except (ValueError, OverflowError) as error:
        raise APIError(400, "PDF document could not be read") from error


def _pages(encoded, budget):
    global _cache_bytes
    if not _pdf_lock.acquire(
        timeout=min(budget.remaining_time(), threading.TIMEOUT_MAX)
    ):
        raise APIError(504, "request timed out", "request_timeout")
    try:
        budget.remaining_time()
        try:
            payload = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise APIError(400, "PDF data is not valid base64") from error
        if len(payload) > MAX_PDF_BYTES:
            raise APIError(400, "PDF document exceeds the size limit")
        key = hashlib.sha256(payload).digest()
        cached = _cache.get(key)
        if cached is not None:
            budget.charge(sum(page.size for page in cached))
            _cache.move_to_end(key)
            return cached
        pages = _render(payload, budget)
        size = sum(page.size for page in pages)
        if size <= CACHE_BYTES:
            _cache[key] = pages
            _cache_bytes += size
            while _cache_bytes > CACHE_BYTES or len(_cache) > CACHE_ENTRIES:
                _, evicted = _cache.popitem(last=False)
                _cache_bytes -= sum(page.size for page in evicted)
        return pages
    finally:
        _pdf_lock.release()


def document_content(block, *, budget=None):
    """Translate an inline PDF into canonical text parts, one per page."""
    if budget is None:
        budget = DocumentBudget()
    source = block.get("source")
    if (
        not isinstance(source, dict)
        or source.get("type") != "base64"
        or source.get("media_type") != "application/pdf"
    ):
        raise APIError(400, "documents require a base64 application/pdf source")
    encoded = source.get("data")
    citations = block.get("citations")
    if citations is not None and (
        not isinstance(citations, dict) or citations.get("enabled", False) is not False
    ):
        raise APIError(400, "document citations are not supported")
    parts = []
    for field in ("title", "context"):
        value = block.get(field)
        if value is not None:
            if not isinstance(value, str):
                raise APIError(400, f"document {field} must be a string")
            budget.charge((len(value) + 1) * 4)
            parts.append({"type": "text", "text": value + "\n"})
    parts.extend(pdf_content(encoded, budget=budget))
    return parts


def pdf_content(encoded, *, budget=None):
    """Render an inline PDF through the shared bounded document pipeline."""
    if not isinstance(encoded, str):
        raise APIError(400, "PDF data must be a base64 string")
    if len(encoded) > 4 * ((MAX_PDF_BYTES + 2) // 3):
        raise APIError(400, "PDF document exceeds the size limit")
    if budget is None:
        budget = DocumentBudget()
    parts = []
    for page in _pages(encoded, budget):
        parts.append({"type": "text", "text": page.text})
    return parts


def file_content(file, *, budget):
    """Translate OpenAI inline PDFs into canonical text parts."""
    if not isinstance(file, dict):
        raise APIError(400, "file must be an object")
    if file.get("file_id") is not None or file.get("file_url") is not None:
        raise APIError(
            400, "file_id and file_url are not supported; use inline file_data"
        )
    filename = file.get("filename")
    if filename is not None and not isinstance(filename, str):
        raise APIError(400, "filename must be a string")
    data = file.get("file_data")
    if not isinstance(data, str):
        raise APIError(400, "file_data must contain a base64 PDF")
    if data.startswith("data:"):
        prefix = "data:application/pdf;base64,"
        if not data.startswith(prefix):
            raise APIError(400, "only application/pdf file data is supported")
        data = data[len(prefix) :]
    return pdf_content(data, budget=budget)
