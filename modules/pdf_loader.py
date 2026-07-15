"""Page-aware PDF text extraction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from utils.text_clean import clean_text


@dataclass
class PageText:
    page: int  # 1-indexed
    text: str


def load_pdf_pages(pdf_path: str | Path) -> list[PageText]:
    """Extract cleaned text per page. Empty pages are skipped."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))
    pages: list[PageText] = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        text = clean_text(raw)
        if text:
            pages.append(PageText(page=i + 1, text=text))
    return pages


def load_pdf(pdf_path: str | Path) -> str:
    """Backward-compatible full-document text (joined pages)."""
    pages = load_pdf_pages(pdf_path)
    return "\n\n".join(p.text for p in pages)
