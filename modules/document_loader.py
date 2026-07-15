"""Unified document loaders (PDF + DOCX + Markdown)."""
from __future__ import annotations

from pathlib import Path

from modules.doc_loader import load_docx_pages
from modules.md_loader import load_md_pages
from modules.pdf_loader import PageText, load_pdf_pages

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt"}


def load_document_pages(path: str | Path) -> list[PageText]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf_pages(path)
    if ext == ".docx":
        return load_docx_pages(path)
    if ext in {".md", ".markdown", ".txt"}:
        return load_md_pages(path)
    raise ValueError(
        f"Unsupported file type: {ext}. Use PDF, DOCX, MD, or TXT."
    )
