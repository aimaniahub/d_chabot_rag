"""Unified document loaders (PDF + DOCX)."""
from __future__ import annotations

from pathlib import Path

from modules.doc_loader import load_docx_pages
from modules.pdf_loader import PageText, load_pdf_pages

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def load_document_pages(path: str | Path) -> list[PageText]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf_pages(path)
    if ext == ".docx":
        return load_docx_pages(path)
    raise ValueError(f"Unsupported file type: {ext}. Use PDF or DOCX.")
