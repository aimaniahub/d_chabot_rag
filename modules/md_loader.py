"""Load Markdown / plain text as page-like units for chunking."""
from __future__ import annotations

from pathlib import Path

from modules.pdf_loader import PageText
from utils.text_clean import clean_text


def load_md_pages(path: str | Path) -> list[PageText]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw = path.read_text(encoding="utf-8", errors="replace")
    text = clean_text(raw)
    if not text:
        return []

    # Split on markdown headings when possible for better retrieval units
    parts: list[str] = []
    buf: list[str] = []
    for line in text.split("\n"):
        if line.startswith("#") and buf:
            parts.append("\n".join(buf).strip())
            buf = [line]
        else:
            buf.append(line)
    if buf:
        parts.append("\n".join(buf).strip())

    pages: list[PageText] = []
    page_no = 1
    acc: list[str] = []
    size = 0
    for part in parts:
        if not part:
            continue
        if size + len(part) > 1800 and acc:
            pages.append(PageText(page=page_no, text="\n\n".join(acc)))
            page_no += 1
            acc = [part]
            size = len(part)
        else:
            acc.append(part)
            size += len(part)
    if acc:
        pages.append(PageText(page=page_no, text="\n\n".join(acc)))
    return pages
