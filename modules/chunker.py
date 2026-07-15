"""Overlap-aware chunking with page metadata."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any

from config import CHUNK_OVERLAP, CHUNK_SIZE
from modules.pdf_loader import PageText


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    page_start: int
    page_end: int
    chunk_index: int

    def to_metadata(self) -> dict[str, Any]:
        meta = asdict(self)
        # Chroma metadata values must be str/int/float/bool
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "page_start": int(self.page_start),
            "page_end": int(self.page_end),
            "chunk_index": int(self.chunk_index),
        }


def _make_chunk_id(doc_id: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{doc_id}:{index}:{text[:64]}".encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}::{index}::{digest}"


def _split_windows(text: str, size: int, overlap: int) -> list[str]:
    """Sliding windows with light boundary snapping."""
    if not text:
        return []
    if size <= 0:
        raise ValueError("CHUNK_SIZE must be > 0")
    overlap = max(0, min(overlap, size - 1)) if size > 1 else 0

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            # Prefer paragraph, then sentence, then space near the end
            snap = max(
                window.rfind("\n\n"),
                window.rfind(". "),
                window.rfind(" "),
            )
            if snap > size * 0.5:
                end = start + snap + (2 if window[snap : snap + 2] == ". " else 1)
                end = min(end, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(0, end - overlap)
        if start >= end:
            start = end
    return chunks


def create_chunks_from_pages(
    pages: list[PageText],
    doc_id: str,
    source: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk page-aware document into overlapping text units with metadata."""
    if not pages:
        return []

    # Build continuous text with page markers for mapping
    parts: list[str] = []
    page_spans: list[tuple[int, int, int]] = []  # start, end, page
    cursor = 0
    for p in pages:
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(p.text)
        cursor += len(p.text)
        page_spans.append((start, cursor, p.page))

    full = "".join(parts)
    windows = _split_windows(full, chunk_size, chunk_overlap)

    chunks: list[Chunk] = []
    search_from = 0
    for i, text in enumerate(windows):
        # Locate window in full text (approximate sequential)
        idx = full.find(text[: min(40, len(text))], search_from)
        if idx < 0:
            idx = full.find(text[: min(40, len(text))])
        if idx < 0:
            idx = search_from
        end_idx = idx + len(text)
        search_from = max(0, end_idx - chunk_overlap)

        page_start, page_end = _pages_for_span(page_spans, idx, end_idx)
        chunk_id = _make_chunk_id(doc_id, i, text)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source=source,
                text=text,
                page_start=page_start,
                page_end=page_end,
                chunk_index=i,
            )
        )
    return chunks


def _pages_for_span(
    page_spans: list[tuple[int, int, int]], start: int, end: int
) -> tuple[int, int]:
    pages_hit = [pg for s, e, pg in page_spans if not (end <= s or start >= e)]
    if not pages_hit:
        return 1, 1
    return min(pages_hit), max(pages_hit)


def create_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Backward-compatible plain string chunks (no metadata)."""
    return _split_windows(text, chunk_size, CHUNK_OVERLAP)
