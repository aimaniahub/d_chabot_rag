"""Document registry for incremental ingest."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import EMBEDDING_MODEL, MANIFEST_PATH, ensure_dirs


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    ensure_dirs()
    p = path or MANIFEST_PATH
    if not p.exists():
        return {
            "version": 1,
            "embedding_model": EMBEDDING_MODEL,
            "documents": {},
        }
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("version", 1)
    data.setdefault("embedding_model", EMBEDDING_MODEL)
    data.setdefault("documents", {})
    return data


def save_manifest(data: dict[str, Any], path: Path | None = None) -> None:
    ensure_dirs()
    p = path or MANIFEST_PATH
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def upsert_document(
    manifest: dict[str, Any],
    doc_id: str,
    *,
    path: str,
    sha256: str,
    mtime: float,
    pages: int,
    chunk_count: int,
) -> None:
    manifest["embedding_model"] = EMBEDDING_MODEL
    manifest["documents"][doc_id] = {
        "doc_id": doc_id,
        "path": path,
        "sha256": sha256,
        "mtime": mtime,
        "pages": pages,
        "chunk_count": chunk_count,
        "indexed_at": _now_iso(),
    }


def remove_document(manifest: dict[str, Any], doc_id: str) -> None:
    manifest.get("documents", {}).pop(doc_id, None)


def list_documents(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return list(manifest.get("documents", {}).values())
