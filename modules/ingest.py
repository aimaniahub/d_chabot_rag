"""Incremental PDF/DOCX ingest into Chroma."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import (
    CHROMA_DIR,
    DATA_DIR,
    EMBEDDING_MODEL,
    PRUNE_MISSING_ON_INGEST,
    ensure_dirs,
)
from modules.chunker import create_chunks_from_pages
from modules.document_loader import SUPPORTED_EXTENSIONS, load_document_pages
from modules.embedder import Embedder
from modules.manifest import (
    list_documents,
    load_manifest,
    remove_document,
    save_manifest,
    upsert_document,
)
from modules.metrics import timer
from modules.vector_store import VectorStore
from utils.hashing import file_sha256


@dataclass
class FileIngestResult:
    path: str
    doc_id: str
    status: str  # indexed | skipped | failed | deleted
    pages: int = 0
    chunks: int = 0
    elapsed_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class IngestReport:
    results: list[FileIngestResult] = field(default_factory=list)
    total_ms: float = 0.0
    embedding_model: str = EMBEDDING_MODEL

    @property
    def indexed(self) -> int:
        return sum(1 for r in self.results if r.status == "indexed")

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == "skipped")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "failed")

    def to_dict(self) -> dict:
        return {
            "embedding_model": self.embedding_model,
            "total_ms": self.total_ms,
            "indexed": self.indexed,
            "skipped": self.skipped,
            "failed": self.failed,
            "results": [r.__dict__ for r in self.results],
        }


def _doc_id_for(path: Path) -> str:
    return path.name


def _list_data_files(data_dir: Path) -> list[Path]:
    files: list[Path] = []
    for ext in sorted(SUPPORTED_EXTENSIONS):
        files.extend(data_dir.glob(f"*{ext}"))
    return sorted(files, key=lambda p: p.name.lower())


def ingest_file(
    path: Path,
    *,
    embedder: Embedder,
    store: VectorStore,
    manifest: dict,
    force: bool = False,
) -> FileIngestResult:
    path = Path(path)
    doc_id = _doc_id_for(path)

    with timer() as t:
        result = _ingest_file_inner(
            path,
            doc_id=doc_id,
            embedder=embedder,
            store=store,
            manifest=manifest,
            force=force,
        )
    result.elapsed_ms = t["ms"]
    return result


def _ingest_file_inner(
    path: Path,
    *,
    doc_id: str,
    embedder: Embedder,
    store: VectorStore,
    manifest: dict,
    force: bool,
) -> FileIngestResult:
    if not path.exists():
        return FileIngestResult(
            path=str(path), doc_id=doc_id, status="failed", error="file not found"
        )
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return FileIngestResult(
            path=str(path),
            doc_id=doc_id,
            status="failed",
            error=f"unsupported type {path.suffix}; use PDF, DOCX, MD, or TXT",
        )

    try:
        sha = file_sha256(path)
        mtime = path.stat().st_mtime
        existing = manifest.get("documents", {}).get(doc_id)
        model_ok = manifest.get("embedding_model") == EMBEDDING_MODEL
        if not force and model_ok and existing and existing.get("sha256") == sha:
            return FileIngestResult(
                path=str(path),
                doc_id=doc_id,
                status="skipped",
                pages=int(existing.get("pages", 0)),
                chunks=int(existing.get("chunk_count", 0)),
            )

        pages = load_document_pages(path)
        if not pages:
            return FileIngestResult(
                path=str(path),
                doc_id=doc_id,
                status="failed",
                error="no extractable text",
            )

        chunks = create_chunks_from_pages(pages, doc_id=doc_id, source=path.name)
        if not chunks:
            return FileIngestResult(
                path=str(path),
                doc_id=doc_id,
                status="failed",
                error="chunker produced no chunks",
            )

        embeddings = embedder.embed_texts([c.text for c in chunks])
        store.delete_doc(doc_id)
        store.upsert_chunks(chunks, embeddings)
        upsert_document(
            manifest,
            doc_id,
            path=str(path),
            sha256=sha,
            mtime=mtime,
            pages=len(pages),
            chunk_count=len(chunks),
        )
        # Best-effort backup of original file to Railway Bucket / S3
        try:
            from modules.object_store import is_enabled, upload_file

            if is_enabled():
                upload_file(path)
        except Exception:  # noqa: BLE001
            pass
        return FileIngestResult(
            path=str(path),
            doc_id=doc_id,
            status="indexed",
            pages=len(pages),
            chunks=len(chunks),
        )
    except Exception as exc:  # noqa: BLE001
        return FileIngestResult(
            path=str(path), doc_id=doc_id, status="failed", error=str(exc)
        )


def prune_missing(
    store: VectorStore,
    manifest: dict,
    data_dir: Path | None = None,
) -> list[FileIngestResult]:
    """Remove index entries for files no longer present in data dir."""
    data_dir = data_dir or DATA_DIR
    present = {p.name for p in _list_data_files(data_dir)}
    results: list[FileIngestResult] = []
    for doc_id in list(manifest.get("documents", {}).keys()):
        if doc_id not in present:
            store.delete_doc(doc_id)
            remove_document(manifest, doc_id)
            results.append(
                FileIngestResult(path=doc_id, doc_id=doc_id, status="deleted")
            )
    return results


def ingest_paths(
    paths: list[Path] | None = None,
    *,
    data_dir: Path | None = None,
    rebuild: bool = False,
    force: bool = False,
    prune_missing_files: bool | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> IngestReport:
    """Ingest one or more PDF/DOCX/MD files (default: all under uploads/)."""
    ensure_dirs()
    data_dir = Path(data_dir or DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    # If local disk lost files but S3 bucket has them, restore first
    try:
        from modules.object_store import is_enabled, restore_missing_to_local

        if is_enabled() and paths is None:
            restore_missing_to_local(data_dir)
    except Exception:  # noqa: BLE001
        pass

    embedder = embedder or Embedder()
    store = store or VectorStore()
    manifest = load_manifest()

    do_prune = (
        PRUNE_MISSING_ON_INGEST
        if prune_missing_files is None
        else prune_missing_files
    )

    with timer() as total_t:
        if rebuild:
            store.reset()
            manifest = {
                "version": 1,
                "embedding_model": EMBEDDING_MODEL,
                "documents": {},
            }
            force = True

        if paths:
            files = [Path(p) for p in paths]
        else:
            files = _list_data_files(data_dir)

        report = IngestReport(embedding_model=embedder.model_name)
        for f in files:
            report.results.append(
                ingest_file(
                    f,
                    embedder=embedder,
                    store=store,
                    manifest=manifest,
                    force=force or rebuild,
                )
            )

        # Default OFF: pruning when disk is empty after redeploy wiped the whole index
        if paths is None and do_prune:
            report.results.extend(prune_missing(store, manifest, data_dir))

        save_manifest(manifest)

    report.total_ms = total_t["ms"]
    return report


def get_index_stats() -> dict:
    ensure_dirs()
    store = VectorStore()
    manifest = load_manifest()
    return {
        "chunk_count": store.count(),
        "documents": list_documents(manifest),
        "embedding_model": manifest.get("embedding_model", EMBEDDING_MODEL),
        "data_dir": str(DATA_DIR),
        "chroma_dir": str(CHROMA_DIR),
    }


def delete_document(
    doc_id: str,
    *,
    delete_file: bool = True,
    store: VectorStore | None = None,
) -> dict:
    """
    Remove a document's vectors from Chroma, drop manifest entry,
    and optionally delete the file under DATA_DIR.
    """
    ensure_dirs()
    doc_id = (doc_id or "").strip()
    if not doc_id or "/" in doc_id or "\\" in doc_id or ".." in doc_id:
        return {"ok": False, "error": "invalid doc_id"}

    store = store or VectorStore()
    manifest = load_manifest()
    meta = (manifest.get("documents") or {}).get(doc_id)

    store.delete_doc(doc_id)
    remove_document(manifest, doc_id)
    save_manifest(manifest)

    # Remove from S3 bucket if enabled
    try:
        from modules.object_store import delete_object, is_enabled, object_key

        if is_enabled():
            delete_object(object_key(doc_id))
    except Exception:  # noqa: BLE001
        pass

    file_deleted = False
    path_tried = None
    if delete_file:
        # Prefer path from manifest; fall back to data_dir / doc_id
        candidates = []
        if meta and meta.get("path"):
            candidates.append(Path(meta["path"]))
        candidates.append(DATA_DIR / doc_id)
        for p in candidates:
            path_tried = str(p)
            try:
                if p.exists() and p.is_file() and p.resolve().parent == DATA_DIR.resolve():
                    p.unlink()
                    file_deleted = True
                    break
                if p.exists() and p.is_file():
                    # still allow delete if under DATA_DIR
                    try:
                        p.resolve().relative_to(DATA_DIR.resolve())
                        p.unlink()
                        file_deleted = True
                        break
                    except ValueError:
                        continue
            except OSError as exc:
                return {
                    "ok": True,
                    "doc_id": doc_id,
                    "vectors_removed": True,
                    "file_deleted": False,
                    "path": path_tried,
                    "warning": str(exc),
                }

    return {
        "ok": True,
        "doc_id": doc_id,
        "vectors_removed": True,
        "file_deleted": file_deleted,
        "path": path_tried,
        "had_manifest": bool(meta),
        "chunk_count_after": store.count(),
    }
