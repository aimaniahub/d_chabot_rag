"""Incremental PDF/DOCX ingest into Chroma."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import CHROMA_DIR, DATA_DIR, EMBEDDING_MODEL, ensure_dirs
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
            error=f"unsupported type {path.suffix}; use PDF or DOCX",
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
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> IngestReport:
    """Ingest one or more PDF/DOCX files (default: all under data/)."""
    ensure_dirs()
    data_dir = Path(data_dir or DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    embedder = embedder or Embedder()
    store = store or VectorStore()
    manifest = load_manifest()

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

        if paths is None:
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
