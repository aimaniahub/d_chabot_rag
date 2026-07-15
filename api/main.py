"""
Darvi RAG API — secured

Public:
  GET  /health
  GET  /admin          (UI; actions need ADMIN_KEY)

Protected (X-API-Key: API_KEY or ADMIN_KEY):
  POST /chat
  GET  /ready, /stats

Admin only (X-API-Key: ADMIN_KEY):
  POST /ingest, /ingest/upload
  GET/POST /admin/api/*
"""
from __future__ import annotations

import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import (
    AUTO_INGEST_ON_START,
    CORS_ORIGINS,
    DATA_DIR,
    DOCS_ENABLED,
    IS_SERVERLESS,
    MIN_SCORE,
    TOP_K,
    ensure_dirs,
    runtime_info,
    sync_seed_pdfs,
)
from modules.auth import require_admin_key, require_api_key

logger = logging.getLogger("pdf_rag.api")
logging.basicConfig(level=logging.INFO)

ADMIN_DIR = Path(__file__).resolve().parent / "admin_static"


def _bootstrap_runtime() -> None:
    try:
        ensure_dirs()
        # Restore originals from Railway Bucket if local volume missing them
        try:
            from modules.object_store import is_enabled, restore_missing_to_local

            if is_enabled():
                n_restored = restore_missing_to_local(DATA_DIR)
                if n_restored:
                    logger.info("Restored %s upload(s) from S3 bucket", n_restored)
        except Exception:  # noqa: BLE001
            logger.exception("S3 restore skipped/failed")

        n = sync_seed_pdfs()
        if n:
            logger.info("Synced %s seed file(s) into %s", n, DATA_DIR)
        from config import SEED_DATA_DIR, storage_health

        health = storage_health()
        if health.get("warning"):
            logger.warning("STORAGE: %s", health["warning"])
        logger.info(
            "Persist root=%s data=%s storage=%s s3=%s",
            health.get("persist_root"),
            health.get("data_dir"),
            health.get("storage_dir"),
            health.get("s3_enabled"),
        )
    except Exception:  # noqa: BLE001
        logger.exception("Runtime bootstrap failed")


def _run_auto_ingest_background() -> None:
    import threading

    def _job() -> None:
        try:
            from modules.ingest import ingest_paths

            report = ingest_paths()
            logger.info(
                "Auto-ingest: indexed=%s skipped=%s failed=%s",
                report.indexed,
                report.skipped,
                report.failed,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Auto-ingest failed")

    threading.Thread(target=_job, name="auto-ingest", daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _bootstrap_runtime()
    try:
        from modules.db import ensure_schema, is_db_configured

        if is_db_configured():
            ensure_schema()
    except Exception:  # noqa: BLE001
        logger.exception("DB bootstrap failed")
    if AUTO_INGEST_ON_START:
        _run_auto_ingest_background()
    yield


app = FastAPI(
    title="Darvi RAG API",
    description="Secured PDF/DOCX RAG. Use /admin for monitoring & uploads.",
    version="2.3.0",
    lifespan=lifespan,
    docs_url="/docs" if DOCS_ENABLED else None,
    redoc_url="/redoc" if DOCS_ENABLED else None,
    openapi_url="/openapi.json" if DOCS_ENABLED else None,
)

_allow_credentials = CORS_ORIGINS != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

if ADMIN_DIR.is_dir():
    app.mount("/admin/assets", StaticFiles(directory=str(ADMIN_DIR)), name="admin_assets")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = None
    min_score: Optional[float] = None
    doc_id: Optional[str] = None
    history: Optional[list[dict[str, str]]] = None
    session_id: Optional[str] = None
    source: Optional[str] = "api"
    language: Optional[str] = "en"


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    abstained: bool = False
    metrics: dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None


class IngestRequest(BaseModel):
    rebuild: bool = False
    force: bool = False
    file: Optional[str] = None


def _do_chat(body: ChatRequest) -> dict[str, Any]:
    _bootstrap_runtime()
    from modules.chat import ChatService
    from modules.ingest import get_index_stats, ingest_paths

    st = get_index_stats()
    if st["chunk_count"] == 0:
        try:
            sync_seed_pdfs()
            ingest_paths()
            st = get_index_stats()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"Index empty: {exc}") from exc

    if st["chunk_count"] == 0:
        raise HTTPException(
            status_code=400,
            detail="Index empty. Upload a PDF/DOCX from /admin first.",
        )

    session_id = (body.session_id or "").strip() or str(uuid.uuid4())
    try:
        service = ChatService(
            top_k=body.top_k or TOP_K,
            min_score=body.min_score if body.min_score is not None else MIN_SCORE,
        )
        resp = service.ask(
            body.question,
            history=body.history,
            top_k=body.top_k,
            min_score=body.min_score,
            doc_id=body.doc_id,
            session_id=session_id,
            source=(body.source or "api").strip() or "api",
            language=(body.language or "en").strip() or "en",
            persist=True,
        )
        return resp.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


_ALLOWED_UPLOAD = {".pdf", ".docx", ".md", ".markdown", ".txt"}


def _validate_upload_name(name: str) -> str:
    name = (name or "").strip()
    lower = name.lower()
    ext = Path(name).suffix.lower()
    if not name or ext not in _ALLOWED_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail="Allowed types: PDF, DOCX, MD, Markdown, TXT",
        )
    # prevent path traversal
    safe = Path(name).name
    if not safe or safe in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return safe


def _do_upload(file: UploadFile) -> dict[str, Any]:
    _bootstrap_runtime()
    from modules.ingest import ingest_paths

    safe_name = _validate_upload_name(file.filename or "")
    dest = DATA_DIR / safe_name
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    report = ingest_paths(paths=[dest], force=True)
    return {"saved_to": str(dest), **report.to_dict()}


def _do_upload_many(files: list[UploadFile]) -> dict[str, Any]:
    """Sequential bulk upload + embed (one report combined)."""
    _bootstrap_runtime()
    from modules.ingest import ingest_paths

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    saved: list[Path] = []
    errors: list[dict[str, str]] = []
    for f in files:
        try:
            safe_name = _validate_upload_name(f.filename or "")
            dest = DATA_DIR / safe_name
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            saved.append(dest)
        except HTTPException as exc:
            errors.append(
                {
                    "file": f.filename or "unknown",
                    "error": str(exc.detail),
                }
            )
        except OSError as exc:
            errors.append({"file": f.filename or "unknown", "error": str(exc)})
        finally:
            try:
                f.file.close()
            except Exception:  # noqa: BLE001
                pass

    if not saved:
        raise HTTPException(
            status_code=400,
            detail={"message": "No valid files uploaded", "errors": errors},
        )

    report = ingest_paths(paths=saved, force=True)
    out = report.to_dict()
    out["uploaded_count"] = len(saved)
    out["saved_paths"] = [str(p) for p in saved]
    out["upload_errors"] = errors
    return out


# ----- Public -----


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "darvi-rag-api", "version": "2.3.0"}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Darvi RAG API",
        "health": "/health",
        "admin": "/admin",
        "note": "API requires X-API-Key. Manage uploads at /admin",
    }


@app.get("/admin")
@app.get("/admin/")
def admin_page() -> FileResponse:
    index = ADMIN_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Admin UI missing")
    return FileResponse(index)


# ----- Chat (API key) -----


@app.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    _key: str = Depends(require_api_key),
) -> dict[str, Any]:
    return _do_chat(body)


@app.get("/ready")
def ready(_key: str = Depends(require_api_key)) -> dict[str, Any]:
    _bootstrap_runtime()
    from modules.db import db_health
    from modules.ingest import get_index_stats

    st = get_index_stats()
    return {
        "status": "ready",
        "chunk_count": st["chunk_count"],
        "documents": len(st.get("documents") or []),
        "postgres": db_health(),
        "paths": runtime_info(),
    }


@app.get("/stats")
def stats(_key: str = Depends(require_api_key)) -> dict[str, Any]:
    _bootstrap_runtime()
    from modules.db import db_health
    from modules.ingest import get_index_stats

    out = get_index_stats()
    out["postgres"] = db_health()
    out["runtime"] = runtime_info()
    return out


# ----- Ingest (admin key) -----


@app.post("/ingest")
def ingest(
    body: IngestRequest,
    _key: str = Depends(require_admin_key),
) -> dict[str, Any]:
    _bootstrap_runtime()
    from modules.ingest import ingest_paths

    paths = [Path(body.file)] if body.file else None
    return ingest_paths(paths=paths, rebuild=body.rebuild, force=body.force).to_dict()


@app.post("/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    _key: str = Depends(require_admin_key),
) -> dict[str, Any]:
    try:
        result = _do_upload(file)
    finally:
        await file.close()
    return result


@app.post("/ingest/upload/bulk")
async def ingest_upload_bulk(
    files: List[UploadFile] = File(...),
    _key: str = Depends(require_admin_key),
) -> dict[str, Any]:
    try:
        return _do_upload_many(list(files))
    finally:
        for f in files:
            try:
                await f.close()
            except Exception:  # noqa: BLE001
                pass


# ----- Admin JSON API (same admin key; used by /admin UI) -----


@app.get("/admin/api/stats")
def admin_stats(_key: str = Depends(require_admin_key)) -> dict[str, Any]:
    _bootstrap_runtime()
    from config import storage_health
    from modules.db import db_health
    from modules.ingest import get_index_stats
    from modules.object_store import status as s3_status

    out = get_index_stats()
    out["postgres"] = db_health()
    out["runtime"] = runtime_info()
    out["storage"] = storage_health()
    out["bucket"] = s3_status()
    return out


@app.get("/admin/api/chats")
def admin_chats(
    limit: int = 40,
    _key: str = Depends(require_admin_key),
) -> dict[str, Any]:
    from modules.db import list_recent_messages

    return {"messages": list_recent_messages(limit=limit)}


@app.post("/admin/api/chat")
def admin_chat(
    body: ChatRequest,
    _key: str = Depends(require_admin_key),
) -> dict[str, Any]:
    body.source = body.source or "admin-ui"
    return _do_chat(body)


@app.post("/admin/api/ingest")
def admin_ingest(
    body: IngestRequest,
    _key: str = Depends(require_admin_key),
) -> dict[str, Any]:
    _bootstrap_runtime()
    from modules.ingest import ingest_paths

    paths = [Path(body.file)] if body.file else None
    return ingest_paths(paths=paths, rebuild=body.rebuild, force=body.force).to_dict()


@app.post("/admin/api/upload")
async def admin_upload(
    file: UploadFile = File(...),
    _key: str = Depends(require_admin_key),
) -> dict[str, Any]:
    try:
        return _do_upload(file)
    finally:
        await file.close()


@app.post("/admin/api/upload/bulk")
async def admin_upload_bulk(
    files: List[UploadFile] = File(...),
    _key: str = Depends(require_admin_key),
) -> dict[str, Any]:
    """Bulk upload many PDF/DOCX/MD files and embed each."""
    try:
        return _do_upload_many(list(files))
    finally:
        for f in files:
            try:
                await f.close()
            except Exception:  # noqa: BLE001
                pass


@app.delete("/admin/api/documents/{doc_id}")
def admin_delete_document(
    doc_id: str,
    delete_file: bool = True,
    _key: str = Depends(require_admin_key),
) -> dict[str, Any]:
    """Remove embeddings (+ optional disk file) for one document."""
    from modules.ingest import delete_document

    result = delete_document(doc_id, delete_file=delete_file)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error") or "delete failed")
    return result


@app.get("/admin/api/analytics")
def admin_analytics(
    days: int = 14,
    _key: str = Depends(require_admin_key),
) -> dict[str, Any]:
    from modules.db import chat_analytics
    from modules.ingest import get_index_stats
    from config import OPENROUTER_MODELS, EMBEDDING_MODEL

    _bootstrap_runtime()
    st = get_index_stats()
    analytics = chat_analytics(days=days)
    return {
        "chat": analytics,
        "index": {
            "chunk_count": st.get("chunk_count"),
            "document_count": len(st.get("documents") or []),
            "documents": st.get("documents") or [],
            "embedding_model": st.get("embedding_model") or EMBEDDING_MODEL,
        },
        "process": {
            "llm_provider": "openrouter",
            "llm_models": OPENROUTER_MODELS,
            "embedding_model": EMBEDDING_MODEL,
            "pipeline": [
                "1. User question (site or admin)",
                "2. Embed question (local fastembed)",
                "3. Retrieve top chunks from Chroma",
                "4. Build plain-text prompt with context",
                "5. OpenRouter free model (rotate on fail)",
                "6. Save Q&A to Postgres",
                "7. Return answer to client",
            ],
            "runtime": runtime_info(),
            "version": "2.3.1",
        },
    }


@app.get("/admin/api/process")
def admin_process(_key: str = Depends(require_admin_key)) -> dict[str, Any]:
    """Bot / pipeline details for monitoring tab."""
    from config import (
        EMBEDDING_MODEL,
        MIN_SCORE,
        OPENROUTER_MODELS,
        TOP_K,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
    from modules.db import db_health
    from modules.ingest import get_index_stats

    st = get_index_stats()
    return {
        "status": "running",
        "version": "2.3.1",
        "pipeline_steps": [
            {"step": 1, "name": "Receive question", "detail": "POST /chat with API_KEY"},
            {"step": 2, "name": "Embed query", "detail": f"Model: {EMBEDDING_MODEL}"},
            {
                "step": 3,
                "name": "Retrieve",
                "detail": f"Chroma top_k={TOP_K}, min_score={MIN_SCORE}",
            },
            {
                "step": 4,
                "name": "Generate",
                "detail": "OpenRouter models: " + ", ".join(OPENROUTER_MODELS),
            },
            {"step": 5, "name": "Log", "detail": "Postgres conversations + messages"},
            {"step": 6, "name": "Respond", "detail": "Plain text header + bullets"},
        ],
        "chunking": {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP},
        "index": {
            "chunks": st.get("chunk_count"),
            "documents": len(st.get("documents") or []),
        },
        "postgres": db_health(),
        "runtime": runtime_info(),
    }
