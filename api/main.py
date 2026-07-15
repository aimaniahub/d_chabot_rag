"""
PDF RAG Backend API — Railway production

  GET  /health
  GET  /ready
  GET  /stats
  POST /ingest
  POST /ingest/upload   (PDF or DOCX)
  POST /chat            (RAG + optional Postgres log)
"""
from __future__ import annotations

import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import (
    AUTO_INGEST_ON_START,
    CORS_ORIGINS,
    DATA_DIR,
    IS_SERVERLESS,
    MIN_SCORE,
    TOP_K,
    ensure_dirs,
    runtime_info,
    sync_seed_pdfs,
)

logger = logging.getLogger("pdf_rag.api")
logging.basicConfig(level=logging.INFO)


def _bootstrap_runtime() -> None:
    try:
        ensure_dirs()
        n = sync_seed_pdfs()
        if n:
            logger.info("Synced %s seed PDF(s) into %s", n, DATA_DIR)
        # also copy docx seeds if any
        from config import SEED_DATA_DIR

        if SEED_DATA_DIR.exists() and SEED_DATA_DIR.resolve() != DATA_DIR.resolve():
            for src in SEED_DATA_DIR.glob("*.docx"):
                dest = DATA_DIR / src.name
                try:
                    if not dest.exists() or dest.stat().st_size != src.stat().st_size:
                        shutil.copy2(src, dest)
                except OSError:
                    pass
        logger.info("Runtime paths: %s", runtime_info())
    except Exception:  # noqa: BLE001
        logger.exception("Runtime bootstrap failed")


def _run_auto_ingest_background() -> None:
    import threading

    def _job() -> None:
        try:
            from modules.ingest import ingest_paths

            logger.info("Background auto-ingest starting...")
            report = ingest_paths()
            logger.info(
                "Auto-ingest finished: indexed=%s skipped=%s failed=%s",
                report.indexed,
                report.skipped,
                report.failed,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Auto-ingest on startup failed")

    threading.Thread(target=_job, name="auto-ingest", daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _bootstrap_runtime()
    try:
        from modules.db import ensure_schema, is_db_configured

        if is_db_configured():
            ensure_schema()
    except Exception:  # noqa: BLE001
        logger.exception("DB schema bootstrap skipped/failed")
    if AUTO_INGEST_ON_START:
        _run_auto_ingest_background()
    yield


app = FastAPI(
    title="PDF RAG API",
    description="Darvi / PDF RAG: ingest PDF+DOCX, chat with sources, log to Postgres.",
    version="2.2.0",
    lifespan=lifespan,
)

_allow_credentials = CORS_ORIGINS != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/health")
def health() -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "ok",
        "service": "pdf-rag-api",
        "serverless": IS_SERVERLESS,
        "version": "2.2.0",
    }
    try:
        from modules.db import db_health

        out["postgres"] = db_health()
    except Exception as exc:  # noqa: BLE001
        out["postgres"] = {"ok": False, "detail": str(exc)}
    return out


@app.get("/ready")
def ready() -> dict[str, Any]:
    try:
        _bootstrap_runtime()
        from modules.ingest import get_index_stats

        st = get_index_stats()
        from modules.db import db_health

        return {
            "status": "ready",
            "chunk_count": st["chunk_count"],
            "documents": len(st.get("documents") or []),
            "paths": runtime_info(),
            "postgres": db_health(),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/stats")
def stats() -> dict[str, Any]:
    _bootstrap_runtime()
    from modules.ingest import get_index_stats

    out = get_index_stats()
    out["runtime"] = runtime_info()
    try:
        from modules.db import db_health

        out["postgres"] = db_health()
    except Exception:  # noqa: BLE001
        pass
    return out


@app.post("/ingest")
def ingest(body: IngestRequest) -> dict[str, Any]:
    _bootstrap_runtime()
    from modules.ingest import ingest_paths

    paths = [Path(body.file)] if body.file else None
    report = ingest_paths(paths=paths, rebuild=body.rebuild, force=body.force)
    return report.to_dict()


@app.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a PDF or DOCX and index it."""
    _bootstrap_runtime()
    from modules.ingest import ingest_paths

    name = (file.filename or "").strip()
    lower = name.lower()
    if not name or not (lower.endswith(".pdf") or lower.endswith(".docx")):
        raise HTTPException(
            status_code=400, detail="Only PDF and DOCX uploads are supported"
        )

    dest = DATA_DIR / Path(name).name
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Cannot write upload to {dest}: {exc}",
        ) from exc
    finally:
        await file.close()

    report = ingest_paths(paths=[dest], force=True)
    return {"saved_to": str(dest), **report.to_dict()}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> dict[str, Any]:
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
            logger.exception("Lazy ingest failed")
            raise HTTPException(
                status_code=503,
                detail=f"Index empty and ingest failed: {exc}",
            ) from exc

    if st["chunk_count"] == 0:
        raise HTTPException(
            status_code=400,
            detail="Index empty. POST /ingest or upload a PDF/DOCX first.",
        )

    session_id = (body.session_id or "").strip() or str(uuid.uuid4())
    source = (body.source or "api").strip() or "api"
    language = (body.language or "en").strip() or "en"

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
            source=source,
            language=language,
            persist=True,
        )
        return resp.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "PDF RAG API",
        "version": "2.2.0",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "chat": "POST /chat",
        "ingest": "POST /ingest",
        "upload": "POST /ingest/upload (pdf|docx)",
        "serverless": IS_SERVERLESS,
    }
