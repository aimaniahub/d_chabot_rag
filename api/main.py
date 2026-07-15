"""
PDF RAG Backend API

Designed to be consumed by any frontend (Next.js/Vercel, React, mobile, etc.).

  GET  /health
  GET  /ready
  GET  /stats
  POST /ingest
  POST /ingest/upload
  POST /chat
"""
from __future__ import annotations

import logging
import shutil
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
    """Writable dirs + seed PDFs. Must never raise on serverless cold start."""
    try:
        ensure_dirs()
        n = sync_seed_pdfs()
        if n:
            logger.info("Synced %s seed PDF(s) into %s", n, DATA_DIR)
        logger.info("Runtime paths: %s", runtime_info())
    except Exception:  # noqa: BLE001
        logger.exception("Runtime bootstrap failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _bootstrap_runtime()
    if AUTO_INGEST_ON_START:
        try:
            from modules.ingest import ingest_paths

            report = ingest_paths()
            logger.info(
                "Auto-ingest finished: indexed=%s skipped=%s failed=%s",
                report.indexed,
                report.skipped,
                report.failed,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Auto-ingest on startup failed")
    yield


app = FastAPI(
    title="PDF RAG API",
    description=(
        "Backend RAG service: ingest PDFs, retrieve grounded context, "
        "answer via Gemini. Connect any frontend via HTTP/JSON."
    ),
    version="2.1.1",
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


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    abstained: bool = False
    metrics: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    rebuild: bool = False
    force: bool = False
    file: Optional[str] = None


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness — no disk writes required."""
    return {
        "status": "ok",
        "service": "pdf-rag-api",
        "serverless": IS_SERVERLESS,
    }


@app.get("/ready")
def ready() -> dict[str, Any]:
    """Readiness: writable storage + index reachable."""
    try:
        _bootstrap_runtime()
        from modules.ingest import get_index_stats

        st = get_index_stats()
        return {
            "status": "ready",
            "chunk_count": st["chunk_count"],
            "documents": len(st.get("documents") or []),
            "paths": runtime_info(),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/stats")
def stats() -> dict[str, Any]:
    _bootstrap_runtime()
    from modules.ingest import get_index_stats

    out = get_index_stats()
    out["runtime"] = runtime_info()
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
    """Upload a PDF into writable data dir and index it."""
    _bootstrap_runtime()
    from modules.ingest import ingest_paths

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

    dest = DATA_DIR / Path(file.filename).name
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
    # Serverless cold start: auto-build index if empty
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
            detail="Index empty. POST /ingest or upload a PDF first.",
        )

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
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "chat": "POST /chat",
        "ingest": "POST /ingest",
        "upload": "POST /ingest/upload",
        "serverless": IS_SERVERLESS,
        "note": (
            "On Vercel, storage is ephemeral under /tmp. "
            "For persistent production RAG, use Docker on Railway."
            if IS_SERVERLESS
            else "Long-running server mode."
        ),
    }
