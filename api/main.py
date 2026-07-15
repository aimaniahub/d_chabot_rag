"""
PDF RAG Backend API

Designed to be consumed by any frontend (Next.js/Vercel, React, mobile, etc.).

  GET  /health
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
    MIN_SCORE,
    TOP_K,
    ensure_dirs,
)

logger = logging.getLogger("pdf_rag.api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
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
    version="2.1.0",
    lifespan=lifespan,
)

# When allow_origins=["*"], credentials must be false (browser CORS rules).
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
    """Liveness for Railway / Docker healthchecks."""
    return {"status": "ok", "service": "pdf-rag-api"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    """Readiness: index reachable."""
    try:
        from modules.ingest import get_index_stats

        st = get_index_stats()
        return {
            "status": "ready",
            "chunk_count": st["chunk_count"],
            "documents": len(st.get("documents") or []),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/stats")
def stats() -> dict[str, Any]:
    from modules.ingest import get_index_stats

    return get_index_stats()


@app.post("/ingest")
def ingest(body: IngestRequest) -> dict[str, Any]:
    from modules.ingest import ingest_paths

    paths = [Path(body.file)] if body.file else None
    report = ingest_paths(paths=paths, rebuild=body.rebuild, force=body.force)
    return report.to_dict()


@app.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a PDF into data/ and index it."""
    from modules.ingest import ingest_paths

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

    ensure_dirs()
    dest = DATA_DIR / Path(file.filename).name
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    finally:
        await file.close()

    report = ingest_paths(paths=[dest], force=True)
    return {"saved_to": str(dest), **report.to_dict()}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> dict[str, Any]:
    from modules.chat import ChatService
    from modules.ingest import get_index_stats

    st = get_index_stats()
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
def root() -> dict[str, str]:
    return {
        "message": "PDF RAG API",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "chat": "POST /chat",
        "ingest": "POST /ingest",
        "upload": "POST /ingest/upload",
    }
