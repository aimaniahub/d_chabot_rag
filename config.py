"""Central configuration for production PDF RAG backend."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", PROJECT_ROOT / "storage"))
CHROMA_DIR = STORAGE_DIR / "chroma"
MANIFEST_PATH = STORAGE_DIR / "manifest.json"
EVAL_REPORTS_DIR = Path(os.getenv("EVAL_REPORTS_DIR", PROJECT_ROOT / "eval_reports"))

# Auth / models
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
# fastembed model (no torch). Aliases resolved in modules.embedder
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.30"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_rag")

# Prompt / chat
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "3"))
ABSTAIN_MESSAGE = "I could not find the answer in the provided document."

# LLM
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_RETRY_BACKOFF_SEC = float(os.getenv("LLM_RETRY_BACKOFF_SEC", "1.5"))

# Eval gates (predeploy)
GATE_HIT_AT_K = float(os.getenv("GATE_HIT_AT_K", "0.70"))
GATE_P95_LATENCY_MS = float(os.getenv("GATE_P95_LATENCY_MS", "15000"))

# Server
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT") or os.getenv("API_PORT", "8000"))
UI_PORT = int(os.getenv("UI_PORT", "8501"))

# CORS — comma-separated origins for separate frontend (e.g. Vercel)
# Use * for open dev; set explicit origins in production
_CORS = os.getenv("CORS_ORIGINS", "*").strip()
CORS_ORIGINS: list[str] = (
    ["*"]
    if _CORS == "*"
    else [o.strip() for o in _CORS.split(",") if o.strip()]
)

# Optional auto-ingest of data/*.pdf on API startup (Docker/Railway)
AUTO_INGEST_ON_START = os.getenv("AUTO_INGEST_ON_START", "false").lower() in (
    "1",
    "true",
    "yes",
)


def ensure_dirs() -> None:
    """Create storage and data directories if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
