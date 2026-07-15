"""Central configuration for production PDF RAG backend."""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("pdf_rag.config")

# ---------------------------------------------------------------------------
# Environment detection
# Vercel / AWS Lambda: package dir is read-only; only /tmp is writable.
# ---------------------------------------------------------------------------
IS_VERCEL = os.getenv("VERCEL", "").lower() in ("1", "true") or bool(
    os.getenv("VERCEL_ENV")
)
IS_LAMBDA = bool(
    os.getenv("AWS_LAMBDA_FUNCTION_NAME") or os.getenv("AWS_EXECUTION_ENV")
)
IS_SERVERLESS = IS_VERCEL or IS_LAMBDA or os.getenv("SERVERLESS", "").lower() in (
    "1",
    "true",
    "yes",
)

PROJECT_ROOT = Path(__file__).resolve().parent


def _runtime_root() -> Path:
    """Root for writable paths (index, uploads, caches)."""
    override = os.getenv("RUNTIME_ROOT") or os.getenv("WRITABLE_ROOT")
    if override:
        return Path(override)
    if IS_SERVERLESS:
        return Path("/tmp/pdf_rag")
    return PROJECT_ROOT


RUNTIME_ROOT = _runtime_root()

# Seed PDFs shipped with the repo (may be read-only on Vercel)
SEED_DATA_DIR = Path(os.getenv("SEED_DATA_DIR", PROJECT_ROOT / "data"))

# Writable data dir (uploads + copies of seeds on serverless)
DATA_DIR = Path(os.getenv("DATA_DIR", RUNTIME_ROOT / "data"))
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", RUNTIME_ROOT / "storage"))
CHROMA_DIR = STORAGE_DIR / "chroma"
MANIFEST_PATH = STORAGE_DIR / "manifest.json"
EVAL_REPORTS_DIR = Path(
    os.getenv("EVAL_REPORTS_DIR", RUNTIME_ROOT / "eval_reports")
)

# Caches must also be under a writable path on serverless
if IS_SERVERLESS:
    _cache = RUNTIME_ROOT / ".cache"
    os.environ.setdefault("HF_HOME", str(_cache / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_cache / "huggingface"))
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(_cache / "fastembed"))
    os.environ.setdefault("XDG_CACHE_HOME", str(_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(_cache / "transformers"))

# Auth / models
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
# 2.0-flash usually has higher free-tier allowance than 2.5-flash
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Retrieval (lower min_score helps table-style plant lists match better)
TOP_K = int(os.getenv("TOP_K", "8"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.22"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_rag")

# Prompt / chat
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "3"))
ABSTAIN_MESSAGE = "I could not find the answer in the provided document."

# LLM
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_RETRY_BACKOFF_SEC = float(os.getenv("LLM_RETRY_BACKOFF_SEC", "1.5"))

# Eval gates
GATE_HIT_AT_K = float(os.getenv("GATE_HIT_AT_K", "0.70"))
GATE_P95_LATENCY_MS = float(os.getenv("GATE_P95_LATENCY_MS", "15000"))

# Server
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT") or os.getenv("API_PORT", "8000"))
UI_PORT = int(os.getenv("UI_PORT", "8501"))

# CORS
_CORS = os.getenv("CORS_ORIGINS", "*").strip()
CORS_ORIGINS: list[str] = (
    ["*"]
    if _CORS == "*"
    else [o.strip() for o in _CORS.split(",") if o.strip()]
)

# Auto-ingest: default ON for serverless (ephemeral /tmp index each cold start)
_auto_default = "true" if IS_SERVERLESS else "false"
AUTO_INGEST_ON_START = os.getenv("AUTO_INGEST_ON_START", _auto_default).lower() in (
    "1",
    "true",
    "yes",
)


def ensure_dirs() -> None:
    """Create writable runtime directories. Safe on read-only package roots."""
    for path in (DATA_DIR, STORAGE_DIR, CHROMA_DIR, EVAL_REPORTS_DIR):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # Never crash the process; log and continue
            logger.error("Could not create directory %s: %s", path, exc)
            if not IS_SERVERLESS:
                raise


def sync_seed_pdfs() -> int:
    """
    Copy bundled data/*.pdf into writable DATA_DIR when needed.
    On Vercel, seeds live under /var/task/data (read-only); index/uploads use /tmp.
    """
    ensure_dirs()
    if not SEED_DATA_DIR.exists():
        return 0
    if SEED_DATA_DIR.resolve() == DATA_DIR.resolve():
        return 0

    copied = 0
    for pattern in ("*.pdf", "*.docx"):
        for src in SEED_DATA_DIR.glob(pattern):
            dest = DATA_DIR / src.name
            try:
                if not dest.exists() or dest.stat().st_size != src.stat().st_size:
                    shutil.copy2(src, dest)
                    copied += 1
            except OSError as exc:
                logger.warning("Could not copy seed file %s: %s", src.name, exc)
    return copied


def runtime_info() -> dict:
    return {
        "is_serverless": IS_SERVERLESS,
        "is_vercel": IS_VERCEL,
        "is_lambda": IS_LAMBDA,
        "project_root": str(PROJECT_ROOT),
        "runtime_root": str(RUNTIME_ROOT),
        "data_dir": str(DATA_DIR),
        "seed_data_dir": str(SEED_DATA_DIR),
        "storage_dir": str(STORAGE_DIR),
        "chroma_dir": str(CHROMA_DIR),
    }
