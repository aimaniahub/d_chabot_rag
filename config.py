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
# Railway sets RAILWAY_ENVIRONMENT / RAILWAY_PROJECT_ID
IS_RAILWAY = bool(
    os.getenv("RAILWAY_ENVIRONMENT")
    or os.getenv("RAILWAY_PROJECT_ID")
    or os.getenv("RAILWAY_SERVICE_ID")
)

PROJECT_ROOT = Path(__file__).resolve().parent


def _persist_root() -> Path:
    """
    Single root for uploads + Chroma + caches.

    Railway allows only ONE volume per service — mount it at /data.
    Layout:
      /data/uploads   <- user PDF/DOCX/MD
      /data/storage   <- chroma + manifest + model cache
    """
    if os.getenv("PERSIST_ROOT"):
        return Path(os.getenv("PERSIST_ROOT"))
    if os.getenv("RUNTIME_ROOT") or os.getenv("WRITABLE_ROOT"):
        return Path(os.getenv("RUNTIME_ROOT") or os.getenv("WRITABLE_ROOT"))
    if IS_SERVERLESS:
        return Path("/tmp/pdf_rag")
    # Production Docker / Railway default
    if IS_RAILWAY or Path("/data").is_dir() or os.getenv("USE_DATA_VOLUME", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return Path("/data")
    return PROJECT_ROOT


PERSIST_ROOT = _persist_root()
RUNTIME_ROOT = PERSIST_ROOT  # alias

# Seed files baked into the image (repo data/) — only for first boot demo
SEED_DATA_DIR = Path(os.getenv("SEED_DATA_DIR", PROJECT_ROOT / "data"))

# Explicit env still wins (legacy)
DATA_DIR = Path(os.getenv("DATA_DIR", PERSIST_ROOT / "uploads"))
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", PERSIST_ROOT / "storage"))
CHROMA_DIR = STORAGE_DIR / "chroma"
MANIFEST_PATH = STORAGE_DIR / "manifest.json"
EVAL_REPORTS_DIR = Path(os.getenv("EVAL_REPORTS_DIR", PERSIST_ROOT / "eval_reports"))

# Model / embed caches under persistent storage
_cache = STORAGE_DIR / ".cache"
os.environ.setdefault("HF_HOME", str(_cache / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_cache / "huggingface"))
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(_cache / "fastembed"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_cache / "transformers"))

# Embeddings
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Security
API_KEY = (os.getenv("API_KEY") or os.getenv("RAG_API_KEY") or "").strip()
ADMIN_KEY = (os.getenv("ADMIN_KEY") or "").strip()
DOCS_ENABLED = os.getenv("DOCS_ENABLED", "false").lower() in ("1", "true", "yes")

# OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OR_API_KEY")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
_default_models = (
    "google/gemma-4-31b-it:free,"
    "google/gemma-4-26b-a4b-it:free,"
    "openai/gpt-oss-20b:free"
)
OPENROUTER_MODELS: list[str] = [
    m.strip()
    for m in os.getenv("OPENROUTER_MODELS", _default_models).split(",")
    if m.strip()
]
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "https://darvigroup.in")
OPENROUTER_SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "Darvi RAG Assistant")

GEMINI_API_KEY = OPENROUTER_API_KEY
GEMINI_MODEL = OPENROUTER_MODELS[0] if OPENROUTER_MODELS else ""

# Chunking / retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "8"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.22"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_rag")

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "3"))
ABSTAIN_MESSAGE = "I could not find the answer in the provided document."

LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))
LLM_RETRY_BACKOFF_SEC = float(os.getenv("LLM_RETRY_BACKOFF_SEC", "1.0"))

GATE_HIT_AT_K = float(os.getenv("GATE_HIT_AT_K", "0.70"))
GATE_P95_LATENCY_MS = float(os.getenv("GATE_P95_LATENCY_MS", "15000"))

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT") or os.getenv("API_PORT", "8000"))
UI_PORT = int(os.getenv("UI_PORT", "8501"))

_CORS = os.getenv("CORS_ORIGINS", "*").strip()
CORS_ORIGINS: list[str] = (
    ["*"]
    if _CORS == "*"
    else [o.strip() for o in _CORS.split(",") if o.strip()]
)

# Auto-ingest: re-scan uploads dir. NEVER prune missing by default (wipes index).
_auto_default = "true" if IS_SERVERLESS else "true" if IS_RAILWAY else "false"
AUTO_INGEST_ON_START = os.getenv("AUTO_INGEST_ON_START", _auto_default).lower() in (
    "1",
    "true",
    "yes",
)
# Dangerous on ephemeral disks — only enable if you know files always live on volume
PRUNE_MISSING_ON_INGEST = os.getenv("PRUNE_MISSING_ON_INGEST", "false").lower() in (
    "1",
    "true",
    "yes",
)

# Railway Bucket (S3-compatible) — optional backup of original files
# Variables typically injected when you connect a Bucket to the service:
#   BUCKET, ACCESS_KEY_ID, SECRET_ACCESS_KEY, ENDPOINT, REGION
S3_BUCKET = (
    os.getenv("BUCKET")
    or os.getenv("S3_BUCKET")
    or os.getenv("AWS_S3_BUCKET")
    or ""
).strip()
S3_ACCESS_KEY = (
    os.getenv("ACCESS_KEY_ID")
    or os.getenv("AWS_ACCESS_KEY_ID")
    or os.getenv("S3_ACCESS_KEY_ID")
    or ""
).strip()
S3_SECRET_KEY = (
    os.getenv("SECRET_ACCESS_KEY")
    or os.getenv("AWS_SECRET_ACCESS_KEY")
    or os.getenv("S3_SECRET_ACCESS_KEY")
    or ""
).strip()
S3_ENDPOINT = (
    os.getenv("ENDPOINT")
    or os.getenv("AWS_ENDPOINT_URL")
    or os.getenv("S3_ENDPOINT")
    or "https://storage.railway.app"
).strip()
S3_REGION = (os.getenv("REGION") or os.getenv("AWS_REGION") or "auto").strip()
S3_PREFIX = (os.getenv("S3_PREFIX") or "rag-uploads/").strip()
S3_ENABLED = bool(S3_BUCKET and S3_ACCESS_KEY and S3_SECRET_KEY)


def ensure_dirs() -> None:
    """Create writable runtime directories."""
    for path in (PERSIST_ROOT, DATA_DIR, STORAGE_DIR, CHROMA_DIR, EVAL_REPORTS_DIR, _cache):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("Could not create directory %s: %s", path, exc)
            if not IS_SERVERLESS:
                raise


def storage_health() -> dict:
    """Report whether persist paths look usable (for admin UI)."""
    ensure_dirs()
    probe = STORAGE_DIR / ".write_test"
    writable = False
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        writable = True
    except OSError as exc:
        logger.error("Storage not writable: %s", exc)

    # Heuristic: volume often mounted at /data
    on_volume = str(PERSIST_ROOT).startswith("/data") or bool(
        os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
    )
    return {
        "writable": writable,
        "persist_root": str(PERSIST_ROOT),
        "data_dir": str(DATA_DIR),
        "storage_dir": str(STORAGE_DIR),
        "chroma_dir": str(CHROMA_DIR),
        "likely_volume_mounted": on_volume,
        "is_railway": IS_RAILWAY,
        "s3_enabled": S3_ENABLED,
        "s3_bucket": S3_BUCKET or None,
        "s3_endpoint": S3_ENDPOINT if S3_ENABLED else None,
        "warning": None
        if (writable and (on_volume or not IS_RAILWAY))
        else (
            "Mount a Railway Volume at /data or files/index will be lost on redeploy. "
            "Optional: connect a Railway Bucket for file backup."
        ),
    }


def sync_seed_pdfs() -> int:
    """Copy bundled seed files into DATA_DIR if missing (demo only)."""
    ensure_dirs()
    if not SEED_DATA_DIR.exists():
        return 0
    try:
        if SEED_DATA_DIR.resolve() == DATA_DIR.resolve():
            return 0
    except OSError:
        pass

    copied = 0
    for pattern in ("*.pdf", "*.docx", "*.md", "*.txt"):
        for src in SEED_DATA_DIR.glob(pattern):
            dest = DATA_DIR / src.name
            try:
                if not dest.exists():
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
        "is_railway": IS_RAILWAY,
        "project_root": str(PROJECT_ROOT),
        "persist_root": str(PERSIST_ROOT),
        "runtime_root": str(RUNTIME_ROOT),
        "data_dir": str(DATA_DIR),
        "seed_data_dir": str(SEED_DATA_DIR),
        "storage_dir": str(STORAGE_DIR),
        "chroma_dir": str(CHROMA_DIR),
        "storage": storage_health(),
    }
