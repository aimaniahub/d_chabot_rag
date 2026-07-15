# PDF RAG Backend — Railway / Docker production image
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Single persistent root — mount Railway Volume at /data
    PERSIST_ROOT=/data \
    DATA_DIR=/data/uploads \
    STORAGE_DIR=/data/storage \
    HF_HOME=/data/storage/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/data/storage/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/data/storage/.cache/fastembed \
    XDG_CACHE_HOME=/data/storage/.cache \
    API_HOST=0.0.0.0 \
    PORT=8000 \
    AUTO_INGEST_ON_START=true \
    PRUNE_MISSING_ON_INGEST=false \
    USE_DATA_VOLUME=true

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# /data is the volume mount point (uploads + chroma live here)
RUN mkdir -p /data/uploads /data/storage /data/storage/chroma /data/storage/.cache /data/eval_reports \
    && sed -i 's/\r$//' /app/scripts/docker-entrypoint.sh \
    && chmod +x /app/scripts/docker-entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
  CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/health" || exit 1

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
