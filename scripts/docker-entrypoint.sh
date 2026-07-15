#!/bin/sh
set -e

# Railway / cloud inject PORT — bind uvicorn to it when using default CMD
if [ "$#" -gt 0 ] && [ "$1" = "uvicorn" ]; then
  # Rewrite --port if PORT env set
  if [ -n "$PORT" ]; then
    # Drop any existing --port N then append
    set -- uvicorn api.main:app --host 0.0.0.0 --port "$PORT"
  fi
fi

# Optional pre-start ingest when not using lifespan only
if [ "${RUN_INGEST_BEFORE_START:-false}" = "true" ]; then
  echo "[entrypoint] Running pre-start ingest..."
  python app.py ingest || true
fi

exec "$@"
