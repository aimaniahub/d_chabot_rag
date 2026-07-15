"""API key auth for chat + admin."""
from __future__ import annotations

import hmac
import secrets
from typing import Optional

from fastapi import Header, HTTPException, Request

from config import ADMIN_KEY, API_KEY, DOCS_ENABLED


def _match(provided: str | None, expected: str | None) -> bool:
    if not expected or not provided:
        return False
    return hmac.compare_digest(provided.strip(), expected.strip())


def extract_key(
    request: Request,
    x_api_key: Optional[str] = None,
    authorization: Optional[str] = None,
) -> Optional[str]:
    if x_api_key:
        return x_api_key.strip()
    # query ?key= for simple admin HTML forms
    q = request.query_params.get("key")
    if q:
        return q.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
) -> str:
    """Chat / read endpoints — site backend uses this."""
    if not API_KEY:
        # Misconfigured production: refuse rather than leave open
        raise HTTPException(
            status_code=503,
            detail="API_KEY not configured on server. Set API_KEY in Railway variables.",
        )
    key = extract_key(request, x_api_key, authorization)
    if _match(key, API_KEY) or _match(key, ADMIN_KEY):
        return key or ""
    raise HTTPException(
        status_code=401,
        detail="Unauthorized. Provide header X-API-Key.",
    )


def require_admin_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
) -> str:
    """Upload / ingest / admin dashboard."""
    if not ADMIN_KEY and not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="ADMIN_KEY not configured. Set ADMIN_KEY in Railway variables.",
        )
    expected = ADMIN_KEY or API_KEY
    key = extract_key(request, x_api_key, authorization)
    if _match(key, expected) or (ADMIN_KEY and API_KEY and _match(key, ADMIN_KEY)):
        return key or ""
    # allow API_KEY only if no separate ADMIN_KEY
    if not ADMIN_KEY and _match(key, API_KEY):
        return key or ""
    raise HTTPException(
        status_code=401,
        detail="Unauthorized. Admin key required (X-API-Key).",
    )


def docs_allowed() -> bool:
    return bool(DOCS_ENABLED)
