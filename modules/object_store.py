"""Optional S3-compatible storage (Railway / Tigris buckets) for original uploads."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from config import (
    S3_ACCESS_KEY,
    S3_BUCKET,
    S3_ENABLED,
    S3_ENDPOINT,
    S3_PREFIX,
    S3_REGION,
    S3_SECRET_KEY,
)

logger = logging.getLogger("pdf_rag.object_store")

_client = None


def is_enabled() -> bool:
    return S3_ENABLED


def _get_client():
    global _client
    if not S3_ENABLED:
        return None
    if _client is not None:
        return _client
    try:
        import boto3
        from botocore.config import Config

        # Works with Railway Buckets and Tigris (t3.storageapi.dev)
        _client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            region_name=S3_REGION or "auto",
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "virtual"},
            ),
        )
        return _client
    except Exception as exc:  # noqa: BLE001
        logger.exception("S3 client init failed: %s", exc)
        return None


def object_key(filename: str) -> str:
    name = Path(filename).name
    prefix = S3_PREFIX if S3_PREFIX.endswith("/") else S3_PREFIX + "/"
    return f"{prefix}{name}"


def upload_file(local_path: Path, key: str | None = None) -> Optional[str]:
    """Upload local file to bucket. Returns object key or None."""
    client = _get_client()
    if client is None:
        return None
    path = Path(local_path)
    if not path.is_file():
        return None
    key = key or object_key(path.name)
    try:
        extra = {}
        # best-effort content type
        suffix = path.suffix.lower()
        ctype = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".md": "text/markdown",
            ".txt": "text/plain",
        }.get(suffix)
        if ctype:
            extra["ExtraArgs"] = {"ContentType": ctype}
        client.upload_file(str(path), S3_BUCKET, key, **extra)
        logger.info("Uploaded %s -> s3://%s/%s", path.name, S3_BUCKET, key)
        return key
    except Exception as exc:  # noqa: BLE001
        logger.exception("S3 upload failed for %s: %s", path, exc)
        return None


def download_file(key: str, dest: Path) -> bool:
    client = _get_client()
    if client is None:
        return False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(S3_BUCKET, key, str(dest))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("S3 download failed %s: %s", key, exc)
        return False


def delete_object(key: str) -> bool:
    client = _get_client()
    if client is None:
        return False
    try:
        client.delete_object(Bucket=S3_BUCKET, Key=key)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("S3 delete failed %s: %s", key, exc)
        return False


def list_keys(prefix: str | None = None) -> list[str]:
    client = _get_client()
    if client is None:
        return []
    prefix = prefix if prefix is not None else S3_PREFIX
    try:
        keys: list[str] = []
        token = None
        while True:
            kwargs = {"Bucket": S3_BUCKET, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            resp = client.list_objects_v2(**kwargs)
            for obj in resp.get("Contents") or []:
                keys.append(obj["Key"])
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return keys
    except Exception as exc:  # noqa: BLE001
        logger.warning("S3 list failed: %s", exc)
        return []


def restore_missing_to_local(data_dir: Path) -> int:
    """Download bucket objects missing on local disk."""
    if not is_enabled():
        return 0
    keys = list_keys()
    restored = 0
    for key in keys:
        name = Path(key).name
        if not name or name.endswith("/"):
            continue
        dest = data_dir / name
        if dest.exists():
            continue
        if download_file(key, dest):
            restored += 1
    if restored:
        logger.info("Restored %s file(s) from S3 bucket to %s", restored, data_dir)
    return restored


def ping() -> dict:
    """Live connectivity check for admin UI."""
    if not is_enabled():
        return {
            "ok": False,
            "enabled": False,
            "detail": "S3 not configured (need BUCKET + ACCESS_KEY_ID + SECRET_ACCESS_KEY)",
        }
    client = _get_client()
    if client is None:
        return {"ok": False, "enabled": True, "detail": "client init failed"}
    try:
        # list a few keys under prefix
        keys = list_keys()
        return {
            "ok": True,
            "enabled": True,
            "bucket": S3_BUCKET,
            "endpoint": S3_ENDPOINT,
            "region": S3_REGION,
            "prefix": S3_PREFIX,
            "object_count": len(keys),
            "sample_keys": keys[:20],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "enabled": True,
            "bucket": S3_BUCKET,
            "endpoint": S3_ENDPOINT,
            "detail": str(exc),
        }


def status() -> dict:
    return {
        "enabled": is_enabled(),
        "bucket": S3_BUCKET or None,
        "endpoint": S3_ENDPOINT if is_enabled() else None,
        "prefix": S3_PREFIX,
        "region": S3_REGION,
    }
