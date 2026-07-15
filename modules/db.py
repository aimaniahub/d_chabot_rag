"""PostgreSQL chat history (Railway DATABASE_URL). Optional if unset."""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Optional

logger = logging.getLogger("pdf_rag.db")

_pool = None
_init_tried = False
_schema_ready = False


def database_url() -> Optional[str]:
    return os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or None


def is_db_configured() -> bool:
    return bool(database_url())


def _get_conn():
    """Lazy connection via psycopg. Returns None if DB unavailable."""
    global _pool, _init_tried
    url = database_url()
    if not url:
        return None
    try:
        import psycopg
        from psycopg.rows import dict_row

        # Railway sometimes gives postgres:// — psycopg wants postgresql://
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://") :]
        conn = psycopg.connect(url, row_factory=dict_row, autocommit=False)
        return conn
    except Exception as exc:  # noqa: BLE001
        if not _init_tried:
            logger.warning("Postgres connect failed: %s", exc)
            _init_tried = True
        return None


def ensure_schema() -> bool:
    """Create tables if missing. Safe to call often."""
    global _schema_ready
    if _schema_ready:
        return True
    conn = _get_conn()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id UUID PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    source TEXT DEFAULT 'api',
                    language TEXT DEFAULT 'en',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_conversations_session
                    ON conversations (session_id);

                CREATE TABLE IF NOT EXISTS messages (
                    id UUID PRIMARY KEY,
                    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    sources JSONB,
                    metrics JSONB,
                    abstained BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages (conversation_id, created_at);
                """
            )
        conn.commit()
        _schema_ready = True
        logger.info("Postgres chat schema ready")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.exception("Schema init failed: %s", exc)
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        return False
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def get_or_create_conversation(
    session_id: str,
    *,
    source: str = "api",
    language: str = "en",
) -> Optional[str]:
    """Return conversation UUID for this session (latest open thread per session)."""
    if not ensure_schema():
        return None
    conn = _get_conn()
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM conversations
                WHERE session_id = %s
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if row:
                cid = str(row["id"])
                cur.execute(
                    "UPDATE conversations SET updated_at = NOW(), language = %s WHERE id = %s",
                    (language, cid),
                )
                conn.commit()
                return cid

            cid = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO conversations (id, session_id, source, language)
                VALUES (%s, %s, %s, %s)
                """,
                (cid, session_id, source, language),
            )
            conn.commit()
            return cid
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_or_create_conversation failed: %s", exc)
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        return None
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def log_chat_turn(
    *,
    session_id: str,
    question: str,
    answer: str,
    sources: list[dict] | None = None,
    metrics: dict | None = None,
    abstained: bool = False,
    source: str = "api",
    language: str = "en",
) -> Optional[dict[str, Any]]:
    """
    Persist one user question + assistant answer.
    Returns {conversation_id, user_message_id, assistant_message_id} or None.
    Never raises to callers — chat must still succeed if DB is down.
    """
    try:
        if not session_id or not question:
            return None
        if not ensure_schema():
            return None
        conv_id = get_or_create_conversation(
            session_id, source=source, language=language
        )
        if not conv_id:
            return None

        conn = _get_conn()
        if conn is None:
            return None

        user_id = str(uuid.uuid4())
        asst_id = str(uuid.uuid4())
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO messages (id, conversation_id, role, content)
                    VALUES (%s, %s, 'user', %s)
                    """,
                    (user_id, conv_id, question),
                )
                cur.execute(
                    """
                    INSERT INTO messages
                        (id, conversation_id, role, content, sources, metrics, abstained)
                    VALUES (%s, %s, 'assistant', %s, %s::jsonb, %s::jsonb, %s)
                    """,
                    (
                        asst_id,
                        conv_id,
                        answer,
                        json.dumps(sources or []),
                        json.dumps(metrics or {}),
                        abstained,
                    ),
                )
                cur.execute(
                    "UPDATE conversations SET updated_at = NOW() WHERE id = %s",
                    (conv_id,),
                )
            conn.commit()
            return {
                "conversation_id": conv_id,
                "user_message_id": user_id,
                "assistant_message_id": asst_id,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("log_chat_turn insert failed: %s", exc)
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001
                pass
            return None
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        logger.exception("log_chat_turn failed: %s", exc)
        return None


def db_health() -> dict[str, Any]:
    if not is_db_configured():
        return {"configured": False, "ok": False, "detail": "DATABASE_URL not set"}
    if not ensure_schema():
        return {"configured": True, "ok": False, "detail": "schema or connect failed"}
    conn = _get_conn()
    if conn is None:
        return {"configured": True, "ok": False, "detail": "connect failed"}
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM messages")
            row = cur.fetchone()
            n = int(row["n"]) if row else 0
            cur.execute("SELECT COUNT(*) AS n FROM conversations")
            row2 = cur.fetchone()
            c = int(row2["n"]) if row2 else 0
        return {
            "configured": True,
            "ok": True,
            "message_count": n,
            "conversation_count": c,
        }
    except Exception as exc:  # noqa: BLE001
        return {"configured": True, "ok": False, "detail": str(exc)}
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def list_recent_messages(limit: int = 40) -> list[dict[str, Any]]:
    """Recent chat turns for admin dashboard."""
    if not ensure_schema():
        return []
    conn = _get_conn()
    if conn is None:
        return []
    limit = max(1, min(int(limit), 200))
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.id, m.role, m.content, m.abstained, m.created_at,
                       c.session_id, c.source, c.language
                FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                ORDER BY m.created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall() or []
        out = []
        for r in rows:
            out.append(
                {
                    "id": str(r["id"]),
                    "role": r["role"],
                    "content": (r["content"] or "")[:2000],
                    "abstained": bool(r.get("abstained")),
                    "created_at": r["created_at"].isoformat()
                    if r.get("created_at")
                    else None,
                    "session_id": r.get("session_id"),
                    "source": r.get("source"),
                    "language": r.get("language"),
                }
            )
        return out
    except Exception as exc:  # noqa: BLE001
        logger.exception("list_recent_messages failed: %s", exc)
        return []
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
