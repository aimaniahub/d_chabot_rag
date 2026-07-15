"""Grounded prompt construction with citations and context budget."""
from __future__ import annotations

from typing import Optional, Sequence

from config import ABSTAIN_MESSAGE, MAX_CONTEXT_CHARS, MAX_HISTORY_TURNS
from modules.metrics import timer
from modules.vector_store import RetrievedChunk


def format_context(hits: Sequence[RetrievedChunk], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Pack ranked chunks as numbered sources under a char budget."""
    if not hits:
        return "(no relevant context retrieved)"

    parts: list[str] = []
    used = 0
    for i, h in enumerate(hits, start=1):
        page = (
            f"p.{h.page_start}"
            if h.page_start == h.page_end
            else f"p.{h.page_start}-{h.page_end}"
        )
        header = f"[{i}] source={h.source} {page} score={h.score:.3f}"
        block = f"{header}\n{h.text.strip()}"
        # +2 for joining newlines
        extra = len(block) + (2 if parts else 0)
        if used + extra > max_chars and parts:
            break
        parts.append(block)
        used += extra
    return "\n\n".join(parts)


def format_history(history: Sequence[dict] | None, max_turns: int = MAX_HISTORY_TURNS) -> str:
    """history items: {role: user|assistant, content: str}"""
    if not history:
        return ""
    turns = list(history)[-max_turns * 2 :]
    lines = []
    for t in turns:
        role = t.get("role", "user")
        content = (t.get("content") or "").strip()
        if content:
            lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def _language_instruction(language: str | None) -> str:
    lang = (language or "en").lower()
    if lang in ("kn", "kannada"):
        return "Reply in Kannada (ಕನ್ನಡ) unless the user clearly uses another language."
    if lang in ("hi", "hindi"):
        return "Reply in Hindi (हिंदी) unless the user clearly uses another language."
    return "Reply in clear English unless the user clearly uses another language."


def build_prompt(
    question: str,
    context: str | Sequence[RetrievedChunk],
    *,
    history: Sequence[dict] | None = None,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    language: str | None = "en",
) -> str:
    """Build grounded QA prompt. `context` may be raw string (legacy) or hits."""
    if isinstance(context, str):
        context_block = context
    else:
        context_block = format_context(context, max_chars=max_context_chars)

    history_block = format_history(history)
    history_section = f"\nPrior conversation:\n{history_block}\n" if history_block else ""
    lang_rule = _language_instruction(language)

    return f"""You are Darvi Assistant for Darvi Group (agriculture, plants, farmland, IoT, registration).

Rules:
- Answer using the provided company documents/context first.
- {lang_rule}
- Be concise, friendly, and structured (short paragraphs or bullets).
- If the answer is not in the context, say you do not have that detail in the documents and suggest contacting Darvi: +91 99868 90777, darvigroup@gmail.com. Do not invent prices or policies.
- When you use document facts, cite sources with [n] markers matching the context blocks.
- Prefer helpful next steps (registration, services, contact) when relevant.

Context:
{context_block}
{history_section}
Question:
{question}

Answer:
"""


def build_prompt_with_metrics(
    question: str,
    hits: Sequence[RetrievedChunk],
    history: Sequence[dict] | None = None,
) -> tuple[str, float]:
    with timer() as t:
        prompt = build_prompt(question, hits, history=history)
    return prompt, t["ms"]


def sources_payload(hits: Sequence[RetrievedChunk]) -> list[dict]:
    return [
        {
            "n": i,
            "chunk_id": h.chunk_id,
            "source": h.source,
            "doc_id": h.doc_id,
            "page_start": h.page_start,
            "page_end": h.page_end,
            "score": round(h.score, 4),
            "preview": h.text[:240],
        }
        for i, h in enumerate(hits, start=1)
    ]
