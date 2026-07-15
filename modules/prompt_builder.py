"""Grounded prompt construction — strict systematic answer format."""
from __future__ import annotations

from typing import Sequence

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
        extra = len(block) + (2 if parts else 0)
        if used + extra > max_chars and parts:
            break
        parts.append(block)
        used += extra
    return "\n\n".join(parts)


def format_history(history: Sequence[dict] | None, max_turns: int = MAX_HISTORY_TURNS) -> str:
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
        return "Write the entire answer in Kannada (ಕನ್ನಡ)."
    if lang in ("hi", "hindi"):
        return "Write the entire answer in Hindi (हिंदी)."
    return "Write the entire answer in clear English."


def build_prompt(
    question: str,
    context: str | Sequence[RetrievedChunk],
    *,
    history: Sequence[dict] | None = None,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    language: str | None = "en",
) -> str:
    """Strict Darvi answer format: header + bullets/para + closing ask."""
    if isinstance(context, str):
        context_block = context
    else:
        context_block = format_context(context, max_chars=max_context_chars)

    history_block = format_history(history)
    history_section = f"\nPrior conversation:\n{history_block}\n" if history_block else ""
    lang_rule = _language_instruction(language)

    return f"""You are Darvi Assistant for Darvi Group (plants, horticulture, farmland, IoT, registration, prices).

## STRICT OUTPUT FORMAT (always follow exactly)
1) First line: a short **Header** (bold markdown like **Guava Varieties**). One line only. No intro fluff.
2) Then the body:
   - For lists (varieties, prices, steps, items): use bullet points only (`- item`). One fact per bullet. Include price/unit if present in context.
   - For explanations: short plain paragraphs. No filler. No marketing language.
3) Do NOT write unnecessary sentences (no "I'd be happy to help", no "Certainly", no long preambles).
4) Last line MUST be exactly one short follow-up question, e.g. "Is there anything else you need?"
5) Optional: if you used a document fact, you may add a small citation like [1] on that line — never invent sources.

## CONTENT RULES
- Answer ONLY from the Context below. Do not invent varieties, prices, or policies.
- {lang_rule}
- If context does not contain the answer, use this exact structure:
  **Not Found**
  - {ABSTAIN_MESSAGE}
  - Contact: +91 99868 90777 | darvigroup@gmail.com
  Is there anything else you need?

## EXAMPLE SHAPE
**Guava Varieties**
- Royl green
- L-49
- Taiwan pink
Is there anything else you need?

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
