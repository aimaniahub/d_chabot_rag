"""Gemini LLM client (google-genai SDK) with retries, 429 handling, model fallback."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from google import genai

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_MAX_RETRIES,
    LLM_RETRY_BACKOFF_SEC,
)
from modules.metrics import timer

# If free-tier quota hits on one model, try others before failing.
_FALLBACK_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-2.5-flash",
]


def _is_quota_error(exc: BaseException) -> bool:
    s = str(exc).lower()
    return (
        "429" in s
        or "resource_exhausted" in s
        or "quota" in s
        or "rate limit" in s
    )


def _model_chain(primary: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in [primary, *os.getenv("GEMINI_FALLBACK_MODELS", "").split(","), *_FALLBACK_MODELS]:
        m = (m or "").strip()
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


@dataclass
class LLMResult:
    text: str
    elapsed_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model: str = GEMINI_MODEL
    raw: Any = None


class GeminiLLM:
    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        key = api_key or GEMINI_API_KEY
        if not key:
            raise RuntimeError(
                "Missing GOOGLE_API_KEY / GEMINI_API_KEY. Set it in .env / Railway variables"
            )
        self.model_name = model_name or GEMINI_MODEL
        self.client = genai.Client(api_key=key)

    def generate(self, prompt: str) -> str:
        return self.generate_with_usage(prompt).text

    def generate_with_usage(self, prompt: str) -> LLMResult:
        last_err: Exception | None = None
        models = _model_chain(self.model_name)

        for model in models:
            for attempt in range(LLM_MAX_RETRIES + 1):
                try:
                    with timer() as t:
                        response = self.client.models.generate_content(
                            model=model,
                            contents=prompt,
                        )
                    text = (getattr(response, "text", None) or "").strip()
                    if not text:
                        try:
                            cands = response.candidates or []
                            if cands and cands[0].content and cands[0].content.parts:
                                text = "".join(
                                    getattr(p, "text", "") or ""
                                    for p in cands[0].content.parts
                                ).strip()
                        except Exception:  # noqa: BLE001
                            text = ""

                    prompt_tokens = completion_tokens = total_tokens = None
                    meta = getattr(response, "usage_metadata", None)
                    if meta is not None:
                        prompt_tokens = getattr(meta, "prompt_token_count", None)
                        completion_tokens = getattr(meta, "candidates_token_count", None)
                        total_tokens = getattr(meta, "total_token_count", None)

                    return LLMResult(
                        text=text,
                        elapsed_ms=t["ms"],
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        model=model,
                        raw=response,
                    )
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    if _is_quota_error(exc):
                        # Try next model immediately (free-tier often per-model)
                        break
                    if attempt >= LLM_MAX_RETRIES:
                        break
                    time.sleep(LLM_RETRY_BACKOFF_SEC * (attempt + 1))

        if last_err and _is_quota_error(last_err):
            raise RuntimeError(
                "Gemini quota exceeded (free tier). Wait a minute, or set a paid "
                "Google AI key / change GEMINI_MODEL on Railway. "
                f"Detail: {last_err}"
            ) from last_err
        raise RuntimeError(f"Gemini generation failed: {last_err}") from last_err
