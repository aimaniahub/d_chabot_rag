"""Gemini LLM client (google-genai SDK) with retries and usage capture."""
from __future__ import annotations

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
                "Missing GOOGLE_API_KEY / GEMINI_API_KEY. Set it in .env"
            )
        self.model_name = model_name or GEMINI_MODEL
        self.client = genai.Client(api_key=key)

    def generate(self, prompt: str) -> str:
        """Return text only."""
        return self.generate_with_usage(prompt).text

    def generate_with_usage(self, prompt: str) -> LLMResult:
        last_err: Exception | None = None
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                with timer() as t:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                    )
                text = (getattr(response, "text", None) or "").strip()
                if not text:
                    # Fallback: pull first candidate parts
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
                    model=self.model_name,
                    raw=response,
                )
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt >= LLM_MAX_RETRIES:
                    break
                time.sleep(LLM_RETRY_BACKOFF_SEC * (attempt + 1))
        raise RuntimeError(f"Gemini generation failed: {last_err}") from last_err
