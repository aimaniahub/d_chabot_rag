"""OpenRouter LLM client with free-model rotation (no Gemini)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from config import (
    LLM_MAX_RETRIES,
    LLM_RETRY_BACKOFF_SEC,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODELS,
    OPENROUTER_SITE_NAME,
    OPENROUTER_SITE_URL,
)
from modules.metrics import timer

logger = logging.getLogger("pdf_rag.llm")


@dataclass
class LLMResult:
    text: str
    elapsed_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model: str = ""
    raw: Any = None


def _should_rotate(status_code: int | None, body: str) -> bool:
    """Rotate model on rate limit, overload, or model unavailable."""
    if status_code in (402, 408, 429, 502, 503, 524):
        return True
    b = (body or "").lower()
    return any(
        x in b
        for x in (
            "rate limit",
            "quota",
            "capacity",
            "overloaded",
            "unavailable",
            "no endpoints",
            "provider returned error",
            "timeout",
        )
    )


class OpenRouterLLM:
    """Chat completions via OpenRouter; rotates free models on failure."""

    def __init__(
        self,
        api_key: str | None = None,
        models: list[str] | None = None,
    ):
        key = api_key or OPENROUTER_API_KEY
        if not key:
            raise RuntimeError(
                "Missing OPENROUTER_API_KEY. Set it in .env / Railway variables."
            )
        self.api_key = key
        self.models = models or list(OPENROUTER_MODELS)
        if not self.models:
            raise RuntimeError("No OpenRouter models configured (OPENROUTER_MODELS).")
        self.base_url = OPENROUTER_BASE_URL.rstrip("/")

    def generate(self, prompt: str) -> str:
        return self.generate_with_usage(prompt).text

    def generate_with_usage(self, prompt: str) -> LLMResult:
        last_err: Exception | None = None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-OpenRouter-Title": OPENROUTER_SITE_NAME,
        }

        for model in self.models:
            for attempt in range(LLM_MAX_RETRIES + 1):
                try:
                    with timer() as t:
                        with httpx.Client(timeout=60.0) as client:
                            resp = client.post(
                                f"{self.base_url}/chat/completions",
                                headers=headers,
                                json={
                                    "model": model,
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": prompt,
                                        }
                                    ],
                                    "temperature": 0.2,
                                    "max_tokens": 1024,
                                },
                            )
                    body_text = resp.text
                    if resp.status_code >= 400:
                        if _should_rotate(resp.status_code, body_text):
                            logger.warning(
                                "OpenRouter model %s failed (%s); rotating. %s",
                                model,
                                resp.status_code,
                                body_text[:200],
                            )
                            last_err = RuntimeError(
                                f"{model} HTTP {resp.status_code}: {body_text[:300]}"
                            )
                            break  # next model
                        last_err = RuntimeError(
                            f"{model} HTTP {resp.status_code}: {body_text[:300]}"
                        )
                        if attempt < LLM_MAX_RETRIES:
                            time.sleep(LLM_RETRY_BACKOFF_SEC * (attempt + 1))
                            continue
                        break

                    data = resp.json()
                    choices = data.get("choices") or []
                    if not choices:
                        last_err = RuntimeError(f"{model}: empty choices")
                        break
                    msg = choices[0].get("message") or {}
                    text = (msg.get("content") or "").strip()
                    if not text:
                        last_err = RuntimeError(f"{model}: empty content")
                        break

                    usage = data.get("usage") or {}
                    return LLMResult(
                        text=text,
                        elapsed_ms=t["ms"],
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                        total_tokens=usage.get("total_tokens"),
                        model=model,
                        raw=data,
                    )
                except httpx.TimeoutException as exc:
                    last_err = exc
                    logger.warning("OpenRouter timeout on %s", model)
                    break  # rotate model
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    logger.warning("OpenRouter error on %s: %s", model, exc)
                    if attempt < LLM_MAX_RETRIES:
                        time.sleep(LLM_RETRY_BACKOFF_SEC * (attempt + 1))
                        continue
                    break

        raise RuntimeError(
            f"OpenRouter generation failed on all models {self.models}: {last_err}"
        ) from last_err


# Backward-compatible alias used by chat module / older imports
GeminiLLM = OpenRouterLLM
