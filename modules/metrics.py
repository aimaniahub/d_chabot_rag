"""Latency / token metrics helpers."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Generator, Optional


@dataclass
class TimingBreakdown:
    total_ms: float = 0.0
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    prompt_ms: float = 0.0
    llm_ms: float = 0.0
    ingest_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class TokenUsage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_prompt_chars: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryMetrics:
    timings: TimingBreakdown = field(default_factory=TimingBreakdown)
    tokens: TokenUsage = field(default_factory=TokenUsage)
    top_k: int = 0
    hits_returned: int = 0
    hits_above_threshold: int = 0
    max_score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timings": self.timings.to_dict(),
            "tokens": self.tokens.to_dict(),
            "top_k": self.top_k,
            "hits_returned": self.hits_returned,
            "hits_above_threshold": self.hits_above_threshold,
            "max_score": self.max_score,
        }


@contextmanager
def timer() -> Generator[dict[str, float], None, None]:
    """Context manager that records elapsed ms into box['ms']."""
    box: dict[str, float] = {"ms": 0.0}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["ms"] = (time.perf_counter() - start) * 1000.0


def estimate_tokens_from_chars(chars: int) -> int:
    """Rough token estimate (~4 chars/token)."""
    return max(1, chars // 4) if chars else 0


def tokens_per_minute(total_tokens: int, elapsed_sec: float) -> float:
    if elapsed_sec <= 0:
        return 0.0
    return (total_tokens / elapsed_sec) * 60.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)
