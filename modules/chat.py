"""Chat / query orchestration over persistent index."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from config import MIN_SCORE, TOP_K
from modules.embedder import Embedder
from modules.llm import GeminiLLM, LLMResult
from modules.metrics import (
    QueryMetrics,
    TokenUsage,
    TimingBreakdown,
    estimate_tokens_from_chars,
    timer,
)
from modules.prompt_builder import build_prompt, sources_payload
from modules.retriever import Retriever
from modules.vector_store import RetrievedChunk, VectorStore


@dataclass
class ChatResponse:
    answer: str
    sources: list[dict]
    hits: list[RetrievedChunk] = field(default_factory=list)
    metrics: QueryMetrics = field(default_factory=QueryMetrics)
    abstained: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "abstained": self.abstained,
            "metrics": self.metrics.to_dict(),
        }


class ChatService:
    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        llm: GeminiLLM | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
    ):
        self.store = store or VectorStore()
        self.embedder = embedder or Embedder()
        self.llm = llm  # lazy optional for retrieve-only tests
        self.top_k = TOP_K if top_k is None else top_k
        self.min_score = MIN_SCORE if min_score is None else min_score
        self.retriever = Retriever(
            store=self.store,
            embedder=self.embedder,
            top_k=self.top_k,
            min_score=self.min_score,
        )

    def _ensure_llm(self) -> GeminiLLM:
        if self.llm is None:
            self.llm = GeminiLLM()
        return self.llm

    def ask(
        self,
        question: str,
        *,
        history: Sequence[dict] | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
        doc_id: str | None = None,
    ) -> ChatResponse:
        question = (question or "").strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        metrics = QueryMetrics(top_k=top_k or self.top_k)
        timings = TimingBreakdown()

        with timer() as total_t:
            with timer() as emb_t:
                q_emb = self.embedder.embed_question(question)
            timings.embed_ms = emb_t["ms"]

            retrieval = self.retriever.retrieve(
                question,
                top_k=top_k,
                min_score=min_score,
                doc_id=doc_id,
                question_embedding=q_emb,
            )
            # retrieve() includes embed if no emb — we passed emb so mostly query time
            timings.retrieve_ms = retrieval.elapsed_ms
            hits = retrieval.hits
            metrics.hits_returned = len(retrieval.all_hits)
            metrics.hits_above_threshold = len(hits)
            metrics.max_score = retrieval.max_score

            with timer() as prompt_t:
                prompt = build_prompt(question, hits, history=history)
            timings.prompt_ms = prompt_t["ms"]
            metrics.tokens.estimated_prompt_chars = len(prompt)
            metrics.tokens.prompt_tokens = estimate_tokens_from_chars(len(prompt))

            llm = self._ensure_llm()
            llm_result: LLMResult = llm.generate_with_usage(prompt)
            timings.llm_ms = llm_result.elapsed_ms
            if llm_result.prompt_tokens is not None:
                metrics.tokens.prompt_tokens = llm_result.prompt_tokens
            metrics.tokens.completion_tokens = llm_result.completion_tokens
            metrics.tokens.total_tokens = llm_result.total_tokens

            answer = llm_result.text
            abstained = len(hits) == 0 or "could not find the answer" in answer.lower()

        timings.total_ms = total_t["ms"]
        metrics.timings = timings

        return ChatResponse(
            answer=answer,
            sources=sources_payload(hits),
            hits=list(hits),
            metrics=metrics,
            abstained=abstained,
        )
