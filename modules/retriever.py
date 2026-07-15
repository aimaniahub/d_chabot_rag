"""Top-k retrieval with score threshold."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from config import MIN_SCORE, TOP_K
from modules.embedder import Embedder
from modules.metrics import timer
from modules.vector_store import RetrievedChunk, VectorStore


@dataclass
class RetrievalResult:
    hits: list[RetrievedChunk]
    all_hits: list[RetrievedChunk]
    elapsed_ms: float
    max_score: Optional[float]


class Retriever:
    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        top_k: int = TOP_K,
        min_score: float = MIN_SCORE,
    ):
        self.store = store or VectorStore()
        self.embedder = embedder or Embedder()
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        doc_id: str | None = None,
        question_embedding: Sequence[float] | None = None,
    ) -> RetrievalResult:
        k = top_k if top_k is not None else self.top_k
        threshold = min_score if min_score is not None else self.min_score

        with timer() as t:
            if question_embedding is None:
                emb = self.embedder.embed_question(question)
            else:
                emb = question_embedding
            all_hits = self.store.query(emb, top_k=k, doc_id=doc_id)
            filtered = [h for h in all_hits if h.score >= threshold]

        max_score = max((h.score for h in all_hits), default=None)
        return RetrievalResult(
            hits=filtered,
            all_hits=all_hits,
            elapsed_ms=t["ms"],
            max_score=max_score,
        )

    def retrieve_top1_from_arrays(
        self,
        chunks: Sequence[str],
        chunk_embeddings: np.ndarray,
        question_embedding: np.ndarray,
    ) -> str:
        """In-memory top-1 cosine (tests / offline helpers). No torch."""
        q = np.asarray(question_embedding, dtype=np.float32).reshape(1, -1)
        m = np.asarray(chunk_embeddings, dtype=np.float32)
        # assume already normalized; still safe to re-normalize lightly
        qn = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
        mn = m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)
        scores = (qn @ mn.T).ravel()
        return chunks[int(np.argmax(scores))]
