"""ONNX embeddings via fastembed (no torch / torchvision)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from fastembed import TextEmbedding

from config import EMBEDDING_MODEL

# Map short aliases → fastembed model ids
_MODEL_ALIASES = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "miniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}


def resolve_model_name(name: str) -> str:
    return _MODEL_ALIASES.get(name, name)


class Embedder:
    """Local embedding model using ONNX Runtime (Docker-friendly, no PyTorch)."""

    def __init__(self, model_name: str | None = None):
        raw = model_name or EMBEDDING_MODEL
        self.model_name = resolve_model_name(raw)
        self.model = TextEmbedding(model_name=self.model_name)
        self.dimension: int | None = None  # set on first embed

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            dim = self.dimension or 384
            return np.zeros((0, dim), dtype=np.float32)

        vectors = list(self.model.embed(list(texts), batch_size=batch_size))
        arr = np.asarray(vectors, dtype=np.float32)
        # L2 normalize for cosine similarity in Chroma
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        arr = arr / norms
        if self.dimension is None and arr.ndim == 2:
            self.dimension = int(arr.shape[1])
        return arr

    def embed_chunks(self, chunks: Sequence[str]) -> np.ndarray:
        return self.embed_texts(chunks)

    def embed_question(self, question: str) -> np.ndarray:
        return self.embed_texts([question])[0]
