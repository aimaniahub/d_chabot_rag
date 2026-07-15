"""Persistent Chroma vector store."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR, COLLECTION_NAME, ensure_dirs
from modules.chunker import Chunk


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    doc_id: str
    source: str
    page_start: int
    page_end: int
    chunk_index: int
    metadata: dict[str, Any]


class VectorStore:
    def __init__(self, collection_name: str = COLLECTION_NAME, persist_dir: str | None = None):
        ensure_dirs()
        path = str(persist_dir or CHROMA_DIR)
        self.client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return int(self.collection.count())

    def delete_doc(self, doc_id: str) -> None:
        # Chroma delete by where filter
        try:
            self.collection.delete(where={"doc_id": doc_id})
        except Exception:
            # Empty collection / no matches
            pass

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.to_metadata() for c in chunks]
        emb_list = [list(map(float, e)) for e in embeddings]

        # Batch to avoid oversized payloads
        batch = 100
        for i in range(0, len(ids), batch):
            self.collection.upsert(
                ids=ids[i : i + batch],
                documents=documents[i : i + batch],
                metadatas=metadatas[i : i + batch],
                embeddings=emb_list[i : i + batch],
            )
        return len(ids)

    def query(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        doc_id: str | None = None,
    ) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []

        kwargs: dict[str, Any] = {
            "query_embeddings": [list(map(float, query_embedding))],
            "n_results": min(top_k, max(self.count(), 1)),
            "include": ["documents", "metadatas", "distances"],
        }
        if doc_id:
            kwargs["where"] = {"doc_id": doc_id}

        result = self.collection.query(**kwargs)
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        hits: list[RetrievedChunk] = []
        for i, doc in enumerate(docs):
            # cosine distance -> similarity score
            dist = float(dists[i]) if i < len(dists) else 1.0
            score = 1.0 - dist
            meta = metas[i] if i < len(metas) else {}
            hits.append(
                RetrievedChunk(
                    chunk_id=ids[i] if i < len(ids) else meta.get("chunk_id", ""),
                    text=doc or "",
                    score=score,
                    doc_id=str(meta.get("doc_id", "")),
                    source=str(meta.get("source", "")),
                    page_start=int(meta.get("page_start", 1)),
                    page_end=int(meta.get("page_end", 1)),
                    chunk_index=int(meta.get("chunk_index", i)),
                    metadata=dict(meta) if meta else {},
                )
            )
        return hits

    def reset(self) -> None:
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
