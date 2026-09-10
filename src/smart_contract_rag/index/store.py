"""Persistent vector store over an embedding backend.

ChromaDB is a lightweight, embeddable, locally-persistent vector database that
runs fine inside a container/process (no separate server), which suits the
portfolio's constraint of no external paid services. It stores chunk text plus
provenance metadata so retrieval can return citable source references.

We expose a :class:`VectorStore` protocol and provide a pure in-memory fake
(:class:`InMemoryVectorStore`) so retrieval/rerank/pipeline tests never need
ChromaDB installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ..models import Chunk, RetrievedChunk
from .embeddings import EmbeddingBackend


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit-normalised vectors.

    Both vectors are expected to be normalised (which the embedders guarantee),
    so the dot product equals the cosine. We normalise defensively anyway so a
    non-normalised embedder still yields a proper similarity in [-1, 1].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class VectorStore(Protocol):
    """Anything able to store chunks and query them by similarity."""

    def add_documents(self, chunks: Sequence[Chunk], embedder: EmbeddingBackend) -> None:
        """Add a batch of chunks (embedding each with ``embedder``)."""
        ...

    def query(self, query: str, embedder: EmbeddingBackend, top_k: int) -> list[RetrievedChunk]:
        """Return the top-k chunks most similar to ``query``."""
        ...

    def count(self) -> int:
        """Number of stored chunks."""
        ...


class ChromaVectorStore:
    """Persistent vector store backed by ChromaDB."""

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "smart_contract_audits",
    ) -> None:
        import chromadb  # heavy; imported lazily so tests can run without it

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, chunks: Sequence[Chunk], embedder: EmbeddingBackend) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = embedder.embed(texts)
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=[self._chunk_metadata(c) for c in chunks],
        )

    def query(self, query: str, embedder: EmbeddingBackend, top_k: int) -> list[RetrievedChunk]:
        query_vector = embedder.embed([query])[0]
        if self._collection.count() == 0 or top_k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_vector.tolist()], n_results=min(top_k, self._collection.count())
        )
        items: list[RetrievedChunk] = []
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        for cid, distance, document, meta in zip(ids, distances, documents, metadatas):
            items.append(
                RetrievedChunk(
                    chunk=Chunk(
                        text=document,
                        source=_source_from_metadata(meta),
                        index=int(meta.get("index", 0)),
                        metadata=dict(meta),
                    ),
                    score=1.0 - float(distance),  # cosine distance -> similarity
                )
            )
        return items

    def count(self) -> int:
        return self._collection.count()

    def _chunk_metadata(self, chunk: Chunk) -> dict[str, str | int]:
        meta: dict[str, str | int] = {
            "index": chunk.index,
            "doc_id": chunk.source.doc_id,
        }
        if chunk.source.page is not None:
            meta["page"] = chunk.source.page
        if chunk.source.section is not None:
            meta["section"] = chunk.source.section
        return meta


def _source_from_metadata(meta: dict) -> object:
    from ..models import SourceRef  # local import to avoid a cycle at module load

    return SourceRef(
        doc_id=str(meta.get("doc_id", "unknown")),
        page=int(meta["page"]) if meta.get("page") is not None else None,
        section=meta.get("section"),
    )


class InMemoryVectorStore:
    """Deterministic in-memory vector store for tests.

    Stores chunks and their vectors locally and performs brute-force cosine
    search. It provides the same behaviour as :class:`ChromaVectorStore` but
    needs no on-disk state and no external package, keeping unit tests fast,
    hermetic and reproducible.
    """

    def __init__(self, embedder: EmbeddingBackend | None = None) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[np.ndarray] = []
        self._embedder = embedder

    def add_documents(self, chunks: Sequence[Chunk], embedder: EmbeddingBackend) -> None:
        if not chunks:
            return
        vectors = embedder.embed([c.text for c in chunks])
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)

    def query(self, query: str, embedder: EmbeddingBackend, top_k: int) -> list[RetrievedChunk]:
        if not self._chunks or top_k <= 0:
            return []
        query_vec = embedder.embed([query])[0]
        scored: list[tuple[float, Chunk]] = []
        for chunk, vec in zip(self._chunks, self._vectors):
            scored.append((cosine_similarity(query_vec, vec), chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [RetrievedChunk(chunk=c, score=s) for s, c in scored[:top_k]]

    def count(self) -> int:
        return len(self._chunks)
