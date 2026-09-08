"""Tests for the vector store and cosine similarity."""

from __future__ import annotations

import numpy as np
import pytest

from smart_contract_rag.index.embeddings import DeterministicFakeEmbeddingBackend
from smart_contract_rag.index.store import InMemoryVectorStore, cosine_similarity
from smart_contract_rag.models import Chunk, SourceRef


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(text=f"chunk number {i} about audits", source=SourceRef(doc_id="doc", page=i + 1), index=i)
        for i in range(n)
    ]


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)

    def test_zero_vector(self) -> None:
        assert cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0


class TestInMemoryStore:
    def test_count(self) -> None:
        store = InMemoryVectorStore()
        embedder = DeterministicFakeEmbeddingBackend()
        store.add_documents(_chunks(5), embedder)
        assert store.count() == 5

    def test_query_returns_requested_top_k(self) -> None:
        store = InMemoryVectorStore()
        embedder = DeterministicFakeEmbeddingBackend()
        store.add_documents(_chunks(10), embedder)
        result = store.query("audits", embedder, top_k=3)
        assert len(result) == 3
        # Sorted best-first.
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_metadata_preserved(self) -> None:
        store = InMemoryVectorStore()
        embedder = DeterministicFakeEmbeddingBackend()
        chunks = _chunks(2)
        store.add_documents(chunks, embedder)
        result = store.query("chunk number 0 about audits", embedder, top_k=1)
        assert result[0].chunk.source.page == 1
        assert result[0].chunk.source.doc_id == "doc"

    def test_query_empty(self) -> None:
        store = InMemoryVectorStore()
        embedder = DeterministicFakeEmbeddingBackend()
        assert store.query("x", embedder, top_k=3) == []

    def test_add_documents_empty(self) -> None:
        store = InMemoryVectorStore()
        embedder = DeterministicFakeEmbeddingBackend()
        store.add_documents([], embedder)
        assert store.count() == 0
