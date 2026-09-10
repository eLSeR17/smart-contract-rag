"""Tests for the vector retriever."""

from __future__ import annotations

from smart_contract_rag.index.embeddings import DeterministicFakeEmbeddingBackend
from smart_contract_rag.index.store import InMemoryVectorStore
from smart_contract_rag.models import Chunk, SourceRef
from smart_contract_rag.retrieval.retriever import VectorRetriever


class TestVectorRetriever:
    def _setup(self, n: int = 4, top_k: int = 2) -> VectorRetriever:
        store = InMemoryVectorStore()
        embedder = DeterministicFakeEmbeddingBackend(dimension=16)
        chunks = [
            Chunk(text=f"about reentrancy guard in contract {i}",
                  source=SourceRef(doc_id=f"doc{i}", page=i + 1), index=i)
            for i in range(n)
        ]
        store.add_documents(chunks, embedder)
        return VectorRetriever(store=store, embedder=embedder, top_k_default=top_k)

    def test_returns_top_k(self) -> None:
        retriever = self._setup(top_k=2)
        result = retriever.retrieve("reentrancy guard")
        assert len(result) == 2

    def test_top_k_override(self) -> None:
        retriever = self._setup(top_k=2)
        result = retriever.retrieve("guard", top_k=4)
        assert len(result) == 4

    def test_default_top_k_used(self) -> None:
        retriever = self._setup(top_k=1)
        assert len(retriever.retrieve("guard")) == 1

    def test_empty_corpus(self) -> None:
        store = InMemoryVectorStore()
        embedder = DeterministicFakeEmbeddingBackend()
        retriever = VectorRetriever(store=store, embedder=embedder, top_k_default=3)
        assert retriever.retrieve("anything") == []

    def test_retrieve_many(self) -> None:
        retriever = self._setup(top_k=1)
        grouped = retriever.retrieve_many(["reentrancy", "contract"], top_k=1)
        assert len(grouped) == 2
        assert all(len(g) == 1 for g in grouped)

    def test_preserves_source_in_retrieved(self) -> None:
        retriever = self._setup(top_k=1)
        result = retriever.retrieve("reentrancy guard in contract 0")
        assert result[0].chunk.source.doc_id in {f"doc{i}" for i in range(4)}
