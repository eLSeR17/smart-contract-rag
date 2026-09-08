"""Retrieval query engine.

The retriever combines the embedding backend and the vector store: it embeds
the query and delegates to the store for top-k search. Keeping this as a thin
wrapper means the store can be swapped (ChromaDB in prod, in-memory in tests)
without changing calling code.
"""

from __future__ import annotations

from typing import Sequence

from ..index.embeddings import EmbeddingBackend
from ..index.store import VectorStore
from ..models import RetrievedChunk


class VectorRetriever:
    """Retrieves the top-k chunks most similar to a query."""

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingBackend,
        *,
        top_k_default: int = 5,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k_default = top_k_default

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the top-k chunks for ``query`` ordered by similarity."""
        k = top_k if top_k is not None else self.top_k_default
        return self.store.query(query, self.embedder, k)

    def retrieve_many(
        self, queries: Sequence[str], *, top_k: int | None = None
    ) -> list[list[RetrievedChunk]]:
        """Retrieve for multiple queries (used by eval/hybrid search)."""
        return [self.retrieve(q, top_k=top_k) for q in queries]
