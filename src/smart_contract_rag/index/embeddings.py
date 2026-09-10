"""Embedding backends behind a common protocol.

Sentence-transformers' ``all-MiniLM-L6-v2`` is a compact, official, widely-deployed
embedder (~22 MB) that produces deterministic 384-dim vectors. We hide it behind
:class:`EmbeddingBackend` so that:
    * real runs use the model inside the docker network,
    * unit tests inject :class:`DeterministicFakeEmbeddingBackend` — no model
      download, no network, fully reproducible.

A small exact-match cache saves recomputation when a chunk is re-embedded.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    """Anything able to turn a sequence of texts into an array of vectors."""

    @property
    def dimension(self) -> int:
        """Length of a single embedding vector."""
        ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float array of shape ``(len(texts), dimension)``."""
        ...


class SentenceTransformerBackend:
    """Real embedder backed by a sentence-transformers model.

    The model is loaded lazily (on first use) so constructing the object costs
    nothing until embeddings are actually needed — important for fast test
    suites that never reach the real backend.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    @property
    def dimension(self) -> int:
        self._ensure_model()
        model = self._model
        assert model is not None
        return int(model.get_sentence_embedding_dimension())

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy; lazy

            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        self._ensure_model()
        model = self._model
        assert model is not None
        vectors = model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


class CachedEmbeddingBackend:
    """Wraps another backend with an exact-match text -> vector cache.

    Caching avoids re-running the model for chunks that never change between
    indexing runs (e.g. when re-indexing a stable corpus). The cache is an
    in-memory dict; it is bounded by the number of unique texts seen, which in
    a corpus pipeline is bounded and acceptable.
    """

    def __init__(self, backend: EmbeddingBackend) -> None:
        self._backend = backend
        self._cache: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    @property
    def dimension(self) -> int:
        return self._backend.dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        uncached: list[str] = []
        positions: list[int] = []
        for i, text in enumerate(texts):
            if text in self._cache:
                self.hits += 1
            else:
                uncached.append(text)
                positions.append(i)

        vectors = np.zeros((len(texts), self._backend.dimension), dtype=np.float32)
        if uncached:
            self.misses += len(uncached)
            new_vectors = self._backend.embed(uncached)
            for pos, text, vec in zip(positions, uncached, new_vectors):
                self._cache[text] = vec
                vectors[pos] = vec
        # Fill cached slots.
        for i, text in enumerate(texts):
            if text in self._cache and vectors[i].sum() == 0.0:
                vectors[i] = self._cache[text]
        return vectors


class DeterministicFakeEmbeddingBackend:
    """Deterministic fake for tests: a hash-based 32-dim vector.

    Identical text always yields an identical vector, so cosine-similarity
    retrieval is deterministic and reproducible across test runs. ``dimension``
    is small (32) to keep tests fast. The vectors are *not* semantically
    meaningful — they exist only to exercise the retrieval/rerank logic.
    """

    def __init__(self, dimension: int = 32) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        rows: list[np.ndarray] = []
        for text in texts:
            # Stable 32-bit hash seeded by the text -> deterministic pseudo-vector.
            rng = np.random.default_rng(abs(hash(text)) % (2**31))
            vec = rng.standard_normal(self._dimension)
            rows.append(vec / np.linalg.norm(vec))
        return np.asarray(rows, dtype=np.float32)
