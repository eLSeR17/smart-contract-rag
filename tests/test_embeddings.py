"""Tests for the embedding backends and the deterministic fake."""

from __future__ import annotations

import numpy as np

from smart_contract_rag.index.embeddings import (
    CachedEmbeddingBackend,
    DeterministicFakeEmbeddingBackend,
)


class TestFakeBackend:
    def test_dimension(self) -> None:
        fake = DeterministicFakeEmbeddingBackend(dimension=16)
        assert fake.dimension == 16

    def test_deterministic_same_text(self) -> None:
        fake = DeterministicFakeEmbeddingBackend()
        a = fake.embed(["hello world"])
        b = fake.embed(["hello world"])
        assert np.array_equal(a, b)
        assert a.shape == (1, 32)

    def test_units_norm(self) -> None:
        fake = DeterministicFakeEmbeddingBackend()
        vec = fake.embed(["something"])[0]
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-6

    def test_empty_input(self) -> None:
        fake = DeterministicFakeEmbeddingBackend()
        out = fake.embed([])
        assert out.shape == (0, 32)


class TestCachingBackend:
    def test_cache_hits_tracked(self) -> None:
        fake = DeterministicFakeEmbeddingBackend()
        cached = CachedEmbeddingBackend(fake)
        cached.embed(["alpha", "beta"])
        cached.embed(["alpha"])
        assert cached.hits >= 1
        assert cached.misses == 2

    def test_cached_result_identical(self) -> None:
        fake = DeterministicFakeEmbeddingBackend()
        cached = CachedEmbeddingBackend(fake)
        first = cached.embed(["alpha"])
        second = cached.embed(["alpha"])
        assert np.array_equal(first, second)

    def test_dimension_delegates(self) -> None:
        fake = DeterministicFakeEmbeddingBackend(dimension=8)
        cached = CachedEmbeddingBackend(fake)
        assert cached.dimension == 8
