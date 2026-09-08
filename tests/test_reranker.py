"""Tests for hybrid rerankers."""

from __future__ import annotations

import pytest

from smart_contract_rag.models import Chunk, RetrievedChunk, SourceRef
from smart_contract_rag.retrieval.reranker import RRF_Reranker, ScoreFusionReranker


def _candidate(text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(text=text, source=SourceRef(doc_id="d1"), index=0),
        score=score,
    )


CANDIDATES = [
    _candidate("the token transfer reentrancy issue affects withdrawals", 0.9),
    _candidate("a completely unrelated chat about weather and food", 0.7),
    _candidate("reentrancy token guard in contract", 0.6),
]


class TestScoreFusionReranker:
    def test_lexical_match_boosted(self) -> None:
        reranker = ScoreFusionReranker(alpha=0.5)
        # Query has exact match "reentrancy token" in third candidate.
        ordered = reranker.rerank("reentrancy token guard", CANDIDATES, top_k=3)
        assert ordered[0].chunk.text.startswith("reentrancy token guard")

    def test_top_k_limits(self) -> None:
        reranker = ScoreFusionReranker()
        ordered = reranker.rerank("reentrancy", CANDIDATES, top_k=2)
        assert len(ordered) == 2

    def test_empty_candidates(self) -> None:
        reranker = ScoreFusionReranker()
        assert reranker.rerank("x", [], top_k=3) == []

    def test_invalid_alpha(self) -> None:
        with pytest.raises(ValueError):
            ScoreFusionReranker(alpha=1.5)

    def test_sets_rerank_score(self) -> None:
        reranker = ScoreFusionReranker()
        ordered = reranker.rerank("reentrancy", CANDIDATES, top_k=3)
        assert all(r.rerank_score is not None for r in ordered)


class TestRRFReranker:
    def test_returns_ordered_results(self) -> None:
        rr = RRF_Reranker()
        ordered = rr.rerank("reentrancy token guard", CANDIDATES, top_k=3)
        assert len(ordered) == 3
        scores = [float(r.rerank_score) for r in ordered]
        assert scores == sorted(scores, reverse=True)

    def test_top_k(self) -> None:
        rr = RRF_Reranker()
        assert len(rr.rerank("x", CANDIDATES, top_k=2)) == 2

    def test_empty(self) -> None:
        rr = RRF_Reranker()
        assert rr.rerank("x", [], top_k=2) == []

    def test_lexical_matches_rank_higher(self) -> None:
        rr = RRF_Reranker()
        # Three candidates where the pure-vector winner (score 0.95) has NO
        # lexical overlap, while an exact-term match has both decent vector rank
        # and the best lexical rank. RRF should surface the one that is strong
        # in *both* orderings above the vector-only winner.
        no_overlap = _candidate("unrelated discussion of stablecoin reserves", 0.95)
        exact_match = _candidate("mev sandwich front-run attacks on AMM liquidity pools", 0.60)
        some_overlap = _candidate("reentrancy protection in the front-run path", 0.50)
        candidates = [no_overlap, exact_match, some_overlap]
        ordered = rr.rerank("mev sandwich front-run", candidates, top_k=3)
        assert ordered[0].chunk.text.startswith("mev sandwich front-run")
