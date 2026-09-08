"""Reranking of candidate chunks for better precision.

Vector-only retrieval can surface lexically-distinct but semantically-adjacent
chunks, and it does not exploit term overlap — a sign a specific keyword
matters (e.g. the exact function/variable a query asks about). A light hybrid
rerank mitigates both:

    * :class:`ScoreFusionReranker` blends the vector similarity with a
      keyword-overlap (BM25-inspired) score, so a chunk that contains the
      query's exact terms ranks higher than one that merely ``feels`` right.

    * :class:`RRF_Reranker` merges two ranked orders (vector + lexical) via
      Reciprocal Rank Fusion, which is robust to differently-scaled score
      distributions.

Both are deterministic and dependency-free, hence easy to unit test and safe
to run in CI.
"""

from __future__ import annotations

import re
from typing import Protocol

from ..models import RetrievedChunk


class Reranker(Protocol):
    """Anything able to reorder a list of candidate chunks for a query."""

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Return ``top_k`` reranked chunks (best first)."""
        ...


def _tokens(text: str) -> set[str]:
    """Lowercased word tokens for keyword overlap scoring."""
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


class ScoreFusionReranker:
    """Blend vector similarity with a lightweight lexical (keyword) score.

    The two scores are on very different scales (cosine ~[0,1], keyword overlap
    ~count of shared terms), so we normalise the keyword count by its maximum
    over the candidate set before fusing. ``alpha`` weights vector vs lexical:
    0.7/0.3 is a sensible default that mostly trusts semantic similarity while
    letting an exact-term chunk rise.
    """

    def __init__(self, *, alpha: float = 0.7) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        query_tokens = _tokens(query)
        lexical_scores = [len(query_tokens & _tokens(c.chunk.text)) for c in candidates]
        max_lex = max(lexical_scores) if lexical_scores else 0
        if max_lex == 0:
            max_lex = 1  # avoid division by zero; all lexical scores are 0

        for candidate, lex in zip(candidates, lexical_scores):
            lex_norm = lex / max_lex
            # Renormalise vector similarity to [0,1]; cosine can be negative.
            vec_norm = max(0.0, candidate.score)
            candidate.rerank_score = self.alpha * vec_norm + (1.0 - self.alpha) * lex_norm

        ordered = sorted(candidates, key=lambda c: float(c.rerank_score), reverse=True)
        return ordered[:top_k]


class RRF_Reranker:
    """Reciprocal Rank Fusion of a vector ranking and a lexical ranking.

    RRF is chosen because it needs no score calibration: it works purely from
    *rank positions*. We compute the lexical ranking by keyword overlap and fuse
    with the scores' original rank order. ``k=60`` is the standard RRF constant.
    """

    def __init__(self, *, k: int = 60) -> None:
        self.k = k

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        query_tokens = _tokens(query)

        # 1) Original order == vector-score ranking (list is passed best-first).
        vector_ranks = {id(c): i + 1 for i, c in enumerate(candidates)}

        # 2) Lexical ranking by keyword overlap.
        lex_ranked = sorted(
            candidates,
            key=lambda c: len(query_tokens & _tokens(c.chunk.text)),
            reverse=True,
        )
        lex_ranks = {id(c): i + 1 for i, c in enumerate(lex_ranked)}

        for candidate in candidates:
            rrf = 0.0
            for rank in (vector_ranks[id(candidate)], lex_ranks[id(candidate)]):
                rrf += 1.0 / (self.k + rank)
            candidate.rerank_score = rrf

        ordered = sorted(candidates, key=lambda c: float(c.rerank_score), reverse=True)
        return ordered[:top_k]
