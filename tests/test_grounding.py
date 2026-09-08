"""Tests for the anti-hallucination grounding checker."""

from __future__ import annotations

from smart_contract_rag.generation.grounding import NaiveGroundedTextCheck
from smart_contract_rag.models import Chunk, RetrievedChunk, SourceRef


def _retrieved(*texts: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(text=t, source=SourceRef(doc_id="doc1")),
            score=0.9,
        )
        for t in texts
    ]


CORPUS = _retrieved(
    "The withdraw function is vulnerable to reentrancy and must use checks-effects-interactions.",
    "Aave v3 uses a liquidity index of 1.05.",
)


class TestNaiveGroundedTextCheck:
    def test_grounded_response_accepted(self) -> None:
        checker = NaiveGroundedTextCheck()
        ok, ratios = checker.check(
            "The withdraw function is vulnerable to reentrancy.",
            CORPUS,
            threshold=0.5,
        )
        assert ok is True
        assert ratios["grounded_ratio"] >= 0.5

    def test_hallucinated_response_rejected(self) -> None:
        checker = NaiveGroundedTextCheck()
        # Answer talks about an unrelated crypto exchange that is not in corpus.
        ok, ratios = checker.check(
            "Binance listed a new token called XYZ next week.",
            CORPUS,
            threshold=0.5,
        )
        assert ok is False
        assert ratios["grounded_ratio"] < 0.5

    def test_empty_answer_is_grounded(self) -> None:
        checker = NaiveGroundedTextCheck()
        ok, _ = checker.check("", CORPUS, threshold=0.5)
        assert ok is True

    def test_number_grounding(self) -> None:
        checker = NaiveGroundedTextCheck()
        # The number 1.05 is grounded; a fabricated number is not.
        ok, ratios = checker.check("The liquidity index is 1.05.", CORPUS, threshold=0.5)
        assert ok is True
        assert ratios["grounded_tokens"] >= 1

        ok_false, _ = checker.check("The fee rate is 42.7 percent now.", CORPUS, threshold=0.8)
        # 42.7 is not in the corpus, pushing ratio down.
        assert ok_false is False

    def test_threshold_controls_strictness(self) -> None:
        checker = NaiveGroundedTextCheck()
        answer = "The withdraw function is vulnerable to reentrancy and exploits are likely."
        # Full grounding yields high ratio; a strict threshold passes.
        ok_strict, ratios = checker.check(answer, CORPUS, threshold=0.5)
        assert ok_strict is True
        # An impossible threshold should reject even a grounded answer.
        ok_impossible, _ = checker.check(answer, CORPUS, threshold=0.99)
        assert ok_impossible is False

    def test_no_retrieved_evidence_flagged(self) -> None:
        checker = NaiveGroundedTextCheck()
        ok, ratios = checker.check("Anything at all here.", [], threshold=0.5)
        assert ok is False
        assert ratios["grounded_ratio"] == 0.0
