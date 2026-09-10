"""Regression-guard tests: the eval must detect a simulated degradation.

These tests use a healthy pipeline as the baseline and then *break* it in
controlled ways (hallucinating, refusing everything, bad citations, degraded
retrieval) to prove the guard flips to FAIL (or WARN) exactly when it should.
"""

from __future__ import annotations

import pytest

from smart_contract_rag.evals.dataset import GoldenDataset
from smart_contract_rag.evals.runner import (
    RegressionStatus,
    RegressionThresholds,
    run_eval,
)
from tests.evals_fixtures import ScriptedPipeline

BASELINE_CASES = [
    {
        "id": "ev-001",
        "question": "What is a reentrancy attack?",
        "topic": "reentrancy",
        "expected_keywords": ["withdraw", "reentrancy", "checks"],
        "relevant_doc_id": "aave-v3",
        "expect_answer": True,
    },
    {
        "id": "ev-002",
        "question": "How does oracle manipulation work in Balancer?",
        "topic": "oracle_manipulation",
        "expected_keywords": ["oracle", "price", "manipulation", "flash"],
        "relevant_doc_id": "balancer-managedpool",
        "expect_answer": True,
    },
    {
        "id": "ev-003",
        "question": "Does the 2024 0x ZK audit exist in this corpus?",
        "topic": "trap",
        "expected_keywords": [],
        "expect_answer": False,
    },
]

HEALTHY_SCRIPT = {
    "default": {"answer": "I DON'T KNOW", "doc_ids": [], "refuse": True},
    "reentrancy": {
        "answer": "The withdraw function is vulnerable to reentrancy via checks-effects-interactions.",
        "doc_ids": ["aave-v3"],
        "grounded_chunks": [
            (
                "The withdraw function is vulnerable to reentrancy and uses "
                "checks-effects-interactions."
            )
        ],
    },
    "oracle": {
        "answer": "An attacker manipulates the oracle price with a flash loan before trading.",
        "doc_ids": ["balancer-managedpool"],
        "grounded_chunks": [
            "An attacker manipulates the oracle price with a flash loan before trading."
        ],
    },
}


def _dataset() -> GoldenDataset:
    return GoldenDataset.from_cases(BASELINE_CASES)


def _default_thresholds() -> RegressionThresholds:
    return RegressionThresholds()


class TestBaselineGreen:
    def test_healthy_pipeline_passes(self) -> None:
        report = run_eval(ScriptedPipeline(HEALTHY_SCRIPT), _dataset())
        assert report.status == RegressionStatus.PASS
        assert report.metrics is not None
        assert report.metrics.hallucination_rate.value == pytest.approx(0.0)
        assert report.metrics.answer_rate.value == pytest.approx(1.0)


class TestDegradations:
    def test_hallucinating_pipeline_fails(self) -> None:
        # The oracle question is answered with fabricated, ungrounded text.
        script = dict(HEALTHY_SCRIPT)
        script["oracle"] = {
            "answer": "LumenPay tokens are printed from nothing on Solana.",
            "doc_ids": [],
            "grounded_chunks": [],
        }
        report = run_eval(ScriptedPipeline(script), _dataset())
        assert report.status == RegressionStatus.FAIL
        assert report.metrics is not None
        assert report.metrics.hallucination_rate.value > 0.0

    def test_refusing_everything_fails(self) -> None:
        script = {
            "default": {"answer": "I DON'T KNOW", "doc_ids": [], "refuse": True}
        }
        report = run_eval(ScriptedPipeline(script), _dataset())
        assert report.status == RegressionStatus.FAIL
        # Refusing answerable questions triggers the answer-rate floor.
        assert report.metrics is not None
        assert report.metrics.answer_rate.value == pytest.approx(0.0)

    def test_wrong_source_citation_fails(self) -> None:
        # The reentrancy answer cites the wrong document.
        script = dict(HEALTHY_SCRIPT)
        script["reentrancy"] = {
            "answer": "The withdraw function is vulnerable to reentrancy via checks-effects-interactions.",
            "doc_ids": ["optimism-security"],
            "grounded_chunks": [
                (
                    "The withdraw function is vulnerable to reentrancy and uses "
                    "checks-effects-interactions."
                )
            ],
        }
        report = run_eval(ScriptedPipeline(script), _dataset())
        assert report.status == RegressionStatus.FAIL
        assert report.metrics is not None
        # One of two scored cases cites the wrong doc -> 0.5 < 0.6 floor.
        assert report.metrics.citation_accuracy.value == pytest.approx(0.5)

    def test_degraded_retrieval_fails_recall_floor(self) -> None:
        # Retrieval returns a chunk from the wrong document for reentrancy.
        script = dict(HEALTHY_SCRIPT)
        script["reentrancy"] = {
            "answer": "The withdraw function is vulnerable to reentrancy via checks-effects-interactions.",
            "doc_ids": ["0x-protocol"],
            "grounded_chunks": [
                (
                    "The withdraw function is vulnerable to reentrancy and uses "
                    "checks-effects-interactions."
                )
            ],
        }
        report = run_eval(ScriptedPipeline(script), _dataset())
        assert report.status == RegressionStatus.FAIL
        assert report.metrics is not None
        assert report.metrics.context_recall.value == pytest.approx(0.5)


class TestWarnBand:
    def test_approaching_threshold_warns(self) -> None:
        # A pipeline whose answers include extra ungrounded terms so that the
        # faithfulness ratio lands *inside* the warn band [guard, guard+margin).
        # Each answer has 7 substantive tokens; only 4 appear in the grounded
        # chunk → 4/7 ≈ 0.571.  With min_faithfulness=0.55 and
        # warn_margin=0.05 the band is [0.55, 0.60) → WARN.
        script = {
            "default": {"answer": "I DON'T KNOW", "doc_ids": [], "refuse": True},
            "reentrancy": {
                "answer": "reentrancy withdraw checks balance borrow oracle flash",
                "doc_ids": ["aave-v3"],
                "grounded_chunks": ["reentrancy withdraw checks oracle"],
            },
            "oracle": {
                "answer": "oracle price manipulation flash balance borrow reentrancy",
                "doc_ids": ["balancer-managedpool"],
                "grounded_chunks": ["oracle price manipulation flash"],
            },
        }
        thresholds = RegressionThresholds(
            min_faithfulness=0.55, min_answer_relevance=0.0, warn_margin=0.05
        )
        report = run_eval(ScriptedPipeline(script), _dataset(), thresholds=thresholds)
        assert report.status == RegressionStatus.WARN

    def test_band_edge_passes_when_above(self) -> None:
        thresholds = RegressionThresholds(
            min_faithfulness=0.5, min_answer_relevance=0.0, warn_margin=0.05
        )
        report = run_eval(ScriptedPipeline(HEALTHY_SCRIPT), _dataset(), thresholds=thresholds)
        assert report.status == RegressionStatus.PASS


class TestThresholdConfiguration:
    def test_threshold_dict_matches_fields(self) -> None:
        thr = RegressionThresholds()
        d = thr.as_dict()
        assert set(d) == {
            "max_hallucination_rate",
            "min_faithfulness",
            "min_answer_relevance",
            "min_context_recall",
            "min_citation_accuracy",
            "min_answer_rate",
            "warn_margin",
        }

    def test_lenient_thresholds_turn_fail_into_pass(self) -> None:
        # Same broken pipeline, lenient thresholds -> PASS instead of FAIL.
        script = {
            "default": {"answer": "I DON'T KNOW", "doc_ids": [], "refuse": True},
            "reentrancy": {
                "answer": "The space is cold and the moon is made of cheese 42.7.",
                "doc_ids": [],
                "grounded_chunks": [],
            },
        }
        lenient = RegressionThresholds(
            max_hallucination_rate=1.0,
            min_faithfulness=0.0,
            min_answer_relevance=0.0,
            min_context_recall=0.0,
            min_citation_accuracy=0.0,
            min_answer_rate=0.0,
        )
        strict = _default_thresholds()
        lenient_report = run_eval(
            ScriptedPipeline(script), _dataset(), thresholds=lenient
        )
        strict_report = run_eval(ScriptedPipeline(script), _dataset(), thresholds=strict)
        assert lenient_report.status == RegressionStatus.PASS
        assert strict_report.status == RegressionStatus.FAIL


class TestStatusEnum:
    def test_enum_values(self) -> None:
        assert RegressionStatus.PASS.value == "PASS"
        assert RegressionStatus.WARN.value == "WARN"
        assert RegressionStatus.FAIL.value == "FAIL"