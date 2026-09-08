"""Tests for the eval runner: report generation, aggregation, judges, verdicts."""

from __future__ import annotations

import pytest

from smart_contract_rag.evals.dataset import GoldenDataset
from smart_contract_rag.evals.judge import JudgeScore
from smart_contract_rag.evals.runner import (
    RegressionStatus,
    RegressionThresholds,
    run_eval,
)
from smart_contract_rag.models import RetrievedChunk

from tests.evals_fixtures import ScriptedPipeline


# ---------------------------------------------------------------------------
# Golden datasets (valid by schema)
# ---------------------------------------------------------------------------
ANSWERABLE_CASE = {
    "id": "ev-001",
    "question": "What is a reentrancy attack and why is it dangerous?",
    "topic": "reentrancy",
    "expected_keywords": ["withdraw", "reentrancy", "checks", "effects", "interactions"],
    "relevant_doc_id": "aave-v3",
    "ground_truth": "An attack re-enters a function through an external call.",
    "expect_answer": True,
}

TRAP_CASE = {
    "id": "ev-999",
    "question": "Does the corpus discuss Rust gas costs in a Solana protocol?",
    "topic": "trap",
    "expected_keywords": [],
    "expect_answer": False,
}

GROUNDED_ANSWER = (
    "The withdraw function is vulnerable to reentrancy and uses the "
    "checks-effects-interactions pattern."
)
GROUNDED_CHUNK = (
    "The withdraw function is vulnerable to reentrancy and uses the "
    "checks-effects-interactions pattern."
)


def _perfect_pipeline() -> ScriptedPipeline:
    return ScriptedPipeline(
        {
            "default": {"answer": "I DON'T KNOW", "doc_ids": [], "refuse": True},
            "reentrancy": {
                "answer": GROUNDED_ANSWER,
                "doc_ids": ["aave-v3"],
                "grounded_chunks": [GROUNDED_CHUNK],
                "refuse": False,
            },
        }
    )


def _dataset(*cases: dict) -> GoldenDataset:
    return GoldenDataset.from_cases(list(cases))


class TestRunEvalPerfect:
    def test_pass_with_perfect_pipeline(self) -> None:
        report = run_eval(
            _perfect_pipeline(), _dataset(ANSWERABLE_CASE, TRAP_CASE)
        )
        assert report.status == RegressionStatus.PASS
        assert report.metrics is not None
        assert report.metrics.faithfulness.value == pytest.approx(1.0)
        assert report.metrics.answer_relevance.value == pytest.approx(1.0)
        assert report.metrics.citation_accuracy.value == pytest.approx(1.0)
        assert report.metrics.context_precision.value == pytest.approx(1.0)
        assert report.metrics.context_recall.value == pytest.approx(1.0)
        assert report.metrics.answer_rate.value == pytest.approx(1.0)
        assert report.metrics.correct_refusal_rate.value == pytest.approx(1.0)
        assert report.metrics.hallucination_rate.value == pytest.approx(0.0)

    def test_per_case_details(self) -> None:
        report = run_eval(_perfect_pipeline(), _dataset(ANSWERABLE_CASE))
        assert len(report.cases) == 1
        case = report.cases[0]
        assert case.case_id == "ev-001"
        assert case.answered is True
        assert case.refused is False
        assert case.faithfulness == pytest.approx(1.0)
        assert case.citation_accuracy == pytest.approx(1.0)
        assert case.context_recall == pytest.approx(1.0)
        assert case.hallucination is False
        assert case.is_trap is False

    def test_answer_metrics_vacuous_when_all_refused(self) -> None:
        # A trap-only dataset with a refusing pipeline must PASS (vacuous floors).
        pipeline = ScriptedPipeline(
            {"default": {"answer": "I DON'T KNOW", "doc_ids": [], "refuse": True}}
        )
        report = run_eval(pipeline, _dataset(TRAP_CASE))
        assert report.status == RegressionStatus.PASS
        assert report.metrics is not None
        assert report.metrics.answer_rate.value == pytest.approx(1.0)
        assert report.metrics.faithfulness.value == pytest.approx(1.0)
        assert report.metrics.hallucination_rate.value == pytest.approx(0.0)
        assert report.metrics.correct_refusal_rate.value == pytest.approx(1.0)

    def test_trap_answered_marks_hallucination_and_fails(self) -> None:
        # The trap is answered confidently instead of refused -> hallucination.
        pipeline = ScriptedPipeline(
            {"default": {"answer": "The corpus says LumenPay is safe.", "doc_ids": ["aave-v3"],
                         "grounded_chunks": ["The corpus says LumenPay is safe."]}}
        )
        report = run_eval(pipeline, _dataset(TRAP_CASE))
        assert report.cases[0].correct_refusal is False
        assert report.status == RegressionStatus.FAIL


class TestJudgeOverride:
    class FixedJudge:
        def __init__(self, faithfulness: float, relevance: float) -> None:
            self._faith = faithfulness
            self._rel = relevance

        def score(
            self, *, question: str, answer: str, expected: str | None, retrieved: list[RetrievedChunk]
        ) -> JudgeScore:
            return JudgeScore(
                faithfulness=self._faith, relevance=self._rel, reasoning="fixed judge"
            )

    def test_judge_scores_normalized_and_used(self) -> None:
        judge = self.FixedJudge(faithfulness=4.5, relevance=3.5)
        report = run_eval(
            _perfect_pipeline(), _dataset(ANSWERABLE_CASE), judge=judge
        )
        assert report.metrics is not None
        assert report.metrics.faithfulness.value == pytest.approx(0.9)
        assert report.metrics.answer_relevance.value == pytest.approx(0.7)
        assert report.cases[0].judge_reason == "fixed judge"

    def test_judge_can_be_disabled(self) -> None:
        judge = self.FixedJudge(faithfulness=1.0, relevance=1.0)
        report = run_eval(
            _perfect_pipeline(),
            _dataset(ANSWERABLE_CASE),
            judge=judge,
            use_judge_for_metrics=False,
        )
        assert report.metrics is not None
        assert report.metrics.faithfulness.value == pytest.approx(1.0)

    def test_failing_judge_falls_back_to_lexical(self) -> None:
        class BrokenJudge:
            def score(self, **kwargs) -> JudgeScore:
                return JudgeScore(
                    faithfulness=0.0, relevance=0.0, reasoning="", error="boom"
                )

        report = run_eval(
            _perfect_pipeline(), _dataset(ANSWERABLE_CASE), judge=BrokenJudge()
        )
        assert report.metrics is not None
        assert report.metrics.faithfulness.value == pytest.approx(1.0)  # lexical fallback


class TestHallucinationDetection:
    def test_ungrounded_answer_flagged(self) -> None:
        pipeline = ScriptedPipeline(
            {
                "default": {
                    "answer": "The moon is made of green cheese 42.7.",
                    "doc_ids": [],
                    "grounded_chunks": [],
                    "refuse": False,
                }
            }
        )
        report = run_eval(pipeline, _dataset(ANSWERABLE_CASE))
        assert report.cases[0].hallucination is True
        assert report.metrics is not None
        assert report.metrics.hallucination_rate.value == pytest.approx(1.0)
        assert report.status == RegressionStatus.FAIL

    def test_grounded_answer_not_flagged(self) -> None:
        report = run_eval(_perfect_pipeline(), _dataset(ANSWERABLE_CASE))
        assert report.metrics is not None
        assert report.metrics.hallucination_rate.value == pytest.approx(0.0)


class TestReportSurface:
    def test_to_dict_round_trip(self) -> None:
        report = run_eval(_perfect_pipeline(), _dataset(ANSWERABLE_CASE, TRAP_CASE))
        data = report.to_dict()
        assert data["status"] == "PASS"
        assert data["n_cases"] == 2
        assert data["n_answered"] == 1
        assert data["n_refused"] == 1
        assert data["thresholds"] == RegressionThresholds().as_dict()
        assert {c["case_id"] for c in data["cases"]} == {"ev-001", "ev-999"}
        assert data["metrics"]["faithfulness"]["value"] == pytest.approx(1.0)

    def test_to_markdown_contains_report(self) -> None:
        report = run_eval(_perfect_pipeline(), _dataset(ANSWERABLE_CASE, TRAP_CASE))
        md = report.to_markdown()
        assert "Evaluation Report" in md
        assert "Verdict" in md
        assert "faithfulness" in md
        assert "ev-001" in md

    def test_str_is_markdown(self) -> None:
        report = run_eval(_perfect_pipeline(), _dataset(ANSWERABLE_CASE))
        assert str(report) == report.to_markdown()


class TestThresholds:
    def test_custom_thresholds_used(self) -> None:
        thresholds = RegressionThresholds(min_faithfulness=0.99)
        report = run_eval(
            _perfect_pipeline(), _dataset(ANSWERABLE_CASE), thresholds=thresholds
        )
        # Faithfulness is exactly 1.0, which meets a 0.99 floor.
        assert report.status == RegressionStatus.PASS
        assert report.thresholds == thresholds

    def test_strict_threshold_fails(self) -> None:
        thresholds = RegressionThresholds(min_answer_rate=1.0, warn_margin=0.0)
        # One of two cases is a trap that refuses -> answer_rate = 1.0 among
        # answerable cases -> still passes; use a stricter definition: make the
        # pipeline refuse the answerable case.
        pipeline = ScriptedPipeline(
            {"default": {"answer": "I DON'T KNOW", "doc_ids": [], "refuse": True}}
        )
        report = run_eval(pipeline, _dataset(ANSWERABLE_CASE), thresholds=thresholds)
        assert report.status == RegressionStatus.FAIL

    def test_bad_citation_fails(self) -> None:
        pipeline = ScriptedPipeline(
            {
                "default": {
                    "answer": GROUNDED_ANSWER,
                    "doc_ids": ["balancer"],  # wrong doc cited
                    "grounded_chunks": [GROUNDED_CHUNK],
                }
            }
        )
        report = run_eval(pipeline, _dataset(ANSWERABLE_CASE))
        assert report.metrics is not None
        assert report.metrics.citation_accuracy.value == pytest.approx(0.0)
        assert report.status == RegressionStatus.FAIL


class TestMetadata:
    def test_metadata_carried_through(self) -> None:
        report = run_eval(
            _perfect_pipeline(),
            _dataset(ANSWERABLE_CASE),
            metadata={"commit": "abc123", "judge": "heuristic"},
        )
        assert report.metadata["commit"] == "abc123"
        assert report.to_dict()["metadata"]["judge"] == "heuristic"