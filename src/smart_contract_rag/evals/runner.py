"""Eval orchestration and regression guard.

``run_eval`` runs a golden dataset through a RAG pipeline and collapses the
per-case metrics into an :class:`EvalReport` with an overall verdict
(:const:`PASS` / :const:`WARN` / :const:`FAIL`) against a set of
:class:`RegressionThresholds`. The thresholds are the **regression guard**: CI
fails (FAIL) when the system degrades on any of the guarded metrics.

Design notes
------------
- The pipeline is injected (``pipeline.answer``), so this module works with the
  real RAG pipeline and equally with any deterministic fake — the runner tests
  never touch Ollama or the network.
- An optional :class:`Judge` (LLM-as-judge) can override the *faithfulness* and
  *answer-relevance* metrics. Without one, the deterministic lexical metrics are
  used, which is what runs in CI.
- Each case is scored with the deterministic metrics regardless, so the report
  always contains a complete, reproducible picture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from ..models import RAGResponse, RetrievedChunk
from .dataset import GoldenCase, GoldenDataset
from .judge import Judge
from .metrics import (
    EvalMetric,
    EvalMetricSet,
    answer_relevance,
    citation_accuracy,
    context_precision,
    context_recall,
    correct_refusal,
    faithfulness,
    is_hallucination,
    is_refusal,
)


class RegressionStatus(str, Enum):
    """Overall verdict of an eval run against the regression thresholds."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class RegressionThresholds:
    """Quality floors for the regression guard.

    Higher is better for the ``min_*`` floors; lower is better for the two
    ``max_*`` rates. ``warn_margin`` (default 0.05) defines the band below/above
    a guard where the run reports ``WARN`` instead of a hard ``FAIL``.
    """

    max_hallucination_rate: float = 0.10
    min_faithfulness: float = 0.70
    min_answer_relevance: float = 0.60
    min_context_recall: float = 0.50
    min_citation_accuracy: float = 0.60
    min_answer_rate: float = 0.70
    warn_margin: float = 0.05

    def as_dict(self) -> dict[str, float]:
        return {
            "max_hallucination_rate": self.max_hallucination_rate,
            "min_faithfulness": self.min_faithfulness,
            "min_answer_relevance": self.min_answer_relevance,
            "min_context_recall": self.min_context_recall,
            "min_citation_accuracy": self.min_citation_accuracy,
            "min_answer_rate": self.min_answer_rate,
            "warn_margin": self.warn_margin,
        }


@dataclass
class EvalCaseResult:
    """Per-case evaluation outcome."""

    case_id: str
    question: str
    topic: str
    answered: bool            # pipeline produced a (non-refused) answer
    refused: bool
    faithfulness: float
    relevance: float
    citation_accuracy: float | None
    context_precision: float
    context_recall: float
    correct_refusal: bool
    hallucination: bool
    is_trap: bool = False  # True if the golden case expects a refusal
    has_relevance_ground_truth: bool = False  # True if the case declares relevant_doc_id
    judge_reason: str = ""


class _PipelineLike(Protocol):
    def answer(self, query: str) -> RAGResponse:
        ...


@dataclass
class EvalReport:
    """Aggregated result of evaluating a golden set against a pipeline."""

    thresholds: RegressionThresholds = field(default_factory=RegressionThresholds)
    status: RegressionStatus = RegressionStatus.PASS
    metrics: EvalMetricSet | None = None
    cases: list[EvalCaseResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "n_cases": len(self.cases),
            "n_answered": sum(1 for c in self.cases if c.answered),
            "n_refused": sum(1 for c in self.cases if c.refused),
            "metrics": self.metrics.as_dict() if self.metrics else {},
            "thresholds": self.thresholds.as_dict(),
            "cases": [_case_dict(c) for c in self.cases],
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Evaluation Report",
            "",
            f"- **Verdict**: `{self.status.value}`",
            (
                f"- **Cases**: {len(self.cases)} (answered "
                f"{sum(1 for c in self.cases if c.answered)}, refused "
                f"{sum(1 for c in self.cases if c.refused)})"
            ),
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        if self.metrics:
            for name, metric in self.metrics.as_dict().items():
                lines.append(f"| {name} | {metric['value']} |")
        lines += [
            "",
            "## Per-case",
            "",
            "| id | topic | answered | faith | rel | cit | p@k | rec | refusal | hall |",
            "|----|-------|----------|-------|-----|-----|-----|-----|---------|------|",
        ]
        for case in self.cases:
            cit = f"{case.citation_accuracy:.2f}" if case.citation_accuracy is not None else "-"
            lines.append(
                f"| {case.case_id} | {case.topic} | {case.answered} | "
                f"{case.faithfulness:.2f} | {case.relevance:.2f} | {cit} | "
                f"{case.context_precision:.2f} | {case.context_recall:.2f} | "
                f"{case.correct_refusal} | {case.hallucination} |"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_markdown()


def _case_dict(case: EvalCaseResult) -> dict:
    return {
        "case_id": case.case_id,
        "topic": case.topic,
        "answered": case.answered,
        "refused": case.refused,
        "faithfulness": round(case.faithfulness, 4),
        "relevance": round(case.relevance, 4),
        "citation_accuracy": round(case.citation_accuracy, 4)
        if case.citation_accuracy is not None
        else None,
        "context_precision": round(case.context_precision, 4),
        "context_recall": round(case.context_recall, 4),
        "correct_refusal": case.correct_refusal,
        "hallucination": case.hallucination,
        "is_trap": case.is_trap,
        "has_relevance_ground_truth": case.has_relevance_ground_truth,
        "judge_reason": case.judge_reason,
    }


def _relevant_doc_ids(case: GoldenCase) -> set[str]:
    rel = case.get("relevant_doc_id")
    if not rel:
        return set()
    if isinstance(rel, str):
        return {rel}
    return set(rel)


def run_eval(
    pipeline: _PipelineLike,
    dataset: GoldenDataset,
    *,
    judge: Judge | None = None,
    thresholds: RegressionThresholds | None = None,
    use_judge_for_metrics: bool = True,
    metadata: dict | None = None,
) -> EvalReport:
    """Evaluate a pipeline against a golden dataset and produce a report.

    For every case the pipeline is asked to ``answer`` the question; the
    response is then scored. If a ``judge`` is provided (LLM-as-judge) and
    ``use_judge_for_metrics`` is True, its faithfulness/relevance scores
    (normalized to ``[0,1]``) override the lexical metrics for those two
    dimensions. All other metrics are always computed lexically.
    """
    thr = thresholds or RegressionThresholds()
    case_results: list[EvalCaseResult] = []

    for case in dataset.cases:
        response = pipeline.answer(case["question"])
        retrieved: list[RetrievedChunk] = list(getattr(response, "retrieved", []))

        rel_docs = _relevant_doc_ids(case)
        expected = case.get("ground_truth")
        keywords = case.get("expected_keywords") or []
        expect_answer = bool(case.get("expect_answer", True))
        has_relevance_gt = bool(case.get("relevant_doc_id"))

        answer_text = getattr(response, "answer", "")
        refused = getattr(response, "refused", False) or is_refusal(answer_text)

        # Faithfulness / relevance — lexical version always available.
        f_lex = faithfulness(answer_text, retrieved)
        r_lex = answer_relevance(
            answer_text,
            expected_keywords=keywords,
            ground_truth=expected,
        )
        f_norm, r_norm = f_lex, r_lex
        judge_reason = ""
        if judge is not None and use_judge_for_metrics:
            score = judge.score(
                question=case["question"],
                answer=answer_text,
                expected=expected,
                retrieved=retrieved,
            )
            judge_reason = score.reasoning
            if not score.error:
                # Normalize the 0-5 judge score to 0-1 to match the metrics.
                f_norm = min(max(score.faithfulness / 5.0, 0.0), 1.0)
                r_norm = min(max(score.relevance / 5.0, 0.0), 1.0)

        cit = citation_accuracy(response, case.get("relevant_doc_id"))
        prec = context_precision(retrieved, rel_docs, top_k=len(retrieved))
        rec = context_recall(retrieved, rel_docs)
        ref_correct = bool(correct_refusal(response, expect_answer))
        hall = bool(is_hallucination(response))

        case_results.append(
            EvalCaseResult(
                case_id=case["id"],
                question=case["question"],
                topic=case.get("topic", ""),
                answered=not refused,
                refused=refused,
                faithfulness=round(f_norm, 4),
                relevance=round(r_norm, 4),
                citation_accuracy=cit,
                context_precision=round(prec, 4),
                context_recall=round(rec, 4),
                correct_refusal=ref_correct,
                hallucination=hall,
                is_trap=not expect_answer,
                has_relevance_ground_truth=has_relevance_gt,
                judge_reason=judge_reason,
            )
        )

    metrics = _aggregate(case_results)
    status = _verdict(metrics, thr)
    return EvalReport(
        thresholds=thr,
        status=status,
        metrics=metrics,
        cases=case_results,
        metadata=metadata or {},
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(cases: list[EvalCaseResult]) -> EvalMetricSet:
    """Aggregate per-case values into :class:`EvalMetric` objects (means)."""

    # Faithfulness and relevance are only meaningful for cases that answered.
    # If nothing answered (e.g. a trap-only eval), both are vacuous → pass.
    answered_cases = [c for c in cases if c.answered]
    faith_values = [c.faithfulness for c in answered_cases] or [1.0]
    rel_values = [c.relevance for c in answered_cases] or [1.0]

    cit_values = [c.citation_accuracy for c in cases if c.citation_accuracy is not None]

    # Context precision/recall are only meaningful for cases that declare a
    # relevant document; a trap question has none, so it must not be penalized.
    relevance_cases = [c for c in cases if c.has_relevance_ground_truth]
    prec_values = [c.context_precision for c in relevance_cases]
    rec_values = [c.context_recall for c in relevance_cases]

    # Correct refusal: 1 if the case made the right refusal decision.
    correct_refusal_values = [1.0 if c.correct_refusal else 0.0 for c in cases]

    hallucination_values = [1.0 if c.hallucination else 0.0 for c in cases]

    # Answer rate: fraction of answerable questions that were answered
    # (trap questions are expected to refuse, so they are excluded). If the
    # dataset has no answerable cases, the metric is vacuous → pass.
    answerable = [c for c in cases if not c.is_trap]
    answer_rate_values = [1.0 if c.answered else 0.0 for c in answerable] or [1.0]

    return EvalMetricSet(
        faithfulness=EvalMetric(
            "faithfulness",
            _mean(faith_values),
            faith_values,
            "answer grounded in retrieved evidence (lexical or judge)",
        ),
        answer_relevance=EvalMetric(
            "answer_relevance",
            _mean(rel_values),
            rel_values,
            "answer addresses the question (keyword/reference overlap or judge)",
        ),
        citation_accuracy=EvalMetric(
            "citation_accuracy",
            _mean(cit_values) if cit_values else 1.0,  # vacuous pass when no case is scored
            cit_values,
            "cited sources include the expected document",
        ),
        context_precision=EvalMetric(
            "context_precision",
            _mean(prec_values) if prec_values else 1.0,  # vacuous pass when unmeasurable
            prec_values,
            "share of retrieved chunks that are relevant",
        ),
        context_recall=EvalMetric(
            "context_recall",
            _mean(rec_values) if rec_values else 1.0,  # vacuous pass when unmeasurable
            rec_values,
            "share of expected documents that were retrieved",
        ),
        answer_rate=EvalMetric(
            "answer_rate",
            _mean(answer_rate_values),
            answer_rate_values,
            "fraction of answerable questions actually answered",
        ),
        correct_refusal_rate=EvalMetric(
            "correct_refusal_rate",
            _mean(correct_refusal_values),
            correct_refusal_values,
            "fraction of refusal decisions that were correct (answer or traps)",
        ),
        hallucination_rate=EvalMetric(
            "hallucination_rate",
            _mean(hallucination_values),
            hallucination_values,
            "fraction of cases where the system invented an ungrounded answer",
        ),
    )


def _verdict(metrics: EvalMetricSet, thr: RegressionThresholds) -> RegressionStatus:
    """Compute PASS/WARN/FAIL from the aggregated metrics and thresholds.

    FAIL if any metric breaches its guard. WARN if any metric is within the
    warn-margin *band* of its guard (approaching it but not past). Otherwise
    PASS.

    The warn band only applies when it falls *inside* the plausible metric
    range [0, 1]:

    - **Rate** (lower better): band is ``(guard − margin, guard]``.  It only
      exists when ``guard < 1`` (not at the maximum) **and**
      ``guard − margin > 0`` (the lower edge is meaningful).  When the guard
      is at the ceiling (e.g. ``max_hallucination_rate=1.0``) the band is
      disabled — any value ≤ guard is simply PASS.
    - **Floor** (higher better): band is ``[guard, guard + margin)``.  It only
      exists when ``guard > 0`` (not at the minimum) **and**
      ``guard + margin < 1`` (the upper edge doesn't overshoot the ceiling).
      When the guard is at the floor (e.g. ``min_faithfulness=0.0``) or very
      close to 1, the band is disabled — any value ≥ guard is simply PASS.
    """
    failed: list[str] = []
    warned: list[str] = []

    # Rate metrics: lower is better.
    rate_thresholds: list[tuple[str, float]] = [
        ("hallucination_rate", thr.max_hallucination_rate),
    ]
    for label, guard in rate_thresholds:
        value = getattr(metrics, label).value
        band_low = guard - thr.warn_margin
        if value > guard:
            failed.append(label)
        elif guard < 1.0 and band_low > 0 and value > band_low:
            warned.append(label)

    # Floor metrics: higher is better.
    floor_thresholds: list[tuple[str, float]] = [
        ("faithfulness", thr.min_faithfulness),
        ("answer_relevance", thr.min_answer_relevance),
        ("context_recall", thr.min_context_recall),
        ("citation_accuracy", thr.min_citation_accuracy),
        ("answer_rate", thr.min_answer_rate),
    ]
    for label, guard in floor_thresholds:
        value = getattr(metrics, label).value
        band_high = guard + thr.warn_margin
        if value < guard:
            failed.append(label)
        elif guard > 0 and band_high < 1 and value < band_high:
            warned.append(label)

    if failed:
        return RegressionStatus.FAIL
    if warned:
        return RegressionStatus.WARN
    return RegressionStatus.PASS
