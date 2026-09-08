"""Evaluation pipeline for SmartContractRAG.

This subpackage measures, with a **golden dataset** and an **LLM-as-judge**,
whether the RAG system actually works: answers are correct, faithful to the
retrieved evidence, correctly cited, and the system refuses rather than
hallucinate. It is the "no-negociable" that distinguishes a production RAG from
a tutorial RAG, and it is what lets CI catch regressions.

Public API
----------
- :class:`GoldenDataset` / :class:`GoldenCase` — load and validate the golden set.
- :module:`metrics` — the deterministic per-sample metric functions and the
  :class:`EvalMetric` type.
- :class:`Judge` / :class:`OllamaJudge` / :class:`HeuristicJudge` — the
  LLM-as-judge scoring layer (with a deterministic fallback for CI).
- :func:`run_eval` / :class:`EvalReport` — orchestrate an evaluation and produce
  a regression verdict (PASS / WARN / FAIL).
"""

from __future__ import annotations

from .dataset import GoldenCase, GoldenDataset, ValidationIssue, ValidationReport
from .judge import HeuristicJudge, Judge, JudgeScore, OllamaJudge, parse_judge_score
from .metrics import EvalMetric, EvalMetricSet
from .runner import (
    EvalCaseResult,
    EvalReport,
    RegressionStatus,
    RegressionThresholds,
    run_eval,
)

__all__ = [
    "EvalCaseResult",
    "EvalMetric",
    "EvalMetricSet",
    "EvalReport",
    "GoldenCase",
    "GoldenDataset",
    "HeuristicJudge",
    "Judge",
    "JudgeScore",
    "OllamaJudge",
    "RegressionStatus",
    "RegressionThresholds",
    "ValidationIssue",
    "ValidationReport",
    "parse_judge_score",
    "run_eval",
]
