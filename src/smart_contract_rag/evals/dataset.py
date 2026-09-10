"""Golden dataset loading, validation, and subsetting.

A *golden dataset* is a curated set of (question, expected-answer) pairs with
enough metadata to score retrieval, groundedness, and citation accuracy
automatically. It is the ground truth the eval loop measures the system
against, and it is versioned in ``data/evals/golden_set.json``.

Each case is a :class:`GoldenCase` with:

- ``question`` — the user query the RAG should answer.
- ``expected_keywords`` — substantive terms that must appear in a correct
  answer (used for lexical answer-relevance scoring).
- ``ground_truth`` — an optional reference answer (human-written).
- ``relevant_doc_id`` — the corpus doc id(s) that back the answer, used for
  context precision/recall and citation accuracy.
- ``expect_answer`` — ``True`` when the question is answerable from the
  corpus; ``False`` for a *trap* question that the system should refuse.
- ``topic`` — the vulnerable area it targets (reentrancy, oracle, ...).

The module validates the schema on load so a malformed golden set fails fast in
CI, and supports slicing into sub-sets (by topic, by id, or by any predicate).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

# The valid choice set is intentionally explicit so typos in the dataset are
# caught at load time rather than silently skewing an eval run.
_KNOWN_TOPICS: frozenset[str] = frozenset(
    {
        "reentrancy",
        "access_control",
        "integer_overflow",
        "oracle_manipulation",
        "flash_loans",
        "privilege_escalation",
        "liquidation",
        "interest_accounting",
        "token_accounting",
        "admin_key_risk",
        "upgrades",
        "input_validation",
        "trap",
        "cross_function",
    }
)


class GoldenCase(TypedDict, total=False):
    """A single golden sample.

    Only ``id`` and ``question`` are required; all other keys are validated
    but optional. ``total=False`` lets a case omit e.g. ``ground_truth``.
    """

    id: str
    question: str
    topic: str
    expected_keywords: list[str]
    relevant_doc_id: str | list[str]
    relevant_doc_hint: str
    ground_truth: str
    expect_answer: bool


@dataclass(frozen=True)
class ValidationIssue:
    """A single schema violation found in a golden set."""

    case_id: str | None
    field: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """Result of validating a golden dataset's schema."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.issues

    def __bool__(self) -> bool:
        return self.valid


@dataclass
class GoldenDataset:
    """A validated collection of golden cases.

    Instances are immutable with respect to their (ordered) list of cases; the
    ``subset_*`` helpers return *new* datasets so callers cannot accidentally
    mutate the original.
    """

    cases: list[GoldenCase]

    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, path: str | Path, *, validate: bool = True) -> GoldenDataset:
        """Load and (by default) validate a golden set from a JSON file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Golden dataset not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            try:
                raw = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in golden dataset {p}: {exc}") from exc
        if not isinstance(raw, list):
            raise TypeError("Golden dataset JSON must be a list of case objects.")
        dataset = cls(cases=[GoldenCase(**case) for case in raw])
        if validate:
            report = dataset.validate()
            if not report.valid:
                raise ValueError(_format_issues(report))
        return dataset

    @classmethod
    def from_cases(cls, cases: list[dict]) -> GoldenDataset:
        """Build a dataset from a list of plain dicts (handy for tests)."""
        return cls(cases=[GoldenCase(**case) for case in cases])

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------
    def validate(self) -> ValidationReport:
        """Return a report of every schema violation (never raises)."""
        issues: list[ValidationIssue] = []
        seen_ids: set[str] = set()
        for i, case in enumerate(self.cases):
            label = f"index {i}"
            case_id = case.get("id")
            if case_id:
                label = case_id

            if "question" not in case or not str(case.get("question", "")).strip():
                issues.append(ValidationIssue(case_id, "question", "missing or empty"))
            if not case_id:
                issues.append(ValidationIssue(None, "id", f"missing id at {label}"))
            elif case_id in seen_ids:
                issues.append(ValidationIssue(case_id, "id", "duplicate id"))
            seen_ids.add(case_id)

            keywords = case.get("expected_keywords")
            if keywords is not None:
                if not isinstance(keywords, list):
                    issues.append(ValidationIssue(case_id, "expected_keywords", "must be a list"))
                elif not all(isinstance(k, str) for k in keywords):
                    issues.append(ValidationIssue(case_id, "expected_keywords", "all entries must be strings"))

            topic = case.get("topic")
            if topic is not None and topic not in _KNOWN_TOPICS:
                issues.append(ValidationIssue(case_id, "topic", f"unknown topic {topic!r}"))

            expect = case.get("expect_answer")
            if expect is not None and not isinstance(expect, bool):
                issues.append(ValidationIssue(case_id, "expect_answer", "must be a boolean"))

            rel = case.get("relevant_doc_id")
            if rel is not None and not (
                isinstance(rel, str) or (isinstance(rel, list) and all(isinstance(x, str) for x in rel))
            ):
                    issues.append(ValidationIssue(case_id, "relevant_doc_id", "must be a str or list[str]"))

            gt = case.get("ground_truth")
            if gt is not None and not isinstance(gt, str):
                issues.append(ValidationIssue(case_id, "ground_truth", "must be a string"))

        return ValidationReport(issues=issues)

    # ------------------------------------------------------------------
    # Subsetting
    # ------------------------------------------------------------------
    def subset_by_topic(self, topic: str) -> GoldenDataset:
        return GoldenDataset(cases=[c for c in self.cases if c.get("topic") == topic])

    def subset_by_ids(self, ids: set[str] | list[str]) -> GoldenDataset:
        id_set = set(ids)
        return GoldenDataset(cases=[c for c in self.cases if c.get("id") in id_set])

    def subset_by_predicate(self, predicate: Callable[[GoldenCase], bool]) -> GoldenDataset:
        return GoldenDataset(cases=[c for c in self.cases if predicate(c)])

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def __getitem__(self, index: int) -> GoldenCase:
        return self.cases[index]


def _format_issues(report: ValidationReport) -> str:
    lines = ["Golden dataset failed validation:"]
    for issue in report.issues:
        where = f"[{issue.case_id}]" if issue.case_id else "[?]"
        lines.append(f"  - {where} {issue.field}: {issue.message}")
    return "\n".join(lines)
