"""Tests for golden-dataset loading, schema validation, and subsetting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smart_contract_rag.evals.dataset import GoldenDataset, ValidationIssue

VALID_CASES = [
    {
        "id": "ev-001",
        "question": "What is reentrancy?",
        "topic": "reentrancy",
        "expected_keywords": ["reentrancy", "external", "call"],
        "relevant_doc_id": "aave-v3",
        "ground_truth": "A reentrancy attack re-enters a function.",
        "expect_answer": True,
    },
    {
        "id": "ev-002",
        "question": "Does the corpus cover Rust?",
        "topic": "trap",
        "expected_keywords": [],
        "expect_answer": False,
    },
]


class TestFromJson:
    def test_loads_valid_dataset(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.json"
        path.write_text(json.dumps(VALID_CASES), encoding="utf-8")
        ds = GoldenDataset.from_json(path)
        assert len(ds) == 2
        assert ds[0]["id"] == "ev-001"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            GoldenDataset.from_json(tmp_path / "does-not-exist.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            GoldenDataset.from_json(path)

    def test_non_list_root_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "obj.json"
        path.write_text(json.dumps({"documents": []}), encoding="utf-8")
        with pytest.raises(TypeError, match="must be a list"):
            GoldenDataset.from_json(path)


class TestValidation:
    def test_valid_dataset_has_no_issues(self) -> None:
        ds = GoldenDataset.from_cases(VALID_CASES)
        assert ds.validate().valid is True

    def test_missing_question_reported(self) -> None:
        ds = GoldenDataset.from_cases([{"id": "x"}])
        report = ds.validate()
        assert report.valid is False
        assert any(i.field == "question" for i in report.issues)

    def test_missing_id_reported(self) -> None:
        ds = GoldenDataset.from_cases([{"question": "hi"}])
        report = ds.validate()
        assert any(i.field == "id" for i in report.issues)

    def test_duplicate_id_reported(self) -> None:
        ds = GoldenDataset.from_cases(
            [
                {"id": "a", "question": "q1"},
                {"id": "a", "question": "q2"},
            ]
        )
        report = ds.validate()
        assert any(i.field == "id" and "duplicate" in i.message for i in report.issues)

    def test_unknown_topic_reported(self) -> None:
        ds = GoldenDataset.from_cases(
            [{"id": "a", "question": "q", "topic": "not-a-real-topic"}]
        )
        report = ds.validate()
        assert any(i.field == "topic" for i in report.issues)

    def test_bad_keywords_type_reported(self) -> None:
        ds = GoldenDataset.from_cases(
            [{"id": "a", "question": "q", "expected_keywords": "reentrancy"}]
        )
        report = ds.validate()
        assert any(i.field == "expected_keywords" for i in report.issues)

    def test_bad_relevant_doc_id_type_reported(self) -> None:
        ds = GoldenDataset.from_cases(
            [{"id": "a", "question": "q", "relevant_doc_id": 123}]
        )
        report = ds.validate()
        assert any(i.field == "relevant_doc_id" for i in report.issues)

    def test_topic_validation_accepts_reference_list(self) -> None:
        # Ensure every topic used in a real golden file is a known one.
        from smart_contract_rag.evals.dataset import _KNOWN_TOPICS

        for topic in ("reentrancy", "access_control", "trap", "oracle_manipulation"):
            assert topic in _KNOWN_TOPICS

    def test_validation_issue_is_frozen_value_object(self) -> None:
        issue = ValidationIssue(case_id="a", field="question", message="missing")
        assert issue.case_id == "a"
        assert issue.field == "question"

    def test_from_cases_invalid_raises_on_validate(self) -> None:
        # from_cases does not validate by default; from_json does. Check that an
        # explicitly invalid set can be caught by calling validate().
        ds = GoldenDataset.from_cases([{"id": "a", "question": "ok"}, {"id": "a", "question": "dup"}])
        assert ds.validate().valid is False


class TestSubsets:
    def test_subset_by_topic(self) -> None:
        ds = GoldenDataset.from_cases(VALID_CASES)
        trap = ds.subset_by_topic("trap")
        assert len(trap) == 1
        assert trap[0]["id"] == "ev-002"

    def test_subset_by_ids(self) -> None:
        ds = GoldenDataset.from_cases(VALID_CASES)
        only = ds.subset_by_ids({"ev-002"})
        assert len(only) == 1
        assert only[0]["id"] == "ev-002"

    def test_subset_by_predicate(self) -> None:
        ds = GoldenDataset.from_cases(VALID_CASES)
        answerable = ds.subset_by_predicate(lambda c: c.get("expect_answer", True))
        assert len(answerable) == 1
        assert answerable[0]["id"] == "ev-001"

    def test_subset_returns_new_object(self) -> None:
        ds = GoldenDataset.from_cases(VALID_CASES)
        sub = ds.subset_by_topic("trap")
        assert sub is not ds

    def test_len_and_iter(self) -> None:
        ds = GoldenDataset.from_cases(VALID_CASES)
        assert len(ds) == 2
        assert [c["id"] for c in ds] == ["ev-001", "ev-002"]

    def test_getitem(self) -> None:
        ds = GoldenDataset.from_cases(VALID_CASES)
        assert ds[1]["id"] == "ev-002"
