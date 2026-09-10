"""Tests for the deterministic evaluation metrics and the EvalMetric types."""

from __future__ import annotations

import pytest

from smart_contract_rag.evals.metrics import (
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
from smart_contract_rag.models import RAGResponse, SourceRef
from tests.evals_fixtures import make_response, make_retrieved


class TestTokensAndRefusal:
    def test_refusal_detected(self) -> None:
        assert is_refusal("I DON'T KNOW") is True
        assert is_refusal("I don't know this from the sources.") is True
        assert is_refusal("No relevant sources were found.") is True

    def test_non_refusal(self) -> None:
        assert is_refusal("The withdraw function is vulnerable.") is False

    def test_empty_is_refusal(self) -> None:
        assert is_refusal("  ") is True


class TestFaithfulness:
    def test_fully_grounded(self) -> None:
        retrieved = [make_retrieved("The withdraw function is vulnerable to reentrancy.", "d")]
        score = faithfulness("The withdraw function is vulnerable to reentrancy.", retrieved)
        assert score == pytest.approx(1.0)

    def test_partially_grounded(self) -> None:
        retrieved = [make_retrieved("withdraw function reentrancy", "d")]
        score = faithfulness("The withdraw function is vulnerable to reentrancy.", retrieved)
        # Substantive terms: withdraw, function, vulnerable, reentrancy — only
        # withdraw/function/reentrancy are in evidence, so 3/4 ≈ 0.75.
        assert score == pytest.approx(0.75)

    def test_ungrounded(self) -> None:
        retrieved = [make_retrieved("aave liquidity index 1.05", "d")]
        score = faithfulness("The moon is made of green cheese 42.7.", retrieved)
        assert score < 0.3

    def test_refusal_counts_grounded(self) -> None:
        retrieved = [make_retrieved("some unrelated text", "d")]
        assert faithfulness("I DON'T KNOW.", retrieved) == pytest.approx(1.0)

    def test_empty_answer_grounded(self) -> None:
        assert faithfulness("", []) == pytest.approx(1.0)

    def test_numbers_counted(self) -> None:
        retrieved = [make_retrieved("the fee rate is 42.7 percent", "d")]
        # `42.7` and `percent` grounded; if answer replicates them, score high.
        score = faithfulness("the fee is 42.7 percent", retrieved)
        assert score > 0.5


class TestAnswerRelevance:
    def test_keyword_overlap_full(self) -> None:
        score = answer_relevance(
            "The withdraw call is vulnerable to reentrancy.",
            expected_keywords=["withdraw", "reentrancy", "call"],
        )
        assert score == pytest.approx(1.0)

    def test_keyword_overlap_partial(self) -> None:
        score = answer_relevance(
            "Only reentrancy is mentioned.",
            expected_keywords=["withdraw", "reentrancy", "call"],
        )
        assert score == pytest.approx(1 / 3)

    def test_no_overlap_zero(self) -> None:
        score = answer_relevance(
            "The moon is lovely.",
            expected_keywords=["withdraw", "reentrancy"],
        )
        assert score == pytest.approx(0.0)

    def test_ground_truth_fallback(self) -> None:
        score = answer_relevance(
            "This answers about withdraw and reentrancy fully.",
            ground_truth="withdraw and reentrancy",
        )
        assert score > 0.0

    def test_refusal_neutral(self) -> None:
        score = answer_relevance(
            "I DON'T KNOW",
            expected_keywords=["withdraw", "reentrancy"],
        )
        assert score == pytest.approx(0.0)

    def test_no_reference_returns_zero(self) -> None:
        assert answer_relevance("anything") == pytest.approx(0.0)


class TestCitationAccuracy:
    def test_correct_doc_cited(self) -> None:
        response = make_response("q", "answer [1]", doc_ids=["aave-v3"])
        assert citation_accuracy(response, "aave-v3") == pytest.approx(1.0)

    def test_wrong_doc_cited(self) -> None:
        response = make_response("q", "answer [1]", doc_ids=["balancer"])
        assert citation_accuracy(response, "aave-v3") == pytest.approx(0.0)

    def test_accepts_list_of_docs(self) -> None:
        response = make_response("q", "answer [1]", doc_ids=["aave-v3"])
        assert citation_accuracy(response, ["balancer", "aave-v3"]) == pytest.approx(1.0)

    def test_no_expected_doc_returns_none(self) -> None:
        response = make_response("q", "answer [1]", doc_ids=["aave-v3"])
        assert citation_accuracy(response, None) is None

    def test_refusal_not_scored(self) -> None:
        response = make_response("q", "I DON'T KNOW", refused=True)
        assert citation_accuracy(response, "aave-v3") is None

    def test_no_citation_while_answer_expected(self) -> None:
        response = RAGResponse(
            query="q",
            answer="Some grounded answer without sources.",
            sources=[],
            retrieved=[],
            refused=False,
        )
        assert citation_accuracy(response, "aave-v3") == pytest.approx(0.0)


class TestContextPrecisionRecall:
    def _retrieved(self, doc_ids: list[str] | None = None) -> list:
        ids = doc_ids or ["a", "b", "c"]
        return [make_retrieved(f"text {id}", id) for id in ids]

    def test_precision_all_relevant(self) -> None:
        retrieved = self._retrieved(["a", "b", "c"])
        assert context_precision(retrieved, {"a", "b", "c"}) == pytest.approx(1.0)

    def test_precision_partial(self) -> None:
        retrieved = self._retrieved(["a", "b", "c"])
        assert context_precision(retrieved, {"a"}) == pytest.approx(1 / 3)

    def test_precision_top_k(self) -> None:
        retrieved = self._retrieved(["a", "b", "c"])
        # top-1 is 'a', which is relevant.
        assert context_precision(retrieved, {"a"}, top_k=1) == pytest.approx(1.0)

    def test_precision_empty_relevance(self) -> None:
        assert context_precision(self._retrieved(), None) == pytest.approx(0.0)

    def test_recall_all_relevant_present(self) -> None:
        retrieved = self._retrieved(["a", "b", "c", "d"])
        assert context_recall(retrieved, {"a", "b"}) == pytest.approx(1.0)

    def test_recall_partial(self) -> None:
        retrieved = self._retrieved(["a"])
        assert context_recall(retrieved, {"a", "b"}) == pytest.approx(0.5)

    def test_recall_missing_all(self) -> None:
        retrieved = self._retrieved(["x"])
        assert context_recall(retrieved, {"a", "b"}) == pytest.approx(0.0)


class TestRefusalAndHallucination:
    def test_correct_refusal_on_trap(self) -> None:
        response = make_response("q", "I DON'T KNOW", refused=True)
        assert correct_refusal(response, expect_answer=False) is True

    def test_wrong_refusal_on_answerable(self) -> None:
        response = make_response("q", "I DON'T KNOW", refused=True)
        assert correct_refusal(response, expect_answer=True) is False

    def test_answer_on_answerable_is_correct(self) -> None:
        response = make_response("q", "A grounded answer.", doc_ids=["a"], chunk_texts=["text"])
        assert correct_refusal(response, expect_answer=True) is True

    def test_answer_on_trap_is_wrong(self) -> None:
        response = make_response("q", "A made-up answer.", doc_ids=["a"], chunk_texts=["text"])
        assert correct_refusal(response, expect_answer=False) is False

    def test_hallucination_when_ungrounded(self) -> None:
        response = RAGResponse(
            query="q",
            answer="The moon is made of green cheese 42.7.",
            sources=[SourceRef(doc_id="a")],
            retrieved=[make_retrieved("reentrancy withdraw", "a")],
            refused=False,
        )
        assert is_hallucination(response) is True

    def test_not_hallucination_when_refused(self) -> None:
        response = make_response("q", "I DON'T KNOW", refused=True)
        assert is_hallucination(response) is False

    def test_hallucination_when_no_evidence(self) -> None:
        response = RAGResponse(
            query="q",
            answer="A confident ungrounded claim here.",
            sources=[],
            retrieved=[],
            refused=False,
        )
        assert is_hallucination(response) is True

    def test_grounded_is_not_hallucination(self) -> None:
        response = make_response(
            "q", "The withdraw function is vulnerable to reentrancy.",
            doc_ids=["a"], chunk_texts=["The withdraw function is vulnerable to reentrancy."],
        )
        assert is_hallucination(response) is False


class TestEvalMetricTypes:
    def test_metric_serialization(self) -> None:
        metric = EvalMetric("faithfulness", 0.8, [0.8, 0.9])
        out = metric.to_dict()
        assert out["value"] == pytest.approx(0.8)
        assert out["per_sample"] == [0.8, 0.9]

    def test_metric_set_as_dict_keys(self) -> None:
        mset = EvalMetricSet(
            faithfulness=EvalMetric("faithfulness", 1.0),
            answer_relevance=EvalMetric("answer_relevance", 1.0),
            citation_accuracy=EvalMetric("citation_accuracy", 1.0),
            context_precision=EvalMetric("context_precision", 1.0),
            context_recall=EvalMetric("context_recall", 1.0),
            answer_rate=EvalMetric("answer_rate", 1.0),
            correct_refusal_rate=EvalMetric("correct_refusal_rate", 1.0),
            hallucination_rate=EvalMetric("hallucination_rate", 0.0),
        )
        d = mset.as_dict()
        assert set(d) == {
            "faithfulness",
            "answer_relevance",
            "citation_accuracy",
            "context_precision",
            "context_recall",
            "answer_rate",
            "correct_refusal_rate",
            "hallucination_rate",
        }
        assert mset["faithfulness"].value == 1.0

    def test_metric_name(self) -> None:
        assert EvalMetric("x", 0.5).name == "x"
