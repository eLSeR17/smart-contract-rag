"""Tests for the LLM-as-judge and its deterministic heuristic fallback."""

from __future__ import annotations

import httpx
import pytest

from smart_contract_rag.evals.judge import (
    HeuristicJudge,
    OllamaJudge,
    parse_judge_score,
)
from tests.evals_fixtures import make_retrieved


class TestParseJudgeScore:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("4", 4.0),
            ("4/5", 4.0),
            ("4.5", 4.5),
            ("Faithfulness: 3/5", 3.0),
            ("Score: 2", 2.0),
            ("5", 5.0),
            ("I score this a 4 out of 5.", 4.0),
        ],
    )
    def test_parses_valid(self, text: str, expected: float) -> None:
        assert parse_judge_score(text) == pytest.approx(expected)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_judge_score("")

    def test_no_number_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_judge_score("no score here at all")

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_judge_score("99/5")

    def test_clamps_above_max(self) -> None:
        # 6 is tolerated by the regex but clamped to 5.
        assert parse_judge_score("6") == pytest.approx(5.0)


class TestHeuristicJudge:
    def _judge(self) -> HeuristicJudge:
        return HeuristicJudge()

    def test_grounded_answer_scores_high(self) -> None:
        retrieved = [make_retrieved("The withdraw function is vulnerable to reentrancy.", "d")]
        score = self._judge().score(
            question="reentrancy?",
            answer="The withdraw function is vulnerable to reentrancy.",
            expected="A reentrancy exploit.",
            retrieved=retrieved,
        )
        assert score.error == ""
        assert score.faithfulness > 4.0
        assert score.relevance > 0.0

    def test_ungrounded_answer_scores_low(self) -> None:
        retrieved = [make_retrieved("aave liquidity index 1.05", "d")]
        score = self._judge().score(
            question="weather?",
            answer="The moon is made of green cheese 42.7.",
            expected=None,
            retrieved=retrieved,
        )
        assert score.faithfulness < 1.0
        assert score.relevance == 0.0

    def test_heuristic_score_in_bounds(self) -> None:
        retrieved = [make_retrieved("some text here", "d")]
        score = self._judge().score(
            question="q", answer="some text here", expected=None, retrieved=retrieved
        )
        assert 0.0 <= score.faithfulness <= 5.0
        assert 0.0 <= score.relevance <= 5.0

    def test_heuristic_reasoning_recorded(self) -> None:
        score = self._judge().score(
            question="q",
            answer="x y z",
            expected=None,
            retrieved=[make_retrieved("x y z", "d")],
        )
        assert "heuristic" in score.reasoning


class TestOllamaJudge:
    def _judge_with_mock(
        self, reply: dict
    ) -> tuple[OllamaJudge, list[httpx.Request]]:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=reply)

        judge = OllamaJudge(base_url="http://ollama:11434", model="qwen2.5-coder:7b")
        judge._client = httpx.Client(
            transport=httpx.MockTransport(handler), base_url="http://ollama:11434"
        )
        return judge, captured

    def test_parses_labeled_scores(self) -> None:
        judge, _ = self._judge_with_mock(
            {
                "message": {
                    "content": (
                        "FAITHFULNESS: 4/5\nRELEVANCE: 3/5\n"
                        "REASON: The answer is grounded in the sources."
                    )
                }
            }
        )
        retrieved = [make_retrieved("some source", "d")]
        score = judge.score(
            question="q", answer="an answer", expected="ref", retrieved=retrieved
        )
        assert score.error == ""
        assert score.faithfulness == pytest.approx(4.0)
        assert score.relevance == pytest.approx(3.0)
        assert "grounded" in score.reasoning

    def test_sends_expected_payload(self) -> None:
        judge, captured = self._judge_with_mock(
            {"message": {"content": "FAITHFULNESS: 5/5\nRELEVANCE: 5/5"}}
        )
        retrieved = [make_retrieved("source text", "d")]
        judge.score(
            question="important?", answer="answer text", expected="ref", retrieved=retrieved
        )
        assert len(captured) == 1
        assert captured[0].url.path == "/api/chat"
        payload = captured[0].read()
        assert b"qwen2.5-coder:7b" in payload
        assert b"important?" in payload

    def test_network_error_graceful(self) -> None:
        judge = OllamaJudge(base_url="http://unreachable:1", model="ignored")
        judge._client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(httpx.ConnectError("boom"))
            ),
            base_url="http://unreachable:1",
        )
        score = judge.score(
            question="q", answer="a", expected=None, retrieved=[make_retrieved("s", "d")]
        )
        assert score.error != ""
        assert score.faithfulness == 0.0

    def test_non_compliant_output_reports_error(self) -> None:
        judge, _ = self._judge_with_mock(
            {"message": {"content": "I think the answer is good."}}
        )
        score = judge.score(
            question="q", answer="a", expected=None, retrieved=[make_retrieved("s", "d")]
        )
        assert score.error == "judge output did not parse"

    def test_constructor_requires_model(self) -> None:
        with pytest.raises(ValueError):
            OllamaJudge(base_url="http://ollama:11434", model="")

    def test_constructor_requires_url(self) -> None:
        with pytest.raises(ValueError):
            OllamaJudge(base_url="", model="qwen2.5-coder:7b")
