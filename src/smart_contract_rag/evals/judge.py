"""LLM-as-judge (and its deterministic fallback) for the eval pipeline.

A *judge* decides, for a single answer, how *faithful* it is to the retrieved
evidence and how *relevant* it is to the question — the two signals that a
token-overlap heuristic alone cannot fully capture. Two implementations are
provided:

- :class:`OllamaJudge` — calls the local Ollama model with a structured scoring
  prompt and parses a ``X/5`` score plus a short reasoning. This is the "real"
  judge, used when Ollama is available in the docker network.
- :class:`HeuristicJudge` — derives 0-5 scores from the deterministic lexical
  metrics. It needs no network, so the full eval loop runs in CI.

Both implement the :class:`Judge` protocol and return a normalized :class:`JudgeScore`.
The runner normalizes these to ``[0, 1]`` for aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

from ..models import RetrievedChunk
from .metrics import answer_relevance, faithfulness

# Scale of the LLM judge (0-5). Heuristic scores are mapped onto the same scale
# so the two are directly comparable downstream.
JUDGE_MAX_SCORE: float = 5.0


class Judge(Protocol):
    """Anything able to score an answer for faithfulness and relevance.

    Implementations MUST NOT raise on unexpected model output; instead they
    return a ``JudgeScore`` with ``error`` describing what went wrong so the
    runner can decide how to treat it.
    """

    def score(
        self,
        *,
        question: str,
        answer: str,
        expected: str | None,
        retrieved: list[RetrievedChunk],
    ) -> JudgeScore:
        ...


@dataclass(frozen=True)
class JudgeScore:
    """The judge's verdict for a single answer.

    ``faithfulness`` and ``relevance`` are on the 0-5 scale; ``reasoning`` is a
    short human-readable justification; ``error`` is non-empty when scoring
    failed and the scores should be treated as unreliable.
    """

    faithfulness: float
    relevance: float
    reasoning: str = ""
    error: str = ""


def parse_judge_score(text: str) -> float:
    """Parse a model's score text into a number on the 0-5 scale.

    Tolerates common output shapes: ``"4"``, ``"4/5"``, ``"4.5"``,
    ``"Faithfulness: 4/5"``, ``"Score: 3"``, trailing prose, etc. If nothing
    parses, raises :class:`ValueError` so the caller can fall back.
    """
    import re

    if not text:
        raise ValueError("empty judge output")
    # Look for a decimal number optionally followed by /5.
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:/\s*5)?", text.strip())
    if not match:
        raise ValueError(f"no numeric score found in {text!r}")
    score = float(match.group(1))
    if score < 0.0 or score > 6.0:  # slight tolerance; clamp below
        raise ValueError(f"score out of range: {score}")
    return min(max(score, 0.0), JUDGE_MAX_SCORE)


class HeuristicJudge:
    """Deterministic judge built from the lexical metrics (no network).

    This is the default for CI: it reuses the same token-overlap ground-truth
    functions as the eval loop, then rescales the ``[0,1]`` result to the 0-5
    judge scale. It is fully reproducible.
    """

    def score(
        self,
        *,
        question: str,
        answer: str,
        expected: str | None,
        retrieved: list[RetrievedChunk],
    ) -> JudgeScore:
        # Faithfulness: token-overlap with evidence.
        f_ratio = faithfulness(answer, retrieved)
        # Relevance: keyword overlap; fall back to expected reference text.
        r_ratio = answer_relevance(answer, ground_truth=expected)
        return JudgeScore(
            faithfulness=round(f_ratio * JUDGE_MAX_SCORE, 3),
            relevance=round(r_ratio * JUDGE_MAX_SCORE, 3),
            reasoning=(
                f"heuristic: faithfulness={f_ratio:.2f}, "
                f"relevance={r_ratio:.2f} (lexical, 0-5)"
            ),
        )


class OllamaJudge:
    """LLM-as-judge that calls a local Ollama model for structured scoring.

    It is constructed with the same endpoint/model as the generator (from
    ``Settings``). The prompt asks the model to return two scores on a 0-5
    scale plus a one-line reason, and the parser extracts the numbers. On any
    failure it returns a ``JudgeScore`` with ``error`` set rather than raising,
    so the eval loop keeps running and can fall back to heuristics.
    """

    def __init__(self, *, base_url: str, model: str, timeout: float = 120.0) -> None:
        if not base_url or not model:
            raise ValueError("OllamaJudge requires base_url and model")
        self._client = httpx.Client(base_url=base_url, timeout=timeout)
        self.model = model
        self.base_url = base_url

    def _system_prompt(self) -> str:
        return (
            "You are a strict, impartial evaluator of a grounded RAG system for "
            "smart-contract security. You will be given a QUESTION, a CANDIDATE ANSWER, "
            "and the RETRIEVED SOURCES the answer was built from.\n\n"
            "Score the candidate answer on TWO axes, each 0 to 5:\n"
            "  - faithfulness: is every claim grounded in the retrieved sources, "
            "    with no invented facts? 5 = fully grounded, 0 = fabricated.\n"
            "  - relevance: does the answer directly and correctly address the "
            "    question? 5 = perfect, 0 = off-topic/empty.\n"
            "If the answer is a refusal (says it does not know), faithfulness is "
            "high (it invented nothing) but relevance is low unless the question "
            "was genuinely unanswerable.\n"
            "Reply on exactly two lines:\n"
            "  FAITHFULNESS: <0-5>/5\n"
            "  RELEVANCE: <0-5>/5\n"
            "Then one line: REASON: <short reason>."
        )

    def _user_prompt(
        self, *, question: str, answer: str, expected: str | None, retrieved: list[RetrievedChunk]
    ) -> str:
        sources = "\n".join(
            f"[{i}] ({r.chunk.source.doc_id}): {r.chunk.text[:500]}"
            for i, r in enumerate(retrieved, start=1)
        )
        expected_line = expected if expected else "(none provided)"
        return (
            f"QUESTION: {question}\n\n"
            f"EXPECTED (reference answer, may be blank): {expected_line}\n\n"
            f"CANDIDATE ANSWER:\n{answer}\n\n"
            f"RETRIEVED SOURCES:\n{sources}\n"
        )

    def score(
        self,
        *,
        question: str,
        answer: str,
        expected: str | None,
        retrieved: list[RetrievedChunk],
    ) -> JudgeScore:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": self._user_prompt(
                    question=question, answer=answer, expected=expected, retrieved=retrieved
                ),
            },
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 256},
        }
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            content = (data.get("message") or {}).get("content", "")
        except Exception as exc:  # noqa: BLE001 — network / parse / HTTP -> graceful fallback
            return JudgeScore(
                faithfulness=0.0,
                relevance=0.0,
                reasoning="",
                error=f"judge call failed: {exc}",
            )

        faithfulness_score = self._extract_score("FAITHFULNESS", content)
        relevance_score = self._extract_score("RELEVANCE", content)
        reason = self._extract_reason(content)
        if faithfulness_score is None or relevance_score is None:
            return JudgeScore(
                faithfulness=0.0,
                relevance=0.0,
                reasoning=content[:300],
                error="judge output did not parse",
            )
        return JudgeScore(
            faithfulness=faithfulness_score,
            relevance=relevance_score,
            reasoning=reason,
        )

    @staticmethod
    def _extract_score(label: str, content: str) -> float | None:
        import re

        # Look for the labelled line "LABEL: X" or "LABEL: X/5".
        match = re.search(rf"{label}\s*:\s*(\d+(?:\.\d+)?)", content, re.IGNORECASE)
        if not match:
            return None
        score = float(match.group(1))
        return min(max(score, 0.0), JUDGE_MAX_SCORE)

    @staticmethod
    def _extract_reason(content: str) -> str:
        import re

        match = re.search(r"REASON\s*:\s*(.+)", content, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return content.strip()
