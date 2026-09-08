"""Input and output guardrails for the RAG pipeline.

Guardrails are the first and last lines of defence:

    * **Input** — reject empty/whitespace queries, cap query length, and
      detect potential prompt-injection payloads.
    * **Output** — a response that is not grounded (see
      :mod:`smart_contract_rag.generation.grounding`) is flagged ``ungrounded``
      so the pipeline can refuse it rather than serve an hallucination.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import RAGResponse


@dataclass
class GuardrailResult:
    """Outcome of evaluating a guardrail against an input/output."""

    passed: bool
    reason: str
    data: dict = field(default_factory=dict)


_INJECTION_SIGNALS: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore all previous",
    "system prompt:",
    "you are now",
    "disregard",
    "forget your instructions",
)


class QueryGuardrail:
    """Validates the incoming query before it reaches retrieval."""

    def __init__(self, *, max_query_length: int = 2000) -> None:
        if max_query_length <= 0:
            raise ValueError("max_query_length must be positive")
        self.max_query_length = max_query_length

    def check(self, query: str) -> GuardrailResult:
        stripped = query.strip()
        if not stripped:
            return GuardrailResult(passed=False, reason="empty_or_whitespace_query")
        if len(stripped) > self.max_query_length:
            return GuardrailResult(
                passed=False,
                reason="query_too_long",
                data={"max_length": self.max_query_length, "actual_length": len(stripped)},
            )
        lowered = stripped.lower()
        for signal in _INJECTION_SIGNALS:
            if signal in lowered:
                return GuardrailResult(passed=False, reason=f"prompt_injection:{signal}")
        return GuardrailResult(passed=True, reason="ok")


class OutputGuardrail:
    """Validates a generated response for grounding and source usage."""

    def __init__(self, *, require_citations: bool = True) -> None:
        self.require_citations = require_citations

    def check(self, response: RAGResponse) -> GuardrailResult:
        if response.refused:
            return GuardrailResult(passed=True, reason="refused")
        if not response.grounding:
            return GuardrailResult(passed=False, reason="no_grounding_check_performed")
        if not response.grounding.ok:
            return GuardrailResult(
                passed=False,
                reason="ungrounded",
                data={"grounded_ratio": response.grounding.grounded_ratios},
            )
        if self.require_citations and not response.sources:
            return GuardrailResult(passed=False, reason="no_citations")
        return GuardrailResult(passed=True, reason="ok")


def reject_if_ungrounded(response: RAGResponse) -> bool:
    """Convenience predicate: should this response be refused/flagged?"""
    if response.grounding is None:
        return True
    return not response.grounding.ok
