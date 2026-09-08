"""Anti-hallucination grounding: do the answer's claims appear in the sources?

Grounding is the output-side safety net of the RAG pipeline. Even if the LLM is
instructed to cite sources, it can still drift. ``GroundedTextCheck`` verifies
that the *substantive tokens* of the answer (numbers, identifiers, technical
terms) actually appear in the retrieved chunks. If too few do, the response is
deemed ungrounded and the pipeline refuses it.

We deliberately focus on distinctive tokens (numbers and multi-char words)
rather than stopwords, because stopwords ("the", "is") are shared by everything
and would inflate the ratio.
"""

from __future__ import annotations

import re
from typing import Protocol

from ..models import RetrievedChunk

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
        "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
        "as", "at", "by", "from", "we", "you", "they", "i", "not", "but",
        "if", "so", "no", "yes", "do", "does", "did", "can", "could", "should",
        "will", "would", "may", "might", "have", "has", "had", "there", "which",
        "than", "about", "against", "between", "over", "into", "during",
    }
)


class GroundedTextCheck(Protocol):
    """Checks whether answer text is grounded in the retrieved evidence."""

    def check(
        self,
        answer: str,
        retrieved: list[RetrievedChunk],
        *,
        threshold: float,
    ) -> tuple[bool, dict[str, float]]:
        """Return ``(grounded, ratios)`` where ratios describe per-term coverage."""
        ...


class NaiveGroundedTextCheck:
    """Token-overlap grounding checker.

    For each substantive token in the answer we test whether it appears in any
    retrieved chunk. The grounded ratio = grounded tokens / total substantive
    tokens. A response is "grounded" when that ratio meets the threshold.
    """

    def __init__(self, *, min_meaningful_len: int = 3) -> None:
        self._min_len = min_meaningful_len

    def check(
        self,
        answer: str,
        retrieved: list[RetrievedChunk],
        *,
        threshold: float,
    ) -> tuple[bool, dict[str, float]]:
        if not answer.strip():
            return True, {"grounded_ratio": 1.0, "substantive_tokens": 0}

        corpus_text = " ".join(r.chunk.text for r in retrieved).lower()
        answer_terms = self._substantive_tokens(answer)

        if len(answer_terms) == 0:
            return True, {"grounded_ratio": 1.0, "substantive_tokens": 0}

        grounded = sum(1 for term in answer_terms if term in corpus_text)
        ratio = grounded / len(answer_terms)
        ratios: dict[str, float] = {
            "grounded_ratio": round(ratio, 4),
            "substantive_tokens": float(len(answer_terms)),
            "grounded_tokens": float(grounded),
        }
        return ratio >= threshold, ratios

    def _substantive_tokens(self, text: str) -> set[str]:
        tokens: set[str] = set()
        for match in re.finditer(r"\b[a-z0-9_.]+\b", text.lower()):
            token = match.group(0)
            # Skip stopwords and very short tokens, but always keep numbers.
            is_number = token.replace(".", "", 1).isdigit()
            if is_number or (len(token) >= self._min_len and token not in _STOPWORDS):
                tokens.add(token)
        return tokens
