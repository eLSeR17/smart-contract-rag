"""Evaluation metrics for the RAG system.

This module provides the **deterministic, lexical** metric functions that are
the backbone of the eval loop (and of the :class:`HeuristicJudge`). They are
pure and fully reproducible, so CI can run them without any LLM.

The metrics cover the four dimensions a production RAG must be held to:

- **Faithfulness / Grounding** — is the answer faithful to the retrieved
  evidence, or does it invent facts? Measured as the ratio of the answer's
  *substantive tokens* (numbers, technical terms) that appear in the retrieved
  chunks, mirroring the pipeline's own anti-hallucination check.
- **Answer relevance / Correctness** — does the answer actually address the
  question? Measured as lexical overlap with the golden ``expected_keywords``
  (falling back to over-lap with the ``ground_truth`` if keywords are absent).
- **Citation accuracy** — if the answer cites a source, does it correspond to
  the document that actually backs it?
- **Context precision / recall** — did retrieval surface the right chunks?

Each function returns a ``float`` in ``[0, 1]`` (or ``None`` when not
applicable, e.g. citation accuracy with no expected document). The runner
aggregates these per-case values into :class:`EvalMetric` objects, optionally
blending in the LLM-as-judge scores for faithfulness/relevance.

All tokenisation is deliberately simple (lowercased word/identifier tokens)
so the metrics are dependency-free and deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..models import RetrievedChunk, RAGResponse

# Stopwords are ignored for faithfulness because they are shared by every
# sentence and would inflate the ratio without adding evidence signal.
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

# Exact (case-insensitive) refusal phrases the system uses to decline.
_DONT_KNOW_SIGNALS: tuple[str, ...] = (
    "i don't know",
    "i do not know",
    "dont know",
    "not enough",
    "no relevant sources",
)

_TOKEN_RE = re.compile(r"[a-z0-9_.]+")
_WORD_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class EvalMetric:
    """A single aggregated metric (mean over the samples it applies to)."""

    name: str
    value: float
    per_sample: list[float] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, float | list[float] | str]:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "per_sample": [round(v, 4) for v in self.per_sample],
            "description": self.description,
        }


@dataclass(frozen=True)
class EvalMetricSet:
    """The full set of aggregated metrics produced by one eval run."""

    faithfulness: EvalMetric
    answer_relevance: EvalMetric
    citation_accuracy: EvalMetric
    context_precision: EvalMetric
    context_recall: EvalMetric
    answer_rate: EvalMetric
    correct_refusal_rate: EvalMetric
    hallucination_rate: EvalMetric

    def as_dict(self) -> dict[str, dict]:
        return {
            "faithfulness": self.faithfulness.to_dict(),
            "answer_relevance": self.answer_relevance.to_dict(),
            "citation_accuracy": self.citation_accuracy.to_dict(),
            "context_precision": self.context_precision.to_dict(),
            "context_recall": self.context_recall.to_dict(),
            "answer_rate": self.answer_rate.to_dict(),
            "correct_refusal_rate": self.correct_refusal_rate.to_dict(),
            "hallucination_rate": self.hallucination_rate.to_dict(),
        }

    def __getitem__(self, name: str) -> EvalMetric:
        return getattr(self, name)


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------
def tokens(text: str) -> set[str]:
    """Lowercased word/identifier tokens (stopwords removed)."""
    out: set[str] = set()
    for match in _TOKEN_RE.finditer(text.lower()):
        # Trim punctuation that the token class may have absorbed at the edges
        # (e.g. the trailing period in "reentrancy."), while keeping interior
        # dots for decimal numbers like "42.7".
        token = match.group(0).strip(".")
        if not token:
            continue
        is_number = token.replace(".", "", 1).isdigit()
        if is_number or (len(token) >= 3 and token not in _STOPWORDS):
            out.add(token)
    return out


def word_tokens(text: str) -> set[str]:
    """Every lowercased word token, including stopwords (for keyword overlap)."""
    return set(_WORD_RE.findall(text.lower()))


def is_refusal(answer: str) -> bool:
    """Return True if the answer is a refusal (declines to answer)."""
    lowered = answer.lower().strip()
    if not lowered:
        return True
    return any(signal in lowered for signal in _DONT_KNOW_SIGNALS)


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------
def faithfulness(answer: str, retrieved: Sequence[RetrievedChunk]) -> float:
    """Groundedness ratio: share of the answer's substantive terms found in
    the retrieved chunks. ``1.0`` = fully grounded, ``0.0`` = fully invented.

    An empty/no-op answer counts as vacuously grounded (``1.0``) because there
    is nothing to substantiate; a refusal is trivially grounded too.
    """
    if not answer.strip() or is_refusal(answer):
        return 1.0
    answer_terms = tokens(answer)
    if not answer_terms:
        return 1.0
    evidence = " ".join(r.chunk.text for r in retrieved).lower()
    grounded = sum(1 for term in answer_terms if term in evidence)
    return grounded / len(answer_terms)


def answer_relevance(
    answer: str,
    *,
    expected_keywords: Iterable[str] | None = None,
    ground_truth: str | None = None,
) -> float:
    """Lexical overlap of the answer with the golden expectation.

    If ``expected_keywords`` are provided they are the primary signal (fraction
    that appear in the answer). Otherwise the answer is compared against the
    ``ground_truth`` reference text by word overlap. Returns ``0.0`` when
    nothing to compare against and the answer is not a refusal.
    """
    if is_refusal(answer):
        # A refusal addresses the question by declining; treat as neutral for
        # relevance (the runner tracks correctness of that refusal separately).
        return 0.0
    answer_tokens = word_tokens(answer)

    keywords = list(expected_keywords) if expected_keywords is not None else []
    if keywords:
        # Tokenize each keyword the same way as the answer, so hyphenated
        # entries like "checks-effects-interactions" match robustly.
        keyword_tokens: set[str] = set()
        for keyword in keywords:
            keyword_tokens |= word_tokens(keyword)
        if not keyword_tokens:
            return 0.0
        matched = keyword_tokens & answer_tokens
        return len(matched) / len(keyword_tokens)

    if ground_truth:
        truth_tokens = word_tokens(ground_truth)
        if not truth_tokens:
            return 0.0
        overlap = len(truth_tokens & answer_tokens)
        return overlap / len(truth_tokens)

    return 0.0


def citation_accuracy(
    response: RAGResponse, expected_doc_id: str | None | list[str]
) -> float | None:
    """Whether the answer cites the document that actually backs it.

    Returns ``None`` when there is no expected document (the case is not
    suitable for citation scoring). Otherwise ``1.0`` if any cited source
    matches the expected doc id, ``0.0`` if it cites a different one (or cites
    nothing while an answer was expected).
    """
    if expected_doc_id is None:
        return None
    expected = {expected_doc_id} if isinstance(expected_doc_id, str) else set(expected_doc_id)
    if is_refusal(response.answer):
        return None
    cited = {s.doc_id for s in response.sources}
    # We only require that *at least one* cited source is the expected one.
    return 1.0 if (cited & expected) else 0.0


def context_precision(
    retrieved: Sequence[RetrievedChunk],
    relevant_doc_ids: set[str] | None,
    *,
    top_k: int | None = None,
) -> float:
    """Precision@k: share of the top ``k`` retrieved chunks that are relevant.

    Relevant = its ``doc_id`` is in ``relevant_doc_ids`` (optional). Returns
    ``0.0`` when there is no relevance signal to score against.
    """
    if not relevant_doc_ids:
        return 0.0
    bucket = list(retrieved)
    if top_k is not None:
        bucket = bucket[:top_k]
    if not bucket:
        return 0.0
    hits = sum(1 for r in bucket if r.chunk.source.doc_id in relevant_doc_ids)
    return hits / len(bucket)


def context_recall(
    retrieved: Sequence[RetrievedChunk],
    relevant_doc_ids: set[str] | None,
) -> float:
    """Recall: fraction of the relevant documents that were retrieved at all.

    ``1.0`` means every expected doc appeared somewhere in the retrieved list.
    Returns ``0.0`` when no relevance signal is available.
    """
    if not relevant_doc_ids:
        return 0.0
    retrieved_docs = {r.chunk.source.doc_id for r in retrieved}
    hit = len(retrieved_docs & relevant_doc_ids)
    return hit / len(relevant_doc_ids)


def correct_refusal(response: RAGResponse, expect_answer: bool) -> bool | None:
    """Did the system refuse *appropriately*?

    ``expect_answer`` is set by the golden case: ``True`` for answerable
    questions, ``False`` for trap questions. Returns the boolean correctness of
    the refusal decision (``None`` is never returned; a trap that refuses is
    correct, an answerable question that answers is correct).
    """
    actually_refused = response.refused or is_refusal(response.answer)
    if expect_answer:
        return not actually_refused
    return actually_refused


def is_hallucination(response: RAGResponse) -> bool:
    """A response is an hallucination when it *answers* (did not refuse) but is
    not grounded in the retrieved evidence."""
    if response.refused or is_refusal(response.answer):
        return False
    if not response.retrieved:
        return True  # answered with no evidence retrieved
    ground = faithfulness(response.answer, response.retrieved)
    return ground < 0.5
