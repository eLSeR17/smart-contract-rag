"""Data models shared across the RAG pipeline.

Centralising the value objects here keeps the components decoupled: the
ingestion layer produces :class:`Chunk` objects, the retrieval layer returns
:class:`RetrievedChunk` objects, and the generation layer returns
:class:`RAGResponse`. They are deliberately plain dataclasses (not ORM
entities) so they serialise easily to JSON for the eval pipeline and stay
trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Reasonable defaults for the sentence-level chunker. These are in *tokens*;
# sentence-transformers tokenisers count ~1 token per word, so 700 tokens is
# roughly a long paragraph — small enough for a 7B context model, large enough
# to keep each chunk semantically self-contained.
DEFAULT_CHUNK_SIZE: int = 700
DEFAULT_OVERLAP: int = 150


@dataclass(frozen=True)
class SourceRef:
    """A specific location inside a source document, used for citations.

    ``doc_id`` matches a manifest entry (or a synthetic id for offline text),
    ``page`` and ``section`` are best-effort provenance so a claim can be
    traced back to a concrete place a reviewer can open.
    """

    doc_id: str
    page: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class AuditPaper:
    """Metadata describing a public smart-contract audit PDF from the manifest."""

    id: str
    title: str
    url: str
    year: int = 0
    project: str = ""
    filename: str = ""

    def __post_init__(self) -> None:
        if not self.filename:
            object.__setattr__(self, "filename", self.url.rsplit("/", 1)[-1])


@dataclass
class Chunk:
    """A single text unit produced by the chunker, ready to be embedded."""

    text: str
    source: SourceRef
    index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Stable id derived from provenance — useful for deduplication."""
        return f"{self.source.doc_id}::chunk-{self.index:04d}"


@dataclass
class RetrievedChunk:
    """A chunk returned by retrieval, together with its similarity score.

    ``score`` is the cosine similarity from the vector store (higher is
    better); ``rerank_score`` is the optional hybrid score from the reranker.
    """

    chunk: Chunk
    score: float
    rerank_score: float | None = None


@dataclass
class GroundingResult:
    """Outcome of the anti-hallucination grounding check."""

    ok: bool
    reason: str
    grounded_ratios: dict[str, float] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """The final, guardrailed answer returned to the caller."""

    query: str
    answer: str
    sources: list[SourceRef] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    grounding: GroundingResult | None = None
    refused: bool = False
