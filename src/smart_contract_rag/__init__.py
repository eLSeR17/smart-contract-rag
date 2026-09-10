"""SmartContractRAG — a production-grade grounded RAG system for smart-contract audits.

The package exposes a clean public API:
    - :class:`pipeline.RAGPipeline` — the orchestration entry point
    - :data:`models.DEFAULT_CHUNK_SIZE` and ``DEFAULT_OVERLAP`` — sensible defaults
    - Re-exported model types used across components

Everything is designed to be testable without network or a local LLM: every
hard dependency (embedder, vector store, retriever, reranker, generator) is
written against a Protocol so tests can inject deterministic fakes.
"""

from __future__ import annotations

from .models import (
    AuditPaper,
    Chunk,
    GroundingResult,
    RAGResponse,
    RetrievedChunk,
    SourceRef,
)
from .pipeline import RAGPipeline

__all__ = [
    "AuditPaper",
    "Chunk",
    "GroundingResult",
    "RAGPipeline",
    "RAGResponse",
    "RetrievedChunk",
    "SourceRef",
]
