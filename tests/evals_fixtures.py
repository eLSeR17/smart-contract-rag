"""Shared helpers for the eval test-suite (deterministic, no network/LLM)."""

from __future__ import annotations

from smart_contract_rag.models import Chunk, RAGResponse, RetrievedChunk, SourceRef


def make_chunk(text: str, doc_id: str, *, page: int | None = 1) -> Chunk:
    return Chunk(text=text, source=SourceRef(doc_id=doc_id, page=page), index=0)


def make_retrieved(text: str, doc_id: str, *, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(chunk=make_chunk(text, doc_id), score=score)


def make_response(
    query: str,
    answer: str,
    *,
    doc_ids: list[str] | None = None,
    chunk_texts: list[str] | None = None,
    refused: bool = False,
) -> RAGResponse:
    """Build a :class:`RAGResponse` with consistent sources + retrieved chunks."""
    doc_ids = doc_ids or []
    chunk_texts = chunk_texts or []
    retrieved = [
        make_chunk(text, doc_id) for text, doc_id in zip(chunk_texts, doc_ids)
    ]
    sources = [SourceRef(doc_id=doc_id) for doc_id in doc_ids]
    return RAGResponse(
        query=query,
        answer=answer,
        sources=sources,
        retrieved=[RetrievedChunk(chunk=c, score=0.9) for c in retrieved],
        refused=refused,
    )


class ScriptedPipeline:
    """A deterministic fake RAG pipeline driven by a script.

    ``script`` is a dict mapping a query *substring* to a behaviour:
        {
          "default": {"answer": "...", "doc_ids": [...], "refuse": False},
          "reentrancy": {"answer": "...", "doc_ids": ["aave-v3"], "grounded": True},
          "trap q": {"answer": "I DON'T KNOW", "doc_ids": [], "refuse": True},
        }
    The behaviour whose key is a substring of the incoming query wins.
    """

    def __init__(self, script: dict) -> None:
        self.script = script
        self.calls: list[str] = []

    def answer(self, query: str) -> RAGResponse:
        self.calls.append(query)
        entry = self.script.get("default", {})
        for key, candidate in self.script.items():
            if key != "default" and key in query:
                entry = candidate
                break

        answer = entry.get("answer", "")
        doc_ids = entry.get("doc_ids", [])
        refusal = entry.get("refuse", False)
        grounded_chunks = entry.get("grounded_chunks", doc_ids)
        return make_response(
            query,
            answer,
            doc_ids=doc_ids,
            chunk_texts=grounded_chunks,
            refused=refusal,
        )
