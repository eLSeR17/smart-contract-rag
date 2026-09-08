"""End-to-end RAG pipeline orchestration.

The pipeline wires the layers together with dependency injection so every unit
is swappable: input guardrail -> retrieve -> rerank -> ground -> generate ->
output guardrail. In production it is constructed with real components; in
tests it is built with deterministic fakes (fake embedder, in-memory store,
fake/mocked generator) so the entire flow runs without network or a local LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .generation.generator import Generator
from .generation.grounding import GroundedTextCheck
from .generation.prompt import build_system_prompt, contextual_chunks
from .guardrails import GuardrailResult, OutputGuardrail, QueryGuardrail
from .models import GroundingResult, RAGResponse, RetrievedChunk, SourceRef


class RetrieverLike(Protocol):
    """Minimal retrieval interface the pipeline depends on."""

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        ...


class RerankerLike(Protocol):
    """Minimal reranking interface the pipeline depends on."""

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        ...


@dataclass
class PipelineConfig:
    """Runtime knobs for a single :class:`RAGPipeline` instance."""

    top_k: int = 5
    context_chunks: int = 4
    grounding_threshold: float = 0.5
    require_citations: bool = True
    max_generation_chunks: int = 4


class RAGPipeline:
    """Orchestrates query -> guardrail -> retrieve -> rerank -> ground -> generate.

    The constructor takes concrete implementations but respects the protocols,
    so any of them can be a fake. This is the seam that lets the whole flow be
    tested deterministically.
    """

    def __init__(
        self,
        *,
        retriever: RetrieverLike,
        reranker: RerankerLike,
        generator: Generator,
        grounded_check: GroundedTextCheck,
        query_guardrail: QueryGuardrail | None = None,
        output_guardrail: OutputGuardrail | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        if config is None:
            config = PipelineConfig()
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.grounded_check = grounded_check
        self.query_guardrail = query_guardrail or QueryGuardrail()
        self.output_guardrail = output_guardrail or OutputGuardrail(
            require_citations=config.require_citations
        )
        self.config = config

    def answer(self, query: str) -> RAGResponse:
        """Run the full pipeline and return a guardrailed response."""
        # 1. Input guardrail.
        input_result: GuardrailResult = self.query_guardrail.check(query)
        if not input_result.passed:
            return RAGResponse(
                query=query,
                answer="",
                refused=True,
                grounding=None,
            )

        # 2. Retrieve.
        raw = self.retriever.retrieve(query, top_k=self.config.top_k)

        # 3. Rerank.
        reranked = self.reranker.rerank(
            query, raw, top_k=self.config.max_generation_chunks
        )

        if not reranked:
            return RAGResponse(
                query=query,
                answer="I DON'T KNOW — no relevant sources were found in the corpus.",
                sources=[],
                retrieved=[],
                grounding=None,
                refused=True,
            )

        # 4. Grounding check on the *evidence* is implicit; the meaningful
        #    check happens after generation below.

        # 5. Generate.
        chunks = contextual_chunks(reranked, self.config.context_chunks)
        answer_text = self.generator.generate(query, chunks, build_system_prompt())

        # 6. Output grounding verification.
        grounded_ok, ratios = self.grounded_check.check(
            answer_text, reranked, threshold=self.config.grounding_threshold
        )

        grounding = GroundingResult(
            ok=grounded_ok,
            reason="grounded" if grounded_ok else "ungrounded",
            grounded_ratios=ratios,
        )
        sources = self._extract_sources(reranked)

        response = RAGResponse(
            query=query,
            answer=answer_text,
            sources=sources,
            retrieved=reranked,
            grounding=grounding,
            refused=False,
        )

        # 7. Output guardrail.
        output_result: GuardrailResult = self.output_guardrail.check(response)
        if not output_result.passed:
            response.refused = True
            response.answer = "I DON'T KNOW — the response could not be verified against the provided sources."
        return response

    def _extract_sources(self, reranked: list[RetrievedChunk]) -> list[SourceRef]:
        seen: set[tuple] = set()
        sources: list[SourceRef] = []
        for r in reranked:
            ref = r.chunk.source
            key = (ref.doc_id, ref.page)
            if key in seen:
                continue
            seen.add(key)
            sources.append(SourceRef(doc_id=ref.doc_id, page=ref.page, section=ref.section))
        return sources
