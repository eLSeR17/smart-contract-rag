"""End-to-end pipeline tests using deterministic fakes (no LLM/network)."""

from __future__ import annotations

from typing import Callable

from smart_contract_rag.index.embeddings import DeterministicFakeEmbeddingBackend
from smart_contract_rag.index.store import InMemoryVectorStore
from smart_contract_rag.models import Chunk, RetrievedChunk, SourceRef
from smart_contract_rag.pipeline import PipelineConfig, RAGPipeline
from smart_contract_rag.retrieval.reranker import ScoreFusionReranker
from smart_contract_rag.retrieval.retriever import VectorRetriever
from smart_contract_rag.generation.grounding import (
    EntailmentGroundedTextCheck,
    NaiveGroundedTextCheck,
)


_CORPUS = [
    "The withdraw function is vulnerable to reentrancy.",
    "Aave v3 uses a liquidity index of 1.05 percent.",
    "Optimism deploys on the Ethereum mainnet.",
]


class FakeGenerator:
    """Deterministic generator whose behaviour is scripted per test.

    ``behaviour`` maps a query substring to the text the fake LLM returns.
    This lets tests inject both a grounded answer and an hallucinating one.
    """

    def __init__(self, behaviour: dict[str, str]) -> None:
        self._behaviour = behaviour
        self.last_query = ""
        self.last_chunks: list[Chunk] = []

    def generate(self, query: str, chunks: list[Chunk], system_prompt: str) -> str:
        self.last_query = query
        self.last_chunks = chunks
        for key, reply in self._behaviour.items():
            if key in query:
                return reply
        return ""


def _build_pipeline(
    *,
    generation: Callable[[str], str] | None = None,
    top_k: int = 3,
    context_chunks: int = 3,
    threshold: float = 0.5,
) -> tuple[RAGPipeline, InMemoryVectorStore, DeterministicFakeEmbeddingBackend]:
    embedder = DeterministicFakeEmbeddingBackend(dimension=16)
    store = InMemoryVectorStore()
    store.add_documents(
        [Chunk(text=t, source=SourceRef(doc_id="d1", page=i)) for i, t in enumerate(_CORPUS)],
        embedder,
    )
    retriever = VectorRetriever(store=store, embedder=embedder, top_k_default=top_k)
    rerank = ScoreFusionReranker(alpha=0.5)
    generator = _CallableGenerator(generation) if generation is not None else FakeGenerator({})
    grounded = NaiveGroundedTextCheck()
    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=rerank,
        generator=generator,
        grounded_check=grounded,
        config=PipelineConfig(top_k=top_k, context_chunks=context_chunks, grounding_threshold=threshold),
    )
    return pipeline, store, embedder


class _CallableGenerator:
    def __init__(self, fn: Callable[[str], str]) -> None:
        self._fn = fn
        self.last_chunks: list[Chunk] = []

    def generate(self, query: str, chunks: list[Chunk], system_prompt: str) -> str:
        self.last_chunks = chunks
        return self._fn(query)


class TestRAGPipelineFlow:
    def test_empty_query_refused(self) -> None:
        pipeline, _, _ = _build_pipeline()
        response = pipeline.answer("")
        assert response.refused is True

    def test_grounded_answer_returned_with_sources(self) -> None:
        pipeline, _, _ = _build_pipeline(
            generation=lambda q: "The withdraw function is vulnerable to reentrancy."
        )
        response = pipeline.answer("Is the withdraw function vulnerable?")
        assert response.refused is False
        assert response.answer.startswith("The withdraw function")
        assert response.sources  # sources extracted from retrieved chunks
        assert response.grounding is not None
        assert response.grounding.ok is True

    def test_hallucination_refused(self) -> None:
        pipeline, _, _ = _build_pipeline(
            generation=lambda q: "The moon is made of green cheese and crypto will 100x."
        )
        response = pipeline.answer("Tell me about the weather")
        assert response.refused is True
        assert "DON'T KNOW" in response.answer

    def test_component_wiring(self) -> None:
        pipeline, store, _ = _build_pipeline()
        assert store.count() == 3  # corpus was indexed
        response = pipeline.answer("reentrancy")
        # The default FakeGenerator returns "" -> grounding sees empty answer -> ok
        assert response.grounding is not None

    def test_grounded_retrieval_evidence_used(self) -> None:
        pipeline, _, _ = _build_pipeline(
            generation=lambda q: "Aave v3 uses a liquidity index of 1.05 percent."
        )
        response = pipeline.answer("What index does Aave v3 use?")
        assert response.grounding is not None and response.grounding.ok is True


class _FakeEntailmentScorer:
    """Scripted entailment scorer (cycles script if more pairs than scripted)."""

    def __init__(self, labels: list[str], scripted: list[list[float]]) -> None:
        self._labels = list(labels)
        self._scripted = scripted

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[list[float]]:
        return [
            list(self._scripted[i % len(self._scripted)])
            for i in range(len(pairs))
        ]


class TestRAGPipelineSemanticGrounding:
    """Integration: pipeline with the entailment (semantic) grounded check.

    The rest of the pipeline is the same deterministic fake stack as
    :class:`TestRAGPipelineFlow`; only the grounded-check seam is swapped.
    """

    def _build(
        self,
        *scripted: list[float],
        generation: Callable[[str], str],
    ) -> RAGPipeline:
        embedder = DeterministicFakeEmbeddingBackend(dimension=16)
        store = InMemoryVectorStore()
        store.add_documents(
            [Chunk(text=t, source=SourceRef(doc_id="d1", page=i)) for i, t in enumerate(_CORPUS)],
            embedder,
        )
        retriever = VectorRetriever(store=store, embedder=embedder, top_k_default=3)
        rerank = ScoreFusionReranker(alpha=0.5)
        generator = _CallableGenerator(generation)
        grounded = EntailmentGroundedTextCheck(
            _FakeEntailmentScorer(
                labels=["contradiction", "entailment", "neutral"],
                scripted=list(scripted),
            )
        )
        return RAGPipeline(
            retriever=retriever,
            reranker=rerank,
            generator=generator,
            grounded_check=grounded,
            config=PipelineConfig(
                top_k=3,
                context_chunks=3,
                grounding_threshold=0.5,
            ),
        )

    def test_entailing_answer_passes_output_guardrail(self) -> None:
        pipeline = self._build(
            [0.1, 0.9, 0.0],  # entails
            generation=lambda q: "The withdraw function is vulnerable to reentrancy.",
        )
        response = pipeline.answer("Is the withdraw function vulnerable?")
        assert response.refused is False
        assert response.grounding is not None and response.grounding.ok is True
        assert response.grounding.grounded_ratios["entailment_max"] == 0.9

    def test_non_entailing_answer_refused_by_output_guardrail(self) -> None:
        pipeline = self._build(
            [0.1, 0.2, 0.7],  # neutral dominates -> not entailed
            generation=lambda q: "The moon is made of green cheese.",
        )
        response = pipeline.answer("Tell me about the weather")
        assert response.refused is True
        assert (
            response.answer
            == "I DON'T KNOW — the response could not be verified against the provided sources."
        )
        assert response.grounding is not None and response.grounding.ok is False
