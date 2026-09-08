"""Assembly of the *real* (non-fake) RAG pipeline from :class:`Settings`.

This is the production wiring; unit tests instead inject fakes directly into
:class:`RAGPipeline`, so this module is not exercised by deterministic tests.
It exists so a small script (``scripts/run_rag.py``) can stand up the full
system in the docker network with a single call.
"""

from __future__ import annotations

from .config import Settings
from .generation.generator import LLMGenerator
from .generation.grounding import NaiveGroundedTextCheck
from .index.embeddings import CachedEmbeddingBackend, SentenceTransformerBackend
from .index.store import ChromaVectorStore
from .pipeline import PipelineConfig, RAGPipeline
from .retrieval.reranker import ScoreFusionReranker
from .retrieval.retriever import VectorRetriever


def build_pipeline(settings: Settings, *, persist_dir: str) -> RAGPipeline:
    """Assemble a fully-wired production pipeline from settings."""
    embedder = CachedEmbeddingBackend(SentenceTransformerBackend(settings.embedding_model))
    store = ChromaVectorStore(persist_dir=persist_dir)
    retriever = VectorRetriever(store=store, embedder=embedder, top_k_default=settings.top_k)
    reranker = ScoreFusionReranker()
    generator = LLMGenerator(base_url=settings.ollama_url, model=settings.ollama_model)
    grounded = NaiveGroundedTextCheck()
    return RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        grounded_check=grounded,
        config=PipelineConfig(top_k=settings.top_k),
    )
