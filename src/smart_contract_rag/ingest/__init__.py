"""Ingestion layer: download, extract and chunk the corpus."""

from __future__ import annotations

from .chunker import RecordChunker, SentenceChunker, split_sentences
from .extractor import PDFTextExtractor
from .fetcher import CorpusFetcher

__all__ = [
    "CorpusFetcher",
    "PDFTextExtractor",
    "RecordChunker",
    "SentenceChunker",
    "split_sentences",
]
