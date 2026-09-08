"""Tests for token-aware and section-aware chunking."""

from __future__ import annotations

import pytest

from smart_contract_rag.ingest.chunker import RecordChunker, SentenceChunker, split_sentences, approx_tokens
from smart_contract_rag.models import SourceRef

SRC = SourceRef(doc_id="aave-v3", page=1)


def _sentence(length: int) -> str:
    """A single sentence of roughly ``length`` letters."""
    word = "word "
    sentence = ""
    while approx_tokens(sentence) < length // 5:
        sentence += word
    return sentence.strip() + "."


class TestSplitSentences:
    def test_splits_on_period_space_uppercase(self) -> None:
        text = "First sentence here. Second sentence here. Third one."
        assert split_sentences(text) == [
            "First sentence here.",
            "Second sentence here.",
            "Third one.",
        ]

    def test_keeps_abbreviations_intact(self) -> None:
        text = "Dr. Smith and e.g. Fig. 3 matter. Next statement."
        parts = split_sentences(text)
        assert any("Dr." in p for p in parts)
        assert any("Fig. 3" in p for p in parts)
        assert parts[-1] == "Next statement."

    def test_handles_whitespace_and_empty(self) -> None:
        assert split_sentences("   ") == []


class TestSentenceChunker:
    def test_respects_chunk_size(self) -> None:
        chunker = SentenceChunker(chunk_size=100, overlap=20)
        # Many small sentences starting with a capital letter (so the sentence
        # splitter recognises the boundaries) distribute across several windows.
        sentences = [f"Sentence{i} word word {i}." for i in range(120)]
        text = " ".join(sentences)
        chunks = chunker.chunk_text(text, SRC)
        assert len(chunks) >= 3
        for c in chunks:
            assert approx_tokens(c.text) <= chunker.chunk_size + 12  # tolerance for whole sentences

    def test_overlap_keeps_shared_sentences(self) -> None:
        chunker = SentenceChunker(chunk_size=200, overlap=50)
        text = " ".join(f"sentence{i} with some extra padding words here." for i in range(40))
        chunks = chunker.chunk_text(text, SRC)
        if len(chunks) > 1:
            first_tail = set(chunks[0].text.split())
            second_head = set(chunks[1].text.split())
            assert first_tail & second_head  # at least one shared word

    def test_single_source_preserved(self) -> None:
        chunker = SentenceChunker(chunk_size=1000, overlap=50)
        chunks = chunker.chunk_text("Only one sentence is here.", SRC)
        assert len(chunks) == 1
        assert chunks[0].source.doc_id == "aave-v3"
        assert chunks[0].index == 0

    def test_empty_text_yields_no_chunks(self) -> None:
        chunker = SentenceChunker(chunk_size=100, overlap=20)
        assert chunker.chunk_text("", SRC) == []

    def test_invalid_params_raise(self) -> None:
        with pytest.raises(ValueError):
            SentenceChunker(chunk_size=0, overlap=0)
        with pytest.raises(ValueError):
            SentenceChunker(chunk_size=100, overlap=100)


class TestRecordChunker:
    def test_never_splits_heading_from_content(self) -> None:
        chunker = RecordChunker(chunk_size=100, overlap=0)
        text = (
            "## Findings\n"
            "There is a reentrancy issue here in the withdraw function.\n"
            "## Recommendations\n"
            "Use checks-effects-interactions pattern to fix it.\n"
        )
        chunks = chunker.chunk_text(text, SRC)
        # Each chunk should carry its heading.
        headings = [c.text.splitlines()[0] for c in chunks if c.source.section]
        assert "## Findings" in headings
        assert "## Recommendations" in headings

    def test_section_boundaries(self) -> None:
        chunker = RecordChunker(chunk_size=10000, overlap=0)
        text = "## A\ncontent a content a content a\n## B\ncontent b content b\n"
        chunks = chunker.chunk_text(text, SRC)
        assert len(chunks) == 2
        assert all(c.source.section in {"## A", "## B"} for c in chunks)


class TestApproxTokens:
    def test_monotonic_with_length(self) -> None:
        assert approx_tokens("a" * 10) < approx_tokens("a" * 100)
