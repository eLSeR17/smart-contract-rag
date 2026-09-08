"""Tests for the shared data models."""

from __future__ import annotations

from smart_contract_rag.models import AuditPaper, Chunk, SourceRef


class TestAuditPaper:
    def test_filename_derived_from_url(self) -> None:
        paper = AuditPaper(
            id="aave",
            title="Aave",
            url="https://raw.example/reviews/2021-11-aave-v3-securityreview.pdf",
            filename="",  # should be derived
        )
        assert paper.filename == "2021-11-aave-v3-securityreview.pdf"


class TestChunk:
    def test_stable_id(self) -> None:
        chunk = Chunk(text="x", source=SourceRef(doc_id="doc1"), index=3)
        assert chunk.id == "doc1::chunk-0003"

    def test_id_uniqueness(self) -> None:
        a = Chunk(text="x", source=SourceRef(doc_id="doc1"), index=0)
        b = Chunk(text="y", source=SourceRef(doc_id="doc1"), index=1)
        assert a.id != b.id
