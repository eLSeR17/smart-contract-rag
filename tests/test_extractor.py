"""Tests for PDF text normalisation and page->chunk stitching.

These tests run without PyMuPDF installed: they exercise the deterministic
normalisation logic and the page-chunker stitching via plain text fixtures.
"""

from __future__ import annotations

from smart_contract_rag.ingest.extractor import PDFTextExtractor, build_chunks_from_pages


class TestNormalisation:
    def test_strips_bare_page_numbers(self) -> None:
        ex = PDFTextExtractor()
        assert ex._normalise_page("Some text\n\n12\nmore text") == "Some text\nmore text"

    def test_strips_header_lines(self) -> None:
        ex = PDFTextExtractor()
        raw = "Trail of Bits\nSECURITY ASSESSMENT\nBody text here."
        assert "Body text here." in ex._normalise_page(raw)
        assert "SECURITY ASSESSMENT" not in ex._normalise_page(raw)

    def test_collapses_whitespace(self) -> None:
        ex = PDFTextExtractor()
        normalised = ex._normalise_page("line1     with   spaces\n\n\n\nline2")
        assert "line1 with spaces" in normalised
        assert "\n\n\n" not in normalised

    def test_empty_input(self) -> None:
        ex = PDFTextExtractor()
        assert ex._normalise_page("   \n  \n") == ""


class TestBuildChunksFromPages:
    def test_chunks_keep_page_provenance(self) -> None:
        pages = ["First page content about Aave.", "Second page with more findings."]
        chunks = build_chunks_from_pages(pages, "doc-x", chunk_size=700, overlap=150)
        assert len(chunks) == 2
        assert all(c.source.doc_id == "doc-x" for c in chunks)
        assert [c.source.page for c in chunks] == [1, 2]

    def test_large_page_splits_into_multiple_chunks(self) -> None:
        page = " ".join("The reentrancy issue appears repeatedly in audits." for _ in range(60))
        chunks = build_chunks_from_pages([page], "doc-y", chunk_size=200, overlap=50)
        assert len(chunks) >= 2
