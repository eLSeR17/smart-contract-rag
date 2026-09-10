"""Tests for PDF text normalisation and page->chunk stitching.

These tests run without PyMuPDF installed: they exercise the deterministic
normalisation logic and the page-chunker stitching via plain text fixtures.
"""

from __future__ import annotations

from smart_contract_rag.ingest.extractor import (
    PDFTextExtractor,
    build_chunks_from_pages,
)


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


class TestMultiPageChunkIdUniqueness:
    """Regression: multi-page documents must produce unique chunk IDs.

    Previously, ``build_chunks_from_pages`` delegated chunking per page to
    ``SentenceChunker`` which resets its internal ``index`` to 0 on every
    call.  This caused ``Chunk.id`` collisions (e.g. ``doc::chunk-0000``
    appearing on both page 1 and page 2).  Persistent vector stores such as
    ChromaDB raise ``DuplicateIDError`` on such collisions.

    The fix introduces a *running* global counter across pages so every chunk
    in a document gets a unique, strictly-increasing index.
    """

    def test_all_chunk_ids_are_unique(self) -> None:
        """Two pages with enough text for multiple chunks each → no ID collisions."""
        page_a = " ".join(
            f"Sentence {i} discusses reentrancy and access control patterns." for i in range(40)
        )
        page_b = " ".join(
            f"Sentence {i} covers oracle manipulation and flash loan vectors." for i in range(40)
        )
        chunks = build_chunks_from_pages(
            [page_a, page_b], "multi-page-doc", chunk_size=200, overlap=50
        )
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), f"Duplicate chunk IDs found: {ids}"

    def test_indices_are_strictly_increasing(self) -> None:
        """Chunk indices must be 0, 1, 2, … across page boundaries."""
        page_a = " ".join(
            f"Sentence {i} discusses reentrancy and access control patterns." for i in range(40)
        )
        page_b = " ".join(
            f"Sentence {i} covers oracle manipulation and flash loan vectors." for i in range(40)
        )
        chunks = build_chunks_from_pages(
            [page_a, page_b], "multi-page-doc", chunk_size=200, overlap=50
        )
        indices = [c.index for c in chunks]
        assert indices == list(range(len(indices)))

    def test_page_provenance_preserved_with_global_index(self) -> None:
        """source.page still reflects the real page; index is global."""
        pages = ["First page text about audit findings here.", "Second page text about recommendations."]
        chunks = build_chunks_from_pages(pages, "doc-z", chunk_size=700, overlap=150)
        assert len(chunks) == 2
        assert chunks[0].source.page == 1
        assert chunks[1].source.page == 2
        assert chunks[0].index == 0
        assert chunks[1].index == 1

    def test_single_page_still_starts_at_zero(self) -> None:
        """A single-page doc should behave identically to the old behaviour."""
        chunks = build_chunks_from_pages(
            ["Some single page content here about audits."], "single-doc", chunk_size=700, overlap=150
        )
        assert len(chunks) == 1
        assert chunks[0].index == 0
        assert chunks[0].id == "single-doc::chunk-0000"
