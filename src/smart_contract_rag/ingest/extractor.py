"""PDF text extraction and normalisation.

We extract per-page text with PyMuPDF (``fitz``) so we can attach page numbers
to the resulting chunks for citation. Raw extraction from PDFs is noisy (page
headers/footers, running page numbers, orphaned whitespace), so we normalise
the text before chunking.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from ..models import Chunk, SourceRef

try:  # PyMuPDF is heavy; make it an optional import for lightweight test runs.
    import fitz  # type: ignore
except Exception:  # noqa: BLE001 — pragma: no cover - import guarded for environments w/o PyMuPDF
    fitz = None  # type: ignore


# Headers/footers commonly seen in audit PDFs that add noise to embeddings.
_NOISE_PATTERNS: tuple[str, ...] = (
    r"^\s*(Trail of Bits|SECURITY ASSESSMENT|SECURITY REVIEW|DRAFT|REPORT)\s*$",
    r"^\s*assessed by\s*$",
    r"^\s*\d{1,3}\s*$",  # bare page numbers
    r"^\s*©\s*20\d\d\s+(Trail of Bits|All rights reserved)\s*$",
)


class TextExtractor(Protocol):
    """Protocol implemented by the real PDF extractor and by test fakes."""

    def extract_pages(self, path: str | Path) -> list[str]:
        """Return one normalised text block per page."""
        ...


class PDFTextExtractor:
    """Extracts and normalises text from a PDF using PyMuPDF."""

    def extract_pages(self, path: str | Path) -> list[str]:
        """Extract text page-by-page, filtering noisy lines."""
        if fitz is None:  # pragma: no cover
            raise RuntimeError(
                "PyMuPDF (fitz) is not available; install the 'pdf' extra to "
                "extract PDF text."
            )
        doc = fitz.open(str(path))
        try:
            return [self._normalise_page(doc[i].get_text()) for i in range(doc.page_count)]
        finally:
            doc.close()

    def _normalise_page(self, raw: str) -> str:
        """Strip headers/footers and collapse whitespace on a single page."""
        lines: list[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if any(re.match(pattern, stripped, re.IGNORECASE) for pattern in _NOISE_PATTERNS):
                continue
            lines.append(stripped)
        text = "\n".join(lines)
        # Collapse runs of blank lines and multiple spaces into single ones.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def build_chunks_from_pages(
    pages: list[str],
    doc_id: str,
    *,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Turn extracted pages into chunk objects.

    This helper stitches the extractor to the chunker: it preserves page numbers
    as provenance so every chunk can be cited back to a specific PDF page.

    The chunk *index* is a **global counter per document** (not per page) so
    that ``Chunk.id`` (``{doc_id}::chunk-{index:04d}``) is unique across all
    pages of a multi-page document — which is required by persistent vector
    stores like ChromaDB that reject duplicate IDs.
    """
    from .chunker import SentenceChunker

    chunker = SentenceChunker(chunk_size=chunk_size, overlap=overlap)
    chunks: list[Chunk] = []
    running_index = 0
    for page_index, page_text in enumerate(pages, start=1):
        source = SourceRef(doc_id=doc_id, page=page_index)
        for chunk in chunker.chunk_text(page_text, source):
            chunk.index = running_index
            running_index += 1
            chunks.append(chunk)
    return chunks
