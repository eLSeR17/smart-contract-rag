"""Token-aware chunking that respects document structure where possible.

Audit reports are densely written technical prose. Two chunkers are provided:

    * :class:`SentenceChunker` — robust fallback that groups whole sentences
      into fixed-size windows with configurable overlap. It works on arbitrary
      free text, which is what you get after PDF extraction loses the
      headline/section tree.

    * :class:`RecordChunker` — a section-aware chunker for text that still has
      explicit section headings (``## Heading``-style or Solidity
      ``contract Foo`` declarations). It never splits a heading from its
      content and never crosses section boundaries, which keeps each chunk
      semantically coherent for embedding.

Token counting is approximated (5 characters ≈ 1 token) so the logic is
dependency-free and fully deterministic for tests, while remaining a good
proxy for the true tokeniser used at embed/LLM time.
"""

from __future__ import annotations

import re
from typing import Protocol

from ..models import Chunk, SourceRef


# Common English abbreviations whose trailing period is NOT a sentence
# boundary. Splitting on these would break "Fig. 3", "e.g.", "Dr. Smith",
# "No.", "Sec.", "Etc." into spurious sentence fragments.
_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "e.g", "i.e", "et al", "etc", "vs", "fig", "no", "sec", "dr", "mr",
        "mrs", "ms", "st", "inc", "ltd", "co", "corp", "jr", "sr",
    }
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation.

    This intentionally avoids heavy NLP: audit prose uses standard sentence
    endings (``.`` ``!`` ``?``) and possessive titles. We keep abbreviations
    (``e.g.``, ``Dr.``, ``Fig. 3``, ``No.``) intact by never splitting on a
    period preceded by a token that is a known abbreviation, and by only
    splitting on a period followed by a space + an uppercase letter or quote.
    """
    # Mask abbreviation-dot sequences (handling capitalisation) so the boundary
    # regex does not split on them; they are restored verbatim afterwards.
    masked = text
    for abbr in _ABBREVIATIONS:
        for variant in (abbr.capitalize(), abbr.upper(), abbr):
            masked = re.sub(rf"\b{re.escape(variant)}\.(\s|$)", rf"{variant}<DOT>\1", masked)

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", masked.strip())
    return [p.replace("<DOT>", ".").strip() for p in parts if p.replace("<DOT>", ".").strip()]


def approx_tokens(text: str) -> int:
    """Rough token estimate: ~5 characters per token (mix of words/code)."""
    return max(1, (len(text) + 4) // 5)


class Chunker(Protocol):
    """Protocol for chunkers so tests can inject fakes."""

    def chunk_text(self, text: str, source: SourceRef) -> list[Chunk]:
        """Split ``text`` into a list of :class:`Chunk` objects."""
        ...


class SentenceChunker:
    """Group whole sentences into fixed-size, overlapping windows.

    Overlap is implemented by carrying the tail sentences of the previous chunk
    into the next window, so no sentence is ever split mid-sentence and
    neighbouring windows share boundary context (important for retrieval when a
    fact straddles a chunk boundary).
    """

    def __init__(self, *, chunk_size: int = 700, overlap: int = 150) -> None:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size must be > 0 and overlap must be 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, source: SourceRef) -> list[Chunk]:
        sentences = split_sentences(text)
        return self.chunk_sentences(sentences, source)

    def chunk_sentences(self, sentences: list[str], source: SourceRef) -> list[Chunk]:
        """Chunk a pre-split list of sentences (kept public for testing)."""
        chunks: list[Chunk] = []
        buffer: list[str] = []
        carry: list[str] = []

        index = 0
        for sentence in sentences:
            buffer.append(sentence)
            if approx_tokens(" ".join(buffer)) >= self.chunk_size:
                chunks.append(self._make_chunk(buffer, source, index))
                index += 1
                # Keep the tail (up to overlap) as context for the next chunk.
                carry = self._tail(buffer)
                buffer = list(carry)

        if buffer:
            chunks.append(self._make_chunk(buffer, source, index))

        # Collapse any chunk that is entirely a carry-over remnant (empty text).
        return [c for c in chunks if c.text.strip()]

    def _tail(self, sentences: list[str]) -> list[str]:
        """Return the trailing sentences whose combined size <= overlap."""
        tail: list[str] = []
        size = 0
        for sentence in reversed(sentences):
            if size + approx_tokens(sentence) > self.overlap:
                break
            tail.insert(0, sentence)
            size += approx_tokens(sentence)
        return tail

    def _make_chunk(self, sentences: list[str], source: SourceRef, index: int) -> Chunk:
        return Chunk(text=" ".join(sentences), source=source, index=index)


_SECTION_HEADING = re.compile(
    r"^(#{1,6}\s+.*|(?:contract|library|interface)\s+\w+.*)$"
)


class RecordChunker:
    """Section-aware chunker that never crosses heading boundaries.

    Text is split into records by section headings (Markdown ``## ...`` lines
    or Solidity ``contract/interface/library`` declarations). Each record is
    chunked independently with :class:`SentenceChunker`, so a heading always
    carries its own content and never bleeds neighbours.
    """

    def __init__(self, *, chunk_size: int = 700, overlap: int = 150) -> None:
        self._sentence_chunker = SentenceChunker(chunk_size=chunk_size, overlap=overlap)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_records(self, text: str) -> list[tuple[str, str]]:
        """Split text into ``(heading, body)`` records.

        A heading line (Markdown ``## ...`` or a Solidity contract declaration)
        begins a new record; everything after it belongs to that record until
        the next heading. This preserves the heading->content association and
        never lets a heading bleed into a neighbouring section.
        """
        records: list[tuple[str, str]] = []
        current_heading = ""
        current_body: list[str] = []

        def flush() -> None:
            if current_body or current_heading:
                records.append((current_heading, "\n".join(current_body).strip()))

        for line in text.splitlines():
            stripped = line.strip()
            if stripped and _SECTION_HEADING.match(stripped):
                # Any heading line starts a new record.
                flush()
                current_heading = stripped
                current_body = []
            else:
                current_body.append(line)

        flush()
        # A record with a heading but empty body is meaningless; drop it.
        return [(h, b) for h, b in records if b]

    def chunk_text(self, text: str, source: SourceRef) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        for heading, body in self._split_records(text):
            section_text = f"{heading}\n{body}" if heading else body
            section_source = SourceRef(source.doc_id, page=source.page, section=heading or None)
            for chunk in self._sentence_chunker.chunk_text(section_text, section_source):
                chunk.index = index
                chunks.append(chunk)
                index += 1
        return chunks
