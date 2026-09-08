#!/usr/bin/env python3
"""Index the downloaded corpus into a persistent ChromaDB vector store.

Pipeline: for each raw PDF in ``data/corpus/raw/``:
    * extract per-page text with PyMuPDF,
    * chunk it into token-aware chunks (keeping page provenance),
    * embed each chunk with the configured local embedder,
    * store it in a persistent ChromaDB collection.

This is an *offline* build step: it requires the PDFs (see fetch_corpus.py) and
the sentence-transformers embedder downloaded on first use. It does not call
the LLM or touch the network beyond the model download.

Each PDF is indexed with the canonical ``id`` from the corpus manifest
(``data/corpus/manifest.json``) as its ``doc_id``, falling back to the PDF
stem when the file is not listed in the manifest.

Usage:
    python scripts/index_corpus.py [--chroma-dir data/chroma] [--manifest data/corpus/manifest.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smart_contract_rag.config import Settings  # noqa: E402
from smart_contract_rag.index.embeddings import (  # noqa: E402
    CachedEmbeddingBackend,
    SentenceTransformerBackend,
)
from smart_contract_rag.index.store import ChromaVectorStore  # noqa: E402
from smart_contract_rag.ingest.extractor import PDFTextExtractor, build_chunks_from_pages  # noqa: E402


def main() -> int:
    default_root = Path(__file__).resolve().parent.parent / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(default_root / "corpus" / "raw"))
    parser.add_argument("--chroma-dir", default=str(default_root / "chroma"))
    parser.add_argument("--manifest", default=str(default_root / "corpus" / "manifest.json"))
    parser.add_argument(
        "--chunk-size", type=int, default=700,
        help="Approx token size of each chunk (default 700).",
    )
    parser.add_argument(
        "--overlap", type=int, default=150,
        help="Overlap between consecutive chunks (default 150).",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    embedder = CachedEmbeddingBackend(SentenceTransformerBackend(settings.embedding_model))
    store = ChromaVectorStore(persist_dir=args.chroma_dir)
    extractor = PDFTextExtractor()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        print("No raw corpus found. Run scripts/fetch_corpus.py first.", file=sys.stderr)
        return 1

    # Map PDF stems to the canonical manifest ``id`` so that citation ids used by
    # the eval golden dataset are stable. The basename of each document's ``url``
    # is the exact filename of the downloaded PDF (see fetch_corpus.py); falling
    # back to the stem keeps files outside the manifest safely indexable.
    stem_to_id: dict[str, str] = {}
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)
        for doc in manifest.get("documents", []):
            doc_id = doc.get("id")
            url = doc.get("url", "")
            url_stem = Path(url).stem
            if doc_id and url_stem:
                stem_to_id[url_stem] = doc_id

    total_chunks = 0
    for pdf_path in sorted(raw_dir.glob("*.pdf")):
        doc_id = stem_to_id.get(pdf_path.stem, pdf_path.stem)
        print(f"  * {pdf_path.name}  (doc_id={doc_id})")
        pages = extractor.extract_pages(pdf_path)
        chunks = build_chunks_from_pages(
            pages,
            doc_id=doc_id,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        store.add_documents(chunks, embedder)
        total_chunks += len(chunks)
        print(f"      -> {len(chunks)} chunks")

    print(f"Indexed {total_chunks} chunks into {args.chroma_dir}.")
    print(f"Total in collection: {store.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
