"""Command-line entry point for answering queries against the indexed corpus.

This wires the *production* components from the environment and runs the full
RAG pipeline for a single query. It requires the real stack (Ollama in the
docker network + an indexed ChromaDB corpus); the deterministic path is
exercised by ``tests/test_pipeline.py`` with fakes instead.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="smart-contract-rag-query",
        description="Answer a question against the smart-contract audit corpus using local RAG.",
    )
    parser.add_argument("query", type=str, help="The question to answer.")
    parser.add_argument(
        "--chroma-dir",
        default="data/chroma",
        help="Persistent ChromaDB directory (default: data/chroma).",
    )
    args = parser.parse_args(argv)

    try:
        from .builders import build_pipeline
        from .config import Settings

        settings = Settings.from_env()
        pipeline = build_pipeline(settings, persist_dir=args.chroma_dir)
        response = pipeline.answer(args.query)

        print(response.answer)
        if response.refused:
            print("\n(Response was refused pending grounding verification.)")
        if response.sources:
            print("\nSources:")
            for source in response.sources:
                where = f"page {source.page}" if source.page else "n/a"
                print(f"  - {source.doc_id} ({where})")
        return 0 if not response.refused else 2
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
