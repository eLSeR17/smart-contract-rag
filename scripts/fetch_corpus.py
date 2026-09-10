#!/usr/bin/env python3
"""Download the audit corpus described by ``data/corpus/manifest.json``.

The corpus is a set of public Trail of Bits audit reports (see the manifest for
URLs and metadata). We download them into ``data/corpus/raw/`` (gitignored).
The manifest is the committed, versioned source of truth; the binary PDFs are
never committed.

The script is deliberate about respecting the upstream source:
    * it downloads sequentially (no parallel burst against GitHub raw),
    * it pauses briefly between files,
    * it skips files that already exist (idempotent) unless ``--force``,
    * it prints a progress line per download.

Requirements: ``pip install requests`` (also pulled in transitively elsewhere).
Usage:
    python scripts/fetch_corpus.py [--force] [--manifest data/corpus/manifest.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a standalone script before the package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smart_contract_rag.ingest.fetcher import CorpusFetcher


def main() -> int:
    default_root = Path(__file__).resolve().parent.parent / "data" / "corpus"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m", "--manifest",
        default=str(default_root / "manifest.json"),
        help="Path to the corpus manifest (default: data/corpus/manifest.json).",
    )
    parser.add_argument(
        "-o", "--out",
        default=str(default_root / "raw"),
        help="Directory to write downloaded PDFs (default: data/corpus/raw).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download files even if they already exist.",
    )
    args = parser.parse_args()

    fetcher = CorpusFetcher(
        manifest_path=Path(args.manifest),
        raw_dir=Path(args.out),
    )
    try:
        papers = fetcher.load_manifest()
        print(f"Manifest: {len(papers)} audit PDF(s).")
        results = fetcher.fetch_all(force=args.force)
    except Exception as exc:  # noqa: BLE001 — CLI tool; report and exit
        print(f"Failed: {exc}", file=sys.stderr)
        return 1

    for paper_id, path in results.items():
        print(f"  [ok] {paper_id} -> {path}")
    print(f"Downloaded/verified {len(results)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
