"""Corpus downloader driven by ``data/corpus/manifest.json``.

Audits are public PDFs hosted on the Trail of Bits publications repository.
We *never* commit the binary PDFs (they are gitignored); instead we version the
manifest (URL + metadata) and download on demand, so the repo itself stays
lightweight and copyright-agnostic.

The fetcher respects the upstream source:
    * sequential downloads (no parallel blast to GitHub raw),
    * a small delay between requests to avoid hammering,
    * idempotent (skips files that already exist at the expected size),
    * a user-agent string identifying the tool for upstream logging.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from ..models import AuditPaper


class CorpusFetcher:
    """Downloads the corpus described by a manifest into a raw directory."""

    def __init__(
        self,
        manifest_path: Path,
        raw_dir: Path,
        *,
        delay_seconds: float = 0.5,
        timeout: int = 60,
        max_size_bytes: int = 20_000_000,
        session: requests.Session | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.raw_dir = Path(raw_dir)
        self.delay_seconds = delay_seconds
        self.timeout = timeout
        self.max_size_bytes = max_size_bytes
        # Allow callers to inject a Session (e.g. a mocked one) for tests.
        self._session = session or requests.Session()
        self._session.headers.setdefault(
            "User-Agent", "SmartContractRAG-corpus-fetcher/1.0 (portfolio demo)"
        )

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------
    def load_manifest(self) -> list[AuditPaper]:
        """Parse the manifest file into a list of :class:`AuditPaper`."""
        with self.manifest_path.open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
        documents = data.get("documents", [])
        return [
            AuditPaper(
                id=doc["id"],
                title=doc.get("title", doc["id"]),
                url=doc["url"],
                year=doc.get("year", 0),
                project=doc.get("project", ""),
                filename=doc.get("file", ""),
            )
            for doc in documents
        ]

    # ------------------------------------------------------------------
    # Downloading
    # ------------------------------------------------------------------
    def fetch_all(self, *, force: bool = False) -> dict[str, Path]:
        """Download every document in the manifest.

        Returns a mapping of paper id -> local raw PDF path. Files that already
        exist are skipped unless ``force`` is True.
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, Path] = {}
        for paper in self.load_manifest():
            destination = self.raw_dir / paper.filename
            if destination.exists() and not force:
                results[paper.id] = destination
                continue
            self._fetch_one(paper, destination)
            time.sleep(self.delay_seconds)  # be polite to the upstream host
            results[paper.id] = destination
        return results

    def _fetch_one(self, paper: AuditPaper, destination: Path) -> None:
        """Download a single PDF, writing it atomically to avoid partial files."""
        response = self._session.get(paper.url, stream=True, timeout=self.timeout)
        response.raise_for_status()

        # Guard against a pathological response: refuse absurd sizes.
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > self.max_size_bytes:
            raise CorpusFetchError(
                f"Refusing download of {paper.url}: declared size "
                f"{content_length} exceeds {self.max_size_bytes} bytes"
            )

        tmp = destination.with_suffix(destination.suffix + ".part")
        bytes_written = 0
        with tmp.open("wb") as out:
            for chunk in response.iter_content(chunk_size=8192):
                out.write(chunk)
                bytes_written += len(chunk)
                if bytes_written > self.max_size_bytes:
                    tmp.unlink(missing_ok=True)
                    raise CorpusFetchError(
                        f"Download of {paper.url} exceeded {self.max_size_bytes} bytes"
                    )
        tmp.rename(destination)


class CorpusFetchError(Exception):
    """Raised when a corpus PDF cannot be downloaded or is malformed."""
