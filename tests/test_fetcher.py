"""Tests for the corpus fetcher using a fake session (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smart_contract_rag.ingest.fetcher import CorpusFetchError, CorpusFetcher


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200, headers: dict | None = None) -> None:
        self.content = content
        self.status_code = status
        self._headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise CorpusFetchError(f"HTTP {self.status_code}")

    @property
    def headers(self):
        return self._headers

    def iter_content(self, chunk_size: int):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]


class FakeSession:
    def __init__(self, mapping: dict[str, bytes]) -> None:
        self._mapping = mapping
        self.headers = {}
        self.requests: list[str] = []

    def setdefault(self, *args, **kwargs) -> None:
        return None

    def get(self, url: str, stream: bool = False, timeout: int = 0) -> FakeResponse:
        self.requests.append(url)
        content = self._mapping.get(url)
        if content is None:
            return FakeResponse(b"", status=404)
        return FakeResponse(content, headers={"Content-Length": str(len(content))})


def _write_manifest(path: Path) -> None:
    manifest = {
        "documents": [
            {
                "id": "doc-a",
                "title": "Doc A",
                "url": "https://raw.example/doc-a.pdf",
                "year": 2022,
                "file": "doc-a.pdf",
            },
            {
                "id": "doc-b",
                "title": "Doc B",
                "url": "https://raw.example/doc-b.pdf",
                "year": 2023,
                "file": "doc-b.pdf",
            },
        ]
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


class TestCorpusFetcher:
    def test_fetch_all_downloads_pdfs(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        raw_dir = tmp_path / "raw"
        _write_manifest(manifest_path)

        session = FakeSession(
            {
                "https://raw.example/doc-a.pdf": b"%PDF-a",
                "https://raw.example/doc-b.pdf": b"%PDF-b",
            }
        )
        fetcher = CorpusFetcher(
            manifest_path, raw_dir, delay_seconds=0.0, session=session
        )
        result = fetcher.fetch_all()
        assert set(result) == {"doc-a", "doc-b"}
        assert (raw_dir / "doc-a.pdf").read_bytes() == b"%PDF-a"
        assert (raw_dir / "doc-b.pdf").read_bytes() == b"%PDF-b"

    def test_skips_existing_without_force(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        raw_dir = tmp_path / "raw"
        _write_manifest(manifest_path)
        raw_dir.mkdir()
        (raw_dir / "doc-a.pdf").write_bytes(b"existing")
        # Both docs available; doc-a already exists so it must be skipped.
        session = FakeSession(
            {
                "https://raw.example/doc-a.pdf": b"new",
                "https://raw.example/doc-b.pdf": b"%PDF-b",
            }
        )
        fetcher = CorpusFetcher(manifest_path, raw_dir, delay_seconds=0.0, session=session)
        fetcher.fetch_all()
        assert (raw_dir / "doc-a.pdf").read_bytes() == b"existing"  # not overwritten
        assert (raw_dir / "doc-b.pdf").read_bytes() == b"%PDF-b"  # downloaded

    def test_force_overwrites(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        raw_dir = tmp_path / "raw"
        _write_manifest(manifest_path)
        raw_dir.mkdir()
        (raw_dir / "doc-a.pdf").write_bytes(b"old")
        session = FakeSession(
            {
                "https://raw.example/doc-a.pdf": b"new-content",
                "https://raw.example/doc-b.pdf": b"%PDF-b",
            }
        )
        fetcher = CorpusFetcher(manifest_path, raw_dir, delay_seconds=0.0, session=session)
        fetcher.fetch_all(force=True)
        assert (raw_dir / "doc-a.pdf").read_bytes() == b"new-content"

    def test_manifest_papers_metadata(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        _write_manifest(manifest_path)
        fetcher = CorpusFetcher(manifest_path, tmp_path / "raw", delay_seconds=0.0)
        papers = fetcher.load_manifest()
        assert len(papers) == 2
        assert papers[0].id == "doc-a"
        assert papers[0].filename == "doc-a.pdf"
        assert papers[0].year == 2022

    def test_max_size_guard(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        raw_dir = tmp_path / "raw"
        _write_manifest(manifest_path)
        session = FakeSession({"https://raw.example/doc-a.pdf": b"x" * 100})
        fetcher = CorpusFetcher(
            manifest_path, raw_dir, delay_seconds=0.0, session=session, max_size_bytes=50
        )
        with pytest.raises(CorpusFetchError):
            fetcher.fetch_all()

    def test_missing_url_raises(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.json"
        raw_dir = tmp_path / "raw"
        _write_manifest(manifest_path)
        session = FakeSession({})  # nothing available -> 404
        fetcher = CorpusFetcher(manifest_path, raw_dir, delay_seconds=0.0, session=session)
        with pytest.raises(Exception):
            fetcher.fetch_all()
