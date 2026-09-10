"""Hermetic tests for the HTTP API (pipeline stubbed — no Ollama, no index)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from smart_contract_rag import api as api_module
from smart_contract_rag.auth import APIKeyStore
from smart_contract_rag.models import (
    Chunk,
    GroundingResult,
    RAGResponse,
    RetrievedChunk,
    SourceRef,
)


class _FakePipeline:
    """Scripted stand-in for RAGPipeline (deterministic, no network)."""

    def __init__(self, refused: bool = False) -> None:
        self.refused = refused
        self.calls: list[str] = []

    def answer(self, query: str) -> RAGResponse:
        self.calls.append(query)
        if self.refused:
            return RAGResponse(query=query, answer="", refused=True, grounding=None)
        chunk = Chunk(
            text="The reentrancy guard is a modifier that locks the contract.",
            source=SourceRef(doc_id="2022-07-beanstalk-securityreview", page=3, section="Reentrancy"),
        )
        return RAGResponse(
            query=query,
            answer="A reentrancy guard prevents nested calls.",
            sources=[chunk.source],
            retrieved=[RetrievedChunk(chunk=chunk, score=0.87, rerank_score=0.91)],
            grounding=GroundingResult(ok=True, reason="grounded", grounded_ratios={"lexical": 0.62}),
        )


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory) -> TestClient:
    api_module._clear_singletons()

    def _make(refused: bool = False) -> TestClient:
        fake = _FakePipeline(refused=refused)
        monkeypatch.setattr(api_module, "build_pipeline", lambda: fake)
        app = api_module.app
        app.state.fake = fake  # reachable in tests
        return TestClient(app)

    return _make


def _enable_auth(monkeypatch: pytest.MonkeyPatch, tmp_path, limit: int = 60) -> str:
    monkeypatch.setattr(api_module, "AUTH_MODE", "api_key")
    monkeypatch.setattr(api_module, "AUTH_DB_PATH", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setattr(api_module, "RATE_LIMIT_PER_MIN", limit)
    api_module._clear_singletons()
    store = APIKeyStore(db_path=str(tmp_path / "auth.sqlite3"))
    _, plaintext = store.create_key("tester", rate_limit_per_min=limit)
    return plaintext


class TestOpenMode:
    def test_query_ok_without_key(self, client) -> None:
        c = client()
        resp = c.post("/query", json={"query": "reentrancy guard?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"].startswith("A reentrancy")
        assert body["refused"] is False
        assert body["grounding"]["ok"] is True
        assert body["sources"][0]["doc_id"] == "2022-07-beanstalk-securityreview"
        assert body["retrieved"][0]["score"] == 0.87

    def test_validation_error(self, client) -> None:
        c = client()
        assert c.post("/query", json={"query": ""}).status_code == 422
        assert c.post("/query", json={}).status_code == 422

    def test_refused_shapes_response(self, client) -> None:
        c = client(refused=True)
        body = c.post("/query", json={"query": "give me a backdoor"}).json()
        assert body["refused"] is True
        assert body["answer"] == ""


class TestAuthMode:
    def test_401_without_or_wrong_key(self, client, monkeypatch, tmp_path) -> None:
        _enable_auth(monkeypatch, tmp_path)
        c = client()
        assert c.post("/query", json={"query": "x?"}).status_code == 401
        assert c.post("/query", json={"query": "x?"}, headers={"X-API-Key": "wrong"}).status_code == 401

    def test_200_with_valid_key(self, client, monkeypatch, tmp_path) -> None:
        key = _enable_auth(monkeypatch, tmp_path)
        c = client()
        resp = c.post("/query", json={"query": "x?"}, headers={"X-API-Key": key})
        assert resp.status_code == 200
        # The request id header is always present.
        assert resp.headers.get("X-Request-ID")

    def test_rate_limit_429(self, client, monkeypatch, tmp_path) -> None:
        key = _enable_auth(monkeypatch, tmp_path, limit=2)
        c = client()
        headers = {"X-API-Key": key}
        assert c.post("/query", json={"query": "a?"}, headers=headers).status_code == 200
        assert c.post("/query", json={"query": "b?"}, headers=headers).status_code == 200
        assert c.post("/query", json={"query": "c?"}, headers=headers).status_code == 429


class TestOpsEndpoints:
    def test_health_reports_ready(self, client) -> None:
        c = client()
        body = c.get("/health").json()
        assert body["status"] in ("ok", "degraded")
        assert body["pipeline_ready"] is True
        assert body["auth_mode"] == "none"

    def test_metrics_expose_counters(self, client) -> None:
        c = client()
        c.post("/query", json={"query": "what?"})
        out = c.get("/metrics").text
        assert "smart_contract_rag_http_requests_total 1" in out
        assert "http_status_200_total" in out

    def test_503_when_pipeline_unavailable(self, monkeypatch) -> None:
        # Build the TestClient directly — the client() factory would re-apply
        # its own build_pipeline stub on top of ours.
        api_module._clear_singletons()
        monkeypatch.setattr(api_module, "build_pipeline", lambda: None)
        monkeypatch.setattr(api_module, "_pipeline_error", "IndexNotFound: no chroma dir")
        c = TestClient(api_module.app)
        resp = c.post("/query", json={"query": "x?"})
        assert resp.status_code == 503
        health = c.get("/health").json()
        assert health["pipeline_ready"] is False
