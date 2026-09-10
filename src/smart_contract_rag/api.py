"""FastAPI server exposing the grounded RAG pipeline over HTTP.

Security model (mirrors the production pattern of the alpha-agent repo):

* ``SCRAG_AUTH=none`` (default, local dev) — open endpoint.
* ``SCRAG_AUTH=api_key`` — every ``POST /query`` must send ``X-API-Key``;
  keys are created with ``scripts/create_api_key.py``, stored hashed.
* Per-key token-bucket rate limit (``SCRAG_RATE_LIMIT`` requests/minute).
* JSON structured request logging + ``X-Request-ID`` on every response.
* ``GET /metrics`` in Prometheus text format (no external dependency).
* ``GET /health`` reports pipeline readiness (index present, model reachable).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .auth import APIKeyStore
from .metrics import Metrics
from .ratelimit import RateLimiter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

AUTH_MODE = os.getenv("SCRAG_AUTH", "none")  # "none" | "api_key"
AUTH_DB_PATH = os.getenv("SCRAG_AUTH_DB", "auth.sqlite3")
RATE_LIMIT_PER_MIN = int(os.getenv("SCRAG_RATE_LIMIT", "60"))
MAX_QUESTION_LEN = int(os.getenv("SCRAG_MAX_QUESTION", "4000"))


class JsonFormatter(logging.Formatter):
    """One JSON object per log line — parseable by any log shipper."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key in ("request_id", "method", "path", "status", "duration_ms", "client"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


_configure_logging()


# ---------------------------------------------------------------------------
# Pipeline factory (singleton, test-stub-able)
# ---------------------------------------------------------------------------

_pipeline_singleton: Any | None = None
_pipeline_error: str | None = None
_auth_singleton: APIKeyStore | None = None
_ratelimiter_singleton: RateLimiter | None = None
_metrics_singleton: Metrics | None = None


def build_pipeline() -> Any:
    """Build (or reuse) the production RAG pipeline from settings.

    Lazy on purpose: the server can boot and report ``/health`` even when
    the corpus/index is not ready yet (or the model is still downloading).
    """
    global _pipeline_singleton, _pipeline_error
    if _pipeline_singleton is not None or _pipeline_error is not None:
        return _pipeline_singleton

    from .builders import build_pipeline as _assemble
    from .config import Settings

    try:
        settings = Settings.from_env()
        _pipeline_singleton = _assemble(settings, persist_dir=str(settings.chroma_dir))
        logger.info("RAG pipeline ready (model=%s, chroma=%s)", settings.ollama_model, settings.chroma_dir)
    except Exception as exc:  # noqa: BLE001 — report readiness in /health
        _pipeline_error = f"{type(exc).__name__}: {exc}"
        logger.warning("Pipeline unavailable: %s", _pipeline_error)
    return _pipeline_singleton


def get_auth() -> APIKeyStore | None:
    """Return the shared APIKeyStore when auth is enabled, else ``None``."""
    global _auth_singleton
    if AUTH_MODE != "api_key":
        return None
    if _auth_singleton is None:
        _auth_singleton = APIKeyStore(db_path=AUTH_DB_PATH)
    return _auth_singleton


def get_ratelimiter() -> RateLimiter:
    """Return the shared RateLimiter."""
    global _ratelimiter_singleton
    if _ratelimiter_singleton is None:
        _ratelimiter_singleton = RateLimiter(default_per_minute=RATE_LIMIT_PER_MIN)
    return _ratelimiter_singleton


def get_metrics() -> Metrics:
    """Return the shared Metrics registry."""
    global _metrics_singleton
    if _metrics_singleton is None:
        _metrics_singleton = Metrics()
    return _metrics_singleton


def _clear_singletons() -> None:
    """Reset module-level state (used by tests)."""
    global _pipeline_singleton, _pipeline_error
    global _auth_singleton, _ratelimiter_singleton, _metrics_singleton
    _pipeline_singleton = None
    _pipeline_error = None
    _auth_singleton = None
    _ratelimiter_singleton = None
    _metrics_singleton = None


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUESTION_LEN)


class SourceInfo(BaseModel):
    doc_id: str
    page: int | None = None
    section: str | None = None


class RetrievedInfo(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    text: str = ""


class GroundingInfo(BaseModel):
    ok: bool
    reason: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    refused: bool = False
    grounding: GroundingInfo | None = None
    sources: list[SourceInfo] = Field(default_factory=list)
    retrieved: list[RetrievedInfo] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    auth_mode: str = AUTH_MODE
    pipeline_ready: bool = False
    pipeline_error: str | None = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SmartContractRAG API",
    version="1.0.0",
    description="Production-grade grounded RAG over public smart-contract audit reports (local models, verifiable citations).",
)


def _serialize(response: Any) -> dict[str, Any]:
    """Convert a RAGResponse dataclass into the public JSON shape."""
    sources = [
        SourceInfo(doc_id=s.doc_id, page=s.page, section=s.section)
        for s in (response.sources or [])
    ]
    retrieved = []
    for r in (response.retrieved or []):
        chunk = r.chunk
        retrieved.append(
            RetrievedInfo(
                chunk_id=chunk.id,
                doc_id=chunk.source.doc_id,
                score=float(r.score),
                text=(chunk.text or "")[:200],
            )
        )
    grounding = None
    if response.grounding is not None:
        grounding = GroundingInfo(ok=response.grounding.ok, reason=response.grounding.reason)
    return QueryResponse(
        query=response.query,
        answer=response.answer,
        refused=response.refused,
        grounding=grounding,
        sources=sources,
        retrieved=retrieved,
    ).model_dump()


@app.middleware("http")
async def request_context(request: Request, call_next: Any) -> Any:
    """Request logging (JSON) + X-Request-ID + coarse latency metric."""
    metrics = get_metrics()
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - start) * 1000.0
    response.headers["X-Request-ID"] = request_id
    metrics.inc("http_requests_total")
    metrics.inc(f"http_status_{response.status_code}_total")
    metrics.observe("request_duration_ms", duration_ms)
    extra = {
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "duration_ms": round(duration_ms, 3),
        "client": request.client.host if request.client else "",
    }
    logger.info("request", extra=extra)
    return response


def require_api_key(x_api_key: str | None = Header(default=None)) -> str | None:
    """FastAPI dependency: validate the API key (only in ``api_key`` mode)."""
    store = get_auth()
    if store is None:
        return None  # auth disabled — open mode
    key_id = store.validate_key(x_api_key or "")
    if key_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    return key_id


def check_rate_limit(key_id: str | None) -> None:
    """Rate-limit dependency: 429 when the per-key budget is exhausted."""
    limiter = get_ratelimiter()
    subject = key_id or "anonymous"
    per_minute = None
    if key_id is not None:
        store = get_auth()
        if store is not None:
            per_minute = store.rate_limit_for(key_id)
    allowed, remaining = limiter.check(subject, per_minute=per_minute)
    if not allowed:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
    get_metrics().inc("rate_limit_exhausted_total" if remaining <= 0 else "rate_limit_ok_total")


@app.post("/query", response_model=QueryResponse)
def query(
    body: QueryRequest,
    key_id: str | None = Depends(require_api_key),
) -> QueryResponse:
    """Answer a question with the grounded RAG pipeline (citations included)."""
    check_rate_limit(key_id)
    pipeline = build_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"RAG pipeline unavailable: {_pipeline_error}",
        )
    metrics = get_metrics()
    metrics.inc("queries_total")
    with metrics.time("query_latency_s"):
        response = pipeline.answer(body.query)
    if response.refused:
        metrics.inc("queries_refused_total")
    return QueryResponse.model_validate(_serialize(response))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + pipeline readiness probe."""
    pipeline = build_pipeline()  # attempt warm-up
    ready = pipeline is not None
    return HealthResponse(
        status="ok" if ready else "degraded",
        auth_mode=AUTH_MODE,
        pipeline_ready=ready,
        pipeline_error=None if ready else _pipeline_error,
    )


@app.get("/metrics")
def metrics() -> str:
    """Prometheus text exposition format."""
    return JSONResponse(content=get_metrics().render())
