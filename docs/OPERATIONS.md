# SmartContractRAG — Operations

Runbook for the REST API deployment (`docker-compose.prod.yml`).

## Deployment

```bash
# build + start (attaches to the external docker_default network for Ollama)
docker compose -f docker-compose.prod.yml up -d --build

# first-time key
docker compose -f docker-compose.prod.yml exec scrag-api \
  python scripts/create_api_key.py "ci-bot"

# logs
docker compose -f docker-compose.prod.yml logs -f scrag-api

# stop / update
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml up -d --build  # after code changes
```

Persistent state lives in the named volume `scrag-data`
(`/app/data`): `chroma/` (vector index), `corpus/` (PDFs + manifest) and
`auth.sqlite3` (API keys).

## Key lifecycle (security)

- **Create** — `scripts/create_api_key.py`; the secret prints once and is
  stored hashed (SHA-256). Treat it like a password.
- **Rotate** — create a new key, switch the consumer, then revoke the old.
- **Revoke** — `SCRAG_AUTH_DB=auth.sqlite3 python -c "from smart_contract_rag.auth import APIKeyStore; s=APIKeyStore('auth.sqlite3'); s.revoke_key('scrag_xxx')"`
- **Audit** — `list_keys()` shows id/name/created/active (never hashes).

## Health & capacity

- `GET /health` — liveness + pipeline readiness. `degraded` means the model
  or index is unreachable; the process stays up to report it.
- `GET /metrics` — request counters, per-status breakdown, latency
  histograms (p50/p90/p95/p99). Wire it into Prometheus/Grafana as-is.
- Typical request budget: retrieval+rerank are milliseconds; generation
  dominates (seconds on a local 7B model). The API layer is not the
  bottleneck — size `SCRAG_RATE_LIMIT` against how many concurrent 7B
  generations your GPU/CPU can sustain.

## Logs (JSON lines)

Each request logs one JSON object: `{"ts", "level", "event": "request",
"request_id", "method", "path", "status", "duration_ms", "client"}` —
parseable by Loki/ELK without custom parsing. Correlate with
`X-Request-ID`.

## Incident response

| Symptom                       | Likely cause                    | Action                                       |
|-------------------------------|---------------------------------|----------------------------------------------|
| `/query` → 503                | Ollama unreachable / index missing | check `docker ps` for Ollama; `SCRAG_*` dirs mounted |
| First query slow / timeout    | Model download or cold load     | wait; raise `WORKER_TIMEOUT` if hitting limits |
| 401 despite correct key       | wrong `SCRAG_AUTH_DB` file      | verify volume path contains `auth.sqlite3`   |
| 429 burst                     | per-key budget too low          | raise `SCRAG_RATE_LIMIT` or recreate key with higher limit |
| `database is locked` in logs  | multi-process SQLite write race | single writer process is expected; if >2 workers, reduce to 1 for the auth DB |
