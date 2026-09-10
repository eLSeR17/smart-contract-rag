# SmartContractRAG — HTTP API

The repository ships a FastAPI server (`src/smart_contract_rag/api.py`) that
exposes the grounded RAG pipeline over HTTP, with API-key auth, per-key rate
limiting, JSON structured logs and Prometheus metrics.

Run it locally:

```bash
pip install -r requirements.txt
uvicorn smart_contract_rag.api:app --port 8000
# or with the production image:
docker compose -f docker-compose.prod.yml up -d --build
```

## Authentication

Two modes, selected with `SCRAG_AUTH`:

| Mode      | Behaviour                                                        |
|-----------|------------------------------------------------------------------|
| `none`    | (default) open endpoint — local dev only.                        |
| `api_key` | every `POST /query` needs `X-API-Key: <secret>`. 401 otherwise.  |

Create a key (plaintext is shown **once**):

```bash
python scripts/create_api_key.py "ci-bot" 60
# key_id : scrag_AbC...   secret : <32 random url-safe chars>
```

Keys are stored as SHA-256 hashes (`auth.sqlite3`). Revocation: use the
store in a one-liner or delete the row after auditing.

## Endpoints

### `POST /query`

Question → grounded answer with verifiable citations.

```bash
curl -s http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: <secret>' \            # only in api_key mode
  -d '{"query": "What reentrancy issues does the Balancer report describe?"}'
```

```json
{
  "query": "What reentrancy issues does the Balancer report describe?",
  "answer": "The report describes a reentrancy risk in managed pool joins...",
  "refused": false,
  "grounding": { "ok": true, "reason": "grounded" },
  "sources": [
    { "doc_id": "2022-10-balancerlabs-managedpoolsmartcontracts-securityreview", "page": 12, "section": null }
  ],
  "retrieved": [
    { "chunk_id": "2022-10-...::chunk-0042", "doc_id": "2022-10-...", "score": 0.87, "text": "The joinPool flow updates balances..." }
  ]
}
```

| Field       | Meaning                                                          |
|-------------|------------------------------------------------------------------|
| `answer`    | Final guardrailed answer (may be empty when `refused`).          |
| `refused`   | `true` when the query failed a guardrail or no sources matched.  |
| `grounding` | Anti-hallucination check result (`ok`/`reason`).                 |
| `sources`   | Citations — `doc_id` matches the corpus manifest.                |
| `retrieved` | Top chunks used with scores (text truncated to 200 chars).       |

### `GET /health`

```json
{ "status": "ok", "auth_mode": "api_key", "pipeline_ready": true, "pipeline_error": null }
```

`status` is `degraded` (and `pipeline_ready` `false`) when the index or the
model is not reachable — the server stays up to report it.

### `GET /metrics`

Prometheus text format, no external dependency:

```
smart_contract_rag_http_requests_total 12
smart_contract_rag_http_status_200_total 11
smart_contract_rag_http_status_401_total 1
smart_contract_rag_query_latency_s_count 10
smart_contract_rag_query_latency_s_quantile{quantile="0.99"} 2.410
```

## Errors

| Code | Meaning                                                        |
|------|----------------------------------------------------------------|
| 400  | Malformed JSON / missing field.                                |
| 401  | Missing or invalid `X-API-Key` (api_key mode).                 |
| 422  | Validation error (e.g. empty `query`, too long).               |
| 429  | Per-key rate limit exceeded (`SCRAG_RATE_LIMIT`, token bucket).|
| 503  | Pipeline unavailable (index/model not ready) — retry later.    |

Every response carries `X-Request-ID` (echoed from the request header, or
generated) for log correlation.

## Environment variables

| Variable           | Default          | Meaning                                  |
|--------------------|------------------|------------------------------------------|
| `SCRAG_AUTH`       | `none`           | `none` or `api_key`                      |
| `SCRAG_AUTH_DB`    | `auth.sqlite3`   | SQLite file for key store                |
| `SCRAG_RATE_LIMIT` | `60`             | requests/minute/key                      |
| `SCRAG_MAX_QUESTION`| `4000`          | max question length (chars)              |
| `OLLAMA_URL`, `OLLAMA_MODEL`, `EMBEDDING_MODEL`, `GROUNDING_*`, `CHROMA_DIR`, `CORPUS_DIR`, `TOP_K` | — | pipeline settings, see `.env.example` |
