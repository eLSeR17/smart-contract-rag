# Security

## Reporting a vulnerability

This is a public portfolio project. If you find a security issue, please open a
private report via **GitHub Security Advisories** →
[Report a vulnerability](https://github.com/eLSeR17/smart-contract-rag/security/advisories/new)
instead of a public issue. Once the report is triaged we will coordinate the
disclosure and a fix.

## Dependabot alert dispositions

Alerts are triaged against **how this repository actually runs ChromaDB**, not
against the full product surface. ChromaDB is used here only as an **embedded,
in-process store** (`chromadb.PersistentClient`, `sqlite`/local files): the
project never runs a Chroma **HTTP server**, never creates tenants or database
scopes, has no authentication boundary to protect, and never sets
`trust_remote_code`. The deployed demo (Streamlit Community Cloud) only *reads*
the bundled index; the optional FastAPI server also uses the client embedded.

| Alert | GHSA / CVE | Affects this repo? | Disposition |
|---|---|---|---|
| ChromaDB code injection (critical) – malicious model repo + `trust_remote_code` via the tenant/collection **HTTP API** | [GHSA-36p7-vc44-83pf](https://github.com/advisories/GHSA-36p7-vc44-83pf) / CVE-2026-45833 | **No** – the affected path is the server's `UPDATE_COLLECTION` endpoint with remote code loading; not present in embedded usage. No upstream fix released (all versions ≤ 1.5.9). | Dismissed: `not_used` (2026-09-16) |
| ChromaDB cross-tenant authorization bypass (high) – any authenticated user accesses any tenant's collections via the **server API** | [GHSA-2wm9-hf6c-p5cr](https://github.com/advisories/GHSA-2wm9-hf6c-p5cr) / CVE-2026-45830 | **No** – requires the multi-tenant server with authentication; this project has a single embedded collection with no server/auth model. No upstream fix released (all versions ≤ 1.5.9). | Dismissed: `not_used` (2026-09-16) |

Both advisories are tracked upstream ([chroma-core/chroma#7588](https://github.com/chroma-core/chroma/issues/7588),
PR [#7602](https://github.com/chroma-core/chroma/pull/7602)); if a patched
release lands and this project's usage grows (e.g. a hosted server mode), the
disposition will be revisited.
