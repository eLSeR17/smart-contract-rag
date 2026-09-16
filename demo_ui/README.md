# SmartContractRAG — Live Demo UI

> A live, interactive front end for the
> [SmartContractRAG](https://github.com/eLSeR17/smart-contract-rag) grounded
> RAG system — deployable for free on **Streamlit Community Cloud**
> (Hugging Face Spaces also works, with a paid PRO plan).

## What this is

A small Streamlit app (three tabs: **Ask**, **Evals**, **About**) that shows
the RAG system answering real questions over the public Trail of Bits audit
corpus with citations, an anti-hallucination grounding badge, and the eval
dashboard. It is intentionally lightweight — `streamlit` + `requests` are the
only hard dependencies. The app connects in one of three ways:

| Mode | Trigger | Behaviour |
|------|---------|-----------|
| REST API | `SCRAG_API_URL` set | `POST /query` to the deployed FastAPI server (full generation; requires hosting the API) |
| Embedded full | no `SCRAG_API_URL`, package + index + Ollama reachable | runs the real pipeline in-process |
| Evidence-first | package + index, but Ollama unreachable | shows retrieval + a lexical grounding verdict, no generation |

## Files

```
demo_ui/
├── app.py               # Streamlit entry point (three tabs)
├── requirements.txt     # streamlit + requests (demo-only deps)
├── .env.example         # documented configuration placeholders
├── .streamlit/
│   └── config.toml      # theme + server settings
└── README.md            # this file
```

All configuration is read from environment variables with sensible defaults
(`env` first, then Streamlit secrets). No secrets are ever committed.

## Run locally — full stack via Docker Compose (recommended)

```bash
# 0. Prerequisites: Docker running with Ollama on the `docker_default`
#    network, and the `qwen2.5-coder:7b` model pulled (see the repo README).

# 1. Start the REST API with the local model in the docker network
docker compose -f docker-compose.prod.yml up -d --build

# 2. Only if SCRAG_AUTH=api_key is enabled (the prod compose default): create a key
python scripts/create_api_key.py your-name 60

# 3. Point the demo at the API and install its deps
export SCRAG_API_URL=http://localhost:8000
export SCRAG_API_KEY=<secret-from-step-2>      # api_key mode only
pip install -r demo_ui/requirements.txt

# 4. Launch
streamlit run demo_ui/app.py
```

Open the printed URL and use the **Ask** tab. The **Evals** tab's
"Run eval live" button needs the local stack (Ollama + index) — it is
disabled in API mode.

## Run locally — embedded mode (no container)

Requires the full repository dependencies and a reachable Ollama:

```bash
python scripts/fetch_corpus.py                # download the 10 audit PDFs
python scripts/index_corpus.py                # build data/chroma/ (546 chunks)
pip install -r requirements.txt               # repo root: full pipeline deps
pip install -r demo_ui/requirements.txt       # demo deps (streamlit + requests)
export OLLAMA_URL=http://<reachable-ollama>:11434   # docker network name by default
streamlit run demo_ui/app.py
```

Without `SCRAG_API_URL` the app auto-detects its mode: full embedded mode when
Ollama answers the liveness probe, otherwise *evidence-first* mode (retrieval
+ grounding verdict on the question, no generation).

## Deploy on Streamlit Community Cloud (free, recommended)

> 2026 note: Hugging Face Spaces no longer offers a free compute tier (Gradio
> and Docker Spaces require a paid PRO plan; only Static Spaces are free). The
> free hosted option for this Streamlit app is **Streamlit Community Cloud** — an
> official Streamlit service that connects to GitHub, deploys in minutes and
> re-deploys on every `git push`.

Honest technical note first: no free hosted platform can route back to your
local machine, so none of them can reach your local Ollama. On hosted
platforms this app therefore runs in **evidence-first mode** — retrieval plus a
lexical grounding verdict over the bundled index, without LLM generation. The
generated answers that the full pipeline produces are documented with real
outputs in the repository (`docs/LIVE_DEMO.md`).

Steps:

1. Sign in at <https://share.streamlit.io> with your GitHub account (you must
   have admin access to the repository).
2. **Create app** → *"Yup, I have an app"*.
3. Fill in: repository `eLSeR17/smart-contract-rag`, branch `main`, main file
   path `demo_ui/app.py`.
4. (Optional) *Advanced settings*: Python 3.12. Do **not** set `SCRAG_API_URL`:
   without it the app auto-detects *evidence-first* mode (package + index
   present, Ollama unreachable) and works with zero configuration.
5. **Deploy**. You get a URL on `*.streamlit.app` (custom subdomain available
   in the app settings).
6. Every `git push` to `main` redeploys the app automatically.

The vector index `data/chroma/` is committed to the repository (force-added;
it is git-ignored otherwise) so Community Cloud can serve evidence-first mode
without a build step. First query downloads the embedding model (~90 MB) and
answers in tens of seconds; subsequent ones are fast.

### Alternative: Hugging Face Spaces (paid PRO)

HF Spaces run Gradio/Docker apps on compute plans behind PRO (~9 $/month);
Static Spaces are free but cannot execute Python. If you already have PRO:

1. Create a Space with SDK **Docker** (or Gradio) and upload this repository.
2. Set the Space app file to `demo_ui/app.py`.
3. Same evidence-first behaviour (no route back to a local Ollama).

### Full REST API mode (only when you host the API yourself)

If you expose the FastAPI server somewhere with internet (`SCRAG_API_URL` +
optional `SCRAG_API_KEY`), the same app switches to full REST API mode with
generated, grounded answers. See `docs/API.md` and `docker-compose.prod.yml`.

### After deploying

Verify the public URL, then link it from the repository-root README (the
orchestrator does that in a later pass).

## Honest limitations

- **Corpus**: the manifest points at 10 public Trail of Bits reviews — a
  demonstration corpus, not an exhaustive security database.
- **Latency**: a 7B model on CPU answers in tens of seconds; the HF Spaces
  free tier runs no LLM at all (see Option B).
- **Grounding heuristic**: the default anti-hallucination check is lexical
  (token overlap) — it detects *absence* of evidence well, but is not a
  semantic entailment check.
- **Not an audit substitute**: outputs are research assistance, never a
  professional smart-contract audit.
- **Archived eval verdict**: the last live run is **FAIL** on
  retrieval/citation metrics, because the harness doc-id bug (KI-01) has been
  fixed and the metrics now measure real retrieval quality, which is below the
  regression guard on 5 of 14 topics. Full record: `docs/LIVE_DEMO.md` and
  `docs/KNOWN_ISSUES.md`.

---

MIT licensed — part of the
[smart-contract-rag](https://github.com/eLSeR17/smart-contract-rag)
repository.