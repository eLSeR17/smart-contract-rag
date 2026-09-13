# SmartContractRAG — Live Demo UI

> A live, interactive front end for the
> [SmartContractRAG](https://github.com/eLSeR17/smart-contract-rag) grounded
> RAG system — deployable for free on Hugging Face Spaces.

## What this is

A small Streamlit app (three tabs: **Ask**, **Evals**, **About**) that shows
the RAG system answering real questions over the public Trail of Bits audit
corpus with citations, an anti-hallucination grounding badge, and the eval
dashboard. It is intentionally lightweight — `streamlit` + `requests` are the
only hard dependencies. The app connects in one of three ways:

| Mode | Trigger | Behaviour |
|------|---------|-----------|
| REST API | `SCRAG_API_URL` set | `POST /query` to the deployed FastAPI server (recommended; the only mode that works on HF Spaces) |
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

## Deploy on Hugging Face Spaces (free)

Honest technical note first: **a free HF Space has no network route back to
your local machine**, so it cannot reach your local Ollama. Two professional
deployment shapes exist — pick one.

### Option A — the Space talks to your hosted API (recommended)

Deploy the API somewhere with internet (a VPS/VM with Ollama, or any Docker
host), following `docs/API.md` and `docker-compose.prod.yml`:

```bash
docker compose -f docker-compose.prod.yml up -d --build   # on that host
python scripts/create_api_key.py space-key 60             # if SCRAG_AUTH=api_key
```

Then create the Space:

1. Create an account at <https://huggingface.co> and log in.
2. **New Space** → name it `smart-contract-rag-demo` → SDK: **Streamlit**
   (simplest: it auto-reads `app.py`, `requirements.txt` and `.streamlit/`,
   which is exactly this folder's layout), or **Docker** (most control when
   you need custom system packages or a pinned Python base image — you write
   a `Dockerfile` that installs `demo_ui/requirements.txt` and runs
   `streamlit run app.py --server.port 7860`).
3. Upload the contents of `demo_ui/` (the 5 files/folders above) as the Space
   root — via the web uploader or a git push of this folder.
4. Go to **Settings → Variables and secrets** and add:
   - `SCRAG_API_URL` → `http://<your-host>:8000`
   - `SCRAG_API_KEY` → the key created above (only if auth is on)
5. The Space rebuilds and you get a public URL:
   `https://huggingface.co/spaces/<you>/smart-contract-rag-demo`.

The demo then works end to end: grounded Ask answers, the Evals golden table
and the About tab. This is the mode the UI was designed for on Spaces.

### Option B — fully in the Space, no LLM (evidence-first)

If you do not want to host the API, the Space can run the *retrieval only*
side of the pipeline — no LLM — by bundling the whole repository instead of
just `demo_ui/`:

1. Upload the **entire repository** as the Space root (`src/`, `data/`,
   `demo_ui/` all present).
2. Make sure the Space-root `requirements.txt` unions both sets of deps:
   append `streamlit` and `requests` to the repo-root `requirements.txt`
   (a one-line change).
3. Set the Space app file to `demo_ui/app.py` in the Space settings.
4. (Recommended) build `data/chroma/` locally first and upload it with the
   repo — runtime indexing in the Space is slow and downloads the embedder
   at first use. Note `data/chroma/` is gitignored, so upload it via the web
   UI or `git add -f`.
5. Leave `OLLAMA_URL` unset (or point it at an unreachable host).

Result: the **Ask** tab runs in *evidence-first* mode — retrieval, chunk
scores and the grounding badge work; only answer generation is skipped,
because no LLM is reachable in the Space. The UI states this clearly
("evidence-first mode: LLM not reachable").

Alternative (Docker runtime): a Space with a custom `Dockerfile` can install
the heavy pipeline deps at build time and pre-fetch the embedder — slower to
build, but the most reproducible shape for evidence-first.

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