"""Streamlit demo UI for SmartContractRAG.

Interactive, multi-tab front end over the grounded RAG system:

* **Ask** -- answers questions about the public Trail of Bits audit corpus with
  citations and an anti-hallucination grounding badge. Two connection modes:
  REST API (``SCRAG_API_URL``) or an embedded local pipeline (package + Ollama).
  When the local LLM is unreachable the tab degrades to *evidence-first* mode:
  the retrieved chunks and a lexical grounding verdict are shown instead of a
  generated answer.
* **Evals** -- the 14-case golden dataset and the archived metrics of the last
  live run (``docs/LIVE_DEMO.md``), plus an optional live re-run of
  ``scripts/run_eval.py`` when the local stack is available.
* **About** -- architecture, quick start and honest limitations.

Dependencies are deliberately minimal (Streamlit + requests only). The full
embedded mode imports ``smart_contract_rag`` from the repository ``src/`` when
available; without it (e.g. a Hugging Face Space that contains only
``demo_ui/``) the UI still works in REST API mode or explains what is missing.

Run from the repository root::

    streamlit run demo_ui/app.py
"""

from __future__ import annotations

import json
import os
import subprocess
import pandas as pd
import sys
from pathlib import Path
from typing import Any

import requests
import streamlit as st

st.set_page_config(
    page_title="SmartContractRAG — Live Demo",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The project is public on GitHub; the About tab links to it.
REPO_URL = "https://github.com/eLSeR17/smart-contract-rag"

# Default Ollama endpoint documented in the repo (.env.example): the docker
# network hostname. Set OLLAMA_URL when Ollama listens somewhere else.
DEFAULT_OLLAMA_URL = "http://ollama:11434"


def _find_repo_root() -> Path:
    """Locate the repository root by walking up from this file's directory.

    When the demo runs inside the repo (``streamlit run demo_ui/app.py``) the
    root is the directory holding ``data/corpus/manifest.json``. When the app
    is deployed as a Hugging Face Space containing only ``demo_ui/`` there is
    no manifest, so the demo directory itself is used as the root.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "data" / "corpus" / "manifest.json").is_file():
            return candidate
    return here


REPO_ROOT = _find_repo_root()


def _get_setting(name: str, default: str | None = None) -> str | None:
    """Read a config value from the environment, falling back to st.secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        if value := st.secrets.get(name):
            return value
    except Exception:
        # No secrets are configured in this environment.
        return default
    return default


def _resolve_chroma_dir() -> Path:
    """Absolute path of the persistent ChromaDB store.

    ``CHROMA_DIR`` wins when set; otherwise ``<repo>/data/chroma`` is used so
    the demo works regardless of the current working directory.
    """
    explicit = os.getenv("CHROMA_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return REPO_ROOT / "data" / "chroma"


def _resolve_golden_set_path() -> Path:
    """Absolute path of the golden eval dataset (overridable via env)."""
    explicit = os.getenv("SCRAG_GOLDEN_SET")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return REPO_ROOT / "data" / "evals" / "golden_set.json"


def _index_ready() -> bool:
    """True when the persistent ChromaDB index exists on disk."""
    return (_resolve_chroma_dir() / "chroma.sqlite3").is_file()


def _ollama_reachable(timeout: float = 3.0) -> bool:
    """Cheap liveness probe against the Ollama ``/api/tags`` endpoint."""
    base_url = os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# Package availability and connection-mode detection
# ---------------------------------------------------------------------------

_PACKAGE_STATUS: dict[str, Any] | None = None


def _package_status() -> dict[str, Any]:
    """Probe importability of ``smart_contract_rag`` (adding ``src/`` to path).

    Returns ``{"available": bool, "error": str | None}``. The top-level import
    is lightweight; heavy components (embeddings, ChromaDB) load lazily when
    the pipeline is actually built.
    """
    global _PACKAGE_STATUS
    if _PACKAGE_STATUS is not None:
        return _PACKAGE_STATUS

    status: dict[str, Any] = {"available": False, "error": None}
    try:
        import smart_contract_rag  # noqa: F401 (imported for its availability)

        status["available"] = True
    except ModuleNotFoundError as exc:
        src_dir = REPO_ROOT / "src"
        if src_dir.is_dir() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
            try:
                import smart_contract_rag  # noqa: F401 (availability probe)

                status["available"] = True
            except Exception as exc2:
                status["error"] = f"{type(exc2).__name__}: {exc2}"
        else:
            status["error"] = str(exc)
    _PACKAGE_STATUS = status
    return status


def _detect_mode() -> dict[str, Any]:
    """Resolve the Ask-tab connection mode from the environment and the stack."""
    api_url = _get_setting("SCRAG_API_URL")
    if api_url:
        return {
            "mode": "api",
            "label": f"REST API ({api_url})",
            "api_url": api_url.rstrip("/"),
        }

    package = _package_status()
    if not package["available"]:
        return {
            "mode": "no-package",
            "label": "embedded — package not importable",
            "error": package["error"],
        }
    if not _index_ready():
        return {
            "mode": "no-index",
            "label": "embedded — index not found",
        }
    if not _ollama_reachable():
        return {
            "mode": "evidence-first",
            "label": "embedded — evidence-first (Ollama not reachable)",
        }
    return {
        "mode": "embedded",
        "label": "embedded — full pipeline (Ollama reachable)",
    }


# ---------------------------------------------------------------------------
# Pipeline loading (embedded modes only; lazy, heavy imports)
# ---------------------------------------------------------------------------


@st.cache_resource(
    show_spinner="Loading the local pipeline (first run may download the embedder)..."
)
def _load_embedded_pipeline() -> tuple[Any | None, str | None]:
    """Build the full local RAG pipeline; returns ``(pipeline, error)``."""
    try:
        from smart_contract_rag.builders import build_pipeline
        from smart_contract_rag.config import Settings

        settings = Settings.from_env()
        pipeline = build_pipeline(settings, persist_dir=str(_resolve_chroma_dir()))
        return pipeline, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_resource(show_spinner="Loading retrieval components...")
def _load_retrieval_components() -> tuple[Any, Any, Any]:
    """Build retriever, reranker and grounding check for evidence-first mode."""
    from smart_contract_rag.config import Settings
    from smart_contract_rag.generation.grounding import NaiveGroundedTextCheck
    from smart_contract_rag.index.embeddings import (
        CachedEmbeddingBackend,
        SentenceTransformerBackend,
    )
    from smart_contract_rag.index.store import ChromaVectorStore
    from smart_contract_rag.retrieval.retriever import VectorRetriever
    from smart_contract_rag.retrieval.reranker import ScoreFusionReranker

    settings = Settings.from_env()
    embedder = CachedEmbeddingBackend(
        SentenceTransformerBackend(settings.embedding_model)
    )
    store = ChromaVectorStore(persist_dir=str(_resolve_chroma_dir()))
    retriever = VectorRetriever(
        store=store,
        embedder=embedder,
        top_k_default=settings.top_k,
    )
    reranker = ScoreFusionReranker()
    grounding = NaiveGroundedTextCheck()
    return retriever, reranker, grounding


# ---------------------------------------------------------------------------
# Query backends
# ---------------------------------------------------------------------------


def _query_api(query: str, base_url: str, timeout: float = 180.0) -> dict[str, Any]:
    """Query the deployed REST API and return the normalized JSON response."""
    headers = {"Content-Type": "application/json"}
    api_key = _get_setting("SCRAG_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        response = requests.post(
            f"{base_url}/query",
            json={"query": query},
            headers=headers,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach {base_url}/query: {exc}. Check that the API "
            "is running and SCRAG_API_URL is correct."
        ) from exc

    if response.status_code == 401:
        raise RuntimeError(
            "Invalid or missing API key (401). Check SCRAG_API_KEY."
        )
    if response.status_code == 429:
        raise RuntimeError("Rate limit exceeded (429) — wait a moment and retry.")
    if response.status_code == 503:
        detail = ""
        try:
            detail = str(response.json().get("detail", ""))
        except ValueError:
            pass
        raise RuntimeError(f"Pipeline unavailable (503): {detail}")
    if response.status_code >= 400:
        raise RuntimeError(
            f"API error {response.status_code}: {response.text[:200]}"
        )
    return response.json()


def _normalize_response(response: Any) -> dict[str, Any]:
    """Convert a RAGResponse into the same JSON shape the REST API returns."""
    grounding = None
    if response.grounding is not None:
        grounding = {
            "ok": response.grounding.ok,
            "reason": response.grounding.reason,
        }
    sources = [
        {"doc_id": ref.doc_id, "page": ref.page, "section": ref.section}
        for ref in (response.sources or [])
    ]
    retrieved = []
    for r in (response.retrieved or []):
        chunk = r.chunk
        retrieved.append(
            {
                "chunk_id": chunk.id,
                "doc_id": chunk.source.doc_id,
                "score": float(r.score),
                "text": (chunk.text or "")[:200],
            }
        )
    return {
        "query": response.query,
        "answer": response.answer,
        "refused": bool(response.refused),
        "grounding": grounding,
        "sources": sources,
        "retrieved": retrieved,
    }


def _query_embedded(query: str) -> dict[str, Any]:
    """Run the full local pipeline (repository package + Ollama)."""
    pipeline, error = _load_embedded_pipeline()
    if pipeline is None:
        raise RuntimeError(f"The local pipeline could not be built: {error}")
    try:
        response = pipeline.answer(query)
    except Exception as exc:
        raise RuntimeError(
            f"The local pipeline failed: {type(exc).__name__}: {exc}"
        ) from exc
    return _normalize_response(response)


def _query_evidence_first(query: str) -> dict[str, Any]:
    """Retrieve and ground a question without calling the LLM.

    Activated when Ollama is unreachable: the tab shows the retrieved evidence
    with a lexical grounding verdict on the question instead of a generated
    answer.
    """
    from smart_contract_rag.config import Settings

    settings = Settings.from_env()
    try:
        retriever, reranker, grounding = _load_retrieval_components()
        raw = retriever.retrieve(query, top_k=settings.top_k)
        reranked = reranker.rerank(query, raw, top_k=settings.top_k)
    except Exception as exc:
        raise RuntimeError(
            f"Evidence retrieval failed: {type(exc).__name__}: {exc}"
        ) from exc

    if not reranked:
        return {
            "query": query,
            "answer": "I DON'T KNOW — no relevant sources were found.",
            "refused": True,
            "grounding": None,
            "sources": [],
            "retrieved": [],
            "evidence_first": True,
        }

    grounded_ok, ratios = grounding.check(
        query, reranked, threshold=settings.grounding_threshold
    )
    view: dict[str, Any] = {
        "query": query,
        "answer": "",
        "refused": False,
        "grounding": {
            "ok": bool(grounded_ok),
            "reason": "grounded" if grounded_ok else "ungrounded",
            "grounded_ratio": float(ratios.get("grounded_ratio", 0.0)),
        },
        "evidence_first": True,
        "sources": [],
        "retrieved": [],
    }
    seen: set[tuple[str, int | None]] = set()
    for r in reranked:
        ref = r.chunk.source
        key = (ref.doc_id, ref.page)
        if key not in seen:
            seen.add(key)
            view["sources"].append(
                {"doc_id": ref.doc_id, "page": ref.page, "section": ref.section}
            )
        view["retrieved"].append(
            {
                "chunk_id": r.chunk.id,
                "doc_id": ref.doc_id,
                "score": float(r.score),
                "text": (r.chunk.text or "")[:200],
            }
        )
    return view


def _run_query(query: str, mode_info: dict[str, Any]) -> dict[str, Any]:
    """Route a question to the active backend and return a normalized view."""
    mode = mode_info["mode"]
    if mode == "api":
        return _query_api(query, mode_info["api_url"])
    if mode == "embedded":
        return _query_embedded(query)
    if mode == "evidence-first":
        return _query_evidence_first(query)
    raise RuntimeError(f"Unsupported mode: {mode}")


# ---------------------------------------------------------------------------
# Answer rendering
# ---------------------------------------------------------------------------


def _render_answer(view: dict[str, Any]) -> None:
    """Render the answer, refusal, grounding badge, sources and chunks."""
    st.divider()

    if view.get("evidence_first"):
        st.warning(
            "**Evidence-first mode: the local LLM (Ollama) is not reachable.** "
            "Answer generation is disabled; below is the raw retrieval plus a "
            "lexical grounding verdict on the question."
        )

    if view["refused"]:
        message = (
            view["answer"]
            or "The query was rejected by the input guardrail."
        )
        st.warning(f"**Refused:** {message}")
    elif view["answer"].strip():
        st.markdown(view["answer"])
    else:
        st.info("No answer to display.")

    _render_grounding(view)
    _render_sources(view)
    _render_retrieved(view)


def _render_grounding(view: dict[str, Any]) -> None:
    """Render the anti-hallucination grounding badge."""
    grounding = view.get("grounding")
    if not grounding:
        st.caption("No grounding check was performed.")
        return
    ok = grounding.get("ok", False)
    reason = grounding.get("reason", "unknown")
    extra = ""
    if "grounded_ratio" in grounding:
        extra = f" · ratio {float(grounding['grounded_ratio']):.3f}"
    if ok:
        st.success(f"**Grounding:** OK ({reason}){extra}")
    else:
        st.error(f"**Grounding:** NOT OK ({reason}){extra}")


def _render_sources(view: dict[str, Any]) -> None:
    """Render the citation list (doc_id + page + optional section)."""
    sources = view.get("sources") or []
    if not sources:
        return
    st.markdown("**Sources**")
    for source in sources:
        page = source.get("page")
        section = source.get("section")
        location = f"page {page}" if page is not None else "page n/a"
        if section:
            location = f"{location} · {section}"
        st.markdown(f"- `{source.get('doc_id')}` — {location}")


def _render_retrieved(view: dict[str, Any]) -> None:
    """Render the top-k retrieved chunks in an expandable section."""
    retrieved = view.get("retrieved") or []
    if not retrieved:
        return
    with st.expander(f"Retrieved chunks (top {len(retrieved)})"):
        for index, chunk in enumerate(retrieved, start=1):
            score = float(chunk.get("score", 0.0))
            excerpt = (chunk.get("text") or "")[:240]
            st.markdown(
                f"**{index}.** `{chunk.get('chunk_id')}` · doc "
                f"`{chunk.get('doc_id')}` · score {score:.4f}"
            )
            st.text(excerpt or "-")


# ---------------------------------------------------------------------------
# Ask tab
# ---------------------------------------------------------------------------


def _render_ask_tab() -> None:
    """Ask tab: question input, connection-mode banner and answer rendering."""
    st.header("Ask the audit corpus")
    st.caption(
        "Grounded, cited answers over 10 public Trail of Bits smart-contract "
        "audit reports. Ungrounded answers are refused by design."
    )

    mode_info = _detect_mode()
    st.caption(f"Connection: **{mode_info['label']}**")

    mode = mode_info["mode"]
    if mode == "no-package":
        st.info(
            "The `smart_contract_rag` package is not importable in this "
            "environment. Install it from the repository root "
            "(`pip install -r requirements.txt` or `pip install -e .`) — or "
            "use **API mode** by setting `SCRAG_API_URL`."
        )
        return
    if mode == "no-index":
        st.info(
            "The local vector index was not found in "
            f"`{_resolve_chroma_dir()}`. Run `python scripts/fetch_corpus.py "
            "&& python scripts/index_corpus.py` first, then reload this tab."
        )
        return

    query = st.text_input(
        "Your question",
        placeholder=(
            "e.g. What reentrancy issues does the Balancer V2 report describe?"
        ),
    )
    if st.button("Ask", type="primary"):
        if not query.strip():
            st.warning("Please enter a question first.")
            return
        if mode == "embedded":
            spinner = (
                "Answering with the local LLM — CPU inference can take "
                "~2 minutes."
            )
        elif mode == "evidence-first":
            spinner = "Retrieving evidence and checking grounding..."
        else:
            spinner = "Querying the REST API..."
        with st.spinner(spinner):
            try:
                view = _run_query(query.strip(), mode_info)
            except RuntimeError as exc:
                st.error(f"Query failed: {exc}")
                return
        _render_answer(view)


# ---------------------------------------------------------------------------
# Evals tab
# ---------------------------------------------------------------------------

# Archived metrics of the last live eval run, after the KI-01 fix
# (index/dataset doc-id mismatch). Source: docs/KNOWN_ISSUES.md
# (KI-01 re-eval, data/demo/eval_report_20260908_KI01-resolved.md, fix f543845).
ARCHIVED_METRICS: list[dict[str, Any]] = [
    {
        "metric": "hallucination_rate",
        "value": 0.0,
        "comment": "lower is better — the anti-hallucination contract",
    },
    {
        "metric": "answer_rate",
        "value": 0.75,
        "comment": "fraction of answerable cases answered",
    },
    {
        "metric": "correct_refusal_rate",
        "value": 0.7857,
        "comment": "unsupported/trap cases correctly refused",
    },
    {
        "metric": "faithfulness",
        "value": 0.6667,
        "comment": "answer supported by the retrieved evidence",
    },
    {
        "metric": "answer_relevance",
        "value": 0.6444,
        "comment": "answer addresses the question",
    },
    {
        "metric": "citation_accuracy",
        "value": 0.4444,
        "comment": "citations resolve to the right document",
    },
    {
        "metric": "context_precision",
        "value": 0.1667,
        "comment": "fraction of retrieved chunks that are relevant",
    },
    {
        "metric": "context_recall",
        "value": 0.5,
        "comment": "fraction of relevant evidence that is retrieved",
    },
]


def _load_golden_set() -> list[dict[str, Any]]:
    """Load the golden eval dataset; returns [] when the file is absent."""
    path = _resolve_golden_set_path()
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, list):
        return []
    return [case for case in payload if isinstance(case, dict)]


def _stack_ready_for_eval() -> bool:
    """True when a live eval run can work: package + index + Ollama."""
    package = _package_status()
    return bool(package["available"]) and _index_ready() and _ollama_reachable()


def _metric_value(payload: Any) -> Any:
    """Pull the scalar value out of an EvalMetricSet entry."""
    if isinstance(payload, dict):
        return payload.get("value", payload)
    return payload


def _extract_json(text: str) -> Any:
    """Parse the first JSON object embedded in a text payload."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in the command output")
    return json.loads(text[start:])


def _render_eval_report(report: dict[str, Any], *, exit_code: int) -> None:
    """Render the EvalReport JSON produced by scripts/run_eval.py."""
    status = str(report.get("status", "?"))
    message = f"Verdict: **{status}** (exit code {exit_code})"
    if status == "PASS":
        st.success(message)
    elif status == "WARN":
        st.warning(message)
    else:
        st.error(message)

    metrics = report.get("metrics") or {}
    metric_rows = [
        {"metric": name, "value": _metric_value(payload)}
        for name, payload in sorted(metrics.items())
    ]
    if metric_rows:
        st.markdown("**Aggregate metrics**")
        st.dataframe(metric_rows, hide_index=True)

    cases = report.get("cases") or []
    if cases:
        st.markdown(f"**Per-case** ({len(cases)})")
        st.dataframe(
            [
                {
                    "id": c.get("case_id", ""),
                    "topic": c.get("topic", ""),
                    "answered": c.get("answered"),
                    "refused": c.get("refused"),
                    "faithfulness": c.get("faithfulness"),
                    "relevance": c.get("relevance"),
                    "citation": c.get("citation_accuracy"),
                    "recall": c.get("context_recall"),
                    "hallucination": c.get("hallucination"),
                }
                for c in cases
            ],
            hide_index=True,
        )


def _run_eval_live() -> None:
    """Run scripts/run_eval.py in a subprocess and render its JSON report."""
    script = REPO_ROOT / "scripts" / "run_eval.py"
    if not script.is_file():
        st.error(
            f"`{script}` not found. The live eval runs from the repository "
            "root; deploy the full repo or use API mode."
        )
        return
    command = [sys.executable, str(script), "--json"]
    with st.spinner("Running the 14-case eval with the LLM judge (up to 300 s)..."):
        try:
            proc = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            st.error(
                "The live eval timed out after 300 s — the local model is "
                "likely busy. Retry when the host is idle."
            )
            return
    try:
        report = _extract_json(proc.stdout)
    except ValueError as exc:
        st.error(f"Could not parse the eval report: {exc}")
        if proc.stderr.strip():
            with st.expander("Command stderr"):
                st.code(proc.stderr)
        return
    _render_eval_report(report, exit_code=proc.returncode)


def _render_evals_tab() -> None:
    """Evals tab: golden dataset, archived metrics and optional live re-run."""
    st.header("Evaluation")

    st.subheader("Golden dataset")
    golden = _load_golden_set()
    if golden:
        rows = [
            {
                "id": case.get("id", ""),
                "topic": case.get("topic", ""),
                "relevant_doc_id": case.get("relevant_doc_id") or "(trap)",
                "expect_answer": str(case.get("expect_answer", True)),
            }
            for case in golden
        ]
        st.dataframe(rows, hide_index=True)
        st.caption(
            "14 curated cases across the vulnerability topics of the corpus; "
            "the two trap questions (ev-013, ev-014) must be refused."
        )
    else:
        st.info(
            f"Golden dataset not found at `{_resolve_golden_set_path()}`. "
            "Run the demo from the repository root or upload `data/evals/`."
        )

    st.subheader("Last live run — archived metrics")
    st.caption("archived — docs/KNOWN_ISSUES.md · post KI-01 re-eval (f543845)")
    st.dataframe(
        [
            {"metric": m["metric"], "value": m["value"], "comment": m["comment"]}
            for m in ARCHIVED_METRICS
        ],
        hide_index=True,
    )
    st.caption(
        "The archived verdict is **FAIL**: retrieval/citation metrics sit "
        "below the regression guard. The structural harness artefact (KI-01) "
        "is fixed and these metrics now measure real retrieval quality — see "
        "docs/KNOWN_ISSUES.md."
    )

    st.subheader("Run the eval live")
    if _stack_ready_for_eval():
        if st.button("Run eval live", type="primary"):
            _run_eval_live()
    else:
        st.info(
            "The live eval requires the local stack: Ollama reachable at "
            "`OLLAMA_URL` + an indexed corpus (`data/chroma/`)."
        )


# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------


def _render_about_tab() -> None:
    """About tab: architecture, quick start and honest limitations."""
    st.header("About")

    st.markdown(
        f"**SmartContractRAG** is a production-grade grounded RAG system for "
        f"smart-contract audit reports running entirely on local models. "
        f"Source, tests and the eval harness: [{REPO_URL}]({REPO_URL})."
    )

    st.subheader("Architecture")
    st.markdown(
        "```text\n"
        "query -------------------------------> input guardrail\n"
        "                                              |\n"
        "                                              v\n"
        "                   retriever (embeddings + ChromaDB, top-k)\n"
        "                                              |\n"
        "                                              v\n"
        "                     reranker (vector + lexical fusion)\n"
        "                                              |\n"
        "                                              v\n"
        "                    generator (local LLM via Ollama)\n"
        "                                              |\n"
        "                                              v\n"
        "               grounding check + output guardrail\n"
        "                                              |\n"
        "                                              v\n"
        "    grounded answer with citations  OR  \"I DON'T KNOW\"\n"
        "\n"
        "offline build:\n"
        "  manifest.json -> fetch_corpus.py -> PDFs -> extractor -> chunker\n"
        "                                                      \\\n"
        "                                                       +> embedder -> ChromaDB\n"
        "```"
    )

    st.subheader("Run it locally")
    st.markdown("**Option 1 — full stack via Docker Compose (recommended):**")
    st.code(
        "# 1. Start the REST API with the local model in the docker network\n"
        "docker compose -f docker-compose.prod.yml up -d --build\n"
        "\n"
        "# 2. Only if SCRAG_AUTH=api_key (the prod compose default): create a key\n"
        "#    python scripts/create_api_key.py your-name 60\n"
        "\n"
        "# 3. Point the demo at the API and install its deps\n"
        "export SCRAG_API_URL=http://localhost:8000\n"
        "# export SCRAG_API_KEY=<secret>    # api_key mode only\n"
        "pip install -r demo_ui/requirements.txt\n"
        "\n"
        "# 4. Launch\n"
        "streamlit run demo_ui/app.py\n",
        language="bash",
    )
    st.markdown("**Option 2 — embedded mode (no container, local Ollama):**")
    st.code(
        "python scripts/fetch_corpus.py\n"
        "python scripts/index_corpus.py\n"
        "pip install -r requirements.txt             # repo root: pipeline deps\n"
        "pip install -r demo_ui/requirements.txt     # streamlit + requests\n"
        "# export OLLAMA_URL=http://<ollama-host>:11434  # docker-network name by default\n"
        "streamlit run demo_ui/app.py\n",
        language="bash",
    )

    st.subheader("How this demo connects")
    st.markdown(
        "- **REST API mode** — set `SCRAG_API_URL` (optional `SCRAG_API_KEY`) "
        "and the Ask tab talks to the deployed FastAPI server. Zero heavy "
        "dependencies.\n"
        "- **Embedded mode** — without `SCRAG_API_URL`, the demo imports "
        "`smart_contract_rag` from the repo and runs the full pipeline locally.\n"
        "- **Evidence-first mode** — embedded, but Ollama is unreachable: "
        "retrieval and grounding still run; only answer generation is skipped.\n"
        "- On **hosted platforms (Streamlit Community Cloud, Hugging Face "
        "Spaces)** there is no route back to your local Ollama — the app "
        "runs in evidence-first mode (full REST API mode when "
        "`SCRAG_API_URL` is set). Step-by-step in `demo_ui/README.md`."
    )

    st.subheader("Honest limitations")
    st.markdown(
        "- **Demo corpus**: 10 public Trail of Bits audit reports — a "
        "demonstration set, not an exhaustive security database.\n"
        "- **Local latency**: a 7B model on CPU answers in tens of seconds; "
        "a GPU would be used in production.\n"
        "- **Lexical grounding** is a heuristic: it detects *absence* of "
        "evidence well, but is not a full semantic entailment check.\n"
        "- **Not an audit substitute**: outputs are research assistance only "
        "and never replace a professional smart-contract audit."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Analytics tab
# ---------------------------------------------------------------------------

_NAMES = {
    "0x-protocol": "0x Protocol",
    "aave-v3": "Aave V3",
    "balancer-managedpool": "Balancer Managed Pool",
    "balancerv2": "Balancer V2",
    "beanstalk-security": "Beanstalk",
    "fraxlend-fraxferry": "FraxLend / FraxFerry",
    "increment-security": "Increment",
    "maplefinance-v1": "Maple Finance V1",
    "optimism-security": "Optimism L2",
    "reserve-security": "Reserve Protocol",
}


@st.cache_data
def _load_analytics() -> dict | None:
    """Read the analytics JSON produced by scripts/extract_analytics.py."""
    path = Path(__file__).resolve().parent.parent / "data" / "analytics" / "analytics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _render_analytics_tab() -> None:
    """Analytics tab: deterministic statistics extracted from the audit corpus."""
    st.header("Corpus analytics")
    st.caption(
        "Deterministic, no-LLM extraction from the 10 Trail of Bits reports "
        "embedded in the corpus. Numbers are cross-verified against the reports' "
        "own declared totals."
    )
    data = _load_analytics()
    if data is None:
        st.info("Analytics data not generated yet -- run scripts/extract_analytics.py.")
        return

    s = data["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reports", s["reports"])
    c2.metric("Findings", s["findings_total"])
    c3.metric("High severity", s["severity"]["High"])
    c4.metric("Informational", s["severity"]["Informational"])

    rows = []
    for r in data["reports"]:
        sev = r["severity"]
        rows.append({
            "Protocol": _NAMES.get(r["doc_id"], r["doc_id"]),
            "Findings": r["findings"]["severity_labels"],
            "High": sev["High"],
            "Medium": sev["Medium"],
            "Low": sev["Low"],
            "Informational": sev["Informational"],
            "Undetermined": sev["Undetermined"],
            "Date": r["date"] or "*",
            "Top categories": ", ".join(t for t, _ in r["types"][:2]) or "*",
            "Tools": ", ".join(r["tools"]) or "*",
        })
    df = pd.DataFrame(rows)

    st.subheader("Findings per protocol")
    st.bar_chart(df[["Protocol", "Findings"]].set_index("Protocol"), horizontal=True)

    st.subheader("Severity distribution")
    st.bar_chart(df.set_index("Protocol")[["High", "Medium", "Low", "Informational", "Undetermined"]])

    st.subheader("Top vulnerability categories (author-provided types)")
    cats = s["types"]
    st.bar_chart(
        pd.DataFrame([{"Category": t, "Count": n} for t, n in cats[:8]]).set_index("Category"),
        horizontal=True,
    )

    st.subheader("Details per report")
    st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("How was this data generated?"):
        st.markdown(
            "- Extracted deterministically by scripts/extract_analytics.py from the "
            "corpus chunks (no LLM, no network, fully reproducible).\n"
            "- Per-finding Severity: and Finding ID: markers are two independent "
            "counters; they agree on every report (verified).\n"
            "- Type: values are the reports' own CATEGORY BREAKDOWN labels, not "
            "LLM-generated guesses.\n"
            "- Full verification: run python scripts/extract_analytics.py --verify."
        )


def main() -> None:
    """Render the four-tab demo application."""
    tab_ask, tab_evals, tab_analytics, tab_about = st.tabs(
        ["Ask", "Evals", "Analytics", "About"]
    )
    with tab_ask:
        _render_ask_tab()
    with tab_evals:
        _render_evals_tab()
    with tab_analytics:
        _render_analytics_tab()
    with tab_about:
        _render_about_tab()


if __name__ == "__main__":
    main()