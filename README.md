# SmartContractRAG — Grounded RAG for Smart-Contract Audit Reports

[![CI](https://github.com/eLSeR17/smart-contract-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/eLSeR17/smart-contract-rag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **What it demonstrates**: a production-grade **Retrieval-Augmented Generation
> (RAG)** system with **grounded, citable answers** and an **anti-hallucination
> layer** — built entirely on **local models** (Ollama + sentence-transformers),
> with a deterministic test suite that runs **without network or a GPU**.

## Problem

Language models confidently answer questions even when they lack the facts. For
a domain like **smart-contract security**, that is dangerous: a hallucinated
"the function is safe because it uses X" can mislead a reviewer into signing off
on a vulnerable codebase. RAG alone does not solve this — retrieval can still
surface *plausible-looking* evidence, and the model can drift from it.

## Solution

SmartContractRAG answers questions about **public smart-contract audit reports**
(Trail of Bits) and *proves* every answer is grounded:

- **Retrieval**: token-aware chunking, local embeddings, hybrid (vector + lexical)
  reranking for better precision.
- **Grounding**: an anti-hallucination layer verifies that the answer's key
  terms actually appear in the retrieved evidence. If there is not enough
  grounding, the system **refuses** rather than guessing.
- **Citations**: every answer is tied to numbered sources and resolved back to a
  document + page, so a reviewer can verify the claim.
- **Guardrails**: input validation (empty/too-long/injected queries) and output
  verification (grounded? sourced?) wrap the whole flow.
- **Eval pipeline**: a golden dataset + dual judge (deterministic heuristic in
  CI, LLM-as-judge locally) scores faithfulness, answer relevance, citation
  accuracy, context precision/recall and hallucination rate, with a regression
  guard that fails CI on quality degradation.

Everything runs on a **local Ollama model (`qwen2.5-coder:7b`)** and a local
embedder — **no cloud, no API keys, no paid services**.

## Architecture

```
                          ┌──────────────┐
  user query  ───────────►│ QueryGuard   │  (empty? too long? injection?)
                          └──────┬───────┘
                                 ▼
                          ┌──────────────┐
                          │  Retriever   │  embed query + top-k vector search
                          └──────┬───────┘
                                 ▼
                          ┌──────────────┐
                          │  Reranker    │  hybrid vector + lexical (precision)
                          └──────┬───────┘
                                 ▼
                          ┌──────────────┐
                          │  Generator   │  local LLM: answer with [1][2] cites
                          └──────┬───────┘
                                 ▼
                          ┌─────────────────────────┐
                          │  Grounding + OutputGuard│  anti-hallucination check
                          └──────┬──────────────────┘
                                 ▼
                     grounded + cited answer  (or "I DON'T KNOW")

   Offline build pipeline:
   manifest.json ─► fetch_corpus.py ─► PDFs ─► extractor ─► chunker
                                                      └► embedder ─► ChromaDB
```

The package exposes a small, dependency-injected core (`src/smart_contract_rag/`)
so each layer can be tested in isolation and swapped for a fake:

```
src/smart_contract_rag/
├── ingest/       fetcher · extractor · chunker
├── index/        embeddings (protocol + real + fake) · vector store
├── retrieval/    retriever · reranker (ScoreFusion / RRF)
├── generation/   prompt · generator · grounding (anti-hallucination)
├── guardrails.py input + output guardrails
├── pipeline.py   end-to-end orchestration
└── config.py     environment-driven settings
```

### Design-for-testability (the core engineering idea)

Every hard dependency is hidden behind a **Protocol** so tests can inject
deterministic fakes:

- `EmbeddingBackend` → real `SentenceTransformerBackend` / deterministic `FakeBackend`
- `VectorStore` → real `ChromaVectorStore` / in-memory fake
- `Generator` → real `LLMGenerator` (Ollama) / scripted fake
- `Reranker`, `GroundedTextCheck`, guardrails — all injectable

This is what makes the **~196 deterministic unit tests** hermetic: they run in
CI with **no Ollama, no ChromaDB server, and no network**.

## Stack

- **Python 3.11+** — type-hinted, protocol-based, no threads
- **Ollama** `qwen2.5-coder:7b` (local) — answer generation
- **sentence-transformers** `all-MiniLM-L6-v2` (local) — embeddings
- **ChromaDB** — persistent, in-process vector store
- **PyMuPDF** — PDF text extraction with page provenance
- **pytest** — ~196 deterministic tests (unit + eval pipeline) + live-path scripts

## Getting Started

### Prerequisites
- Python 3.11+ with pip (or the `python-lab` docker container on the
  `docker_default` network)
- Ollama running locally with `qwen2.5-coder:7b` pulled
- Network access on first model/embedder download

### Run the deterministic tests (no network, no Ollama)
```bash
cd smart-contract-rag
python -m venv .venv && source .venv/bin/activate
pip install pytest numpy httpx
pytest tests/ -v
```

### (Optional) Build the live corpus + index
```bash
# 1. Fetch the public audit PDFs (from data/corpus/manifest.json)
python scripts/fetch_corpus.py

# 2. Index them into a persistent ChromaDB store (downloads the embedder once)
python scripts/index_corpus.py
```

### (Optional) Answer a question end-to-end (requires Ollama in the network)
```bash
PYTHONPATH=src python -m smart_contract_rag.cli \
  "Does the Aave V3 review mention reentrancy?"
```

## Evaluation & Integrity

The project ships a full **eval pipeline** (`src/smart_contract_rag/evals/`) so
quality is measured, not assumed — and regressions are caught in CI before they
ship:

- **Golden dataset**: `data/evals/golden_set.json` — 14 curated
  (question, expected-answer) cases across the DeFi vulnerability topics the
  corpus covers (reentrancy, access control, oracle manipulation, flash loans,
  …), including 2 trap questions that the system must refuse.
- **Metrics** (`metrics.py`): deterministic, dependency-free scoring of
  faithfulness/grounding, answer relevance, citation accuracy, context
  precision/recall, correct-refusal rate and hallucination rate.
- **Dual judge** (`judge.py`): `HeuristicJudge` (lexical, fully reproducible —
  the CI default) and `OllamaJudge` (LLM-as-judge, 0–5 structured scoring on
  the local model when available).
- **Runner + regression guard** (`runner.py`): `run_eval()` collapses the
  per-case scores into an `EvalReport` and compares them against
  `RegressionThresholds` — floors for faithfulness/relevance/recall/citations,
  a ceiling for hallucination rate — producing a **PASS / WARN / FAIL** verdict.

### Run the eval

```bash
# Deterministic mode (no Ollama, no network) — what CI runs:
python scripts/run_eval.py --no-llm-judge

# LLM-as-judge mode (requires Ollama reachable at $OLLAMA_URL):
PYTHONPATH=src python scripts/run_eval.py

# Machine-readable output + topic filter:
PYTHONPATH=src python scripts/run_eval.py --json --topic reentrancy
```

The script exits with a semantic code: `0` = PASS, `1` = FAIL (a metric
breached its threshold), `2` = WARN (approaching a threshold) — so CI can gate
on it.

### Live demo (2026-09-08)

A full end-to-end run against the real corpus, the local LLM and the persistent
ChromaDB store is recorded in [`docs/LIVE_DEMO.md`](docs/LIVE_DEMO.md):
546 chunks indexed from 10 real audit reports, grounded answers with citations
(e.g. `TOB-BALANCER-001`), correct refusals, and an honest **FAIL** verdict from
the regression guard. Known measurement issues found by the live run are tracked
in [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md). A follow-up run after fixing
the measurement issue (commit `f543845`, KI-01 resolved) is recorded in the same
docs — see `LIVE_DEMO.md` (Follow-up section) and `KNOWN_ISSUES.md`.

### Runtime integrity checks

- **Anti-hallucination**: the grounding check (`NaiveGroundedTextCheck`)
  computes the ratio of the answer's substantive terms that appear in the
  retrieved evidence. Below a configurable threshold, the pipeline refuses
  (`I DON'T KNOW`). Tests confirm it accepts grounded answers and rejects
  fabricated ones (e.g. an answer about "Binance listing" against an audit
  corpus).
- **Retrieval precision**: the hybrid reranker favours chunks that share exact
  query terms, reducing noise from purely-semantic near-misses.
- **Deterministic test surface**: chunker boundaries, embedding determinism,
  store/retriever correctness, rerank ordering, grounding thresholds, guardrail
  rejection, the full pipeline flow, and the eval pipeline itself (dataset
  validation, metrics, judges, regression verdicts) — all covered in CI with
  fakes.

## Semantic grounding (experimental)

An optional **NLI entailment gate** (`EntailmentGroundedTextCheck` +
cross-encoder) can replace the lexical anti-hallucination check. It accepts an
answer when it is *semantically entailed* by the retrieved evidence, not just
when it shares tokens with it — which lifts the lexical gate's false refusals
on synthetic/aggregate questions. Measured on the demo corpus in
[`docs/SEMANTIC_GROUNDING.md`](docs/SEMANTIC_GROUNDING.md) (honest verdict:
more strict on anchored topics, no aggregate metric gain, but it unlocks new
capability):

```bash
# lexical (default): the token-overlap gate refuses aggregate questions
GROUNDING_MODE=lexical PYTHONPATH=src python -m smart_contract_rag.cli \
  "What are the more common attack vectors in Web3 smart contracts?"
# → I DON'T KNOW

# semantic: the NLI entailment gate accepts the grounded synthesis
GROUNDING_MODE=semantic PYTHONPATH=src python -m smart_contract_rag.cli \
  "What are the more common attack vectors in Web3 smart contracts?"
# → 10 attack vectors (reentrancy, integer under/overflows, front running, …),
#   each grounded in the retrieved audit reports
```

Caveat: with `GROUNDING_THRESHOLD=0.5` (default) the semantic gate is stricter
than the lexical one on well-anchored topics (e.g. ev-002 reentrancy moved from
answered to refused in the A/B run).

## Roadmap

- ✅ **Eval pipeline (Phase 2, done)**: golden dataset + dual judge
  (deterministic heuristic + LLM-as-judge) for faithfulness/relevance/
  citation/context scoring and hallucination-rate regression, wired into CI via
  `scripts/run_eval.py` exit codes (PASS/WARN/FAIL).
- **Live demo**: a small query UI (Hugging Face Spaces / Streamlit).
- **Larger corpus**: expand the manifest with more published audits + threat models.
- ✅ **Semantic grounding (implemented, experimental)**: `EntailmentGroundedTextCheck`
  NLI gate (cross-encoder, default public `typeform/distilbert-base-uncased-mnli`)
  + `GROUNDING_MODE`/`GROUNDING_THRESHOLD` knobs. A/B-measured against the lexical
  baseline on the demo corpus — capability gain on synthetic/aggregate questions,
  but aggregate metrics did not improve on anchored topics (threshold pending
  calibration). Full write-up:
  [`docs/SEMANTIC_GROUNDING.md`](docs/SEMANTIC_GROUNDING.md).

## Limitations

- **Local model latency**: `qwen2.5-coder:7b` on CPU is slower than cloud LLMs;
  production deployments would benefit from a GPU.
- **Token-level grounding** is a heuristic: it detects *absence* of evidence
  well, but is not a full semantic entailment check. The eval pipeline measures
  this gap: the `OllamaJudge` (LLM-as-judge) scores faithfulness semantically
  when available, and a semantic-entailment judge is planned (see Roadmap).
- **Corpus scope**: the manifest currently points at 10 public Trail of Bits
  reviews; it is a demonstration corpus, not an exhaustive security database.
- **Educational scope**: the tool aids review but is **not** a substitute for a
  professional smart-contract audit.
- **Eval retrieval metrics on the live index** are measured correctly after
  resolving the golden-id/index-id mismatch (KI-01, fixed in commit `f543845`):
  context metrics now reflect real retrieval quality — e.g. context_recall moved
  from 0.083 to 0.5 and citation accuracy from 0.10 to 0.44 — while the verdict
  stays FAIL because several topics still miss their source document, a
  documented functional improvement target (see
  [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md)).

## License

MIT — see [LICENSE](LICENSE). Free to use, modify and distribute with attribution.
