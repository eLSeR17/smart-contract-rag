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

This is what makes the **40+ deterministic unit tests** hermetic: they run in
CI with **no Ollama, no ChromaDB server, and no network**.

## Stack

- **Python 3.11+** — type-hinted, protocol-based, no threads
- **Ollama** `qwen2.5-coder:7b` (local) — answer generation
- **sentence-transformers** `all-MiniLM-L6-v2` (local) — embeddings
- **ChromaDB** — persistent, in-process vector store
- **PyMuPDF** — PDF text extraction with page provenance
- **pytest** — 40+ deterministic tests + live-path scripts

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
  rejection, and the full pipeline flow — all covered in CI with fakes.

## Roadmap

- **Eval pipeline (Phase 2)**: golden dataset + LLM-as-judge for answer
  faithfulness/latency regression, wired into CI.
- **Live demo**: a small query UI (Hugging Face Spaces / Streamlit).
- **Larger corpus**: expand the manifest with more published audits + threat models.

## Limitations

- **Local model latency**: `qwen2.5-coder:7b` on CPU is slower than cloud LLMs;
  production deployments would benefit from a GPU.
- **Token-level grounding** is a heuristic: it detects *absence* of evidence
  well, but is not a full semantic entailment check. The eval pipeline (next
  phase) strengthens this with an LLM-as-judge.
- **Corpus scope**: the manifest currently points at 10 public Trail of Bits
  reviews; it is a demonstration corpus, not an exhaustive security database.
- **Educational scope**: the tool aids review but is **not** a substitute for a
  professional smart-contract audit.

## License

MIT — see [LICENSE](LICENSE). Free to use, modify and distribute with attribution.
