# Architecture & Key Decisions

This document explains the *why* behind the design of SmartContractRAG, so a
reviewer or contributor can see the reasoning without reading all the code.

## Layered, dependency-injected core

The pipeline is split into cross-cutting layers, each behind a Protocol so it
can be swapped for a fake:

| Layer | Responsibility | Protocol | Real | Test fake |
|-------|---------------|----------|------|-----------|
| `index.embeddings` | text -> vector | `EmbeddingBackend` | `SentenceTransformerBackend` | `DeterministicFakeEmbeddingBackend` |
| `index.store` | store / query vectors | `VectorStore` | `ChromaVectorStore` | `InMemoryVectorStore` |
| `retrieval.retriever` | top-k search | `RetrieverLike` | `VectorRetriever` | (uses fakes above) |
| `retrieval.reranker` | re-order candidates | `Reranker` | `ScoreFusionReranker`, `RRF_Reranker` | (pure functions) |
| `generation.generator` | LLM answer | `Generator` | `LLMGenerator` | scripted fake |
| `generation.grounding` | anti-hallucination | `GroundedTextCheck` | `NaiveGroundedTextCheck` | (pure function) |
| `guardrails` | input/output validation | classes | `QueryGuardrail`, `OutputGuardrail` | (pure) |
| `pipeline` | orchestration | — | `RAGPipeline` | (with fakes injected) |

The key insight: **the pipeline never knows which concrete implementation it
holds**. Constructed with real components it is production; constructed with
fakes it is a fully-deterministic test that needs no Ollama, no ChromaDB server,
and no network. This is what lets CI run the entire flow including the
anti-hallucination logic.

## Why token-aware chunking

Audit reports contain long paragraphs and Solidity code snippets. Binomial
character-count chunking would split headings from their content and split
facts mid-sentence. We chunk **by whole sentences** into fixed-size windows with
an overlap that carries boundary context into the next window, so:

- no sentence is ever split in half (keeps embeddings coherent),
- a fact straddling a chunk boundary is still retrievable from either side.

`RecordChunker` additionally never lets a Markdown/Solidity heading bleed into a
neighbouring section, keeping each chunk semantically self-contained.

## Why hybrid reranking

Vector similarity alone can rank a semantically-adjacent chunk above the one
that literally contains the term the user asked about (e.g. a specific function
or vulnerability name). We add a lightweight **lexical** score and fuse it with
the vector score — `ScoreFusionReranker` (weighted sum) and `RRF_Reranker`
(rank fusion). Both are deterministic and dependency-free, so they are simple to
test and safe in CI. We chose a robust, transparent approach over a heavyweight
cross-encoder because:
- it needs no extra model download / GPU,
- it is trivially testable,
- for this corpus the lexical signal meaningfully boosts precision without the
  maintenance cost of a second model.

## Why grounding instead of just "trust the prompt"

Prompting alone does not stop hallucination: models drift, especially on dense
technical prose. The output-side check (`NaiveGroundedTextCheck`) verifies that
the answer's **substantive tokens** — numbers and technical terms, skipping
stopwords — actually appear in the retrieved evidence. Below a threshold the
pipeline **refuses** (`I DON'T KNOW`) rather than serving an unverifiable claim.
This is a heuristic (absence detection), not full semantic entailment; the next
phase adds an LLM-as-judge eval to strengthen it. But it already prevents the
failure mode that matters most in security: *confidently citing a finding that
is not in the report at all*.

## Why local models

The portfolio constraint is **no paid APIs**. Ollama runs `qwen2.5-coder:7b`
inside the Docker network (reachable at `http://ollama:11434`, not `localhost`),
and embeddings come from the local `all-MiniLM-L6-v2`. This keeps the whole
system self-contained, private, and cost-free — and it is the same reason every
external interaction (URLs, model names, paths) is configurable via the
environment rather than hardcoded.

## Security & confidentiality

- No secrets in the repo. Corpus PDFs are public audits, gitignored; the
  committed artifact is just the **manifest** (URLs + metadata).
- The system prompt explicitly instructs the model *not to act on instructions
  embedded in the source text* (the audit PDFs are public and could conceivably
  contain adversarial text), and the output grounding check is the hard backstop.
- No threads are used anywhere (the ecosystem forbids them), keeping the code
  safe to run in sandboxed environments.

## Corpus sourcing

The manifest points at 10 public Trail of Bits audit reports (Aave V3, Balancer
V2, Beanstalk, Reserve, Optimism, Fraxlend/FraxFerry, Maple, Increment, 0x,
Balancer Managed Pool) with verified raw-GitHub URLs. We intentionally pick
reports that are **freely published and stable**, and the fetcher downloads them
sequentially with polite delay and idempotency (skip existing unless `--force`).
