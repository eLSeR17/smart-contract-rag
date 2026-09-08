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
This is a heuristic (absence detection), not full semantic entailment; the
eval pipeline (see *Evaluation Pipeline* below) measures this gap with an
LLM-as-judge. But it already prevents the failure mode that matters most in
security: *confidently citing a finding that is not in the report at all*.

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

## Evaluation pipeline

The eval pipeline (`src/smart_contract_rag/evals/`) is the project's quality
backbone: it turns "the answers look grounded" into **measurable, gated
metrics** that CI can fail on. It exists because for a security-audit assistant,
an unmeasured regression in faithfulness is not a cosmetic issue — it is a
liability.

### Golden dataset (`evals/dataset.py` + `data/evals/golden_set.json`)

A *golden set* is the ground truth the system is measured against: a curated
list of `GoldenCase` objects, each with:

- `question` — the user query;
- `expected_keywords` — substantive terms a correct answer must contain
  (drives lexical answer-relevance scoring);
- `ground_truth` — an optional human-written reference answer;
- `relevant_doc_id` — the corpus doc(s) that back the answer (drives context
  precision/recall and citation accuracy);
- `expect_answer` — `True` for answerable questions, `False` for **traps**:
  questions about protocols/reports *not* in the corpus that the system must
  refuse rather than invent. The 2 trap cases in the current set target
  out-of-corpus reports (e.g. a "2024 zero-knowledge validator upgrade") and a
  non-existent "LumenPay" protocol.

The schema is validated on load (unknown topics, duplicate ids, wrong field
types fail fast), so a malformed golden set breaks CI instead of silently
skewing a run. Subsets by topic/id/predicate let you drill into one
vulnerability area.

### Metrics (`evals/metrics.py`)

Eight aggregate metrics, each a mean over the cases it applies to, all
**deterministic and dependency-free** (simple lowercased tokenization):

| Metric | What it measures |
|--------|------------------|
| `faithfulness` | share of the answer's substantive terms found in the retrieved chunks (grounding ratio; `1.0` = fully grounded) |
| `answer_relevance` | lexical overlap of the answer with `expected_keywords` (or the reference answer) |
| `citation_accuracy` | the cited source(s) include the expected document |
| `context_precision` | precision@k: share of retrieved chunks that are relevant |
| `context_recall` | share of expected documents that were retrieved at all |
| `answer_rate` | fraction of answerable questions actually answered (traps excluded) |
| `correct_refusal_rate` | fraction of refusal decisions that were correct (answer vs. trap cases) |
| `hallucination_rate` | fraction of cases where the system invented an ungrounded answer (answered with no/weak evidence) |

Metrics that are not measurable for a case (e.g. citation accuracy with no
expected document, or context recall for a trap) are excluded from that case
rather than penalized, and vacuous runs pass instead of failing spuriously.

### Dual judge (`evals/judge.py`)

Faithfulness and relevance are the two signals a token-overlap heuristic alone
cannot fully capture, so the runner scores them through a `Judge` — with two
implementations behind the same protocol:

- **`HeuristicJudge`** — derives 0–5 scores from the lexical metrics directly.
  Fully reproducible, no network. This is what CI runs.
- **`OllamaJudge`** — the "real" LLM-as-judge: calls the local Ollama model
  (`$OLLAMA_URL` / `$OLLAMA_MODEL`, the same endpoint as the generator) with a
  structured 0–5 scoring prompt (two score lines + one reasoning line) and
  parses the result. On any failure (network, unparseable output) it returns a
  `JudgeScore` with `error` set instead of raising, so the run degrades to
  heuristics rather than dying.

The runner normalizes judge scores to `[0, 1]` so both judges are directly
comparable and blend into the same metric set. The deterministic metrics are
*always* computed regardless, so every report is complete and reproducible.

### Runner + regression guard (`evals/runner.py`)

`run_eval(pipeline, dataset, judge=…)` is dependency-injected: it works against
the real production pipeline and against any deterministic fake, so the runner
tests never touch Ollama or the network. Per case it scores faithfulness,
relevance, citation accuracy, context precision/recall, refusal correctness and
hallucination, then aggregates into an `EvalReport` (markdown or JSON).

The report carries a **verdict** computed from `RegressionThresholds` — the
regression guard. Default thresholds:

```python
max_hallucination_rate = 0.10   # ceiling: higher-than-this hallucination fails
min_faithfulness       = 0.70   # floor for grounding
min_answer_relevance   = 0.60   # floor for relevance
min_context_recall     = 0.50   # floor for retrieval recall
min_citation_accuracy  = 0.60   # floor for citation correctness
min_answer_rate        = 0.70   # floor for willingness to answer
warn_margin            = 0.05   # width of the WARN band around a guard
```

Verdict logic:

- **FAIL** — any metric breaches its guard (a rate above a `max_*` ceiling or a
  floor metric below a `min_*` threshold);
- **WARN** — no breach, but a metric is inside the `warn_margin` band of its
  guard (approaching it);
- **PASS** — all metrics comfortably inside their guards.

The WARN band only applies inside the plausible metric range `[0, 1]`: for a
*rate* the band is `(guard − margin, guard]` and only exists when
`guard < 1` and `guard − margin > 0`; for a *floor* it is
`[guard, guard + margin)` and only exists when `guard > 0` and
`guard + margin < 1`. When a guard sits at the edge of the range (e.g.
`max_hallucination_rate = 1.0`), the band is disabled and any value on the safe
side is simply PASS. This keeps a run from reporting spurious WARNs on
thresholds that cannot meaningfully be "approached".

### Wiring it into CI (`scripts/run_eval.py`)

`scripts/run_eval.py` builds the *real* pipeline from the environment and
evaluates it against the golden set. It defaults to the LLM judge when Ollama
is reachable and falls back to `HeuristicJudge` in CI-like environments
(`--no-llm-judge` forces determinism). It maps the verdict to an **exit code**
so CI can gate on it:

```bash
PYTHONPATH=src python scripts/run_eval.py --no-llm-judge   # deterministic
PYTHONPATH=src python scripts/run_eval.py                   # LLM-as-judge if available
# exit: 0 = PASS, 1 = FAIL, 2 = WARN
```

### Why this is the project's differentiator

Most RAG demos show *good-looking answers*. This project shows *measured
quality over time*:

- **Regression guard** — every change to retrieval, prompts or chunking is
  checked against the golden set before it ships; a drop in faithfulness or a
  rise in hallucination rate fails the build. This is what makes the project
  maintainable rather than demo-only.
- **Anti-hallucination, measured** — hallucination rate is a first-class gated
  metric, and the trap cases prove the system knows when *not* to answer.
- **Honest judging** — the dual-judge design keeps CI fast and deterministic
  while still offering a semantic LLM-as-judge signal locally, and the judge is
  *auditable* (0–5 scores + reasoning per case), not a black-box "quality
  score".
