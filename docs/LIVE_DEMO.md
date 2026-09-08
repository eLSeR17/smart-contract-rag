# Live Demo — 2026-09-08

A full end-to-end run of the RAG system against the real corpus, the real local
LLM and the persistent vector store. This is the record of what was executed,
the evidence produced, and the issues the live run surfaced that the
deterministic test suite could not (the tests run against in-memory fakes).

## Environment

| Component | Value |
|---|---|
| Model | `qwen2.5-coder:7b` via Ollama (`http://ollama:11434`, docker network `docker_default`) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, local) |
| Vector store | ChromaDB persistent (`data/chroma/`) |
| Runtime | container `python:3.12-slim`, repo mounted at `/app` |
| Repo HEAD | `e0f9c90` at run time |

## What was executed

1. **Install dependencies** in the demo container (`pip install -r requirements.txt`).
2. **Fetch the corpus** — `scripts/fetch_corpus.py` downloaded the 10 public
   Trail of Bits audit PDFs (~11 MB) into `data/corpus/raw/` (gitignored; the
   manifest is the versioned source of truth).
3. **Build the index** — `scripts/index_corpus.py` extracted, chunked (token-aware,
   ~700 tokens, 150 overlap), embedded and stored **546 chunks** in `data/chroma/`.
   The first run failed validation with ChromaDB's `DuplicateIDError`; see bug
   BUG-01 in KNOWN_ISSUES.md (fixed during the demo, commit `e0f9c90`). The second
   run completed cleanly.
4. **Answer live questions** through the CLI (`python -m smart_contract_rag.cli`).
5. **Run the full eval** — `scripts/run_eval.py` with the LLM-as-judge over the
   14-case golden dataset.

## Live question/answer evidence

| Question | Outcome |
|---|---|
| "What issue does the Balancer v2 audit describe regarding vault funds?" | Grounded answer: a malicious asset manager could drain a pool of tokens outside their management — high-severity finding `TOB-BALANCER-001` [1]. Sources: balancer v2 report pp. 4/9/11, managed-pool report p. 5. |
| "What vulnerability classes are described in the Aave v3 audit?" | Grounded answer listing 13 classes (Access Controls, Authentication, Configuration, Cryptography, Data Exposure, Data Validation, Denial of Service, …) all citing [1]. Sources: Aave v3 pp. 41/5, balancer v2 p. 47, managed-pool p. 20. |
| "What are the recommended mitigations for reentrancy vulnerabilities found in these audits?" | Correctly refused: `I DON'T KNOW` — with grounding ratio 0.0, the output guardrail rejected the response instead of letting the model guess. This is the anti-hallucination contract working as designed. |

## Eval results (LLM-as-judge, 14 golden cases)

| Metric | Value |
|---|---|
| hallucination_rate | **0.0** |
| answer_rate | 0.8333 |
| correct_refusal_rate | 0.8571 |
| faithfulness | 0.58 |
| answer_relevance | 0.44 |
| citation_accuracy | 0.10 |
| context_precision | 0.0208 |
| context_recall | 0.0833 |

Verdict: **FAIL** — the regression guard fired. Interpretation is important:
faithfulness, relevance, answer-rate and zero hallucinations measure real system
behaviour; the three near-zero retrieval/citation metrics are **structural
artefacts of a harness mismatch** (the golden dataset references short document
ids, the index stores full PDF stems) — documented in KI-01 of KNOWN_ISSUES.md.
The demo confirmed the guard reliably flags a red run.

## Reproducibility

```bash
# 1. Demo container on the docker_default network (Ollama reachable at ollama:11434)
docker run -d --name scr-rag-demo --network docker_default \
  -v "$PWD":/app -w /app python:3.12-slim sleep infinity

# 2. Dependencies + corpus + index
docker exec scr-rag-demo pip install --no-cache-dir -r /app/requirements.txt
docker exec -e PYTHONPATH=/app/src scr-rag-demo python /app/scripts/fetch_corpus.py
docker exec -e PYTHONPATH=/app/src scr-rag-demo python /app/scripts/index_corpus.py

# 3. Interactive questions
docker exec -e PYTHONPATH=/app/src scr-rag-demo python -m smart_contract_rag.cli "YOUR QUESTION"

# 4. Full eval (LLM-as-judge)
docker exec -e PYTHONPATH=/app/src scr-rag-demo python /app/scripts/run_eval.py --chroma-dir /app/data/chroma

# 5. Teardown
docker stop scr-rag-demo && docker rm scr-rag-demo
```

Raw artifacts from this run are local-only (`data/demo/`, gitignored). The
canonical record is this document.

## Follow-up — KI-01 resolved (2026-09-08)

The first eval above surfaced a document-id mismatch between the golden dataset
and the index (KI-01, tracked in KNOWN_ISSUES.md): `index_corpus.py` derived
`doc_id` from PDF filename stems while the golden set references the manifest's
canonical ids, zeroing the context/citation metrics for 9 of 12 answerable
cases regardless of retrieval quality. The fix (commit `f543845`) maps each
manifest `url` basename to its canonical `id` and uses that as the `doc_id`.
Everything was then re-run end to end: the index was rebuilt (**546 chunks**
with canonical doc_ids) and the 14-case LLM-as-judge eval was executed again
(`data/demo/eval_report_20260908_KI01-resolved.md`).

| Metric | Before (KI-01 active) | After (fix `f543845`) |
|---|---:|---:|
| faithfulness | 0.58 | 0.6667 |
| answer_relevance | 0.44 | 0.6444 |
| citation_accuracy | 0.10 | 0.4444 |
| context_precision | 0.0208 | 0.1667 |
| context_recall | 0.0833 | 0.5 |
| answer_rate | 0.8333 | 0.75 |
| correct_refusal_rate | 0.8571 | 0.7857 |
| hallucination_rate | 0.0 | 0.0 |

The structural zeros are gone (context_recall 0.0833 → 0.5, citation_accuracy
0.10 → 0.4444) and the verdict is still **FAIL** — but it now reflects real
retrieval quality on this corpus, not a harness bug. Five topics still do not
retrieve their source document (access_control, oracle_manipulation,
token_accounting, admin_key_risk, upgrades), which is a documented functional
improvement target rather than a measurement artefact. Both runs are
hallucination-free (hallucination_rate 0.0).

The live citation queries were re-verified after the re-index (Balancer v2 →
`TOB-BALANCER-001`, Aave v3 vulnerability classes) and remain grounded and
correct.
