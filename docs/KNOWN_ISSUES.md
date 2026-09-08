# Known Issues

Tracked issues discovered through live validation. Each entry records the
symptom, root cause, evidence and a proposed fix so the project's quality
process stays auditable.

## Fixed

### BUG-01 — Duplicate chunk ids with a persistent ChromaDB store

- **Discovered**: 2026-09-08, first live index run → `DuplicateIDError` on
  `0x-protocol::chunk-0000`.
- **Root cause**: `build_chunks_from_pages` chunked per page and the chunker
  reset its index to 0 on every page, so multi-page documents produced
  colliding ids `{doc}::chunk-0000`, `::chunk-0001`, …
- **Impact**: indexing failed on any multi-page PDF; the in-memory test store
  silently overwrote duplicates, which is why the suite stayed green.
- **Fix**: global per-document index counter in `build_chunks_from_pages`
  (`running_index`), preserving per-page provenance for citations.
  Regression coverage added (`TestMultiPageChunkIdUniqueness`, 4 tests;
  suite 192 → 196).
- **Commit**: `e0f9c90` — `fix(ingest): use global per-document chunk index to avoid ID collisions`

### KI-01 — Eval retrieval metrics are structurally zeroed by a document-id mismatch

- **Component**: eval harness (`data/evals/golden_set.json` ↔ `scripts/index_corpus.py`)
- **Discovered**: 2026-09-08, first live eval run
- **Severity**: Medium — measurement correctness; does not affect inference-time
  retrieval or generation.
- **Symptom**: `citation_accuracy` 0.10, `context_precision` 0.02,
  `context_recall` 0.08 → regression verdict FAIL in the live eval, while the
  system answers live questions correctly with accurate citations.
- **Root cause**: the golden dataset's `relevant_doc_id` uses the manifest's
  short ids (`aave-v3`, `balancer-managedpool`, …) but `index_corpus.py`
  derives `doc_id` from the PDF **filename stem**
  (`2021-11-aave-v3-securityreview`, …). 10 of 12 answerable cases never
  match, so context precision/recall and citation accuracy are 0.0
  *regardless of retrieval quality*. The manifest states its `id` is the
  canonical citation identifier ("how it is cited"), which the indexer ignores.
- **Evidence**: the only case with non-zero context metrics is `ev-010`
  (citation 1.00 / precision 0.25 / recall 1.00), whose `relevant_doc_id`
  matches because `0x-protocol.pdf` has no date prefix in its filename — while
  ev-003 (also `0x-protocol`) did not surface 0x-protocol chunks for its query.
  All mismatched cases score 0.00 across the board.
- **Proposed fix**: `index_corpus.py` should use the manifest `id` as
  `doc_id` (mapping `id → url-basename` or by fixing `file` in the manifest),
  then re-index and re-run the eval so retrieval metrics reflect real quality.
- **Workaround**: none needed for inference; the eval's faithfulness,
  relevance, answer-rate and hallucination metrics remain valid.
- **Why tests did not catch it**: the eval was only validated against
  in-memory fakes (deterministic store with its own chunk ids); the live index
  is the first consumer with real doc ids.

**Resolution (2026-09-08)**:

- **Fix applied**: `scripts/index_corpus.py` now maps each manifest entry's
  `url` basename to the canonical `id` and uses that as the `doc_id` (fallback:
  PDF filename stem for files outside the manifest). Commit `f543845`.
- **Re-index re-run**: 546 chunks indexed in `data/chroma/` with canonical
  doc_ids (`0x-protocol`, `balancerv2`, `aave-v3`, `maplefinance-v1`,
  `beanstalk-security`, `reserve-security`, `increment-security`,
  `balancer-managedpool`, `fraxlend-fraxferry`, `optimism-security`).
- **Re-eval (llm-judge, 14 cases,
  `data/demo/eval_report_20260908_KI01-resolved.md`)**:

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

  The structural zeros are gone; the verdict is still FAIL, but it now measures
  real retrieval quality on this corpus, not a harness bug.
- **Interpretation**: 4 cases produce a correct citation with complete recall
  (ev-001, ev-002, ev-009 — p@k 0.75 —, ev-010); 2 cases retrieve the right
  document but the LLM refuses conservatively (ev-004, ev-007, recall 1.00);
  5 cases still fail to retrieve the source document (ev-003, ev-005, ev-008,
  ev-011, ev-012 — p@k/recall 0.00); ev-006 now refuses correctly (it answered
  without grounding before the fix); trap cases ev-013 and ev-014 keep refusing.
- **Key nuance**: ev-003 was id-matched *before* the fix (its `id` `0x-protocol`
  already coincided with its filename stem) and failed pre-fix too, so its miss
  is real retrieval quality, not the id bug. The fix removes the structural
  zeros of the 9 cases whose ids had date-prefixed filename stems.
- **Methodological note**: `hallucination_rate` stays 0.0 in both runs;
  `answer_rate` drops 0.8333 → 0.75 because ev-006 moves from an ungrounded
  answer to a correct refusal.
