# Known Issues

Tracked issues discovered through live validation. Each entry records the
symptom, root cause, evidence and a proposed fix so the project's quality
process stays auditable.

## Open

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
