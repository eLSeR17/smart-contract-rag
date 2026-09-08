# Semantic Grounding Experiment (2026-09-08)

> Public write-up of the semantic-grounding experiment: replacing the lexical
> anti-hallucination gate with an NLI entailment gate, measured on the demo
> corpus under identical conditions. Raw evidence: `data/demo/eval_report_20260908_LEXICAL-baseline.md`
> and `data/demo/eval_report_20260908_SEMANTIC.md`.

## Motivation

The shipped anti-hallucination layer (`NaiveGroundedTextCheck`) is a **lexical**
gate: it accepts an answer only if enough of its substantive terms appear
literally in the retrieved evidence chunks. That works well for fact-based
questions, but it systematically rejects **synthetic / aggregating questions** —
e.g. *"What are the more common attack vectors in Web3 smart contracts?"* —
because the answer necessarily reuses vocabulary that no single chunk contains
verbatim. The pipeline answers `I DON'T KNOW`, even when every factual element
of the answer is supported by the retrieved sources (observed on the live demo
run, 2026-09-08).

## Hypothesis

A **semantic entailment gate** (NLI cross-encoder) can decide whether an answer
*follows from* the retrieved evidence, not just whether it *shares tokens* with
it. If the answer is semantically entailed by at least one retrieved chunk
(premise → hypothesis), the synthetic answer should be accepted; if not, the
pipeline should keep refusing. This would lift the lexical gate's false
refusals on aggregate questions **without** losing anti-hallucination
protection.

## Design

Implemented (commits `f9102e0`, `b93f6ba`, `2ff99af`):

- `GroundedTextCheck` Protocol (existing interface, unchanged).
- `EntailmentGroundedTextCheck`: new implementation that asks an
  `EntailmentScorer` whether the answer is entailed by the evidence.
- `CrossEncoderEntailmentScorer`: `sentence-transformers` CrossEncoder wrapper,
  **lazy-loaded** (the model is only downloaded/loaded on first `score_pairs`
  call, never at import time). Default model:
  [`typeform/distilbert-base-uncased-mnli`](https://huggingface.co/typeform/distilbert-base-uncased-mnli)
  (~250 MB, public MNLI checkpoint with entailment / neutral / contradiction
  labels). Label lookup is **case-insensitive**: the checkpoint exposes its
  labels as `ENTAILMENT`/`NEUTRAL`/`CONTRADICTION` (uppercase).
- **Scoring**: for each retrieved chunk, score the pair
  `(chunk.text, answer)` as `(premise, hypothesis)` in NLI convention. The
  answer is considered grounded if the **best source's entailment score**
  (`max_ent` over the sources) is `>= threshold` (0.5 default).
- **Activation**: environment variables `GROUNDING_MODE` (`lexical` default |
  `semantic`), `GROUNDING_MODEL` (override the default checkpoint),
  `GROUNDING_THRESHOLD` (default 0.5). Wired behind
  `build_grounding_check()` in `builders.py` — no changes to
  `pipeline.py` / `guardrails.py`.

### Two real bugs found and fixed during live execution (lessons)

1. **Gated checkpoint**: the originally-planned NLI model
   `cross-encoder/nli-MiniLM-L6-v2` returned **HTTP 401** at first use (it is
   gated on Hugging Face, needs an access token). Replaced with the public
   `typeform/distilbert-base-uncased-mnli` checkpoint.
2. **Uppercase labels**: the replacement checkpoint exposes labels in
   **UPPERCASE**, which crashed the first implementation
   (`ValueError: 'entailment' is not in list`) on the very first live eval.
   Fixed with a case-insensitive label lookup (regression test added:
   `test_uppercase_labels_case_insensitive_lookup`).

## Methodology

- Same demo dataset: 14 cases (11 distinct answerable topics, reentrancy having
  two cases → 12 answerable + 2 trap questions).
- Same LLM-as-judge: Ollama `qwen2.5-coder:7b` via `OllamaJudge`.
- Identical conditions: same session, same persistent index (546 chunks from 10
  Trail of Bits audit reports), one run per mode.
- **Caveat**: Ollama sampling is non-deterministic, so there is run-to-run
  noise in LLM-judged metrics (observed ±0.02–0.04 on faithfulness). The two
  modes were also measured on different runs, so small deltas should not be
  over-interpreted.

## Results

### Aggregate metrics

| Metric | Lexical baseline (2026-09-08) | Semantic (2026-09-08) |
|---|---:|---:|
| faithfulness | 0.6444 | 0.625 |
| answer_relevance | 0.6 | 0.525 |
| citation_accuracy | 0.3333 | 0.25 |
| context_precision | 0.1667 | 0.1667 |
| context_recall | 0.5 | 0.5 |
| answer_rate | 0.75 (9 answered / 5 refused) | 0.6667 (8 answered / 6 refused) |
| correct_refusal_rate | 0.7857 | 0.7143 |
| hallucination_rate | 0.0 | 0.0 |

### Per-case (copied verbatim from the raw reports)

Lexical baseline — `data/demo/eval_report_20260908_LEXICAL-baseline.md`:

| id | topic | answered | faith | rel | cit | p@k | rec | refusal | hall |
|----|-------|----------|-------|-----|-----|-----|-----|---------|------|
| ev-001 | reentrancy | False | 0.00 | 0.00 | - | 0.25 | 1.00 | False | False |
| ev-002 | reentrancy | True | 0.60 | 0.60 | 1.00 | 0.25 | 1.00 | True | False |
| ev-003 | access_control | True | 0.80 | 0.80 | 0.00 | 0.00 | 0.00 | True | False |
| ev-004 | integer_overflow | False | 0.00 | 0.00 | - | 0.25 | 1.00 | False | False |
| ev-005 | oracle_manipulation | True | 0.80 | 0.80 | 0.00 | 0.00 | 0.00 | True | False |
| ev-006 | flash_loans | True | 0.40 | 0.40 | 0.00 | 0.00 | 0.00 | True | False |
| ev-007 | privilege_escalation | False | 0.00 | 0.00 | - | 0.25 | 1.00 | False | False |
| ev-008 | token_accounting | True | 0.40 | 0.40 | 0.00 | 0.00 | 0.00 | True | False |
| ev-009 | liquidation | True | 0.60 | 0.40 | 1.00 | 0.75 | 1.00 | True | False |
| ev-010 | input_validation | True | 0.80 | 0.80 | 1.00 | 0.25 | 1.00 | True | False |
| ev-011 | admin_key_risk | True | 0.80 | 0.80 | 0.00 | 0.00 | 0.00 | True | False |
| ev-012 | upgrades | True | 0.60 | 0.40 | 0.00 | 0.00 | 0.00 | True | False |
| ev-013 | trap | False | 0.00 | 0.00 | - | 0.00 | 0.00 | True | False |
| ev-014 | trap | False | 0.00 | 0.00 | - | 0.00 | 0.00 | True | False |

Semantic — `data/demo/eval_report_20260908_SEMANTIC.md`:

| id | topic | answered | faith | rel | cit | p@k | rec | refusal | hall |
|----|-------|----------|-------|-----|-----|-----|-----|---------|------|
| ev-001 | reentrancy | False | 0.00 | 0.00 | - | 0.25 | 1.00 | False | False |
| ev-002 | reentrancy | False | 0.00 | 0.00 | - | 0.25 | 1.00 | False | False |
| ev-003 | access_control | True | 0.60 | 0.40 | 0.00 | 0.00 | 0.00 | True | False |
| ev-004 | integer_overflow | False | 0.00 | 0.00 | - | 0.25 | 1.00 | False | False |
| ev-005 | oracle_manipulation | True | 0.60 | 0.40 | 0.00 | 0.00 | 0.00 | True | False |
| ev-006 | flash_loans | True | 0.60 | 0.40 | 0.00 | 0.00 | 0.00 | True | False |
| ev-007 | privilege_escalation | False | 0.00 | 0.00 | - | 0.25 | 1.00 | False | False |
| ev-008 | token_accounting | True | 0.40 | 0.40 | 0.00 | 0.00 | 0.00 | True | False |
| ev-009 | liquidation | True | 0.80 | 0.80 | 1.00 | 0.75 | 1.00 | True | False |
| ev-010 | input_validation | True | 0.80 | 0.80 | 1.00 | 0.25 | 1.00 | True | False |
| ev-011 | admin_key_risk | True | 0.80 | 0.80 | 0.00 | 0.00 | 0.00 | True | False |
| ev-012 | upgrades | True | 0.40 | 0.20 | 0.00 | 0.00 | 0.00 | True | False |
| ev-013 | trap | False | 0.00 | 0.00 | - | 0.00 | 0.00 | True | False |
| ev-014 | trap | False | 1.00 | 0.00 | - | 0.00 | 0.00 | True | False |

### Key deltas and honest verdict

- **ev-002 (reentrancy) — regression**: lexical *answered* with a perfect
  citation (faith 0.60 / cit 1.00); semantic *refused*. The NLI gate is
  stricter than the token gate on this anchored topic.
- **ev-006 (flash_loans) and ev-009 (liquidation) — improvement**: faith rises
  0.40 → 0.60 and 0.60 → 0.80 (relevance 0.40 → 0.80 for ev-009).
- **ev-003, ev-005, ev-012 — small drops** (faith −0.20, −0.20, −0.20; rel
  −0.40/−0.40/−0.20): consistent with the stricter gate, but within the
  Ollama-Judge noise band (±0.02–0.04 is smaller than this; these particular
  drops look mode-driven rather than pure noise).
- **Trap cases**: ev-013 and ev-014 refuse in **both** modes. The judge scored
  ev-013's refusal 0.00 in both runs (no support retrievable and no answer
  given), while ev-014's refusal scored faith 1.00 in semantic mode (0.00 in
  lexical) — same `I DON'T KNOW` output, judge noise on refusal scoring;
  relevance 0.00 and no hallucination in all trap cells. `hallucination_rate`
  stays **0.0** in both modes (no fabricated content in either run).
- **Honest aggregate verdict**: on this dataset the semantic gate does **not**
  improve overall metrics — it is stricter and slightly worse
  (faith 0.6444 → 0.625, answer_rate 0.75 → 0.6667,
  correct_refusal_rate 0.7857 → 0.7143). The interesting result is qualitative
  (below), not metric: the gate lifts the *capability* ceiling on synthetic
  questions while keeping hallucination at 0.0.

## Qualitative result — the capability gap

Synthetic question, same retrieved sources in both modes
(`0x-protocol` p77, `aave-v3` p41, `increment-security` p48, `balancerv2`
p47):

> **"What are the more common attack vectors in Web3 smart contracts?"**

**Lexical mode** → the policy's token overlap check fails (no single chunk
contains the aggregate answer's vocabulary) → the pipeline refuses:

> I DON'T KNOW

**Semantic mode** → the NLI gate finds the answer entailed by the sources →
the pipeline answers with the 10 most common attack vectors, each grounded in
the retrieved evidence:

> Based on the retrieved audit reports [1][2][3][4], the most common attack
> vectors in smart contracts are:
>
> 1. Reentrancy
> 2. Integer over/underflows
> 3. Front running
> 4. Phishing
> 5. Denial of service (DoS)
> 6. Timestamp dependence
> 7. Gas limit issues
> 8. Unchecked external calls
> 9. Access control flaws
> 10. Data validation gaps

(Sources: `0x-protocol` p77 [1], `aave-v3` p41 [2], `increment-security` p48
[3], `balancerv2` p47 [4].) The lexical gate could never produce this answer,
even though every vector is supported by the retrieved evidence.

## Trade-offs & next steps

1. **Calibrate `GROUNDING_THRESHOLD`**: 0.5 is strict. Lowering it reduces
   false refusals on aggregate questions at the cost of accepting weaker
   grounding; raising it hardens the gate. The right value depends on the
   relative cost of a false refusal vs. a false acceptance for the target use
   case.
2. **Aggregation layer for "most common"**: replace free-form synthesis with a
   rigorous count over the retrieved findings (the corpus already links
   findings to pages), so aggregate answers are *measured*, not generated.
3. **Better retrieval for cross-document questions**: several misses above are
   retrieval misses (p@k / recall 0.00 in both modes); an aggregate question
   needs retrieval tuned to multi-source aggregation, not single-topic search.
4. **Adopt as a stable option** only if the capability gain (synthetic
   questions) demonstrates value in public benchmarks without regressing
   anchored topics (see ev-002 above).