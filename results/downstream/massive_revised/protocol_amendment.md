# MASSIVE evaluation — protocol amendment (2026-09-16)

Amendment recorded at **2026-09-16 ~04:40 UTC**, before any compressed
checkpoint had been scored and before any test-set prediction existed.

## 1. What changed

| | Original plan (frozen 2026-09-16 ~03:2x UTC) | This amendment |
|---|---|---|
| Compressed conditions | English, Chinese, Korean, Swahili, Bengali, Balanced-character (6) | **English, Balanced-character (2)** |
| Seeds | 0, 1, 2 | 0, 1, 2 (unchanged) |
| Eval languages | EN, ZH, KO, SW, BN | EN, ZH, KO, SW, BN (unchanged) |
| Test items / language | 600 | **200** |
| Model loads | 19 | **7** |
| Total item-scorings | 57,000 | **7,000** |
| Primary contrast | Balanced − own-language *and* Balanced − English | **Balanced − English only** |
| GPUs | 2 (with a 4-GPU option under consideration) | **physical GPU 2,3 only; single stream; no 4-GPU** |
| Bootstrap | 10,000 reps, seed 12345, resample across seeds | **2,000 reps, seed 42, resample jointly across 3 seeds and 5 languages** |
| Budget | none stated | **12 h wall-clock work time** (excludes waiting for exclusive resources); validation+probe+pilot ≤ 2 h |

## 2. Why it changed — and why this is not a result-driven choice

The reduction is driven by **measured processing and merge cost**, recorded
before any test number existed:

* candidate scoring measured at **0.41–0.49 items/s** (microbatch 6) on the
  uncompressed model;
* the uncached reference scorer measured at **0.036 items/s** (~11×, because
  `DeepseekMoE.moe_infer` loops over experts in Python);
* merge measured at **~23 min** per model (1391 s observed in the concurrent
  E2 job), requiring ~62 GB CPU RAM (28 GB Fisher + 34 GB whitening), which
  forces global serialization.

The original grid therefore extrapolated to **~47 h** on two GPUs. The
amendment cuts scope to fit a 12 h budget.

**No test result influenced any of these choices.** At the time of this
amendment:

* **zero** compressed-model scores existed, for any condition, seed, language
  or item;
* **zero** test-set predictions existed;
* **zero** dev accuracies existed (the earlier pilot was aborted during its
  scorer-validation phase and produced no accuracy numbers);
* the only test-set quantity ever computed was **gold-label support counts**
  for the 600-item subset, computed as part of defining the subset and of
  deciding that accuracy is primary and macro-F1 secondary. No model output
  was involved.

Dropping ZH/KO/SW/BN own-language conditions narrows the *question* (see §1 of
the revision) rather than selecting a favourable comparison: the retained
contrast, Balanced − English, was one of the two contrasts pre-registered in
the original protocol, and English-only is the paper's existing reference
condition.

## 3. Subset change: 600 → 200

The 200 items are a deterministic function of the source IDs alone:

    rank by SHA-256("massive-rebuttal-v2|42|" + canonical_source_id), ascending
    take the first 200

drawn from the **already-frozen 600-item common-test manifest**
(`results/massive/massive_test_ids.json`). `canonical_source_id` = decimal
string without leading zeros for numeric IDs (all MASSIVE IDs in the pool are
numeric; 0 non-numeric), original string otherwise. Anyone can reproduce the
subset from the ID list without trusting this run.

No label, prediction or score enters the selection. The same IDs are used in
all five languages.

The unused 400 items of the earlier manifest are **preserved, not deleted**,
as pre-amendment artifacts, as is the entire earlier `results/massive/` tree.

## 4. Reuse boundary

**Reused** (identical code, labels, dtype, prompt hashes):

* the official MASSIVE v1.1 snapshot, sha256
  `4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577`;
* the frozen 60-intent label mapping and both prompt prefixes (hashes in
  `protocol.json`);
* the reference-scorer validation performed *after* the candidate-token cache
  fix (see §5);
* the 8 fixed English demonstrations and the dev subset they came from.

**Separated on purpose:** `inference_signature` (model path, dtype, scorer
version, mode, microbatch, prompt hash, label map, score definition — anything
that changes a produced number) is kept distinct from `protocol_hash` (grid,
budget, subset rules). Shrinking 600 → 200 changes `protocol_hash` only, so
already-scored items would remain reusable; in fact none existed.

**Not reused / not re-run:** the full data audit in
`analysis/data_audit_0916/` is cited, never re-executed. No SIB/Belebele clean
re-evaluation, no XL-Sum backfill, no placebo reconstruction. The FLORES
placebo-derived noise floor is **not** imported as a significance criterion
for MASSIVE — the audit shows its non-overlap premise does not hold.

## 5. A scorer bug found and fixed before any result was produced

`CachedScorer.candidate_tokens_for` cached `(prompt_token_ids, completion_ids)`
keyed on the prompt's **token length**. MASSIVE utterances are short and
similar in length, so the 600-item test subset spanned only ~20 distinct
prompt token lengths and **~96 % of items would have been scored against a
different utterance's prompt** (dev: 86–90 %).

It did not look broken — every item still received a grammatical prompt, so
accuracies would have looked plausible and only a minority of argmax
predictions moved. It was found because the mandated reference-scorer
comparison disagreed on 4/12 items, and because two of this run's own
measurements contradicted each other on identical items.

Fix: prompt ids are always re-tokenized per item and never cached; only the
candidate *completion* ids are cached, justified by every prompt ending in the
identical `\nIntent:` boundary, and that premise is verified against the real
prompt on first use and re-verified every 50 items. Post-fix: **0/12 argmax
disagreements** in EN, ZH and KO (pre-fix: 4/12).

## 6. A measurement limit that survives the fix

This model's candidate scores move by **~0.3 nats** when nothing changes but
tensor shape: the uncached oracle differs from *itself* by 0.48 when every
candidate is padded to one common length. The mechanism is `moe_infer`'s
per-expert variable-row GEMM feeding a hard top-6-of-64 router, compounded over
27 MoE layers. The median top1–top2 decision margin on these prompts is
**~0.31**, i.e. the same size.

Consequences, reported rather than resolved:

* absolute MASSIVE accuracy on this model carries an implementation-dependent
  component;
* scoring is nevertheless **bit-identical on repeat**, and every cell uses
  identical prompt, label order, tensor shapes, batching and dtype, so
  model-vs-model contrasts are not biased by it — they gain variance;
* the paired bootstrap covers test-item sampling, **not** this component.

## 7. Status of the concurrent job on this host

The E2/01 job on physical GPUs 0–1 was **stopped by the user at
2026-09-16T04:29:07Z** (2 of 9 cells complete; the in-flight cell wrote no
partial artifact). This run neither started, stopped, nor resumed it, never
forced the shared merge lock, and does not use GPUs 0–1. Lock and process
state are checked live rather than predicted from earlier estimates.

## 8. Time accounting

Work time under this amendment is counted from **2026-09-16 ~04:36 UTC**,
excludes time blocked on the exclusive merge lock, and is wall-clock (two GPUs
used together count once). Time already spent under the superseded plan is
reported separately in `runtime.json` and is not charged against the 12 h.
