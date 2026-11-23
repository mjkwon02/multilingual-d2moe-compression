# Reporting rules for the clean re-evaluations (fixed 2026-09-16, user instruction)

These bind every table, summary and rebuttal sentence produced from
analysis/clean_eval_0916/. They change interpretation only — no re-run, no new
validation, no change to conditions, seeds, prompts, demonstrations or
evaluation IDs.

## 1. full → clean differences are descriptive, not causal

Dropping exposed items also changes the difficulty mix and the label
distribution of what remains. A full→clean difference therefore cannot be
attributed to exposure.

- Write: "accuracy on the clean subset is lower/higher than on the full subset."
- Do NOT write: "exposure inflated the score", or any equivalent causal claim.
- Collecting all 3 seeds does NOT resolve this; it is a property of changing
  the item set, not of sampling noise. Do not promise that more seeds will
  settle it.

## 2. Never compare the full→clean difference against the reproduction difference

They are different comparisons with different denominators and different
referents:

- full → clean: two DIFFERENT item sets (98 vs 204 for SIB), same run.
- reproduction: the SAME 204 items, two DIFFERENT runs (stored vs re-run).

Reporting them on one scale ("4 items either way, so indistinguishable") is
invalid and is removed wherever it appeared.

Record the reproduction difference as its own line item, per cell. Its cause is
**not established**. Do not state bf16 reduction order, GPU sharding or batch
shape as the cause — the original runs are logged with differing GPU counts,
which is circumstantial only, and no experiment was run to confirm it. No such
experiment is being requested.

## 3. Scope of the current SIB claim

Until all seeds are in, the claim is limited to what has actually been
observed, currently english_only seed 0:

> Against the same clean baseline, EN and ZH are largely preserved, while KO,
> SW and BN show large degradation.

Balanced-vs-English conclusions and any across-seed statement wait for those
cells.

BN on SIB is phrased as:

> 11.2%, which is below the 7-way uniform-chance rate of 14.3% and meets this
> project's existing accuracy-based collapse criterion.

Not "at random level" — no test of equality-to-chance was performed.

## 4. Belebele clean subsets stay per-language

Clean n is per language and is kept as measured: EN 99, ZH 99, SW 99, KO 113,
BN 125 (passage counts 52/52/52/59/65).

- Within a language, every model must be scored on the identical question set —
  verified from the stored per-item `qkey` records at aggregation time.
- Across languages the question sets differ, because the Belebele split order
  differs per language and every reported cell used `--limit 200`. State this
  as a limitation. Do NOT re-evaluate onto a cross-language common subset.
- SW/BN clean-vs-full increases are reported as differences that also carry a
  change in item composition. Do NOT declare them "within sampling noise" on
  the basis of a standard error alone.

## 5. Standing content of the 3-seed report

absolute accuracy; retention against the SAME clean baseline; per-seed values
plus mean and sample std (ddof=1); top-label share and the collapse
determination; per-cell flags for re-aggregated / re-evaluated / not-run.
