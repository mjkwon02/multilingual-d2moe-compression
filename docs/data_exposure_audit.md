# Data exposure audit: calibration vs. evaluation overlap

This document separates the **original (legacy, exposed)** downstream
evaluation protocol from the **corrected (exposure-filtered)** protocol, and
records exactly what was found. Full machine-readable evidence is in
`results/source_data/data_exposure_audit/` (`overlap_summary.csv`,
`overlap_evidence.jsonl.gz`, `calibration_reconstruction.json`,
`data_audit_report.md`).

## What the audit established

Calibration text is drawn from FLORES-200 devtest (1,012 sentences per
language); the first 60 sentences per language are reserved for evaluation
and the calibration pool is the remaining 952.

- **FLORES bits-per-byte (the paper's main metric): no overlap of any kind.**
  Zero exact matches, zero normalized matches, zero translation-parallel
  source IDs, zero overlap with the character windows the Fisher and
  whitening stages actually consume — across all six calibration conditions
  and three seeds, and for the four unseen evaluation languages.
- **XL-Sum out-of-domain bpb: disjoint on every axis tested.** The corpus
  files carry a SHA-256 build manifest (`data/ood_xlsum/manifest.json`).
- **SIB-200 and Belebele: real overlap, reported and corrected.** Both are
  derived from FLORES-200 and their split boundaries are not aligned with the
  60-sentence hold-out. Recovered source IDs show 106/204 SIB-200 test items
  (52.0%) and 351/900 Belebele questions (39.0%) per language originate from
  calibration-pool sentences. 6 of the 14 fixed SIB-200 demonstrations are
  also drawn from calibration-pool sentences.

## Corrected protocol (`results/downstream/exposure_filtered/`) — the default

A common clean subset removes, for **all** conditions and seeds jointly, the
union of affected items (by FLORES source ID, applied across all five
evaluation languages; Belebele filtered at passage level):

- SIB-200: **98 of 204** items retained (all 7 categories represented,
  majority class 26.5%).
- Belebele: **549 of 900** questions retained (all 4 answer positions,
  123–154 each).

We deliberately name this **exposure-filtered test**, not "clean": the
14 fixed SIB-200 **demonstrations were kept unchanged** (including the 6
calibration-pool ones), so demonstration exposure persists by design and is
reported separately from test-item exposure. Clean subsets are roughly half
the original size, so filtered accuracies carry wider uncertainty.

Item ID lists: `data/splits/clean_eval_ids.json`.
Per-model re-evaluations: `results/downstream/exposure_filtered/deepseek_clean_eval/`
and `results/qwen/sib200_clean/`.

## Legacy protocol (`results/downstream/legacy_exposed/`) — kept, not deleted

The original full-test-set results (SIB-200 204 items, Belebele 900
questions) are preserved for transparency and for comparison with the
submitted paper's original numbers. Do not cite these as contamination-free.

### Known issue in the legacy Belebele harness (disclosed, not hidden)

The vendored lm-evaluation-harness Belebele task configs used for the legacy
runs set `test_split: "test"` and `fewshot_split: "test"` with a `first_n`
sampler, which means **the 5-shot demonstrations were drawn from the test
split itself**. This affects the legacy Belebele numbers only. Code that
reproduces the legacy runs is marked `legacy_exposed` in
`docs/provenance/d2moe_patch_manifest.md`; the exposure-filtered Belebele
evaluation (`scripts/evaluation/belebele_clean_eval.py`) re-evaluates from
per-item predictions and is the protocol README commands use.

## What this audit does not claim

It says nothing about whether the base models' **pre-training** corpora
contained FLORES-200, SIB-200, Belebele or XL-Sum, which we cannot check.
Switching the evaluation corpus also does not answer whether the
calibration-language effect reproduces when the *calibration* corpus itself
is replaced.
