# Experiment ↔ code ↔ results mapping

Maps every experiment reported in the paper (*Tracing Calibration-Language
Effects in D²-MoE Compression*), including the rebuttal additions, to its
script(s), configuration, and preserved result files, with a verification
status. Complements `docs/experiment_registry.md` (file-level registry) and
`manifests/verification_report.md` (numeric verification evidence).

## Status vocabulary

| Status | Meaning |
|---|---|
| `FULLY_VERIFIED` | Entire pipeline re-run from scratch (incl. GPU) and results confirmed. **No experiment carries this status**: GPU stages were not re-run during release preparation. |
| `ANALYSIS_REPRODUCED` | Aggregation/table generation re-run from the preserved per-seed result files and verified numerically identical to the legacy results (2026-09-28 full comparison; re-run 2026-11-23). |
| `SMOKE_TESTED` | Script executes end-to-end without error on the preserved inputs; outputs produced but not compared value-by-value beyond spot checks. |
| `CODE_ONLY` | The exact code that produced the results is included and statically checked, but was not executed here (requires GPU and/or deleted tensors). |
| `PARTIAL` | Code present and some results preserved, but part of the result grid or a sub-analysis is missing; the gap is stated in the row. |
| `MISSING` | Neither runnable code nor results could be located. |
| `UNVERIFIED` | Present but no verification of any level was performed. |

Seeds are 0/1/2 everywhere unless noted. "cond" ranges over
`english_only, chinese_only, korean_only, swahili_only, bengali_only,
mixed_5lang` (the paper's Balanced = `mixed_5lang`, equal **character**
budget mixed-sequence construction; the separate token-balanced arm is
`balanced_tok_mono`).

## A. Main calibration experiment (DeepSeek-MoE-16B)

| Experiment | Paper section | Script | Config | Result source | Status |
|---|---|---|---|---|---|
| Calibration construction (FLORES-200, 64 windows × 2,048 chars, ≤512 tokens) | §4 / App. A | `scripts/calibration/phase1_calib_data.py` | in-script constants; `configs/languages.yaml` | deterministic from seed (determinism re-checked 2026-11-23) | ANALYSIS_REPRODUCED |
| Fisher / whitening / SVD / merge (delta_ratio 0.8), per cond × seed | §4–5 / App. B | `scripts/compression/phase1_fisher.py`, `phase1_svd_scale.py`, `phase1_run_freq_and_scale.py`, `phase1_merge_eval.py`; driver `run_phase1_41_diagonal.py` | `configs/deepseek/device_map.json` | per-seed `results/deepseek/<cond>/seed*/scale_*/eval_ppl.json` (tensors deleted; ~62 GB/cell) | CODE_ONLY (GPU re-run), outputs preserved |
| Δbpb% grid + headline gate (Table 1, App. Table 13) | §5.1 | `scripts/analysis/phase1_41_headline_gate.py` | — | `results/deepseek/phase1_41_headline_gate_result.json` | ANALYSIS_REPRODUCED (re-verified numerically identical to legacy raw copy on 2026-11-23; Own/Balanced rows match the OpenReview table exactly) |
| Placebo arms (shuffled text) | App. D (Table 15; noise floors feed Table 13) | same pipeline, conds `*_only_placebo`; `_make_p1_5.py`, `phase1_placebo_gate.py` | — | `results/deepseek/*_only_placebo/`, `p1_5_placebo_zh_bn.json` | ANALYSIS_REPRODUCED (gate level); placebo Fisher tensors deleted 2026-08-02 → upstream recompute CODE_ONLY |
| Seed / placebo robustness gates | §5 robustness | `phase1_{seed,placebo}_gate.py` | — | `results/deepseek/phase1_*_gate_result.json` | ANALYSIS_REPRODUCED (the layout-stale legacy `phase1_bengali_gate.py` / `phase1_swahili_gate.py` were moved to the private archive 2026-11-23; their quantities are produced and verified by the headline gate — see `docs/limitations.md` #5) |

## B. Stage-separation (Fisher × Whitening 2×2)

| Experiment | Paper section | Script | Config | Result source | Status |
|---|---|---|---|---|---|
| EN × KO 2×2 (whitening 61.44 pp vs Fisher 23.46 pp) | §5.4 / B.6 (Table 9) | `phase1_2x2_gate.py`, `phase1_2x2_verify_gate.py`; GPU driver `run_phase1_2x2*.py` | — | `results/deepseek/phase1_2x2_gate_result.json`, `phase1_2x2_verify_gate_result.json` | ANALYSIS_REPRODUCED (3-seed symmetric scale effect recomputes to 61.43 pp) |
| EN × SW 2×2 (whitening 23.85 pp vs Fisher 9.29 pp) | B.6 (Table 10) | `phase1_2x2_ensw_effects.py` | pre-registration quoted inside the result JSON | `results/source_data/table_b6_2x2_ensw_effects.json`, `p1_4_stage2x2_ensw.json` | ANALYSIS_REPRODUCED (23.8547 / 9.2854 pp) |
| Structured pruning on/off (pp_ratio 0.2) | §5.4 / App. B.4 (Table 8 — the script/file are named "table4" from the submission numbering) | `phase1_structured_prune.py` (GPU), `make_table4_pruning_lang.py`, `phase1_pruning_gate.py` | — | `results/deepseek/<cond>/seed*/.../pp_ratio_0.2/eval_ppl.json`, `table4_pruning_lang.json` | ANALYSIS_REPRODUCED (aggregation); GPU stage CODE_ONLY |

## C. Internal analysis (DeepSeek only — mechanism claims do not extend to Qwen)

| Experiment | Paper section | Script | Config | Result source | Status |
|---|---|---|---|---|---|
| Fisher importance rank correlations | App. F.1–F.3 (Tables 18–19), Fig 2A | `phase1_51_fisher_rank_correlation.py`, `phase1_51b_merge_weight_rank_correlation.py` | — | scalar caches `results/deepseek/fisher_expert_scalar*_cache/` + `phase1_51*_result_seed*.json` | PARTIAL — full-condition correlations reproduce identically from the caches; the **placebo-comparison section is not recomputable** (placebo Fisher tensors deleted 2026-08-02); preserved per-seed JSONs are authoritative |
| Placebo rank ρ (ZH/BN) | App. F.4 (Table 20) | `placebo_fisher_rank_zh_bn.py`, `placebo_switch1_zh_bn.py` | — | `results/source_data/placebo_*` (unique copies — `docs/provenance/placebo_unique_results.md`) | PARTIAL — preserved derived files only; not recomputable |
| Routing coverage deficiencies | §6.2 / App. E (Tables 16–17), Fig 2B | `scan_disagreement_experts.py` (GPU), `phase1_52{,b,c}_*.py` | — | `results/deepseek/aux/scan_results.json`, `phase1_52*_result.json` | ANALYSIS_REPRODUCED (analysis); scan CODE_ONLY |
| Activation covariance / whitening geometry, principal angles, between/within ratios, permutation tests | §6.3 (Table 4), Fig 2C, App. G (Figures 3–5; output files `figure4_*`, `figureG1_*`, `figureG2_*` — see `docs/PAPER_FIGURE_TABLE_MAPPING.md`) | `phase1_whitening_geometry.py` (GPU), `phase1_whitening_geometry_analysis.py`, `scripts/figures/make_figure{3,4,_whitening_geometry}*.py` | — | `results/source_data/whitening_geometry_pairs.csv` + analysis JSON/NPZ | ANALYSIS_REPRODUCED (analysis + figures regenerate; the submission-only overview figure was removed from the tree 2026-11-23 — archived privately — and its Panel-B provenance `agg_covariance_spectra.npz` is kept, constants verified exact before removal) |

## D. Expert restoration

| Experiment | Paper section | Script | Config | Result source | Status |
|---|---|---|---|---|---|
| Target selection (high-disagreement + matched low-disagreement control) | §6.4 | `phase1_sec54_select_targets.py` | selection rules in-script (pre-registered) | `results/deepseek/sec54/` | ANALYSIS_REPRODUCED |
| Bundle / individual / full-layer restoration (GPU interventions, incl. integrity check of restored original weights) | §6.4 (Table 5), App. H (Tables 21–22; the recovery script is named "tableH1" from the submission labels) | `phase1_sec54_intervene.py` (GPU), `make_table5_sec54_bundle.py`, `make_tableH1_sec54_recovery.py` | — | `results/deepseek/sec54/`, `table5_sec54_bundle.json` (72 values), `tableH1_sec54_recovery.json` (188 values) | ANALYSIS_REPRODUCED (tables); interventions CODE_ONLY |

Pre-registered vs. exploratory: the gate scripts embed their
pre-registration text/logs (e.g., the EN/SW 2×2 JSON quotes its
mini-preregistration verbatim); analyses without a quoted registration are
exploratory follow-ups.

## E. Additional robustness

| Experiment | Paper section | Script | Config | Result source | Status |
|---|---|---|---|---|---|
| Character- vs token-balanced calibration | App. A.1–A.2 (Tables 6–7), App. C (Table 11), §5.5 / App. K (Tables 31/32/35) | `phase1_6_robust_mix_calib_tokens.py`, `phase1_a2_balanced_token_share.py`, `phase1_tokens_per_byte_all_langs.py`, `e1_token_balanced_design.py` | — | `balanced_tok_mono/` evals, `calib_token_length_results.json`, `balanced_token_share_*` | ANALYSIS_REPRODUCED |
| XL-Sum OOD bpb, 4 conditions + unseen languages (AR/HI/TR/RU) | §5.5, App. K (camera-ready Tables 33–34; rebuttal-era docs/scripts call the XL-Sum grid "Table 31") | `e2_build_ood_corpus.py`, `e2e3_ood_unseen_eval.py` (GPU), `make_external_bpb_grid_0916.py` | — | `data/ood_xlsum/` (SHA-256 manifest), `eval_ood_unseen.json`, `external_bpb_grid.csv` | ANALYSIS_REPRODUCED for the reported 4-condition grid |
| XL-Sum extension to all 6 conditions (rebuttal §C-2) | rebuttal (not citable) | same | — | `scripts/analysis/data_audit_0916/external_bpb_missing_cells.json` | PARTIAL — **7 of 9 added cells were never produced** (chinese_only seed2; swahili_only and bengali_only all seeds). Not imputed; do not cite a 6-condition external grid |
| Disagreement-targeted budget reallocation | §6 | `phase1_6_targeted_budget.py`, pipeline unchanged; gates `phase1_6_budget_gate{, _v2}.py` | — | `disagreement_targeted{,_cap50}/` (3 seeds, eval files present), gate result JSONs | ANALYSIS_REPRODUCED (gates; v1 verdict NOT_SUPPORTED preserved as reported) |
| Vulnerability-targeted / robust-mix (whitening mixture) arms | exploratory | same pipeline | — | `vulnerability_targeted*/seed*/budget_allocation.json`; `robust_mix/seed0/` frequency file only | PARTIAL — allocations preserved; per-seed eval files absent; `robust_mix` has seed0 frequency stats only |
| Synthetic surrogate diagnosis | design appendix | (analysis preserved) | — | `results/source_data/{mass_diagnosis,relfrob_mixed_vs_lang}.csv` | PARTIAL — derived CSVs preserved; generating sweep code lives in the legacy git commit `a51215d`, not in this tree |
| Fisher Pilot A (design validation) | pre-paper design, no paper item | `fisher_pilot_a.py` (kept: its `LAYER_ROLE`/capture helpers are imported by `phase1_51_fisher_rank_correlation.py`, `phase1_52_deficiency_layer_distribution.py`, `phase1_6_targeted_budget.py`, `scan_disagreement_experts.py`) | — | pilot analysis script + `aux/pilot_a_*.json` moved to the private archive 2026-11-23 | ARCHIVED (supplementary; shared constants remain in-tree) |

## F. Rebuttal additions

| Experiment | Paper section | Script | Config | Result source | Status |
|---|---|---|---|---|---|
| **Qwen1.5-MoE-A2.7B cross-model grid** (6 cond × 5 lang × 3 seeds, FLORES Δbpb%) | §5.6 / App. L (Table 36) | `scripts/compression/qwen15_{fetch,freq,fisher,svd_scale,merge_eval}.py` (GPU; `qwen15_config.py`, `qwen15_moe_adapter.py`), `scripts/analysis/qwen15_table1_aggregate.py` | `qwen15_config.py` (bf16, 2 GPUs, pinned revision) | `results/qwen/phase1/<cond>/seed*/`, `report/table1_qwen_seeds012.json` — complete 6×5×3 grid, `complete: true`, no problems | ANALYSIS_REPRODUCED (Own / Balanced rows match the OpenReview table exactly: Own 8.50/2.80/2.75/0.79/2.70, Balanced 9.70/4.32/4.05/3.58/4.77) |
| **Exposure / contamination audit** (SIB-200 106/204 items, 6/14 demos; Belebele 351/900 questions, 202/488 passages) | rebuttal / revised audit section | `audit_calib_eval_overlap.py`, `make_clean_eval_ids_0916.py`, `make_audit_manifest_0916.py` | dataset versions in `docs/REPRODUCIBILITY.md` | `results/source_data/data_exposure_audit/` (`overlap_summary.csv`, `overlap_evidence.jsonl.gz`, `data_audit_report.md`), clean-ID manifest `data/splits/clean_eval_ids.json` | ANALYSIS_REPRODUCED (ID manifests + evidence preserved; re-deriving from fresh downloads = CODE_ONLY, needs dataset download) |
| **SIB-200 exposure-filtered re-evaluation** (98 test items, 14 demos kept — 6 exposed; revised Table 2 in absolute accuracy, mean ± SD, [k/3] collapsed) | §5.2 (Table 2); Qwen: App. L (Table 37) | `sib200_clean_eval.py` (GPU), `_make_table2_accuracy.py`, `qwen15_sib200_clean_{eval,aggregate}.py` | demos/protocol pinned in result JSONs | `results/downstream/exposure_filtered/{deepseek_clean_eval,sib200_accuracy_tables}/`, `results/qwen/sib200_clean/` (per-item predictions included) | ANALYSIS_REPRODUCED (both models; Qwen Balanced = 0/15 collapsed cells, matching the rebuttal claim; 204-item legacy results kept separately under `legacy_exposed/`) |
| **Belebele passage-filtered re-evaluation** (per-language n: EN 99, ZH 99, KO 113, SW 99, BN 125) | rebuttal evidence only — the camera-ready App. I.2 (Table 29) prints the **legacy n=200 5-shot** evaluation; see `docs/PAPER_FIGURE_TABLE_MAPPING.md` §4 | `belebele_clean_eval.py`, `_make_belebele_clean.py` | — | `results/downstream/exposure_filtered/{deepseek_clean_eval/belebele,belebele_tables}/` | ANALYSIS_REPRODUCED (per-language n recorded in every table; legacy 900-item results kept separately; legacy few-shot-from-test issue disclosed in `docs/data_exposure_audit.md`) |
| **MASSIVE cross-dataset evaluation** (intent classification, 200 sampled test items/lang, calibration-overlap-checked) | §5.2 / App. I.3 (Table 30) | `massive_revised_{protocol,run,aggregate,overlap}.py` + `massive_data.py` | `results/downstream/massive_revised/protocol.json` (+ documented `protocol_amendment.md`) | `results/downstream/massive_revised/` (`summary.json`, per-seed metrics, predictions, confusion matrices, demo/test ID lists in `data/splits/`) | ANALYSIS_REPRODUCED (KO baseline 33.5 %, Balanced 17.8 %, English 4.8 %, direction consistent across 3 seeds; Balanced **below** English calibration on EN/ZH — reported as such) |

## G. Table/figure numbering — RESOLVED against the camera-ready PDF

**Resolved 2026-11-23**: the final camera-ready PDF was audited and every
figure/table number in this repository's user-facing docs now follows it.
The authoritative correspondence (all 5 figures, all 37 tables, plus the
filename↔number mismatches that were deliberately left in place) is
**`docs/PAPER_FIGURE_TABLE_MAPPING.md`**.

Historical context: script/file names and older internal records carry the
*submission* numbering (Table 3 = SIB-200, Table 4 = pruning, Table 6 =
stability, Fig 1 = overview, Fig 3/4/G1/G2 = geometry figures) or the
*rebuttal* numbering (Table 31 = XL-Sum). Camera-ready equivalents:
SIB-200 clean = **Table 2**, stability = **Table 3**, pruning =
**Table 8**, XL-Sum = **Table 33**, geometry figures = **Figures 2–5**,
teaser = **Figure 1** (the submission overview figure `fig1_overview` was
dropped from the paper and is kept as LEGACY). The Qwen tables are
camera-ready **Tables 36–37**; §5.6's references resolve correctly in the
final PDF. Filenames were not renamed — scripts, result paths, and the
checksum history depend on them.

## H. Precision note: the Korean 2×2 whitening effect (61.44 vs 61.43 pp)

Recomputed 2026-11-23 from the raw per-seed cross-scale `eval_ppl.json`
cells: the symmetric whitening (scale) effect for Korean is **61.4306 pp**
and the Fisher effect **23.4606 pp** — identical under both aggregation
orders (per-seed Δbpb% then mean, or mean bpb then Δbpb%), so the
full-precision value rounds to **61.43**. The paper's **61.44**
is reproducible only by first rounding the two per-row mean gains to two
decimals (83.92 and 38.95) and averaging ((83.92+38.95)/2 = 61.435 → 61.44).
The **camera-ready PDF prints 61.44** (§5.4 and B.6, checked 2026-11-23),
i.e. the 2-decimal-intermediate convention was kept; this is a cosmetic
0.01 pp presentation difference, flagged to the authors in
`docs/PAPER_FIGURE_TABLE_MAPPING.md` §4. The Fisher effect 23.46 matches at
full precision. No stored result was altered.

## I. Tables and figures — generation commands

See the command table in `docs/REPRODUCIBILITY.md` (Level 1) and the
per-figure/per-table mapping in `docs/PAPER_FIGURE_TABLE_MAPPING.md`. All commands
listed there were last re-run successfully on 2026-11-23
(`tests/smoke_test.sh`), and the 2026-09-28 numeric comparison against the
legacy result files is recorded in `manifests/verification_report.md`.
