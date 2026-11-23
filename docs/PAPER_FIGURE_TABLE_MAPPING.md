# Camera-ready figure/table ↔ repository mapping

Authoritative mapping between the **camera-ready PDF** of *Tracing
Calibration-Language Effects in D²-MoE Compression* (audited against the
final PDF on 2026-11-23) and this repository. Figure/table
numbers below are the **final paper numbers**. File and script names in
this repository were frozen during the research and sometimes carry
*older* (submission- or rebuttal-era) numbers; they were **deliberately
not renamed** (renaming would break script constants, result paths, and
the checksum history). When a filename contains a number, trust this
document, not the filename.

Classification used below:

- **PAPER_MAIN** — used in the camera-ready main body (§1–8)
- **PAPER_APPENDIX** — used in the camera-ready appendix (A–L)
- **SUPPLEMENTARY** — not in the camera-ready, but a related analysis
  kept as part of the research record
- **LEGACY** — used only by an earlier (submission) version of the paper

## 1. Figures (camera-ready has exactly 5)

| Paper figure | Content | Generating script | Output file(s) | Source data | Class |
|---|---|---|---|---|---|
| **Figure 1** (§1) | Teaser: calibration-language gap, English vs Balanced calibration, per-language Δbpb% | `scripts/figures/make_figure_teaser_calibration_gap.py` | `figures/fig_teaser_calibration_gap.{pdf,png}` | in-script constants = the Table 1 English/Balanced rows (7.28…89.02 / 9.29…2.90), verified against `results/deepseek/phase1_41_headline_gate_result.json` | PAPER_MAIN |
| **Figure 2** (§6) | Layer-wise traces, 3 panels: (A) Fisher-rank Spearman ρ vs placebo, (B) coverage deficiencies by layer, (C) between/within ratios | `scripts/figures/make_figure3_whitening_evidence.py` | `figures/fig3_layerwise.{pdf,png}` ⚠️ filename says "fig3" | A: `results/deepseek/phase1_51_fisher_rank_correlation_result_seed0.json`; B: `results/deepseek/phase1_52b_deficiency_pairs_list_result.json`; C: `results/source_data/whitening_geometry_pairs.csv` | PAPER_MAIN |
| **Figure 3** (App. G) | Within- vs between-condition distributions (mean principal angle k=64; covariance rel. Frobenius distance) | `scripts/figures/make_figure4_whitening_distributions.py` | `figures/figure4_whitening_distributions.{pdf,png}` ⚠️ filename says "figure4" | `results/source_data/whitening_geometry_pairs.csv` | PAPER_APPENDIX |
| **Figure 4** (App. G) | Layer-wise raw curves of the mean principal angle (between vs within, late layers marked) | `scripts/figures/make_figure_whitening_geometry.py` | `figures/figureG1_whitening_layer_curve_raw.{pdf,png}` ⚠️ filename says "G1" | `results/source_data/whitening_geometry_analysis_layer_curve.json` | PAPER_APPENDIX |
| **Figure 5** (App. G) | Heatmap: mean principal angle between whitening bases for all condition pairs (diagonal = within-condition) | `scripts/figures/make_figure_whitening_geometry.py` | `figures/figureG2_whitening_heatmap.{pdf,png}` ⚠️ filename says "G2" | `results/source_data/whitening_geometry_analysis_heatmap.npz` | PAPER_APPENDIX |

### Repository figure files not in the camera-ready

None — the tree now contains exactly the five camera-ready figures.
The submission version's Figure 1 (`fig1_overview`, a 3-panel overview:
RQ1 damage heatmap / RQ2 whitening-rotation schematic / RQ3 restoration
recovery bars) was dropped from the camera-ready in favour of the teaser
and was **removed from the public tree on 2026-11-23** (figure + its
generator script archived verbatim in the private provenance archive,
`release_prep_20261123/fig1_overview_legacy/`). Its Panel-B provenance
data, `results/source_data/agg_covariance_spectra.npz`, **stays in the
tree**: it is a unique, non-regenerable derived record (the aggregated
activation covariances it was extracted from were not preserved) and is
independent of the removed script.

Filename ↔ paper-number mismatches (all three are *kept* on purpose):
`fig3_layerwise` → paper **Figure 2**; `figure4_whitening_distributions` →
paper **Figure 3**; `figureG1_*` / `figureG2_*` → paper **Figures 4/5**.
These exact paths are what the camera-ready .tex `\includegraphics`
references (visible verbatim in the draft-mode placeholders of the
camera-ready PDF), so renaming the files would break the paper build —
another reason the names stay.

## 2. Tables (camera-ready has exactly 37)

"Generator" = the no-GPU script that reproduces the table's numbers from
the preserved per-seed results (all generators below are exercised by
`tests/smoke_test.sh` unless noted).

### Main body

| Paper table | Content | Generator | Output / result source | Class |
|---|---|---|---|---|
| **Table 1** (§5.1) | Δbpb% grid, 6 calibration × 5 evaluation languages, 3-seed means | `scripts/analysis/phase1_41_headline_gate.py` | `results/deepseek/phase1_41_headline_gate_result.json` | PAPER_MAIN |
| **Table 2** (§5.2) | SIB-200 absolute accuracy on 98 exposure-filtered test items (14 demos kept), mean±SD, [k/3] collapsed | `results/downstream/exposure_filtered/sib200_accuracy_tables/_make_table2_accuracy.py` | `results/downstream/exposure_filtered/sib200_accuracy_tables/` (per-seed raw: `table2_raw_per_seed.csv`) | PAPER_MAIN |
| **Table 3** (§5.3) | Stability of Balanced vs single-language (regret, worst-language, collapses, seed std) | `results/source_data/tables_main_submission/_make_p0_3.py` | `results/source_data/tables_main_submission/p0_3_table6_source.json` ⚠️ filename says "table6" (submission number) | PAPER_MAIN |
| **Table 4** (§6.3) | Between/within ratios per whitening-geometry metric, overall + layer bands | `scripts/analysis/phase1_whitening_geometry_analysis.py` | `results/source_data/whitening_geometry_analysis.json` | PAPER_MAIN |
| **Table 5** (§6.4) | Restoration recovery (pp), treatment vs control bundles, 3 runs | `scripts/analysis/make_table5_sec54_bundle.py` | `results/source_data/table5_sec54_bundle.json` | PAPER_MAIN |

### Appendix

| Paper table | Content | Generator / source script | Output / result source | Class |
|---|---|---|---|---|
| **Table 6** (A.1) | Realized calibration token lengths per condition | `scripts/calibration/phase1_6_robust_mix_calib_tokens.py` | `results/source_data/calib_token_length_results.json`, `realized_tokens.json` | PAPER_APPENDIX |
| **Table 7** (A.2) | Token-space composition of the Balanced condition, per seed | `scripts/calibration/phase1_a2_balanced_token_share.py` | `results/source_data/balanced_token_share_seed{0,1,2}.json` | PAPER_APPENDIX |
| **Table 8** (B.4) | Structured-pruning additional damage ΔΔbpb% (ON−OFF, pp_ratio 0.2) | `scripts/analysis/make_table4_pruning_lang.py` ⚠️ script says "table4" (submission number) | `results/source_data/table4_pruning_lang.json` (+ `tables_main_submission/_make_p0_2.py` for the recomputed chinese_only row) | PAPER_APPENDIX |
| **Table 9** (B.6) | {EN,KO} 2×2 Fisher × whitening cell means (KO evaluation) | `scripts/analysis/phase1_2x2_gate.py`, `phase1_2x2_verify_gate.py` | `results/deepseek/phase1_2x2_gate_result.json`, `phase1_2x2_verify_gate_result.json` | PAPER_APPENDIX |
| **Table 10** (B.6) | {EN,SW} 2×2 per-seed Swahili-column Δbpb% | `scripts/analysis/phase1_2x2_ensw_effects.py` | `results/source_data/table_b6_2x2_ensw_effects.json`, `tables_main_submission/p1_4_stage2x2_ensw.json` | PAPER_APPENDIX |
| **Table 11** (C) | Tokenization granularity (tokens/byte) on the FLORES eval set | `scripts/calibration/phase1_tokens_per_byte_all_langs.py` | `results/source_data/phase1_tokens_per_byte_all_langs.json` | PAPER_APPENDIX |
| **Table 12** (C) | Per-language baseline bpb of the uncompressed model | (values read by every Δbpb aggregation) | `results/deepseek/baseline/eval_ppl.json` | PAPER_APPENDIX |
| **Table 13** (D) | Headline own-gain verdicts (gains, noise floors, ×floor) | `scripts/analysis/phase1_41_headline_gate.py` | `results/deepseek/phase1_41_headline_gate_result.json` | PAPER_APPENDIX |
| **Table 14** (D) | Per-seed Δbpb% for every Table 1 cell, with seed SD | `scripts/analysis/phase1_41_headline_gate.py` | same gate result (per-seed section) | PAPER_APPENDIX |
| **Table 15** (D) | Placebo-condition per-seed Δbpb% (own-language diagonal) | `results/source_data/tables_main_submission/_make_p1_5.py` | `results/source_data/tables_main_submission/p1_5_placebo_zh_bn.json`, `results/deepseek/*_only_placebo/` | PAPER_APPENDIX |
| **Table 16** (E) | Full list of flagged (layer, expert) deficiency pairs (threshold 20) | `scripts/analysis/phase1_52b_deficiency_pairs_list.py` | `results/deepseek/phase1_52*_result.json`, `results/deepseek/aux/scan_results.json` | PAPER_APPENDIX |
| **Table 17** (E) | Threshold sensitivity (15/20/25/30) of the deficiency counts | `scripts/analysis/phase1_52c_deficiency_robustness_check.py` | `results/deepseek/phase1_52*_result.json` | PAPER_APPENDIX |
| **Table 18** (F.1) | Per-pair Spearman ρ of Fisher expert rankings + placebo rows (seed 0) | `scripts/analysis/phase1_51_fisher_rank_correlation.py` | `results/deepseek/phase1_51_fisher_rank_correlation_result_seed*.json` (placebo section **not recomputable** — tensors deleted 2026-08-02; preserved JSONs authoritative) | PAPER_APPENDIX |
| **Table 19** (F.2) | Seed stability of the headline ρ (language-pair vs placebo, front/back half) | same | same | PAPER_APPENDIX |
| **Table 20** (F.4) | ZH/BN placebo rank correlations (raw Fₑ and merge-weight definitions) | `scripts/analysis/placebo_fisher_rank_zh_bn.py`, `placebo_switch1_zh_bn.py` | `results/source_data/placebo_fisher_rank_zh_bn.{json,md}`, `placebo_switch1_zh_bn.json` (unique preserved copies — `docs/provenance/placebo_unique_results.md`) | PAPER_APPENDIX |
| **Table 21** (H.1) | Intervention target list (12 treatment + 12 control pairs) | `scripts/analysis/phase1_sec54_select_targets.py` | `results/deepseek/sec54/` | PAPER_APPENDIX |
| **Table 22** (H.5) | Recovery rates by restoration scope (T late/early, T+C 24, full layers) | `scripts/analysis/make_tableH1_sec54_recovery.py` ⚠️ script says "H1" (submission label) | `results/source_data/tableH1_sec54_recovery.json` | PAPER_APPENDIX |
| **Table 23** (I.1) | Original SIB-200 retention on 204 items (pre-filtering) | `scripts/analysis/sib200_extended_report.py` | `results/deepseek/sib200_extended/SIB_EXTENDED_REPORT.md` | PAPER_APPENDIX |
| **Tables 24–28** (I.1) | Per-seed SIB-200 raw data (204 items), one table per language EN/ZH/KO/SW/BN | `scripts/analysis/sib200_extended_report.py` | `results/deepseek/sib200_extended/<cond>_seed*.json` | PAPER_APPENDIX |
| **Table 29** (I.2) | Belebele retention, **legacy 5-shot n=200 protocol**, SW/BN floor-excluded | `scripts/analysis/phase1_belebele_gate.py` (per-condition invocations) | `results/deepseek/<cond>/.../fewshot_5/eval_belebele.json`, `results/downstream/legacy_exposed/` — see the note in §4 below | PAPER_APPENDIX |
| **Table 30** (I.3) | MASSIVE v1.1 intent-classification accuracy, 200 items/lang, zero-shot | `scripts/evaluation/massive_revised_aggregate.py` | `results/downstream/massive_revised/` | PAPER_APPENDIX |
| **Table 31** (K.2) | FLORES Δbpb% for the token-balanced variant (E1 grid, incl. ∆∆ row) | rows reproduced from `results/deepseek/balanced_tok_mono/seed*/scale_*/eval_ppl.json` + Table 1 cells; ranking/regret via `scripts/analysis/make_external_bpb_grid_0916.py` | `results/deepseek/balanced_tok_mono/`, design in `results/source_data/e1_token_balanced_design.json` | PAPER_APPENDIX |
| **Table 32** (K.2) | Preregistered per-cell E1 verdicts (SHIFTED / NO VERDICT) | verdict = ∆∆ (Table 31) vs 2× placebo noise floor (Table 13); noise floors **not recomputable** (placebo-arm outputs deleted — stated in paper K.3) | preserved values; thresholds from `phase1_41_headline_gate_result.json` | PAPER_APPENDIX |
| **Table 33** (K.2, E2) | Out-of-domain Δbpb% on XL-Sum ⚠️ repo docs/scripts of rebuttal vintage call this "Table 31" | `scripts/analysis/make_external_bpb_grid_0916.py` | per-cell `eval_ood_unseen.json`; corpus manifest `data/ood_xlsum/manifest.json` (text rebuilt via `e2_build_ood_corpus.py`) | PAPER_APPENDIX |
| **Table 34** (K.2, E3) | Δbpb% on four unseen languages (AR/HI/TR/RU) | `scripts/analysis/make_external_bpb_grid_0916.py` | per-cell `eval_ood_unseen.json` | PAPER_APPENDIX |
| **Table 35** (K.2) | SIB-200 accuracy for E1 (204 items, 4 conditions incl. token-balanced) | `scripts/analysis/sib200_extended_report.py` (4 main rows) + `results/deepseek/balanced_tok_mono/seed*/scale_*/eval_sib200.json` (token row) | as listed | PAPER_APPENDIX |
| **Table 36** (L) | Qwen1.5-MoE-A2.7B FLORES grid, 6 cond × 5 lang, 3-seed mean±SD | `scripts/analysis/qwen15_table1_aggregate.py --seeds 0 1 2` ⚠️ script says "table1" (DeepSeek-analogue naming) | `results/qwen/phase1/report/table1_qwen_seeds012.json` | PAPER_APPENDIX |
| **Table 37** (L) | Qwen SIB-200 clean-98 absolute accuracy, [k/3] collapsed | `scripts/analysis/qwen15_sib200_clean_aggregate.py` | `results/qwen/sib200_clean/report/table_qwen_clean98.json` | PAPER_APPENDIX |

## 3. Repository items with no camera-ready figure/table

| Item | Class | Note |
|---|---|---|
| `agg_covariance_spectra.npz` | LEGACY (provenance) | Panel-B data of the removed submission Figure 1 (see §1); unique derived record, kept |
| Belebele **passage-filtered** re-evaluation (`belebele_clean_eval.py`, `_make_belebele_clean.py`, `results/downstream/exposure_filtered/belebele_tables/`, n = EN 99 / ZH 99 / KO 113 / SW 99 / BN 125) | SUPPLEMENTARY | Produced for the rebuttal; the camera-ready appendix (Table 29) prints the **legacy n=200 5-shot** table instead. Kept as the corrected-protocol evidence; see §4. |
| XL-Sum 6-condition extension stub (`external_bpb_missing_cells.json`) | SUPPLEMENTARY (kept) | 7/9 cells never produced; not citable (documented in `docs/limitations.md`) — kept as limitations evidence |
| Budget-reallocation / robust-mix / vulnerability-targeted / synthetic-surrogate artifacts | SUPPLEMENTARY (kept) | These negative results ARE reported in paper §7 / App. J (without numbered tables) — kept as their evidence |
| XNLI preliminary evals (`phase1_xnli_eval.py`, `eval_xnli_v2.json` ×3) | SUPPLEMENTARY (kept) | Back the Limitations claim that preliminary XNLI collapsed to chance |
| `fisher_pilot_a.py`, `fisher_damage_pair_analysis.py`, MASSIVE scorer ablation/diagnostic | REPRO_SUPPORT (kept) | `fisher_pilot_a` is imported by in-tree analysis scripts; the others validate the §6.1 association claim and the MASSIVE scorer |
| Fisher Pilot A analysis (`analyze_fisher_pilot_a.py`, `aux/pilot_a_*.json`), legacy gates (`phase1_bengali_gate.py`, `phase1_swahili_gate.py`), legacy MASSIVE protocol scripts, unreported Qwen-MASSIVE exploration, smoke/backup/deprecated result files, timestamped Qwen report snapshots, submission-era appendix inventory | SUPPLEMENTARY / LEGACY / TEMPORARY — **moved out** | Moved to `ARR_8/private_release_archive/` in the 2026-11-23 cleanup (89 files; SHA-256-verified; see `ARR_8/PUBLIC_REPOSITORY_CLEANUP_REPORT.md`) |

Every camera-ready figure/table has a repository source — there are **no
paper items missing from the repository**. The only two items whose exact
paper rendering has no single dedicated repo script are Table 12 (constants
read from `baseline/eval_ppl.json`) and Table 32 (verdict rule applied to
Table 31/13 quantities; noise floors preserved, not recomputable).

## 4. Notes for the authors (paper-side, not repository defects)

1. **Table 29 protocol**: camera-ready I.2 prints the *legacy* Belebele
   evaluation (5-shot, n=200/language, demonstrations drawn from the test
   split — disclosed in `docs/data_exposure_audit.md`), while §4 of the
   main text says Belebele was "filter[ed] at the passage level". The
   passage-filtered results exist in this repository
   (`results/downstream/exposure_filtered/belebele_tables/`) but are not
   the table printed in the PDF. Worth a deliberate author decision
   (either print the filtered table or soften the §4 sentence).
2. **61.44 pp** (§5.4, B.6, Table-9-derived): the camera-ready prints
   61.44; full-precision recomputation from the raw cells gives 61.4306 →
   61.43 (see `docs/EXPERIMENT_MAPPING.md` §H). 61.44 arises only from
   2-decimal intermediate rounding. Cosmetic, but flagged.
3. The Qwen table labels proposed in the rebuttal materials
   (`tab:qwen-flores-full`, `tab:qwen-sib-acc-clean`) correspond to
   camera-ready **Tables 36/37**; §5.6's references resolve correctly in
   the final PDF (no "Table ??" remains).

## 5. Verification trail

- Figure/table inventory extracted directly from the camera-ready PDF
  (2026-11-23); no numbers were inferred from filenames or from the
  submission PDF.
- All generators above re-run via `tests/smoke_test.sh` after this audit;
  regenerated figures visually compared against the camera-ready panels.
- No experiment algorithm, raw result file, or stored metric was modified
  in this audit; only documentation (`README.md`,
  `docs/EXPERIMENT_MAPPING.md`, `docs/REPRODUCIBILITY.md`,
  `docs/experiment_registry.md`, this file).
