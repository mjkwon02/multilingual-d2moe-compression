# Experiment registry

Maps every experiment family to its code, its preserved results, and the
paper items it feeds. "cond" ranges over the six calibration conditions
(+`balanced_tok_mono` where noted), seeds are 0/1/2. Paper items use the
**camera-ready** numbering (updated 2026-11-23); the full figure/table
correspondence, including legacy filename numbers, is
`docs/PAPER_FIGURE_TABLE_MAPPING.md`.

| Experiment | Code | Preserved results | Paper items |
|---|---|---|---|
| DeepSeek Δbpb grid (6×5, whitening-ON diagonal) | `phase1_calib_data` → `phase1_fisher`/`phase1_svd_scale` → `phase1_merge_eval`; gate: `phase1_41_headline_gate` | `results/deepseek/<cond>/seed*/scale_*/eval_ppl.json`, `phase1_41_headline_gate_result.json` | Table 1, Tables 13–14, Figure 1 (teaser) |
| Placebo arms (shuffled text) | same pipeline, conditions `*_only_placebo` | `results/deepseek/*_only_placebo/`, `_make_p1_5` outputs | Table 13 noise floors, Table 15 |
| Seed / placebo gates | `phase1_{seed,placebo}_gate` (the layout-stale legacy swahili/bengali gates were archived privately 2026-11-23; their quantities come from the headline gate) | `results/deepseek/phase1_*_gate_result.json` | robustness checks |
| 2×2 Fisher×Scale (EN/KO; EN/SW) | `phase1_2x2_gate`, `phase1_2x2_verify_gate`, `phase1_2x2_ensw_effects` | cross-scale eval files, `phase1_2x2*_result.json`, `table_b6_2x2_ensw_effects.json` | §5.4, B.6 (Tables 9–10) |
| SIB-200 (full 204) | `phase1_sib200_eval`, `sib200_extended_report` | `results/deepseek/sib200_extended/` | Tables 23–28, 35 |
| SIB-200 exposure-filtered (98) | `sib200_clean_eval`, `_make_table2_accuracy` | `results/downstream/exposure_filtered/` | Table 2 (Qwen: Table 37) |
| Belebele (full / filtered) | `phase1_belebele_eval`, `phase1_belebele_gate`, `belebele_clean_eval`, `_make_belebele_clean` | `fewshot_5/eval_belebele.json`, gate results, `belebele_tables/` | Table 29 (legacy protocol; the filtered tables are rebuttal evidence, not in the camera-ready) |
| MASSIVE (revised protocol) | `massive_revised_*` | `results/downstream/massive_revised/`, demo IDs in `data/splits/` | Table 30 (App. I.3) |
| XL-Sum OOD bpb | `e2_build_ood_corpus`, `e2e3_ood_unseen_eval`, `make_external_bpb_grid_0916` | `data/ood_xlsum/`, `eval_ood_unseen.json`, `external_bpb_grid.csv` | §5.5, App. K (Tables 33–34) |
| Structured-pruning ablation | `phase1_structured_prune`, `make_table4_pruning_lang` | `pp_ratio_0.2/eval_ppl.json`, `table4_pruning_lang.json` | Table 8 (App. B.4) |
| Expert restoration (§5.4) | `phase1_sec54_select_targets`, `phase1_sec54_intervene`, `make_table5_sec54_bundle`, `make_tableH1_sec54_recovery` | `results/deepseek/sec54/`, table JSONs | Table 5, App. H (Tables 21–22) |
| Fisher rank correlations | `phase1_51_fisher_rank_correlation`, `phase1_51b_*` | scalar caches + `phase1_51*_result_seed*.json` | App. F.1–F.3 (Tables 18–19), Fig 2A |
| Placebo rank ρ (unique copies) | `placebo_*_zh_bn` | `results/source_data/placebo_*` — see provenance | App. F.4 (Table 20) |
| Whitening geometry | `phase1_whitening_geometry{,_analysis}`, figures 3/4/G | `whitening_geometry_pairs.csv` + analysis JSON/NPZ | Table 4, Fig 2C, Figures 3–5 (output files `figure4_*`, `figureG1_*`, `figureG2_*`) |
| Expert disagreement scan | `scan_disagreement_experts`, `phase1_52*` | `results/deepseek/aux/scan_results.json`, `phase1_52*_result.json` | App. E (Tables 16–17), Fig 2B |
| Token/char-balanced analysis | `phase1_6_robust_mix_calib_tokens`, `phase1_a2_balanced_token_share`, `phase1_tokens_per_byte_all_langs`, `e1_token_balanced_design` | `calib_token_length_results.json`, `balanced_token_share_*`, `phase1_tokens_per_byte_all_langs.json` | Tables 6–7 (App. A), Table 11 (App. C), Tables 31/32/35 (App. K) |
| Fisher Pilot A (design validation) | `fisher_pilot_a` (kept — its constants are imported by in-tree analysis scripts); `analyze_fisher_pilot_a` + `aux/pilot_a_*` moved to the private archive (2026-11-23 cleanup) | private archive | pre-paper, no paper item |
| Qwen1.5-MoE validation (6×3) | `qwen15_{freq,fisher,svd_scale,merge_eval}`, `qwen15_table1_aggregate` | `results/qwen/phase1/` + `report/` | Table 36 (App. L) |
| Qwen SIB filtered | `qwen15_sib200_clean_{eval,aggregate}` | `results/qwen/sib200_clean/` | Table 37 (App. L) |
| Data exposure audit | `audit_calib_eval_overlap`, `make_clean_eval_ids_0916`, `make_audit_manifest_0916` | `results/source_data/data_exposure_audit/`, `data/splits/clean_eval_ids.json` | audit section |
| Legacy submission-Fig-1 panel-B provenance | covariance spectra extracted from the (not preserved) aggregated activation covariances | `results/source_data/agg_covariance_spectra.npz` (full eigenspectra + top-16 eigenvectors, 27 layers × gate/down × 4 languages; verified to reproduce the removed script's constants) | none — the submission-only overview figure and its script were removed from the tree 2026-11-23 (archived privately); camera-ready Fig 1 = teaser |
