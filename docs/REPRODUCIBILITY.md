# Reproduction guide

Two levels of reproduction are supported:

1. **Verification (no GPU, minutes)** — regenerate every paper table and
   figure from the preserved per-seed metrics in `results/`.
2. **Full re-run (GPU, days)** — recompute Fisher/whitening statistics,
   compressed models, and evaluations from scratch.

## Environment

- Python 3.10 (conda env: `requirements/deepseek_conda_env.yml`, pip pins in
  `requirements/deepseek.txt`)
- Reference system: torch 2.5.1+cu121, transformers 5.13.1, accelerate
  1.14.0, datasets 5.0.0, safetensors 0.8.0; 4× RTX 3090 (24 GB), CUDA 12.1,
  driver 590.48. See `requirements/environment.md`.
- Set `HF_HOME` if you want the Hugging Face cache somewhere specific; all
  scripts read it from the environment (default `~/.cache/huggingface`).
- Run every command from the **repository root**.

## Level 1: verification (no GPU)

`tests/smoke_test.sh` runs all of the following and was verified to
reproduce the preserved numbers exactly (see
`manifests/verification_report.md`):

Figure/table numbers are the **camera-ready** ones (full correspondence,
including filename↔number mismatches: `docs/PAPER_FIGURE_TABLE_MAPPING.md`).

| Paper item (camera-ready) | Command | Source data |
|---|---|---|
| Table 1 (Δbpb grid) + Table 13 (verdicts) + Table 14 (per-seed) | `python scripts/analysis/phase1_41_headline_gate.py` | `results/deepseek/<cond>/seed*/scale_*/eval_ppl.json` |
| Tables 23–28 (SIB-200, full 204 + per-seed) | `python scripts/analysis/sib200_extended_report.py` | `results/deepseek/sib200_extended/` |
| Table 2 (SIB-200 exposure-filtered 98) | `python results/downstream/exposure_filtered/sib200_accuracy_tables/_make_table2_accuracy.py` | `results/downstream/exposure_filtered/deepseek_clean_eval/` |
| Table 8 (pruning ΔΔbpb%) | `python scripts/analysis/make_table4_pruning_lang.py` (+ `_make_p0_2.py` for the chinese_only row) | per-seed `eval_ppl.json` + `pp_ratio_0.2/` |
| Table 5 (restoration bundle) | `python scripts/analysis/make_table5_sec54_bundle.py` | `results/deepseek/sec54/` |
| Table 3 (stability of Balanced) | `python results/source_data/tables_main_submission/_make_p0_3.py` | gate results + SIB report |
| Table 9 (EN/KO 2×2) | `python scripts/analysis/phase1_2x2_gate.py` and `phase1_2x2_verify_gate.py` | cross-scale `eval_ppl.json` |
| Table 10 (EN/SW 2×2 per-seed) | `python scripts/analysis/phase1_2x2_ensw_effects.py` | cross-scale `eval_ppl.json` |
| Table 15 (placebo per-seed diagonal) | `python results/source_data/tables_main_submission/_make_p1_5.py` | `*_only_placebo` eval files |
| Tables 16–17 (deficiency pairs + threshold sensitivity) | `python scripts/analysis/phase1_52b_deficiency_pairs_list.py` | `results/deepseek/aux/scan_results.json` |
| Tables 18–19 (Fisher rank ρ) | `python scripts/analysis/phase1_51_fisher_rank_correlation.py` | `results/deepseek/fisher_expert_scalar_cache/` |
| Table 4 + Figures 2–5 (whitening geometry) | `python scripts/analysis/phase1_whitening_geometry_analysis.py` then the three `scripts/figures/make_figure*` scripts (outputs `fig3_layerwise` = Fig 2, `figure4_whitening_distributions` = Fig 3, `figureG1_*`/`figureG2_*` = Figs 4/5) | `results/source_data/whitening_geometry_pairs.csv` |
| Tables 21–22 (§6.4 targets + recovery rates) | `python scripts/analysis/phase1_sec54_select_targets.py`, `python scripts/analysis/make_tableH1_sec54_recovery.py` | `results/deepseek/sec54/` |
| Table 29 (legacy 5-shot Belebele + gates) | `python scripts/analysis/phase1_belebele_gate.py --condition korean_only --scale-condition korean_only --pp-ratio 0.2 --num-fewshot 5` (and `--off-only` variants) | `fewshot_5/eval_belebele.json` |
| Belebele exposure-filtered (rebuttal evidence; no camera-ready table) | `python results/downstream/exposure_filtered/belebele_tables/_make_belebele_clean.py` | `.../deepseek_clean_eval/belebele/` |
| Table 30 (MASSIVE, revised protocol) | `python scripts/evaluation/massive_revised_aggregate.py` | `results/downstream/massive_revised/` |
| Table 36 (Qwen FLORES grid) | `python scripts/analysis/qwen15_table1_aggregate.py --seeds 0 1 2` | `results/qwen/phase1/` |
| Table 37 (Qwen SIB exposure-filtered) | `python scripts/analysis/qwen15_sib200_clean_aggregate.py` | `results/qwen/sib200_clean/` |
| Figure 1 (teaser) | `python scripts/figures/make_figure_teaser_calibration_gap.py` | in-script constants = Table 1 English/Balanced rows (the submission-only overview figure was removed 2026-11-23 and archived privately; its Panel-B provenance `results/source_data/agg_covariance_spectra.npz` is kept) |
| Tables 31, 33, 34 (token-balanced grid, XL-Sum OOD, unseen languages) | `python scripts/analysis/make_external_bpb_grid_0916.py` | `eval_ood_unseen.json` + `eval_ppl.json` files incl. `balanced_tok_mono/` |
| Tables 6, 7, 11 (calibration token statistics) | `python scripts/calibration/phase1_6_robust_mix_calib_tokens.py`, `phase1_a2_balanced_token_share.py`, `phase1_tokens_per_byte_all_langs.py` — **need the FLORES download + DeepSeek tokenizer** (no GPU), so they are not in the smoke test; the preserved outputs are the table sources | `results/source_data/calib_token_length_results.json`, `balanced_token_share_*`, `phase1_tokens_per_byte_all_langs.json` |

Known caveats:

- `phase1_51_fisher_rank_correlation.py` recomputes the full-condition
  correlations from the preserved scalar caches (verified identical), but the
  **placebo-arm comparison section cannot be recomputed** because the placebo
  Fisher tensors were deleted on 2026-08-02. The preserved
  `results/deepseek/phase1_51_fisher_rank_correlation_result_seed*.json`
  files (which include that section) are the authoritative copies; the
  summary print crashes after writing the per-seed file when the placebo
  tensors are absent.
- The legacy `phase1_bengali_gate.py` / `phase1_swahili_gate.py` predated
  the final placebo-arm layout and were moved to the private archive in the
  2026-11-23 cleanup; the noise floors they were built for are computed
  (and verified) by `phase1_41_headline_gate.py`.

## Level 2: full re-run (GPU)

1. **Models**
   - `deepseek-ai/deepseek-moe-16b-base`, revision
     `521d2bc4fb69a3f3ae565310fcc3b65f97af2580`. Apply
     `patches/deepseek_sdpa_fallback.patch` to `modeling_deepseek.py`
     (both the hub snapshot and the `transformers_modules` copy if trust
     remote code has already cached it). See
     `docs/provenance/deepseek_patch.md`.
   - `Qwen/Qwen1.5-MoE-A2.7B` (unmodified).
2. **Third-party code**: clone https://github.com/lliai/D2MoE at commit
   `b3092f0` into `third_party/D2MoE` and apply the changes listed in
   `docs/provenance/d2moe_patch_manifest.md` (memory caps, CPU Fisher
   accumulation, an import fix; the Belebele task-config edits are
   `legacy_exposed` and should NOT be applied for corrected-protocol runs).
   Scripts honor a `D2MOE_DIR` environment variable if the clone lives
   elsewhere.
3. **Datasets** (auto-downloaded via `datasets`): `israel/flores-parallel`
   (FLORES-200 devtest), `Davlan/sib200`, `facebook/belebele`, `wikitext`
   (wikitext-2), and MASSIVE 1.1 (`amazon-massive-dataset-1.1.tar.gz` from
   the official Amazon Science release) unpacked to `data/massive/1.1/`.
   **XL-Sum OOD corpus**: the text is not redistributed (CC BY-NC-SA 4.0);
   rebuild it first with
   `python scripts/calibration/e2_build_ood_corpus.py --bytes-per-lang 51200 --seed 0`,
   which downloads the official `csebuetnlp/xlsum` v2.0 language tarballs
   and deterministically reproduces the exact corpus used in the paper —
   verified on 2026-11-23 to be **byte-identical** to the original files
   (SHA-256 per language in `data/ood_xlsum/manifest.json`, which also
   records the sampled XL-Sum document IDs). The GPU OOD evaluation
   (`e2e3_ood_unseen_eval.py`) and the calibration-overlap audit read the
   rebuilt files; all *stored* XL-Sum results and the camera-ready
   Table 33 aggregation (XL-Sum OOD — called "Table 31" in rebuttal-era
   script comments) work without them.
4. **Pipeline per condition × seed** (DeepSeek):
   `scripts/calibration/phase1_calib_data.py` →
   `scripts/compression/phase1_run_freq_and_scale.py` →
   `scripts/compression/phase1_fisher.py` →
   `scripts/compression/phase1_svd_scale.py` →
   `scripts/compression/phase1_merge_eval.py` (+`phase1_structured_prune.py`
   for Table 8, the pruning ablation) → evaluation scripts. Qwen equivalents:
   `scripts/compression/qwen15_*.py`.
   Seeds are 0/1/2 and enter through each script's `--seed` argument;
   calibration budget is 64 samples × seqlen 512 unless a script's docstring
   says otherwise.
5. Expected intermediate sizes: ~62 GB of Fisher+whitening tensors per
   condition × seed (DeepSeek), ~75 GB per condition × seed (Qwen, including
   the compressed model). These are deliberately not preserved in this
   repository; every paper number is reproducible from the small files that
   are.

## Hardware, precision, and measured cost

- All GPU runs used **NVIDIA RTX 3090 (24 GB)** cards on a shared 4×3090
  host (`requirements/environment.md`); models are loaded in **bfloat16**
  (`torch_dtype=torch.bfloat16` in `phase1_fisher.py`, `phase1_merge_eval.py`,
  `qwen15_config.py`).
- The DeepSeek pipeline uses **1–2 GPUs depending on the stage**
  (`device_map="auto"` with a 22 GiB/GPU memory cap; Fisher accumulation on
  CPU). The Qwen pipeline asserts **exactly two GPUs, no CPU/disk offload**
  (`qwen15_config.py`).
- **Measured cost (Qwen)**: the preserved run log
  (`results/qwen/followup_0917/logs/followup2.log`) records ~8,150–8,520 s
  (~2.3 h) per condition × seed cell for compression + evaluation on that
  host. DeepSeek per-cell GPU time was **not systematically measured** and no
  estimate is given here.
- Level-1 verification (tables/figures from preserved results) needs **no
  GPU** and finishes in minutes.

## Verification levels

Per-experiment verification status — which paper items were re-verified by
aggregation from preserved per-seed results, which have runnable-but-unrun
GPU code, and which are partially missing — is tracked in
`docs/EXPERIMENT_MAPPING.md`. `tests/smoke_test.sh` re-runs the Level-1 set;
it last passed on 2026-11-23 (see `manifests/verification_report.md`).

## Shared-GPU etiquette

`scripts/infra/safe_gpus.sh` and `scripts/infra/gpu_status.py` detect GPUs
occupied by other users and exclude them automatically; the run drivers were
written for a shared 4×3090 host.
