# Placebo Fisher rank correlation — ZH / BN (post-hoc supplement to Table F1)

**POST-HOC SUPPLEMENT to frozen Table F1. Table F1's body is NOT modified and these two placebo pairs are NOT retro-fitted into it; they are reported separately. Extracted 2026-08-02, immediately before the chinese_only_b / bengali_only_b large binaries were deleted for disk recovery.**

Method identical to Table F1: per-expert Fisher scalar F_e (sum of the sample-averaged
squared gradients over that expert's 3 projections), per-layer Spearman over 64 experts,
mean ± std over the 27 MoE layers (ddof=0, F1's convention). The reduction function is
imported from `phase1_51_fisher_rank_correlation.py`, not reimplemented.

## Placebo ρ (same language, disjoint calibration text)

| pair | seed 0 | seed 1 | seed 2 | 3-seed mean |
|---|---:|---:|---:|---:|
| chinese_only vs chinese_only_b | 0.936 ± 0.024 | 0.890 ± 0.051 | 0.916 ± 0.044 | **0.914** |
| bengali_only vs bengali_only_b | 0.972 ± 0.019 | 0.979 ± 0.013 | 0.980 ± 0.009 | **0.977** |

Each cell is the 27-layer mean ± std of the per-layer Spearman ρ for that seed.

## Why these two pairs were missing from F1

At F1's computation time `chinese_only_b` was absent from
`phase1_calib_data.PLACEBO_CONDITIONS`, and `bengali_only_b`'s real Fisher had never
been run (the directory existed but was empty). Both placebo arms have real
`fisher_processed.pt` for all 3 seeds now, so the pairs are computable. They are
reported here only — F1 keeps its original 3 placebo pairs (EN/KO/SW).
