# ZH / BN placebo — Fisher rank ρ and SWITCH-1 equivalence (post-hoc supplement)

**POST-HOC SUPPLEMENT to frozen §6.1 / Appendix F. The frozen text and Tables F1 and F3 are NOT modified; these ZH/BN placebo numbers are reported only as a separate supplementary table and are not retro-fitted into F1/F3. Extracted 2026-08-02, immediately before the chinese_only_b / bengali_only_b large binaries were deleted for disk recovery.**

Method is not reimplemented. 2-A imports `reduce_condition()` from
`phase1_51_fisher_rank_correlation.py`; 2-B imports `reduce_condition_by_proj()`,
`load_expert_freq()`, `merge_weighted()` and `part_a_definition_agreement()` from
`phase1_51b_merge_weight_rank_correlation.py`. Neither source file was modified.
Per-layer Spearman over 64 experts, mean ± std over the 27 MoE layers, ddof=0 (F1 convention).

## 2-A — placebo ρ, raw F_e definition (Table F1's definition)

| pair | seed 0 | seed 1 | seed 2 | 3-seed mean |
|---|---:|---:|---:|---:|
| chinese_only vs chinese_only_b | 0.936 ± 0.024 | 0.890 ± 0.051 | 0.916 ± 0.044 | **0.914** |
| bengali_only vs bengali_only_b | 0.972 ± 0.019 | 0.979 ± 0.013 | 0.980 ± 0.009 | **0.977** |

Reference (not a criterion): the EN/KO/SW placebo pairs in F1 sat at 0.92–0.98.

## 2-B(b) — placebo ρ, merge-weight definition (F3 / SWITCH-1's definition)

| pair | seed 0 | seed 1 | seed 2 | 3-seed mean |
|---|---:|---:|---:|---:|
| chinese_only vs chinese_only_b | 0.946 ± 0.019 | 0.914 ± 0.037 | 0.931 ± 0.033 | **0.930** |
| bengali_only vs bengali_only_b | 0.985 ± 0.007 | 0.988 ± 0.006 | 0.989 ± 0.005 | **0.988** |

The within-language ceiling does not fall when the definition changes — for both pairs the
merge-weighted ρ is slightly **higher** than the raw-F_e ρ. That is F3's question, checked on ZH/BN.

## 2-B(a) — definition agreement (raw F_e ranking vs merge-weighted ranking)

- observed mean ρ = **0.9328** (EN/KO/SW reference mean 0.939)
- observed min ρ  = **0.7697** (EN/KO/SW reference min 0.831; gate 0.83)
- gate verdict: **FLAG_BELOW_REFERENCE_MIN**

The mean is level with the reference; the FLAG comes from the minimum alone, and that minimum
is highly localized: **13 of 1296** (condition × layer × projection) cells fall below 0.83, and
**12 of those 13 are layer 9**, all in the Chinese arms (chinese_only 9, chinese_only_b 4;
gate_proj 6, up_proj 4, combined 2, down_proj 1). Every Bengali cell stays above 0.86.

Two caveats on comparing the minima at all:

1. The reference min 0.831 was computed over seed0 only; this run spans 3 seeds × 4 conditions
   (1296 cells vs the reference's far smaller draw). A minimum is not sample-size invariant, so a
   lower minimum here is expected even under an identical distribution.
2. The FLAG applies to the definition-agreement metric, not to the placebo ρ values above.

Per the run plan, the FLAG does not block the deletion step (memory recovery has priority); it
withholds the definition-agreement number from the paper pending review.
