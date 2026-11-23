# SIB-200 extended grid -- §4.2 main evidence

Protocol: 14-shot fixed English demonstrations, n=204/language, all 5 calibration languages evaluated, natural-sentence continuation scored over 7 single-token category words (`phase1_sib200_eval.py`, protocol `sib200_natural_continuation_v1`). Every run 2-GPU / fully GPU-resident -- no disk offload, per-item NaN guard armed.

Retention = (mean acc over available seeds) / (baseline acc for that language). A cell reads **[COLLAPSED]** instead of a retention number when the predictions degenerated onto a single label: top-label fraction > 0.9 or acc < 0.25 (SIB-200's majority-class floor, categories are imbalanced).

## 1. Grid -- retention by condition x eval language

| condition | EN | KO | ZH | SW | BN |
|---|---|---|---|---|---|
| *baseline acc* | 0.784 | 0.765 | 0.804 | 0.422 | 0.559 |
| mixed_5lang | 0.842 ± 0.028 | 0.782 ± 0.032 | 0.858 ± 0.043 | 0.876 ± 0.018 | 0.520 ± 0.109 (top=0.59, 1/3 seed collapsed) |
| english_only | 0.979 ± 0.053 | 0.551 ± 0.089 | 0.963 ± 0.059 | 0.682 ± 0.154 | **[COLLAPSED]** top=0.79 |
| korean_only | 0.800 ± 0.033 | 0.686 ± 0.028 | 0.870 ± 0.015 | 0.643 ± 0.070 (top=0.56, 1/3 seed collapsed) | 0.471 ± 0.054 (top=0.81, 1/3 seed collapsed) |
| chinese_only | 0.948 ± 0.036 | 0.511 ± 0.021 | 0.949 ± 0.025 | 0.543 ± 0.099 (top=0.62, 2/3 seed collapsed) | **[COLLAPSED]** top=0.90 |
| swahili_only | 0.790 ± 0.081 | 0.509 ± 0.088 | 0.817 ± 0.068 | 0.818 ± 0.048 | **[COLLAPSED]** top=0.96 |
| bengali_only | 0.677 ± 0.177 | 0.673 ± 0.194 | 0.801 ± 0.133 | 0.531 ± 0.076 (top=0.66, 2/3 seed collapsed) | 0.705 ± 0.224 (top=0.39) |

Per-seed raw accuracies (top-label fraction in parentheses):

| condition/seed | EN | KO | ZH | SW | BN |
|---|---|---|---|---|---|
| mixed_5lang/seed0 | 0.681 (0.35) | 0.574 (0.32) | 0.721 (0.26) | 0.368 (0.33) | 0.221 (0.72) **C** |
| mixed_5lang/seed1 | 0.637 (0.35) | 0.598 (0.25) | 0.652 (0.30) | 0.377 (0.43) | 0.324 (0.54) |
| mixed_5lang/seed2 | 0.662 (0.34) | 0.623 (0.26) | 0.696 (0.26) | 0.363 (0.49) | 0.328 (0.52) |
| english_only/seed0 | 0.784 (0.24) | 0.387 (0.54) | 0.794 (0.21) | 0.250 (0.49) | 0.093 (0.94) **C** |
| english_only/seed1 | 0.799 (0.26) | 0.500 (0.33) | 0.809 (0.28) | 0.363 (0.34) | 0.108 (0.88) **C** |
| english_only/seed2 | 0.721 (0.30) | 0.377 (0.54) | 0.721 (0.32) | 0.250 (0.56) | 0.186 (0.55) **C** |
| korean_only/seed0 | 0.618 (0.49) | 0.510 (0.61) | 0.686 (0.42) | 0.299 (0.75) | 0.289 (0.72) |
| korean_only/seed1 | 0.608 (0.50) | 0.515 (0.60) | 0.711 (0.35) | 0.240 (0.48) **C** | 0.270 (0.78) |
| korean_only/seed2 | 0.657 (0.42) | 0.549 (0.57) | 0.701 (0.34) | 0.275 (0.45) | 0.230 (0.93) **C** |
| chinese_only/seed0 | 0.721 (0.20) | 0.397 (0.55) | 0.770 (0.22) | 0.186 (0.75) **C** | 0.088 (0.99) **C** |
| chinese_only/seed1 | 0.775 (0.21) | 0.402 (0.49) | 0.779 (0.28) | 0.270 (0.53) | 0.108 (0.95) **C** |
| chinese_only/seed2 | 0.735 (0.36) | 0.373 (0.60) | 0.740 (0.35) | 0.230 (0.57) **C** | 0.221 (0.77) **C** |
| swahili_only/seed0 | 0.623 (0.31) | 0.377 (0.37) | 0.598 (0.27) | 0.338 (0.32) | 0.088 (0.96) **C** |
| swahili_only/seed1 | 0.681 (0.21) | 0.461 (0.37) | 0.706 (0.18) | 0.328 (0.45) | 0.123 (0.93) **C** |
| swahili_only/seed2 | 0.554 (0.42) | 0.328 (0.47) | 0.667 (0.20) | 0.368 (0.50) | 0.093 (0.99) **C** |
| bengali_only/seed0 | 0.407 (0.43) | 0.358 (0.44) | 0.564 (0.26) | 0.191 (0.66) **C** | 0.270 (0.42) |
| bengali_only/seed1 | 0.681 (0.22) | 0.652 (0.18) | 0.765 (0.26) | 0.255 (0.61) | 0.520 (0.34) |
| bengali_only/seed2 | 0.505 (0.51) | 0.534 (0.38) | 0.603 (0.32) | 0.225 (0.73) **C** | 0.392 (0.41) |

(**C** = collapsed cell by the rule above.)

## 2. Column verdicts -- does the diagonal (own-language) condition win its column?

Within one eval-language column, the own-language-calibrated condition is compared to every other condition **seed by seed** (`+` = diagonal higher, `-` = diagonal lower). `3/3 +` means the ordering held under every seed.

| eval lang | diagonal | vs. condition | seed0 | seed1 | seed2 | sign consistency | mean acc (diag vs cross) |
|---|---|---|---|---|---|---|---|
| EN | english_only | mixed_5lang | + (0.784 vs 0.681) | + (0.799 vs 0.637) | + (0.721 vs 0.662) | **3/3 +** | 0.768 vs 0.660 |
| EN | english_only | korean_only | + (0.784 vs 0.618) | + (0.799 vs 0.608) | + (0.721 vs 0.657) | **3/3 +** | 0.768 vs 0.627 |
| EN | english_only | chinese_only | + (0.784 vs 0.721) | + (0.799 vs 0.775) | - (0.721 vs 0.735) | 2/3 + | 0.768 vs 0.743 |
| EN | english_only | swahili_only | + (0.784 vs 0.623) | + (0.799 vs 0.681) | + (0.721 vs 0.554) | **3/3 +** | 0.768 vs 0.619 |
| EN | english_only | bengali_only | + (0.784 vs 0.407) | + (0.799 vs 0.681) | + (0.721 vs 0.505) | **3/3 +** | 0.768 vs 0.531 |
| KO | korean_only | mixed_5lang | - (0.510 vs 0.574) | - (0.515 vs 0.598) | - (0.549 vs 0.623) | **0/3 +** (reversed) | 0.525 vs 0.598 |
| KO | korean_only | english_only | + (0.510 vs 0.387) | + (0.515 vs 0.500) | + (0.549 vs 0.377) | **3/3 +** | 0.525 vs 0.422 |
| KO | korean_only | chinese_only | + (0.510 vs 0.397) | + (0.515 vs 0.402) | + (0.549 vs 0.373) | **3/3 +** | 0.525 vs 0.391 |
| KO | korean_only | swahili_only | + (0.510 vs 0.377) | + (0.515 vs 0.461) | + (0.549 vs 0.328) | **3/3 +** | 0.525 vs 0.389 |
| KO | korean_only | bengali_only | + (0.510 vs 0.358) | - (0.515 vs 0.652) | + (0.549 vs 0.534) | 2/3 + | 0.525 vs 0.515 |
| ZH | chinese_only | mixed_5lang | + (0.770 vs 0.721) | + (0.779 vs 0.652) | + (0.740 vs 0.696) | **3/3 +** | 0.763 vs 0.690 |
| ZH | chinese_only | english_only | - (0.770 vs 0.794) | - (0.779 vs 0.809) | + (0.740 vs 0.721) | 2/3 - | 0.763 vs 0.775 |
| ZH | chinese_only | korean_only | + (0.770 vs 0.686) | + (0.779 vs 0.711) | + (0.740 vs 0.701) | **3/3 +** | 0.763 vs 0.699 |
| ZH | chinese_only | swahili_only | + (0.770 vs 0.598) | + (0.779 vs 0.706) | + (0.740 vs 0.667) | **3/3 +** | 0.763 vs 0.657 |
| ZH | chinese_only | bengali_only | + (0.770 vs 0.564) | + (0.779 vs 0.765) | + (0.740 vs 0.603) | **3/3 +** | 0.763 vs 0.644 |
| SW | swahili_only | mixed_5lang | - (0.338 vs 0.368) | - (0.328 vs 0.377) | + (0.368 vs 0.363) | 2/3 - | 0.345 vs 0.369 |
| SW | swahili_only | english_only | + (0.338 vs 0.250) | - (0.328 vs 0.363) | + (0.368 vs 0.250) | 2/3 + | 0.345 vs 0.288 |
| SW | swahili_only | korean_only | + (0.338 vs 0.299) | +ᶜ (0.328 vs 0.240) | + (0.368 vs 0.275) | **3/3 +** | 0.345 vs 0.271 |
| SW | swahili_only | chinese_only | +ᶜ (0.338 vs 0.186) | + (0.328 vs 0.270) | +ᶜ (0.368 vs 0.230) | **3/3 +** | 0.345 vs 0.229 |
| SW | swahili_only | bengali_only | +ᶜ (0.338 vs 0.191) | + (0.328 vs 0.255) | +ᶜ (0.368 vs 0.225) | **3/3 +** | 0.345 vs 0.224 |
| BN | bengali_only | mixed_5lang | +ᶜ (0.270 vs 0.221) | + (0.520 vs 0.324) | + (0.392 vs 0.328) | **3/3 +** | 0.394 vs 0.291 |
| BN | bengali_only | english_only | +ᶜ (0.270 vs 0.093) | +ᶜ (0.520 vs 0.108) | +ᶜ (0.392 vs 0.186) | **3/3 +** | 0.394 vs 0.129 |
| BN | bengali_only | korean_only | - (0.270 vs 0.289) | + (0.520 vs 0.270) | +ᶜ (0.392 vs 0.230) | 2/3 + | 0.394 vs 0.263 |
| BN | bengali_only | chinese_only | +ᶜ (0.270 vs 0.088) | +ᶜ (0.520 vs 0.108) | +ᶜ (0.392 vs 0.221) | **3/3 +** | 0.394 vs 0.139 |
| BN | bengali_only | swahili_only | +ᶜ (0.270 vs 0.088) | +ᶜ (0.520 vs 0.123) | +ᶜ (0.392 vs 0.093) | **3/3 +** | 0.394 vs 0.101 |

(ᶜ = at least one side of that comparison is a collapsed cell -- the sign is not interpretable as a calibration effect there.)

## 3. Baseline reproduction check

| lang | recorded reference | current baseline json | delta |
|---|---|---|---|
| EN | 0.784 | 0.784 | +0.0003 |
| KO | 0.765 | 0.765 | -0.0003 |
| ZH | 0.804 | 0.804 | -0.0001 |
| SW | 0.422 | 0.422 | -0.0004 |
| BN | 0.559 | 0.559 | -0.0002 |

Baseline unchanged (this campaign never re-ran it; the file is the 2-GPU rerun's).

## Run provenance

| run | source json |
|---|---|
| mixed_5lang/seed0 | `results/deepseek/sib200_extended/mixed_5lang_seed0.json` |
| mixed_5lang/seed1 | `results/deepseek/sib200_extended/mixed_5lang_seed1.json` |
| mixed_5lang/seed2 | `results/deepseek/sib200_extended/mixed_5lang_seed2.json` |
| english_only/seed0 | `results/deepseek/sib200_extended/english_only_seed0.json` |
| english_only/seed1 | `results/deepseek/sib200_extended/english_only_seed1.json` |
| english_only/seed2 | `results/deepseek/sib200_extended/english_only_seed2.json` |
| korean_only/seed0 | `results/deepseek/korean_only/seed0/scale_korean_only_seed0/eval_sib200.json` |
| korean_only/seed1 | `results/deepseek/korean_only/seed1/scale_korean_only_seed1/eval_sib200.json` |
| korean_only/seed2 | `results/deepseek/korean_only/seed2/scale_korean_only_seed2/eval_sib200.json` |
| chinese_only/seed0 | `results/deepseek/sib200_extended/chinese_only_seed0.json` |
| chinese_only/seed1 | `results/deepseek/sib200_extended/chinese_only_seed1.json` |
| chinese_only/seed2 | `results/deepseek/sib200_extended/chinese_only_seed2.json` |
| swahili_only/seed0 | `results/deepseek/swahili_only/seed0/scale_swahili_only_seed0/eval_sib200.json` |
| swahili_only/seed1 | `results/deepseek/swahili_only/seed1/scale_swahili_only_seed1/eval_sib200.json` |
| swahili_only/seed2 | `results/deepseek/swahili_only/seed2/scale_swahili_only_seed2/eval_sib200.json` |
| bengali_only/seed0 | `results/deepseek/sib200_extended/bengali_only_seed0.json` |
| bengali_only/seed1 | `results/deepseek/sib200_extended/bengali_only_seed1.json` |
| bengali_only/seed2 | `results/deepseek/sib200_extended/bengali_only_seed2.json` |
| baseline | `results/deepseek/baseline/eval_sib200.json` |
