# Supplementary — SIB-200 retention (×, ratio to uncompressed accuracy on the same subset)

## (a) Full test set, n=204 (denominator: full uncompressed accuracy)

| Calibration | EN | ZH | KO | SW | BN |
|:---|:---:|:---:|:---:|:---:|:---:|
| English calibration | 0.979 ± 0.053 | 0.963 ± 0.059 | 0.551 ± 0.089 | 0.682 ± 0.154 | 0.231 ± 0.090 [3/3] |
| Chinese calibration | 0.948 ± 0.036 | 0.949 ± 0.025 | 0.511 ± 0.021 | 0.543 ± 0.099 [2/3] | 0.249 ± 0.128 [3/3] |
| Korean calibration | 0.800 ± 0.033 | 0.870 ± 0.015 | 0.686 ± 0.028 | 0.643 ± 0.070 [1/3] | 0.471 ± 0.054 [1/3] |
| Swahili calibration | 0.790 ± 0.081 | 0.817 ± 0.068 | 0.509 ± 0.088 | 0.818 ± 0.048 | 0.181 ± 0.033 [3/3] |
| Bengali calibration | 0.677 ± 0.177 | 0.801 ± 0.133 | 0.673 ± 0.194 | 0.531 ± 0.076 [2/3] | 0.705 ± 0.224 |
| Balanced-character calibration | 0.842 ± 0.028 | 0.858 ± 0.043 | 0.782 ± 0.032 | 0.876 ± 0.018 | 0.520 ± 0.109 [1/3] |

## (b) Clean subset, n=98 (denominator: clean uncompressed accuracy)

| Calibration | EN | ZH | KO | SW | BN |
|:---|:---:|:---:|:---:|:---:|:---:|
| English calibration | 0.978 ± 0.053 | 1.014 ± 0.027 | 0.557 ± 0.087 | 0.689 ± 0.211 [1/3] | 0.296 ± 0.116 [3/3] |
| Chinese calibration | 0.934 ± 0.080 | 1.000 ± 0.024 | 0.534 ± 0.036 | 0.508 ± 0.066 [3/3] | 0.259 ± 0.160 [3/3] |
| Korean calibration | 0.768 ± 0.015 | 0.909 ± 0.021 | 0.763 ± 0.021 | 0.644 ± 0.129 [1/3] | 0.519 ± 0.074 [1/3] |
| Swahili calibration | 0.750 ± 0.095 | 0.872 ± 0.124 | 0.461 ± 0.075 | 0.742 ± 0.052 | 0.191 ± 0.028 [3/3] |
| Bengali calibration | 0.693 ± 0.208 | 0.877 ± 0.135 | 0.685 ± 0.188 | 0.515 ± 0.069 [2/3] | 0.679 ± 0.167 |
| Balanced-character calibration | 0.833 ± 0.033 | 0.886 ± 0.034 | 0.836 ± 0.063 | 0.811 ± 0.035 | 0.488 ± 0.123 [1/3] |

**Caption.** SIB-200 retention (ratio, ×; not a percentage), defined as compressed-model accuracy divided by uncompressed accuracy on the same evaluation subset: the full-test baseline (n = 204) for the full panel and the clean-subset baseline (n = 98) for the clean panel. Mean ± sample standard deviation (ddof = 1) of per-seed retention over 3 seeds, collapsed seeds included; [k/3] as in the accuracy tables. A high retention in a cell with collapsed seeds, or where the uncompressed accuracy is itself low (e.g. SW), should not be read as preserved task ability. Pending: fewer than 3 seeds completed.
