# DeepSeek-MoE-16B — SIB-200 test-item-clean accuracy (%), same 98 items (from analysis/clean_eval_0916)

| Calibration | EN | ZH | KO | SW | BN |
|:---|:---:|:---:|:---:|:---:|:---:|
| Uncompressed baseline | 77.6 | 74.5 | 74.5 | 44.9 | 55.1 |
| English | 75.9 ± 4.1 [0/3] | 75.5 ± 2.0 [0/3] | 41.5 ± 6.5 [0/3] | 31.0 ± 9.5 [1/3] | 16.3 ± 6.4 [3/3] |
| Chinese | 72.4 ± 6.2 [0/3] | 74.5 ± 1.8 [0/3] | 39.8 ± 2.7 [0/3] | 22.8 ± 2.9 [3/3] | 14.3 ± 8.8 [3/3] |
| Korean | 59.5 ± 1.2 [0/3] | 67.7 ± 1.6 [0/3] | 56.8 ± 1.6 [0/3] | 28.9 ± 5.8 [1/3] | 28.6 ± 4.1 [1/3] |
| Swahili | 58.2 ± 7.4 [0/3] | 65.0 ± 9.3 [0/3] | 34.4 ± 5.6 [0/3] | 33.3 ± 2.4 [0/3] | 10.5 ± 1.6 [3/3] |
| Bengali | 53.7 ± 16.1 [0/3] | 65.3 ± 10.0 [0/3] | 51.0 ± 14.0 [0/3] | 23.1 ± 3.1 [2/3] | 37.4 ± 9.2 [0/3] |
| Balanced-character | 64.6 ± 2.6 [0/3] | 66.0 ± 2.6 [0/3] | 62.2 ± 4.7 [0/3] | 36.4 ± 1.6 [0/3] | 26.9 ± 6.8 [1/3] |

Absolute accuracy (%) on the 98 test-item-clean SIB-200 items per language (same keep set as the DeepSeek clean run); 14 fixed English demonstrations unchanged (6 of 14 calibration-exposed per the audit), so this is TEST-ITEM clean, not calibration-exposure-free. Conditions: mean ± sample std (ddof=1) over seeds 0,1,2, all seeds included; [k/3] = seeds meeting the collapse criterion (top-label share > 0.9 OR accuracy < 25%) on the clean subset. Majority-class floor of the clean subset ~26.5% (26/98 science/technology).
