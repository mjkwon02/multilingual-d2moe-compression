# Qwen1.5-MoE-A2.7B — SIB-200 test-item-clean accuracy (%), 14-shot, 98 items/language

| Calibration | EN | ZH | KO | SW | BN |
|:---|:---:|:---:|:---:|:---:|:---:|
| Uncompressed baseline | 78.6 | 61.2 | 70.4 | 37.8 | 52.0 |
| English | 77.6 ± 4.7 [0/3] | 61.2 ± 6.2 [0/3] | 59.2 ± 11.4 [0/3] | 19.0 ± 5.0 [3/3] | 20.1 ± 9.2 [2/3] |
| Chinese | 82.0 ± 1.6 [0/3] | 70.1 ± 5.6 [0/3] | 66.3 ± 1.8 [0/3] | 28.2 ± 4.6 [1/3] | 29.3 ± 3.3 [0/3] |
| Korean | 77.9 ± 3.6 [0/3] | 67.0 ± 4.6 [0/3] | 75.5 ± 1.0 [0/3] | 26.5 ± 7.1 [2/3] | 34.7 ± 8.1 [0/3] |
| Swahili | 77.9 ± 3.3 [0/3] | 61.9 ± 8.2 [0/3] | 61.2 ± 8.4 [0/3] | 33.0 ± 7.9 [0/3] | 27.9 ± 10.6 [2/3] |
| Bengali | 72.8 ± 7.2 [0/3] | 62.6 ± 4.6 [0/3] | 68.0 ± 6.9 [0/3] | 30.6 ± 7.1 [1/3] | 54.8 ± 6.9 [0/3] |
| Balanced-character | 77.6 ± 6.1 [0/3] | 64.3 ± 12.5 [0/3] | 70.1 ± 3.6 [0/3] | 41.2 ± 14.8 [0/3] | 52.7 ± 12.5 [0/3] |

Absolute accuracy (%) on the 98 test-item-clean SIB-200 items per language (same keep set as the DeepSeek clean run); 14 fixed English demonstrations unchanged (6 of 14 calibration-exposed per the audit), so this is TEST-ITEM clean, not calibration-exposure-free. Conditions: mean ± sample std (ddof=1) over seeds 0,1,2, all seeds included; [k/3] = seeds meeting the collapse criterion (top-label share > 0.9 OR accuracy < 25%) on the clean subset. Majority-class floor of the clean subset ~26.5% (26/98 science/technology).
