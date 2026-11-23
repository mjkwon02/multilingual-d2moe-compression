# Qwen1.5-MoE-A2.7B — FLORES Δbpb% vs uncompressed (seeds [0, 1, 2])

| Calibration | EN | ZH | KO | SW | BN |
|:---|:---:|:---:|:---:|:---:|:---:|
| English | +8.50 ± 0.32 | +12.76 ± 0.56 | +45.38 ± 1.39 | +46.57 ± 0.54 | +84.58 ± 2.72 |
| Chinese | +12.56 ± 0.30 | +2.80 ± 0.27 | +45.11 ± 0.80 | +47.64 ± 0.92 | +76.77 ± 2.38 |
| Korean | +12.51 ± 0.07 | +9.47 ± 0.21 | +2.75 ± 0.10 | +44.45 ± 1.00 | +36.66 ± 0.19 |
| Swahili | +11.91 ± 0.70 | +13.06 ± 0.32 | +46.49 ± 0.63 | +0.79 ± 0.21 | +64.84 ± 2.53 |
| Bengali | +13.86 ± 0.56 | +11.97 ± 0.27 | +13.83 ± 0.28 | +34.31 ± 0.39 | +2.70 ± 0.34 |
| Balanced-character | +9.70 ± 0.55 | +4.32 ± 0.23 | +4.05 ± 0.14 | +3.58 ± 0.11 | +4.77 ± 0.21 |

Mean ± sample std (ddof=1) over calibration seeds; FLORES devtest first 60 sentences, batch 8, primary bpb (Qwen has no BOS; first token unscored, bytes counted).
