# Belebele accuracy (%), clean subset (per-language n: EN 99, ZH 99, KO 113, SW 99, BN 125)

| Calibration | EN | ZH | KO | SW | BN |
|:---|:---:|:---:|:---:|:---:|:---:|
| Uncompressed | 56.6 | 51.5 | 34.5 | 35.4 | 29.6 |
| English calibration | 38.0 ± 4.1 | 40.1 ± 5.9 | 24.5 ± 0.5 [c] | -- | -- |
| Chinese calibration | 34.3 ± 4.4 | 36.7 ± 0.6 | 23.9 ± 0.9 [c] | -- | -- |
| Korean calibration | 36.0 ± 4.6 | 33.7 ± 4.2 | 27.7 ± 1.0 | 23.9 ± 3.5 [c] | 26.7 ± 1.2 |
| Korean calibration, pruning 0.2 | 35.0 ± 5.6 | 32.7 ± 4.6 | 25.1 ± 1.4 | 20.9 ± 3.1 [c] | 24.5 ± 2.0 [c] |
| Balanced-character calibration | 39.4 ± 3.5 | 42.1 ± 4.8 | 30.7 ± 4.5 | -- | -- |

**Caption.** Belebele multiple-choice reading-comprehension accuracy (%) after compression, clean subset, 5-shot. The clean subset removes calibration-exposed passages at passage granularity from the evaluated 200-question split, leaving EN 99, ZH 99, KO 113, SW 99, BN 125 questions (EN 52, ZH 52, KO 59, SW 52, BN 65 passages respectively). The question set is identical across all models within a language, verified from the stored per-item records; across languages the sets differ, because the Belebele split order is language-specific and every reported cell used a 200-question limit -- cross-language comparisons carry that caveat. Compressed cells are mean ± sample standard deviation (ddof = 1) over 3 calibration seeds; the Uncompressed row is a single evaluation. [c] marks a value at or below the 4-way uniform-chance rate of 25%; no test of equality to chance was performed, so a cell without [c] is not thereby shown to differ from chance. The project's collapse criterion is not applied: its top-label-share half needs the predicted option, which the stored Belebele records do not carry. Dashes are cells the submitted paper never reported and that were not run (swahili_only and bengali_only have no Belebele cell; the three English/Chinese/Balanced conditions reported EN, KO and ZH only).
