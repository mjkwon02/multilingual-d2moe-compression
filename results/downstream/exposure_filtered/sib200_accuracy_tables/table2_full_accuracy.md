# Table 2 (revised) — SIB-200 accuracy (%), full test set, n=204

| Calibration | EN | ZH | KO | SW | BN |
|:---|:---:|:---:|:---:|:---:|:---:|
| Uncompressed | 78.4 | 80.4 | 76.5 | 42.2 | 55.9 |
| English calibration | **76.8 ± 4.2** | 77.5 ± 4.7 | 42.2 ± 6.8 | 28.8 ± 6.5 | 12.9 ± 5.0 [3/3] |
| Chinese calibration | 74.3 ± 2.8 | **76.3 ± 2.0** | 39.1 ± 1.6 | 22.9 ± 4.2 [2/3] | 13.9 ± 7.1 [3/3] |
| Korean calibration | 62.7 ± 2.6 | 69.9 ± 1.2 | 52.5 ± 2.1 | 27.1 ± 3.0 [1/3] | 26.3 ± 3.0 [1/3] |
| Swahili calibration | 61.9 ± 6.4 | 65.7 ± 5.5 | 38.9 ± 6.7 | 34.5 ± 2.0 | 10.1 ± 1.9 [3/3] |
| Bengali calibration | 53.1 ± 13.9 | 64.4 ± 10.7 | 51.5 ± 14.8 | 22.4 ± 3.2 [2/3] | **39.4 ± 12.5** |
| Balanced-character calibration | 66.0 ± 2.2 | 69.0 ± 3.5 | **59.8 ± 2.5** | **36.9 ± 0.7** | 29.1 ± 6.1 [1/3] |

**Caption.** SIB-200 topic-classification accuracy (%) after compression, full test set (n = 204 per language; 14-shot, fixed English demonstrations). Cells for compressed models are mean ± sample standard deviation (ddof = 1) over 3 calibration seeds, in percentage points; the Uncompressed row is a single evaluation and carries no standard deviation. All seeds, including collapsed ones, enter the mean and standard deviation. [k/3] marks the number of seeds whose predictions collapsed (top-label share > 0.9 or accuracy < 25%); cells without a bracket have no collapsed seed. The 25% threshold is the project's pre-existing collapse criterion and is not the 7-way uniform-chance rate (1/7 ≈ 14.3%). Unlike the submitted version, which replaced fully collapsed cells with a placeholder, raw accuracy is now reported for every cell and collapse is marked separately. Bold marks, within each evaluation-language column, the higher mean of the own-language calibration and Balanced-character calibration; it is a descriptive comparison of means, not a significance test. Values are the originally reported runs; retention relative to the uncompressed model is given separately in the retention table.
