# Reproduction differences (own line item, rule 2)

Every cell re-ran the stored evaluation on the same full 200-question split. The
table below is the stored-vs-rerun difference in accuracy on that split. It is
reported on its own and is never compared with the full->clean difference: the
two are different comparisons (same items/different runs, versus different item
sets/same run). **The cause of these differences is not established.** The
original runs are logged with differing GPU counts, which is circumstantial only;
no experiment was run to confirm any mechanism, and none is requested.

- cells compared: 62
- exactly reproduced: 57 / 62
- max |difference|: 0.0100 accuracy (2 of 200 questions)
- mean |difference|: 0.0006

| condition | seed | lang | stored | rerun | abs diff |
|:---|:---:|:---:|:---:|:---:|:---:|
| Uncompressed |  | EN | 0.525 | 0.530 | 0.005 |
| Uncompressed |  | ZH | 0.470 | 0.465 | 0.005 |
| Uncompressed |  | KO | 0.360 | 0.370 | 0.010 |
| Uncompressed |  | SW | 0.285 | 0.275 | 0.010 |
| Uncompressed |  | BN | 0.285 | 0.275 | 0.010 |
