# Revised-run time accounting

Work time is wall-clock (two GPUs used together count once) and **excludes**
time blocked waiting for the exclusive merge lock. Time spent under the
superseded 600-item plan is reported separately and is NOT charged here.

- **Budget start:** 2026-09-16T04:36Z
- **Budget:** 12 h work time → nominal stop **2026-09-16T16:36Z**
- **Validation+probe+pilot sub-budget:** 2 h → **06:36Z**

| UTC | phase | note |
|---|---|---|
| 04:36 | amendment accepted, budget starts | E2 on GPU0-1 already stopped by user at 04:29:07Z |
| 04:36-04:53 | revised protocol built (CPU) | 200 test IDs (hash-ordered), 59 dev IDs, overlap check = 0 hits |
| 04:40-04:53 | P0 reference validation finished (GPU2-3) | 5 languages; reused, P8 half deliberately not run |
| 04:53- | SW/BN near-tie vs cache-error resolution | per-item margins captured |

## Spent before the amendment (reported, not charged)

Roughly 04:36 − 03:05 ≈ **1.5 h** wall-clock on the superseded plan: MASSIVE
data acquisition, the 600-item protocol freeze, the scorer-cache bug discovery
and fix, and the scorer ablations. The bug fix and the reference validation are
reused by this run; the 600-item protocol artifacts are preserved but unused.
