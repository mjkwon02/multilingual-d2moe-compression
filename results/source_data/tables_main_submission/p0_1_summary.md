<!--
PROVENANCE
  source json : results/table5_sec54_bundle.json
  raw runs    : results/deepseek/sec54/step2_bundle.json            (english_only seed0, 2026-07-31 15:45 UTC)
                results/deepseek/sec54/seed1_repro/step2_bundle.json     (english_only seed1, 2026-08-01 01:39)
                results/deepseek/sec54/korean_only_repro/step2_bundle.json (korean_only seed0, 2026-08-02 05:59 UTC)
  grid ref    : results/deepseek/phase1_41_headline_gate_result.json
  generated   : 2026-08-02, by tables_main_submission/_make_p0_1.py  (extraction only, no new runs)
  seeds       : english_only {0,1}; korean_only {0}
-->

# P0-1 — §6.4 korean_only intervention reproduction

Fills the two `[korean_only 조건 재현 결과 대기 — 도착 시 반영]` placeholders (§6.4 and Limitations).
Machine-readable: `p0_1_intervention_korean.json`. Table form: `table5_final.csv`.

**Status: DONE. No STOP.** B-3 integrity passes on all three runs.

---

## 1. korean_only seed0 — bundle restoration gain by eval language

`gain(pp) = Δbpb%_compressed − Δbpb%_intervened`; **positive = damage recovered**.
Only seed 0 exists for korean_only (n=1, no seed replication).

| restored bundle | EN | KO | ZH | SW | BN | vulnerable mean (KO/SW/BN) |
|---|---:|---:|---:|---:|---:|---:|
| treatment 12 pairs | +0.417 | +0.045 | +0.255 | **+1.622** | −0.089 | **+0.526** |
| control 12 pairs | −0.059 | +0.027 | +0.110 | +0.294 | −0.045 | **+0.092** |
| difference (T − C) | **+0.475** | +0.018 | +0.145 | +1.327 | −0.044 | **+0.434** |

Raw values (6 dp), seed 0, single seed:

| row | eng_Latn | kor_Hang | zho_Hans | swh_Latn | ben_Beng | vulnerable_mean |
|---|---:|---:|---:|---:|---:|---:|
| treatment | 0.416645 | 0.044945 | 0.254851 | 1.621834 | −0.089250 | 0.525843 |
| control | −0.058668 | 0.027093 | 0.109994 | 0.294367 | −0.044950 | 0.092170 |
| T − C | 0.475313 | 0.017852 | 0.144857 | 1.327466 | −0.044300 | 0.433673 |

Vulnerable means were independently recomputed from the raw per-language gains
(mean over raw values, not over rounded column entries) and reproduce the stored
`vulnerable_mean_pp` to < 1e-15.

## 2. Vulnerable-column (KO/SW/BN) mean gain — all three runs

| compression condition | seed | treatment | control | difference (T − C) | prereg P1 direction (T > C) |
|---|:--:|---:|---:|---:|:--:|
| english_only | 0 | +0.339731 | +0.383644 | **−0.043913** | ✗ false |
| english_only | 1 | +0.103882 | +0.316852 | **−0.212970** | ✗ false |
| **korean_only** | 0 | +0.525843 | +0.092170 | **+0.433673** | ✓ true |

## 3. Negative control (P3) — English column, T − C

Prereg P3 predicts *no* treatment/control difference on EN; a large EN difference
weakens the "treatment is specific to vulnerable languages" reading.

| compression condition | seed | P3 = EN(T−C) | P1 = vulnerable(T−C) | \|P3\| / \|P1\| |
|---|:--:|---:|---:|---:|
| english_only | 0 | +0.184947 | −0.043913 | **4.21×** |
| english_only | 1 | +0.272269 | −0.212970 | **1.28×** |
| **korean_only** | 0 | **+0.475313** | +0.433673 | **1.10×** |

In korean_only the English negative control moves *more* than the vulnerable-column
effect it is supposed to control for (0.475 vs 0.434). The P3 problem flagged for
english_only/seed0 in `TABLE5_SEC54_BUNDLE_SOURCE.md` §7 is therefore **not** removed by
the korean_only reproduction — it is present in every run, and korean_only has the
largest absolute P3 of the three.

## 4. Direction-replication flags

Flag definition (this document's, stated explicitly because the prereg has no such flag):
`direction_replicated = sign(korean_only seed0) equals the sign observed in **both** english_only seeds`.

```
P1_direction_replicated: false
P3_direction_replicated: true
```

* **P1** — english_only seeds are both negative (−0.044, −0.213); korean_only is **positive** (+0.434).
  Sign flips, magnitude is 2–10× larger. The english_only pattern does **not** reproduce.
* **P3** — positive in all three runs (+0.185 / +0.272 / +0.475). Sign reproduces; magnitude grows.

## 5. B-3 integrity — compressed (un-intervened) model vs the official grid

Each run's `compressed_delta_bpb_pct` was compared against the corresponding
per-seed cell of the official Table 1 grid (`phase1_41_headline_gate_result.json`),
i.e. korean_only seed0 against the **Table 1 korean_only row, seed 0**.

| run | grid row | seed | all 5 langs match at 6 dp | max abs diff | STOP |
|---|---|:--:|:--:|---:|:--:|
| english_only seed0 | english_only | 0 | ✓ | 1.42e-14 | no |
| english_only seed1 | english_only | 1 | ✓ | 1.07e-14 | no |
| **korean_only seed0** | korean_only | 0 | **✓** | **7.11e-15** | **no** |

Expected value was exact agreement, and that is what is observed — the residuals are
float round-off (~1e-14 on values of order 10–90), not a mismatch. The runs' own
`reference_check_bpb_diff` fields are `0.0` for all 5 languages in all three runs.
korean_only seed0 compressed damage, for scale: EN 12.838 / KO 3.659 / ZH 13.007 /
SW 39.831 / BN 25.502 %p — i.e. the +1.622pp SW gain is ~4% of SW's damage, and the
+0.045pp KO gain is ~1.2% of KO's damage.

## 6. Gate information

| run | 2-pair pre-gate | detail |
|---|---|---|
| english_only seed0 | **run** | `sec54/step1_gate.json`. threshold 0.3pp, `max_treatment_gain_pp = 0.102613` → **below threshold**, verdict `BUNDLE_MODE`. Per-pair max: `T_L25E32` 0.001818 (swh), `T_L23E51` 0.102613 (ben). |
| english_only seed1 | **not_run** | no `step1_gate.json` under `sec54/seed1_repro/` (contents: `evals/`, `interventions.csv`, `step2_bundle.json`). |
| **korean_only seed0** | **`gate: not_run`** | no `step1_gate.json` under `sec54/korean_only_repro/` (contents: `evals/`, `interventions.csv`, `step2_bundle.json`). The run went directly to bundle mode, reusing the english_only/seed0 `BUNDLE_MODE` verdict; no korean_only-specific 0.3pp gate value exists. |

## 7. Note for the author — Table 5 placeholder

The body currently has **Table 4 (the pruning table) duplicated into the Table 5 slot**.
The correct Table 5 source is §1–§2 above; `table5_final.csv` exports it in the requested
shape (rows: treatment bundle / control bundle / difference; columns: 5 eval languages +
vulnerable mean), one block per run.

Reporting constraints from `00_docs/TABLE5_SEC54_BUNDLE_SOURCE.md` §5 still apply and are
*strengthened* by this extraction: the main verdict stays the english_only bundle result;
korean_only is a single un-replicated run and must not be used to claim P1 holds; the most
defensible statement is that **the direction of P1 depends on the compression condition**.
The recovery-rate ceiling (0.4–5.7% of the damage) is unchanged in every condition.
