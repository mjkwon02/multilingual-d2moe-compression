<!--
PROVENANCE
  off-diagonal cells : results/deepseek/english_only/seed{0,1,2}/scale_swahili_only_seed{s}/eval_ppl.json
                       results/deepseek/swahili_only/seed{0,1,2}/scale_english_only_seed{s}/eval_ppl.json
                       run 2026-08-02, driver scripts/stage2x2_sw_0802_driver.sh, log logs/stage2x2_sw_0802.log
                       6 cells, 0 skipped, finished 2026-08-02T09:01:14Z
  diagonal cells     : english_only/seed{s}/scale_english_only_seed{s}/eval_ppl.json
                       swahili_only/seed{s}/scale_swahili_only_seed{s}/eval_ppl.json   (REUSED, never recomputed)
  baseline           : results/deepseek/baseline/eval_ppl.json
  generated          : 2026-08-02, by tables_main_submission/_make_p1_4.py
  seeds              : 0, 1, 2
  Definitions and verdict rule quoted verbatim from the mini-preregistration fixed
  2026-08-02T07:29:39Z BEFORE the first merge (logs/stage2x2_sw_0802.log). Nothing new introduced.
-->

# P1-4 — stage2×2 EN/SW main effects and verdict

Machine-readable: `p1_4_stage2x2_ensw.json`.

## VERDICT: **MIXED** (primary, swh_Latn column)

---

## 1. Rule as registered (quoted verbatim from `logs/stage2x2_sw_0802.log`)

```
PRIMARY METRICS (evaluated on the swh_Latn column):
  scale-language effect  = mean over fisher in {EN,SW} of
                           [ dbpb%(scale=EN) - dbpb%(scale=SW) ]
  fisher-language effect = mean over scale  in {EN,SW} of
                           [ dbpb%(fisher=EN) - dbpb%(fisher=SW) ]
  Both are main effects of the 2x2; positive = the mismatched (EN)
  calibration language is worse for Swahili.
  PREDICTION (same direction as the original EN/KO 2x2): scale effect >> fisher effect.

DECISION RULE:
  NO NOISE FLOOR IS AVAILABLE. ...
  VERDICT VOCABULARY:
    SCALE_DOMINANT  = scale effect > 3x fisher effect AND both effects
                      sign-consistent across all 3 seeds
    MIXED           = otherwise
    FISHER_DOMINANT = fisher effect exceeds scale effect
  The headline Swahili floor of 0.91%p may be quoted for orientation ONLY,
  and MUST carry the label 'noise floor from a DIFFERENT level of analysis
  (headline single-condition comparison), not a floor for this 2x2'.

SECONDARY: the identical computation on the eng_Latn column. Descriptive only.
```

## 2. The 2×2 cells (Δbpb% vs uncompressed baseline, swh_Latn column)

| seed | Fisher=EN, Scale=EN | Fisher=EN, Scale=SW | Fisher=SW, Scale=EN | Fisher=SW, Scale=SW |
|:--:|---:|---:|---:|---:|
| 0 | 34.776192 | 0.760366 | 15.851902 | 0.017746 |
| 1 | 30.248408 | 0.992630 | 14.977385 | 0.064206 |
| 2 | 34.253162 | 0.564584 | 15.196069 | −0.224386 |

## 3. Main effects — primary (swh_Latn)

| seed | scale-language effect (%p) | fisher-language effect (%p) |
|:--:|---:|---:|
| 0 | 24.924996 | 9.833450 |
| 1 | 22.084506 | 8.099696 |
| 2 | 24.554508 | 9.923040 |
| **3-seed mean** | **23.854670** | **9.285395** |
| sign-consistent across 3 seeds | ✓ | ✓ |
| seed spread (max − min) | 2.840490 | 1.823344 |
| effect ÷ own seed spread | 8.40× | 5.09× |

**ratio scale / fisher = 2.569053**

## 4. Main effects — secondary (eng_Latn), descriptive only

| seed | scale-language effect (%p) | fisher-language effect (%p) |
|:--:|---:|---:|
| 0 | −3.415156 | −1.597107 |
| 1 | −4.352617 | −1.786732 |
| 2 | −3.729456 | −1.771048 |
| **3-seed mean** | **−3.832410** | **−1.718296** |

Both are negative, which is the expected direction (English calibration helps English), and the
scale path is again the larger of the two in magnitude (|−3.83| / |−1.72| = **2.23×**).

⚠️ **Do not quote a verdict for this column.** The prereg assigns it none, and the vocabulary was
written for a column where "positive = mismatched calibration is worse". Applied mechanically to
two negative numbers, `FISHER_DOMINANT = fisher effect exceeds scale effect` fires on
−1.72 > −3.83 and inverts the intended meaning. The json records this under
`verdict_if_rule_applied_mechanically` with `do_not_report_as_verdict: true`.

## 5. Verdict derivation (primary)

```
FISHER_DOMINANT?  fisher (9.285395) > scale (23.854670)?          NO
SCALE_DOMINANT?   scale (23.854670) > 3 x fisher (27.856186)?     NO  <-- fails here
                  both sign-consistent across 3 seeds?            YES
MIXED             = otherwise                                     --> VERDICT
```

**VERDICT: MIXED.** The scale path is the larger of the two by a factor of 2.57, and the
direction predicted by the original EN/KO 2×2 does reproduce (scale > fisher, both positive, all
three seeds) — but the ratio falls short of the registered 3× bar for SCALE_DOMINANT. It misses
by 4.00%p of scale effect (23.85 observed vs 27.86 required).

## 6. Noise floor

**No placebo noise floor exists for the EN/SW pair.** The prereg is explicit: the original 2×2's
floor came from placebo Fisher artefacts which have since been deleted, and it states this as a
limitation rather than substituting anything. The requested "multiple of the placebo noise
floor" therefore **cannot be computed** and is not reported.

For orientation only, dividing by the headline Swahili floor of 0.91%p — which **must** carry
the label *"noise floor from a DIFFERENT level of analysis (headline single-condition
comparison), not a floor for this 2×2"* — gives scale 26.21× and fisher 10.20×. These are not
floor multiples for this analysis.

The prereg's three registered substitutes are given instead, all in §3:
(a) both effects sign-consistent across all 3 seeds; (b) ratio 2.57; (c) each effect against the
seed spread of the cells that produced it — scale 8.40×, fisher 5.09×.

## 7. Reproducibility record

`english_only/seed0/scale_swahili_only_seed0/eval_ppl.json` already existed from 2026-07-30
04:53 with no log recording how it was produced. Per the prereg it was **not** reused: it was
copied to `eval_ppl_prior_20260730.json` (2026-08-02 06:02) and the cell was re-run.

| lang | 2026-07-30 prior bpb | 2026-08-02 re-run bpb | abs diff |
|---|---:|---:|---:|
| eng_Latn | identical | identical | 0.0 |
| kor_Hang | identical | identical | 0.0 |
| zho_Hans | identical | identical | 0.0 |
| swh_Latn | identical | identical | 0.0 |
| ben_Beng | identical | identical | 0.0 |

**All five languages agree bitwise** (`all_bitwise_identical: true`, max abs diff exactly 0.0) —
stronger than the 6-decimal-place agreement the request asked to record. Exact per-language
values are in `p1_4_stage2x2_ensw.json` → `reproducibility_20260730`. The eval path is
deterministic on this host.

## 8. Where this result is used

Because the verdict is **MIXED**, not SCALE_DOMINANT:

* **Limitations — "경로 분리 실험의 언어 범위" paragraph: KEEP.** The EN/SW extension does not
  license relaxing it. Two reasons, both citable: (i) the registered SCALE_DOMINANT bar was not
  met on the EN/SW pair; (ii) no noise floor exists for this pair, so the extension is weaker
  evidence than the EN/KO 2×2 regardless of its point estimate.
* **§5.4 — add one sentence on language-pair dependence.** The supporting numbers: on EN/KO the
  scale path dominates by a very large margin (own_scale_gain 38.95%p at Fisher=KO vs the
  Fisher-alone reference 5.60%p, SUPPORTED at 5.66× the floor); on EN/SW the same computation
  gives scale 23.85%p vs fisher 9.29%p, a ratio of only 2.57. The qualitative ordering
  (scale > fisher) reproduces on both pairs; the *magnitude* of the separation does not.
* This is a **post-hoc extension** of the 2026-07-26 EN/KO registration and must be labelled as
  such wherever it appears, reported descriptively beside the EN/KO result and never in place of it.
