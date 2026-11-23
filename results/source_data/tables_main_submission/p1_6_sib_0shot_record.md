<!--
PROVENANCE
  run driver / prereg : logs/sib_0shot_0802.log  (mini-preregistration fixed 2026-08-02T09:01:38Z,
                        BEFORE any eval; runs completed 2026-08-02T09:56:19Z, 0 skipped)
  result files:
    baseline            results/deepseek/baseline/eval_sib200_0shot.json
    english_only seed0  results/deepseek/english_only/seed0/scale_english_only_seed0/eval_sib200_0shot.json
    bengali_only seed0  results/deepseek/bengali_only/seed0/scale_bengali_only_seed0/eval_sib200_0shot.json
  14-shot anchors     : same directories, eval_sib200.json;  grid: sib200_extended/SIB_EXTENDED_REPORT.md
  generated           : 2026-08-02
  seeds               : seed 0 only (the experiment is specified at seed 0); baseline has no seed
  NO NEW ANALYSIS. Record only -- both verdicts are withheld by the pre-registered control rule.
-->

# P1-6 — SIB-200 0-shot operating point: record for the appendix / Limitations

**Status: DONE. Verdict: WITHHELD (both), by the pre-registered control rule. 14-shot headline unchanged.**

---

## 1. Results — 3 conditions, Bengali, n=204, 0-shot

| condition | num_fewshot | n | acc | top_label_frac | pred_distribution | collapsed by rule? |
|---|:--:|:--:|---:|---:|---|:--:|
| **baseline** (uncompressed) | 0 | 204 | **0.3039** | **0.9216** | sci/tech 188, geography 15, health 1 | **YES** (top > 0.9) |
| english_only seed0 | 0 | 204 | **0.2500** | **1.0000** | sci/tech 204 | **YES** (both criteria) |
| bengali_only seed0 | 0 | 204 | **0.2794** | **0.9020** | sci/tech 184, geography 20 | **YES** (top > 0.9) |

14-shot anchors at seed 0 (the comparison basis fixed in the prereg — note these are seed-0 cells,
not the 3-seed means printed in the extended report):

| condition | 14-shot acc | 14-shot top_label_frac | collapsed at 14-shot? |
|---|---:|---:|:--:|
| baseline | 0.5588 | 0.382 | no |
| english_only seed0 | 0.0931 | 0.941 | **yes** (both criteria) |
| bengali_only seed0 | 0.2696 | 0.417 | no (0.02 above the acc < 0.25 line) |

## 2. Pre-registration, quoted verbatim (`logs/sib_0shot_0802.log`, fixed 2026-08-02T09:01:38Z)

```
COLLAPSED CRITERION (unchanged): top-label ratio > 0.9 OR acc < 0.25.
  Majority-class floor is ~25%, not 1/7.

VERDICT VOCABULARY:
  COLLAPSE_PERSISTS_0SHOT     = english_only/seed0 BN still meets the criterion at 0-shot
  COLLAPSE_PROTOCOL_DEPENDENT = it does not
  plus a control sanity line: whether baseline BN stays uncollapsed at 0-shot.
  If baseline itself collapses at 0-shot, the 0-shot operating point is
  uninformative about compression and BOTH verdicts are withheld.
No interpretation beyond those words is licensed. The 14-shot protocol is NOT replaced.
```

## 3. Verdict: **WITHHELD**

The control fired. `baseline` Bengali at 0-shot has **top_label_frac = 0.9216 > 0.9**, so the
uncompressed model is itself collapsed at this operating point. By the sentence quoted above,
the 0-shot operating point is uninformative about compression and **both verdicts are withheld**:

* `COLLAPSE_PERSISTS_0SHOT` — **cannot be declared.**
* `COLLAPSE_PROTOCOL_DEPENDENT` — **cannot be declared.**

The observation that english_only/seed0 does still meet the collapse criterion at 0-shot is
recorded above as a number, but it may **not** be reported as `COLLAPSE_PERSISTS_0SHOT`: the
criterion is met by all three arms including the uncompressed control, so it separates nothing.

**The 14-shot headline stands unchanged.** Table 3 and every SIB-200 number in §4.2 remain the
14-shot campaign; this run replaces nothing.

## 4. english_only acc = 0.2500 exactly — NaN exclusion checked, prediction is genuinely degenerate

The exact quarter-value invites the suspicion that items were dropped (e.g. NaN logprobs) and
0.25 is an artefact. It is not.

**Verification, and how it was done:**

1. **All 204 items produced a prediction.** In
   `english_only/seed0/scale_english_only_seed0/eval_sib200_0shot.json` →
   `results.ben_Beng.pred_distribution`, the counts are
   `{science/technology: 204, travel: 0, politics: 0, sports: 0, health: 0, entertainment: 0, geography: 0}`,
   which **sums to exactly 204 = `n_samples`**. The evaluator emits no prediction for an item it
   skips, so a sum of 204 means nothing was NaN-excluded. 204/204 accounted for.
2. **The stored logprobs are all finite.** `results.ben_Beng._sample_items` holds 10 saved items,
   each with all 7 single-token label logprobs — **70 values, all finite**, spanning
   **−9.511370 to −2.216584**. The first item's 7 values are the −2.939998 … −9.439999 range
   quoted in the data request. No `-inf`, no `nan`.
   (Only 10 items are stored with per-item logprobs; the 204/204 claim rests on step 1, which is
   the complete count, not on step 2.)
3. **0.2500 is the real accuracy of a one-label predictor.** The model put *every* one of the 204
   items on `science/technology`. Checked directly against the dataset
   (`Davlan/sib200`, config `ben_Beng`, split `test`, from the local cache at
   `$HF_HOME/datasets/Davlan___sib200/ben_Beng/`): n = 204 with the gold
   distribution `science/technology 51, travel 40, politics 30, sports 25, health 22,
   entertainment 19, geography 17` — **51/204 = 0.25 exactly**. The round number is SIB-200's
   majority-class floor reached by a fully degenerate predictor, not a division by a reduced
   denominator. (This is also the 0.25 in the collapse criterion `acc < 0.25`, which is why
   english_only sits exactly *on* the line rather than below it.)

**Conclusion: a genuine degenerate prediction, not a NaN artefact.** The same check on the other
two arms: baseline 188/204 on one label (top 0.9216), bengali_only 184/204 (top 0.9020) — both
also sum to 204, both all-finite over their 10 stored items.

## 5. Two-to-three sentences for the Limitations shot-sensitivity paragraph

> The 14-shot operating point adopted for the SIB-200 campaign was not validated against
> alternatives, and a 0-shot check run at seed 0 on Bengali could not settle the question: the
> **uncompressed baseline itself degenerates at 0-shot** (top-label fraction 0.92, acc 0.30), so by
> the pre-registered control rule the 0-shot comparison is uninformative about compression and both
> verdicts were withheld. What the check does establish is that the collapses reported in the BN
> column are real degeneracies rather than measurement artefacts — english_only/seed0 places all
> 204 Bengali items on a single label, with all label logprobs finite and every item scored, giving
> exactly the majority-class accuracy of 0.25. Whether the collapse is specific to 14 shots
> therefore remains open, and we do not claim shot-count robustness.

## 6. Related open items (context, not part of this record)

`08_figure_정리.md` Table 3 caveat 5 records the earlier spot check in `SIB_DIAGNOSIS.md` CHECK C
(korean_only/seed0, eng_Latn, n=20): 0-shot 0.850 > 7-shot 0.650 > 14-shot 0.350, i.e. the adopted
14-shot was the worst of the three on that small probe. That probe and this 0-shot run are
separate: the probe is n=20 on English, this is n=204 on Bengali with a registered control.
Neither licenses changing the operating point.
