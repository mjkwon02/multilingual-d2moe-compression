<!--
PROVENANCE
  own arms      : results/deepseek/{cond}/seed{s}/scale_{cond}_seed{s}/eval_ppl.json
  placebo arms  : results/deepseek/{cond}_b/seed{s}/scale_{cond}_b_seed{s}/eval_ppl.json
  baseline      : results/deepseek/baseline/eval_ppl.json
  verdicts      : results/deepseek/phase1_41_headline_gate_result.json
  generated     : 2026-08-02, by tables_main_submission/_make_p1_5.py
  seeds         : 0, 1, 2
  Definitions verbatim from scripts/phase1_41_headline_gate.py; nothing redefined.
-->

# P1-5 — ZH/BN placebo, notes accompanying `p1_5_placebo_zh_bn.csv` / `tableD2_full.csv`

**Status: DONE**, with one source correction and one flag that fails.

---

## 0. ⚠️ Source correction

The request points at `results/placebo_zh_bn_summary.md`. That file (and
`placebo_switch1_zh_bn.json`, `placebo_fisher_rank_zh_bn.json`) contains **Fisher-rank Spearman
ρ for Appendix F/§6.1** — 2-A raw `F_e` ρ, 2-B merge-weight ρ, and the definition-agreement gate
that came back `FLAG_BELOW_REFERENCE_MIN`. It contains **no Δbpb% and no noise-floor numbers**,
so the Table 2 data requested here is not in it.

The bpb placebo values below were read from the **same raw `eval_ppl.json` files that
`phase1_41_headline_gate.py` itself reads** — the authoritative source for Table 2. All 15
placebo eval files (5 conditions × 3 seeds) are present on disk; nothing was reconstructed from
the `placebo_eval_backup_0802.tar.gz` archive.

## 1. Definitions (unchanged)

```
dbpb%(cond, seed, lang) = 100 * (bpb(cond,seed,lang) / bpb_baseline(lang) - 1)
noise_floor(L)          = max over seed of | dbpb%(matched(L),seed,L) - dbpb%(placebo(L),seed,L) |
verdict SUPPORTED       = sign_consistent AND mean(own_gain) > 2 * noise_floor
```

## 2. ZH and BN — own vs placebo on their own language, per seed

**Chinese** (`chinese_only` vs `chinese_only_b`)

| seed | own Δbpb% | placebo Δbpb% | \|diff\| (%p) | placebo ≥ own? |
|:--:|---:|---:|---:|:--:|
| 0 | 3.823641 | 7.979356 | 4.155715 | ✓ |
| 1 | 4.328968 | 8.572869 | **4.243901** | ✓ |
| 2 | 4.566880 | 7.533570 | 2.966690 | ✓ |
| mean | 4.239830 | 8.028598 | — | ✓ |

**Bengali** (`bengali_only` vs `bengali_only_b`)

| seed | own Δbpb% | placebo Δbpb% | \|diff\| (%p) | placebo ≥ own? |
|:--:|---:|---:|---:|:--:|
| 0 | 1.390203 | 1.326236 | 0.063967 | ✗ (placebo better by 0.064) |
| 1 | 1.350312 | 1.212250 | 0.138062 | ✗ (placebo better by 0.138) |
| 2 | 1.248873 | 1.585477 | **0.336603** | ✓ |
| mean | 1.329796 | 1.374654 | — | ✓ (by 0.045) |

## 3. Noise floor and multiple — recomputation vs the body

| lang | noise floor computed | body value | match | own_gain mean | 2× floor | multiple | body multiple | match | verdict |
|---|---:|---:|:--:|---:|---:|---:|---:|:--:|---|
| ZH | 4.243901 | 4.24 | ✓ | 10.758536 | 8.487802 | **2.54×** | 2.54× | ✓ | SUPPORTED |
| BN | 0.336603 | 0.34 | ✓ | 67.850211 | 0.673207 | **201.57×** | 201.57× | ✓ | SUPPORTED |

All five languages reproduce: EN 4.047486 (1.47×, INCONCLUSIVE), KO 0.789616 (76.98×),
ZH 4.243901 (2.54×), SW 0.906945 (39.20×), BN 0.336603 (201.57×). Recomputing the verdict from
the rule reproduces the official verdict for all five.

## 4. Conservativeness flag — the body's claim, checked on ZH/BN

The body claims the placebo arm's own-language degradation is **equal to or slightly higher than**
the matched arm's, so that the comparison does not flatter the matched condition.

```
ZH  conservative_all_seeds : TRUE    (placebo worse on 3/3 seeds, by 2.97-4.24 %p)
BN  conservative_all_seeds : FALSE   (placebo BETTER on seeds 0 and 1)
BN  conservative_on_mean   : TRUE    (placebo worse on the 3-seed mean, by 0.045 %p)
```

* **ZH satisfies it, and by a wide margin** — `chinese_only_b` is 3.8%p worse on its own language
  than `chinese_only`. That is exactly why ZH's floor (4.24) is the second largest of the five and
  why its SUPPORTED verdict has the thin 2.54× margin already noted in the body.
* **BN does not satisfy it seed-by-seed.** On seeds 0 and 1 the placebo sample is *better* on
  Bengali than the matched condition (by 0.064 and 0.138 %p); only seed 2 goes the other way, and
  it is seed 2 that sets the whole floor (0.3366). The 3-seed mean still goes the conservative
  way, but by 0.045 %p.
  This does **not** threaten BN's verdict — a mean own_gain of 67.85 against a floor of 0.34 is
  201.57×, and it would stay SUPPORTED under any floor below 33.9. But the sentence
  *"placebo degradation is equal to or slightly higher than the matched condition's"* should be
  qualified for Bengali, e.g. "on the 3-seed mean" rather than stated per seed. The strict
  per-seed claim holds for EN, KO, ZH and SW, and fails on 2 of 3 BN seeds.

## 5. Appendix D2

`tableD2_full.csv` is the 5-language unified version in the same layout as the existing EN/KO/SW
table: one row per language × seed with own Δbpb%, placebo Δbpb%, absolute and signed difference,
and the conservativeness flag, plus a SUMMARY row per language carrying the noise floor,
own_gain mean, floor multiple and verdict.

Unrelated caveat carried over from `placebo_zh_bn_summary.md`: the **Fisher-rank** ZH/BN placebo
numbers there are a post-hoc supplement whose definition-agreement gate returned
`FLAG_BELOW_REFERENCE_MIN`, and per that run plan they are **withheld from the paper pending
review**. That withholding applies to Appendix F ρ values, **not** to the bpb placebo data in
this item.
