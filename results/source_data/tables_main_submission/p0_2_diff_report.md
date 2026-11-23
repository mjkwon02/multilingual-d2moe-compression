<!--
PROVENANCE
  reference json : results/table4_pruning_lang.json  (4 conditions; chinese_only ABSENT)
  raw eval files : results/deepseek/{cond}/seed{s}/eval_ppl.json              (pruning OFF)
                   results/deepseek/{cond}/seed{s}/pp_ratio_0.2/eval_ppl.json (pruning ON)
                   results/deepseek/baseline/eval_ppl.json
  body reference : 00_docs/08_figure_정리.md "Table 4" + 00_docs/TABLE4_PRUNING_LANG_SOURCE.md §1/§3
                   + the chinese_only row quoted in the 2026-08-02 data request
  generated      : 2026-08-02, by tables_main_submission/_make_p0_2.py
  seeds          : 0, 1, 2 for all five conditions (no missing cells)
  metric (unchanged, phase1_pruning_gate.py):  100*(bpb_ON - bpb_OFF)/bpb_baseline
-->

# P0-2 — Table 4 final, 5 conditions

Machine-readable: `p0_2_table4_final.csv` (per-seed + mean + sign flag), `p0_2_table4_final.json`.

**Status: DONE. Zero mismatches against the body text — all 25 cells and all 5 vulnerable-column means agree.**

---

## 1. ⚠️ Source-file finding: `table4_pruning_lang.json` contains only **4** conditions

`results/table4_pruning_lang.json` holds `english_only`, `korean_only`, `swahili_only`,
`bengali_only` — **`chinese_only` is not in it.** The generator
`scripts/make_table4_pruning_lang.py` was written for the 4-condition pre-registered set.

chinese_only is nevertheless fully computed elsewhere: **`phase1_pruning_gate_chinese_only_result.json`
exists** (2026-08-02 05:04, `pp_ratio=0.2`, all 5 eval languages, 3 seeds), produced by the
official `phase1_pruning_gate.py`. Its numbers are the authoritative ones.

Two independent cross-checks were run:

| check | result |
|---|---|
| recomputation from raw `eval_ppl.json` vs `table4_pruning_lang.json` (4 conds × 5 langs) | max abs diff **1.18e-14** — reproduces exactly |
| recomputation vs the 5 official `phase1_pruning_gate_*_result.json` files | max abs diff **1.24e-14** — reproduces exactly |

So the recomputation path used for the chinese_only row is validated on the four conditions
that already had a json, and independently agrees with chinese_only's own official gate output.

One asymmetry worth noting: `phase1_pruning_gate_korean_only_result.json` (2026-07-27, the
oldest of the five) covers only **EN/KO/ZH** — SW/BN for korean_only come from
`table4_pruning_lang.json` / the raw files, not from that gate json. The other four gate jsons
cover all 5 languages.

**Action for the author:** regenerate `table4_pruning_lang.json` with `chinese_only` included,
or cite `p0_2_table4_final.csv` as the Table 4 source. Right now no single json backs the
5-row table in the body.

## 2. Table 4 final (3-seed mean, %p)

| calibration \ eval | EN | KO | ZH | SW | BN | vulnerable (KO/SW/BN) mean |
|---|---:|---:|---:|---:|---:|---:|
| english_only | +0.29 | +0.71 | +0.31 | +0.61 | +1.17 | **0.83** |
| **chinese_only** | **+0.67** | **+1.20** | **+0.38** | **+1.17** | **+2.11** | **1.49** |
| korean_only | −0.53† | +2.12 | +0.91 | +0.76 | +4.41 | 2.43 |
| swahili_only | +1.97 | +3.51 | +1.56 | +1.82 | +5.70 | 3.68 |
| bengali_only | +2.37 | +5.30 | +1.91 | +2.03 | +3.92 | 3.75 |

† = the only sign-inconsistent cell in the whole 25-cell table.

## 3. Diff against the numbers already in the body

**No mismatched cell.** All 25 cells round to the value printed in the body at 2 dp, and all
five vulnerable-column means match.

* **chinese_only row (the one the request asked to verify): `+0.67 / +1.20 / +0.38 / +1.17 / +2.11` — CONFIRMED**, all five cells, recomputed from raw *and* matching the official
  `phase1_pruning_gate_chinese_only_result.json` to 1.2e-14. Raw values:
  EN 0.669145, KO 1.195218, ZH 0.381759, SW 1.170969, BN 2.110303.
  All five carry `direction: PRUNING_HURTS` in the gate json.
* All 5 chinese_only cells are **sign-consistent across all 3 seeds** — the chinese_only row
  adds no new instability. Per-seed:

  | eval | seed0 | seed1 | seed2 | mean |
  |---|---:|---:|---:|---:|
  | EN | +0.647821 | +0.742595 | +0.617019 | 0.669145 |
  | KO | +1.244340 | +1.099957 | +1.241355 | 1.195218 |
  | ZH | +0.412697 | +0.258710 | +0.473870 | 0.381759 |
  | SW | +1.560352 | +0.843842 | +1.108712 | 1.170969 |
  | BN | +2.073853 | +2.246545 | +2.010511 | 2.110303 |
  | vuln mean | +1.626182 | +1.396781 | +1.453526 | **1.492163** |

## 4. korean_only × EN — final status of the † cell

Unchanged after adding the fifth condition: this remains the **only** sign-inconsistent cell
in the 25-cell table.

| seed0 | seed1 | seed2 | mean | sign consistent |
|---:|---:|---:|---:|:--:|
| **+0.079211** | **−0.090181** | **−1.576277** | −0.529083 | **✗** |

The mean is driven almost entirely by seed2 (−1.576); seeds 0 and 1 are near zero and of
opposite sign to each other. The existing body caveat ("this cell cannot support a
conclusion") stands, and the addition of chinese_only (+0.67 on EN, sign-consistent) means
english_only (+0.29) is now the **smallest sign-consistent** EN value.

## 5. Vulnerable-column ordering

Computed ascending (5 conditions):

```
english_only 0.829724  <  chinese_only 1.492163  <  korean_only 2.429895
             <  swahili_only 3.675519  <  bengali_only 3.750502
```

Rounded: **0.83 < 1.49 < 2.43 < 3.68 < 3.75** — **matches the body's stated ordering exactly.**
`ORDER_MATCHES_BODY: true`. Note the top two (3.68 vs 3.75) are separated by 0.075pp, which is
smaller than the seed spread of either condition (swahili_only 3.460–3.823, bengali_only
3.645–3.876) — the swahili/bengali ordering is not seed-robust and should not be asserted as a
strict ranking.

## 6. §5.4 opening sentence — "four calibration conditions"

**Conditions actually present:**

* in `results/table4_pruning_lang.json`: **4** (`english_only`, `korean_only`, `swahili_only`, `bengali_only`)
* in the raw results tree with complete pruning ON/OFF × 3 seeds: **5** (the four above **+ `chinese_only`**)
* in the final table above and in `p0_2_table4_final.csv`: **5**

So the body's "**네 개의 calibration 조건**" is correct *for the json as it currently stands*
but wrong *for the table as printed*, which has five rows. The evidence for changing the
wording to five conditions: all 15 chinese_only cells (5 langs × 3 seeds) exist on disk under
`results/deepseek/chinese_only/seed{0,1,2}/{,pp_ratio_0.2/}eval_ppl.json`,
none missing, and all 5 means are sign-consistent.

Also note that `00_docs/TABLE4_PRUNING_LANG_SOURCE.md` and `00_docs/08_figure_정리.md` both still
describe Table 4 as "4조건 × 3 seed", and `PRUNING_LANG_PREREG.md` §6 makes the "4조건 × 3 seed"
annotation **mandatory** when claiming the monotone ordering. Adding chinese_only to the table
means either (a) that pre-registration constraint has to be re-stated for 5 conditions with
chinese_only explicitly labelled post-hoc, or (b) chinese_only is reported as a supplementary
row outside the pre-registered ordering claim. **Author's call — not changed here.**
