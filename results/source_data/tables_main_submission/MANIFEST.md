# MANIFEST — tables_main_submission/20260802

Data extraction for the body's unfilled TODOs. **No new compression runs; CPU only.**
Generated 2026-08-02 from already-completed queue output.

Location note: the request said `tables_main_submission/`; this bundle sits at
**`tables_main_submission/`**, matching the convention that all research artefacts live under
`` (as do `results/`, `00_docs/`, `logs/` referenced throughout the request).

**Global rule applied:** where a computed value disagreed with the body, the value was **not**
changed — the discrepancy is recorded in the diff report only. Author does the body edits.
All computations reuse existing pre-registered definitions; **no new definition was introduced**.

---

## Status by item

| item | status | outputs |
|---|:--:|---|
| **P0-1** §6.4 korean_only intervention | **DONE** | `p0_1_intervention_korean.json`, `p0_1_summary.md`, `table5_final.csv` |
| **P0-2** Table 4 final (chinese_only) | **DONE** | `p0_2_table4_final.csv`, `p0_2_table4_final.json`, `p0_2_diff_report.md` |
| **P0-3** Table 6 source (4 axes) | **DONE** (axis 4 partial) | `p0_3_table6_source.csv`, `p0_3_table6_source.json`, `p0_3_table6_draft.md` |
| **P1-4** stage2×2 EN/SW | **DONE** | `p1_4_stage2x2_ensw.json`, `p1_4_summary.md` |
| **P1-5** ZH/BN placebo + D2 | **DONE** | `p1_5_placebo_zh_bn.csv`, `tableD2_full.csv`, `p1_5_placebo_zh_bn.json`, `p1_5_placebo_notes.md` |
| **P1-6** SIB 0-shot record | **DONE** | `p1_6_sib_0shot_record.md` |
| **P2-7** appendix inventory | **DONE** (1 MISSING, expected) | `p2_7_appendix_inventory.md`, `_p2_7_inventory.tsv` |

**No STOP flags raised.** Generating scripts are kept alongside as `_make_p0_1.py` … `_make_p2_7.py`.

---

## Headline results

**P0-1 — korean_only reproduction (fills both `[korean_only 조건 재현 결과 대기]` placeholders)**

| | treatment vuln | control vuln | P1 = T−C | direction |
|---|---:|---:|---:|:--:|
| english_only seed0 | +0.339731 | +0.383644 | −0.043913 | ✗ |
| english_only seed1 | +0.103882 | +0.316852 | −0.212970 | ✗ |
| **korean_only seed0** | **+0.525843** | **+0.092170** | **+0.433673** | **✓** |

* `P1_direction_replicated: false` — sign flips vs english_only, magnitude 2–10× larger.
* `P3_direction_replicated: true` — positive in all three runs (+0.185 / +0.272 / **+0.475**).
* **Negative control worsens, not improves:** in korean_only the English P3 difference (0.475) is
  *larger* than the vulnerable-column effect it controls for (0.434), ratio 1.10×.
* **B-3 integrity: PASS.** All three runs reproduce their official Table 1 grid cells at 6 dp
  (max abs diff 7e-15 for korean_only). No STOP.
* **Gate: `not_run`** for korean_only seed0 (and for english_only seed1) — no `step1_gate.json`
  exists in `sec54/korean_only_repro/`. Only english_only seed0 has one (max gain 0.102613 pp,
  below the 0.3pp threshold, verdict `BUNDLE_MODE`).
* n=1 for korean_only. The defensible statement remains "P1's direction depends on the
  compression condition", not "P1 holds".

**P0-2 — Table 4, 5 conditions: zero mismatches**

* All 25 cells and all 5 vulnerable-column means match the body at 2 dp.
* **chinese_only row `+0.67 / +1.20 / +0.38 / +1.17 / +2.11` CONFIRMED**, all 5 cells
  sign-consistent across 3 seeds.
* Ordering **0.83 < 1.49 < 2.43 < 3.68 < 3.75 confirmed** (but the top two differ by 0.075pp,
  less than either condition's seed spread — not a seed-robust ranking).
* korean_only × EN stays the single sign-inconsistent cell: +0.079211 / −0.090181 / −1.576277.
* ⚠️ **`results/table4_pruning_lang.json` has only 4 conditions — chinese_only is absent.**
  It does have an official gate output (`phase1_pruning_gate_chinese_only_result.json`,
  2026-08-02 05:04). Recomputation reproduces both the 4-condition json (1.18e-14) and all five
  gate jsons (1.24e-14). Regenerate the json or cite `p0_2_table4_final.csv`.
* §5.4's "네 개의 calibration 조건" is correct for the json, wrong for the 5-row printed table.

**P0-3 — Table 6 source, all four axes computed**

| condition | max bpb regret (%p) | worst-lang Δbpb% | SIB collapse (full/partial) | Belebele macro |
|---|---:|---:|:--:|---:|
| english_only | 87.69 | 89.02 | 1 / 0 | 0.702 |
| korean_only | 38.57 | 38.52 | 0 / 2 | 0.703 |
| chinese_only | 86.08 | 87.41 | 1 / 1 | 0.658 |
| swahili_only | 73.73 | 75.06 | 1 / 0 | — |
| bengali_only | 35.35 | 35.30 | 0 / 1 | — |
| **Balanced** | **2.88** | **9.29** | **0 / 1** | **0.790** |

* Balanced regret row **matches the body exactly** (EN 2.01 / KO 1.50 / ZH 2.86 / SW 2.88 / BN 1.57).
* Balanced Belebele macro 0.790 **is the highest** — confirmed.
* Collapse counting: the request's "18셀" is the **18 runs**; the cell grid is 30 (or 90 with seeds).
* **Axis 4 is only 4 of 6 conditions** and 3 of those are off-only on EN/KO/ZH — swahili_only and
  bengali_only were never run on Belebele. Axis choice left to the author (see draft §5).

**P1-4 — stage2×2 EN/SW: VERDICT = MIXED**

* swh_Latn column: scale effect **23.854670**, fisher effect **9.285395**, ratio **2.569**.
  Both sign-consistent across 3 seeds; misses the registered 3× SCALE_DOMINANT bar.
* eng_Latn (descriptive only): scale −3.832410, fisher −1.718296. **Do not report a verdict here** —
  both effects are negative and the rule inverts.
* **No placebo noise floor exists for this pair** (prereg says so explicitly; the placebo Fisher
  artefacts were deleted). The requested "multiple of the noise floor" is therefore not reportable.
  The prereg's three registered substitutes are supplied instead.
* **Reproducibility: the 2026-07-30 `eng×scale_swh/seed0` file and the 2026-08-02 re-run are
  bitwise identical on all 5 languages (max abs diff exactly 0.0)** — stronger than the 6-dp
  agreement asked for.
* Consequence: **Limitations "경로 분리 실험의 언어 범위" paragraph stays**, and §5.4 gains the
  language-pair-dependence sentence (EN/KO ratio very large vs EN/SW 2.57).

**P1-5 — ZH/BN placebo: all body values reproduce**

* ZH floor **4.243901** (body 4.24), multiple **2.54×** — matches. BN floor **0.336603**
  (body 0.34), multiple **201.57×** — matches. All 5 languages reproduce, verdicts unchanged.
* ⚠️ **Source correction:** `placebo_zh_bn_summary.md` is Fisher-rank ρ for Appendix F and contains
  **no Δbpb%**. The bpb data was read from the raw eval files `phase1_41_headline_gate.py` uses.
  All 15 placebo eval files are present on disk.
* ⚠️ **Conservativeness flag fails for BN:** placebo is *better* than the matched condition on
  seeds 0 and 1 (−0.064, −0.138 pp); only seed 2 goes the conservative way, and it sets the floor.
  Conservative on the 3-seed mean (+0.045). ZH satisfies it on 3/3 seeds. The verdict is
  unaffected (201.57× vs a 2× bar) but the per-seed wording should be softened for Bengali.
* `tableD2_full.csv` = 5-language unified Appendix D2.

**P1-6 — SIB 0-shot: WITHHELD, as expected**

* baseline BN 0-shot: acc 0.3039, **top-label 0.9216 > 0.9 → the uncompressed control itself
  collapses**, so by the registered rule **both** `COLLAPSE_PERSISTS_0SHOT` and
  `COLLAPSE_PROTOCOL_DEPENDENT` are undeclarable. 14-shot headline unchanged.
* english_only seed0: acc **0.2500**, top-label **1.00**, all 204 items on `science/technology`.
* **NaN exclusion verified:** `pred_distribution` sums to exactly 204 = `n_samples` (nothing
  skipped); the 10 stored `_sample_items` carry 70 label logprobs, **all finite**, range
  −9.511370 to −2.216584. Gold distribution checked against the dataset itself
  (`Davlan/sib200`, `ben_Beng`, test): 51 `science/technology` of 204 = **0.25 exactly**.
  **Conclusion: a genuine degenerate prediction, not an artefact.** A 3-sentence Limitations
  paragraph is drafted in the record.

**P2-7 — inventory: 45/46 present**

* Only MISSING: `results/calib_token_length_bengali_only.json` — the file the docs already
  marked as broken and "인용 금지". Table A1's real source is present. No action needed.
* 표 F3 exists for **seed0 only**; the ZH/BN Fisher-rank supplement carries a
  `FLAG_BELOW_REFERENCE_MIN` withholding flag.
* 부록 H's raw data is complete, but **`SEC54_RESULT.md` is stale** — english_only/seed1 and
  korean_only/seed0 are not in it. Appendix H cannot be written from it alone.
* The request's description of 표 D1 ("KO↔BN 교차 완화 ±0.7%p") differs from the document's
  definition; both are backed by the same file, and the ±0.7 figure is pinned down in the inventory.

---

## Discrepancies for the author (values NOT changed here)

1. `table4_pruning_lang.json` is missing chinese_only — regenerate or cite the new CSV. (P0-2 §1)
2. §5.4 says "four calibration conditions"; the table has five rows. (P0-2 §6)
3. The Table 5 slot in the body currently holds a duplicate of Table 4. `table5_final.csv` is the
   correct source. (P0-1 §7)
4. The BN placebo conservativeness sentence is false per-seed, true on the mean. (P1-5 §4)
5. `SEC54_RESULT.md` documentation debt — two completed runs unrecorded. (P2-7 note 5)
6. 표 A1's published "±0.5%" seed variation and Swahili fertility 2.45 were already corrected to
   "±1.7%" and 2.44 by the 2026-07-31 re-measurement; check whether the body still has the old
   values. (P2-7 note 7)
7. SIB-200 includes collapsed seeds in retention averages while the Belebele gate excludes flagged
   languages — Table 6 axes 3 and 4 would use opposite conventions. (P0-3 §5)

---

*Update 2026-11-23 (public repository cleanup):* `p0_3_table6_draft.md`,
`p2_7_appendix_inventory.md`, `_p2_7_inventory.tsv`, and `_make_p2_7.py`
were moved to the private archive (`ARR_8/private_release_archive/`) —
submission-era draft/inventory records superseded by the camera-ready
mapping (`docs/PAPER_FIGURE_TABLE_MAPPING.md`). All table-source outputs
and their `_make_*.py` generators remain here unchanged.
