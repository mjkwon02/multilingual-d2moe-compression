"""EN/SW Fisher x Scale 2x2 -- main effects (P3 STEP 1, 1-c; 부록 B.6).

Implements exactly the metrics fixed in the mini-preregistration logged at
2026-08-02T06:03:27Z (logs/stage2x2_sw_0802.log), no more:

  scale-language effect  = mean over fisher in {EN,SW} of
                           [ dbpb%(scale=EN) - dbpb%(scale=SW) ]
  fisher-language effect = mean over scale  in {EN,SW} of
                           [ dbpb%(fisher=EN) - dbpb%(fisher=SW) ]

evaluated on the swh_Latn column (PRIMARY) and on eng_Latn (SECONDARY,
descriptive only). Positive = the mismatched (EN) calibration language is
worse for Swahili.

The dbpb% definition is NOT reimplemented: `incr()` is imported from
phase1_2x2_gate_lang.py, i.e. 100 * (bpb / baseline_bpb - 1), the same
convention that produced every other 2x2 number in this project.

NO NOISE FLOOR IS AVAILABLE (the placebo Fisher artefacts were deleted on
2026-08-02, see logs/item09_placebo_0802.log STEP 3), so -- per the
registration -- this script reports instead:
  (a) sign consistency of both effects across the 3 seeds,
  (b) the ratio scale-effect / fisher-effect,
  (c) each effect against the seed spread of the cells that produced it.
It does not substitute any other floor. The headline Swahili floor of
0.91%p is NOT used here; if quoted at all it must carry the label 'noise
floor from a DIFFERENT level of analysis (headline single-condition
comparison), not a floor for this 2x2'.

Verdict vocabulary (fixed in the registration):
  SCALE_DOMINANT  = scale effect > 3x fisher effect AND both effects
                    sign-consistent across all 3 seeds
  MIXED           = otherwise
  FISHER_DOMINANT = fisher effect exceeds scale effect

Rule-ordering note: the three words are not mutually exclusive as written,
so they are evaluated in the only order that keeps each one's stated
condition true -- FISHER_DOMINANT (its condition is an outright comparison)
is tested first, then SCALE_DOMINANT's conjunction, then MIXED as the
residue. This ordering adds no criterion that the registration did not fix.

Usage:
    python phase1_2x2_ensw_effects.py
"""
import json
from pathlib import Path

from phase1_2x2_gate_lang import incr, load_baseline_bpb

SEEDS = [0, 1, 2]
EN, SW = "english_only", "swahili_only"
CONDS = [EN, SW]
PRIMARY, SECONDARY = "swh_Latn", "eng_Latn"
OUT_PATH = Path(__file__).resolve().parents[2] / "results" / "source_data" / "table_b6_2x2_ensw_effects.json"


def cells_for(lang):
    """{(fisher, scale, seed): dbpb%} for the 2x2 x 3 seeds on `lang`."""
    baseline_bpb = load_baseline_bpb(lang)
    return {(f, s, seed): incr(f, s, seed, lang, baseline_bpb)
            for f in CONDS for s in CONDS for seed in SEEDS}


def effects_for(lang):
    cells = cells_for(lang)

    per_seed_scale, per_seed_fisher = [], []
    for seed in SEEDS:
        # scale effect: hold fisher, swap the scale language, average over fisher
        per_seed_scale.append(
            sum(cells[(f, EN, seed)] - cells[(f, SW, seed)] for f in CONDS) / len(CONDS))
        # fisher effect: hold scale, swap the fisher language, average over scale
        per_seed_fisher.append(
            sum(cells[(EN, s, seed)] - cells[(SW, s, seed)] for s in CONDS) / len(CONDS))

    scale_effect = sum(per_seed_scale) / len(per_seed_scale)
    fisher_effect = sum(per_seed_fisher) / len(per_seed_fisher)

    scale_consistent = all(g > 0 for g in per_seed_scale) or all(g < 0 for g in per_seed_scale)
    fisher_consistent = all(g > 0 for g in per_seed_fisher) or all(g < 0 for g in per_seed_fisher)

    # (c) seed spread of the cells that produced each effect: max-min over the
    # 3 seeds, per 2x2 cell, so the effect can be read against the noise the
    # cells themselves carry.
    cell_spread = {f"fisher={f},scale={s}":
                   max(cells[(f, s, seed)] for seed in SEEDS) - min(cells[(f, s, seed)] for seed in SEEDS)
                   for f in CONDS for s in CONDS}

    ratio = (abs(scale_effect) / abs(fisher_effect)) if fisher_effect != 0 else float("inf")

    if abs(fisher_effect) > abs(scale_effect):
        verdict = "FISHER_DOMINANT"
    elif abs(scale_effect) > 3 * abs(fisher_effect) and scale_consistent and fisher_consistent:
        verdict = "SCALE_DOMINANT"
    else:
        verdict = "MIXED"

    return {
        "lang": lang,
        "cells_dbpb_pct": {f"fisher={f},scale={s},seed{seed}": cells[(f, s, seed)]
                           for f in CONDS for s in CONDS for seed in SEEDS},
        "per_seed_scale_effect_pp": per_seed_scale,
        "per_seed_fisher_effect_pp": per_seed_fisher,
        "scale_effect_pp": scale_effect,
        "fisher_effect_pp": fisher_effect,
        "scale_effect_sign_consistent": scale_consistent,
        "fisher_effect_sign_consistent": fisher_consistent,
        "ratio_scale_over_fisher": ratio,
        "per_cell_seed_spread_pp": cell_spread,
        "verdict": verdict,
    }


def report(e, tag):
    print(f"\n=== {tag}: {e['lang']} column ===\n")
    print("2x2 cells, dbpb% vs uncompressed baseline (mean over 3 seeds in brackets):")
    for f in CONDS:
        for s in CONDS:
            vals = [e["cells_dbpb_pct"][f"fisher={f},scale={s},seed{seed}"] for seed in SEEDS]
            print(f"  fisher={f:<14} scale={s:<14} "
                  + "  ".join(f"s{seed}={v:6.3f}%" for seed, v in zip(SEEDS, vals))
                  + f"   [mean {sum(vals)/len(vals):6.3f}%  spread {max(vals)-min(vals):.3f}%p]")

    print(f"\nper-seed scale-language effect : {[f'{g:+.3f}' for g in e['per_seed_scale_effect_pp']]} %p"
          f"   sign-consistent={e['scale_effect_sign_consistent']}")
    print(f"per-seed fisher-language effect: {[f'{g:+.3f}' for g in e['per_seed_fisher_effect_pp']]} %p"
          f"   sign-consistent={e['fisher_effect_sign_consistent']}")
    print(f"\nscale-language effect  = {e['scale_effect_pp']:+.3f} %p")
    print(f"fisher-language effect = {e['fisher_effect_pp']:+.3f} %p")
    print(f"ratio |scale| / |fisher| = {e['ratio_scale_over_fisher']:.2f}x  (SCALE_DOMINANT needs > 3x)")
    print(f"\nVERDICT ({e['lang']}): {e['verdict']}")


def main():
    primary = effects_for(PRIMARY)
    secondary = effects_for(SECONDARY)

    report(primary, "PRIMARY")
    report(secondary, "SECONDARY (descriptive only, not a verdict basis)")

    print("\nNO NOISE FLOOR: the placebo Fisher artefacts this pair would need were "
          "deleted 2026-08-02; stated as a limitation, not substituted for.")

    out = {
        "metric": "dbpb_pct = 100 * (bpb / baseline_bpb - 1); effects in %p",
        "registration": "logs/stage2x2_sw_0802.log MINI-PREREGISTRATION, fixed 2026-08-02T06:03:27Z",
        "design": "Fisher x Scale 2x2 on EN/SW, whitening ON, delta_ratio=0.8, merge=fisher, "
                  "FLORES devtest 60 sentences x 5 langs, batch=8, seeds 0/1/2",
        "noise_floor": None,
        "noise_floor_note": "placebo Fisher artefacts deleted 2026-08-02 (logs/item09_placebo_0802.log "
                            "STEP 3); no equivalent floor is computable for this pair. The headline "
                            "Swahili 0.91%p floor is from a DIFFERENT level of analysis and is not a "
                            "floor for this 2x2.",
        "primary_swh_Latn": primary,
        "secondary_eng_Latn": secondary,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
