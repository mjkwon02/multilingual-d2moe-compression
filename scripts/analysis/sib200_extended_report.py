"""SIB-200 extended campaign report (§4.2 main-evidence grid).

Reads the campaign's own run JSONs from
    <RESULTS_ROOT>/sib200_extended/<condition>_seed<N>.json
plus the pre-existing, already-valid runs still living at their canonical
paths (baseline, korean_only x3, swahili_only x3 from the 2-GPU rerun), and
emits SIB_EXTENDED_REPORT.md with:

  1. Grid  -- condition (row) x eval language (column) RETENTION
             (= mean acc over seeds / baseline acc for that language),
             +- sd over seeds. Every cell also carries its top-label
             fraction; a cell is flagged [COLLAPSED] instead of reporting a
             retention number when the model degenerated onto one label:
             top-label fraction > 0.9 OR acc < 0.25 (the majority-class
             floor -- SIB-200's 7 categories are imbalanced, so the floor is
             ~25%, not 1/7; see phase1_sib200_eval.py's docstring).
  2. Column verdicts -- within each eval-language column, does the diagonal
             (own-language) condition beat each cross condition? Reported
             per seed as a sign, with 3-seed sign consistency. korean/
             swahili columns are recomputed here in the same format so the
             old and new runs read off one table.
  3. Baseline reproduction check against the recorded reference values.

Usage:  python sib200_extended_report.py [--out <path>]
"""
import argparse
import json
import statistics
from pathlib import Path

RESULTS_ROOT = Path("results/deepseek")
EXT_DIR = RESULTS_ROOT / "sib200_extended"

LANGS = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]
LANG_SHORT = {"eng_Latn": "EN", "kor_Hang": "KO", "zho_Hans": "ZH", "swh_Latn": "SW", "ben_Beng": "BN"}
# eval language -> the condition calibrated on that same language ("diagonal")
DIAGONAL = {"eng_Latn": "english_only", "kor_Hang": "korean_only", "zho_Hans": "chinese_only",
            "swh_Latn": "swahili_only", "ben_Beng": "bengali_only"}
CONDITIONS = ["mixed_5lang", "english_only", "korean_only", "chinese_only", "swahili_only", "bengali_only"]
SEEDS = [0, 1, 2]

# baseline values recorded before this campaign (uncompressed model, 14-shot,
# n=204/lang) -- item 3 of the instruction: assert nothing drifted.
BASELINE_REFERENCE = {"eng_Latn": 0.784, "kor_Hang": 0.765, "zho_Hans": 0.804,
                      "swh_Latn": 0.422, "ben_Beng": 0.559}

COLLAPSE_TOP_FRAC = 0.9
CHANCE_FLOOR = 0.25  # majority-class floor, not 1/7


def canonical_path(cond, seed):
    return RESULTS_ROOT / cond / f"seed{seed}" / f"scale_{cond}_seed{seed}" / "eval_sib200.json"


def load_run(cond, seed):
    """Campaign copy first, then the canonical path (korean/swahili live only
    there). Returns {lang: {acc, top_frac, n}} or None if the run is absent."""
    for p in (EXT_DIR / f"{cond}_seed{seed}.json", canonical_path(cond, seed)):
        if p.exists():
            r = json.loads(p.read_text())["results"]
            return {lang: {"acc": r[lang]["acc"],
                           "top_frac": max(r[lang]["pred_distribution"].values()) / r[lang]["n_samples"],
                           "n": r[lang]["n_samples"]}
                    for lang in LANGS}, p
    return None, None


def load_baseline():
    p = RESULTS_ROOT / "baseline" / "eval_sib200.json"
    r = json.loads(p.read_text())["results"]
    return {lang: r[lang]["acc"] for lang in LANGS}


def is_collapsed(cell):
    return cell["top_frac"] > COLLAPSE_TOP_FRAC or cell["acc"] < CHANCE_FLOOR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(EXT_DIR / "SIB_EXTENDED_REPORT.md"))
    args = ap.parse_args()

    base = load_baseline()
    runs = {}      # (cond, seed) -> {lang: cell}
    sources = {}
    for cond in CONDITIONS:
        for seed in SEEDS:
            cells, p = load_run(cond, seed)
            if cells is not None:
                runs[(cond, seed)] = cells
                sources[(cond, seed)] = p

    L = []
    L.append("# SIB-200 extended grid -- §4.2 main evidence\n")
    L.append("Protocol: 14-shot fixed English demonstrations, n=204/language, all 5 calibration "
             "languages evaluated, natural-sentence continuation scored over 7 single-token category "
             "words (`phase1_sib200_eval.py`, protocol `sib200_natural_continuation_v1`). Every run "
             "2-GPU / fully GPU-resident -- no disk offload, per-item NaN guard armed.\n")
    L.append(f"Retention = (mean acc over available seeds) / (baseline acc for that language). "
             f"A cell reads **[COLLAPSED]** instead of a retention number when the predictions "
             f"degenerated onto a single label: top-label fraction > {COLLAPSE_TOP_FRAC} or "
             f"acc < {CHANCE_FLOOR} (SIB-200's majority-class floor, categories are imbalanced).\n")

    # ---- missing runs -------------------------------------------------
    missing = [f"{c}/seed{s}" for c in CONDITIONS for s in SEEDS if (c, s) not in runs]
    if missing:
        L.append(f"> **Incomplete grid** -- missing runs: {', '.join(missing)}. "
                 f"Cells below are computed over whatever seeds exist; their seed count is shown.\n")

    # ---- 1. grid ------------------------------------------------------
    L.append("## 1. Grid -- retention by condition x eval language\n")
    L.append("| condition | " + " | ".join(LANG_SHORT[l] for l in LANGS) + " |")
    L.append("|---|" + "---|" * len(LANGS))
    L.append("| *baseline acc* | " + " | ".join(f"{base[l]:.3f}" for l in LANGS) + " |")
    for cond in CONDITIONS:
        row = [cond]
        for lang in LANGS:
            seeds_present = [s for s in SEEDS if (cond, s) in runs]
            if not seeds_present:
                row.append("--")
                continue
            cells = [runs[(cond, s)][lang] for s in seeds_present]
            accs = [c["acc"] for c in cells]
            tops = [c["top_frac"] for c in cells]
            n_col = sum(is_collapsed(c) for c in cells)
            mean_top = statistics.mean(tops)
            if n_col == len(cells):
                cell = f"**[COLLAPSED]** top={mean_top:.2f}"
            else:
                ret = [a / base[lang] for a in accs]
                m = statistics.mean(ret)
                sd = statistics.stdev(ret) if len(ret) > 1 else 0.0
                cell = f"{m:.3f} ± {sd:.3f}"
                if lang == "ben_Beng" or n_col:
                    cell += f" (top={mean_top:.2f}"
                    if n_col:
                        cell += f", {n_col}/{len(cells)} seed collapsed"
                    cell += ")"
            if len(seeds_present) < len(SEEDS):
                cell += f" [n={len(seeds_present)} seeds]"
            row.append(cell)
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("Per-seed raw accuracies (top-label fraction in parentheses):\n")
    L.append("| condition/seed | " + " | ".join(LANG_SHORT[l] for l in LANGS) + " |")
    L.append("|---|" + "---|" * len(LANGS))
    for cond in CONDITIONS:
        for s in SEEDS:
            if (cond, s) not in runs:
                continue
            cells = runs[(cond, s)]
            L.append(f"| {cond}/seed{s} | " + " | ".join(
                f"{cells[l]['acc']:.3f} ({cells[l]['top_frac']:.2f})" + (" **C**" if is_collapsed(cells[l]) else "")
                for l in LANGS) + " |")
    L.append("\n(**C** = collapsed cell by the rule above.)\n")

    # ---- 2. column verdicts ------------------------------------------
    L.append("## 2. Column verdicts -- does the diagonal (own-language) condition win its column?\n")
    L.append("Within one eval-language column, the own-language-calibrated condition is compared to every "
             "other condition **seed by seed** (`+` = diagonal higher, `-` = diagonal lower). "
             "`3/3 +` means the ordering held under every seed.\n")
    L.append("| eval lang | diagonal | vs. condition | seed0 | seed1 | seed2 | sign consistency | mean acc (diag vs cross) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for lang in LANGS:
        diag = DIAGONAL[lang]
        diag_seeds = [s for s in SEEDS if (diag, s) in runs]
        if not diag_seeds:
            L.append(f"| {LANG_SHORT[lang]} | {diag} | -- | | | | *diagonal run missing* | |")
            continue
        for cond in CONDITIONS:
            if cond == diag:
                continue
            signs, dvals, cvals = [], [], []
            cellstr = []
            for s in SEEDS:
                if (diag, s) not in runs or (cond, s) not in runs:
                    cellstr.append("--")
                    continue
                d = runs[(diag, s)][lang]["acc"]
                c = runs[(cond, s)][lang]["acc"]
                dvals.append(d)
                cvals.append(c)
                sign = "+" if d > c else "-"
                signs.append(sign)
                note = ""
                if is_collapsed(runs[(diag, s)][lang]) or is_collapsed(runs[(cond, s)][lang]):
                    note = "ᶜ"
                cellstr.append(f"{sign}{note} ({d:.3f} vs {c:.3f})")
            if not signs:
                continue
            npos = signs.count("+")
            consistency = f"{npos}/{len(signs)} +" if npos >= len(signs) - npos else f"{len(signs)-npos}/{len(signs)} -"
            if npos == len(signs):
                consistency = f"**{npos}/{len(signs)} +**"
            elif npos == 0:
                consistency = f"**0/{len(signs)} +** (reversed)"
            means = f"{statistics.mean(dvals):.3f} vs {statistics.mean(cvals):.3f}"
            L.append(f"| {LANG_SHORT[lang]} | {diag} | {cond} | " + " | ".join(cellstr) + f" | {consistency} | {means} |")
    L.append("\n(ᶜ = at least one side of that comparison is a collapsed cell -- the sign is not "
             "interpretable as a calibration effect there.)\n")

    # ---- 3. baseline reproduction ------------------------------------
    L.append("## 3. Baseline reproduction check\n")
    L.append("| lang | recorded reference | current baseline json | delta |")
    L.append("|---|---|---|---|")
    drift = []
    for lang in LANGS:
        ref = BASELINE_REFERENCE[lang]
        cur = base[lang]
        d = cur - ref
        if abs(d) > 0.001:
            drift.append(lang)
        L.append(f"| {LANG_SHORT[lang]} | {ref:.3f} | {cur:.3f} | {d:+.4f} |")
    L.append("")
    if drift:
        L.append(f"> **Baseline drifted** for {', '.join(drift)} -- the baseline was not re-measured by this "
                 f"campaign, so any difference means the file on disk changed under us. Investigate before "
                 f"trusting the retention numbers above.\n")
    else:
        L.append("Baseline unchanged (this campaign never re-ran it; the file is the 2-GPU rerun's).\n")

    # ---- provenance ---------------------------------------------------
    L.append("## Run provenance\n")
    L.append("| run | source json |")
    L.append("|---|---|")
    for cond in CONDITIONS:
        for s in SEEDS:
            if (cond, s) in sources:
                L.append(f"| {cond}/seed{s} | `{sources[(cond, s)]}` |")
    L.append(f"| baseline | `{RESULTS_ROOT / 'baseline' / 'eval_sib200.json'}` |")
    L.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print(f"[sib-ext-report] wrote {out} ({len(runs)} condition-seed runs, {len(missing)} missing)")


if __name__ == "__main__":
    main()
