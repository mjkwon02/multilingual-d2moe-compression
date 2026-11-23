"""Belebele clean re-evaluation (analysis/clean_eval_0916/belebele) turned into
paper tables, the Belebele counterpart of 20260916_table2_accuracy.

CPU only. Reads stored JSON/JSONL; starts no evaluation. Rerunnable.

Panels
  A. clean     : per-language clean subsets as measured (EN/ZH/SW 99, KO 113,
                 BN 125 questions), against the clean baseline of the same
                 campaign.
  B. full      : the same cells on the evaluated full split (n = 200).
  C. retention : clean accuracy divided by the clean baseline of the same
                 language -- same subset in numerator and denominator.
  D. raw       : per-seed accuracy, both subsets, plus the stored-vs-rerun
                 reproduction difference of every cell.

Binding reporting rules (analysis/clean_eval_0916/REPORTING_RULES.md)
  * full vs clean are different item sets: report the difference, never call it
    an exposure effect (rule 1).
  * the reproduction difference is its own line item and its cause is not
    established; never put it on one scale with the full->clean difference
    (rule 2).
  * Belebele clean n stays per language; across languages the question sets
    differ, which is a limitation, not something to re-evaluate away (rule 4).
  * the project collapse rule has two halves (top-label share > 0.9, or
    accuracy < 0.25). The per-item Belebele records store correctness only --
    no predicted option -- so the top-label half is NOT computable here and no
    cell is declared collapsed. Accuracy at or below the 4-way uniform-chance
    rate of 25% is flagged descriptively instead, with no test of equality to
    chance.

Usage: python _make_belebele_clean.py
"""
import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BEL = Path("results/downstream/exposure_filtered/deepseek_clean_eval/belebele")

LANGS = [("EN", "eng_Latn"), ("ZH", "zho_Hans"), ("KO", "kor_Hang"),
         ("SW", "swh_Latn"), ("BN", "ben_Beng")]
# (condition, file suffix, row label); the language set of each row is whatever
# the cell actually evaluated -- Table 28 never reported a Belebele cell for
# swahili_only or bengali_only, and those were not run.
ROWS = [
    ("english_only", "", "English calibration"),
    ("chinese_only", "", "Chinese calibration"),
    ("korean_only", "", "Korean calibration"),
    ("korean_only", "_pp0.2", "Korean calibration, pruning 0.2"),
    ("mixed_5lang", "", "Balanced-character calibration"),
]
SEEDS = [0, 1, 2]
CHANCE = 0.25  # 4 options, uniform


def load(name):
    p = BEL / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def cell_paths(cond, suf):
    return [(s, BEL / f"{cond}_seed{s}{suf}.json") for s in SEEDS]


baseline = load("baseline")
if baseline is None:
    raise SystemExit("baseline.json missing -- nothing to build")
base_res = baseline["results"]

# ---------------------------------------------------------------- gather ----
# cells[(row_label, lang_abbr)] = {"clean": [...], "full": [...], "repro": [...]}
cells, present_seeds, sources = {}, {}, []
sources.append(str(BEL / "baseline.json"))
for cond, suf, label in ROWS:
    for s, p in cell_paths(cond, suf):
        if not p.exists():
            continue
        sources.append(str(p))
        res = json.loads(p.read_text())["results"]
        present_seeds.setdefault(label, []).append(s)
        for ab, code in LANGS:
            if code not in res:
                continue
            d = cells.setdefault((label, ab), {"clean": [], "full": [], "repro": []})
            d["clean"].append((s, res[code]["clean"]["acc"]))
            d["full"].append((s, res[code]["full"]["acc"]))
            rc = res[code]["reproduction_check"]
            d["repro"].append((s, rc["stored_acc"], rc["rerun_full_acc"], rc["abs_diff"]))


def agg(vals):
    """vals: list of (seed, acc). Returns (mean, std, n) with std over 3 seeds."""
    xs = [v for _, v in vals]
    if not xs:
        return None
    m = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else None
    return m, sd, len(xs)


def fmt_pct(a, flag_chance=True):
    if a is None:
        return "--"
    m, sd, n = a
    if n < len(SEEDS):
        return f"partial {n}/3"
    txt = f"{100 * m:.1f} $\\pm$ {100 * sd:.1f}" if sd is not None else f"{100 * m:.1f}"
    if flag_chance and m <= CHANCE:
        txt += " [c]"
    return txt


def fmt_ret(a):
    if a is None:
        return "--"
    m, sd, n = a
    if n < len(SEEDS):
        return f"partial {n}/3"
    return f"{m:.3f} $\\pm$ {sd:.3f}" if sd is not None else f"{m:.3f}"


# --------------------------------------------------------------- render -----
def table(kind):
    """kind in {clean, full, retention}. Returns (header_cells, body_rows)."""
    hdr = ["Calibration"] + [ab for ab, _ in LANGS]
    body = []
    if kind in ("clean", "full"):
        row = ["Uncompressed"]
        for ab, code in LANGS:
            b = base_res.get(code)
            row.append("--" if b is None else f"{100 * b[kind]['acc']:.1f}"
                       + (" [c]" if b[kind]["acc"] <= CHANCE else ""))
        body.append(row)
    for _, _, label in ROWS:
        row = [label]
        for ab, code in LANGS:
            d = cells.get((label, ab))
            if d is None:
                row.append("--")
                continue
            if kind == "retention":
                b = base_res[code]["clean"]["acc"]
                row.append(fmt_ret(agg([(s, v / b) for s, v in d["clean"]])))
            else:
                row.append(fmt_pct(agg(d[kind])))
        body.append(row)
    return hdr, body


def md(hdr, body):
    out = ["| " + " | ".join(hdr) + " |",
           "|:---|" + ":---:|" * (len(hdr) - 1)]
    for r in body:
        out.append("| " + " | ".join(c.replace("$\\pm$", "±") for c in r) + " |")
    return "\n".join(out)


def tex(hdr, body, caption, label):
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\begin{tabular}{l" + "c" * (len(hdr) - 1) + "}", r"\toprule",
           " & ".join(hdr) + r" \\", r"\midrule"]
    for r in body:
        out.append(" & ".join(c.replace("--", "--") for c in r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{" + caption + "}", r"\label{" + label + "}", r"\end{table}"]
    return "\n".join(out)


n_note = ", ".join(f"{ab} {base_res[code]['clean']['n']}"
                   for ab, code in LANGS if code in base_res)
pass_note = ", ".join(f"{ab} {base_res[code]['clean']['n_passages']}"
                      for ab, code in LANGS if code in base_res)

CAP_CLEAN = (
    "Belebele multiple-choice reading-comprehension accuracy (\\%) after compression, clean subset, "
    "5-shot. The clean subset removes calibration-exposed passages at passage granularity from the "
    f"evaluated 200-question split, leaving {n_note} questions ({pass_note} passages respectively). "
    "The question set is identical across all models within a language, verified from the stored "
    "per-item records; across languages the sets differ, because the Belebele split order is "
    "language-specific and every reported cell used a 200-question limit -- cross-language "
    "comparisons carry that caveat. Compressed cells are mean $\\pm$ sample standard deviation "
    "(ddof = 1) over 3 calibration seeds; the Uncompressed row is a single evaluation. [c] marks a "
    "value at or below the 4-way uniform-chance rate of 25\\%; no test of equality to chance was "
    "performed, so a cell without [c] is not thereby shown to differ from chance. The project's "
    "collapse criterion is not applied: its top-label-share half needs the predicted option, which "
    "the stored Belebele records do not carry. Dashes are cells the submitted paper never reported "
    "and that were not run (swahili\\_only and bengali\\_only have no Belebele cell; the three "
    "English/Chinese/Balanced conditions reported EN, KO and ZH only)."
)
CAP_FULL = (
    "Belebele accuracy (\\%) on the evaluated full split (n = 200 per language), the same runs as the "
    "clean table. Clean and full accuracies are measured on different item sets, so their difference "
    "is descriptive and is not evidence of an exposure effect. Notation as in the clean table."
)
CAP_RET = (
    "Belebele retention (ratio, $\\times$; not a percentage): compressed accuracy divided by the "
    "uncompressed accuracy of the same language on the same clean subset. Mean $\\pm$ sample standard "
    "deviation (ddof = 1) of per-seed retention over 3 seeds. Where the uncompressed accuracy is "
    "itself low (KO 34.5\\%, BN 29.6\\%), a retention near 1 does not indicate preserved task ability."
)

files = {}
for kind, title, cap, lab in [
    ("clean", f"Belebele accuracy (%), clean subset (per-language n: {n_note})",
     CAP_CLEAN, "tab:belebele-clean"),
    ("full", "Belebele accuracy (%), evaluated full split, n=200", CAP_FULL, "tab:belebele-full"),
    ("retention", "Belebele retention (x) on the clean subset", CAP_RET, "tab:belebele-retention"),
]:
    hdr, body = table(kind)
    stem = f"table_belebele_{kind}"
    files[stem + ".md"] = f"# {title}\n\n{md(hdr, body)}\n\n**Caption.** " + \
        cap.replace("\\%", "%").replace("\\pm", "±").replace("$", "").replace("\\times", "×") \
           .replace("\\_", "_") + "\n"
    files[stem + ".tex"] = tex(hdr, body, cap, lab) + "\n"

# ------------------------------------------------------------------ raw -----
raw_rows = []
for _, _, label in ROWS:
    for ab, code in LANGS:
        d = cells.get((label, ab))
        if d is None:
            continue
        clean = dict((s, v) for s, v in d["clean"])
        full = dict((s, v) for s, v in d["full"])
        repro = {s: (st, rr, ad) for s, st, rr, ad in d["repro"]}
        for s in sorted(clean):
            st, rr, ad = repro[s]
            raw_rows.append({
                "condition": label, "seed": s, "lang": ab,
                "clean_n": base_res[code]["clean"]["n"],
                "clean_acc": f"{clean[s]:.6f}",
                "full_n": 200, "full_acc": f"{full[s]:.6f}",
                "clean_retention": f"{clean[s] / base_res[code]['clean']['acc']:.6f}",
                "at_or_below_chance_clean": int(clean[s] <= CHANCE),
                "repro_stored_full_acc": f"{st:.6f}",
                "repro_rerun_full_acc": f"{rr:.6f}",
                "repro_abs_diff": f"{ad:.6f}",
            })
for ab, code in LANGS:
    if code not in base_res:
        continue
    b = base_res[code]
    raw_rows.append({
        "condition": "Uncompressed", "seed": "", "lang": ab,
        "clean_n": b["clean"]["n"], "clean_acc": f"{b['clean']['acc']:.6f}",
        "full_n": b["full"]["n"], "full_acc": f"{b['full']['acc']:.6f}",
        "clean_retention": "1.000000",
        "at_or_below_chance_clean": int(b["clean"]["acc"] <= CHANCE),
        "repro_stored_full_acc": f"{b['reproduction_check']['stored_acc']:.6f}",
        "repro_rerun_full_acc": f"{b['reproduction_check']['rerun_full_acc']:.6f}",
        "repro_abs_diff": f"{b['reproduction_check']['abs_diff']:.6f}",
    })

with open(HERE / "belebele_raw_per_seed.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
    w.writeheader()
    w.writerows(raw_rows)

for name, text in files.items():
    (HERE / name).write_text(text)

# --------------------------------------------------------- repro summary ----
diffs = [float(r["repro_abs_diff"]) for r in raw_rows]
n_exact = sum(1 for d in diffs if d == 0.0)
repro = [
    "# Reproduction differences (own line item, rule 2)",
    "",
    "Every cell re-ran the stored evaluation on the same full 200-question split. The",
    "table below is the stored-vs-rerun difference in accuracy on that split. It is",
    "reported on its own and is never compared with the full->clean difference: the",
    "two are different comparisons (same items/different runs, versus different item",
    "sets/same run). **The cause of these differences is not established.** The",
    "original runs are logged with differing GPU counts, which is circumstantial only;",
    "no experiment was run to confirm any mechanism, and none is requested.",
    "",
    f"- cells compared: {len(diffs)}",
    f"- exactly reproduced: {n_exact} / {len(diffs)}",
    f"- max |difference|: {max(diffs):.4f} accuracy ({max(diffs) * 200:.0f} of 200 questions)",
    f"- mean |difference|: {statistics.mean(diffs):.4f}",
    "",
    "| condition | seed | lang | stored | rerun | abs diff |",
    "|:---|:---:|:---:|:---:|:---:|:---:|",
]
for r in raw_rows:
    if float(r["repro_abs_diff"]) > 0:
        repro.append(f"| {r['condition']} | {r['seed']} | {r['lang']} | "
                     f"{float(r['repro_stored_full_acc']):.3f} | "
                     f"{float(r['repro_rerun_full_acc']):.3f} | "
                     f"{float(r['repro_abs_diff']):.3f} |")
(HERE / "reproduction_differences.md").write_text("\n".join(repro) + "\n")

(HERE / "sources_used.txt").write_text(
    "# Belebele clean re-evaluation (analysis/clean_eval_0916/belebele)\n"
    + "\n".join(sorted(set(sources)))
    + "\n\n# per-item records used for the identical-item-set check\n"
    + "(verified by scripts/clean_eval_verify.py: 16 models, identical item set per language)\n"
)

done = {label: sorted(v) for label, v in present_seeds.items()}
print("[belebele] rows built:")
for _, _, label in ROWS:
    langs_here = [ab for ab, _ in LANGS if (label, ab) in cells]
    print(f"  {label:36s} seeds={done.get(label, [])} langs={langs_here}")
print(f"[belebele] wrote {len(files) + 3} files to {HERE}")
