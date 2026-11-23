"""Qwen1.5-MoE-A2.7B main experiment (Table 1 counterpart): FLORES Δbpb% vs the
uncompressed baseline for 6 calibration conditions x the requested seeds.

CPU only, reads JSON. It is also a GATE: exit 0 only if the baseline and EVERY
requested cell exist and validate (all 8 pipeline stages, reload bit-identity);
otherwise the missing/invalid cells are listed and it exits 20. Nothing is
imputed, filtered or re-ranked.

  python qwen15_table1_aggregate.py --seeds 0 1
  python qwen15_table1_aggregate.py --seeds 0 1 2

Δbpb%(cell, lang) = 100 * (bpb_compressed / bpb_baseline - 1), primary bpb
column only (the eos-prefixed sensitivity column is not used). Mean and sample
std (ddof=1) are taken over per-seed Δbpb%.
Output: RESULTS_ROOT/report/table1_qwen_seeds<..>.{json,csv,md}; an existing
file is never overwritten (a timestamped name is used instead).
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import qwen15_followup_check as QK  # noqa: E402

LANGS = [("EN", "eng_Latn"), ("ZH", "zho_Hans"), ("KO", "kor_Hang"), ("SW", "swh_Latn"), ("BN", "ben_Beng")]
CONDITIONS = [("english_only", "English"), ("chinese_only", "Chinese"), ("korean_only", "Korean"),
              ("swahili_only", "Swahili"), ("bengali_only", "Bengali"), ("mixed_5lang", "Balanced-character")]


def target(path):
    return path if not path.exists() else path.with_name(f"{path.stem}.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}{path.suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--outdir", default=None, help="default RESULTS_ROOT/report")
    a = ap.parse_args()
    seeds = a.seeds
    problems = []
    base = QK.jload(QK.RESULTS / "baseline" / "eval_ppl.json")
    ok, why = QK.bpb_ok(base)
    if not ok:
        problems.append(f"baseline eval_ppl.json: {why}")
    rows, per = [], {}
    for c, label in CONDITIONS:
        for s in seeds:
            st = QK.stages_for_cell(c, s)
            bad = [f"{a}={b}" for a, b, _ in st if b != "done"]
            if bad:
                problems.append(f"{c}/seed{s}: " + ", ".join(bad))
                continue
            ev = QK.jload(QK.cell_dir(c, s) / f"scale_{c}_seed{s}" / "eval_ppl.json")
            if not ok:
                continue
            for ab, code in LANGS:
                d = 100.0 * (ev[code]["bits_per_byte"] / base[code]["bits_per_byte"] - 1.0)
                per.setdefault((c, ab), {})[s] = d
                rows.append({"condition": label, "condition_id": c, "seed": s, "lang": ab,
                             "bpb_baseline": base[code]["bits_per_byte"], "bpb_compressed": ev[code]["bits_per_byte"],
                             "delta_bpb_pct": round(d, 6)})
    summ = []
    for c, label in CONDITIONS:
        for ab, _ in LANGS:
            v = per.get((c, ab), {})
            vals = [v[s] for s in seeds if s in v]
            summ.append({"condition": label, "lang": ab, "n_seeds": len(vals),
                         **{f"seed{s}": (round(v[s], 4) if s in v else None) for s in seeds},
                         "mean_delta_bpb_pct": round(statistics.mean(vals), 4) if len(vals) == len(seeds) else None,
                         "std_ddof1": round(statistics.stdev(vals), 4) if len(vals) == len(seeds) and len(vals) > 1 else None})
    tag = "".join(map(str, seeds))
    outdir = Path(a.outdir) if a.outdir else QK.RESULTS / "report"
    outdir.mkdir(parents=True, exist_ok=True)
    js = target(outdir / f"table1_qwen_seeds{tag}.json")
    js.write_text(json.dumps({"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seeds": seeds,
                              "complete": not problems, "problems": problems,
                              "baseline_bpb": {ab: base[code]["bits_per_byte"] for ab, code in LANGS} if ok else None,
                              "per_seed": rows, "summary": summ}, indent=2, ensure_ascii=False))
    with open(target(outdir / f"table1_qwen_seeds{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)
    md = [f"# Qwen1.5-MoE-A2.7B — FLORES Δbpb% vs uncompressed (seeds {seeds})", "",
          "| Calibration | " + " | ".join(ab for ab, _ in LANGS) + " |", "|:---|" + ":---:|" * len(LANGS)]
    for c, label in CONDITIONS:
        cells = []
        for ab, _ in LANGS:
            r = next(x for x in summ if x["condition"] == label and x["lang"] == ab)
            cells.append("--" if r["mean_delta_bpb_pct"] is None else
                         (f"{r['mean_delta_bpb_pct']:+.2f} ± {r['std_ddof1']:.2f}" if r["std_ddof1"] is not None
                          else f"{r['mean_delta_bpb_pct']:+.2f}"))
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    md += ["", "Mean ± sample std (ddof=1) over calibration seeds; FLORES devtest first 60 sentences, batch 8, "
           "primary bpb (Qwen has no BOS; first token unscored, bytes counted)."]
    if problems:
        md += ["", "**INCOMPLETE** — " + "; ".join(problems)]
    target(outdir / f"table1_qwen_seeds{tag}.md").write_text("\n".join(md) + "\n")
    print(f"[qwen-table1] seeds={seeds} complete={not problems} -> {js}")
    for p in problems:
        print("  ", p)
    return 0 if not problems else 20


if __name__ == "__main__":
    sys.exit(main())
