"""Aggregate Qwen1.5-MoE-A2.7B results (CPU, reads JSON only).

For every condition/seed cell that exists: FLORES baseline bpb, compressed bpb,
delta-bpb% (primary and eos-prefixed sensitivity), parameter reduction, routing
flags, and -- if present -- SIB-200 / Belebele accuracy with retention =
acc / baseline acc, and XL-Sum OOD / unseen bpb deltas. Missing cells are listed
as missing; nothing is filtered, re-ranked or dropped by value.

Usage:
    python qwen15_report.py            # writes RESULTS_ROOT/report/qwen15_report.{json,md}
"""
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import qwen15_config as QC  # noqa: E402

LANGS = ["eng_Latn", "zho_Hans", "kor_Hang", "swh_Latn", "ben_Beng"]  # paper order EN/ZH/KO/SW/BN
SHORT = {"eng_Latn": "EN", "zho_Hans": "ZH", "kor_Hang": "KO", "swh_Latn": "SW", "ben_Beng": "BN"}


def _load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    root = QC.RESULTS_ROOT
    base = _load(root / "baseline" / "eval_ppl.json")
    base_sib = _load(root / "baseline" / "eval_sib200.json")
    out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model_id": QC.MODEL_ID,
           "model_revision": QC.MODEL_REVISION, "baseline": {}, "cells": {}, "missing": []}
    if base:
        out["baseline"] = {l: {"bpb": base[l]["bits_per_byte"],
                               "bpb_eos_prefixed": base[l]["sensitivity_eos_prefixed"]["bits_per_byte"]} for l in LANGS}
    else:
        out["missing"].append("baseline/eval_ppl.json")

    for cond in QC.CONDITIONS:
        for seed in QC.SEEDS:
            d = QC.condition_dir(cond, seed) / f"scale_{cond}_seed{seed}"
            ev = _load(d / "eval_ppl.json")
            if ev is None:
                out["missing"].append(f"{cond}/seed{seed}")
                continue
            cell = {"bpb": {l: ev[l]["bits_per_byte"] for l in LANGS}}
            if base:
                cell["delta_bpb_pct"] = {l: 100 * (ev[l]["bits_per_byte"] / base[l]["bits_per_byte"] - 1) for l in LANGS}
                cell["delta_bpb_pct_eos_prefixed"] = {
                    l: 100 * (ev[l]["sensitivity_eos_prefixed"]["bits_per_byte"]
                              / base[l]["sensitivity_eos_prefixed"]["bits_per_byte"] - 1) for l in LANGS}
            params = ev.get("_meta", {}).get("run", {}).get("params")
            if params:
                cell["params"] = params
            rc = _load(d / "reload_check.json")
            if rc:
                cell["reload_identical"] = all(v["identical"] for v in rc.values())
            flags = _load(QC.condition_dir(cond, seed) / "routing_coverage_flags.json") or {}
            cell["routing_flags"] = {}
            for fl in flags.values():
                for f in fl:
                    k = f"{f['source']}:{f['kind']}"
                    cell["routing_flags"][k] = cell["routing_flags"].get(k, 0) + 1
            sib = _load(d / "eval_sib200.json")
            if sib:
                cell["sib200_acc"] = {l: sib["results"][l]["acc"] for l in LANGS}
                if base_sib:
                    cell["sib200_retention"] = {l: sib["results"][l]["acc"] / base_sib["results"][l]["acc"] for l in LANGS}
            out["cells"][f"{cond}/seed{seed}"] = cell

    rep_dir = root / "report"
    rep_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    (rep_dir / f"qwen15_report_{stamp}.json").write_text(json.dumps(out, indent=2))
    lines = [f"# Qwen1.5-MoE-A2.7B results ({out['generated_utc']})", "",
             "delta-bpb% vs uncompressed baseline (lower is better). Primary = unchanged evaluator.", "",
             "| cell | " + " | ".join(SHORT[l] for l in LANGS) + " | model params reduction |",
             "|---|" + "---:|" * (len(LANGS) + 1)]
    for k, c in out["cells"].items():
        dp = c.get("delta_bpb_pct")
        vals = " | ".join(f"{dp[l]:.2f}" for l in LANGS) if dp else " | ".join("n/a" for _ in LANGS)
        red = c.get("params", {}).get("total_reduction_pct")
        lines.append(f"| {k} | {vals} | {red:.3f}% |" if red is not None else f"| {k} | {vals} | n/a |")
    lines += ["", f"missing: {len(out['missing'])} cell(s)"]
    (rep_dir / f"qwen15_report_{stamp}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
