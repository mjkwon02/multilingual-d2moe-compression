"""CPU-only checks for the Qwen follow-up run (qwen15_followup_run.sh).

Nothing here starts a GPU job, writes a result, or deletes anything.

  predecessor            MASSIVE revised campaign state from its CONTENT.
                         exit 0 = DONE, 10 = still running / not finished,
                         20 = FAILED (do not start Qwen)
  unit KIND COND SEED    per-stage state of one queue unit, one line per stage:
                         "<stage>\t<done|todo|invalid>\t<detail>"
                         exit 0 = all done, 10 = work left, 20 = invalid state
  next QUEUE             first queue unit that is not done:
                         "ALL_DONE" | "<kind>\t<cond>\t<seed>" (exit 0) |
                         exit 20 with "INVALID\t<unit>\t<detail>"
  plan QUEUE             every queue unit with its state (dry run)
  paths                  resolve the cache/temp directories libraries will
                         actually use in this environment; exit 20 if any is
                         outside the results tree (legacy check)
  init-queue QUEUE       write the default queue if the file does not exist

Queue units (tab separated): downstream_baseline baseline -
                             cell <condition> <seed>
                             downstream <condition> <seed>
"""
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

RESULTS = Path("results/qwen/phase1")
HDD = ""  # legacy storage-location check disabled in the public tree
REPO = Path(__file__).resolve().parents[2]
MASSIVE = REPO / "results" / "downstream" / "massive_revised"
MASSIVE_LOG = REPO / "logs" / "massive_revised_campaign.log"
MASSIVE_UNITS = ["baseline"] + [f"{c}/seed{s}" for s in (0, 1, 2) for c in ("english_only", "mixed_5lang")]
MASSIVE_MODELS = ["baseline"] + [f"{c}_seed{s}" for s in (0, 1, 2) for c in ("english_only", "mixed_5lang")]
MASSIVE_LANGS = ["EN", "ZH", "KO", "SW", "BN"]

FLORES5 = ["eng_Latn", "kor_Hang", "zho_Hans", "swh_Latn", "ben_Beng"]
LAYERS = range(24)
DOWNSTREAM_TASKS = ["sib200", "xlsum"]   # qwen15_sweep_plan.sh section 3: --tasks sib200 xlsum


# ------------------------------------------------------------------ helpers
def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def latest_meta(dirpath, stem):
    """qwen15_config.write_json_versioned writes <stem>.json, then
    <stem>.<UTC>.json when it already exists -- the newest file is current."""
    cands = [Path(p) for p in glob.glob(str(Path(dirpath) / f"{stem}.json"))
             + glob.glob(str(Path(dirpath) / f"{stem}.*.json"))]
    if not cands:
        return None
    return jload(max(cands, key=lambda p: p.stat().st_mtime))


def finite_leaves(obj):
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return True
    if isinstance(obj, (int, float)):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(finite_leaves(v) for v in obj.values())
    if isinstance(obj, list):
        return all(finite_leaves(v) for v in obj)
    return True


def bpb_ok(d, need_delta=False):
    if not d:
        return False, "missing/unreadable"
    for code in FLORES5:
        x = d.get(code, {}).get("bits_per_byte")
        if not isinstance(x, (int, float)) or not math.isfinite(x):
            return False, f"{code} bits_per_byte={x}"
    if need_delta and "_delta_vs_baseline" not in d:
        return False, "no _delta_vs_baseline"
    return True, "5 languages finite"


# --------------------------------------------------------------- predecessor
def predecessor():
    import psutil
    me = os.getpid()
    running = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.info["pid"] == me:
            continue
        # match the script as its own argv element (interpreter + script path), so a
        # shell whose -c string merely mentions the name is not mistaken for it
        argv = p.info["cmdline"] or []
        if any(os.path.basename(a) in ("massive_revised_campaign.sh", "massive_revised_run.py",
                                       "massive_revised_aggregate.py") for a in argv[:3]):
            running.append(f"{p.info['pid']}:{' '.join(argv)[:80]}")
    if running:
        print("RUNNING\t" + " | ".join(running))
        return 10

    problems = []
    text = MASSIVE_LOG.read_text(errors="replace") if MASSIVE_LOG.exists() else ""
    start = text.rfind("===== MASSIVE revised test campaign start")
    text = text[start:] if start >= 0 else ""
    end = text.rfind("===== campaign end =====")
    if end < 0:
        problems.append("driver not running and no 'campaign end' marker")
    else:
        tail = text[end:]
        if "aggregate FAILED" in tail:
            problems.append("aggregate FAILED")
        elif "aggregate written" not in tail:
            problems.append("no 'aggregate written' after campaign end")
    if "FAIL  " in text:
        problems.append("a unit logged FAIL")

    last, last_done_ts = {}, None
    st = MASSIVE / "campaign_status.tsv"
    if st.exists():
        with open(st) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                last[r["unit"]] = r["status"]
                if r["status"] == "DONE":
                    last_done_ts = r["utc"]
    for u in MASSIVE_UNITS:
        if last.get(u) != "DONE":
            problems.append(f"status {u}={last.get(u)}")

    vp = jload(MASSIVE / "validation_and_pilot.json") or {}
    sig = None
    for m in MASSIVE_MODELS:
        rt = jload(MASSIVE / "runtime" / f"{m}__test.json")
        if not rt:
            problems.append(f"runtime/{m}__test.json missing")
            continue
        sig = sig or rt.get("inference_signature")
        if rt.get("stopped_early"):
            problems.append(f"{m} stopped_early={rt['stopped_early']}")
        if rt.get("failures"):
            problems.append(f"{m} failures={len(rt['failures'])}")
        if rt.get("inference_signature") != sig:
            problems.append(f"{m} inference_signature differs")
    if vp.get("inference_signature") and sig and not vp["inference_signature"].startswith(sig[:16]):
        problems.append("runtime signature != validation_and_pilot signature")

    test_ids = set((jload(MASSIVE / "test_ids.json") or {}).get("ids", []))
    seen = {(m, l): set() for m in MASSIVE_MODELS for l in MASSIVE_LANGS}
    pred = MASSIVE / "predictions.jsonl"
    if pred.exists():
        with open(pred) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    problems.append("unparseable predictions line")
                    break
                if r.get("split") == "test" and r.get("inference_signature") == sig and (r["model"], r["lang"]) in seen:
                    seen[(r["model"], r["lang"])].add(str(r["id"]))
    else:
        problems.append("predictions.jsonl missing")
    for k, ids in seen.items():
        if len(test_ids) != 200 or ids != {str(i) for i in test_ids}:
            problems.append(f"predictions {k[0]}/{k[1]}: {len(ids)} of 200 test ids")

    for name in ("summary.json", "metrics.csv", "completion_grid.csv", "confusion_matrices.json"):
        p = MASSIVE / name
        if not p.exists():
            problems.append(f"aggregate output {name} missing")
        elif last_done_ts:
            import datetime as dt
            t_done = dt.datetime.fromisoformat(last_done_ts.replace("Z", "+00:00")).timestamp()
            if p.stat().st_mtime < t_done:
                problems.append(f"{name} older than last DONE unit")

    if problems:
        print("FAILED\t" + "; ".join(problems[:12]) + (f" (+{len(problems) - 12} more)" if len(problems) > 12 else ""))
        return 20
    print(f"DONE\t7/7 units DONE, 7x5x200 predictions, aggregate written, no MASSIVE process alive")
    return 0


# --------------------------------------------------------------- unit state
def cell_dir(c, s):
    return RESULTS / c / f"seed{s}"


def stages_for_cell(c, s):
    d = cell_dir(c, s)
    sd = d / f"scale_{c}_seed{s}"
    out = []

    # fetch: complete local snapshot per the fetch record (no Hub call needed)
    fr = jload(RESULTS / "_fetch" / "fetch_record.json")
    if fr and fr.get("snapshot"):
        snap = Path(os.path.expandvars(fr["snapshot"]))
        if not snap.is_dir():
            # metrics-only tree: the model snapshot is not kept locally; the
            # signed fetch record (revision + per-file sizes) is the evidence.
            out.append(("fetch", "done", f"record present (rev {fr.get('revision','?')[:8]}); snapshot not on this machine"))
            snap = None
    else:
        snap = None
        out.append(("fetch", "todo", "no fetch record"))
    if snap is not None:
        # same completeness rule as qwen15_fetch.py: judged by the weight files
        # (the record also lists Hub files outside allow_patterns, never downloaded)
        weights = {k: v for k, v in fr.get("files", {}).items() if k.endswith(".safetensors")}
        bad = [k for k, v in weights.items() if not (snap / k).is_file() or (snap / k).stat().st_size != v]
        bad += [k for k in ("config.json",) if not (snap / k).is_file()]
        ok = bool(weights) and not bad
        out.append(("fetch", "done" if ok else "todo",
                    f"{len(weights)} safetensors sizes match" if ok else f"incomplete: {bad[:3]}"))

    tok = jload(RESULTS / "_tokenization" / f"{c}_seed{s}.json")
    out.append(("calib_report", "done" if tok and tok.get("calib_text_hash") else "todo",
                f"hash={tok.get('calib_text_hash')}" if tok else "missing"))

    ok, why = bpb_ok(jload(RESULTS / "baseline" / "eval_ppl.json"))
    out.append(("baseline_eval", "done" if ok else "todo", why))

    freq = d / "qwen_wikitext_2000_expert_frequencies.json"
    out.append(("freq", "done" if jload(freq) is not None else "todo", "present" if freq.exists() else "missing"))

    for stage, sub, stem, hash_check in (("fisher", "fisher_layers", "fisher_run_meta", True),
                                         ("svd_scale", "svd_scale_layers", "svd_scale_run_meta", False)):
        have = sum((d / sub / f"layer_{i}.pt").is_file() for i in LAYERS)
        meta = latest_meta(d, stem)
        # Public tree: per-layer tensors are not preserved (regenerable); a run_meta
        # with status "ok" is the completeness evidence. Tensor count still checked
        # when the layers are actually on disk (full working tree).
        if meta and meta.get("status") == "ok" and (have in (0, 24)):
            if hash_check and tok and meta.get("calib_text_hash") != tok.get("calib_text_hash"):
                out.append((stage, "invalid", f"calib hash {meta.get('calib_text_hash')} != tokenization {tok.get('calib_text_hash')}"))
            else:
                out.append((stage, "done", "24/24 layers, meta ok"))
        else:
            out.append((stage, "todo", f"{have}/24 layers, meta={meta.get('status') if meta else None} (layer-resumable)"))

    ev, man = sd / "eval_ppl.json", sd / "compressed_model" / "manifest.json"
    n_st = len(glob.glob(str(sd / "compressed_model" / "layer_*.safetensors")))
    if not ev.exists() and not man.exists():
        out.append(("merge_save", "todo", "no eval_ppl.json, no manifest"))
    else:
        ok, why = bpb_ok(jload(ev), need_delta=True)
        meta = latest_meta(sd, "merge_eval_run_meta")
        if ok and man.exists() and n_st in (0, 24) and meta and meta.get("status") == "ok":
            # n_st == 0: public tree keeps manifest + eval but not the safetensors themselves.
            out.append(("merge_save", "done", f"eval finite, manifest ok, safetensors={n_st}"))
        else:
            out.append(("merge_save", "invalid", f"partial: eval_ppl={why}, manifest={man.exists()}, safetensors={n_st}, "
                                                 f"meta={meta.get('status') if meta else None}"))

    rc, rev = jload(sd / "reload_check.json"), sd / "eval_ppl_reloaded.json"
    if rc is None and not rev.exists():
        out.append(("reload_eval", "todo", "not run"))
    else:
        good = bool(rc) and all(rc.get(code, {}).get("identical") is True for code in FLORES5)
        out.append(("reload_eval", "done" if good else "invalid",
                    "5/5 bit-identical" if good else f"reload_check={'missing' if rc is None else 'not identical'}"))
    return out


def downstream_targets(c, s):
    base = RESULTS / "baseline" if c == "baseline" else cell_dir(c, s) / f"scale_{c}_seed{s}"
    return {"sib200": base / "eval_sib200.json", "xlsum": base / "eval_ood_unseen.json"}


def stages_for_downstream(c, s):
    out = []
    if c != "baseline":
        cs = dict((st, state) for st, state, _ in stages_for_cell(c, s))
        if cs.get("merge_save") != "done" or cs.get("reload_eval") != "done":
            out.append(("prerequisite_cell", "todo", "cell not complete"))
    for task, p in downstream_targets(c, s).items():
        if task not in DOWNSTREAM_TASKS:
            continue
        d = jload(p)
        if not p.exists():
            out.append((task, "todo", "missing"))
        elif task == "sib200":
            res = (d or {}).get("results", {})
            accs = [res.get(code, {}).get("acc") for code in FLORES5]
            ok = all(isinstance(a, (int, float)) and math.isfinite(a) for a in accs) and \
                all(res.get(code, {}).get("n_samples", 0) > 0 for code in FLORES5)
            out.append((task, "done" if ok else "invalid", "5 languages finite" if ok else f"accs={accs}"))
        else:
            ok = bool(d) and bool(d.get("e2_ood", {}).get("results")) and bool(d.get("e3_unseen", {}).get("results")) \
                and finite_leaves(d.get("e2_ood", {}).get("results")) and finite_leaves(d.get("e3_unseen", {}).get("results"))
            out.append((task, "done" if ok else "invalid", "e2+e3 results finite" if ok else "empty or non-finite"))
    return out


def unit_stages(kind, c, s):
    if kind == "cell":
        return stages_for_cell(c, int(s))
    if kind == "downstream":
        return stages_for_downstream(c, int(s))
    if kind == "downstream_baseline":
        return stages_for_downstream("baseline", None)
    raise SystemExit(f"unknown unit kind {kind}")


def summarize(stages):
    if any(state == "invalid" for _, state, _ in stages):
        return "invalid"
    return "done" if all(state == "done" for _, state, _ in stages) else "todo"


def read_queue(q):
    rows = []
    for line in Path(q).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise SystemExit(f"bad queue line: {line!r}")
        rows.append(tuple(parts))
    return rows


# --------------------------------------------------------------------- paths
def paths():
    import tempfile
    rows = {"tempfile.gettempdir()": tempfile.gettempdir()}
    import huggingface_hub.constants as hc
    rows["huggingface_hub HF_HOME"] = hc.HF_HOME
    rows["huggingface_hub HF_HUB_CACHE"] = hc.HF_HUB_CACHE
    import datasets.config as dc
    rows["datasets HF_DATASETS_CACHE"] = str(dc.HF_DATASETS_CACHE)
    import torch
    rows["torch.hub.get_dir()"] = torch.hub.get_dir()
    rows["TRITON_CACHE_DIR"] = os.environ.get("TRITON_CACHE_DIR", "(unset -> ~/.triton)")
    rows["TORCHINDUCTOR_CACHE_DIR"] = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "(unset -> /tmp)")
    rows["XDG_CACHE_HOME"] = os.environ.get("XDG_CACHE_HOME", "(unset -> ~/.cache)")
    try:
        import matplotlib
        rows["matplotlib.get_cachedir()"] = matplotlib.get_cachedir()
    except Exception:
        rows["matplotlib.get_cachedir()"] = f"{os.environ.get('MPLCONFIGDIR')} (matplotlib not importable)"
    sys.path[:0] = [str(Path(__file__).resolve().parent)] + [str(d) for d in Path(__file__).resolve().parents[1].iterdir() if d.is_dir()] + [str(Path(__file__).resolve().parents[2] / "src" / "multilingual_d2moe")]
    import qwen15_config as QC
    rows["qwen15_config.RESULTS_ROOT"] = str(QC.RESULTS_ROOT)
    rows["qwen15_config.HF_HOME_EXPECTED"] = QC.HF_HOME_EXPECTED
    rows["torch visible devices"] = f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} " \
                                    f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')}"
    bad = []
    for k, v in rows.items():
        if k == "torch visible devices":
            continue
        flag = "" if str(v).startswith(HDD) else "   <-- unexpected location (legacy check)"
        if flag:
            bad.append(k)
        print(f"{k:34s} {v}{flag}")
    if str(QC.RESULTS_ROOT) != str(RESULTS):
        print("RESULTS path in this checker differs from qwen15_config.RESULTS_ROOT")
        return 20
    return 20 if bad else 0


def init_queue(q):
    q = Path(q)
    if q.exists():
        print(f"{q} exists -- not overwritten")
        return 0
    main = ["english_only", "chinese_only", "korean_only", "swahili_only", "bengali_only", "mixed_5lang"]
    placebo = ["english_only_placebo", "chinese_only_placebo", "korean_only_placebo", "swahili_only_placebo", "bengali_only_placebo"]
    lines = [
        "# Qwen1.5-MoE-A2.7B follow-up queue -- order of qwen15_sweep_plan.sh.",
        "# Re-read before every unit: edits take effect at the next unit boundary.",
        "# Completed units are validated and skipped, never re-run or overwritten.",
        "# kind<TAB>condition<TAB>seed",
        "# downstream = qwen15_sweep_plan.sh section 3 (--from-saved --tasks sib200 xlsum), run after each main cell.",
        "# Section 4 (optional EN-KO 2x2) is not queued.",
        "downstream_baseline\tbaseline\t-",
    ]
    for s in (0, 1, 2):                       # section 1 (+3): main grid, seed-major as printed
        for c in main:
            lines += [f"cell\t{c}\t{s}", f"downstream\t{c}\t{s}"]
    for s in (0, 1, 2):                       # section 2: placebo arm B
        for c in placebo:
            lines.append(f"cell\t{c}\t{s}")
    q.parent.mkdir(parents=True, exist_ok=True)
    q.write_text("\n".join(lines) + "\n")
    print(f"wrote {q}")
    return 0


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd = sys.argv[1]
    if cmd == "predecessor":
        return predecessor()
    if cmd == "unit":
        st = unit_stages(*sys.argv[2:5])
        for stage, state, detail in st:
            print(f"{stage}\t{state}\t{detail}")
        return {"done": 0, "todo": 10, "invalid": 20}[summarize(st)]
    if cmd == "next":
        for kind, c, s in read_queue(sys.argv[2]):
            st = unit_stages(kind, c, s)
            state = summarize(st)
            if state == "invalid":
                detail = "; ".join(f"{a}:{d}" for a, b, d in st if b == "invalid")
                print(f"INVALID\t{kind} {c} {s}\t{detail}")
                return 20
            if state == "todo":
                print(f"{kind}\t{c}\t{s}")
                return 0
        print("ALL_DONE")
        return 0
    if cmd == "plan":
        rc = 0
        for kind, c, s in read_queue(sys.argv[2]):
            st = unit_stages(kind, c, s)
            state = summarize(st)
            rc = max(rc, 20 if state == "invalid" else 0)
            todo = [a for a, b, _ in st if b != "done"]
            print(f"{state:7s} {kind:20s} {c:16s} {s:2s}  " + ("(all stages valid)" if not todo else "todo: " + ",".join(todo)))
        return rc
    if cmd == "paths":
        return paths()
    if cmd == "init-queue":
        return init_queue(sys.argv[2])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
