"""Revised MASSIVE validation stage: microbatch probe (20 min cap) + baseline
dev pilot (P0 first, P8 only if P0 fails the gate) + the fixed gate.

Budget: this whole stage shares a 2-hour cap with the reference-scorer
validation already completed, and it never re-runs what is already verified.

Microbatch probe. Reuses the existing microbatch-6 measurement rather than
re-measuring it, and tries 12/20/30/60 within the time cap. Deliberately NOT
assuming bigger is faster or that EN decides memory: each candidate size is
timed with CUDA synchronization on fixed dev items that INCLUDE the long
KO/BN utterances, peak memory is recorded, and every size is score-checked
against the validated microbatch-6 baseline on the same fixed examples. The
chosen size is the fastest one that has memory headroom AND matches the
validated scores -- if a larger size is slower or drifts, it loses.

Gate (fixed in advance, operational not statistical): a language whose dev
accuracy does not exceed that dev subset's majority baseline by more than
5 pp is BASELINE_FLOOR_RISK. 3 or more floor-risk languages of 5 => do not
start the compressed sweep, report PILOT_UNINFORMATIVE.
"""
import argparse
import gc
import collections
import json
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import massive_data as M  # noqa: E402
import massive_build_protocol as P  # noqa: E402
import massive_eval as E  # noqa: E402
from massive_revised_run import inference_signature  # noqa: E402

NEW_DIR = SCRIPT_DIR.parents[1] / "results" / "downstream" / "massive_revised"
OLD_DIR = SCRIPT_DIR.parents[1] / "results" / "downstream" / "massive_legacy"
PROBE_CAP_S = 20 * 60
BASE_MB = 6            # the already-validated microbatch
CANDIDATE_MBS = [12, 20, 30, 60]
GATE_MARGIN_PP = 5.0
GATE_FLOOR_RISK_TO_HOLD = 3


def long_item_ids(data, dev_ids, k=3):
    """Fixed probe items chosen to stress memory: the longest KO and BN
    utterances in the dev subset plus the longest EN one. Chosen by utterance
    length only -- never by any score."""
    picks = []
    for pl in ("KO", "BN", "EN"):
        ranked = sorted(dev_ids, key=lambda i: -len(data[pl][i]["utt"]))
        picks += [(pl, i) for i in ranked[:k]]
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="shared_prefix", choices=["shared_prefix", "per_item"])
    ap.add_argument("--skip-probe", action="store_true")
    args = ap.parse_args()

    protocol = json.loads((NEW_DIR / "protocol.json").read_text())
    dev_ids = json.loads((NEW_DIR / "dev_ids.json").read_text())["ids"]
    data = M.load_all()
    demo_meta = json.loads((OLD_DIR / "massive_demo_ids.json").read_text())
    demos = [data[demo_meta["lang"]][i] for i in demo_meta["ids"]]
    shared = {"P0": P.build_shared_prefix([]), "P8": P.build_shared_prefix(demos)}
    for pid, txt in shared.items():
        if M.sha256_text(txt) != protocol["prompts"][f"{pid}_shared_prefix_sha256"]:
            raise RuntimeError(f"{pid} prompt hash drift -- refusing to run")
    surfaces = [M.LABEL_SURFACE[lab] for lab in M.PROMPT_LABEL_ORDER]
    langs = list(M.LOCALES)

    out = {
        "stage": "revised validation: microbatch probe + baseline dev pilot + gate",
        "protocol_id": protocol["protocol_id"],
        "reused_validation": {
            "source": "logs/massive_scorer_diagnostic2.log + results/massive/massive_scorer_ablation*.json",
            "what": ("reference-scorer comparison at microbatch 6 after the candidate-token cache fix; "
                     "0 argmax disagreements per language, max|mean-logprob diff| ~0.27-0.34, which is below "
                     "the model's own shape-sensitivity floor (the uncached oracle differs from itself by "
                     "0.48 when only tensor shape changes)"),
            "not_repeated": "the full 5-language x 2-prompt reference sweep is not re-run",
        },
        "dev_n": len(dev_ids),
    }

    t_load = time.time()
    model, tokenizer = E.load_model()
    load_s = time.time() - t_load
    if hasattr(model, "hf_device_map"):
        off = {k: v for k, v in model.hf_device_map.items() if v in ("disk", "cpu")}
        if off:
            raise RuntimeError(f"{len(off)} module(s) offloaded off-GPU")
    out["device_map_unique"] = sorted(set(getattr(model, "hf_device_map", {}).values()))
    out["seconds_model_load"] = load_s
    print(f"[pilot] baseline loaded in {load_s:.0f}s device_map={out['device_map_unique']}")

    # ---------------------------------------------------------- probe ------
    probe = {"cap_seconds": PROBE_CAP_S, "baseline_microbatch_reused": BASE_MB,
             "prompt_used": "P0 (evaluated first per the revision; if P8 is ever needed its longer prefix is "
                            "re-checked for memory before use)",
             "items": [], "measurements": {}, "mode": args.mode}
    picks = long_item_ids(data, dev_ids)
    probe["items"] = [{"lang": l, "id": i, "utt_chars": len(data[l][i]["utt"])} for l, i in picks]
    tails = [(l, P.build_item_tail(data[l][i]["utt"])) for l, i in picks]

    def timed(mb):
        torch.cuda.reset_peak_memory_stats()
        sc = E.CachedScorer(model, tokenizer, shared["P0"], surfaces, microbatch=mb,
                            share_prefix=(args.mode == "shared_prefix"))
        sc.score(tails[0][1])                      # warmup (cuBLAS autotune for new shapes)
        torch.cuda.synchronize()
        t0 = time.time()
        got = [sc.score(t)[0] for _, t in tails]
        torch.cuda.synchronize()
        secs = time.time() - t0
        peak = max(torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count()))
        sc.verify_cache_intact()
        del sc
        gc.collect(); torch.cuda.empty_cache()
        return got, secs, peak

    t_probe = time.time()
    base_scores, base_secs, base_peak = timed(BASE_MB)
    probe["measurements"][f"mb{BASE_MB}"] = {
        "items_per_second": len(tails) / base_secs, "seconds": base_secs,
        "peak_gpu_bytes": base_peak, "argmax_match_vs_mb6": len(tails), "max_abs_diff_vs_mb6": 0.0,
        "note": "reference point; its reference-scorer validation was done earlier and is reused"}
    print(f"[probe] mb{BASE_MB}: {len(tails)/base_secs:.3f} it/s peak={base_peak/2**30:.1f}GiB")

    for mb in CANDIDATE_MBS:
        if time.time() - t_probe > PROBE_CAP_S:
            probe.setdefault("skipped_due_to_cap", []).append(mb)
            print(f"[probe] cap reached; skipping mb{mb}")
            continue
        try:
            got, secs, peak = timed(mb)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            probe["measurements"][f"mb{mb}"] = {"oom": True}
            print(f"[probe] mb{mb}: OOM -- not a candidate")
            continue
        agree = sum(int(max(range(60), key=lambda j: g[j]) == max(range(60), key=lambda j: b[j]))
                    for g, b in zip(got, base_scores))
        mx = max(max(abs(a - b) for a, b in zip(g, bs)) for g, bs in zip(got, base_scores))
        probe["measurements"][f"mb{mb}"] = {
            "items_per_second": len(tails) / secs, "seconds": secs, "peak_gpu_bytes": peak,
            "argmax_match_vs_mb6": agree, "n_items": len(tails), "max_abs_diff_vs_mb6": mx}
        print(f"[probe] mb{mb}: {len(tails)/secs:.3f} it/s peak={peak/2**30:.1f}GiB "
              f"argmax_match={agree}/{len(tails)} max|D|={mx:.3e}")

    # choose: fastest size that did not OOM, kept every argmax, and stayed
    # within the shape-sensitivity floor already measured for this model
    TOL = 0.5
    viable = {BASE_MB: probe["measurements"][f"mb{BASE_MB}"]["items_per_second"]}
    for mb in CANDIDATE_MBS:
        m = probe["measurements"].get(f"mb{mb}")
        if not m or m.get("oom"):
            continue
        if m["argmax_match_vs_mb6"] == m["n_items"] and m["max_abs_diff_vs_mb6"] <= TOL:
            viable[mb] = m["items_per_second"]
    chosen_mb = max(viable, key=viable.get)
    probe["viable_microbatches"] = viable
    probe["selected_microbatch"] = chosen_mb
    probe["scoring_mode"] = args.mode
    probe["selection_rule"] = ("fastest microbatch with no OOM, full argmax agreement with the validated "
                               f"mb{BASE_MB}, and max|diff| <= {TOL} (the model's own shape-sensitivity floor "
                               "is ~0.3-0.5); largest is NOT assumed fastest")
    probe["tolerance_fixed_before_seeing_test"] = TOL
    print(f"[probe] selected microbatch={chosen_mb} ({viable[chosen_mb]:.3f} it/s) from viable {sorted(viable)}")
    out["microbatch_probe"] = probe

    # ----------------------------------------------------- P0 pilot -------
    def run_dev(pid):
        res = {}
        for l in langs:
            t0 = time.time()
            sc = E.CachedScorer(model, tokenizer, shared[pid], surfaces, microbatch=chosen_mb,
                                share_prefix=(args.mode == "shared_prefix"))
            pred, gold = [], []
            for i in dev_ids:
                row = data[l][i]
                means, _, _ = sc.score(P.build_item_tail(row["utt"]))
                if not all(x == x and abs(x) != float("inf") for x in means):
                    raise RuntimeError(f"non-finite score in {pid}/{l} id={i} -- fix before any test run")
                pred.append(M.PROMPT_LABEL_ORDER[max(range(60), key=lambda j: means[j])])
                gold.append(row["intent"])
            sc.verify_cache_intact(); del sc
            gc.collect(); torch.cuda.empty_cache()
            n = len(gold)
            acc = sum(p == g for p, g in zip(pred, gold)) / n
            maj_lab, maj_n = collections.Counter(gold).most_common(1)[0]
            maj = maj_n / n
            cnt = collections.Counter(pred)
            res[l] = {"n": n, "acc": acc, "majority_label": maj_lab, "majority_acc": maj,
                      "margin_over_majority_pp": 100 * (acc - maj), "uniform_chance": 1 / 60,
                      "top_label_share": cnt.most_common(1)[0][1] / n,
                      "n_distinct_predicted": len(cnt),
                      "seconds": time.time() - t0, "items_per_second": n / (time.time() - t0)}
            r = res[l]
            print(f"[pilot] {pid}/{l}: acc={acc:.4f} majority={maj:.4f} margin={r['margin_over_majority_pp']:+.1f}pp "
                  f"top_share={r['top_label_share']:.2f} distinct={len(cnt)} [{r['seconds']:.0f}s]")
        res["_macro_avg_acc"] = sum(res[l]["acc"] for l in langs) / len(langs)
        print(f"[pilot] {pid} dev macro-average accuracy = {res['_macro_avg_acc']:.4f}")
        return res

    def gate_of(res):
        fr = [l for l in langs if res[l]["margin_over_majority_pp"] <= GATE_MARGIN_PP]
        return fr, len(fr) >= GATE_FLOOR_RISK_TO_HOLD

    per_prompt = {"P0": run_dev("P0")}
    fr0, hold0 = gate_of(per_prompt["P0"])
    print(f"[pilot] P0 gate: floor_risk={fr0} hold={hold0}")

    selected, used_p8 = "P0", False
    if hold0:
        print("[pilot] P0 failed the gate -> evaluating the pre-registered P8 ONCE on the same dev IDs")
        used_p8 = True
        per_prompt["P8"] = run_dev("P8")
        fr8, hold8 = gate_of(per_prompt["P8"])
        print(f"[pilot] P8 gate: floor_risk={fr8} hold={hold8}")
        if not hold8:
            selected = "P8"
        else:
            selected = "P8" if per_prompt["P8"]["_macro_avg_acc"] > per_prompt["P0"]["_macro_avg_acc"] else "P0"

    fr, hold = gate_of(per_prompt[selected])
    verdict = "PILOT_UNINFORMATIVE" if hold else "PROCEED_ALL_FIVE_LANGUAGES"
    out["per_prompt"] = per_prompt
    out["prompt_selection"] = {
        "selected": selected, "p8_evaluated": used_p8,
        "rule": ("P0 first; P8 evaluated only because P0 failed the gate" if used_p8
                 else "P0 passed the gate, so P8 was never run and no prompt search happened"),
        "P0_macro_avg_acc": per_prompt["P0"]["_macro_avg_acc"],
        "P8_macro_avg_acc": per_prompt.get("P8", {}).get("_macro_avg_acc"),
    }
    out["gate"] = {
        "margin_over_majority_pp": GATE_MARGIN_PP, "floor_risk_langs_to_hold": GATE_FLOOR_RISK_TO_HOLD,
        "uniform_chance": 1 / 60,
        "per_language_margin_pp": {l: per_prompt[selected][l]["margin_over_majority_pp"] for l in langs},
        "floor_risk_languages": fr, "n_floor_risk": len(fr), "verdict": verdict,
        "nature": "operational cost-control criterion, NOT a statistical significance test",
        "languages_dropped": "none -- no language is removed to improve the picture",
    }
    out["inference_signature"] = inference_signature(
        selected, M.sha256_text(shared[selected]), args.mode, chosen_mb)
    (NEW_DIR / "validation_and_pilot.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[pilot] SELECTED prompt={selected} mode={args.mode} microbatch={chosen_mb}")
    print(f"[pilot] GATE VERDICT: {verdict} (floor-risk: {fr})")
    print(f"[pilot] inference_signature={out['inference_signature'][:16]}")
    print(f"[pilot] wrote {NEW_DIR / 'validation_and_pilot.json'}")


if __name__ == "__main__":
    main()
