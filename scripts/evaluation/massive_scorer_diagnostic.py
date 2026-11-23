"""Decide -- on evidence, before the grid runs -- which scoring path the
MASSIVE grid may use.

BACKGROUND. The dev pilot's first equivalence check showed the shared-prefix
cached scorer differing from the uncached reference by up to ~2.8e-01 in a
candidate's mean log-likelihood. That is orders of magnitude beyond bf16
reduction-order noise, and the reason is architectural, not a coding error:
DeepSeek-MoE-16B routes each token to a hard top-6 of 64 experts. Expert
selection is a STEP function of the hidden state, so a ~1e-3 bf16 perturbation
introduced by splitting the prefix into separately-forwarded chunks can flip
which experts a token uses, and the resulting logprob change is a jump, not a
rounding error. (phase1_merge_eval.eval_flores_ppl already documents the
continuous, much milder version of this for bpb across batch sizes.)

A large |Δ| is not automatically disqualifying -- a candidate that was going
to lose by 3 nats can move by 0.3 and still lose. What decides a prediction is
|Δ| against the top1-top2 margin, and ultimately whether the argmax moves. So
this script measures all three paths on the same dev items:

    reference      no cache, one full forward per candidate (the oracle)
    per_item       each item's full prompt forwarded contiguously at batch 1,
                   cache reused only across that item's 60 candidates
    shared_prefix  one KV cache for the instruction+labels+demos span, reused
                   across every item (fastest)

and reports, per mode: argmax agreement with the reference, max |Δ|, and
max |Δ| / margin. The grid uses the FASTEST mode with zero argmax
disagreements on this sample; if none qualifies, it uses `reference` and the
cost is re-estimated rather than the protocol being relaxed.

Run:  CUDA_VISIBLE_DEVICES=2,3 python massive_scorer_diagnostic.py --n-items 20
"""
import argparse
import gc
import json
import sys
import time

import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import massive_data as M  # noqa: E402
import massive_build_protocol as P  # noqa: E402
import massive_eval as E  # noqa: E402

OUT_DIR = SCRIPT_DIR.parents[1] / "results" / "downstream" / "massive_legacy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, default=20)
    ap.add_argument("--prompts", default="P0,P8",
                    help="comma-separated prompts to validate the scoring modes on. Both are validated by "
                         "default because the prompt itself has not been selected yet -- that selection needs "
                         "a trustworthy scorer, so the scorer is settled first, for every prompt it might use.")
    ap.add_argument("--microbatch", type=int, default=6)
    ap.add_argument("--langs", default=None)
    args = ap.parse_args()

    protocol, shared, id_sets, data = E.load_protocol()
    prompts = args.prompts.split(",")
    langs = args.langs.split(",") if args.langs else list(M.LOCALES)
    surfaces = [M.LABEL_SURFACE[lab] for lab in M.PROMPT_LABEL_ORDER]
    ids = id_sets["dev"][:args.n_items]
    print(f"[diag] prompts={prompts} items={len(ids)} langs={langs}")

    model, tokenizer = E.load_model()
    if hasattr(model, "hf_device_map"):
        off = {k: v for k, v in model.hf_device_map.items() if v in ("disk", "cpu")}
        if off:
            raise RuntimeError(f"{len(off)} module(s) offloaded off-GPU")

    out = {"prompts": prompts, "n_items": len(ids), "item_ids": ids, "microbatch": args.microbatch,
           "model": "deepseek-ai/deepseek-moe-16b-base (uncompressed)", "per_prompt": {}}
    for prompt in prompts:
        out["per_prompt"][prompt] = {}
        for l in langs:
            tails = [P.build_item_tail(data[l][i]["utt"]) for i in ids]
            scorers = {
                "shared_prefix": E.CachedScorer(model, tokenizer, shared[prompt], surfaces,
                                                microbatch=args.microbatch, share_prefix=True),
                "per_item": E.CachedScorer(model, tokenizer, shared[prompt], surfaces,
                                           microbatch=args.microbatch, share_prefix=False),
            }
            stats = {k: {"n_argmax_disagree": 0, "max_abs_diff": 0.0, "max_diff_over_margin": 0.0,
                         "max_abs_diff_on_ref_argmax": 0.0, "seconds": 0.0} for k in scorers}
            t_ref = 0.0
            for tail in tails:
                t = time.time()
                rm, rs, rn = E.score_reference(model, tokenizer, shared[prompt] + tail, surfaces)
                t_ref += time.time() - t
                am_r = max(range(60), key=lambda i: rm[i])
                srt = sorted(rm, reverse=True)
                margin = srt[0] - srt[1]
                for name, sc in scorers.items():
                    t = time.time()
                    cm, cs, cn = sc.score(tail)
                    stats[name]["seconds"] += time.time() - t
                    if cn != rn:
                        raise RuntimeError(f"token-count mismatch in {name}")
                    d = max(abs(a - b) for a, b in zip(cm, rm))
                    am_c = max(range(60), key=lambda i: cm[i])
                    s_ = stats[name]
                    s_["n_argmax_disagree"] += int(am_c != am_r)
                    s_["max_abs_diff"] = max(s_["max_abs_diff"], d)
                    s_["max_abs_diff_on_ref_argmax"] = max(s_["max_abs_diff_on_ref_argmax"],
                                                           abs(cm[am_r] - rm[am_r]))
                    if margin > 0:
                        s_["max_diff_over_margin"] = max(s_["max_diff_over_margin"], d / margin)
            for name, sc in scorers.items():
                sc.verify_cache_intact()
                stats[name]["items_per_second"] = len(tails) / stats[name]["seconds"]
            stats["reference"] = {"seconds": t_ref, "items_per_second": len(tails) / t_ref,
                                  "n_argmax_disagree": 0, "max_abs_diff": 0.0, "max_diff_over_margin": 0.0}
            out["per_prompt"][prompt][l] = stats
            for name in ("shared_prefix", "per_item"):
                s_ = stats[name]
                print(f"[diag] {prompt}/{l}/{name}: argmax_disagree={s_['n_argmax_disagree']}/{len(tails)} "
                      f"max|D|={s_['max_abs_diff']:.3e} max|D|/margin={s_['max_diff_over_margin']:.3f} "
                      f"{s_['items_per_second']:.2f} it/s")
            print(f"[diag] {prompt}/{l}/reference: {stats['reference']['items_per_second']:.3f} it/s")
            del scorers
            gc.collect(); torch.cuda.empty_cache()

    agg = {}
    for name in ("shared_prefix", "per_item", "reference"):
        cells = [out["per_prompt"][p][l][name] for p in prompts for l in langs]
        agg[name] = {
            "total_argmax_disagreements": sum(c["n_argmax_disagree"] for c in cells),
            "n_compared": len(ids) * len(langs) * len(prompts),
            "worst_max_abs_diff": max(c["max_abs_diff"] for c in cells),
            "worst_max_diff_over_margin": max(c["max_diff_over_margin"] for c in cells),
            "mean_items_per_second": sum(c["items_per_second"] for c in cells) / len(cells),
        }
    order = ["shared_prefix", "per_item", "reference"]  # fastest first
    chosen = next((n for n in order if agg[n]["total_argmax_disagreements"] == 0), "reference")
    out["aggregate"] = agg
    out["selected_mode"] = chosen
    out["selection_rule"] = ("fastest mode with zero argmax disagreements against the uncached reference on "
                             "this dev sample, across every prompt it might be used with; otherwise the "
                             "reference itself")
    out["caveat"] = ("Zero disagreements on this sample bounds the disagreement rate loosely, it does not prove "
                     "exact equality. The mechanism is MoE top-6 expert flips under bf16 perturbation, so a "
                     "residual disagreement rate of order 1/n_compared cannot be excluded and is reported as a "
                     "limitation rather than claimed away.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "massive_scorer_diagnostic.json").write_text(json.dumps(out, indent=2))
    print(f"[diag] aggregate: " + json.dumps(agg, indent=2))
    print(f"[diag] SELECTED SCORING MODE = {chosen}")


if __name__ == "__main__":
    main()
