"""Pin down WHY cached candidate scoring disagrees with the uncached
reference on this model, before choosing the grid's scoring path.

The first diagnostic showed both cached modes disagreeing with the reference
on 4/12 items with |Δ| up to 4.6 nats. That is too large for rounding, and
`per_item` (whose prompt forward is contiguous at batch 1, exactly as the
reference does it) was no better than `shared_prefix` -- which points at the
CANDIDATE forward, not at prefix caching.

Mechanism under suspicion: DeepseekMoE.moe_infer groups tokens by expert and
runs one variable-row GEMM per expert, so every expert's matmul shape depends
on the whole batch's token composition; the next layer's hard top-6 routing
converts that perturbation into a discrete flip, compounding across 27 MoE
layers.

But the reference is NOT a fixed yardstick either: it forwards
prompt+completion at a DIFFERENT sequence length for each of the 60
candidates (labels are 2-5 tokens), so its own prefix logits vary candidate to
candidate for exactly the same reason. Declaring the cached scorer "wrong"
against it without first measuring that would be comparing against a moving
target. So this script measures the yardstick too:

    ref              per-candidate full forward, batch 1            (oracle)
    ref_repeat       the identical call again                        -> pure run-to-run noise floor
    ref_fixedlen     every candidate padded to one common length     -> sequence-length sensitivity of the oracle
    cache_mb1        per-item cache, microbatch 1, no padding        -> cache effect alone
    cache_mb6_pad    per-item cache, microbatch 6, right-padded      -> + padding + batch mixing
    cache_mb6_group  per-item cache, microbatch 6, equal-length groups -> batch mixing without padding

Read the output as: if ref_repeat is ~0 but ref_fixedlen is large, the model is
intrinsically shape-sensitive and no scorer can agree with the oracle to
better than that -- in which case the requirement becomes internal
CONSISTENCY across all 95 cells (identical shapes everywhere), which every
mode here satisfies by construction, and the oracle comparison is reported as
a measured uncertainty rather than a pass/fail gate.

Run: CUDA_VISIBLE_DEVICES=2,3 python massive_scorer_ablation.py --n-items 6 --langs EN,ZH
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import massive_data as M  # noqa: E402
import massive_build_protocol as P  # noqa: E402
import massive_eval as E  # noqa: E402

OUT_DIR = SCRIPT_DIR.parents[1] / "results" / "downstream" / "massive_legacy"


@torch.inference_mode()
def ref_fixedlen(model, tokenizer, prompt_text, surfaces, pad_id):
    """Reference, but every candidate forward has the SAME tensor shape: each
    prompt+completion is right-padded to the longest candidate's length.
    Padding sits strictly after the scored tokens, so under causal attention
    it cannot affect them mathematically -- only kernel shape changes."""
    pref, comps = E.build_candidate_tokens(tokenizer, prompt_text, surfaces)
    dev = next(model.parameters()).device
    L, mx = len(pref), max(len(c) for c in comps)
    means, sums, ntoks = [], [], []
    for comp in comps:
        ids = pref + comp + [pad_id] * (mx - len(comp))
        out = model(input_ids=torch.tensor([ids], device=dev))
        lp = torch.log_softmax(out.logits[0, L - 1:L + len(comp) - 1].float(), dim=-1)
        tot = sum(lp[t, comp[t]].item() for t in range(len(comp)))
        sums.append(tot); ntoks.append(len(comp)); means.append(tot / len(comp))
        del out, lp
    return means, sums, ntoks


@torch.inference_mode()
def cache_grouped(model, tokenizer, prompt_text, surfaces, microbatch, pad_id, group_by_len):
    """Per-item prompt cache (contiguous batch-1 prompt forward, as the
    reference does), then candidates scored in microbatches. group_by_len=True
    batches only equal-length candidates, so no pad token ever enters the MoE
    router."""
    pref, comps = E.build_candidate_tokens(tokenizer, prompt_text, surfaces)
    dev = next(model.parameters()).device
    out = model(input_ids=torch.tensor([pref], device=dev), use_cache=True)
    item_cache = E._to_legacy(out.past_key_values)
    first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    del out
    n_ctx = len(pref)

    order = list(range(len(comps)))
    if group_by_len:
        buckets = collections.defaultdict(list)
        for i in order:
            buckets[len(comps[i])].append(i)
        batches = []
        for ln in sorted(buckets):
            g = buckets[ln]
            batches += [g[k:k + microbatch] for k in range(0, len(g), microbatch)]
    else:
        batches = [order[k:k + microbatch] for k in range(0, len(order), microbatch)]

    res = {}
    for b in batches:
        chunk = [comps[i] for i in b]
        B, Lc = len(chunk), max(len(c) for c in chunk)
        padded = torch.full((B, Lc), pad_id, dtype=torch.long, device=dev)
        for j, c in enumerate(chunk):
            padded[j, :len(c)] = torch.tensor(c, device=dev)
        exp = tuple((k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)) for k, v in item_cache)
        o = model(input_ids=padded,
                  attention_mask=torch.ones((B, n_ctx + Lc), dtype=torch.long, device=dev),
                  past_key_values=exp, use_cache=True)
        lp = torch.log_softmax(o.logits[:, :max(Lc - 1, 1)].float(), dim=-1)
        for j, i in enumerate(b):
            c = comps[i]
            tot = first[c[0]].item()
            for t in range(1, len(c)):
                tot += lp[j, t - 1, c[t]].item()
            res[i] = (tot / len(c), tot, len(c))
        del o, lp, exp, padded
    del item_cache, first
    means = [res[i][0] for i in order]
    sums = [res[i][1] for i in order]
    ntoks = [res[i][2] for i in order]
    return means, sums, ntoks


def compare(a, b):
    """b is the yardstick."""
    d = max(abs(x - y) for x, y in zip(a, b))
    am_a = max(range(len(a)), key=lambda i: a[i])
    am_b = max(range(len(b)), key=lambda i: b[i])
    srt = sorted(b, reverse=True)
    margin = srt[0] - srt[1]
    return d, int(am_a != am_b), (d / margin if margin > 0 else float("inf")), margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, default=6)
    ap.add_argument("--langs", default="EN,ZH")
    ap.add_argument("--prompt", default="P0")
    ap.add_argument("--microbatch", type=int, default=6)
    args = ap.parse_args()

    protocol, shared, id_sets, data = E.load_protocol()
    langs = args.langs.split(",")
    surfaces = [M.LABEL_SURFACE[lab] for lab in M.PROMPT_LABEL_ORDER]
    ids = id_sets["dev"][:args.n_items]
    model, tokenizer = E.load_model()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    print(f"[abl] prompt={args.prompt} items={len(ids)} langs={langs} microbatch={args.microbatch}")

    modes = ["ref_repeat", "ref_fixedlen", "cache_mb1", "cache_mb6_pad", "cache_mb6_group"]
    acc = {m: {"max_abs": 0.0, "disagree": 0, "max_over_margin": 0.0, "n": 0} for m in modes}
    margins = []
    for l in langs:
        for i in ids:
            pt = shared[args.prompt] + P.build_item_tail(data[l][i]["utt"])
            ref, _, _ = E.score_reference(model, tokenizer, pt, surfaces)
            srt = sorted(ref, reverse=True); margins.append(srt[0] - srt[1])
            got = {
                "ref_repeat": E.score_reference(model, tokenizer, pt, surfaces)[0],
                "ref_fixedlen": ref_fixedlen(model, tokenizer, pt, surfaces, pad_id)[0],
                "cache_mb1": cache_grouped(model, tokenizer, pt, surfaces, 1, pad_id, False)[0],
                "cache_mb6_pad": cache_grouped(model, tokenizer, pt, surfaces, args.microbatch, pad_id, False)[0],
                "cache_mb6_group": cache_grouped(model, tokenizer, pt, surfaces, args.microbatch, pad_id, True)[0],
            }
            for m, v in got.items():
                d, dis, om, _ = compare(v, ref)
                a = acc[m]
                a["max_abs"] = max(a["max_abs"], d); a["disagree"] += dis
                a["max_over_margin"] = max(a["max_over_margin"], om); a["n"] += 1
            torch.cuda.empty_cache()
        print(f"[abl] done {l}")
    print(f"[abl] reference top1-top2 margin: min={min(margins):.4f} median="
          f"{sorted(margins)[len(margins)//2]:.4f} max={max(margins):.4f}")
    for m in modes:
        a = acc[m]
        print(f"[abl] {m:16s} vs ref: disagree={a['disagree']}/{a['n']} max|D|={a['max_abs']:.4e} "
              f"max|D|/margin={a['max_over_margin']:.3f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "massive_scorer_ablation.json").write_text(json.dumps({
        "prompt": args.prompt, "langs": langs, "item_ids": ids, "microbatch": args.microbatch,
        "reference_margin": {"min": min(margins), "median": sorted(margins)[len(margins)//2],
                             "max": max(margins)},
        "vs_reference": acc,
        "reading": ("ref_repeat is the pure run-to-run noise floor; ref_fixedlen is the oracle's own "
                    "sensitivity to nothing but tensor shape. Any mode at or below ref_fixedlen is within "
                    "the model's intrinsic shape-sensitivity and cannot be called incorrect relative to an "
                    "oracle that is itself that unstable."),
    }, indent=2))
    print(f"[abl] wrote {OUT_DIR / 'massive_scorer_ablation.json'}")


if __name__ == "__main__":
    main()
