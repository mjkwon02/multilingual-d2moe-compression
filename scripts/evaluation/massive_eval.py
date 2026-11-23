"""MASSIVE 60-way intent classification for this project's DeepSeek-MoE-16B
baseline and its Phase-1 compressed checkpoints.

WHAT IS DIFFERENT FROM THE OTHER PHASE-1 DOWNSTREAM EVALUATORS. This one
cannot reuse phase1_xnli_eval.score_batch() the way phase1_sib200_eval.py and
phase1_xnli_eval_v2.py do. That helper scores exactly one logit position per
candidate, which is valid only because XNLI-v2's 3 and SIB-200's 7 candidates
were all verified single-token. MASSIVE's 60 official intents are 2-5 tokens
each for this tokenizer (checked 2026-09-16), and -- crucially -- they are
NOT all the same length, so the candidate set cannot be compared on a single
position, and summed vs. averaged log-likelihood are not monotone-equivalent
here. The protocol therefore pre-registers the token-AVERAGED conditional
log-likelihood as the primary score (massive_protocol.json), with the raw sum
and the token count stored alongside, and this module implements real
multi-token completion scoring.

TWO SCORERS, ON PURPOSE.

  score_reference()  Re-tokenizes prompt+completion from scratch and runs one
                     full forward per candidate. No cache, no batching, no
                     sharing. Obviously correct, unusably slow (60 full
                     forwards per item).

  CachedScorer       What actually runs the grid. Every item of a given
                     (model, prompt, language) shares one identical
                     instruction+label-list+demonstration prefix, so that
                     prefix's KV is computed ONCE per cell and reused for all
                     600 items; per item only the short "Utterance: ...
                     Intent:" tail is forwarded; the 60 candidates are then
                     scored in microbatches off the item's cache. ~60x fewer
                     prefix forwards.

The second is only trustworthy if it provably equals the first, so
check_equivalence() runs both on fixed dev items and reports the max absolute
score difference and any argmax disagreement. That check is mandatory before
the grid (massive_run_grid.sh runs it), its tolerance is recorded, and it is
re-run on the first compressed checkpoint too -- cache reuse is exercised
through Merge_deepseekMoE's rewritten MoE forward there, which is a different
code path from the baseline's and deserves its own evidence.

Cache mutation. DynamicCache.update() in transformers 4.42 rebinds its list
slots to freshly torch.cat'ed tensors and never writes into the tensors it
was handed, so feeding the same shared-prefix tensors to every item is safe.
"Safe by reading the library" is not enough for a result we intend to
publish, so the shared cache is fingerprinted before and after every cell and
a drift aborts the run (see CachedScorer.verify_cache_intact).

Padding. Candidate completions are right-padded within a microbatch and the
attention mask is deliberately all-ones. Under causal attention a valid
position can never attend to a later pad, so this is exactly equivalent to
masking, while avoiding the fully-masked-row NaNs that a zero mask can
produce. It also does not trip transformers' `_ignore_causal_mask_sdpa`
shortcut, which only fires when key_value_length == query_length (i.e. no
past) -- verified in this env's modeling_attn_mask_utils.py, and in any case
covered by check_equivalence().

Usage:
    python massive_eval.py --baseline --split dev --prompt P0
    python massive_eval.py --baseline --split dev --prompt P0 --equivalence-check
    python massive_eval.py --condition mixed_5lang --seed 0 --scale-condition mixed_5lang \\
        --split test --prompt P8
"""
import argparse
import fcntl
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
D2MOE_DIR = Path(os.environ.get("D2MOE_DIR", SCRIPT_DIR.parents[1] / "third_party" / "D2MoE"))
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
sys.path.insert(0, str(D2MOE_DIR))
import massive_data as M  # noqa: E402
import massive_build_protocol as P  # noqa: E402 -- prompt construction lives with the frozen protocol
from phase1_merge_eval import load_model, merge_condition, RESULTS_ROOT  # noqa: E402 -- reused unchanged

OUT_ROOT = RESULTS_ROOT / "massive"
PROTOCOL_DIR = SCRIPT_DIR.parents[1] / "results" / "downstream" / "massive_legacy"


# ------------------------------------------------------------- protocol ----
def load_protocol():
    """Loads the frozen protocol and re-derives the prompt prefixes from the
    same code that froze them. A hash mismatch means the prompt changed after
    freezing, which invalidates every already-computed cell -- so it aborts
    rather than silently evaluating a different task."""
    protocol = json.loads((PROTOCOL_DIR / "massive_protocol.json").read_text())
    demo_meta = json.loads((PROTOCOL_DIR / "massive_demo_ids.json").read_text())
    data = M.load_all()
    demos = [data[demo_meta["lang"]][i] for i in demo_meta["ids"]]
    shared = {"P0": P.build_shared_prefix([]), "P8": P.build_shared_prefix(demos)}
    for pid, text in shared.items():
        want = protocol["prompts"][pid]["shared_prefix_sha256"]
        got = M.sha256_text(text)
        if want != got:
            raise RuntimeError(
                f"{pid} shared prefix hash drifted since the protocol was frozen "
                f"(frozen {want[:12]}, now {got[:12]}). Refusing to run: this would silently "
                "evaluate a different prompt than already-completed cells used.")
    ids = {
        "dev": json.loads((PROTOCOL_DIR / "massive_dev_ids.json").read_text())["ids"],
        "test": json.loads((PROTOCOL_DIR / "massive_test_ids.json").read_text())["ids"],
    }
    return protocol, shared, ids, data


# -------------------------------------------------------------- scoring ----
def _ids(tokenizer, text):
    return tokenizer(text)["input_ids"]


def build_candidate_tokens(tokenizer, prefix_text, surfaces):
    """Completion token ids for every candidate, taken as the true suffix of
    the jointly-tokenized prompt+completion -- never by tokenizing the label
    on its own and concatenating. Asserts the prefix invariant for all 60
    candidates, which is what makes the cached scorer's reuse of one
    pre-tokenized prefix legitimate."""
    pref = _ids(tokenizer, prefix_text)
    out = []
    for s in surfaces:
        full = _ids(tokenizer, prefix_text + P.COMPLETION_PREFIX + s)
        if full[:len(pref)] != pref:
            raise RuntimeError(
                f"tokenizer boundary violation: appending {P.COMPLETION_PREFIX + s!r} changed the prompt's "
                "own token ids. Candidate completions cannot be scored as a suffix for this prompt.")
        comp = full[len(pref):]
        if not comp:
            raise RuntimeError(f"empty completion for candidate {s!r}")
        out.append(comp)
    return pref, out


@torch.inference_mode()
def score_reference(model, tokenizer, prompt_text, surfaces):
    """Uncached, one full forward per candidate. The correctness oracle."""
    pref, comps = build_candidate_tokens(tokenizer, prompt_text, surfaces)
    dev = next(model.parameters()).device
    means, sums, ntoks = [], [], []
    L = len(pref)
    for comp in comps:
        full = pref + comp
        out = model(input_ids=torch.tensor([full], device=dev))
        # Slice to the scored positions BEFORE .float()/log_softmax. The full
        # (seq_len, 102400) float32 log-softmax is ~300MB for a 750-token
        # prompt and would be built 60 times per item; only the len(comp)
        # rows that score the completion are ever read. Logits at position j
        # predict token j+1, so completion token t (absolute index L+t) is
        # predicted at L+t-1.
        lp = torch.log_softmax(out.logits[0, L - 1:L + len(comp) - 1].float(), dim=-1)
        tot = sum(lp[t, comp[t]].item() for t in range(len(comp)))
        sums.append(tot)
        ntoks.append(len(comp))
        means.append(tot / len(comp))
        del out, lp
    return means, sums, ntoks


def _to_legacy(cache):
    return cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache


def _fingerprint(legacy):
    """Cheap content fingerprint of a legacy KV cache, used to prove the
    shared prefix cache is not mutated by reuse."""
    return [(float(k.float().sum()), float(v.float().sum())) for k, v in legacy]


class CachedScorer:
    """Shared-prefix KV reuse + candidate microbatching, for one
    (model, prompt, language-independent) cell. The shared prefix is the
    instruction + 60-label list + demonstrations; only the per-item
    "Utterance: ...\\nIntent:" tail and the candidates differ."""

    def __init__(self, model, tokenizer, shared_text, surfaces, microbatch=15, share_prefix=True,
                 recheck_every=50):
        """share_prefix=True reuses ONE KV cache for the instruction+label-list
        +demonstration span across every item (fastest). share_prefix=False
        forwards each item's full prompt contiguously at batch 1 -- i.e.
        exactly as score_reference() does -- and caches only that, so the
        prompt's hidden states are bit-identical to the reference and only the
        candidate continuation is batched. The second mode exists because this
        model's MoE gate takes a hard top-6 of 64 experts, which is a
        DISCONTINUOUS function of the hidden state: chunking the prefix
        perturbs it in bf16 by ~1e-3, which can flip an expert and move a
        candidate's logprob far more than reduction-order noise alone would.
        massive_scorer_diagnostic.py measures which mode is safe here; the
        choice is recorded in the run's output, never left implicit."""
        self.model, self.tok, self.microbatch = model, tokenizer, microbatch
        self.share_prefix = share_prefix
        self.dev = next(model.parameters()).device
        self.shared_ids = _ids(tokenizer, shared_text)
        self.shared_text = shared_text
        self.surfaces = surfaces
        self.shared_cache = None
        self._fp0 = None
        if share_prefix:
            with torch.inference_mode():
                out = model(input_ids=torch.tensor([self.shared_ids], device=self.dev), use_cache=True)
            self.shared_cache = _to_legacy(out.past_key_values)
            self._fp0 = _fingerprint(self.shared_cache)
            del out
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self._comps = None
        self._n_scored = 0
        self.recheck_every = recheck_every

    def verify_cache_intact(self):
        if self.shared_cache is None:
            return True  # nothing is shared between items in share_prefix=False mode
        fp = _fingerprint(self.shared_cache)
        for i, ((a, b), (c, d)) in enumerate(zip(self._fp0, fp)):
            if a != c or b != d:
                raise RuntimeError(
                    f"shared prefix KV cache was mutated at layer {i} -- every item scored against it is "
                    "invalid. Refusing to continue.")
        return True

    def candidate_tokens_for(self, tail_text):
        """(prompt token ids for THIS item, candidate completion token ids).

        The prompt ids are ALWAYS re-tokenized for the actual item and are
        never cached. An earlier version keyed a cache of (pref, comps) on the
        prompt's token LENGTH, which is catastrophically wrong: MASSIVE
        utterances are short and similar in length, so the 600-item test
        subset spans only ~20 distinct prompt token lengths, and ~96% of items
        collided onto an earlier item's cache entry and were scored against a
        DIFFERENT utterance's prompt. It did not look broken -- every item was
        still scored against some grammatical prompt, so accuracies stayed
        plausible and only a minority of argmax predictions actually moved.
        Measured 2026-09-16 (massive_scorer_ablation.py vs
        massive_scorer_diagnostic.py disagreeing on identical items).

        Only the candidate COMPLETION ids are cached, and that is legitimate
        for a different reason than length: every prompt in this protocol ends
        with the identical boundary text ('...\\nIntent:'), so the tokenization
        of ' <label>' appended to it cannot depend on the item. That premise is
        verified against the real prompt on first use and re-verified every
        `recheck_every` items, so a violation surfaces as a failure rather
        than as silently mis-aligned scoring."""
        prefix_text = self.shared_text + tail_text
        pref = _ids(self.tok, prefix_text)
        self._n_scored += 1
        if self._comps is None or (self._n_scored % self.recheck_every) == 0:
            pref_check, comps = build_candidate_tokens(self.tok, prefix_text, self.surfaces)
            if pref_check != pref:
                raise RuntimeError("prompt tokenization is not self-consistent for this item")
            if self._comps is not None and comps != self._comps:
                raise RuntimeError(
                    "candidate completion token ids changed between items -- the constant-boundary premise "
                    "behind caching them does not hold for this prompt. Refusing to continue.")
            self._comps = comps
        return pref, self._comps

    @torch.inference_mode()
    def score(self, tail_text):
        pref, comps = self.candidate_tokens_for(tail_text)
        n_shared = len(self.shared_ids)
        if pref[:n_shared] != self.shared_ids:
            raise RuntimeError("item prompt does not start with the cached shared prefix's exact token ids")

        if self.share_prefix:
            # Extend the shared cache by this item's tail. from_legacy_cache
            # wraps (does not copy) our tensors and update() rebinds with
            # torch.cat, so self.shared_cache survives untouched -- asserted
            # by verify_cache_intact() at the end of every cell.
            fwd_ids, past = pref[n_shared:], self.shared_cache
        else:
            fwd_ids, past = pref, None
        out = self.model(input_ids=torch.tensor([fwd_ids], device=self.dev),
                         past_key_values=past, use_cache=True)
        item_cache = _to_legacy(out.past_key_values)
        first_logits = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        del out

        n_ctx = len(pref)
        means, sums, ntoks = [], [], []
        for start in range(0, len(comps), self.microbatch):
            chunk = comps[start:start + self.microbatch]
            B, L = len(chunk), max(len(c) for c in chunk)
            padded = torch.full((B, L), self.pad_id, dtype=torch.long, device=self.dev)
            for j, c in enumerate(chunk):
                padded[j, :len(c)] = torch.tensor(c, device=self.dev)
            expanded = tuple((k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)) for k, v in item_cache)
            o = self.model(input_ids=padded,
                           attention_mask=torch.ones((B, n_ctx + L), dtype=torch.long, device=self.dev),
                           past_key_values=expanded, use_cache=True)
            # lp[:, t] predicts token t+1, so the final position is never
            # read; dropping it before the float32 log-softmax keeps this
            # allocation at (B, L-1, V) instead of (B, L, V).
            lp = torch.log_softmax(o.logits[:, :max(L - 1, 1)].float(), dim=-1)
            for j, c in enumerate(chunk):
                tot = first_logits[c[0]].item()
                for t in range(1, len(c)):
                    tot += lp[j, t - 1, c[t]].item()
                sums.append(tot)
                ntoks.append(len(c))
                means.append(tot / len(c))
            del o, lp, expanded, padded
        del item_cache, first_logits
        return means, sums, ntoks


def check_equivalence(model, tokenizer, shared_text, surfaces, tails, microbatch, tol=1e-3,
                      share_prefix=True):
    """Cached scorer vs. uncached reference on fixed items. Reports the max
    absolute difference in the primary (mean) score and any argmax
    disagreement. Pre-registered tolerance is on the ARGMAX (must be zero
    disagreements); the numeric delta is recorded, not thresholded, because
    bf16 reduction order legitimately differs between the two batch shapes
    (the same effect phase1_merge_eval.eval_flores_ppl documents for bpb)."""
    scorer = CachedScorer(model, tokenizer, shared_text, surfaces, microbatch=microbatch,
                          share_prefix=share_prefix)
    max_abs, argmax_disagree, rows = 0.0, 0, []
    for tail in tails:
        cm, cs, cn = scorer.score(tail)
        rm, rs, rn = score_reference(model, tokenizer, shared_text + tail, surfaces)
        if cn != rn:
            raise RuntimeError(f"token counts differ between scorers: {cn} vs {rn}")
        d = max(abs(a - b) for a, b in zip(cm, rm))
        am_c = max(range(len(cm)), key=lambda i: cm[i])
        am_r = max(range(len(rm)), key=lambda i: rm[i])
        argmax_disagree += int(am_c != am_r)
        max_abs = max(max_abs, d)
        # A large |Δ| on a hopeless candidate is harmless; what decides the
        # prediction is |Δ| relative to the reference's top1-top2 gap. Both are
        # recorded so the fast path is judged on the quantity that matters.
        srt = sorted(rm, reverse=True)
        margin = srt[0] - srt[1]
        rows.append({"tail_sha256": M.sha256_text(tail)[:16], "max_abs_mean_logprob_diff": d,
                     "abs_diff_on_reference_argmax": abs(cm[am_r] - rm[am_r]),
                     "reference_top1_top2_margin": margin,
                     "max_abs_diff_over_margin": (d / margin) if margin > 0 else float("inf"),
                     "cached_argmax": surfaces[am_c], "reference_argmax": surfaces[am_r],
                     "agree": am_c == am_r})
    scorer.verify_cache_intact()
    return {"n_items": len(tails), "max_abs_mean_logprob_diff": max_abs,
            "share_prefix": share_prefix,
            "max_abs_diff_over_margin": max(r["max_abs_diff_over_margin"] for r in rows),
            "n_argmax_disagreements": argmax_disagree, "argmax_tolerance": 0,
            "numeric_tolerance_recorded_not_enforced": tol,
            "shared_cache_unmutated": True, "per_item": rows}


def check_determinism(model, tokenizer, shared_text, surfaces, tails, microbatch, share_prefix):
    """Score the same items twice with the same scorer and report whether the
    results are bit-identical.

    This is the property the grid's validity actually rests on. The candidate
    batch shapes are fixed by the 60 label token-lengths and the tokenizer, so
    they are IDENTICAL for every item, language, condition and seed -- every
    one of the 19 models is scored through exactly the same tensor shapes and
    the same batching. Model-vs-model contrasts (the scientific question) are
    therefore not confounded by this model's shape sensitivity at all; only
    agreement with a hypothetical shape-free absolute oracle is. Determinism
    is what makes that argument hold, so it is measured rather than asserted.
    """
    sc = CachedScorer(model, tokenizer, shared_text, surfaces, microbatch=microbatch,
                      share_prefix=share_prefix)
    n_ident, worst, argmax_flips = 0, 0.0, 0
    for tail in tails:
        a, _, _ = sc.score(tail)
        b, _, _ = sc.score(tail)
        d = max(abs(x - y) for x, y in zip(a, b))
        worst = max(worst, d)
        n_ident += int(d == 0.0)
        argmax_flips += int(max(range(len(a)), key=lambda i: a[i]) != max(range(len(b)), key=lambda i: b[i]))
    sc.verify_cache_intact()
    return {"n_items": len(tails), "n_bit_identical": n_ident, "worst_abs_diff": worst,
            "n_argmax_flips_on_repeat": argmax_flips,
            "interpretation": ("bit-identical repeats mean the scoring procedure is reproducible, so all 95 "
                              "cells are mutually comparable; any residual gap to an uncached oracle is a "
                              "shared, model-independent offset rather than per-cell noise.")}


# ---------------------------------------------------------------- metrics --
def predicted_entropy(counts, n):
    h = 0.0
    for c in counts.values():
        if c:
            p = c / n
            h -= p * math.log(p)
    return h


# ------------------------------------------------------------------ main ---
def model_key(args):
    return "baseline" if args.baseline else f"{args.condition}_seed{args.seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale-condition", default=None)
    ap.add_argument("--scale-seed", type=int, default=None)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--prompt", choices=["P0", "P8"], required=True)
    ap.add_argument("--langs", default=None, help="comma-separated subset of EN,ZH,KO,SW,BN")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--microbatch", type=int, default=6)
    ap.add_argument("--no-share-prefix", action="store_true",
                    help="forward each item's full prompt contiguously instead of reusing one shared-prefix "
                         "KV cache (slower, numerically closer to score_reference; see CachedScorer)")
    ap.add_argument("--equivalence-check", action="store_true",
                    help="run the cached-vs-reference scorer check on the first --eq-items dev items and exit")
    ap.add_argument("--eq-items", type=int, default=4)
    ap.add_argument("--determinism-check", action="store_true",
                    help="score the same dev items twice with the production scorer and report whether the "
                         "results are bit-identical (see check_determinism)")
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()
    if not args.baseline and not args.condition:
        ap.error("either --condition or --baseline is required")
    if args.scale_condition is not None and args.scale_seed is None:
        args.scale_seed = args.seed

    protocol, shared, id_sets, data = load_protocol()
    langs = args.langs.split(",") if args.langs else list(M.LOCALES)
    for l in langs:
        if l not in M.LOCALES:
            ap.error(f"--langs: unknown {l!r}")
    surfaces = [M.LABEL_SURFACE[lab] for lab in M.PROMPT_LABEL_ORDER]
    shared_text = shared[args.prompt]

    key = model_key(args)
    print(f"[massive] model={key} split={args.split} prompt={args.prompt} langs={langs}")

    t_load = time.time()
    model, tokenizer = load_model()
    load_s = time.time() - t_load

    # Same guard phase1_sib200_eval.py carries: off-GPU execution produced NaN
    # logprobs in this repo's scoring path before (SIB_DIAGNOSIS.md 2026-07-31).
    if hasattr(model, "hf_device_map"):
        off = {k: v for k, v in model.hf_device_map.items() if v in ("disk", "cpu")}
        if off:
            raise RuntimeError(f"{len(off)} module(s) offloaded off-GPU (e.g. {next(iter(off.items()))}) -- refusing to run")
    device_map = dict(getattr(model, "hf_device_map", {}))

    merge_s = 0.0
    expert_freq_file = None
    if not args.baseline:
        cond_dir = RESULTS_ROOT / args.condition / f"seed{args.seed}"
        # merge_condition() picks expert_freq by glob; record which file that
        # actually resolves to, because two conditions have both a 50- and a
        # 2000-sample file on disk and the choice is not visible in its output.
        expert_freq_file = sorted(p.name for p in cond_dir.glob("deepseek_wikitext_*_expert_frequencies.json"))
        globbed = list(cond_dir.glob("deepseek_wikitext_*_expert_frequencies.json"))
        expert_freq_file = {"available": expert_freq_file, "used_by_glob_order": globbed[0].name}
        t = time.time()
        print(f"[massive] merging condition={args.condition} seed={args.seed} "
              f"scale=({args.scale_condition},seed{args.scale_seed})")
        # The lock covers the MERGE ONLY, not the hours of scoring that follow.
        # merge_condition() torch.loads fisher_processed.pt (28G) +
        # svd_scale_processed.pt (34G) into CPU RAM; two concurrent merges do
        # not fit in this host's 125G and the documented failure mode is not a
        # clean OOM but accelerate silently disk-offloading, which yields
        # all-NaN logits that look like a finished result (SIB_DIAGNOSIS.md
        # 2026-07-31). Holding the same /tmp/phase1_merge_eval.lock that
        # e2e3_driver_0803.sh uses serializes against every other merging job
        # on this host. Releasing it immediately afterwards is what lets a
        # second evaluation stream run concurrently on the other GPUs.
        lock_path = "/tmp/phase1_merge_eval.lock"
        t_lock = time.time()
        with open(lock_path, "a") as lf:  # "a" not "w": never truncate a lock file another process may hold
            fcntl.flock(lf, fcntl.LOCK_EX)
            lock_wait_s = time.time() - t_lock
            print(f"[massive] acquired {lock_path} after {lock_wait_s:.0f}s wait; merging")
            model = merge_condition(model, args.condition, args.seed, args.scale_condition, args.scale_seed)
            gc.collect()
            # Free the 62G of Fisher/whitening tensors BEFORE releasing the
            # lock, so the next waiter does not start its own 62G load while
            # ours is still resident.
            print("[massive] merge done; releasing lock")
        merge_s = time.time() - t
        gc.collect()
        torch.cuda.empty_cache()

    if args.determinism_check:
        eq_ids = id_sets["dev"][:args.eq_items]
        rep = {}
        for l in langs:
            tails = [P.build_item_tail(data[l][i]["utt"]) for i in eq_ids]
            rep[l] = check_determinism(model, tokenizer, shared_text, surfaces, tails, args.microbatch,
                                       not args.no_share_prefix)
            r = rep[l]
            print(f"[massive-det] {l}: bit-identical {r['n_bit_identical']}/{r['n_items']} "
                  f"worst|Δ|={r['worst_abs_diff']:.3e} argmax flips on repeat={r['n_argmax_flips_on_repeat']}")
        out = OUT_ROOT / "equivalence"
        out.mkdir(parents=True, exist_ok=True)
        p = out / f"determinism_{key}_{args.prompt}{args.out_tag}.json"
        p.write_text(json.dumps({"model": key, "prompt": args.prompt, "microbatch": args.microbatch,
                                 "share_prefix": not args.no_share_prefix,
                                 "per_language": rep}, indent=2))
        print(f"[massive-det] wrote {p}")
        return

    if args.equivalence_check:
        eq_ids = id_sets["dev"][:args.eq_items]
        report = {}
        for l in langs:
            tails = [P.build_item_tail(data[l][i]["utt"]) for i in eq_ids]
            report[l] = check_equivalence(model, tokenizer, shared_text, surfaces, tails, args.microbatch,
                                          share_prefix=not args.no_share_prefix)
            r = report[l]
            print(f"[massive-eq] {l}: max|Δmean logprob|={r['max_abs_mean_logprob_diff']:.3e} "
                  f"argmax disagreements={r['n_argmax_disagreements']}/{r['n_items']}")
        out = OUT_ROOT / "equivalence"
        out.mkdir(parents=True, exist_ok=True)
        p = out / f"{key}_{args.prompt}{args.out_tag}.json"
        p.write_text(json.dumps({"model": key, "prompt": args.prompt, "microbatch": args.microbatch,
                                 "eq_item_ids": eq_ids, "device_map_unique": sorted(set(device_map.values())),
                                 "per_language": report}, indent=2))
        print(f"[massive-eq] wrote {p}")
        return

    eval_ids = id_sets[args.split]
    if args.limit:
        eval_ids = eval_ids[:args.limit]

    cells, pred_rows = {}, []
    for l in langs:
        t0 = time.time()
        scorer = CachedScorer(model, tokenizer, shared_text, surfaces, microbatch=args.microbatch,
                              share_prefix=not args.no_share_prefix)
        pred, gold = [], []
        for i in eval_ids:
            row = data[l][i]
            means, sums, ntoks = scorer.score(P.build_item_tail(row["utt"]))
            if not all(math.isfinite(x) for x in means):
                raise RuntimeError(
                    f"non-finite candidate score for {l} id={i} -- protocol gate requires aborting the "
                    "whole run (see massive_protocol.json::gate.abort_on_nonfinite_scores)")
            k = max(range(60), key=lambda j: means[j])
            p_lab = M.PROMPT_LABEL_ORDER[k]
            pred.append(p_lab)
            gold.append(row["intent"])
            pred_rows.append({
                "model": key, "condition": ("baseline" if args.baseline else args.condition),
                "seed": (None if args.baseline else args.seed), "lang": l, "split": args.split,
                "prompt": args.prompt, "id": i, "gold": row["intent"], "pred": p_lab,
                "correct": int(p_lab == row["intent"]),
                "mean_logprob": [round(x, 6) for x in means],
                "sum_logprob": [round(x, 6) for x in sums],
                "n_completion_tokens": ntoks,
            })
        scorer.verify_cache_intact()
        secs = time.time() - t0

        n = len(gold)
        acc = sum(p == g for p, g in zip(pred, gold)) / n
        counts = {lab: 0 for lab in M.PROMPT_LABEL_ORDER}
        for p_lab in pred:
            counts[p_lab] += 1
        cm = [[0] * 60 for _ in range(60)]
        for g, p_lab in zip(gold, pred):
            cm[M.LABEL_TO_IDX[g]][M.LABEL_TO_IDX[p_lab]] += 1
        cells[l] = {
            "n": n, "acc": acc,
            "confusion_matrix_60x60": cm,
            "pred_counts": counts,
            "top_predicted_label": max(counts, key=counts.get),
            "top_predicted_label_fraction": max(counts.values()) / n,
            "predicted_label_entropy_nats": predicted_entropy(counts, n),
            "n_distinct_predicted_labels": sum(1 for v in counts.values() if v),
            "seconds": secs, "items_per_second": n / secs,
        }
        print(f"[massive] {l}: acc={acc:.4f} (n={n}) top_frac={cells[l]['top_predicted_label_fraction']:.2f} "
              f"H={cells[l]['predicted_label_entropy_nats']:.2f} distinct={cells[l]['n_distinct_predicted_labels']} "
              f"[{secs:.0f}s, {n/secs:.2f} it/s]")
        del scorer
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "protocol_id": protocol["protocol_id"],
        "model": key, "baseline": args.baseline, "condition": args.condition, "seed": args.seed,
        "scale_condition": args.scale_condition, "scale_seed": args.scale_seed,
        "compression": protocol["model_grid"]["compression_config"] if not args.baseline else None,
        "expert_freq_file": expert_freq_file,
        "split": args.split, "prompt": args.prompt, "n_items": len(eval_ids),
        "is_sample_not_full_split": args.split == "test",
        "prompt_sha256": M.sha256_text(shared_text),
        "label_order": M.PROMPT_LABEL_ORDER,
        "microbatch": args.microbatch,
        "share_prefix_kv_cache": not args.no_share_prefix,
        "device_map_unique": sorted(set(device_map.values())),
        "seconds_model_load": load_s, "seconds_merge": merge_s,
        "seconds_merge_lock_wait": (lock_wait_s if not args.baseline else 0.0),
        "results": cells,
    }
    cdir = OUT_ROOT / "cells"
    pdir = OUT_ROOT / "predictions"
    cdir.mkdir(parents=True, exist_ok=True)
    pdir.mkdir(parents=True, exist_ok=True)
    stem = f"{key}__{args.split}__{args.prompt}{args.out_tag}"
    (cdir / f"{stem}.json").write_text(json.dumps(payload, indent=2))
    with open(pdir / f"{stem}.jsonl", "w") as f:
        for r in pred_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[massive] wrote {cdir / (stem + '.json')}")


if __name__ == "__main__":
    main()
