"""Phase 1: SIB-200 downstream topic-classification accuracy -- a second
lightweight downstream task, chosen specifically to close a coverage gap
phase1_xnli_eval.py documents and phase1_belebele_eval.py alone doesn't
fully offset: official XNLI (facebook/xnli) has NO Korean or Bengali split,
so this project's XNLI results only ever cover 3 of its 5 calibration
languages (en/zh/sw). Belebele covers all 5, but it's the only downstream
task doing so and is comparatively expensive (~900 4-way MC items/language,
long FLORES passages). SIB-200 (Adelani et al., EACL 2024) is FLORES-200-
based topic classification spanning 205 languages/dialects, so it covers all
5 calibration languages (en/ko/zh/sw/bn) natively -- confirmed by direct
`datasets.load_dataset("Davlan/sib200", <config>, split="test")` for
eng_Latn/kor_Hang/zho_Hans/swh_Latn/ben_Beng: 204 test examples per language,
7 categories (science/technology, travel, politics, sports, health,
entertainment, geography), same eng_Latn/kor_Hang/zho_Hans/swh_Latn/ben_Beng
config-code convention this project already standardized on for FLORES/
Belebele. NOTE the 7-way "chance" floor is NOT a uniform 1/7 (14.3%) --
categories are imbalanced (confirmed by direct inspection: science/
technology is 51/204 = 25.0% of the English test set, same distribution
across languages since SIB-200's splits are parallel/translated) -- so the
real majority-class floor here is ~25%, about the same as Belebele's 4-way
25% chance floor, not lower. Any writeup citing this task's floor should say
~25% (majority-class), not 1/7.

Scoring protocol deliberately follows phase1_xnli_eval_v2.py's lesson, NOT
phase1_xnli_eval.py's (v1) lettered-MC template: v1 diagnosed that this base
model (never instruction-tuned) fails to read a live "A. X / B. Y / C. Z"
choice layout per-item under few-shot demonstrations and instead collapses
onto the demonstration's letter distribution. So this script uses the same
fix v2 used for XNLI -- a natural-sentence continuation, scored via one
single-token log-prob per candidate, never a letter token:
    {text}
    Question: What is this text about -- Science, Travel, Politics, Sports,
    Health, Entertainment, or Geography?
    Answer:
scored against 7 single-token continuations (" Science" stands in for the
dataset's combined "science/technology" label -- confirmed single-token for
this tokenizer, see CHOICE_TOKENS; all 7 category words were checked with
this project's actual tokenizer and are single tokens, so no multi-token
summed-logprob loop is needed here, same as XNLI v2's candidates). The
question wording is fixed in English for every eval language (same rationale
as XNLI v2: isolates the eval-language variable from an instruction-
translation-quality variable -- only `text` switches per language).

Few-shot demonstrations: fixed English (eng_Latn) demonstrations, drawn from
the VALIDATION split only (never test), `per_label` examples per one of the
7 categories (confirmed validation has at least 8 examples per category, see
2026-07-30 direct inspection), identical across every checkpoint/eval-
language call -- same "fixed English-demo -> multilingual-test transfer"
design as XNLI v1/v2, generalized from 3 labels to 7. This is a fresh
build_demonstrations()/build_prompt() (not an import from phase1_xnli_eval.py)
because the field names (text/category, string labels) and label count (7,
not 3) don't match that module's premise/hypothesis/int-label interface --
per this repo's own "reuse unchanged if the interface matches, write fresh
if it doesn't" convention (see phase1_xnli_eval_v2.py's own docstring for
the same distinction). score_batch() IS reused unchanged from
phase1_xnli_eval.py -- it was already generic over any list of single-token
choice ids, not XNLI-specific.

Reuses phase1_merge_eval.py's load_model()/merge_condition() and
phase1_structured_prune.apply_structured_pruning() UNCHANGED, exactly like
every other Phase 1 eval script.

Usage:
    conda run -n d2moe_env python phase1_sib200_eval.py --baseline --smoke --limit 50 --langs eng_Latn   # gate
    conda run -n d2moe_env python phase1_sib200_eval.py --baseline --limit 204                            # full (per-lang test size)
    conda run -n d2moe_env python phase1_sib200_eval.py --condition korean_only --seed 0 \\
        --scale-condition korean_only --scale-seed 0 --limit 204
"""
import argparse
import gc
import json
import random
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
D2MOE_DIR = Path(os.environ.get("D2MOE_DIR", SCRIPT_DIR.parents[1] / "third_party" / "D2MoE"))
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
sys.path.insert(0, str(D2MOE_DIR))
import phase1_calib_data  # noqa: E402
import phase1_structured_prune  # noqa: E402
from phase1_merge_eval import load_model, merge_condition, RESULTS_ROOT  # noqa: E402 -- reused unchanged
from phase1_xnli_eval import score_batch  # noqa: E402 -- reused unchanged, already generic (see module docstring)

SIB_DATASET = "Davlan/sib200"
# Same FLORES-200 config codes this project already standardized on for
# phase1_merge_eval.py's EVAL_LANGS / phase1_belebele_eval.py's BELEBELE_LANGS
# -- SIB-200 (built on FLORES-200) uses the identical codes, so no new
# calibration/eval language mapping is needed and results are directly
# comparable across all three downstream/PPL tables.
SIB_LANGS = {
    "eng_Latn": "English", "kor_Hang": "Korean", "zho_Hans": "Chinese",
    "swh_Latn": "Swahili", "ben_Beng": "Bengali",
}
DEMO_LANG = "eng_Latn"  # fixed English demonstrations for every eval language, see module docstring

# Dataset's own category strings (confirmed by direct inspection, 2026-07-30)
# -> single-token continuation word. " Science" stands in for the dataset's
# combined "science/technology" label (see module docstring on the 1/7 vs
# ~25% floor).
CATEGORY_ORDER = ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]
CHOICE_WORDS = [" Science", " Travel", " Politics", " Sports", " Health", " Entertainment", " Geography"]
CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(CATEGORY_ORDER)}

QUESTION = "Question: What is this text about -- Science, Travel, Politics, Sports, Health, Entertainment, or Geography?"

DEMO_SEED_DEFAULT = 42
DEMOS_PER_LABEL_DEFAULT = 2  # 14-shot total, same per-label generosity as XNLI's 2-per-label (6-shot for 3 labels)
DEMO_MAX_WORDS = 50  # SIB-200 sentences run up to ~46 words in eng_Latn validation (checked 2026-07-30)


def _choice_token_ids(tokenizer):
    ids = []
    for word in CHOICE_WORDS:
        toks = tokenizer(word, add_special_tokens=False)["input_ids"]
        if len(toks) != 1:
            raise RuntimeError(
                f"choice word {word!r} is {len(toks)} tokens for this tokenizer, not 1 -- score_batch() (reused from "
                f"phase1_xnli_eval.py) assumes single-token candidates; CHOICE_WORDS needs a real multi-token "
                f"summed-logprob loop for this model.")
        ids.append(toks[0])
    return ids


def build_demonstrations(seed=DEMO_SEED_DEFAULT, per_label=DEMOS_PER_LABEL_DEFAULT, max_words=DEMO_MAX_WORDS):
    """Fixed English-validation demonstrations, `per_label` examples per
    category, identical across every checkpoint/eval-language call. Dev
    (validation) split only, never test -- same convention as
    phase1_xnli_eval.build_demonstrations()."""
    from datasets import load_dataset

    ds = load_dataset(SIB_DATASET, DEMO_LANG, split="validation")
    rng = random.Random(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)

    buckets = {cat: [] for cat in CATEGORY_ORDER}
    for i in order:
        ex = ds[i]
        cat = ex["category"]
        if len(buckets[cat]) >= per_label:
            continue
        if len(ex["text"].split()) > max_words:
            continue
        buckets[cat].append(ex)
        if all(len(v) >= per_label for v in buckets.values()):
            break
    missing = [c for c, v in buckets.items() if len(v) < per_label]
    if missing:
        raise RuntimeError(f"could not find {per_label} demonstrations under {max_words} words for category(ies) {missing}")

    demos = [ex for cat in CATEGORY_ORDER for ex in buckets[cat]]
    rng.shuffle(demos)  # interleave category order in the printed demo block
    return demos


def format_example(text, answer_word=None):
    block = f"{text}\n{QUESTION}\nAnswer:"
    if answer_word is not None:
        block += f"{answer_word}\n\n"
    return block


def build_prompt(demos, text):
    parts = [format_example(ex["text"], CHOICE_WORDS[CATEGORY_TO_IDX[ex["category"]]]) for ex in demos]
    parts.append(format_example(text))
    return "\n".join(parts)


def eval_sib200(model, tokenizer, batch_size=4, limit=None, demo_seed=DEMO_SEED_DEFAULT,
                 demos_per_label=DEMOS_PER_LABEL_DEFAULT, langs=None, n_sample_items=10):
    """Returns {lang_code: {...}} for every requested SIB-200 language
    (default: all of SIB_LANGS), evaluated on the test split. Mirrors
    phase1_xnli_eval_v2.eval_xnli()'s output shape (acc, acc_unnormalized,
    pred_distribution, _sample_items) for the same downstream analysis
    tooling to read either eval_xnli_v2.json or eval_sib200.json uniformly."""
    from datasets import load_dataset

    if langs is None:
        langs = list(SIB_LANGS.keys())

    demos = build_demonstrations(seed=demo_seed, per_label=demos_per_label)
    choice_token_ids = _choice_token_ids(tokenizer)

    results = {}
    for lang_code in langs:
        lang_name = SIB_LANGS[lang_code]
        ds = load_dataset(SIB_DATASET, lang_code, split="test")
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))

        pred, gold = [], []
        sample_items = []
        for i in range(0, len(ds), batch_size):
            chunk = ds[i:i + batch_size]
            prompts = [build_prompt(demos, t) for t in chunk["text"]]
            logprobs = score_batch(model, tokenizer, prompts, choice_token_ids)  # (batch, 7), n_tokens=1 per candidate
            # 2026-07-31 (SIB_DIAGNOSIS.md): per-item NaN guard, not just the
            # final acc. argmax() over an all-NaN row deterministically returns
            # index 0 ("science/technology") -- a real prediction-shaped value
            # that the old acc-only NaN check (main()'s nan_langs) never caught,
            # which is exactly how the disk-offload collapse went unnoticed.
            # Fail the run the instant it happens instead of writing junk.
            if torch.isnan(logprobs).any():
                nan_items = [chunk["text"][j][:80] for j in range(len(prompts)) if torch.isnan(logprobs[j]).any()]
                raise RuntimeError(
                    f"NaN logprob(s) in {lang_code} batch at item {i} (texts: {nan_items}) -- refusing to "
                    "continue. See SIB_DIAGNOSIS.md: this is the signature of accelerate disk-offloaded "
                    "execution inside score_batch() (check model.hf_device_map for 'disk'/'cpu' entries).")
            batch_pred = logprobs.argmax(dim=-1).tolist()
            batch_gold = [CATEGORY_TO_IDX[c] for c in chunk["category"]]
            pred.extend(batch_pred)
            gold.extend(batch_gold)

            for j in range(len(batch_pred)):
                if len(sample_items) >= n_sample_items:
                    continue
                sample_items.append({
                    "text": chunk["text"][j], "gold": CATEGORY_ORDER[batch_gold[j]], "pred": CATEGORY_ORDER[batch_pred[j]],
                    "logprobs": {CATEGORY_ORDER[k]: logprobs[j, k].item() for k in range(len(CATEGORY_ORDER))},
                })

        n = len(gold)
        acc = sum(p == g for p, g in zip(pred, gold)) / n if n else float("nan")
        dist = {cat: 0 for cat in CATEGORY_ORDER}
        for p in pred:
            dist[CATEGORY_ORDER[p]] += 1
        results[lang_code] = {
            "name": lang_name, "acc": acc, "acc_unnormalized": acc, "n_samples": n,
            "pred_distribution": dist, "_sample_items": sample_items,
        }
        top_frac = max(dist.values()) / n if n else float("nan")
        print(f"[sib200-eval] {lang_name} ({lang_code}): acc={acc:.4f} (n={n}) "
              f"pred_dist={dist} top_label_frac={top_frac:.2f}")

    results["_demonstrations"] = [{"text": ex["text"], "category": ex["category"]} for ex in demos]
    results["_demo_seed"] = demo_seed
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=phase1_calib_data.CONDITIONS)
    parser.add_argument("--baseline", action="store_true", help="evaluate the uncompressed model instead")
    parser.add_argument("--seed", type=int, default=0,
                         help="must match the seed used for phase1_run_freq_and_scale.py / phase1_fisher.py")
    parser.add_argument("--scale-condition", default=None,
                         help="load svd_scale (whitened/truncation-aware SVD) from this condition; "
                              "default None = plain SVD. Pass --scale-condition <same as --condition> for the "
                              "'own-language whitened SVD' full-pipeline configuration.")
    parser.add_argument("--scale-seed", type=int, default=None, help="seed for --scale-condition (defaults to --seed)")
    parser.add_argument("--delta-ratio", type=float, default=None, help="override DELTA_RATIO=0.8 (see phase1_merge_eval.py)")
    parser.add_argument("--cpu-offload-gib", type=int, default=0)
    parser.add_argument("--pp-ratio", type=float, default=None,
                         help="structured-pruning ratio (third pipeline stage); default None = off")
    parser.add_argument("--pp-condition", default=None, help="calibration condition for pruning importance calib (default: --condition)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--demo-seed", type=int, default=DEMO_SEED_DEFAULT,
                         help="fixed English-validation demonstration selection seed (must match across every checkpoint/language for a valid comparison)")
    parser.add_argument("--demos-per-label", type=int, default=DEMOS_PER_LABEL_DEFAULT,
                         help="demonstrations per category (default 2 -> 14-shot total; 0 -> 0-shot, "
                              "prompt formatting otherwise unchanged)")
    parser.add_argument("--out-suffix", default="",
                         help="suffix for the output filename, e.g. '_0shot' -> eval_sib200_0shot.json. "
                              "Default '' keeps every existing call site writing eval_sib200.json. Required "
                              "for any run that must NOT overwrite an already-frozen result in the same "
                              "out_dir (a 0-shot robustness check sits in the same directory as the 14-shot "
                              "anchor it is compared against).")
    parser.add_argument("--langs", default=None,
                         help="comma-separated subset of eng_Latn,kor_Hang,zho_Hans,swh_Latn,ben_Beng (default: all five)")
    parser.add_argument("--limit", type=int, default=None, help="subsample this many test examples per language (max 204/language)")
    parser.add_argument("--smoke", action="store_true",
                         help="write to the _smoke output filename; also sets limit=20 unless --limit is also given")
    args = parser.parse_args()
    if not args.baseline and not args.condition:
        parser.error("either --condition or --baseline is required")
    if args.scale_condition is not None and args.scale_seed is None:
        args.scale_seed = args.seed
    if args.pp_ratio is not None and args.pp_condition is None:
        args.pp_condition = args.condition

    limit = args.limit if args.limit is not None else (20 if args.smoke else None)
    langs = args.langs.split(",") if args.langs else None
    for lang in (langs or []):
        if lang not in SIB_LANGS:
            parser.error(f"--langs: {lang!r} not in {list(SIB_LANGS)}")

    print(f"[sib200-eval] {'BASELINE (uncompressed)' if args.baseline else f'condition={args.condition}'} "
          f"smoke={args.smoke} limit={limit} langs={langs or list(SIB_LANGS)}")
    model, tokenizer = load_model(cpu_gib=args.cpu_offload_gib)

    # 2026-07-31 (SIB_DIAGNOSIS.md, GPU-count root cause): device_map="auto"
    # falls back to disk offload for whatever doesn't fit in max_memory's
    # per-GPU budget, and disk-offloaded parameters were found to produce NaN
    # logprobs specifically inside score_batch() (belebele's different forward
    # call pattern apparently tolerates it; this script's does not). This
    # script is only meant to run with enough GPUs that nothing offloads --
    # fail fast and loud instead of silently producing collapsed results again.
    if hasattr(model, "hf_device_map"):
        offloaded = {k: v for k, v in model.hf_device_map.items() if v in ("disk", "cpu")}
        if offloaded:
            raise RuntimeError(
                f"model has {len(offloaded)} module(s) offloaded off-GPU (e.g. {next(iter(offloaded.items()))}) "
                "-- refusing to run. See SIB_DIAGNOSIS.md 2026-07-31: disk/cpu-offloaded execution produces "
                "NaN logprobs in score_batch(). Re-run with more GPUs (CUDA_VISIBLE_DEVICES) so the model "
                "fits fully in GPU max_memory.")

    if not args.baseline:
        scale_note = f" scale=({args.scale_condition},seed{args.scale_seed})" if args.scale_condition else " scale=none(plain SVD)"
        pp_note = f" pp_ratio={args.pp_ratio}(calib={args.pp_condition})" if args.pp_ratio is not None else " pp_ratio=off"
        print(f"[sib200-eval] merging condition={args.condition} seed={args.seed}{scale_note}{pp_note}")
        model = merge_condition(model, args.condition, args.seed, args.scale_condition, args.scale_seed,
                                 delta_ratio=args.delta_ratio)
        if args.pp_ratio is not None:
            phase1_structured_prune.apply_structured_pruning(
                model, args.pp_condition, args.seed, args.pp_ratio, tokenizer)

        # 2026-07-30 (same fix as phase1_belebele_eval.py's seed2 OOM retry): free
        # merge_condition()'s leftover intermediate tensors before the eval forward
        # pass -- this script reuses the identical load_model()/merge_condition()
        # path and was seen to run just as close to the single-GPU memory ceiling.
        gc.collect()
        torch.cuda.empty_cache()

    results = eval_sib200(model, tokenizer, batch_size=args.batch_size, limit=limit, demo_seed=args.demo_seed,
                          demos_per_label=args.demos_per_label, langs=langs)

    eval_langs = langs or list(SIB_LANGS)
    nan_langs = [lang for lang in eval_langs if results[lang]["acc"] != results[lang]["acc"]]
    if nan_langs:
        raise RuntimeError(
            f"sib200 eval produced NaN acc for {nan_langs} -- likely a silent model failure (e.g. disk-offload "
            "under host memory pressure, see phase1_merge_eval.py's NaN guard for the same failure mode). Refusing "
            "to write eval_sib200 json; let the caller retry.")

    if args.baseline:
        out_dir = RESULTS_ROOT / "baseline"
    elif args.scale_condition is not None:
        out_dir = RESULTS_ROOT / args.condition / f"seed{args.seed}" / f"scale_{args.scale_condition}_seed{args.scale_seed}"
    else:
        out_dir = RESULTS_ROOT / args.condition / f"seed{args.seed}"
    if args.delta_ratio is not None:
        out_dir = out_dir / f"delta_ratio_{args.delta_ratio}"
    if args.pp_ratio is not None:
        out_dir = out_dir / f"pp_ratio_{args.pp_ratio}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "eval_sib200_smoke" if args.smoke else "eval_sib200"
    out_path = out_dir / f"{stem}{args.out_suffix}.json"
    out_path.write_text(json.dumps({
        "protocol": "sib200_natural_continuation_v1",
        "num_fewshot": len(CATEGORY_ORDER) * args.demos_per_label,
        "demo_seed": args.demo_seed, "limit": limit, "langs": eval_langs, "results": results,
    }, indent=2))
    print(f"[sib200-eval] wrote {out_path}")


if __name__ == "__main__":
    main()
