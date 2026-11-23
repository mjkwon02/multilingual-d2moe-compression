"""Freeze the MASSIVE evaluation protocol BEFORE any model is scored.

Everything this writes -- evaluation subsets, demonstration IDs, prompt
templates and their hashes, candidate label surface forms and their order,
the score definition, seeds, the baseline gate thresholds, and the metric
definitions -- is fixed here and is never re-derived later. massive_eval.py
reads these files and refuses to run if their hashes have moved
(see massive_eval.py::load_protocol). The ordering is deliberate and is the
whole point of running this as a separate script: no compressed-model score
and no test-split label exists anywhere on disk at the time this runs.

Outputs (all under results/massive/):
    massive_data_manifest.json  dataset revision, splits, schema, license,
                                label coverage, cross-language ID checks,
                                and the FLORES-calibration contamination scan
    massive_protocol.json       the frozen protocol itself
    massive_demo_ids.json       the 8 fixed English demonstrations
    massive_dev_ids.json        the 120-item dev pilot subset
    massive_test_ids.json       the 600-item final test subset

Run:  python massive_build_protocol.py
"""
import collections
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import massive_data as M  # noqa: E402

OUT_DIR = SCRIPT_DIR.parents[1] / "results" / "downstream" / "massive_legacy"

# ---------------------------------------------------------------- subsets --
DEV_SUBSET_SEED = 42
DEV_PER_LABEL = 2            # target; 60 labels x 2 = 120 = DEV_MAX
DEV_MAX = 120
TEST_SUBSET_SEED = 42
TEST_N = 600                 # a SAMPLE of the 2974-item test split, not the full split
DEMO_SEED = 42
N_DEMOS = 8
DEMO_LANG = "EN"             # fixed English demonstrations for every eval language
DEMO_MAX_WORDS = 20          # keeps the 8-demo prefix short; applied before selection

# ------------------------------------------------------- prompt templates --
# Fixed English instruction for every eval language. Only the utterance
# changes across languages -- same rationale phase1_xnli_eval_v2.py and
# phase1_sib200_eval.py already use here (isolates the eval-language variable
# from an instruction-translation-quality variable).
INSTRUCTION = (
    "Classify the intent of the user's utterance to a voice assistant.\n"
    "Answer with exactly one intent from the following list.\n"
)
LABEL_LIST_HEADER = "\nIntents:\n"
ITEM_TEMPLATE = "\nUtterance: {utt}\nIntent:"   # completion is " <label surface>"
COMPLETION_PREFIX = " "


def build_label_block():
    return "".join(f"- {M.LABEL_SURFACE[lab]}\n" for lab in M.PROMPT_LABEL_ORDER)


def build_shared_prefix(demos):
    """The part of the prompt that is identical for every item of a given
    (prompt_id, eval language) -- instruction + label list + demonstrations.
    massive_eval.py caches exactly this span's KV, so it is defined here,
    once, rather than being re-spliced at eval time."""
    parts = [INSTRUCTION, LABEL_LIST_HEADER, build_label_block()]
    for d in demos:
        parts.append(ITEM_TEMPLATE.format(utt=d["utt"]) + COMPLETION_PREFIX + M.LABEL_SURFACE[d["intent"]] + "\n")
    return "".join(parts)


def build_item_tail(utt):
    return ITEM_TEMPLATE.format(utt=utt)


# --------------------------------------------------------------- the gate --
# §5: operational thresholds on whether the ORIGINAL model can read each
# language at all. Pre-registered here; not a significance test.
GATE = {
    "uniform_chance": 1.0 / 60,
    "margin_over_majority_pp": 5.0,
    "max_floor_risk_langs_before_hold": 2,   # >=3 of 5 floor-risk => hold the test sweep
    "abort_on_nonfinite_scores": True,
    "note": ("The 5pp margin is a pre-registered operational criterion for 'the original model can "
             "read this language on this task at all'. It is NOT a statistical guarantee and must "
             "not be reported as one."),
}


def stratified_dev(usable_by_part, data, per_label=DEV_PER_LABEL, cap=DEV_MAX, seed=DEV_SUBSET_SEED):
    """Intent-stratified dev pilot subset over the COMMON dev IDs: up to
    `per_label` items per intent, capped at `cap` total. Selection uses only
    the gold intent for stratification and a fixed RNG -- never any model
    output. Records the realized support per label either way."""
    rng = random.Random(seed)
    pool = sorted(usable_by_part["dev"], key=int)
    by_label = collections.defaultdict(list)
    for i in pool:
        by_label[data["EN"][i]["intent"]].append(i)
    chosen, support = [], {}
    for lab in M.PROMPT_LABEL_ORDER:
        avail = by_label.get(lab, [])
        take = sorted(rng.sample(avail, min(per_label, len(avail))), key=int)
        support[lab] = {"available_in_dev": len(avail), "selected": len(take)}
        chosen.extend(take)
    rng.shuffle(chosen)
    chosen = sorted(chosen[:cap], key=int)
    return chosen, support


def uniform_test(usable_by_part, data, n=TEST_N, seed=TEST_SUBSET_SEED):
    """Uniform random sample of the COMMON test IDs. No stratification and no
    performance-dependent selection of any kind; label coverage is whatever
    it turns out to be and is reported, never topped up afterwards."""
    pool = sorted(usable_by_part["test"], key=int)
    ids = sorted(random.Random(seed).sample(pool, n), key=int)
    cov = collections.Counter(data["EN"][i]["intent"] for i in ids)
    coverage = {lab: cov.get(lab, 0) for lab in M.PROMPT_LABEL_ORDER}
    return ids, coverage


def pick_demos(usable_by_part, data, n=N_DEMOS, seed=DEMO_SEED, max_words=DEMO_MAX_WORDS):
    """`n` fixed demonstrations from the English TRAIN split only, distinct
    intents, identical for every eval language and every checkpoint."""
    rng = random.Random(seed)
    pool = [i for i in sorted(usable_by_part["train"], key=int)
            if len(data[DEMO_LANG][i]["utt"].split()) <= max_words]
    rng.shuffle(pool)
    chosen, seen = [], set()
    for i in pool:
        lab = data[DEMO_LANG][i]["intent"]
        if lab in seen:
            continue
        seen.add(lab)
        chosen.append(i)
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise RuntimeError(f"only {len(chosen)} distinct-intent demonstrations available under {max_words} words")
    return chosen


def contamination_scan(data, dev_ids, test_ids):
    """§3: compare this project's FLORES calibration text against the MASSIVE
    dev/test utterances actually evaluated.

    Scope note recorded in the output: the comparison uses each language's
    FULL FLORES-200 pool (the superset from which phase1_calib_data._pool_for
    carves every condition's arm-A/arm-B sentences, and hence a superset of
    the realized 512-token calibration windows). A superset is the
    conservative direction -- it can only over-report overlap.

    MASSIVE ids and FLORES row indices live in unrelated id spaces (SLURP-
    derived utterances vs FLORES-200 sentence indices), so no source-ID join
    exists; this is a string-level scan, at three tightness levels, and short
    formulaic utterances are bucketed separately rather than being counted as
    document-level leakage."""
    import data_utils
    from phase1_calib_data import SOURCE, SPLIT, COL_PREFIX

    out = {
        "flores_pool": {"source": SOURCE, "split": SPLIT, "column_prefix": COL_PREFIX,
                        "scope": "full per-language FLORES-200 pool (superset of realized calibration windows)"},
        "id_join": "not possible: MASSIVE ids (SLURP-derived) and FLORES-200 row indices are unrelated id spaces",
        "per_language": {},
    }
    eval_ids = {"dev": list(dev_ids), "test": list(test_ids)}
    for pl in M.LOCALES:
        flores = data_utils.load_full_column(SOURCE, SPLIT, COL_PREFIX, M.FLORES_CODE[pl])
        levels = {
            "exact": {s for s in flores},
            "nfc": {M.nfc(s) for s in flores},
            "loose_nfc_casefold_ws": {M.loose_key(s) for s in flores},
        }
        lang_out = {"flores_code": M.FLORES_CODE[pl], "n_flores_sentences": len(flores)}
        for split_name, ids in eval_ids.items():
            utts = [data[pl][i]["utt"] for i in ids]
            hits = {}
            for level, keyset in levels.items():
                fn = {"exact": lambda s: s, "nfc": M.nfc, "loose_nfc_casefold_ws": M.loose_key}[level]
                matched = [(i, u) for i, u in zip(ids, utts) if fn(u) in keyset]
                # A MASSIVE utterance is a short assistant command; a FLORES
                # sentence is a Wikipedia-style sentence. Anything matching at
                # <= 5 words is a formulaic collision, not evidence that a
                # calibration document was memorized.
                short = [(i, u) for i, u in matched if len(u.split()) <= 5]
                hits[level] = {
                    "n_matched": len(matched),
                    "n_short_formulaic_leq5_words": len(short),
                    "n_substantive": len(matched) - len(short),
                    "examples": [{"id": i, "utt": u} for i, u in matched[:10]],
                }
            lang_out[split_name] = {"n_items": len(ids), "matches": hits}
        out["per_language"][pl] = lang_out
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = M.load_all()
    manifest, usable, by_part = M.verify_manifest(data)

    dev_ids, dev_support = stratified_dev(by_part, data)
    test_ids, test_coverage = uniform_test(by_part, data)
    demo_ids = pick_demos(by_part, data)

    # No evaluated item may ever appear as a demonstration, and dev must not
    # bleed into test -- both are guaranteed by split disjointness above, but
    # asserted explicitly because that is the cheap half of the check.
    assert not (set(demo_ids) & (set(dev_ids) | set(test_ids)))
    assert not (set(dev_ids) & set(test_ids))

    demos = [data[DEMO_LANG][i] for i in demo_ids]
    shared_p0 = build_shared_prefix([])
    shared_p8 = build_shared_prefix(demos)

    # §2/§4 leak scan over the frozen prompts. The guarantee is structural
    # first: an evaluated item contributes to the prompt ONLY through
    # ITEM_TEMPLATE.format(utt=...), a template with no label slot, so no
    # gold intent can be appended by construction. Two further checks catch
    # the ways a gold hint could still ride in on the text itself.
    #
    # NOTE a naive `row["annot_utt"] in prompt` test is useless here and was
    # discarded: MASSIVE sets annot_utt == utt verbatim whenever an utterance
    # has no slots, so that test flags ~2.4k slot-free items whose "leak" is
    # nothing but the utterance the model is supposed to read. Only a
    # *slot-annotated* annot_utt (annot_utt != utt) carries gold information.
    leak = []
    for pid, shared in (("P0", shared_p0), ("P8", shared_p8)):
        for split_name, ids in (("dev", dev_ids), ("test", test_ids)):
            for i in ids:
                for pl in M.LOCALES:
                    row = data[pl][i]
                    tail = build_item_tail(row["utt"])
                    why = None
                    if tail != ITEM_TEMPLATE.format(utt=row["utt"]):
                        why = "item tail not built solely from utt"
                    elif tail.rstrip() != tail.rstrip().split("Intent:")[0] + "Intent:":
                        why = "text present after the 'Intent:' cue"
                    elif row["annot_utt"] != row["utt"] and row["annot_utt"] in shared + tail:
                        why = "slot-annotated annot_utt present in prompt"
                    if why:
                        leak.append({"prompt": pid, "split": split_name, "id": i, "lang": pl, "why": why})
    if leak:
        raise RuntimeError(f"gold information leaked into {len(leak)} prompt(s), e.g. {leak[:3]}")
    # Demonstrations legitimately carry their own labels (that is what makes
    # P8 few-shot), but must not carry slot annotations either.
    demo_leak = [d["id"] for d in demos if d["annot_utt"] != d["utt"] and d["annot_utt"] in shared_p8]
    if demo_leak:
        raise RuntimeError(f"slot-annotated annot_utt leaked into the P8 demonstration block: {demo_leak}")

    contam = contamination_scan(data, dev_ids, test_ids)
    manifest["contamination_scan_vs_flores_calibration"] = contam
    manifest["subset_note"] = (
        f"The final test evaluation uses a {TEST_N}-item RANDOM SAMPLE of the {len(by_part['test'])}-item "
        "common test split, not the full test split. Every number derived from it must be reported as a "
        "sample estimate.")

    protocol = {
        "protocol_id": "massive_intent60_base_lm_v1",
        "frozen_before": "any compressed-model score; any inspection of test-split labels",
        "task": manifest["task"],
        "dataset": {k: manifest[k] for k in ("dataset", "source_url", "tarball_sha256", "version", "license",
                                             "homepage", "label_set_provenance")},
        "languages": {pl: {"locale": M.LOCALES[pl], "flores_code": M.FLORES_CODE[pl],
                           "own_language_condition": M.OWN_LANG_CONDITION[pl]} for pl in M.LOCALES},
        "language_order": list(M.LOCALES),
        "labels": {
            "n": 60,
            "order": M.PROMPT_LABEL_ORDER,
            "surface_transform": "intent.replace('_', ' ')  [pre-registered; injective; no generated definitions]",
            "surface_map": M.LABEL_SURFACE,
            "all_candidates_scored": True,
        },
        "prompts": {
            "chat_template_applied": False,
            "chat_template_note": "base (never instruction-tuned) model; no chat template is imposed",
            "instruction": INSTRUCTION,
            "label_list_header": LABEL_LIST_HEADER,
            "item_template": ITEM_TEMPLATE,
            "completion_prefix": COMPLETION_PREFIX,
            "P0": {"n_demos": 0, "shared_prefix_sha256": M.sha256_text(shared_p0),
                   "shared_prefix_chars": len(shared_p0)},
            "P8": {"n_demos": N_DEMOS, "demo_lang": DEMO_LANG, "demo_seed": DEMO_SEED,
                   "shared_prefix_sha256": M.sha256_text(shared_p8),
                   "shared_prefix_chars": len(shared_p8)},
            "selection_rule": ("the single setting with the higher BASE-model dev macro-average accuracy over "
                               "the five languages; exact tie -> P0. No prompt outside {P0, P8} is explored, "
                               "and the choice is made and recorded before any compressed checkpoint is scored."),
        },
        "scoring": {
            "primary": "mean_token_logprob",
            "primary_definition": ("token-averaged conditional log-likelihood of the label completion, i.e. "
                                   "sum_t log p(c_t | prompt, c_<t) / n_completion_tokens"),
            "also_recorded": ["sum_token_logprob", "n_completion_tokens"],
            "eos_policy": "EOS is NOT scored and is not appended to any candidate completion",
            "context_tokens_scored": False,
            "prediction_rule": "argmax over all 60 candidates of the primary score",
            "frozen_note": ("mean is the pre-registered primary. sum is recorded for transparency only and "
                            "must not be promoted to primary after seeing results."),
        },
        "subsets": {
            "dev_pilot": {"n": len(dev_ids), "seed": DEV_SUBSET_SEED, "per_label_target": DEV_PER_LABEL,
                          "cap": DEV_MAX, "selection": "intent-stratified over common dev IDs",
                          "same_ids_across_languages": True},
            "test": {"n": len(test_ids), "seed": TEST_SUBSET_SEED,
                     "selection": "uniform random over common test IDs; no stratification",
                     "same_ids_across_languages": True,
                     "is_sample_not_full_split": True,
                     "full_common_test_size": len(by_part["test"])},
        },
        "metrics": {
            "accuracy": "exact match of argmax label to gold intent",
            "confusion_matrix": "fixed 60x60 in labels.order, for every cell",
            "macro_f1_support": "macro-F1 averaged over labels with non-zero support in the test subset",
            "macro_f1_fixed60": "macro-F1 over all 60 labels with zero_division=0",
            "collapse_diagnostics": ["top_predicted_label_fraction", "predicted_label_entropy_nats"],
            "collapse_threshold_note": ("SIB-200's acc<0.25 collapse criterion is NOT reused here; it was "
                                        "derived from a 7-way task with a ~25% majority floor and does not "
                                        "transfer to a 60-way task."),
            "retention": "compressed_acc / base_acc, same language",
            "drop_pp": "100 * (base_acc - compressed_acc)",
            "seed_aggregation": "mean +/- sample std (ddof=1) over the 3 calibration seeds",
        },
        "primary_contrast": {
            "definition_pp": ["Balanced - own_language", "Balanced - English"],
            "balanced_condition": "mixed_5lang",
            "own_language_definition": "the condition whose calibration language equals the evaluation language",
            "uncertainty": ("paired bootstrap over the shared sample IDs: resample IDs jointly across all "
                            "conditions, recompute each seed's contrast on the resampled IDs, then average "
                            "over the 3 seeds within each replicate. The CI is conditional on the three "
                            "observed calibration seeds and does not treat them as 3x the test sample."),
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 12345,
            "multiplicity": ("default reporting is effect size + CI + per-seed direction. No multiple-comparison "
                             "p-value claim is made; doing so would require a pre-defined comparison family "
                             "and correction, which this protocol does not register."),
        },
        "gate": GATE,
        "model_grid": {
            "baseline": "deepseek-ai/deepseek-moe-16b-base, uncompressed, evaluated once per language",
            "conditions": ["english_only", "chinese_only", "korean_only", "swahili_only", "bengali_only",
                           "mixed_5lang"],
            "seeds": [0, 1, 2],
            "eval_languages": list(M.LOCALES),
            "cells": "5 baseline + 6*3*5 = 90 compressed = 95 model-language evaluation units",
            "baseline_replication_note": "the single baseline is never duplicated as if it were 3 seed results",
            "compression_config": {
                "delta_ratio": 0.8, "share_ratio": 1.0, "merge_method": "fisher",
                "svd_scale": "own condition/seed (whitened SVD), i.e. --scale-condition == --condition",
                "structured_pruning": "OFF (pp_ratio=None)",
                "source": "existing Phase-1 artifacts; nothing is recalibrated, retrained or re-swept",
            },
            "not_in_this_run": ["balanced_tok_*", "placebo _b arms", "Qwen1.5-MoE", "other delta_ratio",
                                "structured pruning on"],
        },
        "subset_seeds": {"dev": DEV_SUBSET_SEED, "test": TEST_SUBSET_SEED, "demo": DEMO_SEED},
    }

    (OUT_DIR / "massive_data_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (OUT_DIR / "massive_protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
    (OUT_DIR / "massive_demo_ids.json").write_text(json.dumps({
        "seed": DEMO_SEED, "lang": DEMO_LANG, "split": "train", "max_words": DEMO_MAX_WORDS,
        "n": len(demo_ids), "ids": demo_ids,
        "demonstrations": [{"order": k, "id": i, "intent": data[DEMO_LANG][i]["intent"],
                            "surface": M.LABEL_SURFACE[data[DEMO_LANG][i]["intent"]],
                            "utt": data[DEMO_LANG][i]["utt"]} for k, i in enumerate(demo_ids)],
    }, indent=2, ensure_ascii=False))
    (OUT_DIR / "massive_dev_ids.json").write_text(json.dumps({
        "seed": DEV_SUBSET_SEED, "n": len(dev_ids), "ids": dev_ids,
        "per_label_support": dev_support,
        "labels_with_fewer_than_target": {l: s for l, s in dev_support.items() if s["selected"] < DEV_PER_LABEL},
    }, indent=2, ensure_ascii=False))
    (OUT_DIR / "massive_test_ids.json").write_text(json.dumps({
        "seed": TEST_SUBSET_SEED, "n": len(test_ids), "ids": test_ids,
        "label_coverage": test_coverage,
        "n_labels_with_support": sum(1 for v in test_coverage.values() if v > 0),
        "labels_absent": [l for l, v in test_coverage.items() if v == 0],
        "note": "absent labels are NOT backfilled later; fixed-60 macro-F1 quantifies their effect",
    }, indent=2, ensure_ascii=False))

    print(f"[massive-protocol] wrote 5 files to {OUT_DIR}")
    print(f"[massive-protocol] usable ids={len(usable)} excluded={manifest['n_excluded_total']} "
          f"(train/dev/test = {manifest['usable_split_sizes']})")
    print(f"[massive-protocol] dev pilot n={len(dev_ids)}; labels below target: "
          f"{sum(1 for s in dev_support.values() if s['selected'] < DEV_PER_LABEL)}")
    print(f"[massive-protocol] test n={len(test_ids)} of {len(by_part['test'])} (SAMPLE); "
          f"labels with support={sum(1 for v in test_coverage.values() if v > 0)}/60")
    print(f"[massive-protocol] P0 shared prefix {len(shared_p0)} chars sha={M.sha256_text(shared_p0)[:12]}")
    print(f"[massive-protocol] P8 shared prefix {len(shared_p8)} chars sha={M.sha256_text(shared_p8)[:12]}")
    for pl, lo in contam["per_language"].items():
        for sp in ("dev", "test"):
            h = lo[sp]["matches"]
            print(f"[massive-protocol] contamination {pl}/{sp}: exact={h['exact']['n_matched']} "
                  f"nfc={h['nfc']['n_matched']} loose={h['loose_nfc_casefold_ws']['n_matched']} "
                  f"(substantive loose={h['loose_nfc_casefold_ws']['n_substantive']})")


if __name__ == "__main__":
    main()
