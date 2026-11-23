"""Build the REVISED (2026-09-16) MASSIVE protocol: 200 test items, English
vs Balanced-character only, seeds 0/1/2, GPU 2-3 single stream.

This supersedes the 600-item / 6-condition grid frozen earlier today. It does
NOT overwrite that earlier protocol: everything lands in a separate
analysis/massive_0916_revised/ tree and protocol_amendment.md records what
changed, when, and why. The earlier results/massive/ artifacts stay exactly
as they were.

Test selection is a pure function of the source IDs -- SHA-256 of
"massive-rebuttal-v2|42|<canonical_source_id>", ascending, first 200 -- drawn
from the ALREADY-FROZEN 600-item common-test manifest. No label, prediction or
score enters the choice, and the hash makes the subset verifiable by anyone
from the ID list alone.

Run: python massive_revised_protocol.py
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import massive_data as M  # noqa: E402

OLD_DIR = SCRIPT_DIR.parents[1] / "results" / "downstream" / "massive_legacy"
NEW_DIR = SCRIPT_DIR.parents[1] / "results" / "downstream" / "massive_revised"

HASH_PREFIX = "massive-rebuttal-v2|42|"
TEST_N = 200
DEV_MAX = 60
CONDITIONS = {"english_only": "English", "mixed_5lang": "Balanced-character"}
SEEDS = [0, 1, 2]


def canonical_source_id(raw):
    """Fixed representation rule: a purely numeric MASSIVE id is canonicalized
    as a decimal string with no leading zeros; anything else keeps its original
    string form. Recorded rather than assumed because the hash ordering -- and
    therefore the whole subset -- depends on this exact spelling."""
    s = str(raw)
    if s.isdigit():
        return str(int(s))
    return s


def id_hash(raw):
    return hashlib.sha256((HASH_PREFIX + canonical_source_id(raw)).encode("utf-8")).hexdigest()


def main():
    NEW_DIR.mkdir(parents=True, exist_ok=True)
    data = M.load_all()

    old_test = json.loads((OLD_DIR / "massive_test_ids.json").read_text())
    old_dev = json.loads((OLD_DIR / "massive_dev_ids.json").read_text())
    old_protocol = json.loads((OLD_DIR / "massive_protocol.json").read_text())

    # ---- candidate pool: the already-frozen 600 common-test IDs (§4) ----
    candidates = list(old_test["ids"])
    # every candidate must still satisfy the cross-language conditions
    for i in candidates:
        intents = {data[pl][i]["intent"] for pl in M.LOCALES}
        parts = {data[pl][i]["partition"] for pl in M.LOCALES}
        if len(intents) != 1 or parts != {"test"}:
            raise RuntimeError(f"candidate {i} is not a clean common test ID: intents={intents} parts={parts}")
    nonnum = [i for i in candidates if not str(i).isdigit()]
    dup_canon = [k for k, v in collections.Counter(canonical_source_id(i) for i in candidates).items() if v > 1]
    if dup_canon:
        raise RuntimeError(f"canonicalization collapses distinct ids: {dup_canon[:5]}")

    ranked = sorted(candidates, key=lambda i: (id_hash(i), canonical_source_id(i)))
    test_ids = ranked[:TEST_N]

    cov = collections.Counter(data["EN"][i]["intent"] for i in test_ids)
    coverage = {lab: cov.get(lab, 0) for lab in M.PROMPT_LABEL_ORDER}
    lengths = {pl: sorted(len(data[pl][i]["utt"].split()) for i in test_ids) for pl in M.LOCALES}

    # ---- dev: <=60 IDs, one per intent, taken from the frozen 118-item dev
    # subset in its existing frozen order. Derived from the already-frozen set
    # (never re-drawn), so it cannot be a fresh draw chosen to look good, and
    # no dev accuracy had been produced when this was built. ----
    dev_by_intent = {}
    for i in old_dev["ids"]:
        lab = data["EN"][i]["intent"]
        dev_by_intent.setdefault(lab, i)
    dev_ids = [dev_by_intent[lab] for lab in M.PROMPT_LABEL_ORDER if lab in dev_by_intent][:DEV_MAX]

    assert not (set(dev_ids) & set(test_ids))
    demo_ids = set(json.loads((OLD_DIR / "massive_demo_ids.json").read_text())["ids"])
    assert not (demo_ids & (set(dev_ids) | set(test_ids)))

    (NEW_DIR / "test_ids.json").write_text(json.dumps({
        "n": len(test_ids),
        "selection_rule": f'SHA-256("{HASH_PREFIX}<canonical_source_id>") ascending, first {TEST_N}',
        "canonical_id_rule": "numeric ids -> decimal string without leading zeros; others -> original string",
        "n_non_numeric_ids_in_pool": len(nonnum),
        "candidate_pool": f"the frozen {len(candidates)}-item common-test manifest (results/massive/massive_test_ids.json)",
        "candidate_pool_size": len(candidates),
        "independent_of_labels_and_scores": True,
        "same_ids_across_all_five_languages": True,
        "ids": test_ids,
        "id_hashes": {i: id_hash(i)[:16] for i in test_ids},
        "label_coverage": coverage,
        "n_labels_with_support": sum(1 for v in coverage.values() if v > 0),
        "labels_absent": [l for l, v in coverage.items() if v == 0],
        "utt_word_length_by_language": {pl: {"min": v[0], "median": v[len(v) // 2], "max": v[-1]}
                                        for pl, v in lengths.items()},
        "note": ("A 200-item SAMPLE of MASSIVE test, not the full test split. accuracy is the primary metric; "
                 "macro-F1 is secondary because many of the 60 intents have little or no support at n=200."),
    }, indent=2, ensure_ascii=False))

    (NEW_DIR / "dev_ids.json").write_text(json.dumps({
        "n": len(dev_ids), "max_allowed": DEV_MAX,
        "selection_rule": ("one item per intent, taken in frozen order from the earlier 118-item dev subset "
                           "(results/massive/massive_dev_ids.json); not a fresh draw"),
        "ids": dev_ids,
        "intents_covered": len(dev_ids),
        "intent_absent_from_dev_split": [l for l in M.PROMPT_LABEL_ORDER if l not in dev_by_intent],
        "no_dev_accuracy_existed_when_selected": True,
    }, indent=2, ensure_ascii=False))

    protocol = {
        "protocol_id": "massive_intent60_base_lm_revised200_v1",
        "revision_date": "2026-09-16",
        "supersedes": old_protocol["protocol_id"],
        "question": ("Does Balanced-character calibration show the same performance preservation and "
                     "per-language trade-off against English-only on MASSIVE, a downstream source "
                     "independent of FLORES?"),
        "dataset": old_protocol["dataset"],
        "languages": old_protocol["languages"],
        "language_order": old_protocol["language_order"],
        "labels": {k: old_protocol["labels"][k] for k in ("n", "order", "surface_transform", "all_candidates_scored")},
        "label_surface_map_sha256": M.sha256_text(json.dumps(M.LABEL_SURFACE, sort_keys=True)),
        "prompts": {
            "candidates": ["P0", "P8"],
            "order": "P0 evaluated first; P8 only if P0 fails the baseline gate",
            "P0_shared_prefix_sha256": old_protocol["prompts"]["P0"]["shared_prefix_sha256"],
            "P8_shared_prefix_sha256": old_protocol["prompts"]["P8"]["shared_prefix_sha256"],
            "chat_template_applied": False,
        },
        "scoring": old_protocol["scoring"],
        "grid": {
            "baseline": "DeepSeek-MoE-16B base, uncompressed, once per language",
            "conditions": CONDITIONS,
            "seeds": SEEDS,
            "eval_languages": list(M.LOCALES),
            "model_loads": "1 baseline + 6 compressed = 7",
            "test_items_per_language": TEST_N,
            "total_item_scorings": 7 * 5 * TEST_N,
            "compression": {"delta_ratio": 0.8, "share_ratio": 1.0, "merge_method": "fisher",
                            "svd_scale": "own condition/seed (whitened SVD)",
                            "structured_pruning": "OFF",
                            "source": "existing Phase-1 stats only; no new calibration/Fisher/whitening"},
            "execution_order": ["baseline", "seed0: english_only -> mixed_5lang",
                               "seed1: english_only -> mixed_5lang", "seed2: english_only -> mixed_5lang"],
            "not_in_this_run": ["Balanced-token", "ZH/KO/SW/BN own-language", "placebo arms", "Qwen1.5-MoE",
                                "other delta_ratio", "structured pruning on", "new compression algorithms"],
        },
        "resources": {"physical_gpus": [2, 3], "streams": 1, "four_gpu_use": False,
                      "merge_serialized_by": "/tmp/phase1_merge_eval.lock (global; ~62GB CPU RAM per merge)"},
        "budget": {"wall_clock_hours": 12, "excludes": "time waiting for exclusive resources",
                   "validation_probe_pilot_hours_max": 2,
                   "gpu_hours_cap_equivalent": 24},
        "primary_metrics": {
            "absolute_accuracy_pct": "per language x condition x seed",
            "primary_contrast": "Balanced-character MINUS English, same seed, in pp",
            "not_reported": "Balanced minus own-language (those models are not evaluated in this run)",
            "macro_accuracy_and_worst_language": "over the same five languages, baseline in the same table",
            "drop_pp_vs_baseline": True,
        },
        "uncertainty": {
            "paired_bootstrap_replicates": 2000, "bootstrap_seed": 42,
            "resampling_unit": ("source IDs, resampled jointly across the 3 calibration seeds AND the 5 "
                                "languages, preserving translation correspondence and model pairing"),
            "ci_meaning": "test-sampling uncertainty conditional on the three observed calibration seeds",
            "seed_variability_reported_separately": "sample std (ddof=1) over 3 seeds",
            "no_equivalence_claims": "a wide CI is reported as uncertain, never as equivalence",
        },
        "gate": {
            "margin_over_majority_pp": 5.0,
            "floor_risk_langs_to_hold": 3,
            "uniform_chance": 1 / 60,
            "note": "operational cost-control criterion, not a significance test",
        },
        "secondary": ["macro_f1_support", "macro_f1_fixed60", "label_support", "top_label_share",
                      "predicted_label_entropy", "confusion_matrix", "retention (limited interpretation)"],
        "explicitly_not_imported": ["SIB-200 collapse threshold (7-way, ~25% floor)",
                                    "FLORES placebo-derived noise floor (the audit shows the placebo "
                                    "non-overlap premise does not hold)"],
    }
    (NEW_DIR / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False))

    print(f"[revised] test n={len(test_ids)} from pool of {len(candidates)}")
    print(f"[revised] label support: {protocol['primary_metrics'] and sum(1 for v in coverage.values() if v>0)}/60 "
          f"labels present; absent={len([1 for v in coverage.values() if v==0])}")
    print(f"[revised] support distribution: {dict(sorted(collections.Counter(coverage.values()).items()))}")
    print(f"[revised] dev n={len(dev_ids)} (one per intent, from the frozen 118)")
    for pl in M.LOCALES:
        v = lengths[pl]
        print(f"[revised]   {pl} utt words: min={v[0]} median={v[len(v)//2]} max={v[-1]}")
    print(f"[revised] first 5 test ids by hash: {test_ids[:5]}")
    print(f"[revised] wrote protocol.json / test_ids.json / dev_ids.json to {NEW_DIR}")


if __name__ == "__main__":
    main()
