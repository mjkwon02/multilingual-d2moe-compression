"""Light calibration-overlap check for the revised 200-item MASSIVE subset.

SCOPE IS DELIBERATELY SMALL (§2/§4 of the revision). The full audit already
ran -- analysis/data_audit_0916/ -- and is NOT re-executed here. This only
answers: do any of the 200 evaluated MASSIVE utterances appear verbatim in the
calibration text that the two conditions in THIS run actually consume?

Two things the audit changes about how the check is framed:

  * get_expert_freq.py consumes the ENTIRE single-language pool (952 sentences),
    not merely the 64 sampled 512-token windows. So "calibration text actually
    consumed" means the whole condition pool, and describing calibration
    exposure via the windows alone would understate it. The pool is therefore
    what is compared against, which is also the conservative direction.
  * MASSIVE (SLURP-derived) and FLORES-200 share no provenance ID space, so no
    ID-level join exists and none is assumed. This is a string check only.

An exact calibration match excludes that source ID in ALL five languages and it
is replaced by the next ID in hash order; short formulaic collisions are
flagged separately and are NOT treated as document-level leakage. Passing this
check does not demonstrate document-level independence, and says nothing about
pretraining contamination.
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import massive_data as M  # noqa: E402
import phase1_calib_data as C  # noqa: E402
from massive_revised_protocol import canonical_source_id, id_hash, TEST_N  # noqa: E402

NEW_DIR = SCRIPT_DIR.parents[1] / "results" / "downstream" / "massive_revised"
AUDIT_DIR = SCRIPT_DIR.parents[1] / "results" / "source_data" / "data_exposure_audit"
CONDITIONS = ["english_only", "mixed_5lang"]
SEEDS = [0, 1, 2]


def norm_ws(s):
    return " ".join(M.nfc(s).split())


def main():
    data = M.load_all()
    tmeta = json.loads((NEW_DIR / "test_ids.json").read_text())
    test_ids = list(tmeta["ids"])
    old_test = json.loads((SCRIPT_DIR.parents[1] / "results" / "downstream" / "massive_legacy" / "massive_test_ids.json").read_text())
    pool_all = list(old_test["ids"])

    # ---- calibration text actually consumed by the two conditions in this run ----
    consumed_exact, consumed_ws = set(), set()
    per_cond = {}
    for cond in CONDITIONS:
        for seed in SEEDS:
            sents = C.build_condition_sentences(cond, seed)
            per_cond[f"{cond}/seed{seed}"] = {"n_sentences": len(sents),
                                              "n_chars": sum(len(s) for s in sents)}
            for s in sents:
                consumed_exact.add(s)
                consumed_ws.add(norm_ws(s))
    print(f"[overlap] calibration sentences pooled over {len(per_cond)} condition-seeds: "
          f"{len(consumed_exact)} distinct exact strings")

    def scan(ids):
        hits = []
        for i in ids:
            for pl in M.LOCALES:
                u = data[pl][i]["utt"]
                if u in consumed_exact:
                    hits.append({"id": i, "lang": pl, "level": "exact", "utt": u,
                                 "n_words": len(u.split())})
                elif norm_ws(u) in consumed_ws:
                    hits.append({"id": i, "lang": pl, "level": "nfc_ws", "utt": u,
                                 "n_words": len(u.split())})
        return hits

    hits = scan(test_ids)
    # exclusion rule: an EXACT calibration match removes that source ID in all
    # five languages; replace from hash order. Short formulaic collisions are
    # recorded but not excluded.
    excluded = sorted({h["id"] for h in hits if h["level"] == "exact" and h["n_words"] > 5})
    short_flagged = sorted({h["id"] for h in hits if h["n_words"] <= 5})
    final_ids = list(test_ids)
    replacements = []
    if excluded:
        remaining = [i for i in sorted(pool_all, key=lambda i: (id_hash(i), canonical_source_id(i)))
                     if i not in set(test_ids)]
        final_ids = [i for i in test_ids if i not in set(excluded)]
        for i in remaining:
            if len(final_ids) >= TEST_N:
                break
            if scan([i]):
                continue
            final_ids.append(i)
            replacements.append(i)
        final_ids = sorted(final_ids, key=lambda i: (id_hash(i), canonical_source_id(i)))

    out = {
        "scope": ("light check only; the full audit in analysis/data_audit_0916/ was NOT re-run "
                  "(see that tree for FLORES/SIB/Belebele/XL-Sum exposure)"),
        "audit_source": {
            "dir": str(AUDIT_DIR.relative_to(SCRIPT_DIR.parent)),
            "files_cited": ["artifact_manifest.json", "data_audit_report.md",
                            "calibration_reconstruction.json", "clean_eval_ids.json"],
        },
        "calibration_text_definition": ("the FULL condition pool, because get_expert_freq.py consumes the "
                                       "entire single-language pool (audit finding), not only the 64 sampled "
                                       "512-token windows"),
        "conditions_checked": per_cond,
        "id_join_possible": False,
        "id_join_note": "MASSIVE (SLURP-derived) and FLORES-200 have unrelated provenance ID spaces",
        "levels": ["exact", "NFC + whitespace-normalized"],
        "n_items_scanned": len(test_ids),
        "n_language_instances_scanned": len(test_ids) * len(M.LOCALES),
        "hits": hits,
        "n_hits_total": len(hits),
        "exclusion_rule": ("an EXACT match longer than 5 words excludes that source ID in all five languages; "
                           "replaced by the next ID in hash order that is itself clean"),
        "n_excluded": len(excluded),
        "excluded_ids": excluded,
        "n_replacements": len(replacements),
        "replacement_ids": replacements,
        "short_formulaic_flagged_not_excluded": short_flagged,
        "final_n": len(final_ids),
        "limits": ("Passing this check does not establish document-level independence between MASSIVE and the "
                   "calibration corpus, and establishes nothing about pretraining contamination."),
    }
    (NEW_DIR / "calibration_overlap_check.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    if final_ids != test_ids:
        tmeta["ids"] = final_ids
        tmeta["id_hashes"] = {i: id_hash(i)[:16] for i in final_ids}
        tmeta["overlap_exclusions_applied"] = {"excluded": excluded, "replacements": replacements}
        (NEW_DIR / "test_ids.json").write_text(json.dumps(tmeta, indent=2, ensure_ascii=False))
        print(f"[overlap] excluded {len(excluded)} id(s), added {len(replacements)} replacement(s)")
    print(f"[overlap] hits: total={len(hits)} exact>5w={len(excluded)} short<=5w={len(short_flagged)}")
    print(f"[overlap] final test n={len(final_ids)}")
    print(f"[overlap] wrote {NEW_DIR / 'calibration_overlap_check.json'}")


if __name__ == "__main__":
    main()
