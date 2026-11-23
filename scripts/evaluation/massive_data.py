"""MASSIVE (Amazon, FitzGerald et al. 2022) intent-classification data access
for this project's cross-domain downstream check.

WHY A NEW MODULE, NOT A REUSE. Every existing downstream evaluator here
(phase1_sib200_eval.py, phase1_belebele_eval.py, phase1_xnli_eval*.py) is
built on `datasets.load_dataset(<hub id>, <flores-200 config code>)` and on
single-token candidate scoring. MASSIVE matches neither premise:

  1. Source. It is NOT FLORES-derived. SIB-200 is literally FLORES-200
     sentences re-annotated with topics, and Belebele's passages are FLORES
     passages, so both share their *text* with this project's calibration
     pool's source corpus. MASSIVE is SLURP-derived (Bastianelli et al.
     2020) virtual-assistant utterances, localized by native speakers --
     a genuinely different domain and provenance. That is the entire point
     of running it (Limitations pp.8-9: "cross-domain downstream
     unvalidated").

  2. Loading. AmazonScience/massive on the Hub is a *loading script*
     (massive.py), and this env's `datasets` is 3.2.0, which removed dataset
     scripts entirely. So we go to the script's own canonical source instead:
     the official S3 tarball it downloads (_URL1 in massive.py), pinned by
     sha256 here. That is strictly more faithful than the Hub's auto-converted
     parquet branch, and it is the artifact the data card, the GitHub repo
     (alexa/massive) and the paper all point at.

  3. Labels. 60 intents, and their surface forms are 2-5 tokens for this
     tokenizer -- not the single-token candidates score_batch() (reused by
     xnli_v2/sib200) hard-assumes. massive_eval.py therefore implements real
     multi-token completion scoring rather than importing that helper.

The 60-intent list is NOT re-derived from the data: it is transcribed from
the official loader's `_INTENTS` constant and then *checked* against the
data (see verify_manifest), so a silent upstream label-set change surfaces
as a failure instead of as a quietly different task.
"""
import collections
import hashlib
import json
import unicodedata
from pathlib import Path

# Official v1.1 release tarball, i.e. massive.py's `_URL1`, downloaded
# 2026-09-16 and pinned. v1.1 (not the paper's v1.0) is what the Hub's
# per-locale configs serve; it adds ca-ES, so it has 52 locales where the
# paper says 51. None of our five locales differ between 1.0 and 1.1, but
# the snapshot is pinned rather than assumed -- see 00_docs on version drift.
MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
MASSIVE_TARBALL_SHA256 = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
MASSIVE_VERSION = "1.1"
MASSIVE_LICENSE = "CC BY 4.0"
MASSIVE_ROOT = Path("data/massive/1.1/data")

# Paper-notation language order EN/ZH/KO/SW/BN (00_docs 논문 표기 통일 규약)
# -> MASSIVE locale. Kept as an ordered dict so every table, figure and
# confusion-matrix axis in this task inherits that order for free.
LOCALES = {
    "EN": "en-US",
    "ZH": "zh-CN",
    "KO": "ko-KR",
    "SW": "sw-KE",
    "BN": "bn-BD",
}
# This project's FLORES-200 config codes for the same five languages, so
# MASSIVE results can be joined to eval_ppl.json / eval_sib200.json rows.
FLORES_CODE = {
    "EN": "eng_Latn", "ZH": "zho_Hans", "KO": "kor_Hang", "SW": "swh_Latn", "BN": "ben_Beng",
}
# Calibration condition whose calibration language is this eval language --
# the "own-language" cell of §7's primary contrast.
OWN_LANG_CONDITION = {
    "EN": "english_only", "ZH": "chinese_only", "KO": "korean_only",
    "SW": "swahili_only", "BN": "bengali_only",
}

# Transcribed verbatim from massive.py's `_INTENTS` (AmazonScience/massive
# @ ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740). Order preserved as-is for
# provenance; PROMPT_LABEL_ORDER below is what the prompt/confusion matrix
# actually use.
OFFICIAL_INTENTS = [
    'datetime_query', 'iot_hue_lightchange', 'transport_ticket', 'takeaway_query', 'qa_stock',
    'general_greet', 'recommendation_events', 'music_dislikeness', 'iot_wemo_off', 'cooking_recipe',
    'qa_currency', 'transport_traffic', 'general_quirky', 'weather_query', 'audio_volume_up',
    'email_addcontact', 'takeaway_order', 'email_querycontact', 'iot_hue_lightup',
    'recommendation_locations', 'play_audiobook', 'lists_createoradd', 'news_query',
    'alarm_query', 'iot_wemo_on', 'general_joke', 'qa_definition', 'social_query',
    'music_settings', 'audio_volume_other', 'calendar_remove', 'iot_hue_lightdim',
    'calendar_query', 'email_sendemail', 'iot_cleaning', 'audio_volume_down',
    'play_radio', 'cooking_query', 'datetime_convert', 'qa_maths', 'iot_hue_lightoff',
    'iot_hue_lighton', 'transport_query', 'music_likeness', 'email_query', 'play_music',
    'audio_volume_mute', 'social_post', 'alarm_set', 'qa_factoid', 'calendar_set',
    'play_game', 'alarm_remove', 'lists_remove', 'transport_taxi', 'recommendation_movies',
    'iot_coffee', 'music_query', 'play_podcasts', 'lists_query',
]

# Fixed label order for the prompt's candidate list, for the 60x60 confusion
# matrix axes, and for the fixed-60 macro-F1. Sorted, i.e. a property of the
# label strings alone -- reproducible from this file without trusting the
# loader's (arbitrary) declaration order.
PROMPT_LABEL_ORDER = sorted(OFFICIAL_INTENTS)
LABEL_TO_IDX = {lab: i for i, lab in enumerate(PROMPT_LABEL_ORDER)}

# Deterministic intent -> prompt surface form. The ONLY transformation is
# '_' -> ' ' (pre-registered in massive_protocol.json; no LLM ever generates
# or paraphrases a label definition). Injective because no official intent
# string contains a space, so no two intents can collapse onto one surface
# form -- asserted in verify_manifest(), not assumed.
def label_surface(intent):
    return intent.replace("_", " ")


LABEL_SURFACE = {i: label_surface(i) for i in OFFICIAL_INTENTS}
SURFACE_TO_LABEL = {v: k for k, v in LABEL_SURFACE.items()}

# Fields that may never reach a prompt: the gold intent itself, its coarser
# parent label, the slot-annotated utterance (which contains the gold slots
# inline), and annotator metadata. Only `utt` is model input. Enforced by
# massive_eval.py's prompt builders, which take a plain string, and checked
# by massive_build_protocol.py's leak scan over the frozen prompts.
FORBIDDEN_PROMPT_FIELDS = ("intent", "scenario", "annot_utt", "worker_id", "judgments", "slot_method")


def load_locale(paper_lang):
    """All rows for one language, as-is from the official jsonl. Original
    `utt` strings are preserved byte-for-byte; no normalization here."""
    path = MASSIVE_ROOT / f"{LOCALES[paper_lang]}.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_all():
    """{paper_lang: {id: row}} for all five languages."""
    return {pl: {r["id"]: r for r in load_locale(pl)} for pl in LOCALES}


def nfc(s):
    return unicodedata.normalize("NFC", s)


def loose_key(s):
    """NFC + casefold + whitespace collapse -- the loosest match used by the
    contamination scan, deliberately over-inclusive so that a real overlap
    cannot hide behind punctuation or case."""
    return " ".join(nfc(s).casefold().split())


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def verify_manifest(data=None):
    """Structural checks that must hold before any protocol is frozen.
    Returns the manifest dict; raises on anything that would change what the
    task *is* (label set, split schema, cross-language ID correspondence)."""
    if data is None:
        data = load_all()

    # --- label set: the transcribed constant must match the data exactly ---
    if len(OFFICIAL_INTENTS) != 60 or len(set(OFFICIAL_INTENTS)) != 60:
        raise RuntimeError(f"OFFICIAL_INTENTS is not 60 unique intents ({len(set(OFFICIAL_INTENTS))})")
    if len(set(LABEL_SURFACE.values())) != 60:
        collide = [s for s, n in collections.Counter(LABEL_SURFACE.values()).items() if n > 1]
        raise RuntimeError(f"label_surface() is not one-to-one; colliding surface form(s): {collide}")

    per_lang = {}
    for pl, rows in data.items():
        intents = {r["intent"] for r in rows.values()}
        unknown = intents - set(OFFICIAL_INTENTS)
        if unknown:
            raise RuntimeError(f"{pl}: intents in data not in OFFICIAL_INTENTS: {sorted(unknown)}")
        parts = collections.Counter(r["partition"] for r in rows.values())
        per_lang[pl] = {
            "locale": LOCALES[pl],
            "n_rows": len(rows),
            "n_unique_ids": len(rows),
            "splits": {k: parts[k] for k in ("train", "dev", "test")},
            "n_intents_present": len(intents),
            "all_60_intents_present": len(intents) == 60,
            "n_empty_utt": sum(1 for r in rows.values() if not r["utt"].strip()),
        }

    # --- cross-language ID correspondence ---
    id_sets = {pl: set(rows) for pl, rows in data.items()}
    common = set.intersection(*id_sets.values())
    union = set.union(*id_sets.values())
    mismatch_intent = sorted(i for i in common if len({data[pl][i]["intent"] for pl in data}) > 1)
    mismatch_part = sorted(i for i in common if len({data[pl][i]["partition"] for pl in data}) > 1)
    # Pre-registered exclusion rule (§2): an ID is usable only if it exists in
    # all five languages, carries the same intent in all five, and sits in the
    # same partition in all five. Everything else is excluded from every
    # language at once, so all five languages always see the identical item set.
    excluded = {
        "not_in_all_languages": sorted(union - common),
        "intent_disagrees_across_languages": mismatch_intent,
        "partition_disagrees_across_languages": mismatch_part,
    }
    excluded_ids = set().union(*(set(v) for v in excluded.values()))
    usable = sorted(common - excluded_ids, key=int)

    # --- split disjointness by source ID (a MASSIVE id is one source
    # utterance across all five languages, so this is the cross-language
    # train/dev/test contamination check §3 asks for) ---
    by_part = collections.defaultdict(set)
    for i in usable:
        by_part[data["EN"][i]["partition"]].add(i)
    overlaps = {
        "train&dev": sorted(by_part["train"] & by_part["dev"]),
        "train&test": sorted(by_part["train"] & by_part["test"]),
        "dev&test": sorted(by_part["dev"] & by_part["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"source-ID overlap between splits: { {k: len(v) for k, v in overlaps.items()} }")

    return {
        "dataset": "MASSIVE (FitzGerald et al. 2022, arXiv:2204.08582)",
        "task": "text intent classification, official 60-class label set (utt -> intent)",
        "source_url": MASSIVE_URL,
        "tarball_sha256": MASSIVE_TARBALL_SHA256,
        "version": MASSIVE_VERSION,
        "license": MASSIVE_LICENSE,
        "homepage": "https://github.com/alexa/massive",
        "label_set_provenance": "massive.py::_INTENTS @ AmazonScience/massive ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740",
        "n_locales_in_release": 52,
        "n_locales_in_paper": 51,
        "locales_used": LOCALES,
        "schema_fields": sorted(next(iter(data["EN"].values())).keys()),
        "model_input_field": "utt",
        "forbidden_prompt_fields": list(FORBIDDEN_PROMPT_FIELDS),
        "per_language": per_lang,
        "n_common_ids": len(common),
        "n_union_ids": len(union),
        "id_exclusions": {k: {"n": len(v), "ids": v[:50]} for k, v in excluded.items()},
        "n_excluded_total": len(excluded_ids),
        "n_usable_ids": len(usable),
        "usable_split_sizes": {k: len(v) for k, v in by_part.items()},
        "split_source_id_overlap": {k: len(v) for k, v in overlaps.items()},
    }, usable, by_part
