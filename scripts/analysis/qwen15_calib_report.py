"""CPU-only tokenisation record for the Qwen port (tokenizer files only, no
model weights, no forward pass). Writes, per condition/seed:

  calibration windows (phase1_fisher.build_samples, unchanged):
    - calib_text_hash, and whether that hash appears among the hashes the
      DeepSeek runs of the SAME condition logged (proves identical text);
    - per window: characters, raw Qwen tokens (no truncation), realised tokens
      after truncation at 512, cap hit -> truncation rate and dropped-token share;
    - Balanced (mixed_5lang): language-attributed token shares of the realised
      tokens, with phase1_a2_balanced_token_share's manifest/attribution rules
      (first/last character label; boundary / separator / special buckets);
    - slow (use_fast=False, what the pipeline uses) vs fast tokenizer: ids must
      be identical on every window, because attribution needs fast offsets;
  expert-frequency input (the fake-wikitext sentences, shuffle(seed).select(2000)):
    - sentences, tokens, padded positions at batch 4 (pad positions are routed
      and counted by the vendored script);
  eval sets (written once under _tokenization/eval_sets.json):
    - FLORES devtest 60 sentences x 5 languages: tokens, bytes, tokens/byte,
      per-batch-of-8 padded length and fp32 logits bytes (memory planning),
      with and without a prepended <|endoftext|>;
    - XL-Sum OOD segments and the 4 unseen FLORES languages, same numbers.

The Balanced character-budget rule and the 512-token truncation are NOT
changed; this only measures what they produce under the Qwen tokenizer.

Usage:
    python -u qwen15_calib_report.py --conditions english_only --seeds 0
    python -u qwen15_calib_report.py --conditions all --seeds 0 1 2
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import qwen15_config as QC  # noqa: E402
import phase1_calib_data  # noqa: E402
import phase1_fisher  # noqa: E402
import data_utils  # noqa: E402

LOG_DIRS = [SCRIPT_DIR.parents[1] / "logs", SCRIPT_DIR / "logs"]


def deepseek_logged_hashes(condition, n_samples, seqlen):
    """calib_text_hash values the DeepSeek phase1_fisher/svd_scale runs printed
    for this condition at this budget (non-smoke). The DeepSeek log line does
    not carry the seed, so this is a set membership check."""
    found = set()
    head = re.compile(r"\[phase1-(fisher|svd-scale)\] condition=(\S+) smoke=(\w+) .*?n_samples=(\d+) seqlen=(\d+)")
    hsh = re.compile(r"calib_text_hash=([0-9a-f]{12})")
    for d in LOG_DIRS:
        for p in d.glob("*.log"):
            cur = None
            try:
                for line in p.open(errors="ignore"):
                    m = head.search(line)
                    if m:
                        cur = (m.group(2), m.group(3), int(m.group(4)), int(m.group(5)))
                        continue
                    h = hsh.search(line)
                    if h and cur == (condition, "False", n_samples, seqlen):
                        found.add(h.group(1))
            except OSError:
                continue
    return sorted(found)


def window_stats(tok, texts, seqlen):
    rows = []
    for t in texts:
        raw = len(tok(t)["input_ids"])
        realised = len(tok(t, truncation=True, max_length=seqlen)["input_ids"])
        rows.append({"chars": len(t), "raw_tokens": raw, "realised_tokens": realised, "cap_hit": raw >= seqlen})
    n = len(rows)
    raw_sum = sum(r["raw_tokens"] for r in rows)
    real_sum = sum(r["realised_tokens"] for r in rows)
    return rows, {
        "n_windows": n, "mean_raw_tokens": raw_sum / n, "mean_realised_tokens": real_sum / n,
        "total_realised_tokens": real_sum, "cap_hit_fraction": sum(r["cap_hit"] for r in rows) / n,
        "dropped_token_share": 1 - real_sum / raw_sum if raw_sum else 0.0,
        "fertility_chars_per_token": (sum(r["chars"] for r in rows) / n) / (raw_sum / n),
    }


def slow_fast_identical(slow, fast, texts, seqlen):
    mism = []
    for i, t in enumerate(texts):
        a = slow(t, truncation=True, max_length=seqlen)["input_ids"]
        b = fast(t, truncation=True, max_length=seqlen)["input_ids"]
        if a != b:
            mism.append(i)
    return mism


def balanced_attribution(fast, seed, n_samples, seqlen):
    import phase1_a2_balanced_token_share as A2  # CONDITION == "mixed_5lang"
    pairs = A2.build_labeled_sentences(seed)  # asserts equality with phase1_calib_data
    tot_text, starts, labels = A2.build_char_manifest(pairs)
    samples = phase1_fisher.build_samples(A2.CONDITION, n_samples, seqlen, seed)
    offsets = A2.window_offsets(tot_text, n_samples, seqlen, seed)
    for k, (text, off) in enumerate(zip(samples, offsets)):
        if tot_text[off:off + seqlen * 4] != text:
            raise AssertionError(f"window {k}: offset reproduction failed")
    counts = {lang: 0 for lang in A2.LANG_ORDER}
    boundary = sep_only = special = total = 0
    for text, off in zip(samples, offsets):
        enc = fast(text, truncation=True, max_length=seqlen, return_offsets_mapping=True)
        for (a, b) in enc["offset_mapping"]:
            total += 1
            if b <= a:
                special += 1
                continue
            first = A2.label_at(starts, labels, off + a)
            last = A2.label_at(starts, labels, off + b - 1)
            if first != last:
                boundary += 1
            elif first == "sep":
                sep_only += 1
            else:
                counts[first] += 1
    lang_total = sum(counts.values())
    return {
        "total_tokens": total,
        "language_tokens": counts,
        "share_pct_of_language_attributed": {k: 100.0 * v / lang_total for k, v in counts.items()},
        "share_pct_of_all": {k: 100.0 * v / total for k, v in counts.items()},
        "boundary_tokens": boundary, "separator_only_tokens": sep_only, "special_tokens": special,
        "rule": "phase1_a2_balanced_token_share: first==last char label -> that language; else boundary",
    }


def freq_input_stats(tok, condition, seed):
    ds = phase1_calib_data.make_fake_wikitext_dataset(condition, seed)
    ds = ds.shuffle(seed=seed).select(range(min(QC.FREQ_MAX_SAMPLES, len(ds))))
    texts = ds["text"]
    lens = [len(tok(t, truncation=True, max_length=2048)["input_ids"]) for t in texts]
    padded = 0
    for i in range(0, len(lens), QC.FREQ_BATCH_SIZE):
        chunk = lens[i:i + QC.FREQ_BATCH_SIZE]
        padded += max(chunk) * len(chunk)
    return {"n_sentences": len(texts), "real_tokens": sum(lens), "padded_positions_counted": padded,
            "pad_share": 1 - sum(lens) / padded if padded else 0.0, "max_sentence_tokens": max(lens)}


def seq_batch_stats(tok, seqs, batch_size, vocab, bos_id):
    lens = [len(tok(s)["input_ids"]) for s in seqs]
    nbytes = sum(len(s.encode("utf-8")) for s in seqs)
    batches = [lens[i:i + batch_size] for i in range(0, len(lens), batch_size)]
    max_pad = max(max(b) for b in batches)
    out = {"n": len(seqs), "tokens": sum(lens), "bytes": nbytes, "tokens_per_byte": sum(lens) / nbytes,
           "scored_tokens_default": sum(lens) - len(lens), "max_tokens": max(lens),
           "max_batch_padded_len": max_pad,
           "max_batch_fp32_logits_gib": batch_size * max_pad * vocab * 4 / 2 ** 30,
           "max_batch_fp32_logits_gib_bos_prefixed": batch_size * (max_pad + 1) * vocab * 4 / 2 ** 30}
    out["bos_id_used_for_sensitivity"] = bos_id
    return out


def eval_set_stats(tok):
    import phase1_merge_eval as PM  # constants only
    import e2e3_ood_unseen_eval as E2
    V, bos = QC.EXPECTED_CONFIG["vocab_size"], tok.eos_token_id
    res = {"flores": {}, "xlsum_ood": {}, "unseen": {}}
    for lang in PM.EVAL_LANGS:
        s = data_utils.load_flores_sentences(PM.EVAL_SOURCE, PM.EVAL_SPLIT, PM.EVAL_PREFIX, lang, PM.EVAL_N_SENTENCES)
        res["flores"][lang] = seq_batch_stats(tok, s, QC.EVAL_BATCH_SIZE, V, bos)
    for lang in E2.OOD_LANGS:
        segs = E2.segment((E2.OOD_DIR / f"{lang}.txt").read_text(encoding="utf-8"))
        res["xlsum_ood"][lang] = seq_batch_stats(tok, segs, E2.EVAL_BATCH_SIZE, V, bos)
    for lang in E2.UNSEEN_LANGS:
        s = data_utils.load_flores_sentences(PM.EVAL_SOURCE, PM.EVAL_SPLIT, PM.EVAL_PREFIX, lang, E2.UNSEEN_N_SENTENCES)
        res["unseen"][lang] = seq_batch_stats(tok, s, E2.EVAL_BATCH_SIZE, V, bos)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--skip-eval-sets", action="store_true")
    args = ap.parse_args()
    QC.check_env()
    conds = QC.CONDITIONS if args.conditions == ["all"] else args.conditions
    for c in conds:
        if c not in QC.CONDITIONS:
            raise SystemExit(f"unknown condition {c}")
    from transformers import AutoTokenizer
    snap = QC.model_snapshot_dir()
    slow = AutoTokenizer.from_pretrained(snap, use_fast=False)
    fast = AutoTokenizer.from_pretrained(snap, use_fast=True)
    out_root = QC.RESULTS_ROOT / "_tokenization"

    if not args.skip_eval_sets and not (out_root / "eval_sets.json").exists():
        ev = eval_set_stats(slow)
        ev["tokenizer"] = {"class": type(slow).__name__, "bos_token": slow.bos_token, "eos_token": slow.eos_token,
                           "pad_token": slow.pad_token, "padding_side": slow.padding_side}
        QC.write_json_new(out_root / "eval_sets.json", ev)
        for lang, r in ev["flores"].items():
            print(f"[qwen15-tok] FLORES {lang}: {r['tokens']} tok / {r['bytes']} B = {r['tokens_per_byte']:.4f} tok/B, "
                  f"max batch len {r['max_batch_padded_len']} -> fp32 logits {r['max_batch_fp32_logits_gib']:.2f} GiB",
                  flush=True)

    for c in conds:
        for seed in args.seeds:
            path = out_root / f"{c}_seed{seed}.json"
            if path.exists():
                print(f"[qwen15-tok] {path} exists, skipping", flush=True)
                continue
            texts = phase1_fisher.build_samples(c, QC.N_SAMPLES, QC.SEQLEN, seed)
            h = hashlib.md5("".join(texts).encode()).hexdigest()[:12]
            ds_hashes = deepseek_logged_hashes(c, QC.N_SAMPLES, QC.SEQLEN)
            rows, summary = window_stats(slow, texts, QC.SEQLEN)
            mism = slow_fast_identical(slow, fast, texts, QC.SEQLEN)
            rec = {"condition": c, "seed": seed, "calib_text_hash": h,
                   "deepseek_logged_hashes_same_condition": ds_hashes,
                   "hash_matches_a_deepseek_run": h in ds_hashes,
                   "summary": summary, "windows": rows, "slow_fast_mismatch_windows": mism,
                   "freq_input": freq_input_stats(slow, c, seed)}
            if c == "mixed_5lang":
                if mism:
                    raise RuntimeError(f"slow/fast tokenizers disagree on windows {mism[:5]} -- attribution invalid")
                rec["balanced_attribution"] = balanced_attribution(fast, seed, QC.N_SAMPLES, QC.SEQLEN)
            QC.write_json_new(path, rec)
            print(f"[qwen15-tok] {c} seed{seed}: hash={h} (in DeepSeek logs: {h in ds_hashes}) "
                  f"mean_raw={summary['mean_raw_tokens']:.1f} mean_realised={summary['mean_realised_tokens']:.1f} "
                  f"cap_hit={summary['cap_hit_fraction']:.3f} slow!=fast: {len(mism)}", flush=True)


if __name__ == "__main__":
    main()
