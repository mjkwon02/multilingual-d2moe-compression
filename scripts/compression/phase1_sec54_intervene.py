"""§5.4 STEP 1/2 -- causal-chain intervention: restore selected (layer, expert)
pairs of the COMPRESSED model back to their ORIGINAL (uncompressed) weights and
re-measure FLORES devtest bpb, to test whether the damage localized by §5.1/§5.2
(cross-language disagreement in proxy importance) is causally responsible for the
cross-language bpb loss.

Design is FIXED by 00_docs/SEC54_PREREG.md (read-only, registered before any GPU
run). Nothing here may deviate from it:
  - compressed model: english_only seed0, delta_ratio=0.8, svd_scale=None
    (plain SVD), merge_method=fisher
  - eval: FLORES devtest 60 sentences x 5 langs, metric = delta-bpb% vs the
    uncompressed baseline (identical definition to every other phase1 gate)
  - intervention unit: one (layer, expert); restoration = replace that expert's
    compressed representation (Wmean + low-rank delta) by the original W_j
  - targets: phase1/sec54/targets.json (12 treatment / 12 control, 1:1 matched)

STEP 1 (this script's default) evaluates only pairs 0 and 1 -- the two
highest-signal treatments (L25E32, L23E51) plus their matched controls (L25E45,
L23E62) plus the unmodified compressed model = 5 evals. Per prereg §5 the
result decides individual vs bundle mode for the remaining 20 interventions;
this script REPORTS that verdict and stops. It never auto-continues.

Why this is only 1 merge + 1 eval per intervention (prereg §2, re-verified in
code 2026-07-31):
  - With this pilot's cfg (test_stage=False, no_probe_process=False,
    calibration_stage=False, skip_layers=[[]], and meanW_deltaUV.layer_order
    left at its default []), merge_deepseek.py:622's
    `self.layer_order not in cfg['skip_layers']` is False, so forward() takes
    the final else branch (647-651):
        gate = Wmean_gate(x) + delta_u1(delta_v1(x))
        up   = Wmean_up(x)   + delta_u3(delta_v3(x))
        out  = Wmean_down(act(gate)*up) + delta_u2(delta_v2(act(gate)*up))
    Setting expert j's Wmean_* to its ORIGINAL gate/up/down_proj weights and
    zeroing delta_u{1,2,3} therefore makes forward compute exactly the original
    expert -- no low-rank approximation left.
  - Merge_deepseekMoE.update_Wmean() assigns the block's SHARED Wmean_* module
    to every expert, so rebinding the attribute on expert j only affects j.
  - delta_share_V/delta_share_U are both False (phase1_merge_eval.py), so
    delta_u/v are per-expert modules -- zeroing them cannot leak to other
    experts. This script asserts all of the above at runtime.
  - Restoration is a pure attribute rebind, so it is exactly reversible and no
    re-merge is needed between interventions.

Usage (2 GPUs required -- 1-GPU runs disk-offload and produce NaN logits):
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate d2moe_env
    CUDA_VISIBLE_DEVICES=2,3 python -u phase1_sec54_intervene.py            # STEP 1 gate (pairs 0,1)
    CUDA_VISIBLE_DEVICES=2,3 python -u phase1_sec54_intervene.py --pairs all --mode individual
    CUDA_VISIBLE_DEVICES=2,3 python -u phase1_sec54_intervene.py --pairs all --mode bundle
    python phase1_sec54_intervene.py --dry-run                              # CPU, no model: print plan
"""
import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)] + [str(d) for d in SCRIPT_DIR.parent.iterdir() if d.is_dir()] + [str(SCRIPT_DIR.parents[1] / "src" / "multilingual_d2moe")]
import phase1_merge_eval as pme  # noqa: E402 -- merge/eval/cfg single source of truth

RESULTS_ROOT = pme.RESULTS_ROOT
SEC54_DIR = RESULTS_ROOT / "sec54"
TARGETS_PATH = SEC54_DIR / "targets.json"
EVAL_DIR = SEC54_DIR / "evals"
BASELINE_PATH = RESULTS_ROOT / "baseline" / "eval_ppl.json"

# prereg §1, as amended. AMENDMENT 1 (00_docs/SEC54_RESULT.md, 2026-07-31 12:02
# UTC, recorded with zero intervention evals in existence): svd_scale=None ->
# whitening ON. prereg's stated reason for plain SVD ("max cross-damage") was
# factually wrong -- KO/BN damage is ~4x larger with whitening ON (KO 88.68 vs
# 23.00, BN 89.73 vs 35.00), so whitening ON is what actually maximizes the
# restoration signal's dynamic range, and it is also the condition §4.4/§5.3's
# whitening claims are about. Everything else (targets, P1/P2/P3, verdict
# vocabulary, 0.3pp gate, eval protocol) is unchanged from the prereg.
CONDITION, SEED = "english_only", 0
SCALE_CONDITION, SCALE_SEED = "english_only", 0   # whitening ON, official grid's scale_english_only_seed0
# B-3 sanity check: the unmodified compressed model must reproduce the official
# grid cell it is supposed to be, bit-for-bit within tolerance.
REFERENCE_PATH = (RESULTS_ROOT / CONDITION / f"seed{SEED}" /
                  f"scale_{SCALE_CONDITION}_seed{SCALE_SEED}" / "eval_ppl.json")
REFERENCE_TOL_BPB = 0.001
LANGS = list(pme.EVAL_LANGS)    # eng/kor/zho/swh/ben, canonical order
VULNERABLE = ["kor_Hang", "swh_Latn", "ben_Beng"]   # prereg P1's "취약 언어"
CONTROL_LANG = "eng_Latn"                            # prereg P3's negative control

# targets.json's routing covariates (hit_count_by_lang, proxy_by_lang) are keyed
# by CALIBRATION CONDITION name, not by FLORES language code -- the expert-freq
# scan is per-condition. Indexing them with LANGS raises KeyError (hit 14:03 UTC
# on the first treatment row). P2 will need the same mapping when it correlates
# gain against freq_{e,L}, so it lives here rather than inline.
LANG_TO_CONDITION = {"eng_Latn": "english_only", "kor_Hang": "korean_only",
                     "zho_Hans": "chinese_only", "swh_Latn": "swahili_only",
                     "ben_Beng": "bengali_only"}

STEP1_PAIRS = [0, 1]            # prereg §5 gate pairs (L25E32, L23E51)
STEP1_GATE_PP = 0.3             # < 0.3pp max per-language gain => bundle mode

# EVAL_BATCH_SIZE must equal the batch size that produced REFERENCE_PATH (8),
# or B-3 fails on arithmetic that has nothing to do with the intervention.
# Learned the hard way 2026-07-31: a global demotion to 2 (to dodge a Bengali
# OOM) shifted zho_Hans by +1.2e-3 bpb -- above the 1e-3 B-3 tolerance -- via
# bf16 reduction-order differences, and B-3 correctly rejected the run.
# Only ben_Beng, the split that actually OOMs, may fall back; every other
# language re-raises rather than silently changing its arithmetic.
EVAL_BATCH_SIZE = 8             # == the reference json's batch size. Do not lower globally.
EVAL_OOM_FALLBACK = {"ben_Beng": 2}   # ben measured at 2: diff -2.1e-4 vs ref, inside tolerance

# AMENDMENT 2 (--mode full): whole-layer restoration probes. LATE=25 is a
# targets layer in the late band; MID=11 is the empirical minimum of the §5.3
# U-curve (mean_angle_deg_k64 between-within separation 29.457deg, the lowest
# in the 10-18 bottom band per results/whitening_geometry_analysis_layer_curve
# .json) and carries no target expert, so it probes the U-bottom without
# reusing any selected (layer, expert). Cloning all 64 experts of a layer costs
# 64 x 3 x 2048 x 1408 bf16 ~= 1.1GiB CPU per layer.
PROBE_LAYER_LATE = 25
PROBE_LAYER_MID = 11
N_EXPERTS = 64

EVAL_SECONDS = {}               # tag -> wall-clock seconds (STEP 2 budgeting)


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------
def load_targets():
    if not TARGETS_PATH.exists():
        raise FileNotFoundError(f"{TARGETS_PATH} missing -- run phase1_sec54_select_targets.py first")
    t = json.loads(TARGETS_PATH.read_text())
    by_key = {(r["layer"], r["expert"]): r for r in t["treatment"] + t["control"]}
    items = []
    for i, p in enumerate(t["pairs"]):
        items.append({
            "pair": i, "layer": p["layer"], "band": p["band"], "e1_lang": p["e1_lang"],
            "treatment": by_key[(p["layer"], p["treatment_expert"])],
            "control": by_key[(p["layer"], p["control_expert"])],
        })
    return items


def select(items, pairs):
    if pairs == "all":
        return items
    want = set(pairs)
    missing = want - {it["pair"] for it in items}
    if missing:
        raise ValueError(f"unknown pair ids {sorted(missing)}")
    return [it for it in items if it["pair"] in want]


def needed_experts(sel, arms):
    """{layer: set(expert)} -- exactly the originals that must be cloned."""
    need = {}
    for it in sel:
        for arm in arms:
            need.setdefault(it["layer"], set()).add(it[arm]["expert"])
    return need


# --------------------------------------------------------------------------
# original-weight capture (requirement: inside the merge loop, before del)
# --------------------------------------------------------------------------
def make_clone_hook(need, store):
    """Returns a pre_merge_hook for phase1_merge_eval.merge_condition().

    merge_condition() calls this INSIDE its per-layer loop, before that layer's
    merge_experts() and therefore before `del old_mlp` + gc.collect() frees the
    original 64-expert MLP. There is no correct place to do this from outside
    merge_condition() -- by the time it returns, every original expert weight is
    already gone. Clones land on CPU (24 experts x 3 matrices, bf16 ~415MB;
    --mode full adds two whole layers at ~1.1GiB each).
    """
    def hook(layer_idx, original_mlp):
        wanted = need.get(layer_idx)
        if not wanted:
            return
        for e in sorted(wanted):
            src = original_mlp.experts[e]
            store[(layer_idx, e)] = {
                "gate": src.gate_proj.weight.detach().to("cpu", copy=True),
                "up": src.up_proj.weight.detach().to("cpu", copy=True),
                "down": src.down_proj.weight.detach().to("cpu", copy=True),
            }
            print(f"[sec54] cloned original weights: layer {layer_idx} expert {e}")
    return hook


# --------------------------------------------------------------------------
# runtime assertions on the compressed model (the mechanism prereg §2 relies on)
# --------------------------------------------------------------------------
def assert_mechanism(model, need):
    import os
    old_cwd = Path.cwd()
    os.chdir(pme.D2MOE_DIR)
    try:
        from config import cfg
    finally:
        os.chdir(old_cwd)

    for key, want in [("test_stage", False), ("no_probe_process", False),
                      ("calibration_stage", False), ("skip_layers", [[]])]:
        assert cfg[key] == want, f"cfg[{key!r}]={cfg[key]!r}, expected {want!r} -- forward() would not take the plain mean+delta branch"

    for layer, experts in need.items():
        blk = model.model.layers[layer].mlp
        assert type(blk).__name__ == "Merge_deepseekMoE", f"layer {layer} was not merged"
        assert blk.shared_infer is False, "shared_infer must be False for per-expert restoration"
        assert blk.delta_share_V is False and blk.delta_share_U is False, \
            "delta_u/v are shared across experts -- zeroing them would corrupt other experts"
        for e in experts:
            ex = blk.experts[e]
            assert ex.layer_order == [], f"layer {layer} expert {e}: layer_order={ex.layer_order!r}, forward() would take the probe branch"
            assert ex.Wmean_gate is blk.Wmean_gate, f"layer {layer} expert {e}: Wmean_gate is not the block's shared module"
    print(f"[sec54] mechanism asserts passed on {sum(len(v) for v in need.values())} target experts")


# --------------------------------------------------------------------------
# restore / undo
# --------------------------------------------------------------------------
def _linear_like(weight, device):
    lin = nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=torch.bfloat16)
    lin.weight.data.copy_(weight)
    # accelerate's dispatch hooks live on the modules created at load time; a
    # module rebound afterwards gets NO hook, so it must be placed on the
    # layer's device explicitly or forward() dies with a device mismatch.
    return lin.to(device)


def restore_expert(model, layer, expert, originals):
    """Replace expert's compressed representation by its original weights.
    Returns a snapshot for undo_expert(); pure attribute rebinding, so the
    compressed modules themselves are never mutated."""
    blk = model.model.layers[layer].mlp
    ex = blk.experts[expert]
    device = blk.Wmean_gate.weight.device
    orig = originals[(layer, expert)]

    snap = {name: getattr(ex, name) for name in
            ("Wmean_gate", "Wmean_up", "Wmean_down", "delta_u1", "delta_u2", "delta_u3")}

    ex.Wmean_gate = _linear_like(orig["gate"], device)
    ex.Wmean_up = _linear_like(orig["up"], device)
    ex.Wmean_down = _linear_like(orig["down"], device)
    for name in ("delta_u1", "delta_u2", "delta_u3"):
        w = snap[name].weight
        z = nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=torch.bfloat16)
        z.weight.data.zero_()
        setattr(ex, name, z.to(device))

    # verify the restoration is exact, not approximately right
    assert torch.equal(ex.Wmean_gate.weight.detach().cpu(), orig["gate"])
    assert torch.equal(ex.Wmean_up.weight.detach().cpu(), orig["up"])
    assert torch.equal(ex.Wmean_down.weight.detach().cpu(), orig["down"])
    for name in ("delta_u1", "delta_u2", "delta_u3"):
        assert not getattr(ex, name).weight.detach().any(), f"{name} not zeroed"
    return snap


def undo_expert(model, layer, expert, snap):
    ex = model.model.layers[layer].mlp.experts[expert]
    for name, mod in snap.items():
        setattr(ex, name, mod)
    blk = model.model.layers[layer].mlp
    assert ex.Wmean_gate is blk.Wmean_gate and ex.Wmean_up is blk.Wmean_up \
        and ex.Wmean_down is blk.Wmean_down, "undo did not restore the shared Wmean references"
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------
def batch_is_current(batch_used):
    """Is a cached eval's per-language batch one this run would itself produce?
    Each language must have run at EVAL_BATCH_SIZE, or at its pre-authorised
    fallback. Caches with no recorded batch (written before 2026-07-31 13:40)
    return False -- their arithmetic is unknown, so they are re-run, not
    trusted. This is the guard that stops superseded batch-2 numbers from
    silently surviving a config change."""
    if not isinstance(batch_used, dict):
        return False
    return all(batch_used.get(l) in {EVAL_BATCH_SIZE, EVAL_OOM_FALLBACK.get(l)} for l in LANGS)


def run_eval(model, tokenizer, tag, force=False):
    """One FLORES devtest eval. Resume-safe: a valid cached json is reused."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / f"{tag}.json"
    if out.exists() and not force:
        cached = json.loads(out.read_text())
        finite = all(l in cached and isinstance(cached[l].get("bits_per_byte"), float)
                     and cached[l]["bits_per_byte"] == cached[l]["bits_per_byte"] for l in LANGS)
        # A cache written under a different batch size is NOT reusable: bpb is
        # only reproducible at fixed batch (see eval_flores_ppl's docstring).
        # Caches predating this field have no _meta and are refused outright --
        # that is what silently resurrected batch-2 numbers otherwise.
        cached_batch = (cached.get("_meta") or {}).get("batch_used")
        if finite and batch_is_current(cached_batch):
            print(f"[sec54] reuse cached eval {tag} (batch {cached_batch})")
            return cached
        why = ("non-finite bpb" if not finite else
               f"batch not current: cached={cached_batch}, this run allows "
               f"{EVAL_BATCH_SIZE} (fallback {EVAL_OOM_FALLBACK})")
        print(f"[sec54] cached {tag} not reusable ({why}) -- re-running")

    print(f"[sec54] === eval {tag} ===")
    t0 = time.time()
    res, batch_used = pme.eval_flores_ppl(model, tokenizer, batch_size=EVAL_BATCH_SIZE,
                                          oom_fallback=EVAL_OOM_FALLBACK)
    print(f"[sec54] eval {tag} batch actually used per language: "
          + "  ".join(f"{l}={batch_used[l]}" for l in LANGS))
    EVAL_SECONDS[tag] = time.time() - t0
    print(f"[sec54] eval {tag} took {EVAL_SECONDS[tag]:.1f}s")
    log_mem(f"after eval {tag}")
    nan = [l for l in LANGS if res[l]["bits_per_byte"] != res[l]["bits_per_byte"]]
    if nan:
        raise RuntimeError(
            f"[sec54] {tag}: NaN bits_per_byte for {nan} -- almost certainly accelerate disk-offload "
            "(the SIB-200 failure mode). Refusing to write a garbage result.")
    res["_meta"] = {"batch_used": batch_used,
                    "batch_requested": EVAL_BATCH_SIZE,
                    "oom_fallback": EVAL_OOM_FALLBACK,
                    "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")}
    out.write_text(json.dumps(res, indent=2))
    return res


def delta_pct(res, baseline):
    return {l: (res[l]["bits_per_byte"] - baseline[l]["bits_per_byte"]) / baseline[l]["bits_per_byte"] * 100.0
            for l in LANGS}


def gains(dp_intervened, dp_compressed):
    """Restoration gain in pp: how much delta-bpb% the restoration removed.
    Positive = restoring this expert recovered performance."""
    return {l: dp_compressed[l] - dp_intervened[l] for l in LANGS}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def check_environment():
    n = torch.cuda.device_count()
    if n < 2:
        raise SystemExit(
            f"[sec54] {n} GPU visible -- downstream/eval on this merged model needs 2. On 1 GPU accelerate "
            "silently disk-offloads and every logit comes back NaN (00_docs/SIB_DIAGNOSIS.md). Aborting.")
    print(f"[sec54] {n} GPUs visible")
    # A-4 (2026-07-31): the 12:47 run's allocator setting could not be verified
    # after the fact because nothing logged it. Record the env that actually
    # shapes memory behaviour, so a later diff never has to rely on shell history.
    for var in ("PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES", "HF_HOME"):
        print(f"[sec54][env] {var}={os.environ.get(var, '<unset>')}")


def log_mem(where, min_available_gib=20):
    """B-1: swap was exhausted (562Mi free of 8G) when this run was approved, so
    RAM headroom is logged at every stage boundary and the run aborts before it
    can trigger an OOM kill (the failure mode that killed idea1_diagnosis)."""
    import re
    import subprocess
    out = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout
    mem = re.search(r"^Mem:\s+(\d+)\s+(\d+)\s+(\d+)\s+\S+\s+\S+\s+(\d+)", out, re.M)
    swp = re.search(r"^Swap:\s+(\d+)\s+(\d+)\s+(\d+)", out, re.M)
    avail = int(mem.group(4)) if mem else -1
    print(f"[sec54][mem] {where}: used={mem.group(2)}Gi available={avail}Gi "
          f"swap_used={swp.group(2)}Gi/{swp.group(1)}Gi" if mem and swp else f"[sec54][mem] {where}: {out}")
    if 0 <= avail < min_available_gib:
        raise SystemExit(f"[sec54] available RAM {avail}Gi < {min_available_gib}Gi at {where} -- aborting "
                         "before an OOM kill (B-1 guard). Nothing was deleted; rerun is resume-safe.")


def verify_reference(res):
    """B-3: the unmodified compressed model must match the official grid's
    english_only/seed0 whitening-ON cell. A mismatch means the merge or eval
    path drifted, and every gain computed against it would be meaningless."""
    if not REFERENCE_PATH.exists():
        raise SystemExit(f"[sec54] reference {REFERENCE_PATH} missing -- cannot validate the eval path")
    ref = json.loads(REFERENCE_PATH.read_text())
    diffs = {l: res[l]["bits_per_byte"] - ref[l]["bits_per_byte"] for l in LANGS}
    print("[sec54] B-3 vs official grid cell " + str(REFERENCE_PATH))
    for l in LANGS:
        flag = "OK " if abs(diffs[l]) <= REFERENCE_TOL_BPB else "MISMATCH"
        print(f"[sec54]   {l}: ours={res[l]['bits_per_byte']:.6f} ref={ref[l]['bits_per_byte']:.6f} "
              f"diff={diffs[l]:+.6f}  {flag}")
    bad = [l for l in LANGS if abs(diffs[l]) > REFERENCE_TOL_BPB]
    if bad:
        raise SystemExit(f"[sec54] compressed-model bpb differs from the official grid by >"
                         f"{REFERENCE_TOL_BPB} for {bad} -- eval path not sound. Stopping (B-3).")
    print("[sec54] B-3 passed: eval path reproduces the official grid cell")
    return diffs


def check_placement(model):
    dm = getattr(model, "hf_device_map", None) or {}
    bad = {k: v for k, v in dm.items() if v in ("disk", "cpu") or (isinstance(v, str) and "disk" in v)}
    if bad:
        raise SystemExit(f"[sec54] model was offloaded to disk/cpu ({list(bad)[:5]}...) -- results would be NaN. Aborting.")
    print("[sec54] device_map clean (no disk/cpu offload)")


# --------------------------------------------------------------------------
# AMENDMENT 2 analysis (prereg's original P1/P2 on the completed 12 pairs)
# --------------------------------------------------------------------------
def sign_test(diffs):
    """prereg P1: paired sign test over the 12 pairs, exact two-sided binomial.
    Ties (diff == 0) are dropped, the standard convention."""
    from scipy import stats
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    p = stats.binomtest(pos, n, 0.5, alternative="two-sided").pvalue if n else float("nan")
    return {"n_pairs": len(diffs), "n_used": n, "n_ties": len(diffs) - n,
            "n_treatment_wins": pos, "n_control_wins": neg,
            "p_two_sided_exact": p, "mean_diff_pp": (sum(diffs) / len(diffs)) if diffs else float("nan")}


def spearman(xs, ys):
    from scipy import stats
    r = stats.spearmanr(xs, ys)
    return {"rho": float(r.statistic), "p_two_sided": float(r.pvalue), "n": len(xs)}


def recovery_rates(gain_row, dp_compressed):
    """Gain as a fraction of the damage the compression caused, per language.
    dp_compressed is the unmodified compressed model's delta-bpb% (the damage);
    gain is in the same pp units, so the ratio is dimensionless."""
    return {l: (gain_row[f"gain_{l}"] / dp_compressed[l] * 100.0) if dp_compressed[l] else float("nan")
            for l in LANGS}


def write_csv(rows, path):
    fields = (["tag", "arm", "pair", "layer", "expert", "band", "e1_lang", "disagreement_var", "total_hits"]
              + [f"hit_{l}" for l in LANGS] + [f"bpb_{l}" for l in LANGS]
              + [f"dpct_{l}" for l in LANGS] + [f"gain_{l}" for l in LANGS]
              + ["gain_vuln_mean", f"gain_{CONTROL_LANG}"])
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[sec54] wrote {path}")


def main():
    global CONDITION, SCALE_CONDITION, SEED, SCALE_SEED, REFERENCE_PATH, SEC54_DIR, EVAL_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=",".join(str(p) for p in STEP1_PAIRS),
                    help="comma-separated pair ids, or 'all' (default: STEP 1 gate pairs 0,1)")
    ap.add_argument("--mode", choices=["individual", "bundle", "full"], default="individual",
                    help="individual = one eval per expert (prereg default); bundle = all treatments "
                         "restored at once vs all controls at once (2 evals, prereg §5 fallback, P2 dropped); "
                         "full = AMENDMENT 2: individual 12 pairs + late/early halves + whole-layer probes, "
                         "all on ONE merge")
    ap.add_argument("--dry-run", action="store_true", help="CPU only: print the plan, load no model")
    ap.add_argument("--force", action="store_true", help="ignore cached per-eval jsons")
    ap.add_argument("--out-suffix", default="", help="suffix for interventions{suffix}.csv")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="calibration seed for BOTH the merge condition and the whitening scale "
                         "(default 0 = the registered §5.4 run). Any other value is a REPRODUCTION "
                         "run: the target list stays the seed0 selection (targets.json is never "
                         "re-selected), and --out-dir must be given so seed0's artefacts are not "
                         "overwritten.")
    ap.add_argument("--out-dir", default="",
                    help="subdirectory under phase1/sec54/ for this run's evals/ and result jsons. "
                         "Empty = the canonical seed0 location.")
    ap.add_argument("--condition", default=CONDITION,
                    help="calibration condition of the COMPRESSED model (default english_only = the "
                         "registered §5.4 run). Any other value is a CONDITION-GENERALIZATION run: "
                         "targets.json stays the registered english_only/seed0 selection (never "
                         "re-selected), and --out-dir must be given so the registered artefacts are "
                         "not overwritten.")
    ap.add_argument("--scale-condition", default="",
                    help="calibration condition of the whitening scale (default: same as --condition, "
                         "i.e. the official grid's scale_{condition}_seed{seed} cell).")
    args = ap.parse_args()

    # Seed/condition override (reproduction + condition-generalization runs).
    # Rebinding the module globals keeps a single source of truth: REFERENCE_PATH,
    # EVAL_DIR and every output path are derived here, so no call site can
    # silently keep a registered-run path.
    if (args.seed != SEED or args.condition != CONDITION) and not args.out_dir:
        raise SystemExit(f"[sec54] --seed {args.seed} --condition {args.condition} without --out-dir "
                         "would overwrite the registered english_only/seed0 artefacts. Refusing.")
    CONDITION = args.condition
    SCALE_CONDITION = args.scale_condition or CONDITION
    SEED = SCALE_SEED = args.seed
    REFERENCE_PATH = (RESULTS_ROOT / CONDITION / f"seed{SEED}" /
                      f"scale_{SCALE_CONDITION}_seed{SCALE_SEED}" / "eval_ppl.json")
    if args.out_dir:
        SEC54_DIR = SEC54_DIR / args.out_dir
        EVAL_DIR = SEC54_DIR / "evals"
        print(f"[sec54] REPRODUCTION run: condition={CONDITION} seed={SEED}, artefacts -> {SEC54_DIR}")
        print(f"[sec54] targets stay the registered english_only/seed0 selection: {TARGETS_PATH}")

    pairs = "all" if args.pairs.strip() == "all" else [int(x) for x in args.pairs.split(",") if x.strip()]
    sel = select(load_targets(), pairs)
    arms = ["treatment", "control"]
    need = needed_experts(sel, arms)
    if args.mode == "full":
        if len(sel) != 12:
            raise SystemExit(f"[sec54] --mode full completes the prereg's 12-pair design; got "
                             f"{len(sel)} pairs. Pass --pairs all.")
        # AMENDMENT 2: whole-layer probes need every expert of the two probe
        # layers cloned, on top of the 24 targets.
        for layer in (PROBE_LAYER_LATE, PROBE_LAYER_MID):
            need.setdefault(layer, set()).update(range(N_EXPERTS))
    n_evals = (1 + (2 if args.mode == "bundle" else 2 * len(sel))
               + (5 if args.mode == "full" else 0))

    print(f"[sec54] condition={CONDITION} seed={SEED} "
          f"svd_scale={SCALE_CONDITION}/seed{SCALE_SEED} (whitening ON, AMENDMENT 1) "
          f"delta_ratio={pme.DELTA_RATIO} merge={pme.MERGE_METHOD} eval_batch={EVAL_BATCH_SIZE}")
    print(f"[sec54] mode={args.mode} pairs={[it['pair'] for it in sel]} -> {n_evals} evals "
          f"({sum(len(v) for v in need.values())} experts over {len(need)} layers)")
    for it in sel:
        t, c = it["treatment"], it["control"]
        print(f"  pair {it['pair']:>2} L{it['layer']:<2} {it['band']:>5}  T=E{t['expert']:<2} "
              f"(var {t['disagreement_var']:.4f}, hits {t['total_hits']})  "
              f"C=E{c['expert']:<2} (var {c['disagreement_var']:.4f}, hits {c['total_hits']})  e1={it['e1_lang']}")
    # Pre-flight clone-coverage assert (CPU, before any GPU work). The merge
    # hook fires per layer, so a target on a layer the hook never sees would
    # only surface AFTER the ~25min merge. In bundle mode `need` spans 8 layers
    # instead of STEP 1's 2, so check the plan against targets.json up front.
    planned = {(it[arm]["layer"], it[arm]["expert"]) for it in sel for arm in arms}
    from_need = {(layer, e) for layer, es in need.items() for e in es}
    if args.mode == "full":
        planned |= {(layer, e) for layer in (PROBE_LAYER_LATE, PROBE_LAYER_MID)
                    for e in range(N_EXPERTS)}
        assert PROBE_LAYER_MID not in {it["layer"] for it in sel}, \
            f"probe MID layer {PROBE_LAYER_MID} must carry no target expert"
        gib = len(planned) * 3 * 2048 * 1408 * 2 / 2**30
        print(f"[sec54] AMENDMENT 2 probe layers: LATE={PROBE_LAYER_LATE} (targets layer), "
              f"MID={PROBE_LAYER_MID} (§5.3 U-curve minimum in the 10-18 bottom band, "
              f"between-within 29.457deg, carries no target) -- clone budget ~{gib:.2f}GiB CPU")
    assert planned == from_need, f"clone plan != selection: {planned ^ from_need}"
    print(f"[sec54] clone plan covers {len(planned)} experts over layers "
          f"{sorted(need)} -- verified against targets.json before load")
    if args.dry_run:
        print("[sec54] --dry-run: no model loaded, nothing evaluated")
        return

    check_environment()
    baseline = json.loads(BASELINE_PATH.read_text())

    log_mem("before load")
    model, tokenizer = pme.load_model()
    check_placement(model)

    originals = {}
    model = pme.merge_condition(model, CONDITION, SEED, scale_condition=SCALE_CONDITION,
                                scale_seed=SCALE_SEED,
                                pre_merge_hook=make_clone_hook(need, originals))
    log_mem("after merge")
    missing = {k for layer, es in need.items() for k in ((layer, e) for e in es)} - set(originals)
    if missing:
        raise SystemExit(f"[sec54] originals not captured for {sorted(missing)} -- merge loop never saw those layers")
    assert_mechanism(model, need)

    rows = []
    res0 = run_eval(model, tokenizer, "compressed", force=args.force)
    ref_diffs = verify_reference(res0)
    dp0 = delta_pct(res0, baseline)
    rows.append({"tag": "compressed", "arm": "none", "pair": "", "layer": "", "expert": "",
                 **{f"bpb_{l}": res0[l]["bits_per_byte"] for l in LANGS},
                 **{f"dpct_{l}": dp0[l] for l in LANGS}})
    print("[sec54] unmodified compressed model delta-bpb%: "
          + "  ".join(f"{l}={dp0[l]:+.2f}" for l in LANGS))

    def record(tag, arm, it, rec, res):
        dp = delta_pct(res, baseline)
        g = gains(dp, dp0)
        rows.append({
            "tag": tag, "arm": arm, "pair": it["pair"] if it else "", "layer": rec["layer"] if rec else "",
            "expert": rec["expert"] if rec else "", "band": it["band"] if it else "",
            "e1_lang": (it["e1_lang"] if arm == "treatment" and it else ""),
            "disagreement_var": rec["disagreement_var"] if rec else "",
            "total_hits": rec["total_hits"] if rec else "",
            **{f"hit_{l}": (rec["hit_count_by_lang"][LANG_TO_CONDITION[l]] if rec else "") for l in LANGS},
            **{f"bpb_{l}": res[l]["bits_per_byte"] for l in LANGS},
            **{f"dpct_{l}": dp[l] for l in LANGS},
            **{f"gain_{l}": g[l] for l in LANGS},
            "gain_vuln_mean": sum(g[l] for l in VULNERABLE) / len(VULNERABLE),
        })
        print(f"[sec54] {tag}: gain(pp) " + "  ".join(f"{l}={g[l]:+.3f}" for l in LANGS)
              + f"  | vuln_mean={rows[-1]['gain_vuln_mean']:+.3f}")
        return g

    def multi_eval(tag, targets, label):
        """Restore every (layer, expert) in `targets` at once, eval, undo.
        Same verification the bundle path uses: exact-restore check on each
        target before the eval, shared-Wmean check on each after the undo."""
        snaps = [(layer, e, restore_expert(model, layer, e, originals)) for layer, e in targets]
        print(f"[sec54] {label}: restored {len(targets)} experts -> "
              + ", ".join(f"L{layer}E{e}" for layer, e in targets[:16])
              + (f", ... (+{len(targets) - 16} more)" if len(targets) > 16 else ""))
        assert len(set(targets)) == len(targets), f"{label} has duplicate targets"
        for layer, e in targets:
            ex, orig = model.model.layers[layer].mlp.experts[e], originals[(layer, e)]
            assert torch.equal(ex.Wmean_gate.weight.detach().cpu(), orig["gate"]) \
                and torch.equal(ex.Wmean_up.weight.detach().cpu(), orig["up"]) \
                and torch.equal(ex.Wmean_down.weight.detach().cpu(), orig["down"]), \
                f"L{layer}E{e} not exactly restored in {label}"
        print(f"[sec54] {label}: all {len(targets)} restorations verified exact (torch.equal)")
        try:
            res = run_eval(model, tokenizer, tag, force=args.force)
        finally:
            for layer, e, snap in snaps:
                undo_expert(model, layer, e, snap)
        for layer, e in targets:
            blk = model.model.layers[layer].mlp
            assert blk.experts[e].Wmean_gate is blk.Wmean_gate, f"L{layer}E{e} undo left a private Wmean"
        print(f"[sec54] {label}: undo verified -- all {len(targets)} experts back to compressed")
        return res

    if args.mode in ("individual", "full"):
        for it in sel:
            for arm in arms:
                rec = it[arm]
                tag = f"{arm[0].upper()}_L{rec['layer']}E{rec['expert']}"
                snap = restore_expert(model, rec["layer"], rec["expert"], originals)
                try:
                    res = run_eval(model, tokenizer, tag, force=args.force)
                finally:
                    undo_expert(model, rec["layer"], rec["expert"], snap)
                record(tag, arm, it, rec, res)

    if args.mode == "full":
        # 2-3 half decomposition (named in the original instruction) --
        # treatment only, late 8 vs early 4, so the split is by prereg band.
        for band in ("late", "early"):
            sub = [it for it in sel if it["band"] == band]
            tgts = [(it["treatment"]["layer"], it["treatment"]["expert"]) for it in sub]
            res = multi_eval(f"T_{band}{len(sub)}", tgts, f"half {band}")
            record(f"T_{band}{len(sub)}", "treatment_half", None, None, res)
        # 2-4 exploratory probes -- labelled, NOT verdict inputs.
        combos = [("explor_TC24", [(it[a]["layer"], it[a]["expert"]) for it in sel for a in arms],
                   "exploratory T12+C12"),
                  (f"explor_L{PROBE_LAYER_LATE}_all", [(PROBE_LAYER_LATE, e) for e in range(N_EXPERTS)],
                   f"exploratory whole layer {PROBE_LAYER_LATE} (late)"),
                  (f"explor_L{PROBE_LAYER_MID}_all", [(PROBE_LAYER_MID, e) for e in range(N_EXPERTS)],
                   f"exploratory whole layer {PROBE_LAYER_MID} (mid, U-bottom)")]
        for tag, tgts, label in combos:
            res = multi_eval(tag, tgts, label)
            record(tag, "exploratory", None, None, res)

    if args.mode == "bundle":
        for arm in arms:
            tag = f"bundle_{arm[0].upper()}"
            snaps = [(it[arm]["layer"], it[arm]["expert"],
                      restore_expert(model, it[arm]["layer"], it[arm]["expert"], originals)) for it in sel]
            # 12 simultaneous restorations across 8 distinct layers: print every
            # one and re-check it against targets.json, so a silently-skipped or
            # mis-targeted restoration cannot masquerade as a null result.
            restored = [(layer, expert) for layer, expert, _ in snaps]
            expected = [(it[arm]["layer"], it[arm]["expert"]) for it in sel]
            print(f"[sec54] bundle {arm}: restored {len(restored)} experts -> "
                  + ", ".join(f"L{layer}E{e}" for layer, e in restored))
            assert restored == expected, f"bundle {arm} restored {restored} != targets {expected}"
            assert len(set(restored)) == len(restored), f"bundle {arm} has duplicate targets: {restored}"
            for layer, e in restored:
                ex = model.model.layers[layer].mlp.experts[e]
                orig = originals[(layer, e)]
                assert torch.equal(ex.Wmean_gate.weight.detach().cpu(), orig["gate"]) \
                    and torch.equal(ex.Wmean_up.weight.detach().cpu(), orig["up"]) \
                    and torch.equal(ex.Wmean_down.weight.detach().cpu(), orig["down"]), \
                    f"L{layer}E{e} not exactly restored at bundle-eval time"
            print(f"[sec54] bundle {arm}: all {len(restored)} restorations verified exact "
                  "(torch.equal on gate/up/down) with every other expert untouched")
            try:
                res = run_eval(model, tokenizer, tag, force=args.force)
            finally:
                for layer, expert, snap in snaps:
                    undo_expert(model, layer, expert, snap)
            # undo must put the shared Wmean references back on every expert
            for layer, e in restored:
                ex = model.model.layers[layer].mlp.experts[e]
                blk = model.model.layers[layer].mlp
                assert ex.Wmean_gate is blk.Wmean_gate, f"L{layer}E{e} undo left a private Wmean"
            print(f"[sec54] bundle {arm}: undo verified -- all {len(restored)} experts back to compressed")
            record(tag, arm, None, None, res)

    SEC54_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, SEC54_DIR / f"interventions{args.out_suffix}.csv")

    # prereg §5 gate -- reported only. This script never auto-continues to the
    # remaining pairs; that is a human decision point.
    if args.mode == "individual":
        t_rows = [r for r in rows if r["arm"] == "treatment"]
        max_gain = max((max(r[f"gain_{l}"] for l in LANGS) for r in t_rows), default=float("nan"))
        verdict = "INDIVIDUAL_MODE" if max_gain >= STEP1_GATE_PP else "BUNDLE_MODE"
        gate = {"pairs": [it["pair"] for it in sel], "threshold_pp": STEP1_GATE_PP,
                "max_treatment_gain_pp": max_gain, "verdict": verdict,
                "per_treatment": {r["tag"]: {l: r[f"gain_{l}"] for l in LANGS} for r in t_rows},
                "scale_condition": f"{SCALE_CONDITION}/seed{SCALE_SEED} (whitening ON, AMENDMENT 1)",
                "reference_check_bpb_diff": ref_diffs,
                "eval_seconds": EVAL_SECONDS,
                "compressed_delta_bpb_pct": dp0}
        (SEC54_DIR / f"step1_gate{args.out_suffix}.json").write_text(json.dumps(gate, indent=2))
        print(f"\n[sec54] STEP 1 GATE: max treatment gain = {max_gain:+.3f}pp "
              f"(threshold {STEP1_GATE_PP}pp) -> {verdict}")
        print("[sec54] stopping here -- continuing to the remaining pairs is a human decision (prereg §5).")
    elif args.mode == "full":
        by_tag = {r["tag"]: r for r in rows}
        t_rows = [r for r in rows if r["arm"] == "treatment"]
        c_rows = [r for r in rows if r["arm"] == "control"]
        assert len(t_rows) == len(c_rows) == len(sel), "full mode expects one T and one C row per pair"
        tmap = {int(r["pair"]): r for r in t_rows}
        cmap = {int(r["pair"]): r for r in c_rows}

        # 3-1 formal P1: paired sign test on vulnerable-language mean gain.
        diffs = [tmap[i]["gain_vuln_mean"] - cmap[i]["gain_vuln_mean"] for i in sorted(tmap)]
        p1 = sign_test(diffs)
        p1["per_pair_diff_pp"] = {str(i): d for i, d in zip(sorted(tmap), diffs)}

        # 3-2 P1 conditional on total_hits (planned in the pre-registration's
        # "사전 관찰 (a)": control selection required hits >= layer median but
        # treatment had no matching constraint, so the arms are asymmetric).
        bal = [i for i in sorted(tmap) if tmap[i]["total_hits"] >= cmap[i]["total_hits"]]
        p1_cond = sign_test([tmap[i]["gain_vuln_mean"] - cmap[i]["gain_vuln_mean"] for i in bal])
        p1_cond["subset"] = "pairs where treatment total_hits >= control total_hits"
        p1_cond["pairs"] = bal

        # 3-3 P2: gain vs that expert's routing frequency for that language,
        # pooled over 24 experts x 5 languages.
        xs = [r[f"hit_{l}"] for r in t_rows + c_rows for l in LANGS]
        ys = [r[f"gain_{l}"] for r in t_rows + c_rows for l in LANGS]
        p2 = spearman(xs, ys)
        p2_t = spearman([r[f"hit_{l}"] for r in t_rows for l in LANGS],
                        [r[f"gain_{l}"] for r in t_rows for l in LANGS])
        p2["caveat"] = ("effect sizes are at the 0.01-0.1pp scale, i.e. ~1e-3 of the damage being "
                        "repaired -- a significant rho would describe the ordering of near-null "
                        "quantities, not a materially useful predictor")

        # 3-4 which control expert produced bundle_C's KO gain
        ko_contrib = sorted(((r["tag"], r["gain_kor_Hang"], r["total_hits"]) for r in c_rows),
                            key=lambda t: -t[1])

        # 3-5 recovery rate: gain as a % of the damage the compression caused
        recov = {r["tag"]: recovery_rates(r, dp0) for r in rows if r["arm"] != "none"}

        # 3-6 halves and whole-layer probes
        halves = {t: {l: by_tag[t][f"gain_{l}"] for l in LANGS}
                  for t in by_tag if t.startswith("T_late") or t.startswith("T_early")}
        probes = {t: {l: by_tag[t][f"gain_{l}"] for l in LANGS}
                  for t in by_tag if t.startswith("explor_")}

        out = {"mode": "full (AMENDMENT 2)",
               "verdict_hierarchy": ("PRIMARY verdict remains the gate-path bundle result "
                                     "(step2_bundle.json). These individual-12 results complete the "
                                     "prereg's original design and are reported alongside; they do "
                                     "not overwrite the primary verdict."),
               "scale_condition": f"{SCALE_CONDITION}/seed{SCALE_SEED} (whitening ON, AMENDMENT 1)",
               "reference_check_bpb_diff": ref_diffs, "compressed_delta_bpb_pct": dp0,
               "P1_sign_test": p1, "P1_conditional_on_total_hits": p1_cond,
               "P2_spearman_pooled": p2, "P2_spearman_treatment_only": p2_t,
               "control_KO_contributions_sorted": ko_contrib,
               "recovery_rate_pct_of_damage": recov,
               "halves_treatment": halves,
               "exploratory_probes": probes,
               "exploratory_note": "labelled exploratory -- not inputs to any verdict",
               "determinism_note": ("B-3 reproduced the reference to 6 decimals, so eval is "
                                    "deterministic: pair-to-pair differences are this model "
                                    "instance's real values, not measurement noise. Generalisation "
                                    "noise across seeds/conditions is a separate, untested question; "
                                    "everything here is english_only/seed0 only."),
               "eval_seconds": EVAL_SECONDS}
        (SEC54_DIR / f"step2_full{args.out_suffix}.json").write_text(json.dumps(out, indent=2))
        print(f"\n[sec54] P1 sign test (12 pairs, vulnerable mean gain T-C): "
              f"T wins {p1['n_treatment_wins']}/{p1['n_used']}, mean diff {p1['mean_diff_pp']:+.4f}pp, "
              f"exact two-sided p={p1['p_two_sided_exact']:.4f}")
        print(f"[sec54] P1 | total_hits balanced subset ({len(bal)} pairs): "
              f"T wins {p1_cond['n_treatment_wins']}/{p1_cond['n_used']}, p={p1_cond['p_two_sided_exact']:.4f}")
        print(f"[sec54] P2 Spearman pooled (24 experts x 5 langs, n={p2['n']}): "
              f"rho={p2['rho']:+.4f}, p={p2['p_two_sided']:.4f}  | treatment-only rho={p2_t['rho']:+.4f}")
        print(f"[sec54] control KO top contributor: {ko_contrib[0][0]} {ko_contrib[0][1]:+.3f}pp")
        print("[sec54] PRIMARY verdict unchanged (gate-path bundle). Final grade is a human call.")
        print("[sec54] stopping here.")
    else:
        bt = next(r for r in rows if r["tag"] == "bundle_T")
        bc = next(r for r in rows if r["tag"] == "bundle_C")
        p1 = {"treatment_vuln_mean_pp": bt["gain_vuln_mean"],
              "control_vuln_mean_pp": bc["gain_vuln_mean"],
              "difference_pp": bt["gain_vuln_mean"] - bc["gain_vuln_mean"],
              "direction_holds": bt["gain_vuln_mean"] > bc["gain_vuln_mean"]}
        p3 = {"treatment_en_pp": bt[f"gain_{CONTROL_LANG}"],
              "control_en_pp": bc[f"gain_{CONTROL_LANG}"],
              "difference_pp": bt[f"gain_{CONTROL_LANG}"] - bc[f"gain_{CONTROL_LANG}"]}

        # Additivity (recorded, NOT a verdict input). STEP 1 measured only 2 of
        # the 12 pairs individually, so the individual sum is a PARTIAL sum --
        # it bounds the true 12-expert sum from below in magnitude only if the
        # unmeasured 10 contribute with the same sign, which is unknown. Read
        # it as "bundle vs the 2 individuals we actually have", nothing more.
        additivity = None
        prior = SEC54_DIR / "step1_gate.json"
        if prior.exists():
            step1 = json.loads(prior.read_text())
            per_t = step1.get("per_treatment", {})
            partial_sum = {l: sum(v[l] for v in per_t.values()) for l in LANGS}
            additivity = {
                "caveat": (f"individual sum covers only {len(per_t)} of {len(sel)} treatment experts "
                           f"({sorted(per_t)}) -- partial, not the full 12-expert individual sum"),
                "individual_partial_sum_pp": partial_sum,
                "bundle_treatment_pp": {l: bt[f"gain_{l}"] for l in LANGS},
                "bundle_minus_partial_sum_pp": {l: bt[f"gain_{l}"] - partial_sum[l] for l in LANGS},
            }

        out = {"mode": "bundle", "pairs": [it["pair"] for it in sel],
               "n_restored_per_arm": len(sel),
               "condition": f"{CONDITION}/seed{SEED}",
               "scale_condition": f"{SCALE_CONDITION}/seed{SCALE_SEED} (whitening ON, AMENDMENT 1)",
               "reference_check_bpb_diff": ref_diffs,
               "compressed_delta_bpb_pct": dp0,
               "gains_pp": {"treatment": {l: bt[f"gain_{l}"] for l in LANGS},
                            "control": {l: bc[f"gain_{l}"] for l in LANGS}},
               "P1_vulnerable_bundle": p1, "P3_english_control": p3,
               "P2": "dropped -- prereg §5 bundle-mode fallback",
               "additivity_vs_step1": additivity,
               "eval_seconds": EVAL_SECONDS}
        (SEC54_DIR / f"step2_bundle{args.out_suffix}.json").write_text(json.dumps(out, indent=2))
        print(f"\n[sec54] BUNDLE P1 (vulnerable KO/SW/BN mean gain): "
              f"T={p1['treatment_vuln_mean_pp']:+.3f}pp  C={p1['control_vuln_mean_pp']:+.3f}pp  "
              f"diff={p1['difference_pp']:+.3f}pp -> direction {'HOLDS' if p1['direction_holds'] else 'DOES NOT HOLD'}")
        print(f"[sec54] BUNDLE P3 (EN negative control): T={p3['treatment_en_pp']:+.3f}pp  "
              f"C={p3['control_en_pp']:+.3f}pp  diff={p3['difference_pp']:+.3f}pp")
        print("[sec54] P2 dropped (prereg §5 bundle-mode fallback) -- verdict ceiling is PARTIAL.")
        print("[sec54] stopping here -- the final SUPPORTED/PARTIAL/NOT_SUPPORTED grade is a human call.")


if __name__ == "__main__":
    main()
