"""Qwen1.5-MoE-A2.7B structure adapter for the D2-MoE compression step.

What gets compressed, and what does not
---------------------------------------
Qwen2MoeSparseMoeBlock (transformers 4.42.4, modeling_qwen2_moe.py:790-849):
  router      gate: Linear(2048 -> 60), softmax in fp32 over 60 experts, top-4,
              NO renormalisation of the 4 probabilities (norm_topk_prob=False),
              cast back to bf16, output = sum_k p_k * expert_k(x)  (index_add_)
  routed      experts[0..59]: Qwen2MoeMLP(1408): down(silu(gate(x)) * up(x)),
              gate/up weight [1408, 2048], down weight [2048, 1408]
  shared      shared_expert: Qwen2MoeMLP(5632), always on, scaled by
              sigmoid(shared_expert_gate(x)), shared_expert_gate: Linear(2048 -> 1)

The D2-MoE decomposition (Fisher-weighted shared BASE weight Wmean + per-expert
low-rank DELTA) is applied to the 60 ROUTED experts only -- the same target as
the DeepSeek runs, where Merge_deepseekMoE keeps DeepSeek's 2 shared experts
untouched. Qwen's own shared_expert / shared_expert_gate / router are left as
the original module objects: we keep the ORIGINAL Qwen2MoeSparseMoeBlock and
swap only its `experts` ModuleList. Its forward (routing, un-normalised top-k
weights, shared-expert sigmoid gate, the (hidden, router_logits) tuple the
decoder layer unpacks) therefore runs the library's own code, not a copy.
"Shared base weight" (D2-MoE's Wmean) and "shared expert" (Qwen's always-on
MLP) are different things; nothing here merges one into the other.

Why not the vendored D2MoE/model/merge_qwen.py
----------------------------------------------
It targets Qwen2-57B-A14B (D2-qwen.py default), and its svd_delta wraps
torch.linalg.svd in `safe_svd`, which on failure retries with random noise of
growing size (1e-6, 1e-2, 1) -- and D2-qwen.py silently SKIPS a layer on
ValueError. Neither behaviour exists on the DeepSeek path this project's
results come from, so using it would change the method between models. We use
the DeepSeek path's arithmetic instead: Merge_deepseekMoE.svd_delta (called as
the same staticmethod object) and a statement-for-statement copy of its
merge_method="fisher" branch (fisher_merge_base_weights below).

Routing-coverage counting
-------------------------
Qwen's block calls EVERY expert on every forward, with a 0-row tensor when no
token was routed to it (no `continue`, modeling_qwen2_moe.py:829-841). So
"forward hook fired" and "weight.grad is not None" are both true for experts
that received zero tokens. (The DeepSeek remote code's train-mode forward also
calls every expert -- modeling_deepseek.py:387-388 in the HF cache -- which is
why the DeepSeek Fisher runs never logged a dead expert; only the eval-mode
whitening stage, whose moe_infer skips empty experts, did.) RoutingCounter
counts the rows each expert actually receives via a forward PRE-hook, so a
"hit" means >=1 routed token in that sample -- the meaning phase1_*.py intended.
"""
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

import qwen15_config as QC

PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")
_D2 = {}


def fisher_key(e, proj):
    """Same key format phase1_fisher.py writes and Merge_deepseekMoE reads."""
    return f"mlp.experts.{e}.{proj}"


def get_block(model, layer_idx):
    return model.model.layers[layer_idx].mlp


def d2moe():
    """Imports the vendored D2MoE pieces once, with the same cfg keys
    phase1_merge_eval.merge_condition() sets (copied verbatim from
    phase1_merge_eval.py:155-176, including the skip_layers=[[]] trick that
    makes meanW_deltaUV.forward take the plain Wmean + U V branch)."""
    if _D2:
        return _D2
    if str(QC.D2MOE_DIR) not in sys.path:
        sys.path.insert(0, str(QC.D2MOE_DIR))
    old_cwd = Path.cwd()
    os.chdir(QC.D2MOE_DIR)  # config.py reads './config.yml'
    try:
        from config import cfg
        from model.merge_deepseek import Merge_deepseekMoE, meanW_deltaUV
        cfg.setdefault("prune_metric", cfg["control"]["prune_metric"])
        cfg.setdefault("prune_ratio", 0.0)  # structured pruning OFF (main experiment)
        cfg.setdefault("test_stage", False)
        cfg.setdefault("no_probe_process", False)
        cfg.setdefault("skip_layers", [[]])
        cfg.setdefault("calibration_stage", False)
        cfg.setdefault("batch_size", cfg["control"]["batch_size"])
        cfg.setdefault("mode", cfg["control"]["mode"])
        cfg.setdefault("prune_method", cfg["control"]["prune_method"])
        cfg.setdefault("tc_multiple", 64)
        cfg.setdefault("onlyprobe", False)
        cfg.setdefault("gate_probe_ratio", 1.0)
        cfg.setdefault("up_probe_ratio", 1.0)
    finally:
        os.chdir(old_cwd)
    # meanW_deltaUV.forward takes the plain branch only under these values.
    assert cfg["test_stage"] is False and cfg["no_probe_process"] is False
    assert cfg["calibration_stage"] is False and [] in cfg["skip_layers"]
    _D2.update(cfg=cfg, Merge_deepseekMoE=Merge_deepseekMoE, meanW_deltaUV=meanW_deltaUV)
    return _D2


class RoutingCounter:
    """Rows (tokens) dispatched to each expert of one block, per sample."""

    def __init__(self, block):
        n = len(block.experts)
        self.hit_samples = [0] * n
        self.tokens = [0] * n
        self._cur = [0] * n
        self._handles = [ex.register_forward_pre_hook(self._make(e)) for e, ex in enumerate(block.experts)]

    def _make(self, e):
        def pre_hook(module, args):
            self._cur[e] += int(args[0].reshape(-1, args[0].shape[-1]).shape[0])
        return pre_hook

    def start_sample(self):
        self._cur = [0] * len(self._cur)

    def end_sample(self):
        for e, c in enumerate(self._cur):
            if c > 0:
                self.hit_samples[e] += 1
                self.tokens[e] += c

    def remove(self):
        for h in self._handles:
            h.remove()


@torch.no_grad()
def fisher_merge_base_weights(experts, hessian, expert_freq, num_experts):
    """Statement-for-statement copy of Merge_deepseekMoE.merge_experts'
    merge_method == "fisher" branch (D2MoE/model/merge_deepseek.py:261-285),
    with `self.config.n_routed_experts` -> num_experts and `module.experts` ->
    experts. Kept verbatim -- including Python sum() starting from int 0 and the
    per-j recomputation of sum(list) -- so the bf16 rounding order is the same
    as in every DeepSeek result. Element-wise: Wmean = sum_j W_j * (F_j f_j) /
    sum_k (F_k f_k), F = element-wise Fisher, f = routing frequency."""
    scaling_factor = 1.0  # noqa: F841 -- present (unused) in the original too
    scaling_factors_ft_list = expert_freq

    fisher_scale_w_gate = [0]*num_experts
    fisher_scale_w_down = [0]*num_experts
    fisher_scale_w_up = [0]*num_experts

    for j in range(num_experts):
        base_name = f"mlp.experts.{j}."
        fisher_scale_w_gate[j] = hessian[base_name + "gate_proj"].to(experts[j].gate_proj.weight.device) * scaling_factors_ft_list[j]
        fisher_scale_w_down[j] = hessian[base_name + "down_proj"].to(experts[j].down_proj.weight.device) * scaling_factors_ft_list[j]
        fisher_scale_w_up[j] = hessian[base_name + "up_proj"].to(experts[j].up_proj.weight.device) * scaling_factors_ft_list[j]

    fisher_scale_w_gate = [fisher_scale_w_gate[j] / sum(fisher_scale_w_gate) for j in range(num_experts)]
    fisher_scale_w_down = [fisher_scale_w_down[j] / sum(fisher_scale_w_down) for j in range(num_experts)]
    fisher_scale_w_up = [fisher_scale_w_up[j] / sum(fisher_scale_w_up) for j in range(num_experts)]

    Wmean_gate = sum([experts[j].gate_proj.weight * fisher_scale_w_gate[j] for j in range(num_experts)])
    Wmean_down = sum([experts[j].down_proj.weight * fisher_scale_w_down[j] for j in range(num_experts)])
    Wmean_up = sum([experts[j].up_proj.weight * fisher_scale_w_up[j] for j in range(num_experts)])
    return Wmean_gate, Wmean_down, Wmean_up


def zero_denominator_counts(hessian, expert_freq, num_experts):
    """Elements where every F_j * f_j is exactly 0: the verbatim merge divides
    0/0 there and Wmean becomes NaN. Checked on CPU BEFORE merging so a run
    stops with a clear reason instead of a NaN eval."""
    out = {}
    for proj in PROJ_NAMES:
        mask = None
        for j in range(num_experts):
            z = (hessian[fisher_key(j, proj)] == 0) if expert_freq[j] != 0 else None
            if z is None:
                m = torch.ones_like(hessian[fisher_key(j, proj)], dtype=torch.bool)
            else:
                m = z
            mask = m if mask is None else (mask & m)
        out[proj] = int(mask.sum().item())
    return out


def _new_expert_modules(config, num_experts, delta_ratio, device):
    """Same construction as Merge_deepseekMoE.__init__ for delta_share_V=False,
    delta_share_U=False (merge_deepseek.py:56-63): bf16 Linears built on CPU,
    then moved. Every weight is overwritten afterwards, so the random init is
    irrelevant."""
    D = d2moe()
    h, m = config.hidden_size, config.moe_intermediate_size
    Wmean_gate = nn.Linear(h, m, bias=False, dtype=torch.bfloat16)
    Wmean_down = nn.Linear(m, h, bias=False, dtype=torch.bfloat16)
    Wmean_up = nn.Linear(h, m, bias=False, dtype=torch.bfloat16)
    experts = nn.ModuleList([D["meanW_deltaUV"](config, Wmean_gate, Wmean_down, Wmean_up, delta_ratio,
                                               delta_share_V=False, delta_share_U=False)
                             for _ in range(num_experts)])
    holder = nn.Module()
    holder.Wmean_gate, holder.Wmean_down, holder.Wmean_up, holder.experts = Wmean_gate, Wmean_down, Wmean_up, experts
    holder.to(device)
    return holder


def _install(block, holder):
    old = block.experts
    block.experts = holder.experts
    block.Wmean_gate, block.Wmean_down, block.Wmean_up = holder.Wmean_gate, holder.Wmean_down, holder.Wmean_up
    del old
    gc.collect()
    torch.cuda.empty_cache()


def _finite_report(named_tensors):
    return [n for n, t in named_tensors if not bool(torch.isfinite(t).all().item())]


def _experts_numel(experts):
    return sum(p.numel() for p in experts.parameters())


@torch.no_grad()
def compress_block(block, config, hessian, expert_freq, svd_scale, delta_ratio=QC.DELTA_RATIO, scale_type="svdllm"):
    """Replace block.experts by Wmean + per-expert delta factors. `hessian` and
    `svd_scale` are this layer's dicts (CPU, bf16), exactly as
    Merge_deepseekMoE.merge_experts receives them; svd_scale=None means plain
    SVD. svd_scale stays on CPU as on the DeepSeek path (its inverse is then
    taken on CPU inside cal_scale_inv, as there)."""
    D = d2moe()
    svd_delta = D["Merge_deepseekMoE"].svd_delta
    num_experts = len(block.experts)
    device = block.gate.weight.device
    before = _experts_numel(block.experts)

    zero_den = zero_denominator_counts(hessian, expert_freq, num_experts)
    if any(zero_den.values()):
        raise RuntimeError(f"Fisher-weighted merge would divide 0/0 at {zero_den} elements (all F_j*f_j == 0) "
                           "-- Wmean would be NaN there. Stopping before the merge.")

    holder = _new_expert_modules(config, num_experts, delta_ratio, device)
    Wg, Wd, Wu = fisher_merge_base_weights(block.experts, hessian, expert_freq, num_experts)
    holder.Wmean_gate.weight.data = Wg
    holder.Wmean_down.weight.data = Wd
    holder.Wmean_up.weight.data = Wu

    # delta_share_V=False / delta_share_U=False branch, merge_deepseek.py:421-435.
    for j in range(num_experts):
        old = block.experts[j]
        new = holder.experts[j]
        delta_gate = (old.gate_proj.weight - holder.Wmean_gate.weight)
        delta_down = (old.down_proj.weight - holder.Wmean_down.weight)
        delta_up = (old.up_proj.weight - holder.Wmean_up.weight)
        if svd_scale is not None:
            base_name = f"mlp.experts.{j}."
            new.delta_u1.weight.data, new.delta_v1.weight.data = svd_delta(delta_gate, ratio=delta_ratio, svd_scale=svd_scale[base_name + "gate_proj"], scale_type=scale_type)
            new.delta_u2.weight.data, new.delta_v2.weight.data = svd_delta(delta_down, ratio=delta_ratio, svd_scale=svd_scale[base_name + "down_proj"], scale_type=scale_type)
            new.delta_u3.weight.data, new.delta_v3.weight.data = svd_delta(delta_up, ratio=delta_ratio, svd_scale=svd_scale[base_name + "up_proj"], scale_type=scale_type)
        else:
            new.delta_u1.weight.data, new.delta_v1.weight.data = svd_delta(delta_gate, ratio=delta_ratio)
            new.delta_u2.weight.data, new.delta_v2.weight.data = svd_delta(delta_down, ratio=delta_ratio)
            new.delta_u3.weight.data, new.delta_v3.weight.data = svd_delta(delta_up, ratio=delta_ratio)
        del delta_gate, delta_down, delta_up

    r = holder.experts[0].delta_low_rank
    shapes = {n: tuple(getattr(holder.experts[0], n).weight.shape)
              for n in ("delta_u1", "delta_v1", "delta_u2", "delta_v2", "delta_u3", "delta_v3")}
    expected = {"delta_u1": (config.moe_intermediate_size, r), "delta_v1": (r, config.hidden_size),
                "delta_u2": (config.hidden_size, r), "delta_v2": (r, config.moe_intermediate_size),
                "delta_u3": (config.moe_intermediate_size, r), "delta_v3": (r, config.hidden_size)}
    for j in range(num_experts):
        for n, shp in expected.items():
            got = tuple(getattr(holder.experts[j], n).weight.shape)
            if got != shp:
                raise RuntimeError(f"expert {j} {n} shape {got} != expected {shp} (rank {r})")
    bad = _finite_report(list(compressed_tensors(holder).items()))
    if bad:
        raise RuntimeError(f"non-finite values after merge in {len(bad)} tensor(s), e.g. {bad[:5]}")

    _install(block, holder)
    after = _experts_numel(block.experts)
    return {"routed_expert_params_before": before, "routed_expert_params_after": after,
            "delta_rank": r, "delta_factor_shapes": {k: list(v) for k, v in shapes.items()},
            "zero_denominator_elements": zero_den, "svd_scale": svd_scale is not None}


def compressed_tensors(block_or_holder):
    t = {"Wmean_gate": block_or_holder.Wmean_gate.weight, "Wmean_down": block_or_holder.Wmean_down.weight,
         "Wmean_up": block_or_holder.Wmean_up.weight}
    for j, ex in enumerate(block_or_holder.experts):
        for n in ("delta_u1", "delta_v1", "delta_u2", "delta_v2", "delta_u3", "delta_v3"):
            t[f"experts.{j}.{n}"] = getattr(ex, n).weight
    return t


def tensors_digest(tensors):
    h = hashlib.sha256()
    for k in sorted(tensors):
        t = tensors[k].detach().contiguous().cpu()
        h.update(k.encode())
        h.update(str((tuple(t.shape), str(t.dtype))).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def save_compressed_block(block, path, meta):
    from safetensors.torch import save_file
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists -- refusing to overwrite a saved compressed layer")
    tensors = {k: v.detach().contiguous().cpu() for k, v in compressed_tensors(block).items()}
    digest = tensors_digest(tensors)
    tmp = path.with_suffix(".tmp")
    save_file(tensors, str(tmp), metadata={k: str(v) for k, v in meta.items()})
    tmp.rename(path)
    return {"file": path.name, "sha256_tensors": digest, "bytes": path.stat().st_size, "n_tensors": len(tensors)}


@torch.no_grad()
def load_compressed_block(block, config, path, delta_ratio=QC.DELTA_RATIO):
    """Rebuild the compressed experts of one layer from a saved file (no
    Fisher/whitening/SVD recomputation) and install them into the original
    block. Returns the tensor digest for comparison with the manifest."""
    from safetensors.torch import load_file
    device = block.gate.weight.device
    loaded = load_file(str(path), device=str(device))
    holder = _new_expert_modules(config, len(block.experts), delta_ratio, device)
    holder.Wmean_gate.weight.data = loaded["Wmean_gate"]
    holder.Wmean_down.weight.data = loaded["Wmean_down"]
    holder.Wmean_up.weight.data = loaded["Wmean_up"]
    for j, ex in enumerate(holder.experts):
        for n in ("delta_u1", "delta_v1", "delta_u2", "delta_v2", "delta_u3", "delta_v3"):
            getattr(ex, n).weight.data = loaded[f"experts.{j}.{n}"]
    expected_keys = set(compressed_tensors(holder).keys())
    if set(loaded) != expected_keys:
        raise RuntimeError(f"{path}: key set mismatch ({len(loaded)} vs {len(expected_keys)})")
    digest = tensors_digest(compressed_tensors(holder))
    _install(block, holder)
    return digest


def analytic_param_accounting(config, n_moe_layers, delta_ratio=QC.DELTA_RATIO):
    """Shape-only arithmetic (no weights needed) -- used for the pre-run
    feasibility table and cross-checked against measured counts after merge."""
    h, m, E = config.hidden_size, config.moe_intermediate_size, config.num_experts
    per_expert = 3 * h * m
    r = int(m * h * delta_ratio / (m + h))
    per_expert_delta = 3 * r * (h + m)
    before = n_moe_layers * E * per_expert
    after = n_moe_layers * (3 * h * m + E * per_expert_delta)
    return {"rank": r, "per_expert_params": per_expert, "per_expert_delta_params": per_expert_delta,
            "routed_before": before, "routed_after": after, "routed_reduction": before - after,
            "routed_reduction_pct": 100.0 * (before - after) / before}


def load_layer_stats(stage_dir, layer_idx):
    """Per-layer statistic file written by qwen15_fisher.py / qwen15_svd_scale.py."""
    p = Path(stage_dir) / f"layer_{layer_idx}.pt"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- that stage has not finished layer {layer_idx}")
    return torch.load(p, map_location="cpu")


def read_json(p):
    return json.loads(Path(p).read_text())
