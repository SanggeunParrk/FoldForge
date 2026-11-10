"""Released-checkpoint layout -> engine module layout: pure tensor re-shuffles, no module is built here.

The engine (miniworld-engine 2.5.0) has no ``from_released``; FoldForge owns the conversion of the AlphaFold 3 family's
released parameter layout (the names FoldForge's dense modules and the Haiku blob carry) to the state dict of the engine class
that serves the op. Every function maps a state dict (optionally below ``prefix``) to the engine class's own keys and moves no
value except by view / ``contiguous`` / ``chunk``: it is CPU-safe and dtype-agnostic. The offline converter
(``models/checkpoints/miniworld.py``) applies these rules once, to the dense released-layout state dict a Haiku blob loads into, and writes
the miniworld checkpoint format (engine keys); nothing at load time translates. The rules are listed here once:

    split      transition ``transition1`` [2 n c, c] -> ``expand_a`` / ``expand_b`` (``chunk(2)``); TriMul ``projection`` / ``gate``
               [2 h, c] -> ``to_left`` / ``to_right`` (+ gates) by the row interleave (row 2k = branch a of channel k, 2k + 1 =
               branch b; an incoming module takes branch b as ``left``); OPM ``output_w`` [K, K, out] ([G, K, K, out]) ->
               ``to_out.weight`` [out, G K K]; the DiT stacks' shared ``pair_logits_projection`` -> one slice of ``n_head`` rows
               per block
    broadcast  the AF3 token DiT's ONE shared ``pair_input_layer_norm`` -> every block's ``attention.ln_pair`` (and the atom
               stack's shared norm the same way)
    discard    ``DISCARDED`` below: a pair-LayerNorm offset (softmax-invariant), the K / V AdaLN of an atom block whose engine
               class shares one AdaLN, a bias the engine class does not have (RoseTTAFold3's query bias)

Released names per module are listed on each function.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from foldforge.modules.engine.modules import set_norm_eps

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import nn

    from foldforge.modules.engine.transformers import AtomDiTStack, TokenDiTStack

Tensor = torch.Tensor
State = "Mapping[str, Tensor]"

#: Released keys (regular expressions, relative to the module) that no engine class has a parameter for.
DISCARDED = (
    r"(^|\.)pair_norm\.bias$",  # MSA pair-weighted averaging: the offset adds one constant per head, softmax cancels it
    r"(^|\.)pair_input_layer_norm(\.\d+)?\.bias$",  # DiT pair LayerNorm offset (same argument)
    r"(^|\.)single_pair_logits_norm\.bias$",  # the pairformer single attention's pair LayerNorm offset
    r"(^|\.)k_adaptive_layernorm\..*$",  # atom blocks with one shared AdaLN (Boltz-2 / RoseTTAFold3 / Chai-1)
)


class Tracked(dict):
    """A released state dict that remembers which of its keys a conversion READ (``used``, full names), so a converter can demand that every
    key it did not read is a known discard (:func:`is_discarded`). Sub-views made by :func:`_sub` report into the same set."""

    def __init__(self, items: Mapping[str, Tensor], used: set[str] | None = None, full: Mapping[str, str] | None = None) -> None:
        super().__init__(items)
        self.used: set[str] = set() if used is None else used
        self.full: Mapping[str, str] = {} if full is None else full

    def __getitem__(self, key: str) -> Tensor:
        self.used.add(self.full.get(key, key))
        return super().__getitem__(key)


def _sub(state: State, prefix: str) -> dict[str, Tensor]:
    """The entries of ``state`` below ``prefix`` (a dotted path, with or without the trailing dot), prefix removed."""
    if prefix and not prefix.endswith("."):
        prefix += "."
    items = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if isinstance(state, Tracked):
        return Tracked(items, state.used, {short: state.full.get(f"{prefix}{short}", f"{prefix}{short}") for short in items})
    return items


def _need(state: State, *keys: str) -> None:
    missing = [k for k in keys if k not in state]
    if missing:
        raise KeyError(f"released state_dict lacks {missing} (has {sorted(state)[:12]}...)")


def _put(out: dict[str, Tensor], name: str, state: State, key: str) -> None:
    if key in state:
        out[name] = state[key]


def _norm(out: dict[str, Tensor], name: str, state: State, key: str, *, bias: bool = True) -> None:
    _need(state, f"{key}.weight")
    out[f"{name}.weight"] = state[f"{key}.weight"]
    if bias:
        _need(state, f"{key}.bias")
        out[f"{name}.bias"] = state[f"{key}.bias"]


def is_discarded(key: str) -> bool:
    """Whether a released key is one the engine has no parameter for (``DISCARDED``)."""
    return any(re.search(pattern, key) for pattern in DISCARDED)


# ---------------------------------------------------------------------------------------------------------------- pair-track ops
def triangle_multiplication_from_released(state: State, *, outgoing: bool, prefix: str = "") -> dict[str, Tensor]:
    """Engine ``TriangleMultiplication`` weights from the released interleaved ``projection`` / ``gate``.

    Released: ``left_norm_input.{weight,bias} projection.weight gate.weight center_norm.{weight,bias} output_projection.weight
    gating_linear.weight``. The released contraction is ``einsum("cik,cjk->cij", a, b)`` (outgoing) / ``einsum("ckj,cki->cij", a, b)``
    (incoming); the engine's incoming contracts ``left[k, i] * right[k, j]``, so an incoming module takes branch b as ``left``.
    """
    s = _sub(state, prefix)
    _need(s, "projection.weight", "gate.weight", "output_projection.weight", "gating_linear.weight")
    projection, gate = s["projection.weight"], s["gate.weight"]
    hidden = projection.shape[0] // 2
    if projection.shape[0] != 2 * hidden or gate.shape != projection.shape:
        raise ValueError(f"projection {tuple(projection.shape)} / gate {tuple(gate.shape)} must both be [2 * hidden, c_pair]")
    p = projection.reshape(hidden, 2, -1).transpose(0, 1)
    g = gate.reshape(hidden, 2, -1).transpose(0, 1)
    left, right = (0, 1) if outgoing else (1, 0)
    out: dict[str, Tensor] = {}
    _norm(out, "ln_pair", s, "left_norm_input")
    _norm(out, "ln_out", s, "center_norm")
    out["to_left.weight"], out["to_right.weight"] = p[left], p[right]
    out["to_left_gate.weight"], out["to_right_gate.weight"] = g[left], g[right]
    out["to_out.weight"] = s["output_projection.weight"]
    out["to_gate.weight"] = s["gating_linear.weight"]
    return {k: v.contiguous() for k, v in out.items()}


def triangle_attention_from_released(state: State, *, prefix: str = "") -> dict[str, Tensor]:
    """Engine ``TriangleAttention*`` weights (a rename). Released: ``act_norm pair_bias_projection q/k/v_projection gating_query
    output_projection [output_projection_transposed]``; the gate / output biases are carried when present (RoseTTAFold3, Chai-1)."""
    s = _sub(state, prefix)
    _need(s, "pair_bias_projection.weight", "q_projection.weight", "k_projection.weight", "v_projection.weight",
          "gating_query.weight", "output_projection.weight")
    out: dict[str, Tensor] = {}
    _norm(out, "ln_pair", s, "act_norm")
    for name, key in (("to_bias", "pair_bias_projection"), ("to_query", "q_projection"), ("to_key", "k_projection"),
                      ("to_value", "v_projection"), ("to_gate", "gating_query"), ("to_out", "output_projection"),
                      ("to_out_t", "output_projection_transposed")):
        _put(out, f"{name}.weight", s, f"{key}.weight")
        _put(out, f"{name}.bias", s, f"{key}.bias")
    return out


def transition_from_released(state: State, *, prefix: str = "") -> dict[str, Tensor]:
    """Engine ``Transition`` weights. Released: ``input_layer_norm.{weight,bias} transition1.weight [2 n c, c] transition2.weight``."""
    s = _sub(state, prefix)
    _need(s, "transition1.weight", "transition2.weight")
    a, b = s["transition1.weight"].chunk(2, dim=0)
    out: dict[str, Tensor] = {}
    _norm(out, "ln_in", s, "input_layer_norm")
    out["expand_a.weight"], out["expand_b.weight"] = a.contiguous(), b.contiguous()
    out["squeeze.weight"] = s["transition2.weight"]
    return out


def outer_product_mean_from_released(state: State, *, prefix: str = "") -> dict[str, Tensor]:
    """Engine ``OuterProductMean`` weights. Released: ``layer_norm_input.{weight,bias} left/right_projection.{weight[,bias]} output_w
    output_b [product_norm.{weight,bias}]``; ``output_w`` ``[K, K, out]`` (``[G, K, K, out]``) becomes ``to_out.weight`` ``[out, G K K]``."""
    s = _sub(state, prefix)
    _need(s, "left_projection.weight", "right_projection.weight", "output_w", "output_b")
    out: dict[str, Tensor] = {}
    _norm(out, "ln_msa", s, "layer_norm_input")
    for name, key in (("to_left", "left_projection"), ("to_right", "right_projection")):
        _put(out, f"{name}.weight", s, f"{key}.weight")
        _put(out, f"{name}.bias", s, f"{key}.bias")
    w = s["output_w"]
    out["to_out.weight"] = w.reshape(-1, w.shape[-1]).T.contiguous()
    out["to_out.bias"] = s["output_b"]
    if "product_norm.weight" in s:
        _norm(out, "product_norm", s, "product_norm")
    return out


def msa_pair_weighted_averaging_from_released(state: State, *, prefix: str = "") -> dict[str, Tensor]:
    """Engine ``MSAPairWeightedAveraging`` weights. Released: ``act_norm pair_norm.weight pair_logits v_projection gating_query
    output_projection`` (the pair LayerNorm offset is discarded: a per-head constant, softmax-invariant)."""
    s = _sub(state, prefix)
    _need(s, "pair_norm.weight", "pair_logits.weight", "v_projection.weight", "gating_query.weight", "output_projection.weight")
    out: dict[str, Tensor] = {}
    _norm(out, "ln_msa", s, "act_norm")
    out["ln_pair.weight"] = s["pair_norm.weight"]
    out["to_bias.weight"] = s["pair_logits.weight"]
    out["to_value.weight"] = s["v_projection.weight"]
    out["to_gate.weight"] = s["gating_query.weight"]
    out["to_out.weight"] = s["output_projection.weight"]
    return out


def attention_pair_bias_from_released(attention: State, norm: State, projection: State) -> dict[str, Tensor]:
    """Engine ``AttentionPairBias`` weights from the pairformer's single attention (``single_attention_``, AF3 form) and its pair-logit
    ``single_pair_logits_norm`` / ``single_pair_logits_projection`` (each state already stripped of its prefix; the norm's offset is
    discarded)."""
    a = attention
    _need(a, "adaptive_layernorm.layer_norm.weight", "adaptive_layernorm.layer_norm.bias", "q_projection.weight", "q_projection.bias",
          "k_projection.weight", "v_projection.weight", "gating_query.weight", "adaptive_zero_init.transition2.weight")
    _need(norm, "weight")
    _need(projection, "weight")
    return {
        "ln_single.weight": a["adaptive_layernorm.layer_norm.weight"],
        "ln_single.bias": a["adaptive_layernorm.layer_norm.bias"],
        "to_query.weight": a["q_projection.weight"],
        "to_query.bias": a["q_projection.bias"],
        "to_key.weight": a["k_projection.weight"],
        "to_value.weight": a["v_projection.weight"],
        "ln_pair.weight": norm["weight"],
        "to_bias.weight": projection["weight"],
        "to_gate.weight": a["gating_query.weight"],
        "to_out.weight": a["adaptive_zero_init.transition2.weight"],
    }


def pairformer_block_from_released(state: State, *, prefix: str = "") -> dict[str, Tensor]:
    """Engine ``PairformerBlock`` weights from a released block (pair track; the single track when its keys are there). Names:
    ``triangle_multiplication_outgoing/incoming -> tri_multi_outgoing/incoming``, ``pair_attention1/2 -> tri_atten_starting/ending``,
    ``pair_transition -> transition_pair``, ``single_attention_ + single_pair_logits_* -> single_attention``,
    ``single_transition -> single_transition``."""
    s = _sub(state, prefix)
    out: dict[str, Tensor] = {}
    for src, dst, outgoing in (("triangle_multiplication_outgoing", "tri_multi_outgoing", True),
                               ("triangle_multiplication_incoming", "tri_multi_incoming", False)):
        out |= {f"{dst}.{k}": v for k, v in triangle_multiplication_from_released(s, outgoing=outgoing, prefix=src).items()}
    if any(k.startswith("pair_attention1.") for k in s):
        for src, dst in (("pair_attention1", "tri_atten_starting"), ("pair_attention2", "tri_atten_ending")):
            out |= {f"{dst}.{k}": v for k, v in triangle_attention_from_released(s, prefix=src).items()}
    out |= {f"transition_pair.{k}": v for k, v in transition_from_released(s, prefix="pair_transition").items()}
    if any(k.startswith("single_attention_.") for k in s):
        out |= {f"single_attention.{k}": v for k, v in attention_pair_bias_from_released(
            _sub(s, "single_attention_"), _sub(s, "single_pair_logits_norm"), _sub(s, "single_pair_logits_projection")).items()}
        out |= {f"single_transition.{k}": v for k, v in transition_from_released(s, prefix="single_transition").items()}
    return out


# ---------------------------------------------------------------------------------------------------------------- diffusion blocks
def _adaln(s: State, key: str, name: str, out: dict[str, Tensor]) -> bool:
    """Engine ``AdaptiveLayerNorm`` / ``IdentityAdaptiveLayerNorm`` tensors of the released ``AdaptiveLayerNorm`` at ``key``
    (``single_cond_layer_norm.weight``, ``single_cond_scale.{weight[,bias]}``, ``single_cond_bias.weight``); returns whether it is the
    identity-centred form (no ``single_cond_layer_norm``: Chai-1's)."""
    identity = f"{key}.single_cond_layer_norm.weight" not in s
    _need(s, f"{key}.single_cond_scale.weight", f"{key}.single_cond_bias.weight")
    out[f"{name}.to_scale.weight"] = s[f"{key}.single_cond_scale.weight"]
    out[f"{name}.to_bias.weight"] = s[f"{key}.single_cond_bias.weight"]
    if not identity:
        out[f"{name}.ln_cond.weight"] = s[f"{key}.single_cond_layer_norm.weight"]
        _need(s, f"{key}.single_cond_scale.bias")
        out[f"{name}.to_scale.bias"] = s[f"{key}.single_cond_scale.bias"]
    return identity


def _conditioned_transition(s: State, key: str, prefix: str, out: dict[str, Tensor]) -> None:
    """Released ``DiffusionTransition`` / ``GatedDiffusionTransition`` at ``key`` -> the engine ``(UpGated)ConditionedTransition``
    (``ada_ln_in``, ``expand_a/b`` (``chunk(2)`` of ``transition1``), ``up`` (``a_to_b``), ``squeeze``, ``to_scale``)."""
    _need(s, f"{key}.transition1.weight", f"{key}.adaptive_zero_init.transition2.weight",
          f"{key}.adaptive_zero_init.adaptive_zero_cond.weight", f"{key}.adaptive_zero_init.adaptive_zero_cond.bias")
    a, b = s[f"{key}.transition1.weight"].chunk(2, dim=0)
    out[f"{prefix}.expand_a.weight"], out[f"{prefix}.expand_b.weight"] = a.contiguous(), b.contiguous()
    out[f"{prefix}.squeeze.weight"] = s[f"{key}.adaptive_zero_init.transition2.weight"]
    out[f"{prefix}.to_scale.weight"] = s[f"{key}.adaptive_zero_init.adaptive_zero_cond.weight"]
    out[f"{prefix}.to_scale.bias"] = s[f"{key}.adaptive_zero_init.adaptive_zero_cond.bias"]
    _adaln(s, f"{key}.adaptive_layernorm", f"{prefix}.ada_ln_in", out)
    _put(out, f"{prefix}.up.weight", s, f"{key}.a_to_b.weight")


def _attention(s: State, key: str, prefix: str, out: dict[str, Tensor]) -> None:
    """Released ``SelfAttention`` / cross attention at ``key`` -> the engine ``AugmentedAttentionPairBias`` (without the pair bias and
    the AdaLN, which the callers own). ``query_layer_norm`` / ``key_layer_norm`` are RoseTTAFold3's flat q / k LayerNorms."""
    for name, src in (("to_query", "q_projection"), ("to_key", "k_projection"), ("to_value", "v_projection"), ("to_gate", "gating_query")):
        _put(out, f"{prefix}.{name}.weight", s, f"{key}.{src}.weight")
    _put(out, f"{prefix}.to_query.bias", s, f"{key}.q_projection.bias")
    _need(s, f"{key}.adaptive_zero_init.adaptive_zero_cond.weight", f"{key}.adaptive_zero_init.adaptive_zero_cond.bias")
    out[f"{prefix}.to_scale.weight"] = s[f"{key}.adaptive_zero_init.adaptive_zero_cond.weight"]
    out[f"{prefix}.to_scale.bias"] = s[f"{key}.adaptive_zero_init.adaptive_zero_cond.bias"]
    _put(out, f"{prefix}.to_out.weight", s, f"{key}.adaptive_zero_init.transition2.weight")
    for name, src in (("norm_query", "query_layer_norm"), ("norm_key", "key_layer_norm")):
        _put(out, f"{prefix}.{name}.weight", s, f"{key}.{src}.weight")
        _put(out, f"{prefix}.{name}.bias", s, f"{key}.{src}.bias")


def count_blocks(state: State, stem: str) -> int:
    """Number of released blocks ``{stem}.{i}.*`` in a state dict."""
    indices = {int(m.group(1)) for k in state if (m := re.match(rf"{re.escape(stem)}\.(\d+)\.", k))}
    return 1 + max(indices) if indices else 0


def token_dit_blocks_from_released(state: State, *, n_head: int, super_block_size: int = 4, prefix: str = "") -> list[dict[str, Tensor]]:
    """One engine ``TokenDiT*`` block state dict per block of a released token ``DiffusionTransformer``.

    Released: ``self_attention.i.*``, ``transition_block.i.*``, ``pair_input_layer_norm`` (AF3: ONE shared norm; per-block families:
    ``pair_input_layer_norm.i``) and ``pair_logits_projection`` (AF3: ``pair_logits_projection.j`` of ``4 * n_head`` rows per super
    block of ``super_block_size`` blocks; per-block families: ``pair_logits_projection.i`` of ``n_head`` rows). The shared norm is
    broadcast to every block, the shared projection sliced per block, the norm offset discarded. A key the block class has no
    parameter for (``to_gate`` of Chai-1) is simply absent from the released state."""
    s = _sub(state, prefix)
    per_block = "pair_input_layer_norm.0.weight" in s
    blocks: list[dict[str, Tensor]] = []
    for i in range(count_blocks(s, "self_attention")):
        out: dict[str, Tensor] = {}
        _adaln(s, f"self_attention.{i}.adaptive_layernorm", "attention.ada_ln_in", out)
        _attention(s, f"self_attention.{i}", "attention", out)
        _conditioned_transition(s, f"transition_block.{i}", "transition", out)
        out["attention.ln_pair.weight"] = s[f"pair_input_layer_norm.{i}.weight" if per_block else "pair_input_layer_norm.weight"]
        if per_block:
            proj = s[f"pair_logits_projection.{i}.weight"]
        else:
            super_block, offset = divmod(i, super_block_size)
            proj = s[f"pair_logits_projection.{super_block}.weight"][offset * n_head : (offset + 1) * n_head]
        out["attention.to_bias.weight"] = proj
        blocks.append({k: v.contiguous() for k, v in out.items()})
    return blocks


def _check_shared_kv_adaln(s: State, key: str, index: int) -> None:
    """Block classes with ONE AdaLN for queries and keys / values (``kv_norm == "shared"``: Boltz-2, RoseTTAFold3, Chai-1): the released
    ``k_adaptive_layernorm`` may be discarded ONLY if it equals ``q_adaptive_layernorm`` exactly; any difference is a conversion error."""
    for name in ("single_cond_layer_norm.weight", "single_cond_scale.weight", "single_cond_scale.bias", "single_cond_bias.weight"):
        q, k = f"{key}.q_adaptive_layernorm.{name}", f"{key}.k_adaptive_layernorm.{name}"
        if (q in s) != (k in s):
            raise ValueError(f"block {index}: {q if q in s else k} has no counterpart; the K / V AdaLN cannot be discarded")
        if q in s and not torch.equal(s[q], s[k]):
            diff = (s[q].double() - s[k].double()).abs().max().item()
            raise ValueError(f"block {index}: {k} differs from {q} (max |d| {diff:.3e}); the engine class shares one AdaLN")


def atom_dit_blocks_from_released(state: State, *, n_head: int, kv_norm: str, prefix: str = "") -> list[dict[str, Tensor]]:
    """One engine ``AtomDiT*`` block state dict per block of a released ``DiffusionCrossAttTransformer``.

    Released: ``cross_attention.i.{q,k}_adaptive_layernorm`` (the query AdaLN -> ``attention.ada_ln_in``, the key / value AdaLN ->
    ``attention.ada_ln_kv`` unless the class shares one AdaLN, ``kv_norm == "shared"``, when it is discarded), ``transition_block.i``,
    ``pair_input_layer_norm`` / ``pair_logits_projection`` (one shared Linear of ``n_blocks * n_head`` rows, or per block)."""
    if kv_norm not in ("parallel", "chained", "shared"):
        raise ValueError(f"kv_norm {kv_norm!r}")
    s = _sub(state, prefix)
    per_block = "pair_input_layer_norm.0.weight" in s
    n_blocks = count_blocks(s, "cross_attention")
    blocks: list[dict[str, Tensor]] = []
    for i in range(n_blocks):
        out: dict[str, Tensor] = {}
        key = f"cross_attention.{i}"
        _adaln(s, f"{key}.q_adaptive_layernorm", "attention.ada_ln_in", out)
        if kv_norm != "shared":
            _adaln(s, f"{key}.k_adaptive_layernorm", "attention.ada_ln_kv", out)
        else:
            _check_shared_kv_adaln(s, key, i)
        _attention(s, key, "attention", out)
        _conditioned_transition(s, f"transition_block.{i}", "transition", out)
        out["attention.ln_pair.weight"] = s[f"pair_input_layer_norm.{i}.weight" if per_block else "pair_input_layer_norm.weight"]
        proj = s[f"pair_logits_projection.{i}.weight"] if per_block else s["pair_logits_projection.weight"][i * n_head : (i + 1) * n_head]
        out["attention.to_bias.weight"] = proj
        blocks.append({k: v.contiguous() for k, v in out.items()})
    return blocks


# ---------------------------------------------------------------------------------------------------------------- loading
def fit_to_module(module: nn.Module, converted: Mapping[str, Tensor]) -> tuple[dict[str, Tensor], list[str]]:
    """``(kept, dropped)``: ``converted`` restricted to the keys ``module`` has; ``dropped`` names the rest, so a caller can demand that
    every dropped key is a known discard (RoseTTAFold3's absent query bias, Chai-1's absent output projection) before a strict load."""
    own = module.state_dict().keys()
    kept = {k: v for k, v in converted.items() if k in own}
    return kept, sorted(set(converted) - set(kept))


@torch.no_grad()
def load(module: nn.Module, converted: Mapping[str, Tensor], *, norm_eps: float | None = None, allow_dropped: tuple[str, ...] = ()) -> nn.Module:
    """Strict load of converted tensors into ``module`` keeping the module's parameter dtypes (fp32 masters stay fp32: the released values
    are copied in, never rounded to bf16). ``allow_dropped``: keys of ``converted`` the module may lack (checked against
    :func:`fit_to_module`)."""
    kept, dropped = fit_to_module(module, converted)
    unexpected = [k for k in dropped if k not in allow_dropped]
    if unexpected:
        raise KeyError(f"{type(module).__name__} has no parameter for {unexpected}")
    module.load_state_dict(kept, strict=True)
    if norm_eps is not None:
        set_norm_eps(module, norm_eps)
    return module


def diffusion_transition_from_released(state: State, *, prefix: str = "") -> dict[str, Tensor]:
    """Engine ``Transition`` weights from a released UNCONDITIONED ``DiffusionTransition`` (the diffusion head's pair / single
    conditioning transitions): ``adaptive_layernorm.layer_norm.{weight,bias}``, ``transition1.weight`` ``[2 n c, c]`` (``chunk(2)`` ->
    ``expand_a`` / ``expand_b``), ``adaptive_zero_init.transition2.weight`` -> ``squeeze``."""
    s = _sub(state, prefix)
    _need(s, "transition1.weight", "adaptive_zero_init.transition2.weight")
    a, b = s["transition1.weight"].chunk(2, dim=0)
    out: dict[str, Tensor] = {}
    _norm(out, "ln_in", s, "adaptive_layernorm.layer_norm")
    out["expand_a.weight"], out["expand_b.weight"] = a.contiguous(), b.contiguous()
    out["squeeze.weight"] = s["adaptive_zero_init.transition2.weight"]
    return out


def evoformer_block_from_released(state: State, *, prefix: str = "") -> dict[str, Tensor]:
    """Engine parts of an MSA-stack block (``graph.EvoformerBlock``, same attribute names as the released block) from the released
    ``EvoformerBlock``: ``outer_product_mean``, ``msa_attention1`` (pair-weighted averaging), ``msa_transition``,
    ``triangle_multiplication_outgoing`` / ``incoming``, ``pair_attention1`` / ``2`` (when present), ``pair_transition``."""
    s = _sub(state, prefix)
    out: dict[str, Tensor] = {}

    def put(name: str, tensors: dict[str, Tensor]) -> None:
        out.update({f"{name}.{k}": v for k, v in tensors.items()})

    put("outer_product_mean", outer_product_mean_from_released(s, prefix="outer_product_mean"))
    put("msa_attention1", msa_pair_weighted_averaging_from_released(s, prefix="msa_attention1"))
    put("msa_transition", transition_from_released(s, prefix="msa_transition"))
    put("triangle_multiplication_outgoing", triangle_multiplication_from_released(s, outgoing=True, prefix="triangle_multiplication_outgoing"))
    put("triangle_multiplication_incoming", triangle_multiplication_from_released(s, outgoing=False, prefix="triangle_multiplication_incoming"))
    for name in ("pair_attention1", "pair_attention2"):
        if any(k.startswith(f"{name}.") for k in s):
            put(name, triangle_attention_from_released(s, prefix=name))
    put("pair_transition", transition_from_released(s, prefix="pair_transition"))
    return out


# ---------------------------------------------------------------------------------------------------------- stack loading (tests)
#: A key the released state may carry that a block class has no parameter for, and the check that makes dropping it exact.
_DROPPABLE_ZERO = ("attention.to_query.bias",)  # RoseTTAFold3's bias-free query (released as a zero bias)


def _load_blocks(stack: nn.Module, converted: list[dict[str, Tensor]], norm_eps: float | None) -> nn.Module:
    if len(converted) != len(stack.blocks):
        raise ValueError(f"released state has {len(converted)} blocks, the stack {len(stack.blocks)}")
    for index, (block, tensors) in enumerate(zip(stack.blocks, converted, strict=True)):
        own = block.state_dict().keys()
        for key in _DROPPABLE_ZERO:
            if key in tensors and key not in own and bool(tensors[key].any()):
                raise ValueError(f"block {index}: {key} is not zero, the engine class has no such parameter")
        load(block, tensors, norm_eps=norm_eps, allow_dropped=_DROPPABLE_ZERO)
    return stack


def load_token_stack(
    stack: TokenDiTStack, state: Mapping[str, Tensor], *, n_head: int, super_block_size: int = 4, prefix: str = "", norm_eps: float | None = None
) -> TokenDiTStack:
    """Strict load of a released token ``DiffusionTransformer`` (see :func:`~released.token_dit_blocks_from_released`)."""
    _load_blocks(stack, token_dit_blocks_from_released(state, n_head=n_head, super_block_size=super_block_size, prefix=prefix), norm_eps)
    return stack


def load_atom_stack(
    stack: AtomDiTStack, state: Mapping[str, Tensor], *, n_head: int = 4, prefix: str = "", norm_eps: float | None = None
) -> AtomDiTStack:
    """Strict load of a released ``DiffusionCrossAttTransformer`` (the K / V AdaLN follows the block class's ``kv_norm``)."""
    kv_norm = stack.blocks[0].kv_norm
    _load_blocks(stack, atom_dit_blocks_from_released(state, n_head=n_head, kv_norm=kv_norm, prefix=prefix), norm_eps)
    return stack
