# Copyright 2024 xfold authors
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
"""The conditioned (adaptive) layers and the atom cross attention transformer of FoldForge's own atom stack.

ESMFold2's atom stack is FoldForge-own (the engine's ESMFold2 atom block is pair-free sliding-window attention; FoldForge's has an atom pair and
rotary q / k: ``docs/ENGINE-UNSUPPORTED.md``), so :class:`DiffusionCrossAttTransformer` is a runtime module for that family. The released
dense token transformer (``DiffusionTransformer``) that the offline converter reads Haiku blobs into lives in
``foldforge.models.checkpoints.legacy``.
"""

import functools
from collections.abc import Callable
from typing import Any

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from foldforge.modules.dense.spec import ALPHAFOLD3, DenseSpec
from foldforge.utils import atom_layout

Tensor = torch.Tensor
Projection = Callable[[Tensor], Tensor]


def swiglu(x: Tensor, project_a: Projection, project_b: Projection | None = None) -> Tensor:
    """``silu(a) * b`` from one packed projection (chunked) or two separate checkpoint projections."""
    if project_b is None:
        a, b = project_a(x).chunk(2, dim=-1)
    else:
        a, b = project_a(x), project_b(x)
    return F.silu(a) * b


def adaptive_normalization(
    x: Tensor,
    cond: Tensor | None,
    *,
    normalize: Projection,
    normalize_cond: Projection | None = None,
    scale: Projection | None = None,
    shift: Projection | None = None,
    identity_scale: bool = False,
    activate_cond: bool = False,
) -> Tensor:
    """Conditioned normalization, with the original separate normalization steps.

    ``identity_scale`` selects the identity-centred form, ``(s + 1) * x``, over the sigmoid one. Both start near the identity at
    initialisation, so which a checkpoint was trained under is invisible until its weights are loaded. Families on that form also leave the
    conditioning unnormalised, which is why ``normalize_cond`` may be absent here and is not a no-op when it is not: a LayerNorm re-centres
    and re-scales even at unit scale.

    ``activate_cond`` passes the conditioning through SiLU before the scale and shift read it, which one family's atom blocks do in place of
    normalising it.
    """
    value = normalize(x)
    if cond is None:
        return value
    if scale is None or shift is None:
        msg = "Conditioned normalization requires a scale and a shift"
        raise ValueError(msg)
    if normalize_cond is None and not identity_scale:
        msg = "Sigmoid conditioning requires the condition norm"
        raise ValueError(msg)
    context = cond if normalize_cond is None else normalize_cond(cond)
    if activate_cond:
        context = F.silu(context)
    gate = scale(context) + 1.0 if identity_scale else torch.sigmoid(scale(context))
    return gate * value + shift(context)


def conditioned_residual(
    single: Tensor,
    cond: Tensor,
    *,
    attention: Callable[[Tensor], Tensor],
    transition: Callable[[Tensor, Tensor], Tensor],
    residual_path: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Compose raw attention and transition deltas with one residual each (``residual_path`` wraps each delta, e.g. dropout)."""
    delta = attention(single)
    single = single + (delta if residual_path is None else residual_path(delta))
    delta = transition(single, cond)
    return single + (delta if residual_path is None else residual_path(delta))


def reference_adaln(module: nn.Module, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
    """Bind optional single-conditioning to the shared adaptive normalization."""
    conditioned = module.use_single_cond
    if (cond is not None) != conditioned:
        message = "Condition presence must match the checkpoint configuration"
        raise ValueError(message)
    return adaptive_normalization(
        x,
        cond,
        normalize=module.layer_norm,
        normalize_cond=getattr(module, "single_cond_layer_norm", None) if conditioned else None,
        scale=module.single_cond_scale if conditioned else None,
        shift=module.single_cond_bias if conditioned else None,
        identity_scale=getattr(module, "identity_scale", False),
        activate_cond=getattr(module, "activate_cond", False),
    )


def reference_conditioned_transition(module: nn.Module, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
    """Bind the transition layout; normalization, SwiGLU and gating are shared."""
    hidden = swiglu(
        module.adaptive_layernorm(x, cond),
        lambda value: value.to(module.transition1.weight.dtype) @ module.transition1.weight.T,
        None,
    )
    return module.adaptive_zero_init(hidden, cond)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary embedding with TILED cos and sin -- [c|c], not interleaved.

    The tiling pairs with the split-into-halves rotation below. Interleaving
    instead reads correlation 0.88 on the atom encoder: high enough to look like
    noise, low enough to ruin the fold.
    """
    width = cos.shape[-1] * 2
    c = torch.concatenate([cos, cos], dim=-1)[..., None, :]
    s = torch.concatenate([sin, sin], dim=-1)[..., None, :]
    a, b = torch.split(x[..., :width], width // 2, dim=-1)
    rotated = torch.concatenate([-b, a], dim=-1)
    return torch.concatenate([x[..., :width] * c + rotated * s, x[..., width:]], dim=-1)


class RMSNorm(nn.Module):
    """Affine-free RMSNorm at float32 machine epsilon.

    Not `F.rms_norm(eps=None)`, which uses the eps of the INPUT dtype: under
    bf16 that is 7.8e-3 rather than 1.2e-7, which is a different function.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the module output."""
        eps = torch.finfo(torch.float32).eps
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class AdaptiveLayerNorm(nn.Module):
    """Represent adaptive layer norm."""

    def __init__(
        self,
        c_x: int,
        c_single_cond: int | None,
        use_single_cond: bool = False,
        identity_scale: bool = False,
        eps: float = 1e-5,
        rms_norm: bool = False,
        activate_cond: bool = False,
    ) -> None:

        super().__init__()

        #: The activation is normalised with an affine-free RMSNorm and the
        #: conditioning passes through SiLU rather than a LayerNorm. Neither
        #: carries a parameter, so both are invisible to a tree diff.
        self.rms_norm = rms_norm
        self.activate_cond = activate_cond
        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.use_single_cond = use_single_cond
        #: The (s + 1) form leaves the conditioning unnormalised and its scale
        #: projection biasless; there is no norm here to map and keeping one at
        #: unit scale would still re-centre and re-scale.
        self.identity_scale = identity_scale

        if self.use_single_cond is True:
            if self.c_single_cond is None:
                message = "Conditioned layers require a conditioning channel count"
                raise ValueError(message)
            self.layer_norm = (
                RMSNorm()
                if rms_norm
                else nn.LayerNorm(
                    self.c_x, eps=eps, elementwise_affine=False, bias=False
                )
            )
            if not identity_scale:
                self.single_cond_layer_norm = nn.LayerNorm(
                    self.c_single_cond, bias=False
                )
            self.single_cond_scale = nn.Linear(
                self.c_single_cond, self.c_x, bias=not identity_scale
            )
            self.single_cond_bias = nn.Linear(self.c_single_cond, self.c_x, bias=False)
        else:
            self.layer_norm = nn.LayerNorm(self.c_x, eps=eps)

    def forward(
        self, x: torch.Tensor, single_cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the module output."""
        return reference_adaln(self, x, single_cond)


class AdaLNZero(nn.Module):
    """Represent ada l n zero."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        c_single_cond: int | None,
        use_single_cond: bool = False,
        project: bool = True,
        zero_bias: bool = True,
        raw_gate: bool = False,
    ) -> None:
        super().__init__()

        self.raw_gate = raw_gate
        self.c_in = c_in
        self.c_out = c_out
        self.c_single_cond = c_single_cond
        self.use_single_cond = use_single_cond
        #: Without the projection the raw concatenated heads are multiplied by
        #: the conditioning gate and that is the whole output.
        self.project = project

        if project:
            self.transition2 = nn.Linear(self.c_in, self.c_out, bias=False)
        if self.use_single_cond is True:
            if self.c_single_cond is None:
                message = "Conditioned layers require a conditioning channel count"
                raise ValueError(message)
            self.adaptive_zero_cond = nn.Linear(
                self.c_single_cond, self.c_out, bias=zero_bias
            )

    def forward(
        self,
        x: torch.Tensor,
        single_cond: torch.Tensor | None = None,
        gate_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the module output."""
        if (single_cond is None) != (self.use_single_cond is False):
            message = (
                "Invalid state: (single_cond is None) == (self.use_single_cond is"
                " False)"
            )
            raise ValueError(message)

        if not self.project:
            if single_cond is None:
                message = "A gate-only output needs its conditioning"
                raise ValueError(message)
            return torch.sigmoid(self.adaptive_zero_cond(single_cond)) * x
        output = (
            self.transition2(x)
            if gate_logits is None
            else self.transition2(torch.sigmoid(gate_logits) * x)
        )
        if self.use_single_cond is True:
            if self.c_single_cond is None:
                message = "Conditioned layers require a conditioning channel count"
                raise ValueError(message)
            # A gate taken RAW, from a conditioning already through SiLU. Both
            # halves matter: the vendor's fused modulation produces the scale,
            # the shift AND the gate from one projection of silu(c), so the
            # gate reads the SAME activated conditioning the scale does. AF3
            # squashes its gate through a sigmoid and offsets it by -2, and no
            # weight can undo either -- a sigmoid does not fold into a linear
            # map.
            context = F.silu(single_cond) if self.raw_gate else single_cond
            cond = self.adaptive_zero_cond(context)
            output = (cond if self.raw_gate else torch.sigmoid(cond)) * output
        return output


class DiffusionTransition(nn.Module):
    """Represent diffusion transition."""

    def __init__(
        self,
        c_x: int,
        c_single_cond: int | None,
        num_intermediate_factor: int = 2,
        use_single_cond: bool = False,
        identity_scale: bool = False,
        norm_eps: float = 1e-5,
        zero_bias: bool = True,
        rms_conditioning: bool = False,
    ) -> None:
        super().__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.num_intermediate_factor = num_intermediate_factor
        self.use_single_cond = use_single_cond

        self.adaptive_layernorm = AdaptiveLayerNorm(
            self.c_x,
            self.c_single_cond,
            self.use_single_cond,
            identity_scale=identity_scale,
            eps=norm_eps,
            rms_norm=rms_conditioning,
            activate_cond=rms_conditioning,
        )
        self.transition1 = nn.Linear(
            self.c_x, 2 * self.c_x * self.num_intermediate_factor, bias=False
        )

        self.adaptive_zero_init = AdaLNZero(
            self.num_intermediate_factor * self.c_x,
            self.c_x,
            self.c_single_cond,
            self.use_single_cond,
            zero_bias=zero_bias,
            raw_gate=rms_conditioning,
        )

    def forward(
        self, x: torch.Tensor, single_cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the module output."""
        return reference_conditioned_transition(self, x, single_cond)


class GatedDiffusionTransition(DiffusionTransition):
    """Conditioned transition whose SwiGLU output passes a linear up-gate.

    ``b = swiglu(norm(x)) * a_to_b(norm(x))`` before the zero-initialised output.
    It is a class of its own so the engine's conditioned-transition wrapper, which
    has no slot for the gate, leaves it alone and converts only its AdaLN.
    """

    def __init__(self, c_x: int, c_single_cond: int | None, **options: Any) -> None:
        super().__init__(c_x, c_single_cond, **options)
        self.a_to_b = nn.Linear(c_x, self.num_intermediate_factor * c_x, bias=False)

    def forward(
        self, x: torch.Tensor, single_cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the module output."""
        normed = self.adaptive_layernorm(x, single_cond)
        gate, value = self.transition1(normed).chunk(2, dim=-1)
        hidden = torch.nn.functional.silu(gate) * value * self.a_to_b(normed)
        return self.adaptive_zero_init(hidden, single_cond)


def conditioned_transition(spec: DenseSpec) -> type[DiffusionTransition]:
    """Transition class for blocks conditioned on a single representation."""
    return GatedDiffusionTransition if spec.transition_up_gate else DiffusionTransition


class CrossAttention(nn.Module):
    """Represent cross attention."""

    def __init__(
        self,
        key_dim: int = 128,
        value_dim: int = 128,
        c_single_cond: int = 128,
        num_head: int = 4,
        kq_norm: bool = False,
        identity_scale: bool = False,
        norm_eps: float = 1e-5,
        gating_query: bool = True,
        project_output: bool = True,
        zero_bias: bool = True,
        rms_conditioning: bool = False,
        key_masked: bool = False,
    ) -> None:
        super().__init__()

        self.kq_norm = kq_norm
        self.key_masked = key_masked
        self.use_gating_query = gating_query
        if kq_norm:
            self.query_layer_norm = nn.LayerNorm(key_dim)
            self.key_layer_norm = nn.LayerNorm(key_dim)
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.c_single_cond = c_single_cond
        self.num_head = num_head

        self.key_dim_per_head = self.key_dim // self.num_head
        self.value_dim_per_head = self.value_dim // self.num_head

        self.q_scale = self.key_dim_per_head ** (-0.5)
        self.rope_norm = RMSNorm()

        adaln = {
            "c_x": self.key_dim,
            "c_single_cond": self.c_single_cond,
            "use_single_cond": True,
            "identity_scale": identity_scale,
            "eps": norm_eps,
            "rms_norm": rms_conditioning,
            "activate_cond": rms_conditioning,
        }
        self.q_adaptive_layernorm = AdaptiveLayerNorm(**adaln)
        self.k_adaptive_layernorm = AdaptiveLayerNorm(**adaln)

        self.q_projection = nn.Linear(self.key_dim, self.key_dim, bias=True)
        self.k_projection = nn.Linear(self.key_dim, self.key_dim, bias=False)
        self.v_projection = nn.Linear(self.value_dim, self.value_dim, bias=False)

        if self.use_gating_query:
            self.gating_query = nn.Linear(self.key_dim, self.value_dim, bias=False)
        self.adaptive_zero_init = AdaLNZero(
            self.value_dim,
            self.value_dim,
            self.key_dim,
            use_single_cond=True,
            project=project_output,
            zero_bias=zero_bias,
            raw_gate=rms_conditioning,
        )

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        mask_q: torch.Tensor,
        mask_k: torch.Tensor,
        pair_logits: torch.Tensor | None = None,
        single_cond_q: torch.Tensor | None = None,
        single_cond_k: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
        keys_from_queries: Callable[[torch.Tensor], torch.Tensor] | None = None,
        rope_q: tuple[torch.Tensor, torch.Tensor] | None = None,
        rope_k: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute the module output.

        ``keys_from_queries`` CHAINS the two adaptive normalisations instead of
        running them in parallel: the keys are gathered from the already-normed
        queries, so the key norm re-centres and re-scales a tensor whose
        statistics the query norm has already fixed, and sees its learned scale.
        Two chained LayerNorms are not one, and they are not two parallel ones.
        The gather has to happen BETWEEN them, which is why it arrives as a
        callback rather than a ready-made key tensor.
        """
        if tuple(mask_q.shape) != tuple(x_q.shape[-mask_q.ndim - 1 : -1]):
            message = f"{mask_q.shape}, {x_q.shape}"
            raise ValueError(message)
        if tuple(mask_k.shape) != tuple(x_k.shape[-mask_k.ndim - 1 : -1]):
            message = f"{mask_k.shape}, {x_k.shape}"
            raise ValueError(message)

        if self.key_masked:
            # The exact mode's OR form: a padded key is masked from every query.
            # AF3's AND form is only safe because it slides its key window
            # inside the real atoms; the two fold alike (a fast-mode unification).
            bias = -1e9 * (
                mask_q.logical_not()[..., None, :, None].float()
                + mask_k.logical_not()[..., None, None, :].float()
            )
        else:
            bias = (
                1e9
                * mask_q.logical_not()[..., None, :, None]
                * mask_k.logical_not()[..., None, None, :]
            )

        if pair_mask is not None:
            # A family may open its atom attention only within a token. AF3 lets
            # every atom attend across the whole window, which spreads the
            # softmax over some eighty keys where such a family opens nine --
            # an attenuated average whose damage is intra-residue geometry.
            bias = torch.where(pair_mask[..., None, :, :], bias, -1e9)

        x_q = self.q_adaptive_layernorm(x_q, single_cond_q)
        if keys_from_queries is not None:
            x_k = keys_from_queries(x_q)
        x_k = self.k_adaptive_layernorm(x_k, single_cond_k)

        q = self.q_projection(x_q)
        k = self.k_projection(x_k)
        if self.kq_norm:
            q, k = self.query_layer_norm(q), self.key_layer_norm(k)
        q = torch.reshape(q, (*q.shape[:-1], self.num_head, self.key_dim_per_head))
        k = torch.reshape(k, (*k.shape[:-1], self.num_head, self.key_dim_per_head))

        if rope_q is not None and rope_k is not None:
            # The family's ENTIRE positional signal, applied after an
            # affine-free RMSNorm. Queries and keys carry DIFFERENT rotations
            # because they are different subsets of the same flat atom list.
            q = apply_rope(self.rope_norm(q.float()), *rope_q)
            k = apply_rope(self.rope_norm(k.float()), *rope_k)

        logits = (
            # GEMM operands follow native projection precision. Keep masking and
            # softmax in FP32, then restore V's dtype for the weighted sum.
            torch.einsum("...qhc,...khc->...hqk", q * self.q_scale, k).float() + bias
        )
        if pair_logits is not None:
            logits += pair_logits
        weights = torch.softmax(logits, dim=-1)

        v = self.v_projection(x_k)
        v = torch.reshape(v, (*v.shape[:-1], self.num_head, self.value_dim_per_head))
        weighted_avg = torch.einsum("...hqk,...khc->...qhc", weights.to(v.dtype), v)
        weighted_avg = torch.reshape(weighted_avg, (*weighted_avg.shape[:-2], -1))

        gate_logits = self.gating_query(x_q) if self.use_gating_query else None
        return self.adaptive_zero_init(
            weighted_avg, single_cond_q, gate_logits=gate_logits
        )


class DiffusionCrossAttTransformer(nn.Module):
    """Represent diffusion cross att transformer."""

    def __init__(
        self,
        c_query: int = 128,
        c_single_cond: int = 128,
        c_pair_cond: int = 16,
        num_blocks: int = 3,
        num_head: int = 4,
        spec: DenseSpec = ALPHAFOLD3,
    ) -> None:
        super().__init__()

        self.c_query = c_query
        self.c_single_cond = c_single_cond
        self.c_pair_cond = c_pair_cond

        self.num_blocks = num_blocks
        self.num_head = num_head
        #: Per-head width of the atom attention; the rotary tables are built to
        #: half of it.
        self.key_dim_per_head = c_query // num_head

        self.per_block_pair = spec.per_block_atom_pair_layer_norm
        self.parallel = spec.parallel_attention_transition
        self.chained_key_norm = spec.chained_atom_key_norm
        self.key_window = spec.atom_key_window
        self.keys_subset = spec.atom_keys_subset
        #: ESMFold2: the rank-based sliding window (the only pair_mask it builds) and the
        #: rotary q / k; the engine block reproduces both (``engine.modules.atom_blocks``).
        self.window_half = spec.atom_window_half
        self.rope = spec.atom_rope is not None
        if self.per_block_pair:
            self.pair_input_layer_norm = nn.ModuleList(
                [
                    nn.LayerNorm(
                        self.c_pair_cond,
                        bias="pair_input_layer_norm" in spec.affine_norms,
                    )
                    for _ in range(self.num_blocks)
                ]
            )
            self.pair_logits_projection = nn.ModuleList(
                [
                    nn.Linear(self.c_pair_cond, self.num_head, bias=False)
                    for _ in range(self.num_blocks)
                ]
            )
        else:
            self.pair_input_layer_norm = nn.LayerNorm(
                self.c_pair_cond,
                bias="pair_input_layer_norm" in spec.affine_norms,
            )
            self.pair_logits_projection = nn.Linear(
                self.c_pair_cond, self.num_blocks * self.num_head, bias=False
            )

        atom_identity = (
            spec.adaptive_identity_scale
            if spec.atom_adaptive_identity_scale is None
            else spec.atom_adaptive_identity_scale
        )
        self.cross_attention = nn.ModuleList(
            [
                CrossAttention(
                    num_head=self.num_head,
                    kq_norm=spec.attention_kq_norm,
                    identity_scale=atom_identity,
                    gating_query=spec.atom_attention_gating_query,
                    project_output=spec.atom_attention_project_output,
                    zero_bias=spec.atom_adaptive_zero_bias,
                    rms_conditioning=spec.atom_rms_conditioning,
                    key_masked=spec.key_masked_atom_attention,
                    norm_eps=spec.adaptive_norm_eps,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.transition_block = nn.ModuleList(
            [
                conditioned_transition(spec)(
                    c_x=self.c_query,
                    c_single_cond=self.c_single_cond,
                    use_single_cond=True,
                    identity_scale=atom_identity,
                    zero_bias=spec.atom_adaptive_zero_bias,
                    rms_conditioning=spec.atom_rms_conditioning,
                    norm_eps=spec.adaptive_norm_eps,
                )
                for _ in range(self.num_blocks)
            ]
        )

    def forward(
        self,
        queries_act: torch.Tensor,  # (num_subsets, num_queries, ch)
        queries_mask: torch.Tensor,  # (num_subsets, num_queries)
        queries_to_keys: atom_layout.GatherInfo,  # (num_subsets, num_keys)
        keys_mask: torch.Tensor,  # (num_subsets, num_keys)
        queries_single_cond: torch.Tensor,  # (num_subsets, num_queries, ch)
        keys_single_cond: torch.Tensor,  # (num_subsets, num_keys, ch)
        pair_cond: torch.Tensor,  # (num_subsets, num_queries, num_keys, ch)
        pair_mask: torch.Tensor | None = None,  # (num_subsets, num_queries, num_keys)
        rope_q: tuple[torch.Tensor, torch.Tensor] | None = None,
        rope_k: tuple[torch.Tensor, torch.Tensor] | None = None,
        engine_inputs: tuple[torch.Tensor | None, ...] | None = None,
    ) -> torch.Tensor:
        """Compute the module output.

        ``engine_inputs`` is the engine's ``(pair, key_mask[, uid])`` layout made once per fold by the
        caller (``uid``: the atoms' conformer ids, for a family whose pair mask is same-conformer): the engine hoists its pair bias per tensor object, and a view made here would be a
        new object at every call.
        """
        if self.per_block_pair:
            pair_logits = [
                projection(norm(pair_cond)).permute(0, 3, 1, 2)
                for norm, projection in zip(
                    self.pair_input_layer_norm, self.pair_logits_projection, strict=True
                )
            ]
        else:
            pair_logits = self.pair_logits_projection(self.pair_input_layer_norm(pair_cond))
            pair_logits = einops.rearrange(
                pair_logits, "n q k (b h) -> b n h q k", h=self.num_head
            )

        for block_idx in range(self.num_blocks):
            gather = functools.partial(
                atom_layout.convert, queries_to_keys, layout_axes=(-3, -2)
            )
            keys_act = gather(queries_act)
            chained = gather if self.chained_key_norm else None
            if self.parallel:
                queries_act = (
                    queries_act
                    + self.cross_attention[block_idx](
                        x_q=queries_act,
                        x_k=keys_act,
                        mask_q=queries_mask,
                        mask_k=keys_mask,
                        pair_logits=pair_logits[block_idx],
                        single_cond_q=queries_single_cond,
                        single_cond_k=keys_single_cond,
                        pair_mask=pair_mask,
                        keys_from_queries=chained,
                        rope_q=rope_q,
                        rope_k=rope_k,
                    )
                    + self.transition_block[block_idx](queries_act, queries_single_cond)
                )
                continue

            queries_act = conditioned_residual(
                queries_act,
                queries_single_cond,
                attention=lambda value, block_idx=block_idx, keys_act=keys_act, chained=chained: (
                    self.cross_attention[block_idx](
                        x_q=value,
                        x_k=keys_act,
                        mask_q=queries_mask,
                        mask_k=keys_mask,
                        pair_logits=pair_logits[block_idx],
                        single_cond_q=queries_single_cond,
                        single_cond_k=keys_single_cond,
                        pair_mask=pair_mask,
                        keys_from_queries=chained,
                        rope_q=rope_q,
                        rope_k=rope_k,
                    )
                ),
                transition=self.transition_block[block_idx],
            )

        return queries_act
