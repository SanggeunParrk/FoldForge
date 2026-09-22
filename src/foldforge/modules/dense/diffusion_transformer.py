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


from typing import Any

import einops
import torch
import torch.nn as nn
from team_gm.modules.blocks.composition import conditioned_residual
from team_gm.modules.checkpoints.af_family import (
    reference_adaln,
    reference_conditioned_transition,
)
from team_gm.modules.checkpoints.backend_attention import (
    engine_attention_supported,
    engine_dense_attention,
    gated_projection,
    pair_bias_projection,
)

from foldforge.modules import ops as fastnn
from foldforge.modules.dense import atom_layout
from foldforge.modules.dense.spec import ALPHAFOLD3, DenseSpec


class AdaptiveLayerNorm(nn.Module):
    """Represent adaptive layer norm."""

    def __init__(
        self,
        c_x: int,
        c_single_cond: int | None,
        use_single_cond: bool = False,
        identity_scale: bool = False,
        eps: float = 1e-5,
    ) -> None:

        super().__init__()

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
            self.layer_norm = fastnn.LayerNorm(
                self.c_x, eps=eps, elementwise_affine=False, bias=False
            )
            if not identity_scale:
                self.single_cond_layer_norm = fastnn.LayerNorm(
                    self.c_single_cond, bias=False
                )
            self.single_cond_scale = nn.Linear(
                self.c_single_cond, self.c_x, bias=not identity_scale
            )
            self.single_cond_bias = nn.Linear(self.c_single_cond, self.c_x, bias=False)
        else:
            self.layer_norm = fastnn.LayerNorm(self.c_x, eps=eps)

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
    ) -> None:
        super().__init__()

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
                self.c_single_cond, self.c_out, bias=True
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
            else gated_projection(self, self.transition2, gate_logits, x)
        )
        if self.use_single_cond is True:
            if self.c_single_cond is None:
                message = "Conditioned layers require a conditioning channel count"
                raise ValueError(message)
            cond = self.adaptive_zero_cond(single_cond)
            output = torch.sigmoid(cond) * output
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
        )
        self.transition1 = nn.Linear(
            self.c_x, 2 * self.c_x * self.num_intermediate_factor, bias=False
        )

        self.adaptive_zero_init = AdaLNZero(
            self.num_intermediate_factor * self.c_x,
            self.c_x,
            self.c_single_cond,
            self.use_single_cond,
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


class SelfAttention(nn.Module):
    """Represent self attention."""

    def __init__(
        self,
        c_x: int = 768,
        c_single_cond: int = 384,
        num_head: int = 16,
        use_single_cond: bool = False,
        kq_norm: bool = False,
        identity_scale: bool = False,
        norm_eps: float = 1e-5,
        gating_query: bool = True,
        project_output: bool = True,
    ) -> None:

        super().__init__()
        self.kq_norm = kq_norm
        self.use_gating_query = gating_query
        if kq_norm:
            self.query_layer_norm = fastnn.LayerNorm(c_x)
            self.key_layer_norm = fastnn.LayerNorm(c_x)

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.num_head = num_head

        self.qkv_dim = self.c_x // self.num_head
        self.use_single_cond = use_single_cond

        self.adaptive_layernorm = AdaptiveLayerNorm(
            self.c_x,
            self.c_single_cond,
            self.use_single_cond,
            identity_scale=identity_scale,
            eps=norm_eps,
        )

        self.q_projection = nn.Linear(self.c_x, self.c_x, bias=True)
        self.k_projection = nn.Linear(self.c_x, self.c_x, bias=False)
        self.v_projection = nn.Linear(self.c_x, self.c_x, bias=False)

        if self.use_gating_query:
            self.gating_query = nn.Linear(self.c_x, self.c_x, bias=False)

        self.adaptive_zero_init = AdaLNZero(
            self.c_x,
            self.c_x,
            self.c_single_cond,
            self.use_single_cond,
            project=project_output,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pair_logits: torch.Tensor | None = None,
        single_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:

        x (torch.Tensor): (num_tokens, ch)
        mask (torch.Tensor): (num_tokens,)
        pair_logits (torch.Tensor, optional): (num_heads, num_tokens, num_tokens)
        """
        if (single_cond is None) != (self.use_single_cond is False):
            message = (
                "Invalid state: (single_cond is None) == (self.use_single_cond is"
                " False)"
            )
            raise ValueError(message)

        x = self.adaptive_layernorm(x, single_cond)

        q = self.q_projection(x)
        k = self.k_projection(x)
        v = self.v_projection(x)
        if self.kq_norm:
            # Over the flattened head axis, before the per-head scaling.
            q, k = self.query_layer_norm(q), self.key_layer_norm(k)

        q, k, v = (
            einops.rearrange(t, "... n (h c) -> ... h n c", h=self.num_head)
            if x.ndim > 2
            else einops.rearrange(t, "n (h c) -> h n c", h=self.num_head).unsqueeze(0)
            for t in [q, k, v]
        )

        if engine_attention_supported(self, q, self.qkv_dim):
            weighted_avg = engine_dense_attention(
                q, k, v, pair_logits, mask, num_aug=x.shape[0] if x.ndim == 3 else 1
            )
        else:
            weighted_avg = fastnn.dot_product_attention(
                q, k, v, mask=mask, bias=pair_logits
            )

        if x.ndim == 2:
            weighted_avg = weighted_avg.squeeze(0)
        weighted_avg = einops.rearrange(weighted_avg, "... h q c -> ... q (h c)")

        gate_logits = self.gating_query(x) if self.use_gating_query else None
        return self.adaptive_zero_init(
            weighted_avg, single_cond, gate_logits=gate_logits
        )


class DiffusionTransformer(nn.Module):
    """Represent diffusion transformer."""

    def __init__(
        self,
        c_act: int = 768,
        c_single_cond: int = 384,
        c_pair_cond: int = 128,
        num_head: int = 16,
        num_blocks: int = 24,
        super_block_size: int = 4,
        spec: DenseSpec = ALPHAFOLD3,
    ) -> None:

        super().__init__()
        self.per_block_pair = spec.per_block_pair_layer_norm
        self.parallel = spec.parallel_attention_transition

        self.c_act = c_act
        self.c_single_cond = c_single_cond
        self.c_pair_cond = c_pair_cond
        self.num_head = num_head
        self.num_blocks = num_blocks
        self.super_block_size = super_block_size

        self.num_super_blocks = self.num_blocks // self.super_block_size

        if self.per_block_pair:
            # One pair norm and one projection per block, as the vendor trained it.
            self.pair_input_layer_norm = nn.ModuleList(
                [
                    fastnn.LayerNorm(
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
            self.pair_input_layer_norm = fastnn.LayerNorm(
                self.c_pair_cond,
                bias="pair_input_layer_norm" in spec.affine_norms,
            )
            self.pair_logits_projection = nn.ModuleList(
                [
                    nn.Linear(
                        self.c_pair_cond,
                        self.super_block_size * self.num_head,
                        bias=False,
                    )
                    for _ in range(self.num_super_blocks)
                ]
            )

        self.self_attention = nn.ModuleList(
            [
                SelfAttention(
                    self.c_act,
                    self.c_single_cond,
                    use_single_cond=True,
                    kq_norm=spec.attention_kq_norm,
                    identity_scale=spec.adaptive_identity_scale,
                    norm_eps=spec.adaptive_norm_eps,
                    gating_query=spec.token_attention_gating_query,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.transition_block = nn.ModuleList(
            [
                conditioned_transition(spec)(
                    self.c_act,
                    self.c_single_cond,
                    use_single_cond=True,
                    identity_scale=spec.adaptive_identity_scale,
                    norm_eps=spec.adaptive_norm_eps,
                )
                for _ in range(self.num_blocks)
            ]
        )

    def forward(
        self,
        act: torch.Tensor,
        mask: torch.Tensor,
        single_cond: torch.Tensor,
        pair_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the module output."""
        if self.per_block_pair:
            for idx in range(self.num_blocks):
                pair_logits = self.pair_logits_projection[idx](
                    self.pair_input_layer_norm[idx](pair_cond)
                ).permute(2, 0, 1)
                if self.parallel:
                    act = (
                        act
                        + self.self_attention[idx](act, mask, pair_logits, single_cond)
                        + self.transition_block[idx](act, single_cond)
                    )
                    continue
                act = conditioned_residual(
                    act,
                    single_cond,
                    attention=lambda value, idx=idx, pair_logits=pair_logits: (
                        self.self_attention[idx](value, mask, pair_logits, single_cond)
                    ),
                    transition=self.transition_block[idx],
                )
            return act

        pair_act = self.pair_input_layer_norm(pair_cond)

        for super_block_i in range(self.num_super_blocks):
            pair_logits = self.pair_logits_projection[super_block_i](pair_act)
            pair_logits = einops.rearrange(
                pair_logits, "n s (b h) -> b h n s", h=self.num_head
            )
            for j in range(self.super_block_size):
                idx = super_block_i * self.super_block_size + j
                act = conditioned_residual(
                    act,
                    single_cond,
                    attention=lambda value, idx=idx, j=j, pair_logits=pair_logits: (
                        self.self_attention[idx](
                            value, mask, pair_logits[j], single_cond
                        )
                    ),
                    transition=self.transition_block[idx],
                )

        return act


class CrossAttention(nn.Module):
    """Represent cross attention."""

    def __init__(
        self,
        key_dim: int = 128,
        value_dim: int = 128,
        c_single_cond: int = 128,
        num_head: int = 4,
        key_masked: bool = False,
        kq_norm: bool = False,
        identity_scale: bool = False,
        norm_eps: float = 1e-5,
        gating_query: bool = True,
        project_output: bool = True,
    ) -> None:
        super().__init__()

        self.key_masked = key_masked
        self.kq_norm = kq_norm
        self.use_gating_query = gating_query
        if kq_norm:
            self.query_layer_norm = fastnn.LayerNorm(key_dim)
            self.key_layer_norm = fastnn.LayerNorm(key_dim)
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.c_single_cond = c_single_cond
        self.num_head = num_head

        self.key_dim_per_head = self.key_dim // self.num_head
        self.value_dim_per_head = self.value_dim // self.num_head

        self.q_scale = self.key_dim_per_head ** (-0.5)

        adaln = {
            "c_x": self.key_dim,
            "c_single_cond": self.c_single_cond,
            "use_single_cond": True,
            "identity_scale": identity_scale,
            "eps": norm_eps,
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
    ) -> torch.Tensor:
        """Compute the module output."""
        if tuple(mask_q.shape) != tuple(x_q.shape[-mask_q.ndim - 1 : -1]):
            message = f"{mask_q.shape}, {x_q.shape}"
            raise ValueError(message)
        if tuple(mask_k.shape) != tuple(x_k.shape[-mask_k.ndim - 1 : -1]):
            message = f"{mask_k.shape}, {x_k.shape}"
            raise ValueError(message)

        if self.key_masked:
            # OR form: a padded key is masked from every query. AF3's AND form is
            # only safe because it slides its key window inside the real atoms.
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
        x_k = self.k_adaptive_layernorm(x_k, single_cond_k)

        q = self.q_projection(x_q)
        k = self.k_projection(x_k)
        if self.kq_norm:
            q, k = self.query_layer_norm(q), self.key_layer_norm(k)
        q = torch.reshape(q, (*q.shape[:-1], self.num_head, self.key_dim_per_head))
        k = torch.reshape(k, (*k.shape[:-1], self.num_head, self.key_dim_per_head))

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

        self.per_block_pair = spec.per_block_atom_pair_layer_norm
        self.parallel = spec.parallel_attention_transition
        self.mask_act_per_block = spec.mask_atom_act_per_block
        if self.per_block_pair:
            self.pair_input_layer_norm = nn.ModuleList(
                [
                    fastnn.LayerNorm(
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
            self.pair_input_layer_norm = fastnn.LayerNorm(
                self.c_pair_cond,
                bias="pair_input_layer_norm" in spec.affine_norms,
            )
            self.pair_logits_projection = nn.Linear(
                self.c_pair_cond, self.num_blocks * self.num_head, bias=False
            )

        self.cross_attention = nn.ModuleList(
            [
                CrossAttention(
                    num_head=self.num_head,
                    key_masked=spec.key_masked_atom_attention,
                    kq_norm=spec.attention_kq_norm,
                    identity_scale=spec.adaptive_identity_scale,
                    norm_eps=spec.adaptive_norm_eps,
                    gating_query=spec.atom_attention_gating_query,
                    project_output=spec.atom_attention_project_output,
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
                    identity_scale=spec.adaptive_identity_scale,
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
    ) -> torch.Tensor:
        """Compute the module output."""
        if self.per_block_pair:
            pair_logits = [
                projection(norm(pair_cond)).permute(0, 3, 1, 2)
                for norm, projection in zip(
                    self.pair_input_layer_norm, self.pair_logits_projection, strict=True
                )
            ]
        else:
            pair_logits = pair_bias_projection(
                self.pair_input_layer_norm, self.pair_logits_projection, pair_cond
            )
            pair_logits = einops.rearrange(
                pair_logits, "n q k (b h) -> b n h q k", h=self.num_head
            )

        for block_idx in range(self.num_blocks):
            if self.mask_act_per_block:
                # Re-pad the atom axis with zeros before the key gather, so a
                # padded slot cannot carry one block's output into the next
                # block's keys.
                queries_act = queries_act * queries_mask[..., None].to(
                    queries_act.dtype
                )
            keys_act = atom_layout.convert(
                queries_to_keys, queries_act, layout_axes=(-3, -2)
            )
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
                    )
                    + self.transition_block[block_idx](queries_act, queries_single_cond)
                )
                continue

            queries_act = conditioned_residual(
                queries_act,
                queries_single_cond,
                attention=lambda value, block_idx=block_idx, keys_act=keys_act: (
                    self.cross_attention[block_idx](
                        x_q=value,
                        x_k=keys_act,
                        mask_q=queries_mask,
                        mask_k=keys_mask,
                        pair_logits=pair_logits[block_idx],
                        single_cond_q=queries_single_cond,
                        single_cond_k=keys_single_cond,
                        pair_mask=pair_mask,
                    )
                ),
                transition=self.transition_block[block_idx],
            )

        return queries_act
