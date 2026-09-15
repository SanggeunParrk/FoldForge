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


class AdaptiveLayerNorm(nn.Module):
    """Represent adaptive layer norm."""

    def __init__(
        self, c_x: int, c_single_cond: int | None, use_single_cond: bool = False
    ) -> None:

        super().__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.use_single_cond = use_single_cond

        if self.use_single_cond is True:
            if self.c_single_cond is None:
                message = "Conditioned layers require a conditioning channel count"
                raise ValueError(message)
            self.layer_norm = fastnn.LayerNorm(
                self.c_x, elementwise_affine=False, bias=False
            )
            self.single_cond_layer_norm = fastnn.LayerNorm(
                self.c_single_cond, bias=False
            )
            self.single_cond_scale = nn.Linear(self.c_single_cond, self.c_x, bias=True)
            self.single_cond_bias = nn.Linear(self.c_single_cond, self.c_x, bias=False)
        else:
            self.layer_norm = fastnn.LayerNorm(self.c_x)

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
    ) -> None:
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.c_single_cond = c_single_cond
        self.use_single_cond = use_single_cond

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
    ) -> None:
        super().__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.num_intermediate_factor = num_intermediate_factor
        self.use_single_cond = use_single_cond

        self.adaptive_layernorm = AdaptiveLayerNorm(
            self.c_x, self.c_single_cond, self.use_single_cond
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


class SelfAttention(nn.Module):
    """Represent self attention."""

    def __init__(
        self,
        c_x: int = 768,
        c_single_cond: int = 384,
        num_head: int = 16,
        use_single_cond: bool = False,
    ) -> None:

        super().__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.num_head = num_head

        self.qkv_dim = self.c_x // self.num_head
        self.use_single_cond = use_single_cond

        self.adaptive_layernorm = AdaptiveLayerNorm(
            self.c_x, self.c_single_cond, self.use_single_cond
        )

        self.q_projection = nn.Linear(self.c_x, self.c_x, bias=True)
        self.k_projection = nn.Linear(self.c_x, self.c_x, bias=False)
        self.v_projection = nn.Linear(self.c_x, self.c_x, bias=False)

        self.gating_query = nn.Linear(self.c_x, self.c_x, bias=False)

        self.adaptive_zero_init = AdaLNZero(
            self.c_x, self.c_x, self.c_single_cond, self.use_single_cond
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

        q, k, v = (
            einops.rearrange(t, "n (h c) -> h n c", h=self.num_head).unsqueeze(0)
            for t in [q, k, v]
        )

        if engine_attention_supported(self, q, self.qkv_dim):
            weighted_avg = engine_dense_attention(q, k, v, pair_logits, mask)
        else:
            weighted_avg = fastnn.dot_product_attention(
                q, k, v, mask=mask, bias=pair_logits
            )

        weighted_avg = weighted_avg.squeeze(0)
        weighted_avg = einops.rearrange(weighted_avg, "h q c -> q (h c)")

        gate_logits = self.gating_query(x)
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
    ) -> None:

        super().__init__()

        self.c_act = c_act
        self.c_single_cond = c_single_cond
        self.c_pair_cond = c_pair_cond
        self.num_head = num_head
        self.num_blocks = num_blocks
        self.super_block_size = super_block_size

        self.num_super_blocks = self.num_blocks // self.super_block_size

        self.pair_input_layer_norm = fastnn.LayerNorm(self.c_pair_cond)
        self.pair_logits_projection = nn.ModuleList(
            [
                nn.Linear(
                    self.c_pair_cond, self.super_block_size * self.num_head, bias=False
                )
                for _ in range(self.num_super_blocks)
            ]
        )

        self.self_attention = nn.ModuleList(
            [
                SelfAttention(self.c_act, self.c_single_cond, use_single_cond=True)
                for _ in range(self.num_blocks)
            ]
        )
        self.transition_block = nn.ModuleList(
            [
                DiffusionTransition(
                    self.c_act, self.c_single_cond, use_single_cond=True
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
    ) -> None:
        super().__init__()

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.c_single_cond = c_single_cond
        self.num_head = num_head

        self.key_dim_per_head = self.key_dim // self.num_head
        self.value_dim_per_head = self.value_dim // self.num_head

        self.q_scale = self.key_dim_per_head ** (-0.5)

        self.q_adaptive_layernorm = AdaptiveLayerNorm(
            c_x=self.key_dim, c_single_cond=self.c_single_cond, use_single_cond=True
        )
        self.k_adaptive_layernorm = AdaptiveLayerNorm(
            c_x=self.key_dim, c_single_cond=self.c_single_cond, use_single_cond=True
        )

        self.q_projection = nn.Linear(self.key_dim, self.key_dim, bias=True)
        self.k_projection = nn.Linear(self.key_dim, self.key_dim, bias=False)
        self.v_projection = nn.Linear(self.value_dim, self.value_dim, bias=False)

        self.gating_query = nn.Linear(self.key_dim, self.value_dim, bias=False)
        self.adaptive_zero_init = AdaLNZero(
            self.value_dim, self.value_dim, self.key_dim, use_single_cond=True
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
    ) -> torch.Tensor:
        """Compute the module output."""
        if len(mask_q.shape) != len(x_q.shape) - 1:
            message = f"{mask_q.shape}, {x_q.shape}"
            raise ValueError(message)
        if len(mask_k.shape) != len(x_k.shape) - 1:
            message = f"{mask_k.shape}, {x_k.shape}"
            raise ValueError(message)

        bias = (
            1e9
            * mask_q.logical_not()[..., None, :, None]
            * mask_k.logical_not()[..., None, None, :]
        )

        x_q = self.q_adaptive_layernorm(x_q, single_cond_q)
        x_k = self.k_adaptive_layernorm(x_k, single_cond_k)

        q = self.q_projection(x_q)
        k = self.k_projection(x_k)
        q = torch.reshape(q, (*q.shape[:-1], self.num_head, self.key_dim_per_head))
        k = torch.reshape(k, (*k.shape[:-1], self.num_head, self.key_dim_per_head))

        logits = (
            torch.einsum("...qhc,...khc->...hqk", q.float() * self.q_scale, k.float())
            + bias
        )
        if pair_logits is not None:
            logits += pair_logits
        weights = torch.softmax(logits, dim=-1)

        v = self.v_projection(x_k)
        v = torch.reshape(v, (*v.shape[:-1], self.num_head, self.value_dim_per_head))
        weighted_avg = torch.einsum("...hqk,...khc->...qhc", weights.to(v.dtype), v)
        weighted_avg = torch.reshape(weighted_avg, (*weighted_avg.shape[:-2], -1))

        gate_logits = self.gating_query(x_q)
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
    ) -> None:
        super().__init__()

        self.c_query = c_query
        self.c_single_cond = c_single_cond
        self.c_pair_cond = c_pair_cond

        self.num_blocks = num_blocks
        self.num_head = num_head

        self.pair_input_layer_norm = fastnn.LayerNorm(self.c_pair_cond, bias=False)
        self.pair_logits_projection = nn.Linear(
            self.c_pair_cond, self.num_blocks * self.num_head, bias=False
        )

        self.cross_attention = nn.ModuleList(
            [CrossAttention(num_head=self.num_head) for _ in range(self.num_blocks)]
        )

        self.transition_block = nn.ModuleList(
            [
                DiffusionTransition(
                    c_x=self.c_query,
                    c_single_cond=self.c_single_cond,
                    use_single_cond=True,
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
    ) -> torch.Tensor:
        """Compute the module output."""
        pair_logits = pair_bias_projection(
            self.pair_input_layer_norm, self.pair_logits_projection, pair_cond
        )

        pair_logits = einops.rearrange(
            pair_logits, "n q k (b h) -> b n h q k", h=self.num_head
        )

        for block_idx in range(self.num_blocks):
            keys_act = atom_layout.convert(
                queries_to_keys, queries_act, layout_axes=(-3, -2)
            )

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
                    )
                ),
                transition=self.transition_block[block_idx],
            )

        return queries_act
