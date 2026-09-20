# Optional chemistry/GPU backends load at the selected execution boundary.
# ruff: noqa: PLC0415
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
from team_gm.modules.checkpoints.backend_attention import (
    engine_attention_supported,
    engine_triangle_attention,
    gated_projection,
)

from foldforge.modules import ops as fastnn
from foldforge.modules.dense.spec import ALPHAFOLD3, DenseSpec


class GridSelfAttention(nn.Module):
    """Represent grid self attention."""

    def __init__(
        self,
        c_pair: int = 128,
        num_head: int = 4,
        transpose: bool = False,
        spec: DenseSpec = ALPHAFOLD3,
    ) -> None:
        super().__init__()
        self.c_pair = c_pair
        self.num_head = num_head
        self.qkv_dim = self.c_pair // self.num_head
        self.transpose = transpose
        self.transposed_bias = transpose and spec.transposed_column_pair_bias

        self.act_norm = fastnn.LayerNorm(self.c_pair)
        self.pair_bias_projection = nn.Linear(self.c_pair, self.num_head, bias=False)

        self.q_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.k_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.v_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.gating_query = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.output_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)

    def _attention(self, pair: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor):
        q = self.q_projection(pair)
        k = self.k_projection(pair)
        v = self.v_projection(pair)

        q, k, v = (
            einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_head)
            for t in [q, k, v]
        )

        if engine_attention_supported(self, q, self.qkv_dim):
            weighted_avg = engine_triangle_attention(q, k, v, bias, mask)
        elif (
            getattr(self, "foldforge_implementation", None) == "cuequivariance"
            and q.is_cuda
        ):
            from cuequivariance_torch import triangle_attention

            vendor_mask = None if mask is None else mask[None, :, None, None, :]
            vendor_output = triangle_attention(
                q[None].contiguous(),
                k[None].contiguous(),
                v[None].contiguous(),
                bias[None, None].float().contiguous(),
                mask=vendor_mask,
                scale=self.qkv_dim**-0.5,
            )
            if not isinstance(vendor_output, torch.Tensor):
                message = "Triangle attention returned auxiliary values unexpectedly"
                raise TypeError(message)
            weighted_avg = vendor_output.squeeze(0)
        else:
            weighted_avg = fastnn.dot_product_attention(q, k, v, mask=mask, bias=bias)

        weighted_avg = einops.rearrange(weighted_avg, "b h n d -> b n (h d)")

        gate_values = self.gating_query(pair)

        return gated_projection(self, self.output_projection, gate_values, weighted_avg)

    def forward(self, pair: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Args:

            pair (torch.Tensor): [N_token, N_token, c_pair]
            mask (torch.Tensor): [N_token, N_token]
        Returns:
            torch.Tensor: [N_token, N_token, c_pair]
        """
        pair = self.act_norm(pair)
        nonbatched_bias = self.pair_bias_projection(pair).permute(2, 0, 1)
        if self.transposed_bias:
            nonbatched_bias = nonbatched_bias.transpose(-1, -2)

        if self.transpose:
            pair = pair.permute(1, 0, 2)

        pair = self._attention(pair, mask, nonbatched_bias)

        if self.transpose:
            pair = pair.permute(1, 0, 2)

        return pair


class MSAAttention(nn.Module):
    """Represent m s a attention."""

    def __init__(self, c_msa: int = 64, c_pair: int = 128, num_head: int = 8) -> None:
        super().__init__()

        self.c_msa = c_msa
        self.c_pair = c_pair
        self.num_head = num_head

        self.value_dim = self.c_msa // self.num_head

        self.act_norm = fastnn.LayerNorm(self.c_msa)
        self.pair_norm = fastnn.LayerNorm(self.c_pair)
        self.pair_logits = nn.Linear(self.c_pair, self.num_head, bias=False)
        self.v_projection = nn.Linear(
            self.c_msa, self.num_head * self.value_dim, bias=False
        )
        self.gating_query = nn.Linear(self.c_msa, self.c_msa, bias=False)
        self.output_projection = nn.Linear(self.c_msa, self.c_msa, bias=False)

    def forward(self, msa, msa_mask, pair):
        """Compute the module output."""
        msa = self.act_norm(msa)
        pair = self.pair_norm(pair)
        logits = self.pair_logits(pair)
        logits = logits.permute(2, 0, 1)

        logits += 1e9 * (torch.max(msa_mask, dim=0).values - 1.0)
        weights = torch.softmax(logits, dim=-1)

        v = self.v_projection(msa)
        v = einops.rearrange(v, "b k (h c) -> b k h c", h=self.num_head)

        v_avg = torch.einsum("hqk, bkhc -> bqhc", weights.to(v.dtype), v)
        v_avg = torch.reshape(v_avg, (*v_avg.shape[:-2], -1))

        gate_values = self.gating_query(msa)
        return gated_projection(self, self.output_projection, gate_values, v_avg)
