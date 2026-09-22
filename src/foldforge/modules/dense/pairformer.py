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


import torch
import torch.nn as nn
from team_gm.modules.blocks.composition import (
    MSAUpdateConfig,
    msa_pair_update,
    msa_row_update,
)
from team_gm.modules.checkpoints.af_family import transition_residual, triangle_residual

from foldforge.modules import ops as fastnn
from foldforge.modules.dense.attention import GridSelfAttention, MSAAttention
from foldforge.modules.dense.diffusion_transformer import SelfAttention
from foldforge.modules.dense.primitives import OuterProductMean, Transition
from foldforge.modules.dense.spec import ALPHAFOLD3, DenseSpec
from foldforge.modules.dense.triangle_multiplication import TriangleMultiplication


class PairformerBlock(nn.Module):
    """Implement Algorithm 17 [Line2-Line8] in AF3.

    Ref to: openfold/model/evoformer.py and protenix/model/modules/pairformer.py
    """

    def __init__(
        self,
        n_heads: int = 16,
        c_pair: int = 128,
        c_single: int = 384,
        c_hidden_mul: int = 128,  # noqa: ARG002 - shared callback or fixture signature
        n_heads_pair: int = 4,
        num_intermediate_factor: int = 4,
        single_intermediate_factor: int | None = None,
        with_single: bool = True,
        spec: DenseSpec = ALPHAFOLD3,
        pair_qkv_dim: int | None = None,
        dual_output: bool = False,
        tri_hidden_dim: int | None = None,
    ) -> None:
        """Args:

        n_heads (int, optional): number of head [for SelfAttention]. Defaults to
            16.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_hidden_mul (int, optional): hidden dim [for
            TriangleMultiplicationOutgoing].
            Defaults to 128.
        n_heads_pair (int, optional): number of head [for TriangleAttention].
            Defaults to 4.
        """
        super().__init__()
        self.n_heads = n_heads
        self.with_single = with_single
        self.num_intermediate_factor = num_intermediate_factor

        self.triangle_multiplication_outgoing = TriangleMultiplication(
            c_pair=c_pair,
            _outgoing=True,
            divide_by_length=spec.triangle_mul_divide_by_length,
            hidden_dim=tri_hidden_dim,
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            c_pair=c_pair,
            _outgoing=False,
            divide_by_length=spec.triangle_mul_divide_by_length,
            hidden_dim=tri_hidden_dim,
        )
        self.pair_attention1 = GridSelfAttention(
            c_pair=c_pair,
            num_head=n_heads_pair,
            transpose=False,
            spec=spec,
            qkv_dim=pair_qkv_dim,
            dual_output=dual_output,
        )
        self.pair_attention2 = GridSelfAttention(
            c_pair=c_pair,
            num_head=n_heads_pair,
            transpose=True,
            spec=spec,
            qkv_dim=pair_qkv_dim,
            dual_output=dual_output,
        )
        self.pair_transition = Transition(
            c_x=c_pair, num_intermediate_factor=self.num_intermediate_factor
        )
        self.c_single = c_single
        #: Every update reads the representation ENTERING the block and their
        #: results are summed into it, where AF3 threads each update through the
        #: running activation.
        self.parallel = spec.parallel_pairformer_block
        if self.with_single is True:
            self.single_pair_logits_norm = fastnn.LayerNorm(c_pair)
            self.single_pair_logits_projection = nn.Linear(c_pair, n_heads, bias=False)
            self.single_attention_ = SelfAttention(
                c_x=c_single, num_head=n_heads, use_single_cond=False
            )
            # The single transition normally widens by the same factor as the
            # pair one. A stack that narrows only its pair transition says so.
            self.single_transition = Transition(
                c_x=self.c_single,
                num_intermediate_factor=(
                    self.num_intermediate_factor
                    if single_intermediate_factor is None
                    else single_intermediate_factor
                ),
            )

    def forward(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        single: torch.Tensor | None = None,
        seq_mask: torch.Tensor | None = None,
        extra_pair_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the PairformerBlock.

        Args:
            pair (torch.Tensor): [..., N_token, N_token, c_pair]
            pair_mask (torch.Tensor): [..., N_token, N_token]
            single (torch.Tensor, optional): [..., N_token, c_single]
            seq_mask (torch.Tensor, optional): [..., N_token]
            extra_pair_bias (torch.Tensor, optional): [..., N_token, N_token],
                added to the single attention's pair logits on every head. Only
                the structural-token refiner supplies one.

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: pair, single
        """
        if self.parallel:
            # The fused residual helpers below add into the running activation,
            # which is the schedule this family does not use; call the modules
            # for their deltas instead.
            pair_in = pair
            pair = pair_in + (
                self.triangle_multiplication_outgoing(pair_in, pair_mask)
                + self.triangle_multiplication_incoming(pair_in, pair_mask)
                + self.pair_attention1(pair_in, mask=pair_mask)
                + self.pair_attention2(pair_in, mask=pair_mask)
                + self.pair_transition(pair_in)
            )
        else:
            pair = triangle_residual(
                self.triangle_multiplication_outgoing, pair, pair_mask
            )
            pair = triangle_residual(
                self.triangle_multiplication_incoming, pair, pair_mask
            )
            pair += self.pair_attention1(pair, mask=pair_mask)
            pair += self.pair_attention2(pair, mask=pair_mask)
            pair = transition_residual(self.pair_transition, pair)

        if self.with_single is True:
            if single is None or seq_mask is None:
                message = "Single-track pairformer requires single features and a mask"
                raise ValueError(message)
            pair_logits = self.single_pair_logits_projection(
                self.single_pair_logits_norm(pair)
            )

            pair_logits = pair_logits.permute(2, 0, 1)
            if extra_pair_bias is not None:
                pair_logits = pair_logits + extra_pair_bias[None].to(pair_logits.dtype)

            attention_update: torch.Tensor = self.single_attention_(
                single, seq_mask, pair_logits=pair_logits
            )
            if self.parallel:
                # Both single updates read the single entering the block.
                return pair, single + attention_update + self.single_transition(single)
            single += attention_update

            single = transition_residual(self.single_transition, single)
            return pair, single
        return pair


class EvoformerBlock(nn.Module):
    """Represent evoformer block."""

    def __init__(
        self,
        c_msa: int = 64,
        c_pair: int = 128,
        n_heads_pair: int = 4,
        spec: DenseSpec = ALPHAFOLD3,
    ) -> None:
        super().__init__()
        self.msa_update_config = MSAUpdateConfig(
            order="msa_first" if spec.msa_update_before_opm else "opm_first"
        )

        self.outer_product_mean = OuterProductMean(
            c_msa=c_msa,
            num_output_channel=c_pair,
            num_outer_channel=spec.opm_channel,
            bias_after_norm=spec.opm_bias_after_norm,
            projection_bias=spec.opm_projection_bias,
            groups=spec.opm_groups,
            sum_without_norm=spec.opm_sum_without_norm,
        )
        self.msa_attention1 = MSAAttention(
            c_msa=c_msa,
            c_pair=c_pair,
            value_dim=spec.msa_value_dim,
            pair_mask_logits=spec.msa_pair_mask_logits,
        )
        self.msa_transition = Transition(c_x=c_msa)

        self.triangle_multiplication_outgoing = TriangleMultiplication(
            c_pair=c_pair,
            _outgoing=True,
            divide_by_length=spec.triangle_mul_divide_by_length,
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            c_pair=c_pair,
            _outgoing=False,
            divide_by_length=spec.triangle_mul_divide_by_length,
        )
        self.pair_attention1 = GridSelfAttention(
            c_pair=c_pair, num_head=n_heads_pair, transpose=False, spec=spec
        )
        self.pair_attention2 = GridSelfAttention(
            c_pair=c_pair, num_head=n_heads_pair, transpose=True, spec=spec
        )
        self.pair_transition = Transition(c_x=c_pair)
        #: Two parallel stages: the two triangle multiplications and the
        #: transition all read the post-outer-product pair and are summed into
        #: it, then both attention directions read that result and are summed in
        #: turn. AF3 threads all five sequentially.
        self.parallel = spec.parallel_msa_block

    def forward(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the module output."""

        def update_msa(value, updated_pair):
            """Update msa."""
            return msa_row_update(
                value,
                updated_pair,
                attention_delta=lambda m, z: self.msa_attention1(
                    m, msa_mask, z, pair_mask
                ),
                transition_residual=lambda m: transition_residual(
                    self.msa_transition, m
                ),
            )

        def update_pair(value):
            """Update pair."""
            if self.parallel:
                value = value + (
                    self.triangle_multiplication_outgoing(value, pair_mask)
                    + self.triangle_multiplication_incoming(value, pair_mask)
                    + self.pair_transition(value)
                )
                return value + (
                    self.pair_attention1(value, mask=pair_mask)
                    + self.pair_attention2(value, mask=pair_mask)
                )
            value = triangle_residual(
                self.triangle_multiplication_outgoing, value, pair_mask
            )
            value = triangle_residual(
                self.triangle_multiplication_incoming, value, pair_mask
            )
            value = value + self.pair_attention1(value, mask=pair_mask)
            value = value + self.pair_attention2(value, mask=pair_mask)
            return transition_residual(self.pair_transition, value)

        updated_msa, updated_pair = msa_pair_update(
            msa,
            pair,
            config=self.msa_update_config,
            outer_product_delta=lambda m: self.outer_product_mean(m, msa_mask),
            msa_residual=update_msa,
            pair_residual=update_pair,
        )
        if updated_msa is None:
            message = "MSA update unexpectedly removed the MSA track"
            raise RuntimeError(message)
        return updated_msa, updated_pair
