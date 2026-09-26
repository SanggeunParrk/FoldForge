# Optional chemistry/GPU backends load at the selected execution boundary.
# ruff: noqa: PLC0415
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
from team_gm.modules.checkpoints.af_family import reference_transition

from foldforge.modules import ops as fastnn


class Transition(nn.Module):
    """Represent transition."""

    def __init__(self, c_x: int, num_intermediate_factor: int = 4) -> None:
        super().__init__()
        self.num_intermediate_factor = num_intermediate_factor
        self.c_in = c_x
        self.input_layer_norm = fastnn.LayerNorm(c_x)
        self.transition1 = nn.Linear(
            c_x, self.num_intermediate_factor * c_x * 2, bias=False
        )
        self.transition2 = nn.Linear(
            self.num_intermediate_factor * c_x, c_x, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the module output."""
        return reference_transition(self, x)


class OuterProductMean(nn.Module):
    """Represent outer product mean."""

    def __init__(
        self,
        c_msa: int = 64,
        num_output_channel: int = 128,
        num_outer_channel: int = 32,
        projection_bias: bool = False,
        groups: int = 1,
        bias_after_norm: bool = False,
        clamped_norm: bool = False,
    ) -> None:
        super().__init__()

        self.c_msa = c_msa
        self.bias_after_norm = bias_after_norm
        self.clamped_norm = clamped_norm
        self.num_outer_channel = num_outer_channel
        self.num_output_channel = num_output_channel
        self.groups = groups
        self.epsilon = 1e-3

        self.layer_norm_input = fastnn.LayerNorm(self.c_msa)
        width = self.groups * self.num_outer_channel
        self.left_projection = nn.Linear(self.c_msa, width, bias=projection_bias)
        self.right_projection = nn.Linear(self.c_msa, width, bias=projection_bias)

        if self.groups > 1:
            # Grouped: the outer product contracts only the MSA-depth axis and
            # BROADCASTS the group index, giving G*K*K products where one group
            # gives K*K. The product norm carries eps 0.1 because the depth axis
            # is summed and never divided; that norm is what absorbs the scale.
            self.product_norm = fastnn.LayerNorm(
                self.groups * self.num_outer_channel**2, eps=0.1
            )
        self.output_w = nn.Parameter(
            torch.randn(
                *((self.groups,) if self.groups > 1 else ()),
                self.num_outer_channel,
                self.num_outer_channel,
                self.num_output_channel,
            )
        )
        self.output_b = nn.Parameter(torch.randn(self.num_output_channel))

    def forward(self, msa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute the module output."""
        mask = mask.to(msa.dtype).unsqueeze(-1)
        msa = self.layer_norm_input(msa)
        left_act = mask * self.left_projection(msa)
        right_act = mask * self.right_projection(msa)

        if self.groups > 1:
            return self._grouped(left_act, right_act)

        from team_gm.modules.blocks.attention_math import af3_outer_product_mean

        if not self.bias_after_norm and not self.clamped_norm:
            # AF3's normalisation: mean over MSA rows, 1e-3 offset, bias before
            # the divide. The fast mode runs this for every family.
            return af3_outer_product_mean(
                left_act,
                right_act,
                mask,
                self.output_w,
                self.output_b,
                eps=self.epsilon,
            )
        # The exact mode's release variants, which fold alike (<= 0.1 A):
        outer = torch.einsum("acb,ade->dceb", left_act.permute(0, 2, 1), right_act)
        output = torch.einsum("dceb,cef->dbf", outer, self.output_w)
        norm = torch.einsum("abc,adc->bdc", mask, mask)
        # A clamp at one against AF3's 1e-3 offset: a scale, not an offset.
        divisor = norm.clamp_min(1.0) if self.clamped_norm else self.epsilon + norm
        if self.bias_after_norm:
            # Divide FIRST, so the output bias is not scaled by the pair count.
            return output.permute(1, 0, 2) / divisor + self.output_b
        return (output + self.output_b).permute(1, 0, 2) / divisor

    def _grouped(self, left_act: torch.Tensor, right_act: torch.Tensor) -> torch.Tensor:
        """Outer products taken WITHIN each group, summed over MSA depth only."""
        groups, channels = self.groups, self.num_outer_channel
        left = left_act.unflatten(-1, (groups, channels))
        right = right_act.unflatten(-1, (groups, channels))
        product = torch.einsum("sigk,sjgl->ijgkl", left, right)
        product = self.product_norm(product.flatten(-3))
        return (
            torch.einsum(
                "ijn,nf->ijf",
                product,
                self.output_w.reshape(groups * channels**2, self.num_output_channel),
            )
            + self.output_b
        )
