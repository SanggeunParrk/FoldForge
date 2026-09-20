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
        bias_after_norm: bool = False,
        projection_bias: bool = False,
    ) -> None:
        super().__init__()

        self.bias_after_norm = bias_after_norm

        self.c_msa = c_msa
        self.num_outer_channel = num_outer_channel
        self.num_output_channel = num_output_channel
        self.epsilon = 1e-3

        self.layer_norm_input = fastnn.LayerNorm(self.c_msa)
        self.left_projection = nn.Linear(
            self.c_msa, self.num_outer_channel, bias=projection_bias
        )
        self.right_projection = nn.Linear(
            self.c_msa, self.num_outer_channel, bias=projection_bias
        )

        self.output_w = nn.Parameter(
            torch.randn(
                self.num_outer_channel, self.num_outer_channel, self.num_output_channel
            )
        )
        self.output_b = nn.Parameter(torch.randn(self.num_output_channel))

    def forward(self, msa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute the module output."""
        mask = mask.to(msa.dtype).unsqueeze(-1)
        msa = self.layer_norm_input(msa)
        left_act = mask * self.left_projection(msa)
        right_act = mask * self.right_projection(msa)

        from team_gm.modules.blocks.attention_math import af3_outer_product_mean

        if self.bias_after_norm:
            # Divide by the pair count clamped at one, THEN add the bias. The two
            # orders differ by bias * (1 - 1/n): a per-channel constant.
            outer = torch.einsum("acb,ade->dceb", left_act.permute(0, 2, 1), right_act)
            output = torch.einsum("dceb,cef->dbf", outer, self.output_w)
            norm = torch.einsum("abc,adc->bdc", mask, mask)
            return output.permute(1, 0, 2) / norm.clamp_min(1.0) + self.output_b
        return af3_outer_product_mean(
            left_act, right_act, mask, self.output_w, self.output_b, eps=self.epsilon
        )
