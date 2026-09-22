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
from team_gm.modules.checkpoints.af_family import (
    reference_triangle,
)

from foldforge.modules import ops as fastnn


class TriangleMultiplication(nn.Module):
    """Represent triangle multiplication."""

    def __init__(
        self,
        c_pair: int = 128,
        _outgoing: bool = True,
        divide_by_length: bool = False,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.c_pair = c_pair
        #: AF3 ties the projection width to the channel count; one family's
        #: template stack widens it without widening the pair.
        hidden = hidden_dim or c_pair
        #: Read by the shared triangle update: the contraction is divided by the
        #: sequence length before the centre norm.
        self.divide_by_length = divide_by_length
        self.left_norm_input = fastnn.LayerNorm(self.c_pair)
        self.projection = nn.Linear(self.c_pair, 2 * hidden, bias=False)
        self.gate = nn.Linear(self.c_pair, 2 * hidden, bias=False)
        self.center_norm = fastnn.LayerNorm(hidden)
        self.output_projection = nn.Linear(hidden, self.c_pair, bias=False)
        self.gating_linear = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.equation = "ckj,cki->cij"
        if _outgoing is True:
            self.equation = "cik,cjk->cij"

    def forward(self, pair: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute the module output."""
        return reference_triangle(self, pair, mask)
