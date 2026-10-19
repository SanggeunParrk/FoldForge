"""Explicit projection precision for native low-precision model parameters."""

import torch
from torch import nn
from torch.nn import functional as F


class NativeLinear(nn.Linear):
    """Project in parameter dtype, with an explicit optional compute precision.

    Geometry projections can request FP32 arithmetic while retaining BF16
    parameters and BF16 outputs. No autocast or forward hook is involved.
    """

    compute_dtype: torch.dtype | None = None

    @classmethod
    def from_linear(cls, source: nn.Linear) -> "NativeLinear":
        """Reuse loaded parameters without initialization or rounding."""
        if source._forward_pre_hooks or source._forward_hooks:
            raise ValueError("Convert projections before installing forward hooks")
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.in_features = source.in_features
        result.out_features = source.out_features
        result.weight = source.weight
        result.bias = source.bias
        result.compute_dtype = getattr(
            source, "compute_dtype", getattr(source, "precision", None)
        )
        result.train(source.training)
        return result

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return the projection in parameter dtype under the declared policy."""
        dtype = self.weight.dtype
        # Preserve the native parameter-precision input boundary, including for
        # geometry projections which explicitly compute the subsequent GEMM in FP32.
        value = input.to(dtype)
        if self.compute_dtype is None:
            return F.linear(value, self.weight, self.bias)
        return F.linear(
            value.to(self.compute_dtype),
            self.weight.to(self.compute_dtype),
            None if self.bias is None else self.bias.to(self.compute_dtype),
        ).to(dtype)
