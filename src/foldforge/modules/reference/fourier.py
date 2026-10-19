"""One Fourier feature equation for fixed checkpoint projections."""

import torch


def fourier_features(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    cast_input: bool = False,
) -> torch.Tensor:
    """Embed scalar noise with the checkpoint's explicit input-cast policy."""
    if cast_input:
        value = value.to(weight.dtype)
    return torch.cos(2 * torch.pi * (value[..., None] * weight + bias))
