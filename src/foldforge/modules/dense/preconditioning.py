"""Shared EDM coordinate scaling and clean-coordinate reconstruction."""

from typing import Literal

import torch
from miniworld_engine.architecture.diffusion.edm.sampling import expand_to_coords


def coordinate_coefficient(value: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Keep scalar coefficients scalar to preserve Torch's mixed-dtype promotion."""
    return value if value.ndim == 0 else expand_to_coords(value, coords)


def scale_coordinates(
    coords: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply optional atom masking before the released division-based scaling."""
    value = coords if mask is None else coords * mask[..., None]
    return value / coordinate_coefficient(torch.sqrt(sigma_data**2 + sigma**2), coords)


def reconstruct_coordinates(
    coords: torch.Tensor,
    update: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float,
    *,
    formulation: Literal["standard", "ratio"] = "standard",
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Preserve the checkpoint's coefficient rounding and output-cast convention.

    Ratio-form coefficients are evaluated in the update dtype; replacing them
    by algebraically equivalent standard coefficients changes BF16 trajectories.
    """
    if formulation == "ratio":
        ratio = expand_to_coords(sigma / sigma_data, update).to(update.dtype)
        output = (
            1 / (1 + ratio**2) * coords
            + expand_to_coords(sigma, update) / torch.sqrt(1 + ratio**2) * update
        ).to(update.dtype)
    elif formulation == "standard":
        denominator = sigma_data**2 + sigma**2
        keep = coordinate_coefficient(sigma_data**2 / denominator, coords)
        emit = coordinate_coefficient(
            sigma * sigma_data / torch.sqrt(denominator), update
        )
        output = keep * coords + emit * update
    else:
        message = f"Unknown EDM coefficient formulation: {formulation}"
        raise ValueError(message)
    return output if mask is None else output * mask[..., None]
