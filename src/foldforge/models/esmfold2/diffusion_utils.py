"""Small diffusion helpers ESMFold2 needs that the shared package does not have."""

from collections.abc import Callable

import torch
from jaxtyping import Bool, Float

#: A denoiser: ``(scaled_coords, sigma) -> coordinate_update``.
#:
#: The solver hands over coordinates already divided by
#: ``sqrt(sigma^2 + sigma_data^2)`` and the raw noise level. Mapping sigma to a
#: conditioning value is the network's business — ESMFold2 bakes ``sigma_data``
#: into its own config — so the solver does not do it on the network's behalf.
DenoiseFn = Callable[
    [Float[torch.Tensor, "*batch N 3"], torch.Tensor],
    Float[torch.Tensor, "*batch N 3"],
]


def expand_to_coords(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Broadcast per-sample scalars across a coordinate tensor's trailing axes.

    Parameters
    ----------
    value : Tensor
        Per-sample values, one leading axis.
    target : Tensor
        Tensor whose rank the result should match.

    Returns
    -------
    Tensor
        ``value`` reshaped with trailing singleton axes.

    Raises
    ------
    ValueError
        If ``value`` has more dimensions than ``target``.

    """
    if value.ndim > target.ndim:
        msg = f"Cannot broadcast shape {tuple(value.shape)} to {tuple(target.shape)}."
        raise ValueError(msg)
    return value.reshape(*value.shape, *((1,) * (target.ndim - value.ndim)))


def random_rigid_motion(
    coords: Float[torch.Tensor, "*batch N 3"],
    mask: Bool[torch.Tensor, "*batch N"] | None = None,
    trans_scale: float = 1.0,
    *,
    generator: torch.Generator | None = None,
) -> Float[torch.Tensor, "*batch N 3"]:
    """Center, randomly rotate and randomly translate each structure.

    Deliberately not :func:`team_gm.utils.transform.random_augmentation` or
    :meth:`team_gm.diffusion.Diffuser.random_rotation_and_translation`: both draw
    rotations through ``scipy.spatial.transform.Rotation.random``, which reads
    NumPy's global RNG and so cannot be pinned by a :class:`torch.Generator`.
    Rotations here come from normalised quaternions sampled with torch, so a
    seeded generator reproduces the whole sampling trajectory.

    Parameters
    ----------
    coords : Tensor
        Coordinates to re-pose.
    mask : Tensor or None
        Validity mask used for the center of mass.
    trans_scale : float
        Standard deviation of the random translation.
    generator : torch.Generator or None
        Source of randomness.

    Returns
    -------
    Tensor
        Re-posed coordinates.

    """
    batch = coords.shape[:-2]
    weights = (
        torch.ones_like(coords[..., :1])
        if mask is None
        else mask.to(coords.dtype).unsqueeze(-1)
    )
    total = weights.sum(dim=-2, keepdim=True).clamp(min=1.0)
    centered = coords - (coords * weights).sum(dim=-2, keepdim=True) / total

    quat = torch.randn(
        (*batch, 4), device=coords.device, dtype=coords.dtype, generator=generator
    )
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = quat.unbind(-1)
    rotation = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*batch, 3, 3)

    rotated = torch.einsum("...ij,...nj->...ni", rotation, centered)
    shift = trans_scale * torch.randn(
        (*batch, 1, 3), device=coords.device, dtype=coords.dtype, generator=generator
    )
    return rotated + shift
