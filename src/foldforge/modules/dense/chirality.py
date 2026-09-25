"""Gradient of the chiral-centre dihedral error, RoseTTAFold3's atom input.

RoseTTAFold3's diffusion encoder adds a projection of d/dx sum (dih - target)^2
over the plane pairs of every chiral centre (``data.features.chirality``),
taken on the scaled noisy coordinates at every step, so the denoiser can tell
a chiral centre from its mirror image. The release computes it in closed form
rather than by autograd, which also keeps it usable under inference mode; this
is that closed form (rf3 ``calc_ddihedralmse_dxyz``), over any leading axes.
"""

from __future__ import annotations

import torch


def dihedral_error_gradient(
    positions: torch.Tensor, centres: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """d/dx of the summed squared dihedral error, per atom.

    ``positions`` is (..., n_atoms, 3), ``centres`` (n, 4) flat atom indices,
    ``targets`` (n,) radians. Returns (..., n_atoms, 3); zero with no centres.
    """
    grads = torch.zeros_like(positions, dtype=torch.float32)
    if centres.shape[0] == 0:
        return grads
    eps = 1e-6
    x = positions.float()[..., centres, :]  # (..., n, 4, 3)
    a, b, c, d = x.unbind(-2)
    b0, b1, b2 = a - b, c - b, d - c
    b1_norm = torch.linalg.norm(b1, dim=-1, keepdim=True)
    b1n = b1 / (b1_norm + eps)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    px = (v * w).sum(-1)
    py = (torch.cross(b1n, v, dim=-1) * w).sum(-1)
    dih = torch.atan2(py + eps, px + eps)

    eye = torch.eye(3, device=x.device, dtype=x.dtype)
    outer = lambda p, q: p[..., :, None] * q[..., None, :]
    dmse = (2 * (dih - targets.to(x))).unsqueeze(-1)  # (..., n, 1)
    denom = px**2 + py**2 + eps
    ddih_dx = (-py / denom).unsqueeze(-1)
    ddih_dy = (px / denom).unsqueeze(-1)
    dy_dv = -torch.cross(b1n, w, dim=-1)
    dy_dw = torch.cross(b1n, v, dim=-1)
    dx_dv, dx_dw = w, v
    norm2 = (b1_norm**2 + eps).unsqueeze(-1)
    db1n_db1 = (b1_norm + eps).unsqueeze(-1) * eye / norm2 - outer(b1, b1) / norm2
    dw_db1n = -(b2 * b1n).sum(-1)[..., None, None] * eye - outer(b2, b1n)
    dv_db1n = -(b0 * b1n).sum(-1)[..., None, None] * eye - outer(b0, b1n)
    project = eye - outer(b1n, b1n)  # dv/db0 and dw/db2
    row = lambda vec, mat: (vec[..., None, :] @ mat).squeeze(-2)

    def through(dv: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
        """The error's gradient given dv/dp and dw/dp for one atom p."""
        dx = row(dx_dv, dv) + row(dx_dw, dw)
        dy = row(dy_dv, dv) + row(dy_dw, dw)
        return dmse * (ddih_dx * dx + ddih_dy * dy)

    zero = torch.zeros_like(project)
    db1n_db = -db1n_db1
    grad_a = through(project, zero)
    grad_b = through(
        -project + dv_db1n.transpose(-1, -2) @ db1n_db,
        dw_db1n.transpose(-1, -2) @ db1n_db,
    )
    grad_c = through(
        dv_db1n.transpose(-1, -2) @ db1n_db1,
        -project + dw_db1n.transpose(-1, -2) @ db1n_db1,
    )
    grad_d = through(zero, project)
    per_centre = torch.stack([grad_a, grad_b, grad_c, grad_d], dim=-2)  # (..., n, 4, 3)
    grads.index_add_(grads.ndim - 2, centres.reshape(-1), per_centre.flatten(-3, -2))
    return grads
