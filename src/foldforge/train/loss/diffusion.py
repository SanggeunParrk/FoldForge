"""Diffusion training loss on the dense atom layout ``[res, atoms_per_token]`` (AF3 SI Eq. 2-5, EDM weighting).

The weighted rigid alignment (Horn's quaternion, GPU tensor ops only, no SVD / host sync), the GPU-side rotation and the GPU-side noise
draw are copied from MiniWorld's ``training/phase2_graph_safe.py`` (``weighted_align_gs``, ``rot_trans_gs``, ``sample_noise_gs``,
``cal_loss_gs``, ``bond_loss_gs``) so the same step can later be captured into a CUDA graph (phase 3). The head of this graph returns the
preconditioned prediction ``x_pred = c_skip * x_noisy + c_out * update`` directly (``reconstruct_coordinates``), so the loss is taken on it.
"""

from __future__ import annotations

import torch

from foldforge.train.config import EDMNoise


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack(
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
        -1,
    ).reshape(*q.shape[:-1], 3, 3)


def horn_rot(H: torch.Tensor, squarings: int = 24) -> torch.Tensor:
    """R_col maximising tr(R_col H) (H_ab = sum_i w x_a y_b, so R_col x ~ y): top eigenvector of Horn's 4x4 matrix by repeated squaring."""
    Sxx, Sxy, Sxz = H[:, 0, 0], H[:, 0, 1], H[:, 0, 2]
    Syx, Syy, Syz = H[:, 1, 0], H[:, 1, 1], H[:, 1, 2]
    Szx, Szy, Szz = H[:, 2, 0], H[:, 2, 1], H[:, 2, 2]
    N = torch.stack(
        [
            torch.stack([Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx], -1),
            torch.stack([Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz], -1),
            torch.stack([Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy], -1),
            torch.stack([Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz], -1),
        ],
        -2,
    )
    nrm = N.flatten(-2).norm(dim=-1)[:, None, None].clamp_min(1e-12)
    M = (
        N + nrm * torch.eye(4, device=H.device, dtype=H.dtype)
    ) / nrm  # eigenvalues in [0, 2]
    for _ in range(squarings):
        M = (M.unsqueeze(-1) * M.unsqueeze(-3)).sum(
            -2
        )  # M @ M in exact fp32 (no tensor cores)
        M = M / M.flatten(-2).norm(dim=-1)[:, None, None].clamp_min(1e-30)
    u0 = torch.arange(1, 5, device=H.device, dtype=H.dtype)
    q = (M * u0).sum(-1)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return quat_to_rotmat(q)


@torch.no_grad()
def weighted_align(
    x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """``x`` rigidly aligned onto ``y`` with per-atom ``weight`` (``[..., L, 3]``, ``[..., L]``); no gradient."""
    shape, length = x.shape, x.shape[-2]
    x, y, w = (
        x.reshape(-1, length, 3),
        y.reshape(-1, length, 3),
        weight.reshape(-1, length),
    )
    w_sum = w.sum(dim=-1, keepdim=True)
    w = w.unsqueeze(-1)
    xc = ((x * w).sum(dim=-2) / w_sum).unsqueeze(-2)
    yc = ((y * w).sum(dim=-2) / w_sum).unsqueeze(-2)
    x_cen, y_cen = x - xc, y - yc
    H = ((x_cen * w).unsqueeze(-1) * y_cen.unsqueeze(-2)).sum(dim=-3)
    R = horn_rot(H).transpose(-1, -2)  # row-vector convention
    aligned = (x_cen.unsqueeze(-1) * R.unsqueeze(-3)).sum(dim=-2) + yc
    return aligned.reshape(shape)


def sample_sigma(noise: EDMNoise, n: int, device: torch.device) -> torch.Tensor:
    """EDM noise levels ``sigma_data * exp(P_mean + P_std * N(0,1))`` clamped to ``[sigma_min, sigma_max] * sigma_data`` ... as MiniWorld's
    ``sample_noise_gs`` (uniform -> ndtri on the GPU)."""
    u = torch.rand(n, device=device)
    sigma = noise.sigma_data * torch.exp(
        noise.P_mean + noise.P_std * torch.special.ndtri(u)
    )
    return torch.clamp(
        sigma,
        min=noise.sigma_min * noise.sigma_data,
        max=noise.sigma_max * noise.sigma_data,
    )


def random_rotation_translation(
    x: torch.Tensor, translation_noise: float
) -> torch.Tensor:
    """Per-sample uniform rotation (normalised Gaussian quaternion) and Gaussian translation of ``x`` ``[S, ..., 3]``."""
    shape = x.shape
    flat = x.reshape(shape[0], -1, 3)
    q = torch.randn(shape[0], 4, device=x.device, dtype=x.dtype)
    rot = quat_to_rotmat(q / q.norm(dim=-1, keepdim=True))
    t = torch.randn(shape[0], 1, 3, device=x.device, dtype=x.dtype) * translation_noise
    return (torch.bmm(flat, rot.transpose(-1, -2)) + t).reshape(shape)


def atom_weights(
    is_dna: torch.Tensor,
    is_rna: torch.Tensor,
    is_ligand: torch.Tensor,
    atom_mask: torch.Tensor,
    *,
    alpha_dna: float,
    alpha_rna: float,
    alpha_ligand: float,
) -> torch.Tensor:
    """AF3 Eq. 4 per-atom weight ``1 + a_dna * is_dna + a_rna * is_rna + a_ligand * is_ligand`` on ``[res, atoms]`` (token flags broadcast to atoms)."""
    token = (
        1.0
        + alpha_dna * is_dna.float()
        + alpha_rna * is_rna.float()
        + alpha_ligand * is_ligand.float()
    )
    return token[:, None].expand_as(atom_mask).to(torch.float32) * atom_mask.to(
        torch.float32
    ) + (~atom_mask).to(torch.float32)


def diffusion_mse(
    x_pred: torch.Tensor,
    x0: torch.Tensor,
    sigma: torch.Tensor,
    atom_mask: torch.Tensor,
    atom_weight: torch.Tensor,
    sigma_data: float,
) -> torch.Tensor:
    """EDM-weighted, atom-weighted MSE between the prediction and the ground truth aligned onto it (AF3 Eq. 2-4).

    ``x_pred`` / ``x0`` ``[S, res, atoms, 3]`` (``x0`` augmented like the noisy input), ``sigma`` ``[S]``, ``atom_mask`` / ``atom_weight``
    ``[res, atoms]``. The alignment uses the same atom weights as the MSE and carries no gradient."""
    s = x_pred.shape[0]
    pred = x_pred.reshape(s, -1, 3).float()
    truth = x0.reshape(s, -1, 3).float()
    mask = atom_mask.reshape(1, -1).expand(s, -1)
    weight_atom = atom_weight.reshape(1, -1).expand(s, -1).float()
    pred_safe = torch.where(mask[..., None], pred, 0.0)
    truth_safe = torch.where(mask[..., None], truth, 0.0)
    align_weight = (
        torch.where(mask.any(dim=-1, keepdim=True), mask, True).float() * weight_atom
    )
    truth_aligned = weighted_align(truth_safe, pred_safe.detach(), align_weight)
    sq = torch.where(mask[..., None], (pred_safe - truth_aligned).pow(2), 0.0)
    lam = (
        (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
    ).float()  # EDM loss weight
    numerator = (sq * weight_atom[..., None]).sum(dim=(-2, -1)) * lam
    denominator = mask.sum(dim=-1).clamp_min(1).float() * 3
    return (numerator / denominator).mean()


def bond_loss(
    x_pred: torch.Tensor,
    x0: torch.Tensor,
    sigma: torch.Tensor,
    bond_pairs: torch.Tensor | None,
    sigma_data: float,
) -> torch.Tensor:
    """AF3 Eq. 5: MSE of bonded-atom-pair distances, per sample weighted by the EDM weight. ``bond_pairs`` ``[P, 4]`` int
    ``(token_i, atom_i, token_j, atom_j)`` on the dense layout, ``-1`` rows are padding; ``None`` (no bonds) gives 0."""
    if bond_pairs is None or bond_pairs.numel() == 0:
        return x_pred.sum() * 0.0
    valid = (bond_pairs >= 0).all(dim=-1)
    ti, ai, tj, aj = (bond_pairs.clamp_min(0)[:, k] for k in range(4))

    def distance(x: torch.Tensor) -> torch.Tensor:
        delta = x[..., ti, ai, :] - x[..., tj, aj, :]
        return torch.where(valid[:, None], delta, 1.0).norm(dim=-1)

    count = valid.sum().clamp_min(1).float()
    per_sample = ((distance(x_pred.float()) - distance(x0.float())).pow(2) * valid).sum(
        dim=-1
    ) / count  # [S]
    lam = ((sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2).float()
    return (lam * per_sample).mean()
