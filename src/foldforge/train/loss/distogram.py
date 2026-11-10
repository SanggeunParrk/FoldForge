"""Distogram loss: cross entropy between the head's logits and the binned ground-truth pseudo-beta distances.

The bins are the head's own (``DistogramHead.breaks``: ``num_bins - 1`` edges from ``first_break`` to ``last_break``); the ground-truth
pseudo-beta atom of each token is gathered with the batch's ``token_atoms_to_pseudo_beta`` (CB, CA for glycine, ... as the confidence head).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from foldforge.utils import atom_layout


def distogram_loss(
    logits: torch.Tensor,
    breaks: torch.Tensor,
    x0: torch.Tensor,
    atom_mask: torch.Tensor,
    token_atoms_to_pseudo_beta,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """``logits`` ``[res, res, bins]``, ``x0`` ``[res, atoms, 3]``, ``atom_mask`` ``[res, atoms]``, ``token_mask`` ``[res]``."""
    with torch.no_grad():
        beta = atom_layout.convert(
            token_atoms_to_pseudo_beta, x0.float(), layout_axes=(-3, -2)
        )  # [res, 3]
        beta_ok = (
            atom_layout.convert(
                token_atoms_to_pseudo_beta, atom_mask.float(), layout_axes=(-2, -1)
            )
            > 0.5
        )
        distance = torch.cdist(beta, beta)
        target = torch.bucketize(distance, breaks.to(distance.dtype))
        valid = (
            beta_ok[:, None]
            & beta_ok[None, :]
            & (token_mask > 0)[:, None]
            & (token_mask > 0)[None, :]
        )
    per_pair = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
    ).reshape(target.shape)
    return (per_pair * valid).sum() / valid.sum().clamp_min(1)
