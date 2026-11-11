"""Confidence-head loss of the single-stage (full) finetuning: per-bin cross entropy of the pLDDT, PDE and PAE logits against targets computed
from the mini-rollout structure and the ground truth (MiniWorld ``loss/confidence.py`` + ``models/confidence/client.py::loss_fn``, ported to
the dense ``[res, atoms_per_token]`` layout).

All targets are built under ``no_grad``; the bins are the head's own (``distance_breaks`` for PDE, ``pae_breaks`` for PAE, 50 uniform lDDT
bins for pLDDT), so a target bin is the bin the head's expected value is read from.

  * pLDDT: per-atom lDDT of the rollout structure against the ground truth (AF3 section 4.3.1: nucleic neighbours within 30 A, others 15 A;
    a ligand atom counts polymer neighbours only), over the real atoms only.
  * PDE: ``|d_ij(pred) - d_ij(gt)|`` between the token representative (pseudo-beta) atoms.
  * PAE: the frame-aligned error of the representative atoms, in the per-token frames (N, CA, C for protein; C1', C3', C4' for nucleic acids;
    a token without the three atoms, or whose frame is near colinear, has no frame and drops out of the loss).
The experimentally-resolved head has no MiniWorld counterpart and is not trained (weight 0 in the recipe).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from foldforge.data.constants import atom_types, residue_names
from foldforge.utils import atom_layout

if TYPE_CHECKING:
    from foldforge.modules.dense.head import ConfidenceHead

PROTEIN_FRAME = (atom_types.N, atom_types.CA, atom_types.C)
NUCLEIC_FRAME = (atom_types.C1PRIME, atom_types.C3PRIME, atom_types.C4PRIME)


@lru_cache(maxsize=1)
def _frame_slot_table() -> tuple[list[tuple[int, int, int]], list[bool]]:
    """Per ``aatype`` index the dense slots of its three frame atoms (and whether it has a frame)."""
    slots, valid = [], []
    for name in residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP:
        order = atom_types.DENSE_ATOM_ORDER.get(name)
        found = None
        if order:
            for names in (PROTEIN_FRAME, NUCLEIC_FRAME):
                if all(n in order for n in names):
                    found = tuple(order[n] for n in names)
                    break
        slots.append(found or (0, 0, 0))
        valid.append(found is not None)
    return slots, valid


def frame_slots(
    aatype: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(slots [res, 3], has_frame [res])`` of the tokens; an ``aatype`` outside the polymer alphabet (ligand) has no frame."""
    slots, valid = _frame_slot_table()
    table = torch.tensor(slots, dtype=torch.long, device=aatype.device)
    ok = torch.tensor(valid, dtype=torch.bool, device=aatype.device)
    index = aatype.long().clamp(0, len(slots) - 1)
    in_range = (aatype >= 0) & (aatype < len(slots))
    return table[index], ok[index] & in_range


def per_atom_lddt(
    pred: torch.Tensor,
    gt: torch.Tensor,
    is_nuc: torch.Tensor,
    is_ligand: torch.Tensor,
    max_distance: float = 15.0,
    nuc_max_distance: float = 30.0,
    distance_bins: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-atom lDDT ``[N]`` and the validity ``[N]`` (at least one neighbour) of ``N`` real atoms (``pred`` / ``gt`` ``[N, 3]``)."""
    pred_dist = torch.cdist(pred, pred)
    gt_dist = torch.cdist(gt, gt)
    radius = torch.where(
        is_nuc, gt_dist.new_tensor(nuc_max_distance), gt_dist.new_tensor(max_distance)
    )
    pair = (gt_dist > 0.0) & (gt_dist < radius[None, :])
    pair &= ~(is_ligand[:, None] & is_ligand[None, :])
    delta = (pred_dist - gt_dist).abs()
    bins = torch.tensor(distance_bins, dtype=torch.float32, device=pred.device)
    kept = (delta[..., None] <= bins) & pair[..., None]
    frac = kept.sum(dim=-2).float() / (pair.sum(dim=-1, keepdim=True).float() + 1e-8)
    return frac.mean(dim=-1), pair.sum(dim=-1) > 0


def token_frames(
    atoms: torch.Tensor, frame_ok: torch.Tensor, colinear_cos: float = 0.9063
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rigid frames ``(R [res, 3, 3], t [res, 3], valid [res])`` from the three frame atoms ``[res, 3, 3]`` (Gram-Schmidt; centre = atom 1;
    near-colinear atoms (angle below 25 degrees) invalidate the frame)."""
    a0, a1, a2 = atoms[:, 0], atoms[:, 1], atoms[:, 2]
    v1, v2 = a2 - a1, a0 - a1
    e1 = v1 / (v1.norm(dim=-1, keepdim=True) + 1e-8)
    v2n = v2 / (v2.norm(dim=-1, keepdim=True) + 1e-8)
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
    e2 = u2 / (u2.norm(dim=-1, keepdim=True) + 1e-8)
    e3 = torch.cross(e1, e2, dim=-1)
    rot = torch.stack([e1, e2, e3], dim=-1)
    colinear = (e1 * v2n).sum(-1).abs() > colinear_cos
    return rot, a1, frame_ok & ~colinear


def _local(
    rot: torch.Tensor, t: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Representative atom ``j`` in the frame of token ``i``: ``R_i^T (x_j - t_i)`` -> ``[res_i, res_j, 3]`` (``rot`` columns are the basis
    vectors, so component ``k`` is ``sum_c rot[i, c, k] * d[c]``; MiniWorld's ``pae_target_bins`` contracts the other axis of ``rot``, i.e. applies
    ``R`` instead of ``R^T`` -- not carried over)."""
    return torch.einsum("ick,ijc->ijk", rot, x[None, :, :] - t[:, None, :])


@torch.no_grad()
def confidence_targets(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    atom_mask: torch.Tensor,
    token_atoms_to_pseudo_beta,
    token_mask: torch.Tensor,
    aatype: torch.Tensor,
    is_nuc: torch.Tensor,
    is_ligand: torch.Tensor,
    head: ConfidenceHead,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """``{"plddt" | "pde" | "pae": (target bins, mask)}``. ``x_pred`` / ``x_gt`` ``[res, atoms, 3]``, ``atom_mask`` ``[res, atoms]``
    (ground-truth atoms present), token tensors ``[res]``."""
    res, atoms, _ = x_pred.shape
    x_pred, x_gt = x_pred.float(), x_gt.float()
    present = atom_mask.bool()

    # pLDDT over the real atoms only (a dense crop has many empty slots).
    flat = present.reshape(-1)
    index = flat.nonzero(as_tuple=True)[0]
    token_of_atom = index // atoms
    lddt, valid = per_atom_lddt(
        x_pred.reshape(-1, 3)[index],
        x_gt.reshape(-1, 3)[index],
        is_nuc.bool()[token_of_atom],
        is_ligand.bool()[token_of_atom],
    )
    lddt_dense = torch.zeros(res * atoms, device=x_pred.device)
    valid_dense = torch.zeros(res * atoms, dtype=torch.bool, device=x_pred.device)
    lddt_dense[index] = lddt
    valid_dense[index] = valid
    num_bins = head.num_plddt_bins
    plddt_bins = (lddt_dense.clamp(0.0, 1.0 - 1e-6) * num_bins).long().reshape(res, atoms)

    # Representative atoms.
    beta_pred = atom_layout.convert(token_atoms_to_pseudo_beta, x_pred, layout_axes=(-3, -2))
    beta_gt = atom_layout.convert(token_atoms_to_pseudo_beta, x_gt, layout_axes=(-3, -2))
    beta_ok = (
        atom_layout.convert(
            token_atoms_to_pseudo_beta, present.float(), layout_axes=(-2, -1)
        )
        > 0.5
    )
    tok_ok = beta_ok & (token_mask > 0)
    pair_ok = tok_ok[:, None] & tok_ok[None, :]

    pde_err = (torch.cdist(beta_pred, beta_pred) - torch.cdist(beta_gt, beta_gt)).abs()
    pde_bins = torch.bucketize(pde_err, head.distance_breaks.float())

    # Frame-aligned error.
    slots, has_frame = frame_slots(aatype)
    rows = torch.arange(res, device=x_pred.device)[:, None]
    pred_frame_atoms = x_pred[rows, slots]
    gt_frame_atoms = x_gt[rows, slots]
    observed = present[rows, slots].all(dim=-1)
    r_pred, t_pred, ok_pred = token_frames(pred_frame_atoms, has_frame)
    r_gt, t_gt, ok_gt = token_frames(gt_frame_atoms, has_frame & observed)
    pae_err = (
        _local(r_pred, t_pred, beta_pred) - _local(r_gt, t_gt, beta_gt)
    ).norm(dim=-1)
    pae_bins = torch.bucketize(pae_err, head.pae_breaks.float())
    pae_mask = (ok_pred & ok_gt & (token_mask > 0))[:, None] & tok_ok[None, :]

    return {
        "plddt": (plddt_bins, valid_dense.reshape(res, atoms)),
        "pde": (pde_bins, pair_ok),
        "pae": (pae_bins, pae_mask),
    }


def masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean cross entropy over the masked positions (``logits`` ``[..., C]``, ``target`` / ``mask`` ``[...]``)."""
    classes = logits.shape[-1]
    ce = F.cross_entropy(
        logits.float().reshape(-1, classes),
        target.reshape(-1).clamp(0, classes - 1),
        reduction="none",
    )
    weight = mask.reshape(-1).float()
    return (ce * weight).sum() / (weight.sum() + 1e-8)


def confidence_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    plddt_loss: float,
    pde_loss: float,
    pae_loss: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted sum of the per-bin cross entropies of ``outputs["plddt_logits" | "pde_logits" | "pae_logits"]`` (the head run with
    ``save_logits``). Returns ``(total, {name: loss})``."""
    parts = {
        "plddt": masked_ce(outputs["plddt_logits"], *targets["plddt"]),
        "pde": masked_ce(outputs["pde_logits"], *targets["pde"]),
        "pae": masked_ce(outputs["pae_logits"], *targets["pae"]),
    }
    total = plddt_loss * parts["plddt"] + pde_loss * parts["pde"] + pae_loss * parts["pae"]
    return total, parts
