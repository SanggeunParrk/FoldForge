"""Normalize AF-family confidence units and atom/token layouts."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import torch

from foldforge.prediction import Prediction


def from_atom_confidence(output: Any) -> Prediction:
    """Average original-token pLDDT; these decoded atom scores are already [0,1]."""
    full = output["full_data"]
    plddt = []
    for sample in full:
        atom_scores = sample["atom_plddt"].float()
        index = sample["atom_to_token_idx"].long()
        n_token = len(sample["token_asym_id"])
        scores = atom_scores.new_zeros(n_token)
        count = atom_scores.new_zeros(n_token)
        scores.scatter_add_(0, index, atom_scores)
        count.scatter_add_(0, index, torch.ones_like(atom_scores))
        plddt.append(scores / count.clamp_min(1))
    summary = output["summary_confidence"]
    ptm = torch.stack([s["ptm"] for s in summary])
    multi_chain = torch.unique(full[0]["token_asym_id"]).numel() > 1
    iptm = torch.stack([s["iptm"] for s in summary]) if multi_chain else None
    return Prediction(
        coords=output["coordinate"],
        plddt=torch.stack(plddt),
        pae=torch.stack([s["token_pair_pae"] for s in full]),
        ptm=ptm,
        iptm=iptm,
    )


def from_af3(
    output: Any, flat_coordinates: torch.Tensor, token_atom_mask: torch.Tensor
) -> Prediction:
    """AF3 dense atom scores to token pLDDT in [0,1] and flat coordinates."""
    scores = output["predicted_lddt"].float() / 100.0
    mask = token_atom_mask.to(scores.device).float()
    plddt = (scores * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    return Prediction(coords=flat_coordinates, plddt=plddt, pae=output["full_pae"])


def cpu_payload(prediction: Prediction) -> dict[str, torch.Tensor | None]:
    """Save tensors rather than a Python-class pickle; absent heads stay None."""
    return {
        field.name: None
        if (value := getattr(prediction, field.name)) is None
        else value.detach().cpu()
        for field in fields(prediction)
    }


def structure_with_confidence(
    atoms: Any, coordinates: Any, atom_plddt: torch.Tensor
) -> Any:
    """Keep original atom identities and store pLDDT on the CIF's 0-100 scale."""
    if tuple(atom_plddt.shape) != (len(atoms),):
        message = "Atom pLDDT must have exactly one value per output atom"
        raise ValueError(message)
    structure = atoms.copy()
    structure.coord = coordinates
    structure.set_annotation(
        "b_factor", atom_plddt.detach().float().cpu().numpy() * 100.0
    )
    return structure
