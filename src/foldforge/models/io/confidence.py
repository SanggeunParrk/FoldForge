# Optional chemistry/GPU backends load at the selected execution boundary.
# ruff: noqa: PLC0415
"""Normalize AF-family confidence units and atom/token layouts."""

from __future__ import annotations

import math
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
        pde=torch.stack([s["token_pair_pde"] for s in full])
        if all("token_pair_pde" in s for s in full)
        else None,
        ptm=ptm,
        iptm=iptm,
    )


def predicted_tm(tm_adjusted_pae: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
    """AlphaFold 3's pTM / ipTM from a TM-adjusted PAE ``[S, N, N]`` (the expected ``1 / (1 + (e / d0)^2)`` of the PAE bins, ``d0`` from the
    number of tokens, global or interface) and the pairs that count ``[N, N]``: for every alignment token ``i`` the mean over the counted
    ``j``, then the best ``i``. ``pair_mask`` is every valid pair for pTM, the pairs of different chains for ipTM
    (``alphafold3.model.confidences.predicted_tm_score``)."""
    mask = pair_mask.to(tm_adjusted_pae.dtype)
    normed = mask / (1e-8 + mask.sum(-1, keepdim=True))
    return (tm_adjusted_pae * normed).sum(-1).max(-1).values


def from_af3(
    output: Any,
    flat_coordinates: torch.Tensor,
    token_atom_mask: torch.Tensor,
    asym_id: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
) -> Prediction:
    """AF3 dense atom scores to token pLDDT in [0,1] and flat coordinates; pTM and (with more than one chain) ipTM from the head's TM-adjusted
    PAE when the chain index and the token mask of the fold are given."""
    scores = output["predicted_lddt"].float() / 100.0
    mask = token_atom_mask.to(scores.device).float()
    plddt = (scores * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    ptm = iptm = None
    global_apae, interface_apae = output.get("tmscore_adjusted_pae_global"), output.get("tmscore_adjusted_pae_interface")
    if global_apae is not None and interface_apae is not None and asym_id is not None and token_mask is not None:
        device = global_apae.device
        valid = token_mask.to(device).bool()
        asym = asym_id.to(device)
        pair = valid[:, None] & valid[None, :]
        ptm = predicted_tm(global_apae.float(), pair)
        different = pair & (asym[:, None] != asym[None, :])
        if bool(different.any()):
            iptm = predicted_tm(interface_apae.float(), different)
    return Prediction(
        coords=flat_coordinates,
        plddt=plddt,
        pae=output["full_pae"],
        pde=output.get("full_pde"),
        ptm=ptm,
        iptm=iptm,
    )


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


def summary_for_json(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Encode undefined chain-pair PAE as null, keeping other NaNs invalid.

    Released flat-atom checkpoints use NaN when a chain pair has no valid local
    frames (for example, an ion). Canonical confidence tensors and raw checkpoint
    output remain unchanged; this conversion only defines their JSON encoding.
    """
    from foldforge.models.io.output import json_value

    def missing_pae(value: Any) -> Any:
        """Compute missing pae."""
        if isinstance(value, list):
            return [missing_pae(item) for item in value]
        return None if isinstance(value, float) and math.isnan(value) else value

    encoded = json_value(summary)
    for sample in encoded:
        for key in ("chain_pair_pae_mean", "chain_pair_pae_min"):
            if key in sample:
                sample[key] = missing_pae(sample[key])
    return encoded
