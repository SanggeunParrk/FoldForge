# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research

from typing import Any

import torch
from team_gm.metrics.confidence import (
    break_down_to_per_sample_dict,
    calculate_chain_based_gpde,
    calculate_chain_based_ptm,
    calculate_iptm,
    calculate_ptm,
    get_bin_centers,
    logits_to_score,
)
from team_gm.modules.checkpoints.tensor_utils import distogram_bin_tops

from foldforge.eval.clash import Clash
from foldforge.models.config.opendde.config.schema import BinConfig, OpenDDEConfig
from foldforge.models.config.types import ConfigNode
from foldforge.utils.distributed import traverse_and_aggregate


def detailed_merge_per_sample_confidence_scores(
    summary_confidence_list: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge confidence scores from multiple samples into a single dictionary.

    Args:
        summary_confidence_list (list[dict[str, Any]]): List of dictionaries containing
            confidence scores for each sample.

    Returns:
        dict[str, Any]: Merged dictionary of confidence scores.
    """

    def stack_score(tensor_list: list[torch.Tensor]) -> torch.Tensor:
        """Compute stack score."""
        if tensor_list[0].dim() == 0:
            tensor_list = [x.unsqueeze(0) for x in tensor_list]
        score = torch.stack(tensor_list, dim=0)
        return score

    return traverse_and_aggregate(summary_confidence_list, aggregation_func=stack_score)


def _detailed_compute_full_data_and_summary(  # noqa: PLR0915 - released confidence equations
    configs: ConfigNode,
    pae_logits: torch.Tensor,
    plddt_logits: torch.Tensor,
    pde_logits: torch.Tensor,
    contact_probs: torch.Tensor,
    token_asym_id: torch.Tensor,
    token_has_frame: torch.Tensor,
    atom_coordinate: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    atom_is_polymer: torch.Tensor,
    N_recycle: int,
    interested_atom_mask: torch.Tensor | None = None,
    elements_one_hot: torch.Tensor | None = None,
    mol_id: torch.Tensor | None = None,
    return_full_data: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Compute full data and summary confidence scores for the given inputs.

    Args:
        configs: Configuration object.
        pae_logits (torch.Tensor): Logits for PAE (Predicted Aligned Error).
        plddt_logits (torch.Tensor): Logits for pLDDT (Predicted Local Distance
            Difference Test).
        pde_logits (torch.Tensor): Logits for PDE (Predicted Distance Error).
        contact_probs (torch.Tensor): Contact probabilities.
        token_asym_id (torch.Tensor): Asymmetric ID for tokens.
        token_has_frame (torch.Tensor): Indicator for tokens having a frame.
        atom_coordinate (torch.Tensor): Atom coordinates.
        atom_to_token_idx (torch.Tensor): Mapping from atoms to tokens.
        atom_is_polymer (torch.Tensor): Indicator for atoms being part of a polymer.
        N_recycle (int): Number of recycles.
        interested_atom_mask (Optional[torch.Tensor]): Mask for interested atoms.
            Defaults to None.
        elements_one_hot (Optional[torch.Tensor]): One-hot encoding for elements.
            Defaults to None.
        mol_id (Optional[torch.Tensor]): Molecular ID. Defaults to None.
        return_full_data (bool): Whether to return full data. Defaults to False.

    Returns:
        tuple[list[dict], list[dict]]:
            - summary_confidence: List of dictionaries containing summary confidence
            scores.
            - full_data: List of dictionaries containing full data if
            `return_full_data` is True.
    """
    atom_is_ligand = (1 - atom_is_polymer).long()
    token_is_ligand = torch.zeros_like(token_asym_id).scatter_add(
        0, atom_to_token_idx, atom_is_ligand
    )
    token_is_ligand = token_is_ligand > 0

    full_data = {}
    full_data["atom_plddt"] = logits_to_score(
        plddt_logits, **get_bin_params(configs.loss.plddt)
    )  # [N_s, N_atom]
    # Cpu offload for saving cuda memory
    pde_logits = pde_logits.to(plddt_logits.device)
    full_data["token_pair_pde"] = logits_to_score(
        pde_logits, **get_bin_params(configs.loss.pde)
    )  # [N_s, N_token, N_token]
    del pde_logits
    full_data["contact_probs"] = contact_probs.clone()  # [N_token, N_token]
    pae_logits = pae_logits.to(plddt_logits.device)
    full_data["token_pair_pae"], pae_prob = logits_to_score(
        pae_logits, **get_bin_params(configs.loss.pae), return_prob=True
    )  # [N_s, N_token, N_token]
    del pae_logits

    summary_confidence = {}
    summary_confidence["plddt"] = full_data["atom_plddt"].mean(dim=-1) * 100  # [N_s, ]
    summary_confidence["gpde"] = (
        full_data["token_pair_pde"] * full_data["contact_probs"]
    ).sum(dim=[-1, -2]) / full_data["contact_probs"].sum(dim=[-1, -2])

    summary_confidence["ptm"] = calculate_ptm(
        pae_prob, has_frame=token_has_frame, **get_bin_params(configs.loss.pae)
    )  # [N_s, ]
    summary_confidence["iptm"] = calculate_iptm(
        pae_prob,
        has_frame=token_has_frame,
        asym_id=token_asym_id,
        **get_bin_params(configs.loss.pae),
    )  # [N_s, ]

    # Add: 'chain_gpde', 'chain_pair_gpde'
    summary_confidence.update(
        calculate_chain_based_gpde(
            token_pair_pde=full_data["token_pair_pde"],
            contact_probs=full_data["contact_probs"],
            asym_id=token_asym_id,
        )
    )
    # Add: 'chain_pair_iptm', 'chain_pair_iptm_global' 'chain_iptm', 'chain_ptm'
    summary_confidence.update(
        calculate_chain_based_ptm(
            pae_prob,
            has_frame=token_has_frame,
            asym_id=token_asym_id,
            token_is_ligand=token_is_ligand,
            **get_bin_params(configs.loss.pae),
        )
    )
    # Add: 'chain_plddt', 'chain_pair_plddt'
    summary_confidence.update(
        detailed_calculate_chain_based_plddt(
            full_data["atom_plddt"], token_asym_id, atom_to_token_idx
        )
    )
    # Add: 'chain_pair_pae_mean', 'chain_pair_pae_min'
    summary_confidence.update(
        calculate_chain_pair_pae(
            token_pair_pae=full_data["token_pair_pae"],
            asym_id=token_asym_id,
            token_has_frame=token_has_frame,
        )
    )
    del pae_prob
    summary_confidence["has_clash"] = detailed_calculate_clash(
        atom_coordinate,
        token_asym_id,
        atom_to_token_idx,
        atom_is_polymer,
        configs.metrics.clash.af3_clash_threshold,
    )
    summary_confidence["num_recycles"] = torch.tensor(
        N_recycle, device=atom_coordinate.device
    )

    summary_confidence["disorder"] = torch.zeros_like(summary_confidence["ptm"])
    summary_confidence["ranking_score"] = (
        0.8 * summary_confidence["iptm"]
        + 0.2 * summary_confidence["ptm"]
        + 0.5 * summary_confidence["disorder"]
        - 100 * summary_confidence["has_clash"]
    )
    if interested_atom_mask is not None:
        token_idx = atom_to_token_idx[interested_atom_mask[0].bool()].long()
        asym_ids = token_asym_id[token_idx]
        if len(torch.unique(asym_ids)) != 1:
            message = "Invalid state: len(torch.unique(asym_ids)) == 1"
            raise ValueError(message)
        interested_asym_id = int(asym_ids[0].item())
        N_chains = int(token_asym_id.max().long().item()) + 1
        pb_ranking_score = summary_confidence["chain_pair_iptm_global"][
            :, interested_asym_id, torch.arange(N_chains) != interested_asym_id
        ]  # [N_s, N_chain - 1]
        summary_confidence["pb_ranking_score"] = pb_ranking_score[:, 0]
        if elements_one_hot is not None and mol_id is not None:
            vdw_clash = detailed_calculate_vdw_clash(
                pred_coordinate=atom_coordinate,
                asym_id=token_asym_id,
                mol_id=mol_id,
                is_polymer=atom_is_polymer,
                atom_token_idx=atom_to_token_idx,
                elements_one_hot=elements_one_hot,
                threshold=configs.metrics.clash.vdw_clash_threshold,
            )
            N_sample = atom_coordinate.shape[0]
            vdw_clash_per_sample_flag = (
                vdw_clash[:, interested_asym_id, :].reshape(N_sample, -1).max(dim=-1)[0]
            )
            summary_confidence["has_vdw_pl_clash"] = vdw_clash_per_sample_flag
            summary_confidence["pb_ranking_score_vdw_penalized"] = (
                summary_confidence["pb_ranking_score"] - 100 * vdw_clash_per_sample_flag
            )

    summary_confidence = break_down_to_per_sample_dict(
        summary_confidence, shared_keys=["num_recycles"]
    )

    if return_full_data:
        # save extra inputs that are used for computing summary_confidence
        full_data["token_has_frame"] = token_has_frame.clone()
        full_data["token_asym_id"] = token_asym_id.clone()
        full_data["atom_to_token_idx"] = atom_to_token_idx.clone()
        full_data["atom_is_polymer"] = atom_is_polymer.clone()
        full_data["atom_coordinate"] = atom_coordinate.clone()

        full_data = break_down_to_per_sample_dict(
            full_data,
            shared_keys=[
                "contact_probs",
                "token_has_frame",
                "token_asym_id",
                "atom_to_token_idx",
                "atom_is_polymer",
            ],
        )
        return summary_confidence, full_data
    return summary_confidence, [{}]


def get_bin_params(cfg: ConfigNode | BinConfig) -> dict:
    """Extract bin parameters from the configuration object."""
    return {"min_bin": cfg.min_bin, "max_bin": cfg.max_bin, "no_bins": cfg.no_bins}


def detailed_compute_contact_prob(
    distogram_logits: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    thres: float = 8.0,
) -> torch.Tensor:
    """Compute the contact probability from distogram logits.

    Args:
        distogram_logits (torch.Tensor): Logits for the distogram.
            Shape: [N_token, N_token, N_bins]
        min_bin (float): Minimum bin value.
        max_bin (float): Maximum bin value.
        no_bins (int): Number of bins.
        thres (float): Threshold distance for contact probability. Defaults to 8.0.

    Returns:
        torch.Tensor: Contact probability.
            Shape: [N_token, N_token]
    """
    distogram_prob = torch.nn.functional.softmax(
        distogram_logits, dim=-1
    )  # [N_token, N_token, N_bins]
    distogram_bins = get_bin_centers(min_bin, max_bin, no_bins)
    thres_idx = (distogram_bins < thres).sum()
    contact_prob = distogram_prob[..., :thres_idx].sum(-1)
    del distogram_prob
    return contact_prob


def detailed_calculate_vdw_clash(
    pred_coordinate: torch.Tensor,
    asym_id: torch.Tensor,
    mol_id: torch.Tensor,
    atom_token_idx: torch.Tensor,
    is_polymer: torch.Tensor,
    elements_one_hot: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Calculate Van der Waals (VDW) clash for predicted coordinates.

    Args:
        pred_coordinate (torch.Tensor): Predicted coordinates of atoms.
            Shape: [N_sample, N_atom, 3]
        asym_id (torch.LongTensor): Asymmetric ID for tokens.
            Shape: [N_token]
        mol_id (torch.LongTensor): Molecular ID.
            Shape: [N_atom]
        atom_token_idx (torch.LongTensor): Mapping from atoms to tokens.
            Shape: [N_atom]
        is_polymer (torch.BoolTensor): Indicator for atoms being part of a polymer.
            Shape: [N_atom]
        elements_one_hot (torch.Tensor): One-hot encoding for elements.
            Shape: [N_atom, N_elements]
        threshold (float): Threshold for VDW clash detection.

    Returns:
        torch.Tensor: VDW clash summary.
            Shape: [N_sample]
    """
    clash_calculator = Clash(
        details_dtype=torch.bool, vdw_clash_threshold=threshold, compute_af3_clash=False
    )
    # Check ligand-polymer VDW clash
    pred_coordinate.shape[0]
    dummy_is_dna = torch.zeros_like(is_polymer)
    dummy_is_rna = torch.zeros_like(is_polymer)
    clash_dict = clash_calculator(
        pred_coordinate=pred_coordinate,
        asym_id=asym_id,
        atom_to_token_idx=atom_token_idx,
        mol_id=mol_id,
        is_ligand=1 - is_polymer,
        is_protein=is_polymer,
        is_dna=dummy_is_dna,
        is_rna=dummy_is_rna,
        elements_one_hot=elements_one_hot,
    )
    return clash_dict["summary"]["vdw_clash"]


def detailed_calculate_clash(
    pred_coordinate: torch.Tensor,
    asym_id: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    is_polymer: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Check complex clash.

    Args:
        pred_coordinate (torch.Tensor): [N_sample, N_atom, 3]
        asym_id (torch.LongTensor): [N_token, ]
        atom_to_token_idx (torch.LongTensor): [N_atom, ]
        is_polymer (torch.BoolTensor): [N_atom, ]
        threshold: (float)

    Returns:
        torch.Tensor: [N_sample] whether there is a clash in the complex
    """
    N_sample = pred_coordinate.shape[0]
    dummy_is_dna = torch.zeros_like(is_polymer)
    dummy_is_rna = torch.zeros_like(is_polymer)
    clash_calculator = Clash(
        details_dtype=torch.bool, vdw_clash_threshold=threshold, compute_vdw_clash=False
    )
    clash_dict = clash_calculator(
        pred_coordinate,
        asym_id,
        atom_to_token_idx,
        1 - is_polymer,
        is_polymer,
        dummy_is_dna,
        dummy_is_rna,
    )
    return clash_dict["summary"]["af3_clash"].reshape(N_sample, -1).max(dim=-1)[0]


def calculate_chain_pair_pae(
    token_pair_pae: torch.Tensor,
    asym_id: torch.Tensor,
    token_has_frame: torch.Tensor,
    contact_probs: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Calculate chain-pair PAE values.

    Args:
        token_pair_pae (torch.Tensor): PAE (Predicted Aligned Error) of token-token
            pairs.
            [..., N_token, N_token]
        asym_id (torch.LongTensor): Asymmetric ID for tokens.
            [N_token]
        token_has_frame (torch.BoolTensor): Indicator for tokens having a frame.
            [N_token]
        contact_probs (torch.Tensor | None): Optional contact probabilities.
            [..., N_token, N_token]
        eps (float): Small value to avoid division by zero.

    Returns:
        dict[str, torch.Tensor]: Dictionary containing chain-pair PAE values.
            - chain_pair_pae_mean (torch.Tensor): Mean PAE for chain pairs.
            - chain_pair_pae_min (torch.Tensor): Min PAE for chain pairs.
    """
    asym_id = asym_id.long()
    unique_asym_ids = torch.unique(asym_id)
    N_chain = len(unique_asym_ids)
    if N_chain != asym_id.max() + 1:
        # asym_id has gaps (chains were filtered out); remap to contiguous 0..N_chain-1
        remap = {old.item(): new for new, old in enumerate(unique_asym_ids)}
        asym_id = torch.tensor(
            [remap[x.item()] for x in asym_id], dtype=torch.long, device=asym_id.device
        )

    batch_shape = token_pair_pae.shape[:-2]
    device = token_pair_pae.device

    if contact_probs is None:
        contact_probs = torch.ones(
            token_pair_pae.shape[1:], dtype=torch.float64, device=device
        )

    mask = token_has_frame[:, None] & token_has_frame[None, :]  # [N_token, N_token]
    if mask.shape != token_pair_pae.shape[1:]:
        message = "Invalid state: mask.shape == token_pair_pae.shape[1:]"
        raise ValueError(message)

    chain_pair_pae_mean = torch.zeros(
        size=(*batch_shape, N_chain, N_chain), device=device
    )
    chain_pair_pae_min = torch.zeros(
        size=(*batch_shape, N_chain, N_chain), device=device
    )

    for aid_1 in range(N_chain):
        mask_1 = asym_id == aid_1
        sub_pae = token_pair_pae[..., mask_1, :]
        sub_mask = mask[mask_1, :]
        sub_contact_probs = contact_probs[mask_1, :]
        for aid_2 in range(N_chain):
            mask_2 = asym_id == aid_2

            subsub_pae = sub_pae[..., mask_2]
            subsub_mask = sub_mask[..., mask_2]
            subsub_contact_probs = sub_contact_probs[..., mask_2]

            (flat_subsub_mask_idxs,) = torch.where(subsub_mask.flatten() > 0)
            flat_subsub_pae = subsub_pae.view(batch_shape[0], -1)
            flat_subsub_contact_probs = subsub_contact_probs.flatten()

            if flat_subsub_mask_idxs.numel() == 0:
                chain_pair_pae_mean[..., aid_1, aid_2] = torch.nan
                chain_pair_pae_min[..., aid_1, aid_2] = torch.nan
            else:
                valid_pae = flat_subsub_pae[:, flat_subsub_mask_idxs]
                valid_contact_probs = flat_subsub_contact_probs[flat_subsub_mask_idxs]

                # min
                chain_pair_pae_min[..., aid_1, aid_2] = valid_pae.min(dim=-1).values

                # weighted mean
                chain_pair_pae_mean[..., aid_1, aid_2] = (
                    valid_contact_probs * valid_pae
                ).mean(dim=-1) / (valid_contact_probs.mean(dim=-1) + eps)

    return {
        "chain_pair_pae_mean": chain_pair_pae_mean,
        "chain_pair_pae_min": chain_pair_pae_min,
    }


def detailed_calculate_chain_based_plddt(
    atom_plddt: torch.Tensor,
    asym_id: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Calculate chain-based pLDDT scores.

    Args:
        atom_plddt (torch.Tensor): Predicted pLDDT scores for atoms.
            Shape: [N_sample, N_atom]
        asym_id (torch.LongTensor): Asymmetric ID for tokens.
            Shape: [N_token]
        atom_to_token_idx (torch.LongTensor): Mapping from atoms to tokens.
            Shape: [N_atom]

    Returns:
        dict: Dictionary containing chain-based pLDDT scores.
            - chain_plddt (torch.Tensor): pLDDT scores for each chain.
            - chain_pair_plddt (torch.Tensor): Pairwise pLDDT scores between chains.
    """
    asym_id = asym_id.long()
    unique_asym_ids = torch.unique(asym_id)
    if len(unique_asym_ids) != asym_id.max() + 1:
        remap = {old.item(): new for new, old in enumerate(unique_asym_ids)}
        asym_id = torch.tensor(
            [remap[x.item()] for x in asym_id], dtype=torch.long, device=asym_id.device
        )
    asym_id_to_asym_mask = {aid.item(): asym_id == aid for aid in torch.unique(asym_id)}
    N_chain = len(asym_id_to_asym_mask)
    if N_chain != asym_id.max() + 1:
        message = "Invalid state: N_chain == asym_id.max() + 1"
        raise ValueError(message)  # make sure it is from 0 to N_chain-1

    def _calculate_lddt_with_token_mask(token_mask):
        atom_mask = token_mask[atom_to_token_idx]
        sub_plddt = atom_plddt[:, atom_mask].mean(-1)
        return sub_plddt

    batch_shape = atom_plddt.shape[:-1]
    # Chain_plddt
    chain_plddt = torch.zeros(size=(*batch_shape, N_chain)).to(atom_plddt.device)
    for aid, asym_mask in asym_id_to_asym_mask.items():
        chain_plddt[:, aid] = _calculate_lddt_with_token_mask(token_mask=asym_mask)

    # Chain_pair_plddt
    chain_pair_plddt = torch.zeros(size=(*batch_shape, N_chain, N_chain)).to(
        atom_plddt.device
    )
    for aid_1, mask_1 in asym_id_to_asym_mask.items():
        for aid_2, mask_2 in asym_id_to_asym_mask.items():
            if aid_1 == aid_2:
                continue
            pair_mask = mask_1 + mask_2
            chain_pair_plddt[:, aid_1, aid_2] = _calculate_lddt_with_token_mask(
                token_mask=pair_mask
            )

    return {"chain_plddt": chain_plddt, "chain_pair_plddt": chain_pair_plddt}


@torch.no_grad()
def detailed_compute_full_data_and_summary(
    configs: ConfigNode,
    pae_logits: torch.Tensor,
    plddt_logits: torch.Tensor,
    pde_logits: torch.Tensor,
    contact_probs: torch.Tensor,
    token_asym_id: torch.Tensor,
    token_has_frame: torch.Tensor,
    atom_coordinate: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    atom_is_polymer: torch.Tensor,
    N_recycle: int,
    return_full_data: bool = False,
    interested_atom_mask: torch.Tensor | None = None,
    mol_id: torch.Tensor | None = None,
    elements_one_hot: torch.Tensor | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Wrapper of `_compute_full_data_and_summary` by enumerating over N samples."""
    N_sample = pae_logits.size(0)
    if contact_probs.dim() == 2:
        # Convert to [N_sample, N_token, N_token]
        contact_probs = contact_probs.unsqueeze(dim=0).expand(N_sample, -1, -1)
    elif contact_probs.dim() != 3:
        message = "Invalid state: contact_probs.dim() == 3"
        raise ValueError(message)
    if not (
        contact_probs.size(0) == plddt_logits.size(0) == pde_logits.size(0) == N_sample
    ):
        message = (
            "Invalid state: contact_probs.size(0) == plddt_logits.size(0) == "
            "pde_logits.size(0) == N_sample"
        )
        raise ValueError(message)

    summary_confidence = []
    full_data = []
    for i in range(N_sample):
        summary_confidence_i, full_data_i = _detailed_compute_full_data_and_summary(
            configs=configs,
            pae_logits=pae_logits[i : i + 1],
            plddt_logits=plddt_logits[i : i + 1],
            pde_logits=pde_logits[i : i + 1],
            contact_probs=contact_probs[i],
            token_asym_id=token_asym_id,
            token_has_frame=token_has_frame,
            atom_coordinate=atom_coordinate[i : i + 1],
            atom_to_token_idx=atom_to_token_idx,
            atom_is_polymer=atom_is_polymer,
            N_recycle=N_recycle,
            interested_atom_mask=interested_atom_mask,
            return_full_data=return_full_data,
            mol_id=mol_id,
            elements_one_hot=elements_one_hot,
        )
        summary_confidence.extend(summary_confidence_i)
        full_data.extend(full_data_i)
    return summary_confidence, full_data


def compact_merge_per_sample_confidence_scores(
    summary_confidence_list: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge confidence scores from multiple samples into a single dictionary.

    Args:
        summary_confidence_list (list[dict[str, Any]]): List of dictionaries containing
            confidence scores for each sample.

    Returns:
        dict[str, Any]: Merged dictionary of confidence scores.
    """

    def stack_score(value_list: list[Any]) -> Any:
        """Compute stack score."""
        if not all(isinstance(x, torch.Tensor) for x in value_list):
            return value_list
        if value_list[0].dim() == 0:
            value_list = [x.unsqueeze(0) for x in value_list]
        score = torch.stack(value_list, dim=0)
        return score

    return traverse_and_aggregate(summary_confidence_list, aggregation_func=stack_score)


def _offload_confidence_tree_to_cpu(value: Any) -> Any:
    """Release per-sample CUDA confidence outputs before the next sample."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu")
    if isinstance(value, dict):
        return {
            key: _offload_confidence_tree_to_cpu(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_offload_confidence_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_offload_confidence_tree_to_cpu(item) for item in value)
    return value


def _compact_compute_full_data_and_summary(  # noqa: PLR0915 - released confidence equations
    configs: OpenDDEConfig,
    pae_logits: torch.Tensor,
    plddt_logits: torch.Tensor,
    pde_logits: torch.Tensor,
    contact_probs: torch.Tensor,
    token_asym_id: torch.Tensor,
    token_has_frame: torch.Tensor,
    atom_coordinate: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    atom_is_polymer: torch.Tensor,
    N_recycle: int,
    interested_atom_mask: torch.Tensor | None = None,
    elements_one_hot: torch.Tensor | None = None,
    mol_id: torch.Tensor | None = None,
    return_full_data: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Compute full data and summary confidence scores for the given inputs.

    Args:
        configs: Configuration object.
        pae_logits (torch.Tensor): Logits for PAE (Predicted Aligned Error).
        plddt_logits (torch.Tensor): Logits for pLDDT (Predicted Local Distance
            Difference Test).
        pde_logits (torch.Tensor): Logits for PDE (Predicted Distance Error).
        contact_probs (torch.Tensor): Contact probabilities.
        token_asym_id (torch.Tensor): Asymmetric ID for tokens.
        token_has_frame (torch.Tensor): Indicator for tokens having a frame.
        atom_coordinate (torch.Tensor): Atom coordinates.
        atom_to_token_idx (torch.Tensor): Mapping from atoms to tokens.
        atom_is_polymer (torch.Tensor): Indicator for atoms being part of a polymer.
        N_recycle (int): Number of recycles.
        interested_atom_mask (Optional[torch.Tensor]): Mask for interested atoms.
            Defaults to None.
        elements_one_hot (Optional[torch.Tensor]): One-hot encoding for elements.
            Defaults to None.
        mol_id (Optional[torch.Tensor]): Molecular ID. Defaults to None.
        return_full_data (bool): Whether to return full data. Defaults to False.

    Returns:
        tuple[list[dict], list[dict]]:
            - summary_confidence: List of dictionaries containing summary confidence
            scores.
            - full_data: List of dictionaries containing full data if
            `return_full_data` is True.
    """
    atom_is_ligand = (1 - atom_is_polymer).long()
    token_is_ligand = torch.zeros_like(token_asym_id).scatter_add(
        0, atom_to_token_idx, atom_is_ligand
    )
    token_is_ligand = token_is_ligand > 0

    full_data: dict[str, Any] = {}
    atom_plddt = logits_to_score(
        plddt_logits, **get_bin_params(configs.confidence.plddt)
    )  # [N_s, N_atom]
    if return_full_data:
        full_data["atom_plddt"] = atom_plddt
    # Keep confidence logits on the same device while converting logits to scores.
    pde_logits = pde_logits.to(plddt_logits.device)
    token_pair_pde = logits_to_score(
        pde_logits, **get_bin_params(configs.confidence.pde)
    )  # [N_s, N_token, N_token]
    del pde_logits
    if return_full_data:
        full_data["token_pair_pde"] = token_pair_pde
        full_data["contact_probs"] = contact_probs.clone()  # [N_token, N_token]
    pae_logits = pae_logits.to(plddt_logits.device)
    pae_score, pae_prob = logits_to_score(
        pae_logits,
        **get_bin_params(configs.confidence.pae),
        return_prob=True,
    )  # [N_s, N_token, N_token]
    del pae_logits
    if return_full_data:
        full_data["token_pair_pae"] = pae_score
    else:
        del pae_score

    summary_confidence: dict[str, Any] = {}
    summary_confidence["plddt"] = atom_plddt.mean(dim=-1) * 100  # [N_s, ]
    summary_confidence["gpde"] = (token_pair_pde * contact_probs).sum(
        dim=[-1, -2]
    ) / contact_probs.sum(dim=[-1, -2])

    summary_confidence["ptm"] = calculate_ptm(
        pae_prob, has_frame=token_has_frame, **get_bin_params(configs.confidence.pae)
    )  # [N_s, ]
    summary_confidence["iptm"] = calculate_iptm(
        pae_prob,
        has_frame=token_has_frame,
        asym_id=token_asym_id,
        **get_bin_params(configs.confidence.pae),
    )  # [N_s, ]

    # Add: 'chain_gpde', 'chain_pair_gpde'
    summary_confidence.update(
        calculate_chain_based_gpde(
            token_pair_pde=token_pair_pde,
            contact_probs=contact_probs,
            asym_id=token_asym_id,
        )
    )
    # Add: 'chain_pair_iptm', 'chain_pair_iptm_global' 'chain_iptm', 'chain_ptm'
    summary_confidence.update(
        calculate_chain_based_ptm(
            pae_prob,
            has_frame=token_has_frame,
            asym_id=token_asym_id,
            token_is_ligand=token_is_ligand,
            **get_bin_params(configs.confidence.pae),
        )
    )
    # Add: 'chain_plddt', 'chain_pair_plddt'
    summary_confidence.update(
        compact_calculate_chain_based_plddt(
            atom_plddt, token_asym_id, atom_to_token_idx
        )
    )
    del pae_prob
    summary_confidence["has_clash"] = compact_calculate_clash(
        atom_coordinate,
        token_asym_id,
        atom_to_token_idx,
        atom_is_polymer,
        configs.metrics.clash.af3_clash_threshold,
    )
    summary_confidence["num_recycles"] = torch.tensor(
        N_recycle, device=atom_coordinate.device
    )

    summary_confidence["disorder"] = torch.zeros_like(summary_confidence["ptm"])
    summary_confidence["ranking_score"] = (
        0.8 * summary_confidence["iptm"]
        + 0.2 * summary_confidence["ptm"]
        + 0.5 * summary_confidence["disorder"]
        - 100 * summary_confidence["has_clash"]
    )
    if interested_atom_mask is not None:
        token_idx = atom_to_token_idx[interested_atom_mask[0].bool()].long()
        asym_ids = token_asym_id[token_idx]
        if len(torch.unique(asym_ids)) != 1:
            message = "Invalid state: len(torch.unique(asym_ids)) == 1"
            raise ValueError(message)
        interested_asym_id = int(asym_ids[0].item())
        N_chains = int(token_asym_id.max().long().item()) + 1
        pb_ranking_score = summary_confidence["chain_pair_iptm_global"][
            :, interested_asym_id, torch.arange(N_chains) != interested_asym_id
        ]  # [N_s, N_chain - 1]
        summary_confidence["pb_ranking_score"] = pb_ranking_score[:, 0]
        if elements_one_hot is not None and mol_id is not None:
            vdw_clash = compact_calculate_vdw_clash(
                pred_coordinate=atom_coordinate,
                asym_id=token_asym_id,
                mol_id=mol_id,
                is_polymer=atom_is_polymer,
                atom_token_idx=atom_to_token_idx,
                elements_one_hot=elements_one_hot,
                threshold=configs.metrics.clash.vdw_clash_threshold,
            )
            N_sample = atom_coordinate.shape[0]
            vdw_clash_per_sample_flag = (
                vdw_clash[:, interested_asym_id, :].reshape(N_sample, -1).max(dim=-1)[0]
            )
            summary_confidence["has_vdw_pl_clash"] = vdw_clash_per_sample_flag
            summary_confidence["pb_ranking_score_vdw_penalized"] = (
                summary_confidence["pb_ranking_score"] - 100 * vdw_clash_per_sample_flag
            )

    summary_confidence_by_sample = break_down_to_per_sample_dict(
        summary_confidence, shared_keys=["num_recycles"]
    )

    if return_full_data:
        # save extra inputs that are used for computing summary_confidence
        full_data["token_has_frame"] = token_has_frame.clone()
        full_data["token_asym_id"] = token_asym_id.clone()
        full_data["atom_to_token_idx"] = atom_to_token_idx.clone()
        full_data["atom_is_polymer"] = atom_is_polymer.clone()
        full_data["atom_coordinate"] = atom_coordinate.clone()

        full_data_by_sample = break_down_to_per_sample_dict(
            full_data,
            shared_keys=[
                "contact_probs",
                "token_has_frame",
                "token_asym_id",
                "atom_to_token_idx",
                "atom_is_polymer",
            ],
        )
        return summary_confidence_by_sample, full_data_by_sample
    return summary_confidence_by_sample, [{}]


def compact_compute_contact_prob(
    distogram_logits: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    thres: float = 8.0,
) -> torch.Tensor:
    """Compute the contact probability from distogram logits.

    Args:
        distogram_logits (torch.Tensor): Logits for the distogram.
            Shape: [N_token, N_token, N_bins]
        min_bin (float): Minimum bin value.
        max_bin (float): Maximum bin value.
        no_bins (int): Number of bins.
        thres (float): Threshold distance for contact probability. Defaults to 8.0.

    Returns:
        torch.Tensor: Contact probability.
            Shape: [N_token, N_token]
    """
    distogram_prob = torch.nn.functional.softmax(
        distogram_logits, dim=-1
    )  # [N_token, N_token, N_bins]
    bin_tops = distogram_bin_tops(
        min_bin=min_bin,
        max_bin=max_bin,
        no_bins=no_bins,
        device=distogram_logits.device,
        dtype=distogram_logits.dtype,
    )
    contact_prob = distogram_prob[..., bin_tops <= thres].sum(-1)
    del distogram_prob
    return contact_prob


def compact_calculate_vdw_clash(
    pred_coordinate: torch.Tensor,
    asym_id: torch.Tensor,
    mol_id: torch.Tensor,
    atom_token_idx: torch.Tensor,
    is_polymer: torch.Tensor,
    elements_one_hot: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Calculate Van der Waals (VDW) clash for predicted coordinates.

    Args:
        pred_coordinate (torch.Tensor): Predicted coordinates of atoms.
            Shape: [N_sample, N_atom, 3]
        asym_id (torch.LongTensor): Asymmetric ID for tokens.
            Shape: [N_token]
        mol_id (torch.LongTensor): Molecular ID.
            Shape: [N_atom]
        atom_token_idx (torch.LongTensor): Mapping from atoms to tokens.
            Shape: [N_atom]
        is_polymer (torch.BoolTensor): Indicator for atoms being part of a polymer.
            Shape: [N_atom]
        elements_one_hot (torch.Tensor): One-hot encoding for elements.
            Shape: [N_atom, N_elements]
        threshold (float): Threshold for VDW clash detection.

    Returns:
        torch.Tensor: VDW clash summary.
            Shape: [N_sample]
    """
    clash_calculator = Clash(vdw_clash_threshold=threshold, compute_af3_clash=False)
    # Check ligand-polymer VDW clash
    pred_coordinate.shape[0]
    dummy_is_dna = torch.zeros_like(is_polymer)
    dummy_is_rna = torch.zeros_like(is_polymer)
    clash_dict = clash_calculator(
        pred_coordinate=pred_coordinate,
        asym_id=asym_id,
        atom_to_token_idx=atom_token_idx,
        mol_id=mol_id,
        is_ligand=1 - is_polymer,
        is_protein=is_polymer,
        is_dna=dummy_is_dna,
        is_rna=dummy_is_rna,
        elements_one_hot=elements_one_hot,
    )
    return clash_dict["summary"]["vdw_clash"]


def compact_calculate_clash(
    pred_coordinate: torch.Tensor,
    asym_id: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    is_polymer: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Check complex clash.

    Args:
        pred_coordinate (torch.Tensor): [N_sample, N_atom, 3]
        asym_id (torch.LongTensor): [N_token, ]
        atom_to_token_idx (torch.LongTensor): [N_atom, ]
        is_polymer (torch.BoolTensor): [N_atom, ]
        threshold: (float)

    Returns:
        torch.Tensor: [N_sample] whether there is a clash in the complex
    """
    N_sample = pred_coordinate.shape[0]
    dummy_is_dna = torch.zeros_like(is_polymer)
    dummy_is_rna = torch.zeros_like(is_polymer)
    clash_calculator = Clash(vdw_clash_threshold=threshold, compute_vdw_clash=False)
    clash_dict = clash_calculator(
        pred_coordinate,
        asym_id,
        atom_to_token_idx,
        1 - is_polymer,
        is_polymer,
        dummy_is_dna,
        dummy_is_rna,
    )
    return clash_dict["summary"]["af3_clash"].reshape(N_sample, -1).max(dim=-1)[0]


def compact_calculate_chain_based_plddt(
    atom_plddt: torch.Tensor,
    asym_id: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Calculate chain-based pLDDT scores.

    Args:
        atom_plddt (torch.Tensor): Predicted pLDDT scores for atoms.
            Shape: [N_sample, N_atom]
        asym_id (torch.LongTensor): Asymmetric ID for tokens.
            Shape: [N_token]
        atom_to_token_idx (torch.LongTensor): Mapping from atoms to tokens.
            Shape: [N_atom]

    Returns:
        dict: Dictionary containing chain-based pLDDT scores.
            - chain_plddt (torch.Tensor): pLDDT scores for each chain.
            - chain_pair_plddt (torch.Tensor): Pairwise pLDDT scores between chains.
    """
    asym_id = asym_id.long()
    unique_asym_ids = torch.unique(asym_id)
    if len(unique_asym_ids) != asym_id.max() + 1:
        remap = {old.item(): new for new, old in enumerate(unique_asym_ids)}
        asym_id = torch.tensor(
            [remap[x.item()] for x in asym_id], dtype=torch.long, device=asym_id.device
        )
    asym_id_to_asym_mask = {aid.item(): asym_id == aid for aid in torch.unique(asym_id)}
    N_chain = len(asym_id_to_asym_mask)
    if N_chain != asym_id.max() + 1:
        message = "Invalid state: N_chain == asym_id.max() + 1"
        raise ValueError(message)  # make sure it is from 0 to N_chain-1

    def _calculate_lddt_with_token_mask(token_mask):
        atom_mask = token_mask[atom_to_token_idx]
        sub_plddt = atom_plddt[:, atom_mask]
        # MPS has no float64; fall back to float32 accumulation there.
        accum_dtype = torch.float32 if sub_plddt.device.type == "mps" else torch.float64
        return sub_plddt.to(accum_dtype).mean(-1).to(atom_plddt.dtype)

    batch_shape = atom_plddt.shape[:-1]
    # Chain_plddt
    chain_plddt = torch.zeros(size=(*batch_shape, N_chain)).to(atom_plddt.device)
    for aid, asym_mask in asym_id_to_asym_mask.items():
        chain_plddt[:, aid] = _calculate_lddt_with_token_mask(token_mask=asym_mask)

    # Chain_pair_plddt
    chain_pair_plddt = torch.zeros(size=(*batch_shape, N_chain, N_chain)).to(
        atom_plddt.device
    )
    for aid_1, mask_1 in asym_id_to_asym_mask.items():
        for aid_2, mask_2 in asym_id_to_asym_mask.items():
            if aid_1 == aid_2:
                continue
            pair_mask = mask_1 + mask_2
            chain_pair_plddt[:, aid_1, aid_2] = _calculate_lddt_with_token_mask(
                token_mask=pair_mask
            )

    return {"chain_plddt": chain_plddt, "chain_pair_plddt": chain_pair_plddt}


@torch.no_grad()
def compact_compute_full_data_and_summary(
    configs: OpenDDEConfig,
    pae_logits: torch.Tensor,
    plddt_logits: torch.Tensor,
    pde_logits: torch.Tensor,
    contact_probs: torch.Tensor,
    token_asym_id: torch.Tensor,
    token_has_frame: torch.Tensor,
    atom_coordinate: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    atom_is_polymer: torch.Tensor,
    N_recycle: int,
    return_full_data: bool = False,
    interested_atom_mask: torch.Tensor | None = None,
    mol_id: torch.Tensor | None = None,
    elements_one_hot: torch.Tensor | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Wrapper of `_compute_full_data_and_summary` by enumerating over N samples."""
    N_sample = pae_logits.size(0)
    compute_device = plddt_logits.device
    token_asym_id = token_asym_id.to(device=compute_device)
    token_has_frame = token_has_frame.to(device=compute_device)
    atom_coordinate = atom_coordinate.to(device=compute_device)
    atom_to_token_idx = atom_to_token_idx.to(device=compute_device)
    atom_is_polymer = atom_is_polymer.to(device=compute_device)
    if interested_atom_mask is not None:
        interested_atom_mask = interested_atom_mask.to(device=compute_device)
    if mol_id is not None:
        mol_id = mol_id.to(device=compute_device)
    if elements_one_hot is not None:
        elements_one_hot = elements_one_hot.to(device=compute_device)
    if contact_probs.dim() == 2:
        # Convert to [N_sample, N_token, N_token]
        contact_probs = contact_probs.unsqueeze(dim=0).expand(N_sample, -1, -1)
    elif contact_probs.dim() != 3:
        message = "Invalid state: contact_probs.dim() == 3"
        raise ValueError(message)
    if not (
        contact_probs.size(0) == plddt_logits.size(0) == pde_logits.size(0) == N_sample
    ):
        message = (
            "Invalid state: contact_probs.size(0) == plddt_logits.size(0) == "
            "pde_logits.size(0) == N_sample"
        )
        raise ValueError(message)

    summary_confidence = []
    full_data = []
    for i in range(N_sample):
        summary_confidence_i, full_data_i = _compact_compute_full_data_and_summary(
            configs=configs,
            pae_logits=pae_logits[i : i + 1].to(device=compute_device),
            plddt_logits=plddt_logits[i : i + 1],
            pde_logits=pde_logits[i : i + 1].to(device=compute_device),
            contact_probs=contact_probs[i].to(device=compute_device),
            token_asym_id=token_asym_id,
            token_has_frame=token_has_frame,
            atom_coordinate=atom_coordinate[i : i + 1],
            atom_to_token_idx=atom_to_token_idx,
            atom_is_polymer=atom_is_polymer,
            N_recycle=N_recycle,
            interested_atom_mask=interested_atom_mask,
            return_full_data=return_full_data,
            mol_id=mol_id,
            elements_one_hot=elements_one_hot,
        )
        # ``return_full_data`` retains O(N_token^2) PAE/PDE/contact maps for
        # every sample.  Move each completed tree off CUDA before evaluating
        # the next sample instead of waiting for the outer model return path.
        summary_confidence_i = _offload_confidence_tree_to_cpu(summary_confidence_i)
        full_data_i = _offload_confidence_tree_to_cpu(full_data_i)
        summary_confidence.extend(summary_confidence_i)
        full_data.extend(full_data_i)
    return summary_confidence, full_data
