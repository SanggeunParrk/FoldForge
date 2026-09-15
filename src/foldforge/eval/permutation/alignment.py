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

import torch

from foldforge.eval.rmsd import align_pred_to_true


def get_optimal_transform(
    src_atoms: torch.Tensor,
    tgt_atoms: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A function that obtain the transformation that optimally align.

    src_atoms to tgt_atoms.

    Args:
        src_atoms: ground-truth centre atom positions, shape: [N, 3]
        tgt_atoms: predicted centre atom positions, shape: [N, 3]
        mask: a vector of boolean values, shape: [N]

    Returns:
        tuple[torch.Tensor]: A rotation matrix that records the optimal rotation
                             that will best align src_atoms to tgt_atoms.
                             A tanslation matrix records how the atoms should be
                             shifted after applying r.
    """
    if src_atoms.shape != tgt_atoms.shape:
        message = (src_atoms.shape, tgt_atoms.shape)
        raise ValueError(message)
    if src_atoms.shape[-1] != 3:
        message = "Invalid state: src_atoms.shape[-1] == 3"
        raise ValueError(message)
    if mask is not None:
        mask = mask.bool()
        if mask.dim() != 1:
            message = "mask should have the shape of [N]"
            raise ValueError(message)
        if mask.shape[-1] != src_atoms.shape[-2]:
            message = "Invalid state: mask.shape[-1] == src_atoms.shape[-2]"
            raise ValueError(message)
        src_atoms = src_atoms[mask, :]
        tgt_atoms = tgt_atoms[mask, :]

    with torch.autocast("cuda", enabled=False):
        _, rot, trans = align_pred_to_true(
            pred_pose=src_atoms.to(dtype=torch.float32),
            true_pose=tgt_atoms.to(dtype=torch.float32),
            allowing_reflection=False,
        )  # svd alignment does not support BF16

    return rot, trans


def apply_transform(pose, rot, trans):
    """Apply transform."""
    return torch.matmul(pose, rot.transpose(-1, -2)) + trans


def num_unique_matches(match_list: list[dict]) -> int:
    """Compute num unique matches."""
    return len({tuple(sorted(match.items())) for match in match_list})
