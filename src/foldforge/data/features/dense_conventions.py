"""Input conventions a dense-graph family was trained under.

These act on the featurised example before it reaches the network. They are part
of a family's weights as much as its parameters are: applying one family's
convention to another is a mistake, not a feature, and a missing one fails
silently as a worse fold. Each is a field of the family's ``DenseSpec``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from foldforge.modules.dense.spec import DenseSpec


def centre_conformers(example: dict[str, Any]) -> None:
    """Subtract each reference conformer's own masked mean from ``ref_pos``.

    Grouped by ``ref_space_uid`` as the vendors group it. Padding stays zero: the
    mask is what every consumer reads.
    """
    positions = np.asarray(example["ref_pos"], dtype=np.float32)
    flat = positions.reshape(-1, 3)
    weight = (np.asarray(example["ref_mask"]).reshape(-1) > 0).astype(np.float32)
    group = np.asarray(example["ref_space_uid"]).reshape(-1).astype(np.int64)
    if not group.size:
        return
    groups = int(group.max()) + 1
    count = np.maximum(np.bincount(group, weights=weight, minlength=groups), 1.0)
    centre = (
        np.stack(
            [
                np.bincount(group, weights=weight * flat[:, axis], minlength=groups)
                for axis in range(3)
            ],
            axis=-1,
        )
        / count[:, None]
    )
    example["ref_pos"] = ((flat - centre[group]) * weight[:, None]).reshape(
        positions.shape
    )


def apply(example: dict[str, Any], spec: DenseSpec) -> dict[str, Any]:
    """Apply ``spec``'s input conventions to one featurised example, in place."""
    if spec.centre_ref_conformers:
        centre_conformers(example)
    return example
