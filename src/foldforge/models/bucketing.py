"""Inference padding uses explicit semantic axes.

Autotune floor/clamping selects a *configuration*, never a physical tensor size.
Tokens round up to multiples of 128. Atoms retain the engine allocation ladder;
MSA uses its own ladder. Real inputs are never truncated.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

import torch
from miniworld_engine.autotune.shape_key import ATOM_SHAPES

TOKEN_BUCKET_SIZE = 128
# Finite sequence for AF3's featurizer API, covering every supported atom count.
# This is an inference allocation list, not the engine's autotune TOKEN_SHAPES.
TOKEN_SHAPES = tuple(range(TOKEN_BUCKET_SIZE, ATOM_SHAPES[-1] + 1, TOKEN_BUCKET_SIZE))
# 1024 is the shared inference policy's per-recycle MSA sample; without it every
# adapter padded those rows to 2048 and ran the MSA stack half on padding.
MSA_SHAPES = (1024, 2048, 4096, 8192, 16384)
# Inference atom allocation ladder. The engine tunes up to 8192 atoms and clamps
# larger launches onto that configuration; the two extra rungs only size padding
# so long homo-oligomer inputs can run until memory, not the ladder, stops them.
ATOM_BUCKETS = (*ATOM_SHAPES, 16384, 32768)


def token_bucket(size: int) -> int:
    """Round real or expanded structural tokens up without an autotune-size cap."""
    if size < 1:
        message = f"tokens must be positive, got {size}"
        raise ValueError(message)
    return ((size + TOKEN_BUCKET_SIZE - 1) // TOKEN_BUCKET_SIZE) * TOKEN_BUCKET_SIZE


def ceiling(size: int, edges: tuple[int, ...], axis: str) -> int:
    """Choose the smallest supported extent containing every real element."""
    if size < 1:
        message = f"{axis} must be positive, got {size}"
        raise ValueError(message)
    index = bisect_left(edges, size)
    if index == len(edges):
        message = f"{axis}={size} exceeds largest inference bucket {edges[-1]}"
        raise ValueError(message)
    return edges[index]


def pad_axis(
    value: torch.Tensor, axis: int, extent: int, fill: int = 0
) -> torch.Tensor:
    """Right-pad one declared axis without changing dtype or real values."""
    axis %= value.ndim
    count = value.shape[axis]
    if count > extent:
        message = f"Cannot truncate axis {axis}: {count} > {extent}"
        raise ValueError(message)
    if count == extent:
        return value
    shape = list(value.shape)
    shape[axis] = extent - count
    return torch.cat((value, value.new_full(shape, fill)), dim=axis)


@dataclass(frozen=True)
class BucketShape:
    """Real and allocated inference dimensions."""

    tokens: int
    atoms: int
    msa: int | None
    token_bucket: int
    atom_bucket: int
    msa_bucket: int | None

    @classmethod
    def select(cls, tokens: int, atoms: int, msa: int | None) -> BucketShape:
        """Round tokens by 128 and preserve the atom/MSA allocation ladders."""
        return cls(
            tokens,
            atoms,
            msa,
            token_bucket(tokens),
            ceiling(atoms, ATOM_BUCKETS, "atoms"),
            None if msa is None else ceiling(msa, MSA_SHAPES, "msa"),
        )


def bucket_af3(example: dict) -> tuple[dict, BucketShape]:
    """Rebuild flat-atom gathers with the official layout API, trim empty MSA rows.

    The caller must request the 128-step inference TOKEN_SHAPES from the
    official featurizer first (not the engine autotune ladder).
    Dense token-by-24 geometry stays dense; only flat cross-attention uses A.
    """
    import numpy as np  # noqa: PLC0415 - optional AF3 input dependencies
    from alphafold3.model import feat_batch, features  # noqa: PLC0415

    batch = feat_batch.Batch.from_data_dict(example)
    layout = batch.convert_model_output.token_atoms_layout
    tokens = int(np.asarray(batch.token_features.mask).sum())
    atoms = int(np.count_nonzero(layout.atom_name))
    valid_rows = np.flatnonzero(np.asarray(batch.msa.mask).any(axis=-1))
    msa_depth = int(valid_rows[-1]) + 1 if len(valid_rows) else 1
    shape = BucketShape.select(tokens, atoms, msa_depth)
    if batch.token_features.mask.shape[0] != shape.token_bucket:
        message = "AF3 inputs were not featurized with 128-step inference token buckets"
        raise ValueError(message)
    result = dict(example)
    # Every removed row is empty; keep internal row ordering, holes and deletion values.
    for name in ("msa", "msa_mask", "deletion_matrix"):
        value = result[name]
        if value.shape[0] < shape.msa_bucket:
            result[name] = np.pad(
                value, ((0, shape.msa_bucket - value.shape[0]), (0, 0))
            )
        else:
            result[name] = value[: shape.msa_bucket]
    if shape.msa_bucket is None:
        message = "Dense inputs require an MSA bucket"
        raise ValueError(message)
    padding = features.PaddingShapes(
        num_tokens=shape.token_bucket,
        msa_size=shape.msa_bucket,
        num_chains=1000,
        num_templates=4,
        num_atoms=shape.atom_bucket,
    )
    gathers = features.AtomCrossAtt.compute_features(
        all_token_atoms_layout=layout,
        queries_subset_size=32,
        keys_subset_size=128,
        padding_shapes=padding,
    )
    result.update(gathers.as_data_dict())
    return result, shape
