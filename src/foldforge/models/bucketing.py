"""Model-specific padding around the shared team-gm inference bucket policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from team_gm.modules.bucketing import TOKEN_SHAPES, BucketShape, pad_axis

__all__ = [
    "TOKEN_SHAPES",
    "BucketShape",
    "bucket_af3",
    "pad_esmfold2",
    "unpad_esmfold2",
]

if TYPE_CHECKING:
    from foldforge.models.architectures.esmfold2 import ESMFold2Output

# These names refer to model_kwargs, not an upstream feature dictionary. No
# shape-based guessing: an atom count can equal a token count or a channel width.
_ESM_TOKENS = (
    "residue_type",
    "residue_index",
    "asym_id",
    "sym_id",
    "entity_id",
    "token_index",
    "mol_type",
    "mask",
    "representative_atom_index",
    "deletion_mean",
    "lm_hidden_states",
)
_ESM_ATOMS = (
    "ref_pos",
    "ref_charge",
    "ref_element",
    "ref_atom_name_chars",
    "ref_space_uid",
    "atom_mask",
    "atom_to_token",
)
_ESM_MSA = ("msa", "msa_mask", "has_deletion", "deletion_value")


def pad_esmfold2(kwargs: dict) -> tuple[dict, BucketShape]:
    """Pad model inputs after LM lookup; masked atoms never enter token means."""
    tokens, atoms = kwargs["mask"].shape[1], kwargs["atom_mask"].shape[1]
    msa = kwargs.get("msa")
    shape = BucketShape.select(tokens, atoms, None if msa is None else msa.shape[1])
    result = dict(kwargs)
    if msa is not None and result.get("msa_mask") is None:
        result["msa_mask"] = torch.ones_like(msa, dtype=torch.bool)
    for name in _ESM_TOKENS:
        if result.get(name) is not None:
            result[name] = pad_axis(result[name], 1, shape.token_bucket)
    for name in _ESM_ATOMS:
        result[name] = pad_axis(result[name], 1, shape.atom_bucket)
    result["token_bonds"] = pad_axis(
        pad_axis(result["token_bonds"], 1, shape.token_bucket), 2, shape.token_bucket
    )
    for name in _ESM_MSA:
        if result.get(name) is not None:
            if shape.msa_bucket is None:
                message = "MSA features require an MSA bucket"
                raise ValueError(message)
            result[name] = pad_axis(
                pad_axis(result[name], 1, shape.msa_bucket), 2, shape.token_bucket
            )
    return result, shape


def unpad_esmfold2(output: ESMFold2Output, shape: BucketShape) -> ESMFold2Output:
    """Restore public token/atom dimensions, including confidence logits."""
    confidence = output.confidence
    token_names = ("plddt", "plddt_ca")
    pair_names = ("pae", "pde", "pae_logits", "pde_logits")
    atom_names = ("plddt_per_atom", "plddt_logits", "resolved_logits")
    updates = {
        name: getattr(confidence, name)[:, : shape.tokens] for name in token_names
    }
    updates.update(
        {
            name: getattr(confidence, name)[:, : shape.tokens, : shape.tokens]
            for name in pair_names
        }
    )
    updates.update(
        {name: getattr(confidence, name)[:, : shape.atoms] for name in atom_names}
    )
    return output._replace(
        coords=output.coords[:, : shape.atoms],
        distogram_logits=None
        if output.distogram_logits is None
        else output.distogram_logits[:, : shape.tokens, : shape.tokens],
        confidence=confidence._replace(**updates),
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
