"""Model-specific padding around the shared team-gm inference bucket policy."""

from __future__ import annotations

from team_gm.modules.bucketing import TOKEN_SHAPES, BucketShape

__all__ = ["TOKEN_SHAPES", "BucketShape", "bucket_af3"]


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
