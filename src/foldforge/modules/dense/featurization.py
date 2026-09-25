# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


from collections.abc import Sequence

import torch
import torch.nn.functional as F

from foldforge.data.constants import residue_names
from foldforge.data.features import dense as features
from foldforge.data.features import dense_batch as feat_batch
from foldforge.modules.dense import utils


def gumbel_noise(
    shape: Sequence[int],
    device: torch.device,
    eps: float = 1e-6,
    generator=None,
) -> torch.Tensor:
    """Generate Gumbel Noise of given Shape.

    This generates samples from Gumbel(0, 1).

    Args:
        shape: Shape of noise to return.

    Returns:
        Gumbel noise of given shape.
    """
    uniform_noise = torch.rand(
        shape, dtype=torch.float32, device=device, generator=generator
    )
    gumbel = -torch.log(-torch.log(uniform_noise + eps) + eps)
    return gumbel


def gumbel_argsort_sample_idx(
    logits: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Sample with replacement from a distribution given by 'logits'.

    This uses Gumbel trick to implement the sampling an efficient manner. For a
    distribution over k items this samples k times without replacement, so this
    is effectively sampling a random permutation with probabilities over the
    permutations derived from the logprobs.

    Args:
      key: prng key
      logits: logarithm of probabilities to sample from, probabilities can be
        unnormalized.

    Returns:
      Sample from logprobs in one-hot form.
    """
    z = gumbel_noise(logits.shape, device=logits.device, generator=generator)
    return torch.argsort(logits + z, dim=-1, descending=True)


#: Chains the vendor's paired-row feature distinguishes; more than any complex
#: this runs on.
_PAIRED_CHAIN_BOUND = 16
#: Database labels of the vendor's MSA source feature: the query row, then the
#: rest. A row from a third source is not distinguishable to us.
_MSA_SOURCE_CLASSES = 6
_MSA_SOURCE_QUERY = 4
_MSA_SOURCE_OTHER = 2


def create_msa_feat(
    msa: features.MSA,
    layout: str = "af3",
    is_ligand: torch.Tensor | None = None,
    asym_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create and concatenate MSA features in the layout the weights expect."""
    classes = residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP + 1
    rows = msa.rows.to(dtype=torch.int64)
    if layout == "chai1" and is_ligand is not None:
        # A non-polymer token has no MSA, and the vendor does not mark it as a
        # gap: its QUERY row carries the unknown residue and every other row the
        # mask class, where AF3 puts a gap on all of them.
        unknown = residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP.index(
            residue_names.UNK
        )
        mask_class = residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
        ligand = is_ligand.to(torch.bool)[None, :]
        query = torch.arange(rows.shape[0], device=rows.device)[:, None] == 0
        rows = torch.where(ligand, torch.where(query, unknown, mask_class), rows)
    msa_1hot = torch.nn.functional.one_hot(rows, classes)
    deletion_matrix = msa.deletion_matrix
    has_deletion = torch.clip(deletion_matrix, 0.0, 1.0)[..., None]
    deletion_value = (torch.arctan(deletion_matrix / 3.0) * (2.0 / torch.pi))[..., None]

    if layout != "chai1":
        return torch.concatenate([msa_1hot, has_deletion, deletion_value], dim=-1)

    # Alphabetical feature order, and deletion VALUE precedes HAS-deletion --
    # the opposite of AF3's.
    dtype = msa_1hot.dtype
    depth, tokens = msa_1hot.shape[:2]
    if asym_id is not None:
        # A row is paired exactly when it covers tokens of more than one chain,
        # which is what pairing means. Zero throughout for a monomer.
        chains = torch.nn.functional.one_hot(
            asym_id.to(torch.int64).clamp(0, _PAIRED_CHAIN_BOUND - 1),
            _PAIRED_CHAIN_BOUND,
        ).to(torch.float32)
        covers = msa.mask.to(torch.float32) @ chains > 0
        paired = (covers.sum(-1) > 1).to(dtype)
        is_paired = paired[:, None, None].expand(depth, tokens, 1)
    else:
        is_paired = msa_1hot.new_zeros((depth, tokens, 1))
    source = torch.where(
        torch.arange(depth, device=rows.device) == 0,
        _MSA_SOURCE_QUERY,
        _MSA_SOURCE_OTHER,
    )
    source = torch.nn.functional.one_hot(source, _MSA_SOURCE_CLASSES).to(dtype)
    source = source[:, None, :].expand(depth, tokens, _MSA_SOURCE_CLASSES)
    return torch.concatenate(
        [is_paired, source, deletion_value, has_deletion, msa_1hot], dim=-1
    )


def truncate_msa_batch(msa: features.MSA, num_msa: int) -> features.MSA:
    """Compute truncate msa batch."""
    indices = torch.arange(num_msa, device=msa.rows.device, dtype=torch.int64)
    return msa.index_msa_rows(indices)


def create_target_feat(
    batch: feat_batch.Batch,
    append_per_atom_features: bool,
) -> torch.Tensor:
    """Make target feat."""
    token_features = batch.token_features
    target_features = []

    target_features.append(
        torch.nn.functional.one_hot(
            token_features.aatype.to(dtype=torch.int64),
            residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP,
        )
    )
    target_features.append(batch.msa.profile)
    target_features.append(batch.msa.deletion_mean[..., None])

    # Reference structure features
    if append_per_atom_features:
        ref_mask = batch.ref_structure.mask
        element_feat = torch.nn.functional.one_hot(batch.ref_structure.element, 128)
        element_feat = utils.mask_mean(
            mask=ref_mask[..., None], value=element_feat, dim=-2, eps=1e-6
        )
        target_features.append(element_feat)
        pos_feat = batch.ref_structure.positions
        pos_feat = pos_feat.reshape([pos_feat.shape[0], -1])
        target_features.append(pos_feat)
        target_features.append(ref_mask)

    return torch.concatenate(target_features, dim=-1)


def create_relative_encoding(
    seq_features: features.TokenFeatures,
    max_relative_idx: int,
    max_relative_chain: int,
    *,
    chain_bucket_on_same_chain: bool = False,
) -> torch.Tensor:
    """Add relative position encodings."""
    rel_feats = []
    token_index = seq_features.token_index
    residue_index = seq_features.residue_index
    asym_id = seq_features.asym_id
    entity_id = seq_features.entity_id
    sym_id = seq_features.sym_id

    left_asym_id = asym_id[:, None]
    right_asym_id = asym_id[None, :]

    left_residue_index = residue_index[:, None]
    right_residue_index = residue_index[None, :]

    left_token_index = token_index[:, None]
    right_token_index = token_index[None, :]

    left_entity_id = entity_id[:, None]
    right_entity_id = entity_id[None, :]

    left_sym_id = sym_id[:, None]
    right_sym_id = sym_id[None, :]

    # Embed relative positions using a one-hot embedding of distance along chain
    offset = left_residue_index - right_residue_index
    clipped_offset = torch.clip(
        offset + max_relative_idx, min=0, max=2 * max_relative_idx
    )
    asym_id_same = left_asym_id == right_asym_id
    final_offset = torch.where(
        asym_id_same,
        clipped_offset,
        (2 * max_relative_idx + 1) * torch.ones_like(clipped_offset),
    )
    rel_pos = torch.nn.functional.one_hot(
        final_offset.to(dtype=torch.int64), 2 * max_relative_idx + 2
    )
    rel_feats.append(rel_pos)

    # Embed relative token index as a one-hot embedding of distance along residue
    token_offset = left_token_index - right_token_index
    clipped_token_offset = torch.clip(
        token_offset + max_relative_idx, min=0, max=2 * max_relative_idx
    )
    residue_same = (left_asym_id == right_asym_id) & (
        left_residue_index == right_residue_index
    )
    final_token_offset = torch.where(
        residue_same,
        clipped_token_offset,
        (2 * max_relative_idx + 1) * torch.ones_like(clipped_token_offset),
    )
    rel_token = torch.nn.functional.one_hot(
        final_token_offset.to(dtype=torch.int64), 2 * max_relative_idx + 2
    )
    rel_feats.append(rel_token)

    # Embed same entity ID
    entity_id_same = left_entity_id == right_entity_id
    rel_feats.append(entity_id_same.to(dtype=rel_pos.dtype)[..., None])

    # Embed relative chain ID inside each symmetry class
    rel_sym_id = left_sym_id - right_sym_id

    max_rel_chain = max_relative_chain

    clipped_rel_chain = torch.clip(
        rel_sym_id + max_rel_chain, min=0, max=2 * max_rel_chain
    )

    pad_class = (2 * max_rel_chain + 1) * torch.ones_like(clipped_rel_chain)
    if chain_bucket_on_same_chain:
        # Keyed on same-CHAIN, sending the MATCH to the pad class, where AF3
        # keys on same-ENTITY and sends the MISMATCH there. On a MONOMER this
        # flips every pair: same-chain is universally true, so one convention
        # makes the whole one-hot the pad class and the other the zero-offset
        # class -- a different constant column into a trained projection.
        same_chain = left_asym_id == right_asym_id
        final_rel_chain = torch.where(same_chain, pad_class, clipped_rel_chain)
    else:
        final_rel_chain = torch.where(entity_id_same, clipped_rel_chain, pad_class)
    rel_chain = torch.nn.functional.one_hot(
        final_rel_chain.to(dtype=torch.int64), 2 * max_relative_chain + 2
    )

    rel_feats.append(rel_chain)

    return torch.concatenate(rel_feats, dim=-1)


def shuffle_msa(msa: features.MSA) -> features.MSA:
    """Shuffle MSA randomly, return batch with shuffled MSA.

    Args:
      key: rng key for random number generation.
      msa: MSA object to sample msa from.

    Returns:
      Protein with sampled msa.
    """
    # Sample uniformly among sequences with at least one non-masked position.
    logits = (torch.clip(torch.sum(msa.mask, dim=-1), 0.0, 1.0) - 1.0) * 1e6
    index_order = gumbel_argsort_sample_idx(logits)

    return msa.index_msa_rows(index_order)


def subsample_msa_keep_query(msa: features.MSA, num_msa: int) -> features.MSA:
    """Subsample to ``num_msa`` rows KEEPING THE QUERY at row 0, in a3m order.

    AF3's shuffle-then-truncate is a different function. Its gumbel shuffle
    ranks EVERY row including row 0, so the query lands somewhere random and the
    truncation drops it outright with probability 1 - num_msa/depth -- about
    half the time on a 2145-row alignment at depth 1024. A trunk handed an
    alignment with no query in it folds WORSE the moment it is given a real one,
    which is how this was found.

    Depth one is the case that hides it: a self-MSA has nothing to shuffle and
    nothing to drop, so every gate that folds from a single sequence agrees.
    """
    logits = (torch.clip(torch.sum(msa.mask, dim=-1), 0.0, 1.0) - 1.0) * 1e6
    # Row 0 is taken explicitly, so it must not also be drawn into the tail:
    # the query would occupy two slots and one real sequence would be lost.
    # Pushing it below the padded rows is enough, since padding sits at -1e6
    # and only the first num_msa - 1 draws are used.
    logits = logits.clone()
    logits[0] -= 2e6
    order = gumbel_argsort_sample_idx(logits)
    index = torch.concatenate([order.new_zeros(1), order[: num_msa - 1]])
    # Sorted, as the vendor sorts: an a3m is ordered by similarity to the
    # query, so its row order is information and a permutation of it is a
    # different input to any module that is not row-equivariant.
    return msa.index_msa_rows(torch.sort(index).values)


def chai_relative_encoding(
    token_features: features.TokenFeatures, dtype: torch.dtype
) -> torch.Tensor:
    """Two separation one-hots, the vendor's own, in place of AF3's four blocks.

    Sequence separation is a searchsorted over bins -32..32, which collapses to
    ``clip(rel + 33, 0, 65)`` -- so +32 and anything beyond SHARE the top class --
    with a class of its own for an inter-chain pair. Token separation is over the
    TOKEN index rather than the residue index, and takes that same out-of-range
    class wherever the pair is not in one residue of one chain, so on a plain
    protein everything off the diagonal lands there.

    Then the chain and entity relations, which only a complex exercises:
    AF-Multimer's relative chain (six classes: the copy offset within one entity,
    clipped to +-2, and one class for a pair across entities) and the sign of
    the entity offset (three). They were once folded into the bias at their
    single-chain values, so every pair across chains read as one inside a
    chain -- exact on one protein chain, wrong on every complex.

    The rest of the vendor's token-pair stream (restraints, docking) is constant
    for a fold and the converter folds it into this projection's bias.
    """
    classes = 67
    outside = classes - 1
    residue = token_features.residue_index.to(torch.int64)
    token = token_features.token_index.to(torch.int64)
    same_chain = token_features.asym_id[:, None] == token_features.asym_id[None, :]

    separation = torch.clamp(residue[:, None] - residue[None, :] + 33, 0, 65)
    separation = torch.where(same_chain, separation, outside)

    same_residue = (residue[:, None] == residue[None, :]) & same_chain
    token_separation = torch.clamp(token[:, None] - token[None, :] + 32, 0, 65)
    token_separation = torch.where(same_residue, token_separation, outside)

    entity = token_features.entity_id.to(torch.int64)
    same_entity = entity[:, None] == entity[None, :]
    sym = token_features.sym_id.to(torch.int64)
    chain_offset = torch.clamp(sym[:, None] - sym[None, :] + 2, 0, 4)
    chain_offset = torch.where(same_entity, chain_offset, 5)
    entity_order = torch.clamp(entity[:, None] - entity[None, :] + 1, 0, 2)

    return torch.cat(
        [
            F.one_hot(separation, classes),
            F.one_hot(token_separation, classes),
            F.one_hot(chain_offset, 6),
            F.one_hot(entity_order, 3),
        ],
        dim=-1,
    ).to(dtype)


#: AF3's polymer classes: 20 residues, UNK, the gap, then the nucleic acids.
_AF3_POLYMER_CLASSES = 31
#: ESMFold2's: slot 0 unused, 1 the gap, 2..22 the residues and UNK, 23..31 the
#: nucleic acids, 32 the unknown deoxyribonucleotide AF3's block does not carry.
_ESM_POLYMER_CLASSES = 33


def widen_to_esm_classes(features: torch.Tensor) -> torch.Tensor:
    """Re-lay a 31-class restype and profile pair out over ESMFold2's 33.

    The widening is a PERMUTATION, not a shift: ESMFold2 puts the gap at class
    one, BELOW the residues, where AF3 puts it at 21 between UNK and the nucleic
    acids. Padding with two leading zeros instead puts AF3's gap column on the
    first NUCLEIC class and shifts every nucleic class down one -- which no
    single-sequence protein fold can see, since both the gap column and the
    nucleic columns are then identically zero.

    ``features`` is (..., 447): restype, profile, deletion mean, token act.
    """
    n = _AF3_POLYMER_CLASSES
    zero = torch.zeros_like(features[..., :1])

    def widen(block: torch.Tensor) -> list[torch.Tensor]:
        return [zero, block[..., 21:22], block[..., :21], block[..., 22:n], zero]

    return torch.concatenate(
        [
            *widen(features[..., :n]),
            *widen(features[..., n : 2 * n]),
            features[..., 2 * n :],
        ],
        dim=-1,
    )
