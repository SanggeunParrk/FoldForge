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

_NAME_CLASSES = 64


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


def drop_atoms(example: dict[str, Any], names: tuple[str, ...]) -> int:
    """Remove atoms the vendor's tokenizer never creates, from STANDARD residues only.

    AF3 gives a C-terminal residue an OXT and a 5' nucleotide an OP3; several vendors
    emit neither, so the weights have never seen them. The names mean other things
    elsewhere (O3P is a phosphoserine side-chain atom), so an atomised residue, one
    real atom per token, keeps every atom. The atoms are removed from the layouts,
    not only masked: a masked hole would shift every 32-atom attention block after
    it and leave the output asking for an atom the model no longer predicts.
    """
    import dataclasses  # noqa: PLC0415

    from alphafold3.model import features  # noqa: PLC0415 - AF3 input dependency

    chars = np.asarray(example["ref_atom_name_chars"])
    ref_mask = np.array(example["ref_mask"]).astype(bool)
    real = ref_mask.sum(axis=1)

    def name(row: np.ndarray) -> str:
        # Characters are stored as ASCII minus 32 in 64 classes.
        return "".join(chr(int(c) + 32) for c in row if 0 <= c < _NAME_CLASSES).strip()

    drop = [
        (token, atom)
        for token, atom in zip(*np.nonzero(ref_mask), strict=True)
        if real[token] > 1 and name(chars[token, atom]) in names
    ]
    if not drop:
        return 0
    width = ref_mask.shape[1]
    flat = [token * width + atom for token, atom in drop]
    for key in ("ref_mask", "pred_dense_atom_mask"):
        if key in example:
            value = np.array(example[key])
            for token, atom in drop:
                value[token, atom] = 0
            example[key] = value
    beta = "token_atoms_to_pseudo_beta"
    example[f"{beta}:gather_mask"] = np.asarray(
        example[f"{beta}:gather_mask"]
    ) & ~np.isin(np.asarray(example[f"{beta}:gather_idxs"]), flat)

    layout = features._unwrap(example["token_atoms_layout"])  # noqa: SLF001
    atom_names = np.array(layout.atom_name, copy=True)
    gone = set()
    for token, atom in drop:
        gone.add(
            (
                str(layout.chain_id[token, atom]),
                int(layout.res_id[token, atom]),
                str(atom_names[token, atom]),
            )
        )
        atom_names[token, atom] = ""
    layout = dataclasses.replace(layout, atom_name=atom_names)
    example["token_atoms_layout"] = np.array(layout, object)

    queries = np.asarray(example["token_atoms_to_queries:gather_idxs"])
    keys = np.asarray(example["queries_to_keys:gather_idxs"]).shape[1]
    example.update(
        features.AtomCrossAtt.compute_features(
            all_token_atoms_layout=layout,
            queries_subset_size=queries.shape[1],
            keys_subset_size=keys,
            padding_shapes=features.PaddingShapes(
                num_tokens=layout.shape[0],
                msa_size=0,
                num_chains=0,
                num_templates=0,
                num_atoms=queries.size,
            ),
        ).as_data_dict()
    )

    output = features._unwrap(example["flat_output_layout"])  # noqa: SLF001
    keep = np.array(
        [
            (str(chain), int(residue), str(atom)) not in gone
            for chain, residue, atom in zip(
                output.chain_id, output.res_id, output.atom_name, strict=True
            )
        ]
    )
    if not keep.all():
        example["flat_output_layout"] = np.array(output[keep], object)
        structure = features._unwrap(example["empty_output_struc"])  # noqa: SLF001
        example["empty_output_struc"] = np.array(structure.filter(mask=keep), object)
    return len(drop)


def dedupe_self_msa(example: dict[str, Any]) -> None:
    """Keep the query ONCE when every live MSA row is the query.

    AF3 concatenates a paired and an unpaired MSA and both begin with the query, so
    a chain with no alignments arrives with two identical rows; these vendors emit
    depth one. It fires only on a self-MSA: with a real alignment the duplicate is
    what the vendor's own pipeline produces too.
    """
    msa, mask = np.asarray(example["msa"]), np.asarray(example["msa_mask"])
    live = np.flatnonzero(mask.any(-1))
    duplicated = live.size > 1 and all(
        np.array_equal(msa[live[0]], msa[i]) for i in live[1:]
    )
    if not duplicated:
        return
    # Only the first duplicate goes, as in the reference: depth two is what AF3 builds.
    mask = mask.copy()
    mask[live[1]] = False
    example["msa_mask"] = mask
    if "num_alignments" in example:
        count = np.asarray(example["num_alignments"])
        example["num_alignments"] = np.asarray(count - 1, dtype=count.dtype)


def key_window(example: dict[str, Any], policy: str) -> None:
    """Rewrite the atom-attention key window of a FINISHED example.

    Each block of queries takes a key window centred on it; the policies differ at
    the ends. AF3 ("slide") shifts an out-of-range window back inside the real
    atoms. "pad" clips and masks, so edge blocks see fewer neighbours at different
    key slots. "slide_qblock" slides against the atom count rounded up to a whole
    query block. Runs after bucketing, which rebuilds the window AF3's way.
    """
    query_mask = np.asarray(example["token_atoms_to_queries:gather_mask"])
    subsets, query_size = query_mask.shape
    padded = subsets * query_size
    name = "queries_to_keys:gather_idxs"
    key_size = np.asarray(example[name]).shape[1]
    starts = np.arange(subsets) * query_size + (query_size - key_size) // 2
    flat_mask = query_mask.reshape(-1)
    if policy == "slide_qblock":
        bound = -(-int(flat_mask.sum()) // query_size) * query_size
        starts = np.clip(starts, 0, max(bound - key_size, 0))
    elif policy != "pad":
        message = f"unknown key-window policy {policy!r}"
        raise ValueError(message)
    window = starts[:, None] + np.arange(key_size)[None]
    inside = (window >= 0) & (window < padded)
    window = np.clip(window, 0, padded - 1)
    keep = inside & flat_mask[window]
    example[name] = window.astype(np.asarray(example[name]).dtype)
    example["queries_to_keys:gather_mask"] = keep
    tokens = np.asarray(example["tokens_to_queries:gather_idxs"]).reshape(-1)
    target = "tokens_to_keys:gather_idxs"
    example[target] = tokens[window].astype(np.asarray(example[target]).dtype)
    token_mask = np.asarray(example["tokens_to_queries:gather_mask"]).reshape(-1)
    example["tokens_to_keys:gather_mask"] = (
        keep if policy == "pad" else token_mask[window]
    )


def apply(example: dict[str, Any], spec: DenseSpec) -> dict[str, Any]:
    """Apply ``spec``'s input conventions to one featurised example, in place."""
    if spec.centre_ref_conformers:
        centre_conformers(example)
    if spec.drop_atoms:
        drop_atoms(example, spec.drop_atoms)
    if spec.dedupe_self_msa:
        dedupe_self_msa(example)
    return example


def apply_after_bucketing(example: dict[str, Any], spec: DenseSpec) -> dict[str, Any]:
    """Conventions that act on the final atom-attention gathers."""
    if spec.atom_key_window != "slide":
        key_window(example, spec.atom_key_window)
    return example
