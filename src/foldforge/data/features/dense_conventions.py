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

#: AF3 encodes an atom name character as its ordinal less this.
_NAME_OFFSET = 32

#: Gap class in FoldForge's polymer vocabulary.
_GAP_RESTYPE = 21


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


def widen_key_subset(example: dict[str, Any], keys: int) -> None:
    """Rebuild the atom-attention gathers with a wider key subset.

    AF3 takes 128 keys per 32-query block, which is wide enough for a window it
    only block-aligns. A family whose window is an explicit +/-N by atom rank
    needs a subset that contains it, or the mask reaches past the keys the
    gather supplies and the window is silently truncated at the block edges.
    """
    from alphafold3.model import features  # noqa: PLC0415 - AF3 input dependency

    layout = features._unwrap(example["token_atoms_layout"])  # noqa: SLF001
    queries = np.asarray(example["token_atoms_to_queries:gather_idxs"])
    example.update(
        features.AtomCrossAtt.compute_features(
            all_token_atoms_layout=layout,  # pyright: ignore[reportArgumentType]
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


def override_ref_conformers(example: dict[str, Any], conformers: dict) -> int:
    """Replace `ref_pos` with another model's reference geometry, by atom name.

    AF3 takes reference coordinates from the CCD ideal values. A family trained
    on its own idealised frame was trained on THAT one, and the feature goes
    straight into a Linear, so it is not pose-invariant. Atoms whose names the
    table does not carry keep the coordinates they already had.
    """
    from alphafold3.constants import residue_names  # noqa: PLC0415 - AF3 input

    positions = np.array(example["ref_pos"])
    mask = np.asarray(example["ref_mask"])
    chars = np.asarray(example["ref_atom_name_chars"])
    aatype = np.asarray(example["aatype"])
    codes = residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP

    def name(row: np.ndarray) -> str:
        codes = (
            chr(int(c) + _NAME_OFFSET) if 0 <= c < _NAME_CLASSES else "" for c in row
        )
        return "".join(codes).strip()

    replaced = 0
    for token in range(positions.shape[0]):
        index = int(aatype[token])
        entry = conformers.get(codes[index]) if index < len(codes) else None
        if entry is None:
            continue
        names, reference = entry
        lookup = {atom: slot for slot, atom in enumerate(names)}
        for atom in range(positions.shape[1]):
            if not mask[token, atom]:
                continue
            slot = lookup.get(name(chars[token, atom]))
            if slot is not None:
                positions[token, atom] = reference[slot]
                replaced += 1
    example["ref_pos"] = positions.astype(np.asarray(example["ref_pos"]).dtype)
    return replaced


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


#: AF3's gap class in the MSA and profile.
_GAP = 21


def nonprotein_msa_as_query(example: dict[str, Any]) -> None:
    """Fill a non-protein chain's all-gap MSA rows with its own residue types.

    AF3 writes a ligand's MSA column as the gap -- the query row included -- and
    a nucleic chain's column as the gap in every protein alignment row. The
    Protenix lineage fills any row where a non-protein chain is entirely gap
    with that chain's query ("forward compatibility patch for non-protein
    entities"), and its query row for a ligand is the ligand's residue type,
    UNK, so no non-protein token ever reads as gap. The profile follows: a
    non-protein token whose profile is pure gap takes its residue type. Missed,
    a zinc ion's profile was the gap where the release's is UNK, and it moved
    every token's initial single through the outer sum.
    """
    msa = np.array(example["msa"])
    live = np.asarray(example["msa_mask"]).any(-1)
    aatype = np.asarray(example["aatype"])
    real = np.asarray(example["seq_mask"]).astype(bool)
    other = real & ~np.asarray(example["is_protein"]).astype(bool)
    asym = np.asarray(example["asym_id"])
    for chain in np.unique(asym[other]):
        cols = np.flatnonzero(other & (asym == chain))
        rows = np.flatnonzero(live & np.all(msa[:, cols] == _GAP, axis=1))
        msa[np.ix_(rows, cols)] = aatype[cols]
    example["msa"] = msa.astype(np.asarray(example["msa"]).dtype)
    profile = np.array(example["profile"])
    gap_only = profile[:, _GAP] >= 1.0
    fix = np.flatnonzero(other & gap_only)
    profile[fix] = 0.0
    profile[fix, aatype[fix]] = 1.0
    example["profile"] = profile.astype(np.asarray(example["profile"]).dtype)


def empty_template_gap(example: dict[str, Any], slots: str) -> None:
    """Fill an ABSENT template's restype with the gap class rather than zero.

    Both pipelines pad the template axis and mask every atom, so the distogram,
    the unit vector and both masks come out zero either way. The RESTYPE one-hot
    does not: AF3 fills it with class zero where these vendors fill it with the
    gap. Applied only when there is no real template anywhere, which is the case
    they build that way; a batch carrying one is left alone.

    ``slots`` is the vendor's own padding: "first" gives one gap template and
    zero-pads the rest, "all" makes every slot a gap template.
    """
    aatype = example.get("template_aatype")
    mask = example.get("template_atom_mask")
    if aatype is None or mask is None:
        return
    aatype, mask = np.asarray(aatype), np.asarray(mask)
    if not aatype.shape[0] or mask.any():
        return
    aatype = aatype.copy()
    if slots == "all":
        aatype[:] = _GAP_RESTYPE
    else:
        aatype[0] = _GAP_RESTYPE
    example["template_aatype"] = aatype


def apply(example: dict[str, Any], spec: DenseSpec) -> dict[str, Any]:
    """Apply ``spec``'s input conventions to one featurised example, in place."""
    if spec.ref_conformers != "af3":
        from foldforge.data.constants import residue_geometry  # noqa: PLC0415

        override_ref_conformers(
            example, residue_geometry.as_conformers(spec.ref_conformers)
        )
    if spec.centre_ref_conformers:
        centre_conformers(example)
    if spec.drop_atoms:
        drop_atoms(example, spec.drop_atoms)
    if spec.empty_template_gap is not None:
        empty_template_gap(example, spec.empty_template_gap)
    if spec.nonprotein_msa_as_query:
        nonprotein_msa_as_query(example)
    if spec.dedupe_self_msa:
        dedupe_self_msa(example)
    return example


def apply_after_bucketing(example: dict[str, Any], spec: DenseSpec) -> dict[str, Any]:
    """Conventions that act on the final atom-attention gathers."""
    if spec.atom_keys_subset != 128:  # noqa: PLR2004 - AF3's own key subset size
        widen_key_subset(example, spec.atom_keys_subset)
    return example
