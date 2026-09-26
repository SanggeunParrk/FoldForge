# AF3's featurisation is heavy and only one family reaches this module, so its
# imports stay inside the functions that need them.
# ruff: noqa: PLC0415
"""Re-tokenise an AF3 residue-level example onto OpenDDE's structural tokens.

OpenDDE folds on structural tokens, not residues: each standard protein residue
becomes a BACKBONE token (N, CA, C, O, OXT; representative CA) and a SIDECHAIN
token (the rest; representative CB), and each standard nucleotide becomes a
backbone and a base token. Glycine, ligands, modified residues and anything with
an empty split stay one token, and a non-standard residue becomes one token per
atom.

The trunk still runs on residues. Only the diffusion and the confidence heads see
the expanded set, so this module produces a SECOND feature set -- the same AF3
dataclasses over the structural layout -- alongside the residue one, plus the
bookkeeping the expander needs to carry the trunk's output across.

Every array here is numpy and per-example: this is featurisation, and it runs
once per input, before anything reaches the graph.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

#: Role ids, from opendde's tokenizer: an atomised token is 0, then backbone and
#: child alternate per polymer class.
ROLE_ATOM = 0
ROLE_PROTEIN_BB = 1
ROLE_PROTEIN_SC = 2
_ROLE = {"protein": (1, 2), "dna": (3, 4), "rna": (5, 6)}

#: No twin subtoken: a residue that did not split.
NO_TWIN = -1

_PROTEIN_BACKBONE = frozenset(["N", "CA", "C", "O", "OXT"])
_NUCLEIC_BACKBONE = frozenset(
    [
        "P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P",
        "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
        "O5*", "C5*", "C4*", "O4*", "C3*", "O3*", "C2*", "O2*", "C1*",
        "O5T", "O3T",
    ]
)  # fmt: skip
_BACKBONE = {
    "protein": _PROTEIN_BACKBONE,
    "dna": _NUCLEIC_BACKBONE,
    "rna": _NUCLEIC_BACKBONE,
}

#: Only a STANDARD residue splits; everything else is atomised one token per atom.
_STANDARD = {
    "protein": frozenset(
        [
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
            "TYR", "VAL", "UNK",
        ]
    ),
    "rna": frozenset(["A", "G", "C", "U", "N"]),
    "dna": frozenset(["DA", "DG", "DC", "DT", "DN"]),
}  # fmt: skip

#: Representative-atom preference per (polymer class, is backbone).
_CENTRE = {
    ("protein", True): ["CA", "N", "C"],
    ("protein", False): ["CB"],
    ("dna", True): ["C4'", "C4*", "C1'", "C1*"],
    ("rna", True): ["C4'", "C4*", "C1'", "C1*"],
}

#: A nucleic BASE token's representative depends on the ring system, which one
#: preference list cannot express. Purines carry an N1 too, in the six-membered
#: ring, so a plain ``["N1", "N9", ...]`` silently picks N1 for every A/G/DA/DG
#: where OpenDDE picks N9 -- the wrong atom on about half of all nucleotides,
#: and invisible to protein.
_PURINE = frozenset(["A", "G", "DA", "DG"])
_PYRIMIDINE = frozenset(["C", "U", "DC", "DT"])

_LAYOUT_FIELDS = (
    "atom_name",
    "res_id",
    "chain_id",
    "atom_element",
    "res_name",
    "chain_type",
)


def _base_centre(res_name: str) -> list[str]:
    """OpenDDE's per-residue base-centre preference list."""
    if res_name in _PURINE:
        return ["N9", "C4", "C8", "N7", "C5"]
    if res_name in _PYRIMIDINE:
        return ["N1", "C2", "C6", "C5", "C4"]
    return ["C1'", "C1*", "N9", "N1"]


def _mol_type(chain_type: str | None) -> str:
    """AF3's chain-type string to the polymer class OpenDDE tokenises by."""
    text = (chain_type or "").lower()
    if "polypeptide" in text:
        return "protein"
    if "deoxyribo" in text:
        return "dna"
    if "ribonucleotide" in text:
        return "rna"
    return "ligand"


def _choose(idxs: Sequence[int], names: np.ndarray, priority: Sequence[str]) -> int:
    """Return the first atom of ``idxs`` matching ``priority``, else the first atom."""
    for wanted in priority:
        for k in idxs:
            if names[k] == wanted:
                return k
    return idxs[0]


def build_structural_layout(  # noqa: C901, PLR0912, PLR0915 - one pass over a token array, and splitting it up would hide the tokenizer
    all_tokens: Any, all_token_atoms_layout: Any
) -> dict:
    """Residue-level AtomLayouts to structural-token ones, plus the bookkeeping.

    ``all_tokens`` is (n_res,), one representative atom per residue token, and
    ``all_token_atoms_layout`` is (n_res, max_atoms). Returns the same pair over
    structural tokens together with the per-token parent, role, twin and
    chain-adjacent parents the expander reads, and the two gathers that bring a
    structural prediction back to the residue layout.
    """
    from alphafold3.model.atom_layout import atom_layout

    n_res, max_atoms = all_token_atoms_layout.shape
    layout = all_token_atoms_layout

    rows: list[dict[str, np.ndarray]] = []
    representatives: list[tuple[int, int]] = []
    sources: list[tuple[int, list[int]]] = []
    parent: list[int] = []
    role: list[int] = []

    for r in range(n_res):
        names = layout.atom_name[r]
        valid_idx = np.where(np.array([bool(n) for n in names]))[0]
        if valid_idx.size == 0:
            continue
        chain_type = layout.chain_type
        res_names = layout.res_name
        kind = _mol_type(
            chain_type[r][valid_idx[0]] if chain_type is not None else None
        )
        res_name = res_names[r][valid_idx[0]] if res_names is not None else ""

        if kind in _ROLE and res_name in _STANDARD[kind] and valid_idx.size > 1:
            backbone_set = _BACKBONE[kind]
            bb_role, child_role = _ROLE[kind]
            backbone = [k for k in valid_idx if names[k] in backbone_set]
            child = [k for k in valid_idx if names[k] not in backbone_set]
            if not child or not backbone:
                # Glycine, or any residue whose split leaves one side empty.
                groups = [(bb_role, list(valid_idx), _CENTRE[(kind, True)])]
            else:
                child_centre = (
                    _base_centre(res_name)
                    if kind in ("rna", "dna")
                    else _CENTRE[(kind, False)]
                )
                groups = [
                    (bb_role, backbone, _CENTRE[(kind, True)]),
                    (child_role, child, child_centre),
                ]
        else:
            groups = [(ROLE_ATOM, [k], [names[k]]) for k in valid_idx]

        for role_id, idxs, priority in groups:
            row = {
                field: (
                    np.zeros(max_atoms, dtype=np.int64)
                    if field == "res_id"
                    else np.full(max_atoms, "", dtype=object)
                )
                for field in _LAYOUT_FIELDS
            }
            for slot, k in enumerate(idxs):
                for field in _LAYOUT_FIELDS:
                    source = getattr(layout, field)
                    if source is not None:
                        row[field][slot] = source[r][k]
            rows.append(row)
            sources.append((r, [int(k) for k in idxs]))
            representatives.append(
                (r, _choose([int(k) for k in idxs], names, priority))
            )
            parent.append(r)
            role.append(role_id)

    n_struct = len(parent)
    parent_idx = np.array(parent, dtype=np.int64)
    role_id = np.array(role, dtype=np.int64)

    struct_atoms = atom_layout.AtomLayout(
        **{
            field: np.stack([row[field] for row in rows], axis=0)
            for field in _LAYOUT_FIELDS
        }
    )
    struct_tokens = atom_layout.AtomLayout(
        **{
            field: np.array(
                [
                    getattr(layout, field)[r][k]
                    if getattr(layout, field) is not None
                    else ("" if field != "res_id" else 0)
                    for (r, k) in representatives
                ],
                dtype=np.int64 if field == "res_id" else object,
            )
            for field in _LAYOUT_FIELDS
        }
    )

    # The twin is the other subtoken of the same residue, which exists only for
    # a residue that actually split in two.
    twin = np.full(n_struct, NO_TWIN, dtype=np.int64)
    for r in range(n_res):
        members = np.where(parent_idx == r)[0]
        if members.size == 2:  # noqa: PLR2004 - a residue splits in two or not at all
            twin[members[0]] = members[1]
            twin[members[1]] = members[0]

    # Chain-adjacent parents: the previous and next residue of the SAME chain.
    res_chain = np.asarray(all_tokens.chain_id, dtype=object)
    prev_parent = np.full(n_struct, -1, dtype=np.int64)
    next_parent = np.full(n_struct, -1, dtype=np.int64)
    for i in range(n_struct):
        r = int(parent_idx[i])
        if r - 1 >= 0 and res_chain[r - 1] == res_chain[r]:
            prev_parent[i] = r - 1
        if r + 1 < n_res and res_chain[r + 1] == res_chain[r]:
            next_parent[i] = r + 1

    # residue_atom_gather[r, j] is the flat structural slot (t * max_atoms + s)
    # holding residue r's atom j, so a structural prediction scatters back onto
    # the residue layout the mmCIF writer already understands. -1 is padding.
    residue_atom_gather = np.full((n_res, max_atoms), -1, dtype=np.int64)
    for t, (r, idxs) in enumerate(sources):
        for s, j in enumerate(idxs):
            residue_atom_gather[r, j] = t * max_atoms + s

    # residue_rep_token[r] stands for residue r where a PER-TOKEN quantity has to
    # come back to the residue layout (PAE, PDE, and the pTM they feed). That is
    # a CHOICE, not a reconstruction: a residue has several subtokens, and we
    # take its first, the one carrying its backbone role.
    residue_rep_token = np.zeros((n_res,), dtype=np.int64)
    seen = np.zeros((n_res,), dtype=bool)
    for t, r in enumerate(parent_idx):
        if 0 <= r < n_res and not seen[r]:
            residue_rep_token[r] = t
            seen[r] = True

    return {
        "all_tokens": struct_tokens,
        "all_token_atoms_layout": struct_atoms,
        "parent_residue_idx": parent_idx,
        "subtoken_role_id": role_id,
        "twin_token_idx": twin,
        "prev_parent_residue_idx": prev_parent,
        "next_parent_residue_idx": next_parent,
        "residue_atom_gather": residue_atom_gather,
        "residue_rep_token": residue_rep_token,
        "n_struct": n_struct,
    }


def _relabel_atom_cross_att(
    residue: Any,
    residue_atom_gather: np.ndarray,
    *,
    struct_num_tokens: int,
    max_atoms: int,
    num_residue_tokens: int,
) -> Any:
    """Build the structural AtomCrossAtt by relabelling the residue-level one.

    THE ATOM AXIS IS THE SAME. AF3 cuts its query and key windows on the flat
    atom list, which it gets by flattening the (token, slot) layout ROW-MAJOR.
    OpenDDE instead keeps the structure's own atom order and has each token index
    into it -- and that order is exactly the row-major flatten of the RESIDUE
    layout. The two differ under a structural flatten, because a token's atoms
    stop being contiguous in residue order at a non-glycine chain terminus: OXT
    is in the BACKBONE set, so the sidechain sits between O and OXT.

        residue-major (OpenDDE's)  N CA C O CB SG OXT
        structural-major           N CA C O OXT | CB SG

    No ordering of the two tokens fixes that, since the backbone set interleaves.
    But because the axis OpenDDE wants is the residue layout's own, the windows
    AF3 builds for the residue layout are already the right ones -- every atom
    sits at the same place on the axis, and the windows are cut on the axis.
    Only the gathers that name a TOKEN change, and `residue_atom_gather` is the
    relabelling: it maps each residue (token, slot) to the structural one holding
    the same atom.
    """
    from alphafold3.model import features
    from alphafold3.model.atom_layout import atom_layout

    # Pad the relabelling to the residue TOKEN bucket the gathers were built at:
    # a padded row holds no atom and is masked out everywhere below.
    padded_gather = np.full((num_residue_tokens, max_atoms), -1, dtype=np.int64)
    rows = min(num_residue_tokens, residue_atom_gather.shape[0])
    padded_gather[:rows] = residue_atom_gather[:rows]
    flat_gather = padded_gather.reshape(-1)

    def to_struct(info: atom_layout.GatherInfo) -> tuple[np.ndarray, np.ndarray]:
        """Residue flat slots named by ``info`` to structural flat slots."""
        safe = np.clip(info.gather_idxs, 0, flat_gather.shape[0] - 1)
        struct = flat_gather[safe]
        mask = np.asarray(info.gather_mask) & (struct >= 0)
        return np.where(mask, struct, 0), mask

    queries_idxs, queries_mask = to_struct(residue.token_atoms_to_queries)
    token_atoms_to_queries = atom_layout.GatherInfo(
        gather_idxs=queries_idxs,
        gather_mask=queries_mask,
        input_shape=np.array((struct_num_tokens, max_atoms)),
    )

    # The keys are a gather OF the queries, so the same relabelling reaches them
    # through `queries_to_keys` rather than being recomputed.
    queries_to_keys = residue.queries_to_keys
    flat_queries = queries_idxs.reshape(-1)
    flat_queries_mask = queries_mask.reshape(-1)
    key_safe = np.clip(queries_to_keys.gather_idxs, 0, flat_queries.shape[0] - 1)
    keys_idxs = flat_queries[key_safe]
    keys_mask = np.asarray(queries_to_keys.gather_mask) & flat_queries_mask[key_safe]

    tokens_to_queries = atom_layout.GatherInfo(
        gather_idxs=np.where(queries_mask, queries_idxs // max_atoms, 0),
        gather_mask=queries_mask,
        input_shape=np.array((struct_num_tokens,)),
    )
    tokens_to_keys = atom_layout.GatherInfo(
        gather_idxs=np.where(keys_mask, keys_idxs // max_atoms, 0),
        gather_mask=keys_mask,
        input_shape=np.array((struct_num_tokens,)),
    )

    # queries_to_token_atoms points the other way, so it is a SCATTER: each
    # residue slot hands its query position to the structural slot holding it.
    source = residue.queries_to_token_atoms
    scatter_idxs = np.zeros((struct_num_tokens, max_atoms), dtype=np.int64)
    scatter_mask = np.zeros((struct_num_tokens, max_atoms), dtype=bool)
    valid = np.asarray(source.gather_mask) & (padded_gather >= 0)
    token, slot = np.divmod(padded_gather[valid], max_atoms)
    scatter_idxs[token, slot] = np.asarray(source.gather_idxs)[valid]
    scatter_mask[token, slot] = True
    queries_to_token_atoms = atom_layout.GatherInfo(
        gather_idxs=scatter_idxs,
        gather_mask=scatter_mask,
        input_shape=np.asarray(source.input_shape),
    )

    # AF3's feature dataclasses reach us without type information, so the
    # checker cannot see these field names.
    return features.AtomCrossAtt(
        token_atoms_to_queries=token_atoms_to_queries,  # pyright: ignore[reportCallIssue]
        tokens_to_queries=tokens_to_queries,  # pyright: ignore[reportCallIssue]
        tokens_to_keys=tokens_to_keys,  # pyright: ignore[reportCallIssue]
        queries_to_keys=queries_to_keys,  # pyright: ignore[reportCallIssue]
        queries_to_token_atoms=queries_to_token_atoms,  # pyright: ignore[reportCallIssue]
    )


def build_structural_batch(
    capture: dict[str, Any],
    residue_atom_cross_att: Any,
    num_residue_tokens: int,
    *,
    struct_num_tokens: int | None = None,
    pad_multiple: int = 32,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Build the diffusion-facing feature set over structural tokens.

    ``capture`` carries the residue-level AtomLayouts and the builder arguments
    taken while the example was featurised -- see `capture_layouts`. Everything
    but the atom cross attention is built by AF3's own feature builders, driven
    on the structural layout; the cross attention is relabelled instead, because
    its windows are cut on an atom axis the structural layout does not change.

    ``residue_atom_cross_att`` is the example's CURRENT residue cross attention,
    not the captured one: bucketing rebuilds those gathers on a different atom
    bucket, and the relabelling has to follow the axis actually in play.
    """
    from alphafold3.model import features
    from alphafold3.model.atom_layout import atom_layout

    info = build_structural_layout(
        capture["all_tokens"], capture["all_token_atoms_layout"]
    )
    struct_tokens = info["all_tokens"]
    struct_atoms = info["all_token_atoms_layout"]
    n_struct = info["n_struct"]
    max_atoms = capture["all_token_atoms_layout"].shape[1]

    if struct_num_tokens is None:
        struct_num_tokens = int(np.ceil(n_struct / pad_multiple) * pad_multiple)
    if struct_num_tokens < n_struct:
        message = f"{n_struct} structural tokens do not fit {struct_num_tokens}"
        raise ValueError(message)
    padding = dataclasses.replace(
        capture["padding_shapes"],
        num_tokens=struct_num_tokens,
        # The padded atom count is the query layout's own size: bucketing may
        # have rebuilt it at a different one than the capture saw.
        num_atoms=int(
            np.asarray(residue_atom_cross_att.token_atoms_to_queries.gather_idxs).size
        ),
    )

    atom_cross_att = _relabel_atom_cross_att(
        residue_atom_cross_att,
        info["residue_atom_gather"],
        struct_num_tokens=struct_num_tokens,
        max_atoms=max_atoms,
        num_residue_tokens=num_residue_tokens,
    )
    token_features = features.TokenFeatures.compute_features(
        all_tokens=struct_tokens, padding_shapes=padding
    )
    predicted_structure_info = features.PredictedStructureInfo.compute_features(
        all_tokens=struct_tokens,
        all_token_atoms_layout=struct_atoms,
        padding_shapes=padding,
    )
    # AF3 re-derives the pseudo-beta atom from a token's own atoms and res_name:
    # CB or CA for protein, and for a nucleic token the ring atoms C4 or C2. A
    # structural token holds only the backbone OR only the base, so a nucleic
    # BACKBONE token contains neither and falls through to its first atom, P.
    # We already chose the right representative when the tokens were built, and
    # `struct_tokens` IS that one-atom-per-token layout, so gather from it.
    pseudo_beta_info = features.PseudoBetaInfo(
        token_atoms_to_pseudo_beta=atom_layout.compute_gather_idxs(  # pyright: ignore[reportCallIssue]
            source_layout=struct_atoms,
            target_layout=struct_tokens.copy_and_pad_to((struct_num_tokens,)),
        )
    )
    ref_structure, _ = features.RefStructure.compute_features(
        all_token_atoms_layout=struct_atoms,
        ccd=capture["ccd"],
        padding_shapes=padding,
        chemical_components_data=capture["chemical_components_data"],
        random_state=np.random.RandomState(random_seed),
        ref_max_modified_date=capture["ref_max_modified_date"],
        conformer_max_iterations=capture["conformer_max_iterations"],
    )
    frames = features.Frames.compute_features(
        all_tokens=struct_tokens,
        all_token_atoms_layout=struct_atoms,
        ref_structure=ref_structure,
        padding_shapes=padding,
    )

    def pad_tokens(array: np.ndarray, fill: int = 0) -> np.ndarray:
        out = np.full((struct_num_tokens, *array.shape[1:]), fill, dtype=array.dtype)
        out[: array.shape[0]] = array
        return out

    return {
        "atom_cross_att": atom_cross_att,
        "token_features": token_features,
        "predicted_structure_info": predicted_structure_info,
        "pseudo_beta_info": pseudo_beta_info,
        "ref_structure": ref_structure,
        "frames": frames,
        # Padded tokens are inert: their parent and role read 0, and the token
        # mask discards them everywhere.
        "parent_residue_idx": pad_tokens(info["parent_residue_idx"]),
        "subtoken_role_id": pad_tokens(info["subtoken_role_id"]),
        "twin_token_idx": pad_tokens(info["twin_token_idx"], fill=NO_TWIN),
        "prev_parent_residue_idx": pad_tokens(info["prev_parent_residue_idx"], fill=-1),
        "next_parent_residue_idx": pad_tokens(info["next_parent_residue_idx"], fill=-1),
        "residue_atom_gather": info["residue_atom_gather"],
        "residue_rep_token": info["residue_rep_token"],
        "n_struct": n_struct,
    }


#: Bookkeeping the expander and the output path read, under `structbook/`.
BOOKKEEPING = (
    "parent_residue_idx",
    "subtoken_role_id",
    "twin_token_idx",
    "prev_parent_residue_idx",
    "next_parent_residue_idx",
)


def attach(
    example: dict[str, Any],
    capture: dict[str, Any],
    *,
    struct_num_tokens: int | None = None,
    pad_multiple: int = 32,
) -> dict[str, Any]:
    """Return ``example`` with the structural feature set attached.

    The structural batch goes under `struct/` and the expander bookkeeping under
    `structbook/`, so both ride the ordinary example dict and the graph pulls out
    what it needs. The residue-level keys are untouched: the trunk still runs on
    them.
    """
    from alphafold3.model import feat_batch, features

    residue = feat_batch.Batch.from_data_dict(example)
    built = build_structural_batch(
        capture,
        features.AtomCrossAtt.from_data_dict(example),
        int(np.asarray(example["aatype"]).shape[0]),
        struct_num_tokens=struct_num_tokens,
        pad_multiple=pad_multiple,
    )
    # The diffusion and the confidence heads read only these six; MSA, templates,
    # bonds and the output conversion are never reached on this path, so they
    # carry over untouched rather than being rebuilt on a token set they do not
    # describe.
    structural = dataclasses.replace(
        residue,  # pyright: ignore[reportArgumentType]
        token_features=built["token_features"],
        ref_structure=built["ref_structure"],
        predicted_structure_info=built["predicted_structure_info"],
        pseudo_beta_info=built["pseudo_beta_info"],
        atom_cross_att=built["atom_cross_att"],
        frames=built["frames"],
    )

    out = dict(example)
    for key, value in structural.as_data_dict().items():
        out["struct/" + key] = value
    out["struct/ref_pos"] = _residue_ref_pos_onto_structural(
        np.asarray(example["ref_pos"]),
        np.asarray(out["struct/ref_pos"]),
        built["residue_atom_gather"],
    )
    for key in BOOKKEEPING:
        out["structbook/" + key] = built[key]

    # The two residue-shaped gathers are built at the TRUE residue count, but the
    # residue batch they map back into is padded to a token bucket -- so pad them
    # to match, or every consumer broadcasts one length against the other. A
    # padded row gathers nothing and stands for token 0, and the mask discards it.
    num_tokens = int(np.asarray(example["aatype"]).shape[0])
    gather = built["residue_atom_gather"]
    rows = num_tokens - gather.shape[0]
    if rows < 0:
        message = f"{gather.shape[0]} residues do not fit {num_tokens} tokens"
        raise ValueError(message)
    out["structbook/residue_atom_gather"] = np.pad(
        gather, ((0, rows), (0, 0)), constant_values=-1
    )
    out["structbook/residue_rep_token"] = np.pad(built["residue_rep_token"], (0, rows))
    out["structbook/n_struct"] = np.asarray(built["n_struct"])
    return out


def _residue_ref_pos_onto_structural(
    residue: np.ndarray, structural: np.ndarray, residue_atom_gather: np.ndarray
) -> np.ndarray:
    """Give the structural batch the residue batch's reference conformer, atom for atom.

    The release keeps ONE atom array: the denoiser reads the same ``ref_pos`` as
    the trunk, posed once per residue at random. Rebuilt here from the CCD, the
    structural copy missed the pose: our 5I28 folds sat 2.4 A from the release's
    (rel-rel 1.5) -- exactly the distance between the release with its pose and
    without it.
    """
    gather = np.asarray(residue_atom_gather)
    rows = residue[: gather.shape[0]].reshape(-1, 3)
    index = gather.reshape(-1)
    keep = index >= 0
    out = structural.reshape(-1, 3).copy()
    out[index[keep]] = rows[keep]
    return out.reshape(structural.shape).astype(structural.dtype)


def structural_to_residue_positions(
    positions: np.ndarray, residue_atom_gather: np.ndarray
) -> np.ndarray:
    """Scatter structural-token coordinates back onto the residue atom layout.

    ``positions`` is (..., n_struct, max_atoms, 3) and the result is
    (..., n_res, max_atoms, 3), which is what the mmCIF writer understands. The
    reconstruction is exact: every residue atom lives in exactly one structural
    slot, and `residue_atom_gather` names it.
    """
    values = np.asarray(positions)
    lead = values.shape[:-3]
    flat = values.reshape((*lead, -1, 3))
    gather = np.asarray(residue_atom_gather)
    index = np.clip(gather, 0, flat.shape[-2] - 1).reshape(-1)
    out = flat[..., index, :].reshape((*lead, *gather.shape, 3))
    return np.where((gather >= 0)[..., None], out, 0.0).astype(np.float32)


def structural_to_residue_atoms(
    values: np.ndarray, residue_atom_gather: np.ndarray
) -> np.ndarray:
    """Apply the same scatter to a per-(token, slot) scalar, such as atom pLDDT."""
    array = np.asarray(values)
    lead = array.shape[:-2]
    flat = array.reshape((*lead, -1))
    gather = np.asarray(residue_atom_gather)
    index = np.clip(gather, 0, flat.shape[-1] - 1).reshape(-1)
    out = flat[..., index].reshape((*lead, *gather.shape))
    return np.where(gather >= 0, out, 0.0)


@contextlib.contextmanager
def capture_layouts() -> Iterator[list[dict[str, Any]]]:
    """Record what the structural layout has to be rebuilt from, while featurising.

    AF3's featurisation keeps its AtomLayouts internal and publishes only the
    arrays it derives from them, so there is nothing in the example dict to
    re-tokenise. Rather than featurise a second time to get at them, this takes
    them from the one call that already happens: the tokenizer's output, and the
    arguments the reference-structure and cross-attention builders were given.

    Yields a list that fills with one capture per example, in featurisation order.
    """
    from alphafold3.model import features

    captured: list[dict[str, Any]] = []
    tokenizer = features.tokenizer
    cross_att = features.AtomCrossAtt.compute_features.__func__
    ref_structure = features.RefStructure.compute_features.__func__

    def tokenize(flat_output_layout: Any, **kwargs: Any) -> Any:
        all_tokens, all_token_atoms_layout, standard_token_idxs = tokenizer(
            flat_output_layout, **kwargs
        )
        # One capture per TOKENISED example. The builders below are each called
        # more than once per example (the reference structure is rebuilt for the
        # templates), so they update this entry rather than opening a new one.
        captured.append(
            {
                "all_tokens": all_tokens,
                "all_token_atoms_layout": all_token_atoms_layout,
            }
        )
        return all_tokens, all_token_atoms_layout, standard_token_idxs

    def build_cross_att(
        cls: Any, all_token_atoms_layout: Any, *args: Any, **kwargs: Any
    ) -> Any:
        built = cross_att(cls, all_token_atoms_layout, *args, **kwargs)
        bound = inspect.signature(cross_att).bind(
            cls, all_token_atoms_layout, *args, **kwargs
        )
        captured[-1]["atom_cross_att"] = built
        captured[-1]["padding_shapes"] = bound.arguments["padding_shapes"]
        return built

    def build_ref_structure(
        cls: Any, all_token_atoms_layout: Any, *args: Any, **kwargs: Any
    ) -> Any:
        bound = inspect.signature(ref_structure).bind(
            cls, all_token_atoms_layout, *args, **kwargs
        )
        bound.apply_defaults()
        for name in (
            "ccd",
            "chemical_components_data",
            "ref_max_modified_date",
            "conformer_max_iterations",
        ):
            captured[-1][name] = bound.arguments[name]
        return ref_structure(cls, all_token_atoms_layout, *args, **kwargs)

    features.tokenizer = tokenize
    features.AtomCrossAtt.compute_features = classmethod(build_cross_att)  # pyright: ignore[reportAttributeAccessIssue]
    features.RefStructure.compute_features = classmethod(build_ref_structure)  # pyright: ignore[reportAttributeAccessIssue]
    try:
        yield captured
    finally:
        features.tokenizer = tokenizer
        features.AtomCrossAtt.compute_features = classmethod(cross_att)  # pyright: ignore[reportAttributeAccessIssue]
        features.RefStructure.compute_features = classmethod(ref_structure)  # pyright: ignore[reportAttributeAccessIssue]
