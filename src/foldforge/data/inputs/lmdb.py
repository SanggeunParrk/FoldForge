# Optional chemistry/GPU backends load at the selected execution boundary.
# ruff: noqa: PLC0415
"""Read MiniWorld/StructCooker MSA and aligned TemplateMol records by key.

No key enumeration, DB conversion, template search, or fragment tokenization.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lmdb
import numpy as np
from biomol.core.utils import load_bytes

from foldforge.data.inputs.template_mol import TemplateMol

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

# Each lookup opens/closes a readonly environment inside the lock. No live LMDB
# handles survive input preparation or are pickled into dataloader workers.
_READ_LOCK = threading.RLock()


MIN_ALIGNED_BACKBONES = 4


def _after_fork() -> None:
    global _READ_LOCK  # noqa: PLW0603 - reset inherited lock after fork
    _READ_LOCK = threading.RLock()


os.register_at_fork(after_in_child=_after_fork)


def read_record(path: Path, key: str) -> dict[str, Any]:
    """Read record."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    with (
        _READ_LOCK,
        lmdb.open(
            str(path),
            subdir=path.is_dir(),
            readonly=True,
            create=False,
            lock=False,
            readahead=False,
        ) as env,
        env.begin() as txn,
    ):
        raw = txn.get(key.encode("utf-8"))
    if raw is None:
        msg = f"LMDB key {key!r} not found in {path}"
        raise KeyError(msg)
    return dict(load_bytes(raw))


def text(value: object) -> str:
    """Compute text."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


ALPHABETS = {
    "protein": dict(enumerate("ARNDCQEGHILKMFPSTWYVX")) | {31: "-"},
    "rna": dict(enumerate("AUGCN", start=21)) | {31: "-"},
    "dna": dict(enumerate("ATGCN", start=26)) | {31: "-"},
}
UNPAIRED = frozenset({"", "query", "unknown", "none", "nan", "-1", "n/a"})


@dataclass(frozen=True)
class Alignment:
    """Original alignment tokens, counts and species; no lossy re-alignment."""

    query: np.ndarray
    tokens: np.ndarray
    deletions: np.ndarray
    species: tuple[str, ...]
    profile: np.ndarray
    deletion_mean: np.ndarray
    kind: str

    @classmethod
    def load(cls, path: Path, key: str, kind: str, sequence: str) -> Alignment:  # noqa: C901 - validate one complete MSA record
        """Load ."""
        if kind not in ALPHABETS:
            msg = f"MSA-LMDB is not defined for {kind}"
            raise ValueError(msg)
        data = read_record(path, key)["msa_dict"]
        seq, headers = data["sequences"], data["headers"]
        query, tokens, deletions = (
            np.asarray(seq[k])
            for k in ("query_sequence", "aligned_sequences", "deletions")
        )
        species = tuple(text(x) for x in headers["species"])
        if tokens.ndim != 2 or tokens.shape != deletions.shape or tokens.shape[0] < 1:  # noqa: PLR2004 - tensor rank or format cardinality
            msg = f"{key}: invalid MSA/deletion shape"
            raise ValueError(msg)
        if query.shape != (tokens.shape[1],) or len(species) != len(tokens):
            msg = f"{key}: inconsistent query/species dimensions"
            raise ValueError(msg)
        # StructCooker stores query_sequence as letters; query-only MiniWorld
        # records may use integer tokens. Normalize both without guessing alphabet.
        if query.dtype.kind in {"U", "S"}:
            letters = [text(x) for x in query]
            mapping = {v: k for k, v in ALPHABETS[kind].items()}
            if any(x not in mapping for x in letters):
                msg = f"{key}: invalid query letters"
                raise ValueError(msg)
            query = np.asarray([mapping[x] for x in letters], dtype=np.int32)
        if not np.issubdtype(tokens.dtype, np.integer) or not np.issubdtype(
            query.dtype, np.integer
        ):
            msg = f"{key}: MSA tokens must be integers"
            raise ValueError(msg)
        if (
            not np.isin(tokens, list(ALPHABETS[kind])).all()
            or not np.isin(query, list(ALPHABETS[kind])).all()
        ):
            msg = f"{key}: token outside MiniWorld {kind} alphabet"
            raise ValueError(msg)
        decoded = "".join(ALPHABETS[kind][int(x)] for x in query)
        if decoded != sequence or not np.array_equal(query, tokens[0]):
            msg = f"{key}: LMDB query does not match FASTA sequence"
            raise ValueError(msg)
        if not np.issubdtype(deletions.dtype, np.integer) or (deletions < 0).any():
            msg = f"{key}: deletions must be nonnegative integers"
            raise ValueError(msg)
        if deletions[0].any():
            msg = f"{key}: query row must not contain insertions"
            raise ValueError(msg)
        profile, mean = np.asarray(seq["profile"]), np.asarray(seq["deletion_mean"])
        if (
            profile.ndim != 2  # noqa: PLR2004 - tensor rank or format cardinality
            or profile.shape[0] != len(query)
            or mean.shape != query.shape
        ):
            msg = f"{key}: invalid profile/deletion_mean shape"
            raise ValueError(msg)
        return cls(query, tokens, deletions, species, profile, mean, kind)

    def a3m(self, aliases: dict[str, str], depth: int | None = None) -> str:
        # Only insertion counts are retained in MiniWorld's DB. Lowercase x
        # encodes those exact counts; no original inserted amino acids are invented.
        """Compute a3m."""
        lines = []
        alphabet = ALPHABETS[self.kind]
        for i, (row, deletion, species) in enumerate(
            zip(self.tokens, self.deletions, self.species, strict=False)
        ):
            if depth is not None and i >= depth:
                break
            alias = aliases.get(species) if i else None
            # Native AF3 and P/O recognize UniProt-shaped species identifiers.
            # These are explicitly transport aliases, not biological accessions.
            header = f"tr|000000|MW_{alias}" if alias else f"row{i}"
            lines.extend(
                [
                    ">" + header,
                    "".join(
                        "x" * int(d) + alphabet[int(t)]
                        for t, d in zip(row, deletion, strict=False)
                    ),
                ]
            )
        return "\n".join(lines) + "\n"


def species_aliases(alignments: Iterable[Alignment]) -> dict[str, str]:
    """Compute species aliases."""
    species = sorted(
        {s for a in alignments for s in a.species[1:] if s.lower() not in UNPAIRED}
    )
    if len(species) >= 36**5:
        msg = "Too many species for native MSA header transport"
        raise ValueError(msg)
    return {s: np.base_repr(i + 1, base=36).zfill(5) for i, s in enumerate(species)}


@dataclass(frozen=True)
class Template:
    """Represent template."""

    id: str
    sequence: str
    positions: np.ndarray  # [query_length, 4, 3], N/CA/C/CB
    mask: np.ndarray
    release_date: str | None = None

    def payload(self) -> dict[str, Any]:
        """Compute payload."""
        return {
            "id": self.id,
            "sequence": self.sequence,
            "positions": self.positions.tolist(),
            "mask": self.mask.tolist(),
            "release_date": self.release_date,
        }

    def af3(self) -> dict[str, Any]:
        """One-chain mmCIF + exact query mapping, retaining partial atom masks."""
        from io import StringIO

        from biotite.sequence import ProteinSequence
        from biotite.structure.io.pdbx import CIFBlock, CIFCategory, CIFFile

        # Compact out alignment gaps; include sequence positions with no atoms
        # in entity_poly_seq so AF3 preserves their indices and atom masks.
        indices = [i for i, aa in enumerate(self.sequence) if aa != "-"]
        names = [
            ProteinSequence.convert_letter_1to3(self.sequence[i])
            if self.sequence[i] != "X"
            else "UNK"
            for i in indices
        ]
        block = CIFBlock()
        block["entry"] = CIFCategory({"id": ["miniworld_template"]})
        # AF3 requires a release date even for explicitly supplied templates.
        # Unknown stays explicit in payloads; the native transport sentinel is
        # the same future date used by Protenix for supplied JSON templates.
        block["pdbx_audit_revision_history"] = CIFCategory(
            {"ordinal": ["1"], "revision_date": [self.release_date or "9999-12-31"]}
        )
        block["entity"] = CIFCategory({"id": ["1"], "type": ["polymer"]})
        block["struct_asym"] = CIFCategory({"id": ["A"], "entity_id": ["1"]})
        block["entity_poly"] = CIFCategory(
            {
                "entity_id": ["1"],
                "type": ["polypeptide(L)"],
                "pdbx_strand_id": ["A"],
                "pdbx_seq_one_letter_code_can": [
                    "".join(self.sequence[i] for i in indices)
                ],
            }
        )
        block["entity_poly_seq"] = CIFCategory(
            {
                "entity_id": ["1"] * len(indices),
                "num": list(map(str, range(1, len(indices) + 1))),
                "mon_id": names,
                "hetero": ["n"] * len(indices),
            }
        )
        columns = [
            "group_PDB",
            "id",
            "type_symbol",
            "label_atom_id",
            "label_alt_id",
            "label_comp_id",
            "label_asym_id",
            "label_entity_id",
            "label_seq_id",
            "pdbx_PDB_ins_code",
            "Cartn_x",
            "Cartn_y",
            "Cartn_z",
            "occupancy",
            "B_iso_or_equiv",
            "auth_seq_id",
            "auth_comp_id",
            "auth_asym_id",
            "auth_atom_id",
            "pdbx_PDB_model_num",
        ]
        rows = []
        for j, (q, name) in enumerate(zip(indices, names, strict=False), 1):
            for a, atom in enumerate(("N", "CA", "C", "CB")):
                if self.mask[q, a]:
                    rows.append(
                        [
                            "ATOM",
                            str(len(rows) + 1),
                            atom[0],
                            atom,
                            ".",
                            name,
                            "A",
                            "1",
                            str(j),
                            "?",
                            *[str(float(x)) for x in self.positions[q, a]],
                            "1",
                            "0",
                            str(j),
                            name,
                            "A",
                            atom,
                            "1",
                        ]
                    )
        block["atom_site"] = CIFCategory(
            dict(zip(columns, map(list, zip(*rows, strict=False)), strict=False))
        )
        cif = CIFFile({"miniworld_template": block})
        stream = StringIO()
        cif.write(stream)
        return {
            "mmcif": stream.getvalue(),
            "queryIndices": indices,
            "templateIndices": list(range(len(indices))),
        }


def load_templates(
    path: Path, key: str, length: int, count: int
) -> tuple[Template, ...]:
    """Load templates."""
    if count == 0:
        return ()
    mols = read_record(path, key)["template_mols"]
    templates = []
    for name, item in mols.items():
        mol = TemplateMol.from_dict(item)
        if len(mol.residues) != length:
            msg = f"{key}/{name}: aligned template length differs from query {length}"
            raise ValueError(msg)
        ids = np.asarray(mol.atoms.id.value).reshape(length, 4)
        ids = np.asarray([[text(x) for x in row] for row in ids])
        allowed = np.array(["N", "CA", "C", "CB"])
        valid_ids = (ids == allowed) | (ids == "")
        valid_ids[:, 3] |= ids[:, 3] == "CA"
        if not valid_ids.all():
            msg = f"{key}/{name}: expected MiniWorld N/CA/C/CB atom slots"
            raise ValueError(msg)
        positions = np.asarray(mol.atoms.xyz.value, dtype=np.float32).reshape(
            length, 4, 3
        )
        mask = np.isfinite(positions).all(axis=-1) & (ids == allowed)
        # Slot 4 can contain CA as MiniWorld pseudo-beta fallback. This is not
        # an observed CB; released AF3-family featurizers select glycine CA.
        # Keep actual atom identity rather than manufacturing a CB atom.
        # StructCooker pads unaligned residues with empty strings and NaNs.
        sequence = "".join(
            text(x) or "-" for x in mol.residues.one_letter_code_can.value
        )
        if len(sequence) != length or any(
            x not in "ARNDCQEGHILKMFPSTWYVX-" for x in sequence
        ):
            msg = f"{key}/{name}: invalid protein template sequence"
            raise ValueError(msg)
        mask[np.array(list(sequence)) == "-"] = False
        if int(mask[:, :3].all(axis=-1).sum()) < MIN_ALIGNED_BACKBONES:
            continue  # Same minimum backbone coverage as MiniWorld.
        templates.append(
            Template(
                str(name),
                sequence,
                np.where(mask[..., None], positions, 0),
                mask,
                mol.metadata.get("release_date"),
            )
        )
        if len(templates) >= count:
            break
    return tuple(templates)


def template_features(
    payloads: Sequence[Mapping[str, Any]], query_sequence: str, family: str
) -> list[dict[str, Any]]:
    """Use aligned coordinates directly at each released featurizer's boundary."""
    from foldforge.data.constants import chemistry as constants
    from foldforge.data.template import features as utils

    if family not in {"protenix", "opendde"}:
        msg = f"Unsupported flat template layout: {family}"
        raise ValueError(msg)
    results = []
    for payload in payloads:
        sequence = payload["sequence"]
        pos = np.asarray(payload["positions"], dtype=np.float32)
        mask = np.asarray(payload["mask"], dtype=bool)
        length = len(query_sequence)
        if (
            len(sequence) != length
            or pos.shape != (length, 4, 3)
            or mask.shape != (length, 4)
        ):
            msg = "Invalid aligned MiniWorld template shape"
            raise ValueError(msg)
        if not np.isfinite(pos).all():
            msg = "Template positions must be finite with an explicit mask"
            raise ValueError(msg)
        full_pos = np.zeros((length, 37, 3), dtype=np.float32)
        full_mask = np.zeros((length, 37), dtype=np.float32)
        for a, atom in enumerate(("N", "CA", "C", "CB")):
            full_pos[:, utils.ATOM37_ORDER[atom]] = np.where(
                mask[:, a, None], pos[:, a], 0
            )
            full_mask[:, utils.ATOM37_ORDER[atom]] = mask[:, a]
        results.append(
            {
                "template_all_atom_positions": full_pos,
                "template_all_atom_masks": full_mask,
                "template_sequence": sequence.encode(),
                "template_aatype": np.asarray(
                    utils.encode_template_restype(constants.PROTEIN_CHAIN, sequence),
                    dtype=np.int32,
                ),
                "template_domain_names": np.array(payload["id"].encode(), dtype=object),
                "template_sum_probs": [1.0],
                "template_release_date": np.array(
                    (payload.get("release_date") or "9999-12-31").encode(), dtype=object
                ),
            }
        )
    return results
