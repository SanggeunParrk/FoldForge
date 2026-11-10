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
    from collections.abc import Iterable

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


# Token order of the nucleic alphabets differs between MSA LMDBs of the same family: MiniWorld's ResidueMapping writes RNA as A U G C and
# DNA as A T G C (21.. / 26..), the FoldBench LMDBs as A G C U and A G C T (AlphaFold 3's order). The first order of each kind is the
# default; ``Alignment.load`` takes the one under which the stored query letters equal the first aligned row, so a record is always decoded
# with the order it was written in.
_ORDERS = {
    "protein": ("ARNDCQEGHILKMFPSTWYVX",),
    "rna": ("AUGCN", "AGCUN"),
    "dna": ("ATGCN", "AGCTN"),
}
_FIRST_TOKEN = {"protein": 0, "rna": 21, "dna": 26}
#: letters that stand for "unknown / modified residue" on either side of a query-versus-FASTA comparison
_WILDCARD = {"protein": frozenset("X"), "rna": frozenset("XN"), "dna": frozenset("XN")}


def _agree(a: str, b: str, wildcard: frozenset[str]) -> bool:
    """Equal length, and equal letter by letter wherever neither side is an unknown / modified residue."""
    return len(a) == len(b) and all(x == y or x in wildcard or y in wildcard for x, y in zip(a, b, strict=True))


def alphabet_of(kind: str, order: str) -> dict[int, str]:
    """Token -> letter for one ``order`` of ``kind`` (gap = 31)."""
    return dict(enumerate(order, start=_FIRST_TOKEN[kind])) | {31: "-"}


ALPHABETS = {kind: alphabet_of(kind, orders[0]) for kind, orders in _ORDERS.items()}
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
    #: token -> letter of the order this record was written in (None: the default ``ALPHABETS[kind]``)
    alphabet: dict[int, str] | None = None

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
        wildcard = _WILDCARD[kind]
        unknown = "X" if kind == "protein" else "N"
        alphabet = ALPHABETS[kind]
        if query.dtype.kind in {"U", "S"}:
            # an unknown / modified nucleotide is spelled N or X in a query; it is the alphabet's N either way
            letters = [unknown if text(x) in wildcard else text(x) for x in query]
            if not np.issubdtype(tokens.dtype, np.integer):
                msg = f"{key}: MSA tokens must be integers"
                raise ValueError(msg)
            for order in _ORDERS[kind]:
                candidate = alphabet_of(kind, order)
                mapping = {v: k for k, v in candidate.items()}
                if all(x in mapping for x in letters) and np.array_equal(
                    np.asarray([mapping[x] for x in letters], dtype=np.int32), tokens[0]
                ):
                    alphabet = candidate
                    query = np.asarray([mapping[x] for x in letters], dtype=np.int32)
                    break
            else:
                msg = f"{key}: the query letters match the first aligned row under no known {kind} token order {_ORDERS[kind]}"
                raise ValueError(msg)
        if not np.issubdtype(tokens.dtype, np.integer) or not np.issubdtype(
            query.dtype, np.integer
        ):
            msg = f"{key}: MSA tokens must be integers"
            raise ValueError(msg)
        if (
            not np.isin(tokens, list(alphabet)).all()
            or not np.isin(query, list(alphabet)).all()
        ):
            msg = f"{key}: token outside MiniWorld {kind} alphabet"
            raise ValueError(msg)
        decoded = "".join(alphabet[int(x)] for x in query)
        # a cross-linked chromophore (GFP's CR2 / CH6 ...) is ONE residue in the FASTA and three in an MSA written for the expanded chain: the
        # extra columns after the residue are folded away (query, every aligned row, deletions, profile); `keep` selects the columns that stay.
        keep = None
        if len(decoded) > len(sequence):
            extra = len(decoded) - len(sequence)
            for p, s in enumerate(sequence):
                if s in wildcard and _agree(decoded[:p], sequence[:p], wildcard) and _agree(decoded[p + 1 + extra :], sequence[p + 1 :], wildcard):
                    keep = np.ones(len(decoded), dtype=bool)
                    keep[p + 1 : p + 1 + extra] = False
                    query, tokens, deletions = query[keep], tokens[:, keep], deletions[:, keep]
                    decoded = "".join(alphabet[int(x)] for x in query)
                    break
        # the FASTA spells an unknown / modified residue X (its parent is not in the parser's table); the MSA query carries the entity's
        # canonical letter there. Everywhere else the two must agree.
        same = _agree(decoded, sequence, wildcard)
        if not same or not np.array_equal(query, tokens[0]):
            msg = f"{key}: LMDB query does not match FASTA sequence"
            raise ValueError(msg)
        if not np.issubdtype(deletions.dtype, np.integer) or (deletions < 0).any():
            msg = f"{key}: deletions must be nonnegative integers"
            raise ValueError(msg)
        if deletions[0].any():
            msg = f"{key}: query row must not contain insertions"
            raise ValueError(msg)
        profile, mean = np.asarray(seq["profile"]), np.asarray(seq["deletion_mean"])
        if keep is not None:
            profile, mean = profile[keep], mean[keep]
        if (
            profile.ndim != 2  # noqa: PLR2004 - tensor rank or format cardinality
            or profile.shape[0] != len(query)
            or mean.shape != query.shape
        ):
            msg = f"{key}: invalid profile/deletion_mean shape"
            raise ValueError(msg)
        return cls(query, tokens, deletions, species, profile, mean, kind, alphabet if alphabet is not ALPHABETS[kind] else None)

    def merge_query(self, sequence: str) -> str:
        """``sequence`` with its unknown / modified positions (``X``; ``N`` for nucleic acids) replaced by the MSA query's canonical letter.

        An MSA is written for the entity's canonical sequence, so where the FASTA names a modified residue the query carries its parent; the
        AF3 parser requires the MSA's first row to equal the chain's sequence exactly. The chain's modified residues are one-letter in the
        released adapters (``modifications`` is empty), so taking the parent letter changes nothing else."""
        wildcard = _WILDCARD[self.kind]
        alphabet = self.alphabet or ALPHABETS[self.kind]
        decoded = "".join(alphabet[int(x)] for x in self.query)
        return "".join(d if s in wildcard and d not in wildcard else s for s, d in zip(sequence, decoded, strict=True))

    def a3m(self, aliases: dict[str, str], depth: int | None = None) -> str:
        # Only insertion counts are retained in MiniWorld's DB. Lowercase x
        # encodes those exact counts; no original inserted amino acids are invented.
        """Compute a3m."""
        lines = []
        alphabet = self.alphabet or ALPHABETS[self.kind]
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


#: ambiguous / rare one-letter codes a template sequence may carry (Glx, Asx, Xle, selenocysteine, pyrrolysine): an unknown residue
_AMBIGUOUS = str.maketrans("BZJUO", "XXXXX")
MIN_TEMPLATE_IDENTITY = 0.25


def _template_start(template: str, query: str, offset: int) -> int | None:
    """Where the query starts inside the template's aligned frame, or None when the template cannot be placed on it.

    The template records are aligned to the entity's full sequence; a FoldBench input may carry a trimmed one. Same length: start 0. An
    explicit ``offset`` (the residues trimmed from the front) is taken as given. A longer template with no offset: the start whose letters
    agree best with the query over the aligned residues (a homolog, so a clear winner and not a perfect match is what to look for)."""
    length, full = len(query), len(template)
    if offset:
        return offset if full >= offset + length else None
    if full == length:
        return 0
    if full < length:
        return None
    best, best_start = 0.0, None
    for start in range(full - length + 1):
        pairs = [(a, b) for a, b in zip(template[start : start + length], query, strict=True) if a != "-" and a != "X" and b != "X"]
        if len(pairs) >= MIN_ALIGNED_BACKBONES:
            identity = sum(a == b for a, b in pairs) / len(pairs)
            if identity > best:
                best, best_start = identity, start
    return best_start if best >= MIN_TEMPLATE_IDENTITY else None


def load_templates(
    path: Path, key: str, query: "str | int", count: int, offset: int = 0
) -> tuple[Template, ...]:
    """Load the templates of ``key`` placed on ``query``. ``query`` is the chain length (int): every template must be aligned to exactly that
    many residues, else the record is refused. ``query`` as the chain's one-letter sequence (str) also takes a template aligned in a longer
    frame: placed by ``offset`` (residues trimmed from the front of the chain, FoldBench's ``template_offset``) or, with none, by the start that
    agrees best with the sequence; a template that cannot be placed is skipped."""
    if count == 0:
        return ()
    exact = isinstance(query, int)
    length = query if exact else len(query)
    mols = read_record(path, key)["template_mols"]
    templates = []
    seen: set[str] = set()
    for name, item in mols.items():
        mol = TemplateMol.from_dict(item)
        full = len(mol.residues)
        if exact and full != length:
            msg = f"{key}/{name}: aligned template length differs from query {length}"
            raise ValueError(msg)
        ids = np.asarray(mol.atoms.id.value).reshape(full, 4)
        ids = np.asarray([[text(x) for x in row] for row in ids])
        allowed = np.array(["N", "CA", "C", "CB"])
        valid_ids = (ids == allowed) | (ids == "")
        valid_ids[:, 3] |= ids[:, 3] == "CA"
        if not valid_ids.all():
            msg = f"{key}/{name}: expected MiniWorld N/CA/C/CB atom slots"
            raise ValueError(msg)
        positions = np.asarray(mol.atoms.xyz.value, dtype=np.float32).reshape(
            full, 4, 3
        )
        # StructCooker pads unaligned residues with empty strings and NaNs.
        sequence = "".join(
            text(x) or "-" for x in mol.residues.one_letter_code_can.value
        ).translate(_AMBIGUOUS)
        if len(sequence) != full or any(
            x not in "ARNDCQEGHILKMFPSTWYVX-" for x in sequence
        ):
            msg = f"{key}/{name}: invalid protein template sequence"
            raise ValueError(msg)
        start = 0 if exact else _template_start(sequence, query, offset)
        if start is None:
            continue  # cannot be placed on this (trimmed) chain
        window = slice(start, start + length)
        ids, positions, sequence = ids[window], positions[window], sequence[window]
        mask = np.isfinite(positions).all(axis=-1) & (ids == allowed)
        # Slot 4 can contain CA as MiniWorld pseudo-beta fallback. This is not
        # an observed CB; released AF3-family featurizers select glycine CA.
        # Keep actual atom identity rather than manufacturing a CB atom.
        mask[np.array(list(sequence)) == "-"] = False
        if int(mask[:, :3].all(axis=-1).sum()) < MIN_ALIGNED_BACKBONES:
            continue  # Same minimum backbone coverage as MiniWorld.
        # One template per hit sequence, as AF3's and Protenix's template search
        # keep it: the LMDB stores every chain of an entry (5SYD_A, 5SYD_B, ...),
        # so the first ``count`` records would repeat the same structure.
        if sequence.replace("-", "") in seen:
            continue
        seen.add(sequence.replace("-", ""))
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
