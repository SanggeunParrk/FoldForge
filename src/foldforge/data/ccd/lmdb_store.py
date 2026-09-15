# Optional chemistry/GPU backends load at the selected execution boundary.
# ruff: noqa: PLC0415
"""MiniWorld-compatible, per-component BioMol LMDB access.

The default LMDB database contains only CCD IDs and BioMol.to_bytes records.
Model-specific full CIF and reference-conformer views live in record metadata;
the atom/residue/chain schema remains readable by MiniWorld CCDMol.from_bytes.
"""

from __future__ import annotations

import os
import threading
import weakref
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import lmdb
import numpy as np

from foldforge.data.ccd.mol import CCDMol

if TYPE_CHECKING:
    from foldforge.data.ccd.fragmented_mol import FragmentedCCDMol

SCHEMA = "miniworld.ccd.biomol.v1"

_LOCK = threading.RLock()
_ENVS: weakref.WeakValueDictionary[tuple[int, Path], _Reader] = (
    weakref.WeakValueDictionary()
)


class _Reader:
    """Represent reader."""

    def __init__(self, path: Path) -> None:
        self.env: lmdb.Environment | None = lmdb.open(
            str(path),
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=4096,
            subdir=True,
        )

    def close(self) -> None:
        """Release the open resources."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def __del__(self) -> None:
        if hasattr(self, "env"):
            self.close()


def _after_fork() -> None:
    # LMDB forbids opening a path twice in one process. Close inherited reader
    # handles before lazy reopening in a dataloader child; parent is unaffected.
    global _LOCK  # noqa: PLW0603 - reset inherited lock after fork
    for reader in list(_ENVS.values()):
        reader.close()
    _ENVS.clear()
    _LOCK = threading.RLock()


os.register_at_fork(after_in_child=_after_fork)


@dataclass(frozen=True)
class CCDResidue:
    """Canonical heavy atoms and model coordinates, as in MiniWorld."""

    chemcomp_id: str
    atom_ids: np.ndarray
    atom_elements: np.ndarray
    atom_charges: np.ndarray
    atom_xyz: np.ndarray

    @property
    def n_atoms(self) -> int:
        """Compute n atoms."""
        return len(self.atom_xyz)


class CCDLookup(Mapping):
    """Lazy MiniWorld CCD lookup with process-local, shared LMDB readers."""

    def __init__(self, ccd_db_path: Path) -> None:
        self.ccd_db_path = Path(ccd_db_path).expanduser().resolve()
        if not (self.ccd_db_path / "data.mdb").is_file():
            raise FileNotFoundError(self.ccd_db_path / "data.mdb")
        self._reader: _Reader | None = None
        self._pid = None
        self._ccdmol_cache = {}
        self._residue_cache = {}
        self._fragments_cache = {}

    @property
    def _env(self) -> lmdb.Environment:
        pid = os.getpid()
        if self._pid != pid or self._reader is None:
            with _LOCK:
                key = (pid, self.ccd_db_path)
                reader = _ENVS.get(key)
                if reader is None:
                    reader = _Reader(self.ccd_db_path)
                    _ENVS[key] = reader
                self._reader, self._pid = reader, pid
        reader = self._reader
        if reader is None or reader.env is None:
            message = "The component database reader is closed"
            raise RuntimeError(message)
        return reader.env

    def __getstate__(self) -> dict[str, Path]:
        return {"ccd_db_path": self.ccd_db_path}

    def __setstate__(self, state: dict[str, Path]) -> None:
        self.__init__(state["ccd_db_path"])

    def __iter__(self) -> Iterator[str]:
        with self._env.begin() as txn:
            keys = [bytes(key).decode() for key, _ in txn.cursor()]
        return iter(keys)

    def __len__(self) -> int:
        return self._env.stat()["entries"]

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        with self._env.begin() as txn:
            return txn.get(key.encode()) is not None

    def raw(self, key: str) -> bytes:
        """Compute raw."""
        with self._env.begin() as txn:
            raw = txn.get(key.encode())
        if raw is None:
            msg = f"CCD entry {key!r} not found in {self.ccd_db_path}"
            raise KeyError(msg)
        return bytes(raw)

    def ccdmol(self, key: str) -> CCDMol:
        """Compute ccdmol."""
        if key not in self._ccdmol_cache:
            mol = CCDMol.from_bytes(self.raw(key))
            if list(mol.chains.id.value) != [key]:
                msg = f"CCD key/chain identity mismatch: {key}"
                raise ValueError(msg)
            self._ccdmol_cache[key] = mol
        return self._ccdmol_cache[key]

    _load_ccdmol = ccdmol

    def fragments(self, key: str) -> dict[int, FragmentedCCDMol]:
        """MiniWorld v2 merge levels, with explicit unavailable-chemistry errors."""
        from foldforge.data.ccd.fragment import fragment_ccdmol_all_merges

        if key not in self._fragments_cache:
            mol = self.ccdmol(key)
            if (
                len(mol.atoms) == 0
                or mol.metadata.get("fragmentation_available") is False
            ):
                msg = f"CCD {key} lacks valid heavy-atom fragmentation chemistry"
                raise ValueError(msg)
            self._fragments_cache[key] = fragment_ccdmol_all_merges(mol)
        return self._fragments_cache[key]

    def __getitem__(self, key: str) -> CCDResidue:
        if key not in self._residue_cache:
            mol = self.ccdmol(key)
            xyz = np.asarray(mol.atoms.model_xyz.value, dtype=object).copy()
            xyz[(xyz == "?") | (xyz == ".")] = 0.0
            xyz = np.nan_to_num(xyz.astype(np.float32), nan=0.0)
            charge = np.asarray(mol.atoms.charge.value)
            self._residue_cache[key] = CCDResidue(
                key,
                np.asarray(mol.atoms.id.value),
                np.asarray(mol.atoms.element.value),
                np.array(
                    [0.0 if c in {"?", "."} else float(c) for c in charge],
                    dtype=np.float32,
                ),
                xyz,
            )
        return self._residue_cache[key]
