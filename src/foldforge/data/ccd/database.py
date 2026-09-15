# Heavy chemistry libraries load only when the selected database needs a view.
# ruff: noqa: PLC0415
"""One explicitly selected CCD database for every FoldForge predictor.

Like MiniWorld's CCDLookup, the database owns its entry caches. Model adapters
choose reference-coordinate policy; they never choose an independent database.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from alphafold3.constants.chemical_components import Ccd
    from biotite.structure.io.pdbx import CIFBlock
    from rdkit.Chem.rdchem import Mol

from pydantic import BaseModel, ConfigDict

_CURRENT: ContextVar[CCDDatabase | None] = ContextVar("foldforge_ccd", default=None)


class CCDConfig(BaseModel):
    """Files generated from one CCD release, independent of model checkpoints."""

    model_config = ConfigDict(frozen=True)
    ccd_db: Path
    source_sha256: str


class CCDDatabase:
    """Shared chemistry source and instance-owned lazy caches."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        from foldforge.data.ccd.lmdb_store import SCHEMA, CCDLookup

        manifest = self.root / "manifest.json"
        data = json.loads(manifest.read_text())
        if data.get("schema") != SCHEMA:
            msg = (
                "CCD must use MiniWorld BioMol LMDB; migrate with foldforge ccd prepare"
            )
            raise ValueError(msg)
        self.manifest = data
        self.config = CCDConfig(
            ccd_db=self.root, source_sha256=data["sources"]["components_sha256"]
        )
        for item in data["files"].values():
            path = (self.root / item["path"]).resolve()
            if not path.is_relative_to(self.root):
                msg = "CCD manifest paths must stay inside its database"
                raise ValueError(msg)
            if not path.is_file():
                raise FileNotFoundError(path)
        self.lookup = CCDLookup(self.root)
        self._caches: dict[str, dict] = {}

    def cache(self, name: str) -> dict:
        """Return a cache owned by this database, never another active source."""
        return self._caches.setdefault(name, {})

    @functools.cached_property
    def cif(self) -> Mapping:
        """Lazy CIF-shaped compatibility view of individual BioMol records."""
        return _CIFView(self)

    @functools.cached_property
    def molecules(self) -> Mapping:
        """Lazy references; no whole-database pickle is loaded at runtime."""
        return _MoleculeView(self)

    def af3_ccd(self, user_ccd: str | None = None) -> Ccd:
        """AF3 mapping over the exact same records, with explicit user overrides."""
        from alphafold3.constants.chemical_components import Ccd
        from alphafold3.cpp import cif_dict

        view: Mapping = _AF3View(self)
        if user_ccd is not None:
            if not user_ccd:
                msg = "User CCD cannot be empty"
                raise ValueError(msg)
            overrides = {
                k: {c: tuple(v) for c, v in record.items()}
                for k, record in cif_dict.parse_multi_data_cif(user_ccd).items()
            }
            view = _OverlayView(overrides, view)
        # Ccd's public Mapping contract is retained, without reading a pickle.
        result = Ccd.__new__(Ccd)
        result._dict = view  # noqa: SLF001 - shared runtime integration hook
        return result

    @property
    def chemical_component_sets(self) -> dict[str, frozenset[str]]:
        """Compute chemical component sets."""
        return {
            key: frozenset(value)
            for key, value in self.manifest["chemical_component_sets"].items()
        }

    def molecule(self, name: str) -> Any:
        """Validate atom identities before sharing a prepared reference molecule."""
        molecule = self.molecules.get(name)
        if molecule is None or molecule.GetNumAtoms() == 0:
            return molecule
        verified = self.cache("verified_molecule")
        if name not in verified:
            atom_map = molecule.atom_map
            names = self.cif[name]["chem_comp_atom"]["atom_id"].as_array()
            if set(atom_map) != set(names) or set(atom_map.values()) != set(
                range(molecule.GetNumAtoms())
            ):
                message = f"CCD and RDKit atom identities disagree for {name}"
                raise ValueError(message)
            molecule.GetConformer(molecule.ref_conf_id)
            if len(molecule.ref_mask) != molecule.GetNumAtoms():
                message = f"Invalid CCD reference mask for {name}"
                raise ValueError(message)
            verified[name] = True
        return molecule

    @functools.cached_property
    def esm_molecules(self) -> Mapping:
        """ESM atom properties over the same reference molecules."""
        return _ESMMolecules(self)

    def verify(self) -> None:
        """Verify all stored assets against the preparation manifest."""
        for item in self.manifest["files"].values():
            with (self.root / item["path"]).open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
            if actual != item["sha256"]:
                message = f"CCD asset checksum mismatch: {item['path']}"
                raise ValueError(message)

    @contextmanager
    def activate(self) -> Iterator[CCDDatabase]:
        """Scope model feature preparation to this database; nesting is safe."""
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)

    def describe(self) -> dict:
        """Source identities recorded with every prediction."""
        return {
            "root": str(self.root),
            "schema": self.manifest["schema"],
            "lmdb_sha256": self.manifest["files"]["data"]["sha256"],
            "components_sha256": self.config.source_sha256,
            "rdkit_sha256": self.manifest["sources"]["rdkit_sha256"],
        }


class _RecordView(Mapping):
    """Represent record view."""

    def __init__(self, database: CCDDatabase) -> None:
        self.database = database

    def __iter__(self) -> Iterator[str]:
        return iter(self.database.lookup)

    def __len__(self) -> int:
        return len(self.database.lookup)


class _AF3View(_RecordView):
    """Represent a f3 view."""

    def __getitem__(self, key: str) -> dict[str, list[str]]:
        metadata = self.database.lookup.ccdmol(key).metadata
        return {
            "data_": [key],
            **metadata["ccd_cif"],
            **metadata.get("af3_text_overrides", {}),
        }


class _CIFView(_RecordView):
    """Represent c i f view."""

    def __getitem__(self, key: str) -> CIFBlock:
        from biotite.structure.io.pdbx import CIFBlock, CIFCategory

        cache = self.database.cache("cif_block")
        if key not in cache:
            raw = self.database.lookup.ccdmol(key).metadata["ccd_cif"]
            categories = {}
            for column, values in raw.items():
                category, name = column[1:].split(".", 1)
                categories.setdefault(category, {})[name] = values
            block = CIFBlock()
            for category, columns in categories.items():
                block[category] = CIFCategory(columns)
            cache[key] = block
        return cache[key]


class _MoleculeView(_RecordView):
    """Represent molecule view."""

    def __getitem__(self, key: str) -> Mol | None:
        import numpy as np
        from rdkit import Chem

        cache = self.database.cache("rdkit_reference")
        if key not in cache:
            reference = self.database.lookup.ccdmol(key).metadata["reference"]
            if reference is None:
                cache[key] = None
            else:
                # RDKit supports binary construction; its stubs omit this overload.

                from_binary = cast("Callable[[bytes], Mol]", Chem.Mol)
                mol = from_binary(base64.b64decode(reference["rdkit_binary"]))
                mol.__dict__.update(reference["properties"])
                mol.ref_mask = np.asarray(mol.ref_mask, dtype=bool)
                cache[key] = mol
        return cache[key]


class _ESMMolecules(Mapping):
    """Lazy RDKit property view; reference coordinates and atom indices stay intact."""

    def __init__(self, database: CCDDatabase) -> None:
        self.database = database

    def __iter__(self) -> Iterator[str]:
        return iter(self.database.molecules)

    def __len__(self) -> int:
        return len(self.database.molecules)

    def __getitem__(self, key: str) -> Any:
        from rdkit import Chem

        cache = self.database.cache("esm_molecule_view")
        if key in cache:
            return cache[key]
        original = self.database.molecule(key)
        if original is None:
            raise KeyError(key)
        mol = Chem.Mol(original)
        atom_map = original.atom_map
        component = self.database.cif[key]["chem_comp_atom"]
        names = component["atom_id"].as_array()
        leaving = dict(
            zip(names, component["pdbx_leaving_atom_flag"].as_array(), strict=True)
        )
        for name, index in atom_map.items():
            atom = mol.GetAtomWithIdx(index)
            atom.SetProp("name", name)
            atom.SetProp("leaving_atom", "1" if leaving.get(name) == "Y" else "0")
        for conformer in mol.GetConformers():
            # Preserve the prepared reference choice, do not generate a new
            # conformer or assume a shared RDKit RNG matches another release.
            conformer.SetProp(
                "name",
                "Computed"
                if conformer.GetId() == original.ref_conf_id
                else "Alternative",
            )
        cache[key] = mol
        return mol


def current_database() -> CCDDatabase:
    """Return the caller-selected database or fail before a hidden fallback."""
    database = _CURRENT.get()
    if database is None:
        message = "CCD access requires an explicit CCDDatabase.activate() context"
        raise RuntimeError(message)
    return database


def database_cache(function: Callable | None = None, **_options: Any) -> Callable:
    """Cache a lookup inside the selected database, including missing entries."""

    def decorate(fn: Callable) -> Callable:
        """Compute decorate."""

        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            """Compute wrapped."""
            cache = current_database().cache(fn.__module__ + "." + fn.__qualname__)
            key = (args, tuple(sorted(kwargs.items())))
            if key not in cache:
                cache[key] = fn(*args, **kwargs)
            return cache[key]

        return wrapped

    return decorate(function) if function is not None else decorate


def default_path() -> Path:
    """Shared CLI default, independent of the selected model."""
    return Path(os.environ.get("FOLDFORGE_CCD_DB", "data/ccd/preprocessed_CCD.lmdb"))


class _OverlayView(Mapping):
    """Read user overrides ahead of lazy CCD records without materializing the DB."""

    def __init__(self, overrides: Mapping, base: Mapping) -> None:
        self.overrides = overrides
        self.base = base

    def __getitem__(self, key: str) -> Any:
        if key in self.overrides:
            return self.overrides[key]
        return self.base[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.overrides
        yield from (key for key in self.base if key not in self.overrides)

    def __len__(self) -> int:
        return len(self.base) + sum(key not in self.base for key in self.overrides)
