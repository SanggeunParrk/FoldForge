"""Model-independent input loading and validation."""
# ruff: noqa: PLC0415 - native chemistry parsing remains behind explicit loading

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from foldforge.data.inputs.build import Input
from foldforge.errors import InputError


@dataclass(frozen=True, repr=False)
class Inputs:
    """Resolved target resources and their origin, reusable across model families."""

    target: Input | None
    payload: dict[str, Any] | None
    origin: Path

    @property
    def name(self) -> str:
        """Target name independent of model choice."""
        return (
            (self.target.spec.name or "prediction")
            if self.target is not None
            else (self.payload or {}).get("name", "prediction")
        )

    def __repr__(self) -> str:
        chains = (
            len(self.target.chains)
            if self.target is not None
            else len((self.payload or {}).get("sequences", []))
        )
        return (
            f"Inputs(name={self.name!r}, chain_records={chains}, "
            f"origin={str(self.origin)!r})"
        )


def protein(
    sequence: str, *, name: str = "prediction", chain_id: str = "A"
) -> dict[str, Any]:
    """Build a single protein without performing an MSA or template search."""
    if not isinstance(sequence, str):
        message = "Protein sequence must be a string"
        raise InputError(message)
    sequence = "".join(sequence.split()).upper()
    payload = {
        "name": name,
        "dialect": "alphafold3",
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": chain_id,
                    "sequence": sequence,
                    "unpairedMsa": "",
                    "pairedMsa": "",
                    "templates": [],
                }
            }
        ],
    }
    return normalize_payload(payload)


def normalize_payload(data: Any) -> dict[str, Any]:
    """Validate raw AF3 inputs, then supply explicit empty conditioning defaults."""
    if not isinstance(data, dict) or not isinstance(data.get("sequences"), list):
        message = "Inputs must contain an AlphaFold3 sequences list"
        raise InputError(message)
    data.setdefault("name", "prediction")
    data.setdefault("dialect", "alphafold3")
    data.setdefault("version", 1)
    for chain in data["sequences"]:
        if isinstance(chain, dict) and isinstance(chain.get("protein"), dict):
            for key, default in (
                ("unpairedMsa", ""),
                ("pairedMsa", ""),
                ("templates", []),
            ):
                if f"{key}Path" not in chain["protein"]:
                    chain["protein"].setdefault(key, deepcopy(default))
    _validate_structure(data)
    return data


def _validate_structure(data: dict[str, Any]) -> None:
    """Catch model-independent mistakes before weights or CCD features are loaded."""
    name = data.get("name")
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or Path(name).name != name
        or "\\" in name
    ):
        message = "Input name must be a nonempty target name, not a path"
        raise InputError(message)
    if not data["sequences"]:
        message = "Inputs must contain at least one chain"
        raise InputError(message)
    used = set()
    for chain in data["sequences"]:
        _validate_chain(chain, used)


def _validate_chain(chain: Any, used: set[str]) -> None:
    """Validate chain identities and released AF3 polymer/ligand syntax."""
    if (
        not isinstance(chain, dict)
        or len(chain) != 1
        or next(iter(chain)) not in {"protein", "rna", "dna", "ligand"}
    ):
        message = "Each chain must contain exactly one protein, rna, dna or ligand"
        raise InputError(message)
    kind, body = next(iter(chain.items()))
    if not isinstance(body, dict):
        message = "A chain body must be a mapping"
        raise InputError(message)
    ids = body.get("id")
    ids = [ids] if isinstance(ids, str) else ids
    if (
        not isinstance(ids, list)
        or not ids
        or any(
            not isinstance(i, str) or not i or not i.isalpha() or not i.isupper()
            for i in ids
        )
    ):
        message = "Chain ids must be nonempty uppercase alphabetic strings"
        raise InputError(message)
    if any(i in used for i in ids) or len(set(ids)) != len(ids):
        message = "Chain ids must be unique across the target"
        raise InputError(message)
    used.update(ids)
    if kind == "ligand":
        valid = bool(body.get("ccdCodes")) != bool(body.get("smiles"))
    else:
        sequence = body.get("sequence")
        valid = (
            isinstance(sequence, str)
            and bool(sequence)
            and sequence.isalpha()
            and sequence.isascii()
            and sequence.isupper()
        )
    if not valid:
        message = f"Invalid {kind} sequence; ligands require either ccdCodes or smiles"
        raise InputError(message)


def read_inputs(
    source: Inputs | Input | Mapping[str, Any] | str | Path, base_dir: str | Path | None
) -> tuple[Input | None, Any, Path]:
    """Read supported source forms without writing intermediate adapter files."""
    from foldforge.data.inputs.build import load

    if isinstance(source, Inputs):
        return source.target, deepcopy(source.payload), source.origin
    origin = Path(base_dir or Path.cwd()).expanduser().resolve() / "input.json"
    if isinstance(source, Input):
        return source, None, origin
    if isinstance(source, Mapping):
        return None, deepcopy(dict(source)), origin
    origin = Path(source).expanduser().resolve()
    if origin.suffix.lower() == ".json":
        return None, json.loads(origin.read_text()), origin
    return load(origin), None, origin


def load_inputs(
    source: Inputs | Input | Mapping[str, Any] | str | Path,
    *,
    base_dir: str | Path | None = None,
) -> Inputs:
    """Load and validate reusable input resources independently of a model."""
    if isinstance(source, Inputs):
        return source
    target, data, origin = read_inputs(source, base_dir)
    if target is not None:
        chains = tuple(
            replace(chain, a3m_text=target.msa_text(chain))
            if chain.a3m is not None and chain.alignment is None
            else chain
            for chain in target.chains
        )
        return Inputs(
            replace(target, chains=chains)
            if any(a is not b for a, b in zip(chains, target.chains, strict=True))
            else target,
            None,
            origin,
        )
    data = normalize_payload(data)
    from alphafold3.common import folding_input

    try:
        parsed = folding_input.Input.from_json(
            json.dumps({**data, "modelSeeds": data.get("modelSeeds") or [0]}),
            json_path=origin,
        )
    except (ValueError, KeyError, TypeError) as error:
        raise InputError(str(error)) from error
    return Inputs(None, json.loads(parsed.fill_missing_fields().to_json()), origin)
