"""One persistence contract for every checkpoint's decoded output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from foldforge.models.io.paths import run_directory

if TYPE_CHECKING:
    from foldforge.prediction import Prediction


def cpu_tree(value: Any) -> Any:
    """Compute cpu tree."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu_tree(v) for v in value]
    return value


def numpy_tree(value: Any) -> Any:
    """Compute numpy tree."""
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    if isinstance(value, dict):
        return {k: numpy_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [numpy_tree(v) for v in value]
    return value


def json_value(value: Any) -> Any:  # noqa: PLR0911 - explicit JSON value variants
    """Compute json value."""
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def cpu_payload(prediction: Prediction) -> dict[str, torch.Tensor | None]:
    """Absent heads stay None; no model-class pickle is needed to read results."""
    return {f.name: cpu_tree(getattr(prediction, f.name)) for f in fields(prediction)}


def target_name(name: str) -> str:
    """Do not silently rename a target or let it escape its output directory."""
    if not name or name in {".", ".."} or Path(name).name != name or "\\" in name:
        msg = "input name must identify a target, not a path"
        raise ValueError(msg)
    return name


@dataclass
class Decoded:
    """Represent decoded."""

    name: str
    prediction: Prediction
    cifs: list[str]
    report: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    arrays: dict[str, Any] | None = None
    # Keep existing CLI filenames; the manifest always lists every sample.
    singleton_cif: bool = False
    image_inputs: dict[str, Any] = field(default_factory=dict)
    image_distogram: Any = None


def write_output(
    result: Decoded,
    directory: Path,
    *,
    expected_samples: int,
    images: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Validate before writing and emit the same manifest for every layout."""
    directory = run_directory(directory)
    name = target_name(result.name)
    prediction = result.prediction
    if prediction.n_samples != expected_samples or len(result.cifs) != expected_samples:
        msg = "Decoded sample count differs from the requested count"
        raise ValueError(msg)
    payload = cpu_payload(prediction)
    for key, value in payload.items():
        if value is not None and not torch.isfinite(value).all():
            msg = f"non-finite {key}"
            raise FloatingPointError(msg)
    plddt = payload["plddt"]
    if plddt is not None and ((plddt < 0).any() or (plddt > 1).any()):
        msg = "Prediction pLDDT must use the [0, 1] scale"
        raise ValueError(msg)
    if any(not isinstance(cif, str) or not cif.strip() for cif in result.cifs):
        msg = "Each decoded sample must provide a nonempty mmCIF"
        raise ValueError(msg)
    paths = [
        directory
        / (
            f"{name}.cif"
            if expected_samples == 1 and result.singleton_cif
            else f"{name}-{i}.cif"
        )
        for i in range(expected_samples)
    ]
    report = {
        **result.report,
        "prediction_cif": str(paths[0]),
        "prediction_cifs": [str(p) for p in paths],
        "prediction_schema": 1,
        "confidence_units": {
            "plddt": "0..1",
            "cif_b_factor": "0..100",
            "pae": "angstrom",
            "pde": "angstrom",
        },
    }
    # Reject invalid report values before creating any prediction artifact.
    report_text = json.dumps(json_value(report), indent=2, allow_nan=False) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(payload, directory / f"{name}.prediction.pt")
    for path, cif in zip(paths, result.cifs, strict=True):
        path.write_text(cif)
    if result.raw is not None:
        torch.save(cpu_tree(result.raw), directory / f"{name}.pt")
    if result.arrays is not None:
        np.savez_compressed(directory / f"{name}.npz", **result.arrays)
    if images:
        from foldforge.models.io.images import write_images  # noqa: PLC0415

        report["images"] = write_images(result, directory, images)
        report_text = json.dumps(json_value(report), indent=2, allow_nan=False) + "\n"
    (directory / f"{name}.json").write_text(report_text)
    return report
