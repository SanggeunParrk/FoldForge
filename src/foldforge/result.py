"""Public prediction results, independent of CLI parsing and automatic persistence."""
# ruff: noqa: PLC0415 - output/Torch dependencies stay behind result operations

# ruff: noqa: TC001, TC002, TC003 - public runtime type introspection

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from foldforge.models.io.output import Decoded
from foldforge.prediction import Prediction


@dataclass(frozen=True)
class PredictionResult:
    """One target's CPU prediction, mmCIF strings and execution metadata."""

    name: str
    prediction: Prediction
    cifs: tuple[str, ...]
    report: dict[str, Any]
    _decoded: Decoded = field(repr=False, compare=False)
    output_dir: Path | None = None
    _images: tuple[str, ...] = field(default=(), repr=False)

    @property
    def coords(self) -> torch.Tensor:
        """Coordinates in angstroms, shaped [samples, atoms, 3]."""
        return self.prediction.coords

    @property
    def plddt(self) -> torch.Tensor | None:
        """Per-token confidence on [0, 1], when the model provides it."""
        return self.prediction.plddt

    @classmethod
    def from_decoded(
        cls, decoded: Decoded, *, expected_samples: int, images: tuple[str, ...] = ()
    ) -> PredictionResult:
        """Detach outputs from model/graph storage and validate without writing."""
        import json

        from foldforge.models.io.output import (
            cpu_tree,
            json_value,
            prediction_report,
            validate_decoded,
        )
        from foldforge.prediction import Prediction

        prediction = Prediction(**validate_decoded(decoded, expected_samples))
        report = json_value(prediction_report(decoded))
        json.dumps(report, allow_nan=False)
        detached = replace(
            decoded,
            prediction=prediction,
            report=report,
            raw=cpu_tree(decoded.raw),
            arrays=cpu_tree(decoded.arrays),
            image_distogram=cpu_tree(decoded.image_distogram),
        )
        return cls(
            decoded.name,
            prediction,
            tuple(decoded.cifs),
            report,
            detached,
            _images=images,
        )

    def save(self, directory: str | Path) -> PredictionResult:
        """Save standard prediction artifacts to a caller-owned directory."""
        from foldforge.models.io.output import write_output
        from foldforge.models.io.paths import output_directory

        path = output_directory(directory)
        report = write_output(
            replace(self._decoded, report=dict(self.report)),
            path,
            expected_samples=self.prediction.n_samples,
            images=self._images,
            within_runs=False,
        )
        return replace(self, report=report, output_dir=path)
