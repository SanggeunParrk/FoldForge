"""Resolve every structure-prediction destination inside the repository runs tree."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from foldforge.models.checkpoints import ROOT

RUNS_ROOT = ROOT / "runs"


def run_directory(
    value: str | Path | None = None, *, model: str = "prediction"
) -> Path:
    """Resolve a run name or contained absolute path without creating directories.

    ``example`` and ``runs/example`` both name ``<repo>/runs/example``.
    An omitted destination receives a unique directory under the model name.
    Traversal and directory symlinks cannot redirect output outside runs.
    """
    root = RUNS_ROOT.resolve()
    if value is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        value = Path(model) / f"{stamp}-{uuid4().hex[:8]}"
    selected = Path(value).expanduser()
    if not selected.is_absolute():
        if selected.parts and selected.parts[0] == "runs":
            selected = Path(*selected.parts[1:])
        selected = root / selected
    selected = selected.resolve()
    if not selected.is_relative_to(root):
        message = f"Prediction output must be inside {root}; got {value}"
        raise ValueError(message)
    return selected
