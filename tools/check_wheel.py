"""Verify a built FoldForge wheel's public imports, types and dependency metadata."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from email.parser import Parser
from pathlib import Path
from typing import get_type_hints


def main() -> int:
    """Check metadata and optionally an installed wheel in an isolated prefix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args()
    with zipfile.ZipFile(args.wheel) as wheel:
        metadata_path = next(
            p for p in wheel.namelist() if p.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(wheel.read(metadata_path).decode())
        requires = metadata.get_all("Requires-Dist", [])
        assert any(r.startswith("miniworld-engine==2.3.12") for r in requires)
        assert any(r.startswith("foldforge-af3-data==3.0.2") for r in requires)
        assert "foldforge/py.typed" in wheel.namelist()
        assert "weights" in metadata.get_all("Provides-Extra", [])
        for name in (
            "sources.json",
            "conversions.lock",
            "download.py",
            "convert.py",
            "commands.py",
        ):
            assert f"foldforge/models/checkpoints/{name}" in wheel.namelist()
    if args.installed:
        import foldforge as ff

        assert Path(ff.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        for function in (
            ff.load_model,
            ff.load_inputs,
            ff.predict,
            ff.Predictor.run_trunk,
            ff.Predictor.sample_diffusion,
            ff.PredictionResult.save,
        ):
            assert get_type_hints(function)
        loaded = ff.load_inputs(ff.protein("AG"))
        assert loaded.payload["sequences"][0]["protein"]["sequence"] == "AG"
        assert ff.Config().trunk.recycles is None
    print(  # noqa: T201 - command output
        json.dumps(
            {
                "wheel": str(args.wheel),
                "metadata": "passed",
                "installed": args.installed,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
