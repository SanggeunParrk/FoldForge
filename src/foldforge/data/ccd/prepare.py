# Native AF3 feature dependencies are loaded only for preparation.
# ruff: noqa: PLC0415
"""Prepare the shared CCD database once, outside model weight directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from .build import prepare


def main(argv: list[str] | None = None) -> int:
    """Prepare or verify a local shared database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--components", type=Path)
    parser.add_argument("--rdkit", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ccd-db", type=Path)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        if any(v is None for v in (args.components, args.rdkit, args.out)):
            parser.error("prepare requires --components, --rdkit and --out")
        prepare(args.components, args.rdkit, args.out)
    else:
        from .database import CCDDatabase, default_path

        CCDDatabase(args.ccd_db or default_path()).verify()
    return 0
