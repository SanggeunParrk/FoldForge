"""Diff an AF3-format parameter blob against a dense-graph family's expected tree.

Porting aid: lists records the graph lacks, records the blob lacks, and shared
records whose element counts differ. A family is ready when all three are empty.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import torch

from foldforge.models.checkpoints import haiku
from foldforge.modules.dense.spec import SPECS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blob", type=Path, required=True)
    parser.add_argument("--family", required=True, choices=sorted(SPECS))
    parser.add_argument("--name", help="report under this family name")
    args = parser.parse_args()
    from foldforge.models.architectures.af3 import AlphaFold3

    spec = SPECS[args.family]
    if args.name:
        spec = replace(spec, family=args.name)
    with torch.device("meta"):
        model = AlphaFold3(spec=spec)
    params = haiku.get_alphafold3_params(args.blob)
    flat = haiku._process_translations_dict(  # noqa: SLF001 - porting aid
        d=haiku.get_translation_dict(model), _key_prefix="diffuser/"
    )
    skip = ("__meta__/", "fourier_embedding_")
    blob = {k for k in params if not any(s in k for s in skip)}
    short = lambda key: key.replace("diffuser/", "")
    print(f"blob {len(blob)} graph {len(flat)}")  # noqa: T201
    for key in sorted(set(flat) - blob):
        print("GRAPH ONLY", short(key))  # noqa: T201
    for key in sorted(blob - set(flat)):
        print("BLOB ONLY ", short(key), tuple(params[key].shape))  # noqa: T201
    for key in sorted(set(flat) & blob):
        param = flat[key]
        targets = param.param if param.stacked else [param.param]
        if math.prod(params[key].shape) != sum(t.numel() for t in targets):
            shape = (len(targets), *targets[0].shape)
            print("SIZE      ", short(key), tuple(params[key].shape), shape)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
