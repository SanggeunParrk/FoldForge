"""Convert a released AF3-family checkpoint into the AF3 parameter blob.

Offline provenance step, not part of inference. The key maps are the Apache-2.0
converters of sokrypton/alphafold3 (branch af3-any-model); pass that checkout with
--reference. They are imported, never copied, and they write through FoldForge's
own record codec, so neither this tool nor the FoldForge runtime needs Haiku or
the AF3 package.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import os
import sys
import types
from pathlib import Path

MODELS = (
    "intellifold2",
    "openfold3",
    "openbind0",
    "boltz2",
    "rosettafold3",
    "chai1",
    "protenix1",
    "protenix2",
    "opendde",
)


def _seed_reference_params() -> None:
    """Give the key maps FoldForge's record codec instead of the JAX-side module.

    The reference converters import ``alphafold3.model.params`` only for the record
    codec; importing the real module would pull in Haiku and the AF3 package.
    """
    from foldforge.models.checkpoints import haiku

    for name in ("alphafold3", "alphafold3.model", "alphafold3.model.params"):
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module
    params = sys.modules["alphafold3.model.params"]
    params.encode_record = haiku.encode_record  # type: ignore[attr-defined]
    params.read_records = haiku.read_records  # type: ignore[attr-defined]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()
    # Full precision everywhere; the runtime's precision policy decides the rest.
    os.environ["IF2_FP32_BLOB"] = "1"
    _seed_reference_params()
    sys.path.insert(0, str(args.reference))
    converters = importlib.import_module("converters")
    result = converters.CONVERTERS[args.model](args.checkpoint, args.out)
    print(f"{args.model}: {result}")  # noqa: T201 - CLI output contract
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
