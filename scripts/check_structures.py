"""Per-sample geometry and numerical drift for the ESMFold2, Protenix and OpenDDE rows.

Writes the ``structure_checks.json`` that ``render_benchmark_results.py`` publishes:
for every mode and diffusion sample, local geometry counts plus the same-index
comparison with that model's compiled default-precision reference. Atom
correspondence uses chain, residue index, residue identity and atom name. This is
a drift and gross-geometry diagnostic, not an experimental accuracy score; AF3 has
its own report from ``compare_af3_precision.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_af3_precision import atoms, compare, geometry

MODELS = ("esmfold2", "protenix", "opendde")
REFERENCE = "pytorch_compile_reference"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result: dict[str, dict[str, list[dict]]] = {}
    for model in MODELS:
        rows = json.loads((args.results / f"e2e-{model}.json").read_text())
        if any(row["status"] for row in rows):
            message = f"Failed benchmark rows for {model}"
            raise ValueError(message)
        samples = {
            row["mode"]: [atoms(Path(p)) for p in row["report"]["prediction_cifs"]]
            for row in rows
        }
        references = samples[REFERENCE]
        result[model] = {}
        for mode, predicted in samples.items():
            if len(predicted) != len(references):
                message = f"Different sample count: {model}/{mode}"
                raise ValueError(message)
            checks = []
            for index, (sample, reference) in enumerate(
                zip(predicted, references, strict=True)
            ):
                if set(sample) != set(reference):
                    message = f"Different atom identities: {model}/{mode}/{index}"
                    raise ValueError(message)
                checks.append(
                    {
                        "sample": index,
                        "geometry": geometry(sample),
                        "vs_reference": compare(sample, reference),
                    }
                )
            result[model][mode] = checks
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
