"""Fold one reference input with FoldForge under the references' conditions.

Usage: fold_ours.py FAMILY TARGET SEED RUN_ROOT [--precision P] [-- extra config]
Writes <RUN_ROOT>/<family>/<target>/seed<k>/ (RUN_ROOT is relative to runs/),
the tree ``foldforge validate`` reads.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from foldforge.cli import main as foldforge  # noqa: E402
from foldforge.eval.references import FAMILIES, OUTPUTS  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("family", choices=list(FAMILIES))
parser.add_argument("target")
parser.add_argument("seed", type=int)
parser.add_argument("run_root")
parser.add_argument("--precision", default="af3_default")
parser.add_argument("--mode", choices=("fast", "exact"), default="fast")
args = parser.parse_args()
family = FAMILIES[args.family]
config = {
    "backend": "pytorch",
    "precision": args.precision,
    "mode": args.mode,
    "trunk_seed": args.seed,
    "diffusion_seed": args.seed,
    "trunk": {"recycles": family["recycles"], "msa_depth": 16384},
    "diffusion": {"steps": 200},
    "execution": {"compile": False, "cuda_graph": False, "bucketing": True},
}
if "variant" in family:
    config["variant"] = family["variant"]
with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
    yaml.safe_dump(config, fh)
out = f"{args.run_root}/{args.family}/{args.target}/seed{args.seed}"
raise SystemExit(
    foldforge(
        [
            "fold",
            family["model"],
            "--spec",
            str(OUTPUTS / "inputs" / args.target / "foldforge.yaml"),
            "--config",
            fh.name,
            "--out",
            out,
        ]
    )
)
