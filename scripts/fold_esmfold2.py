"""Fold a validation target with FoldForge's ESMFold2 and check the structure.

The first question after a port is not "is it fast" but "is it the same
structure". This folds a target and scores the answer three ways:

* **pLDDT** — an absolute sanity floor. A port with a doubled residual still
  produces coordinates, but its confidence collapses.
* **RMSD to a known-good prediction** — the same target folded by the same model
  before the port, if one is passed. This is the real check.
* **RMSD to the deposited structure**, when the CA counts line up.

The acceptance bar for the RMSD is not zero. The fold is not bit-deterministic
(cuBLAS split-k, Triton atomics), and two runs of identical code on 4YX2 differ
by ~0.38 A; the engine's kernels also changed under the port. What a real
mistake looks like is not 0.5 A but 5-10 A with pLDDT falling off a cliff.

Usage (GPU node)::

    sbatch scripts/fold_esmfold2.sbatch --target 1ubq
    sbatch scripts/fold_esmfold2.sbatch --target 4yx2 \
        --compare validation/results/esmfold2-teamgm-resfuse/4yx2/prediction.cif
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from team_gm.modules.exceptions import ImplementationType

from foldforge.eval.structure import deposit_ca, score_against_deposit
from foldforge.models.esmfold2 import ESMFold2Config, convert
from foldforge.models.esmfold2 import ESMFold2Model as Model
from foldforge.models.esmfold2.features import (
    ESMFold2InputBuilder,
    build_input,
    ca_coords_from_cif,
    kabsch_rmsd,
    model_kwargs,
)

logger = logging.getLogger("fold_esmfold2")

#: One-letter to three-letter residue names. The port has no name table (its
#: residue vocabulary is 33 integer tokens), so names come from the sequence that
#: was actually folded — which is the authoritative answer to "what is residue i"
#: anyway, and lets a wrong chain map or numbering fail loudly.
THREE_LETTER = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


def predicted_ca(
    coords: torch.Tensor, features: dict, manifest: dict
) -> dict[tuple[str, int], tuple[str, tuple[float, ...]]]:
    """``(chain, residue number) -> (residue name, xyz)`` for the prediction.

    Chain letters follow the order ``build_input`` assigns them; residue numbers
    come from the feature dict, so they are whatever the input builder produced
    rather than an assumption. Non-protein chains are skipped — a ligand has no
    CA to compare.
    """
    index = features["distogram_atom_idx"].long().reshape(-1)
    representative = coords[0].float().cpu()[index.cpu()]
    asym = features["asym_id"].long().reshape(-1).cpu()
    numbers = features["residue_index"].long().reshape(-1).cpu()

    sequences = {}
    for position, chain in enumerate(manifest["chains"]):
        if chain["type"] != "protein":
            continue
        sequences[position] = chain["sequence"]

    out: dict[tuple[str, int], tuple[str, tuple[float, ...]]] = {}
    seen: dict[int, int] = {}
    for token in range(representative.shape[0]):
        chain_index = int(asym[token])
        sequence = sequences.get(chain_index)
        if sequence is None:
            continue
        offset = seen.get(chain_index, 0)
        seen[chain_index] = offset + 1
        if offset >= len(sequence):
            continue
        name = THREE_LETTER.get(sequence[offset])
        if name is None:
            continue
        letter = chr(ord("A") + chain_index)
        out[(letter, int(numbers[token]))] = (
            name,
            tuple(representative[token].tolist()),
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="1ubq")
    parser.add_argument("--data-root", default="validation/data")
    parser.add_argument("--out", default="validation/results/foldforge")
    parser.add_argument("--checkpoint", default="model_checkpoints/esmfold2")
    parser.add_argument("--msa-depth", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument(
        "--implementation", default=ImplementationType.MINIWORLD_ENGINE.value
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="mmCIF of a known-good prediction for the same target and seed",
    )
    parser.add_argument(
        "--experimental",
        default=None,
        help="deposited mmCIF; defaults to <data-root>/<target>/<target>.cif",
    )
    parser.add_argument(
        "--chain-map",
        default=None,
        help="predicted=deposited chain pairs, e.g. A=A,B=H,C=L (default: identity)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    checkpoint = Path(args.checkpoint)
    sample = Path(args.data_root) / args.target

    builder = ESMFold2InputBuilder(ccd_cache=checkpoint)
    features, _ = builder.prepare_input(
        build_input(sample, args.msa_depth), seed=args.seed, device=device
    )
    lm_hidden = torch.load(
        sample / "cache" / "esmc_hidden_states.pt", map_location=device
    ).to(dtype)

    config = ESMFold2Config.from_json(checkpoint)
    backend = ImplementationType(args.implementation)
    model = Model(config, backend).to(device=device, dtype=dtype).eval()
    model.load_state_dict(
        convert.convert_model(load_file(checkpoint / "model.safetensors"), config)
    )
    kwargs = model_kwargs(features, dtype)

    def fold() -> object:
        return model(
            **kwargs,
            lm_hidden_states=lm_hidden,
            num_diffusion_samples=1,
            generator=torch.Generator(device=device).manual_seed(args.seed),
        )

    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fold()
        torch.cuda.synchronize()
        first = time.perf_counter() - start
        # Second fold: the first pays Triton autotune, which on an un-tuned GPU
        # dwarfs the fold and describes the tuner rather than the model.
        start = time.perf_counter()
        out = fold()
        torch.cuda.synchronize()
        warm = time.perf_counter() - start

    n_tokens = int(features["token_attention_mask"].shape[1])
    report = {
        "target": args.target,
        "n_tokens": n_tokens,
        "device": torch.cuda.get_device_name(device),
        "dtype": args.dtype,
        "backend": backend.value,
        "seconds_first": round(first, 3),
        "seconds_warm": round(warm, 3),
        "plddt_mean": float(out.confidence.plddt.mean()),
        "ptm": float(out.confidence.ptm),
        "iptm": float(out.confidence.iptm) if out.confidence.iptm is not None else None,
    }

    if args.compare:
        reference = ca_coords_from_cif(Path(args.compare))
        # Representative (CA) atoms only, so the comparison is against the CIF's
        # CA rows rather than every atom.
        index = features["distogram_atom_idx"].long().reshape(-1)
        ours = out.coords[0].float().cpu()[index.cpu()]
        report["compare"] = {
            "path": args.compare,
            "n_ca_reference": int(reference.shape[0]),
            "n_ca_ours": int(ours.shape[0]),
            "rmsd": kabsch_rmsd(ours, reference)
            if reference.shape == ours.shape
            else None,
        }

    experimental = (
        Path(args.experimental) if args.experimental else sample / f"{args.target}.cif"
    )
    if experimental.is_file():
        manifest = json.loads((sample / "target.json").read_text())
        chains = sorted({c for c, _ in predicted_ca(out.coords, features, manifest)})
        chain_map = (
            dict(pair.split("=") for pair in args.chain_map.split(","))
            if args.chain_map
            else {c: c for c in chains}
        )
        report["experimental"] = {
            "path": str(experimental),
            "chain_map": chain_map,
            **score_against_deposit(
                predicted_ca(out.coords, features, manifest),
                deposit_ca(experimental),
                chain_map,
            ),
        }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.target}.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("%s", json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
