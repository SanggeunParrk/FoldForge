# GPU-specific dependencies are loaded at this inference boundary.
# ruff: noqa: PLC0415
"""Run released ESMFold2, recording precision, execution and structure metrics."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from team_gm.modules.exceptions import ImplementationType

from foldforge import checkpoints
from foldforge.data.ccd import CCDDatabase, default_path
from foldforge.eval.structure import deposit_ca, score_against_deposit
from foldforge.models import get_model
from foldforge.models.esmfold2 import compute_lm_hidden_states, load_esmc
from foldforge.models.esmfold2.features import (
    ESMFold2InputBuilder,
    build_input,
    model_kwargs,
)
from foldforge.models.esmfold2.precision import inference_precision
from foldforge.prediction import Prediction

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
    # Distogram representatives are CB for most proteins, not CA.
    names = features["ref_atom_name_chars"].reshape(-1, 4).cpu()
    is_ca = (names == torch.tensor([35, 33, 0, 0])).all(-1)
    is_ca &= features["atom_attention_mask"].reshape(-1).cpu().bool()
    atom_indices = is_ca.nonzero().flatten().tolist()
    atom_to_token = features["atom_to_token"].reshape(-1).cpu()
    positions = coords[0].float().cpu()
    asym = features["asym_id"].long().reshape(-1).cpu()
    numbers = features["residue_index"].long().reshape(-1).cpu()

    sequences = {}
    for position, chain in enumerate(manifest["chains"]):
        if chain["type"] != "protein":
            continue
        sequences[position] = chain["sequence"]

    out: dict[tuple[str, int], tuple[str, tuple[float, ...]]] = {}
    seen: dict[int, int] = {}
    for atom in atom_indices:
        token = int(atom_to_token[atom])
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
            tuple(positions[atom].tolist()),
        )
    return out


def confidence_json(value: torch.Tensor | None) -> float | list[float] | None:
    """Keep single-sample reports scalar and retain every multi-sample score."""
    if value is None:
        return None
    if value.numel() == 1:
        return float(value.item())
    return value.detach().float().cpu().reshape(-1).tolist()


LM_HIDDEN_RANK = 4


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0915  # noqa: PLR0915
    """Run one complete checkpoint correctness check and save its artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccd-db", type=Path, default=default_path())
    parser.add_argument("--input-spec", type=Path)
    parser.add_argument("--lm-cache", type=Path)
    parser.add_argument("--recycles", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--target", default="1ubq")
    parser.add_argument("--data-root", default="validation/data")
    parser.add_argument("--out", default="validation/results/foldforge")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--lm-checkpoint", default=None)
    parser.add_argument(
        "--lm-source",
        choices=("compute", "cache"),
        default="compute",
        help="Run ESMC-6B, or explicitly reuse the target's saved embeddings",
    )
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
    parser.add_argument("--max-graphs", type=int, default=4)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--execution-scope", choices=("denoiser", "model"), default="denoiser"
    )
    parser.add_argument("--benchmark-repeats", type=int, default=0)
    args = parser.parse_args(argv)
    if any(
        value is not None and value < 1
        for value in (args.samples, args.recycles, args.steps, args.msa_depth)
    ):
        parser.error("samples, recycles, steps and msa-depth must be positive")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    checkpoint = (
        Path(args.checkpoint) if args.checkpoint else checkpoints.resolve("esmfold2")
    )
    sample = Path(args.data_root) / args.target

    if args.input_spec:
        from foldforge.data.inference.build import load as load_input

        target = load_input(args.input_spec)
        if target.spec.ccd_db.resolve() != args.ccd_db.resolve():
            message = "Spec and selected CCD database disagree"
            raise ValueError(message)
        structure_input = target.esmfold2(args.msa_depth)
        manifest = target.manifest()
    else:
        structure_input = build_input(sample, args.msa_depth)
        manifest = json.loads((sample / "target.json").read_text())
    database = CCDDatabase(args.ccd_db)
    builder = ESMFold2InputBuilder(ccd_db=database)
    features, chain_infos = builder.prepare_input(
        structure_input, seed=args.seed, device=device
    )
    lm_seconds = None
    if args.lm_source == "cache":
        lm_path = args.lm_cache or sample / "cache" / "esmc_hidden_states.pt"
        lm_hidden = torch.load(lm_path, map_location=device, weights_only=True).to(
            dtype
        )
    else:
        lm_path = (
            Path(args.lm_checkpoint)
            if args.lm_checkpoint
            else checkpoints.resolve("esmc-6b")
        )
        logger.info("Loading ESMC from %s", lm_path)
        language_model = load_esmc(str(lm_path), device=device, dtype=dtype)
        kwargs = model_kwargs(features, dtype)
        torch.cuda.synchronize()
        lm_start = time.perf_counter()
        with torch.no_grad():
            lm_hidden = compute_lm_hidden_states(
                language_model,
                *[
                    kwargs[name]
                    for name in (
                        "residue_type",
                        "asym_id",
                        "residue_index",
                        "mol_type",
                        "mask",
                    )
                ],
            ).to(dtype)
        torch.cuda.synchronize()
        lm_seconds = time.perf_counter() - lm_start
        logger.info("ESMC forward completed in %.3f s", lm_seconds)
        del language_model
        torch.cuda.empty_cache()

    expected_tokens = tuple(features["token_attention_mask"].shape)
    if (
        lm_hidden.ndim != LM_HIDDEN_RANK
        or tuple(lm_hidden.shape[:2]) != expected_tokens
    ):
        message = (
            f"ESMC cache token shape {tuple(lm_hidden.shape[:2])} does not match "
            f"input token shape {expected_tokens}; recompute ESMC for this exact input"
        )
        raise ValueError(message)
    if not torch.isfinite(lm_hidden).all():
        message = "ESMC produced non-finite hidden states"
        raise RuntimeError(message)
    backend = ImplementationType(args.implementation)
    model = get_model("esmfold2")(checkpoint, implementation=backend)
    inference_precision(model, device, dtype)
    from foldforge.models.config import ExecutionConfig
    from foldforge.models.execution import Execution, measured_forward

    execution = Execution(
        model,
        "esmfold2",
        ExecutionConfig(
            compile=args.compile,
            cuda_graph=args.cuda_graph,
            scope=args.execution_scope,
            max_graphs=args.max_graphs,
            benchmark_repeats=args.benchmark_repeats,
        ),
    )
    kwargs = model_kwargs(features, dtype)

    def fold() -> object:
        return model(
            **kwargs,
            lm_hidden_states=lm_hidden,
            num_diffusion_samples=args.samples,
            num_loops=args.recycles,
            num_sampling_steps=args.steps,
            generator=torch.Generator(device=device).manual_seed(args.seed),
        )

    with torch.no_grad():
        out, measurements = measured_forward(fold, max(1, args.benchmark_repeats))
    first = measurements["model_seconds_cold"]
    warm = measurements["model_seconds_warm_median"]

    n_tokens = int(features["token_attention_mask"].shape[1])
    prediction = Prediction(
        coords=out.coords,
        plddt=out.confidence.plddt,
        pae=out.confidence.pae,
        ptm=out.confidence.ptm,
        iptm=out.confidence.iptm,
        distogram_logits=out.distogram_logits,
    )
    for name in ("coords", "plddt", "pae", "ptm"):
        value = getattr(prediction, name)
        if value is not None and not torch.isfinite(value).all():
            message = f"ESMFold2 produced non-finite {name}"
            raise RuntimeError(message)
    report = {
        "target": args.target,
        "ccd": database.describe(),
        "n_tokens": n_tokens,
        "device": torch.cuda.get_device_name(device),
        "dtype": args.dtype,
        "backend": backend.value,
        "seed": args.seed,
        "msa_depth": args.msa_depth,
        "checkpoint": str(checkpoint.resolve()),
        "lm_source": args.lm_source,
        "lm_path": str(lm_path.resolve()),
        "seconds_lm_forward": lm_seconds,
        **execution.report(),
        **measurements,
        "precision": "native dtype, FP32 norm parameters, no autocast",
        "scoring_atoms": "CA, matched by chain/residue/name",
        "num_diffusion_samples": args.samples,
        "num_sampling_steps": args.steps
        if args.steps is not None
        else model.config.structure_head.inference_num_steps,
        "num_loops": args.recycles
        if args.recycles is not None
        else model.config.num_loops,
        "seconds_first": round(first, 3),
        "seconds_warm": round(warm, 3),
        "plddt_mean": float(out.confidence.plddt.mean()),
        "ptm": confidence_json(out.confidence.ptm),
        "iptm": confidence_json(out.confidence.iptm),
        "structure_score_sample": 0,
    }

    ca_prediction = predicted_ca(out.coords, features, manifest)
    if args.compare:
        reference = deposit_ca(Path(args.compare))
        comparison = score_against_deposit(
            ca_prediction, reference, {chain: chain for chain, _ in ca_prediction}
        )
        report["compare"] = {
            "path": args.compare,
            "n_ca_reference": len(reference),
            "n_ca_ours": len(ca_prediction),
            **comparison,
        }

    experimental = (
        Path(args.experimental) if args.experimental else sample / f"{args.target}.cif"
    )
    if experimental.is_file():
        chains = sorted({c for c, _ in ca_prediction})
        chain_map = (
            dict(pair.split("=") for pair in args.chain_map.split(","))
            if args.chain_map
            else {c: c for c in chains}
        )
        report["experimental"] = {
            "path": str(experimental),
            "chain_map": chain_map,
            **score_against_deposit(
                ca_prediction,
                deposit_ca(experimental),
                chain_map,
            ),
        }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    from foldforge.models.af_prediction import cpu_payload

    torch.save(cpu_payload(prediction), out_dir / f"{args.target}.prediction.pt")
    decoded = builder.decode(
        {
            "sample_atom_coords": prediction.coords,
            "plddt": prediction.plddt,
            "ptm": prediction.ptm,
            "iptm": prediction.iptm,
            "pae": prediction.pae,
            "distogram_logits": prediction.distogram_logits,
            "residue_index": features["residue_index"],
            "entity_id": features["entity_id"],
        },
        features,
        chain_infos,
        num_diffusion_samples=args.samples,
        complex_id=args.target,
    )
    decoded_samples = decoded if isinstance(decoded, list) else [decoded]
    if len(decoded_samples) != args.samples:
        message = "Decoded sample count differs from the requested count"
        raise ValueError(message)
    cif_paths = []
    for index, decoded_sample in enumerate(decoded_samples):
        filename = (
            f"{args.target}.cif" if args.samples == 1 else f"{args.target}-{index}.cif"
        )
        cif_path = out_dir / filename
        cif_path.write_text(decoded_sample.complex.to_mmcif())
        cif_paths.append(str(cif_path))
    report["prediction_cif"] = cif_paths[0]
    report["prediction_cifs"] = cif_paths
    (out_dir / f"{args.target}.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("%s", json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
