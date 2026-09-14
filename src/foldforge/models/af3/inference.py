# GPU dependencies are imported at the model boundary so registry/help stay cheap.
# ruff: noqa: PLC0415
"""AF3 official input features and native BF16 PyTorch inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foldforge.data.ccd import CCDDatabase


def main(argv: list[str] | None = None) -> int:
    """Run the model-specific command."""
    p = argparse.ArgumentParser(prog="foldforge fold af3")
    p.add_argument(
        "--input", type=Path, required=True, help="AF3 JSON with prepared MSA/templates"
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    from foldforge.data.ccd import CCDDatabase, default_path

    p.add_argument("--ccd-db", type=Path, default=default_path())
    p.add_argument("--backend", choices=("pytorch", "miniworld"), default="miniworld")
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--recycles", type=int, default=10)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--max-graphs", type=int, default=4)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--cuda-graph", action="store_true")
    p.add_argument(
        "--execution-scope", choices=("denoiser", "model"), default="denoiser"
    )
    p.add_argument("--benchmark-repeats", type=int, default=0)
    args = p.parse_args(argv)
    if min(args.recycles, args.steps, args.samples) < 1:
        p.error("recycles, steps and samples must be positive")
    database = CCDDatabase(args.ccd_db)
    from alphafold3.constants import chemical_component_sets

    with (
        database.activate(),
        chemical_component_sets.use_ccd_sets(database.chemical_component_sets),
    ):
        return _run(args, database)


def _run(args: argparse.Namespace, database: CCDDatabase) -> int:  # noqa: PLR0915
    import numpy as np
    import torch
    from alphafold3.common import folding_input
    from alphafold3.data import featurisation
    from alphafold3.model import feat_batch
    from alphafold3.model.atom_layout import atom_layout

    from foldforge.models.af3.model import load
    from foldforge.models.af_prediction import cpu_payload, from_af3

    args.out.mkdir(parents=True, exist_ok=True)
    text = args.input.read_text()
    fold_input = folding_input.Input.from_json(text, json_path=args.input)
    ccd = database.af3_ccd(user_ccd=fold_input.user_ccd)
    examples = featurisation.featurise_input(
        fold_input, ccd, buckets=None, verbose=True
    )
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model = load(
        args.checkpoint,
        backend=args.backend,
        dtype=dtype,
        recycles=args.recycles,
        samples=args.samples,
        steps=args.steps,
    )
    from foldforge.models.config import ExecutionConfig
    from foldforge.models.execution import Execution, measured_forward

    execution = Execution(
        model,
        "af3",
        ExecutionConfig(
            compile=args.compile,
            cuda_graph=args.cuda_graph,
            scope=args.execution_scope,
            max_graphs=args.max_graphs,
            benchmark_repeats=args.benchmark_repeats,
        ),
    )
    print("Loaded", model.foldforge_load_report, flush=True)  # noqa: T201
    for seed, example in zip(fold_input.rng_seeds, examples, strict=True):
        torch.manual_seed(seed)
        start = time.monotonic()
        tensors = {
            k: torch.from_numpy(v).to("cuda")
            for k, v in example.items()
            if isinstance(v, np.ndarray) and v.dtype.kind in "biuf"
        }
        tensors["deletion_mean"] = tensors["deletion_mean"].float()
        with torch.inference_mode():
            result, measurements = measured_forward(
                lambda tensors=tensors: model(tensors), args.benchmark_repeats
            )
        torch.cuda.synchronize()
        cpu = {}

        def numpy_tree(x: torch.Tensor) -> Any:
            if isinstance(x, torch.Tensor):
                return x.detach().float().cpu().numpy()
            if isinstance(x, dict):
                return {k: numpy_tree(v) for k, v in x.items()}
            return x

        cpu = numpy_tree(result)
        positions = cpu["diffusion_samples"]["atom_positions"]
        if not np.isfinite(positions).all():
            msg = "non-finite AF3 coordinates"
            raise FloatingPointError(msg)
        batch = feat_batch.Batch.from_data_dict(example)
        layout = batch.convert_model_output
        gather = atom_layout.compute_gather_idxs(
            source_layout=layout.token_atoms_layout,
            target_layout=layout.flat_output_layout,
        )
        if not np.all(gather.gather_mask):
            msg = "AF3 output contains unmapped atoms"
            raise ValueError(msg)
        coords = atom_layout.convert(
            gather_info=gather, arr=positions, layout_axes=(-3, -2)
        )
        plddt = atom_layout.convert(
            gather_info=gather, arr=cpu["predicted_lddt"], layout_axes=(-2, -1)
        )
        name = Path(fold_input.name).name
        canonical = from_af3(
            result, torch.from_numpy(coords), tensors["pred_dense_atom_mask"]
        )
        torch.save(
            cpu_payload(canonical), args.out / f"{name}-seed{seed}.prediction.pt"
        )
        for i, xyz in enumerate(coords):
            structure = layout.empty_output_struc.copy_and_update_atoms(
                atom_x=xyz[:, 0],
                atom_y=xyz[:, 1],
                atom_z=xyz[:, 2],
                atom_b_factor=plddt[i],
                atom_occupancy=np.ones(len(xyz)),
            )
            (args.out / f"{name}-seed{seed}-{i}.cif").write_text(structure.to_mmcif())
        np.savez_compressed(
            args.out / f"{name}-seed{seed}.npz",
            coordinates=coords,
            plddt=plddt,
            **{k: v for k, v in cpu.items() if isinstance(v, np.ndarray)},
        )
        report = {
            "model": "af3",
            "ccd": database.describe(),
            "backend": args.backend,
            "precision": args.precision,
            "autocast": False,
            **execution.report(),
            **measurements,
            "seed": seed,
            "recycles": args.recycles,
            "steps": args.steps,
            "samples": args.samples,
            "checkpoint": model.foldforge_load_report,
            "elapsed_seconds": time.monotonic() - start,
            "mean_plddt": float(plddt.mean()),
            "confidence_fields": [
                k for k, v in cpu.items() if isinstance(v, np.ndarray)
            ],
        }
        (args.out / f"{name}-seed{seed}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(report, flush=True)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
