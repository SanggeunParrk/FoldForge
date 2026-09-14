# GPU dependencies are imported at the model boundary so registry/help stay cheap.
# ruff: noqa: PLC0415
"""Protenix/OpenDDE inference in the FoldForge environment."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foldforge.data.ccd import CCDDatabase


def json_value(value: Any) -> Any:
    """Convert decoded tensors into JSON-compatible confidence values."""
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def run(model_name: str, argv: list[str] | None = None) -> int:
    """Prepare input features, predict coordinates and save decoded confidence."""
    parser = argparse.ArgumentParser(prog=f"foldforge fold {model_name}")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Model input JSON with sequences and optional prepared MSAs/templates",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    from foldforge.data.ccd import CCDDatabase, default_path

    parser.add_argument("--ccd-db", type=Path, default=default_path())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("miniworld", "pytorch"), default="miniworld"
    )
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recycles", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--templates", action="store_true")
    parser.add_argument("--no-msa", action="store_true")
    if model_name == "protenix":
        parser.add_argument(
            "--variant",
            default="protenix_base_default_v1.0.0",
            choices=(
                "protenix_base_default_v1.0.0",
                "protenix_base_20250630_v1.0.0",
                "protenix-v2",
            ),
        )
    parser.add_argument("--max-graphs", type=int, default=4)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--execution-scope", choices=("denoiser", "model"), default="denoiser"
    )
    parser.add_argument("--benchmark-repeats", type=int, default=0)
    args = parser.parse_args(argv)
    if min(args.recycles, args.steps, args.samples) < 1:
        parser.error("recycles, steps and samples must be positive")
    database = CCDDatabase(args.ccd_db)
    with database.activate():
        return _run(model_name, args, database)


def _run(model_name: str, args: argparse.Namespace, database: CCDDatabase) -> int:  # noqa: PLR0915 - sequential target runner
    import numpy as np
    import torch
    from biotite.structure.io import pdbx

    from foldforge.models.af_prediction import cpu_payload, from_atom_confidence

    model_api = importlib.import_module(f"foldforge.models.{model_name}.model")
    config = (
        model_api.configuration(args.variant)
        if model_name == "protenix"
        else model_api.configuration()
    )
    config.input_json_path = str(args.input.resolve())
    config.dump_dir = str(args.out.resolve())
    config.use_template = args.templates
    config.use_msa = not args.no_msa
    config.model.N_cycle = args.recycles
    config.model.N_model_seed = 1
    config.sample_diffusion.N_step = args.steps
    config.sample_diffusion.N_sample = args.samples
    config.num_workers = 0
    config.data.ccd_components_file = str(database.config.ccd_db)
    config.data.ccd_components_rdkit_mol_file = str(database.config.ccd_db)
    data_api = importlib.import_module(
        f"foldforge.models.{model_name}.ported.data.inference.infer_dataloader"
    )
    torch_utils = importlib.import_module(
        f"foldforge.models.{model_name}.ported.utils.torch_utils"
    )
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = data_api.InferenceDataset(config)
    model = model_api.load(
        args.checkpoint, configs=config, backend=args.backend, dtype=dtype
    )
    from foldforge.models.config import ExecutionConfig
    from foldforge.models.execution import Execution, copy_containers, measured_forward

    execution = Execution(
        model,
        model_name,
        ExecutionConfig(
            compile=args.compile,
            cuda_graph=args.cuda_graph,
            scope=args.execution_scope,
            max_graphs=args.max_graphs,
            benchmark_repeats=args.benchmark_repeats,
        ),
    )
    print("Loaded", model.foldforge_load_report, flush=True)  # noqa: T201
    for idx in range(len(dataset)):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        start = time.monotonic()
        data, atoms, error = dataset[idx]
        if error:
            raise RuntimeError(error)
        features = torch_utils.to_device(data["input_feature_dict"], "cuda")
        torch.save(data["input_feature_dict"], args.out / f"features-{idx}.pt")
        with torch.inference_mode():
            (prediction, _, logs), measurements = measured_forward(
                lambda features=features: model(
                    copy_containers(features), None, None, mode="inference"
                ),
                args.benchmark_repeats,
            )
        torch.cuda.synchronize()
        canonical = from_atom_confidence(prediction)
        positions = canonical.coords.detach().float().cpu().numpy()
        if not np.isfinite(positions).all():
            msg = "non-finite coordinates"
            raise FloatingPointError(msg)
        name = str(
            data.get("sample_name", dataset.inputs[idx].get("name", f"sample-{idx}"))
        )
        # Input names identify a target; they are not output directory paths.
        name = Path(name).name
        if name in {"", ".", ".."}:
            msg = "input name must identify a target"
            raise ValueError(msg)
        torch.save(cpu_payload(canonical), args.out / f"{name}.prediction.pt")
        for sample, coordinate in enumerate(positions):
            from foldforge.models.af_prediction import structure_with_confidence

            structure = structure_with_confidence(
                atoms, coordinate, prediction["full_data"][sample]["atom_plddt"]
            )
            cif = pdbx.CIFFile()
            pdbx.set_structure(cif, structure)
            cif.write(args.out / f"{name}-{sample}.cif")

        def cpu_tree(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {k: cpu_tree(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [cpu_tree(v) for v in value]
            return value

        torch.save(cpu_tree(prediction), args.out / f"{name}.pt")
        report = {
            "model": model_name,
            "ccd": database.describe(),
            "backend": args.backend,
            "precision": args.precision,
            "autocast": False,
            **execution.report(),
            **measurements,
            "seed": args.seed,
            "recycles": args.recycles,
            "steps": args.steps,
            "samples": args.samples,
            "checkpoint": model.foldforge_load_report,
            "elapsed_seconds": time.monotonic() - start,
            "confidence": json_value(prediction.get("summary_confidence", {})),
            "logs": json_value(logs),
        }
        (args.out / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, report["elapsed_seconds"], "seconds", flush=True)  # noqa: T201
    return 0
