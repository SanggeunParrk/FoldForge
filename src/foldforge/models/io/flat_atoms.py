# Optional chemistry/GPU backends load at the selected execution boundary.
# ruff: noqa: PLC0415
"""Checkpoint tensor-layout conversion for the common inference runtime."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from foldforge.models.execution import copy_containers
from foldforge.models.io.output import Decoded, json_value, target_name
from foldforge.models.io.runtime import Case, Runtime, to_device
from foldforge.utils.seed import seed_all

if TYPE_CHECKING:
    from collections.abc import Iterator

    from biotite.structure import AtomArray

    from foldforge.data.ccd import CCDDatabase
    from foldforge.models.io.request import Request


def prepare(args: Request, database: CCDDatabase, runtime: Runtime) -> Iterator[Case]:  # noqa: PLR0915 - case tensors stay alive through the decode closure
    """Prepare ."""
    import io

    import numpy as np
    import torch
    from biotite.structure.io import pdbx

    from foldforge.models.io.confidence import (
        from_atom_confidence,
        structure_with_confidence,
        summary_for_json,
    )

    if args.out is None or args.input is None:
        message = "The checkpoint adapter requires input and output paths"
        raise ValueError(message)

    model_name = args.model
    from foldforge.models import load
    from foldforge.models.config import configuration

    config = configuration(model_name)
    config.input_json_path = str(args.input.resolve())
    config.dump_dir = str(args.out.resolve())
    if args.guidance is not None:
        config.sample_diffusion.guidance.enable = args.guidance
    config.use_template = args.templates
    config.use_msa = not args.no_msa
    config.model.N_cycle = args.recycles
    config.model.N_model_seed = 1
    config.sample_diffusion.N_step = args.steps
    config.sample_diffusion.N_sample = args.samples
    config.num_workers = 0
    config.data.ccd_components_file = str(database.config.ccd_db)
    config.data.ccd_components_rdkit_mol_file = str(database.config.ccd_db)
    from foldforge.data.inputs.dataset import StructuralInferenceDataset

    # OpenDDE is the last model on this path; every other predictor reads its
    # inputs through the dense layout.
    dataset_type = {"opendde": StructuralInferenceDataset}[model_name]
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = dataset_type(config)
    model = load(
        model_name,
        args.checkpoint,
        configs=config,
        backend=args.backend,
        dtype=dtype,
        precision_policy=args.precision,
    )
    runtime.bind(model)

    print("Loaded", model.foldforge_load_report, flush=True)  # noqa: T201
    for idx in range(len(dataset)):
        seed_all(args.trunk_seed)
        start = time.monotonic()
        data, atoms, error = dataset[idx]
        if error:
            raise RuntimeError(error)
        features = to_device(data["input_feature_dict"], "cuda")

        from foldforge.models.io.images import input_images

        image_inputs = input_images(
            data["input_feature_dict"], args.output.image_names, gap_id=31
        )

        def decode(
            output: tuple[dict[str, Any], Any, dict[str, Any]],
            _measurements: dict[str, float],
            *,
            atoms: AtomArray = atoms,
            data: dict[str, Any] = data,
            features: dict[str, torch.Tensor] = features,
            idx: int = idx,
            start: float = start,
            image_inputs: dict[str, Any] = image_inputs,
        ) -> Decoded:
            prediction, _, logs = output
            canonical = from_atom_confidence(prediction)
            positions = canonical.coords.detach().float().cpu().numpy()
            if not np.isfinite(positions).all():
                msg = "non-finite coordinates"
                raise FloatingPointError(msg)
            name = str(
                data.get(
                    "sample_name", dataset.inputs[idx].get("name", f"sample-{idx}")
                )
            )
            # Input names identify a target; they are not output directory paths.
            name = target_name(name)
            if name in {"", ".", ".."}:
                msg = "input name must identify a target"
                raise ValueError(msg)
            cifs = []
            for sample, coordinate in enumerate(positions):
                structure = structure_with_confidence(
                    atoms, coordinate, prediction["full_data"][sample]["atom_plddt"]
                )
                cif = pdbx.CIFFile()
                pdbx.set_structure(cif, structure)
                stream = io.StringIO()
                cif.write(stream)
                cifs.append(stream.getvalue())

            from team_gm.modules.bucketing import BucketShape

            shape = (
                BucketShape.select(
                    features["residue_index"].shape[-1],
                    features["ref_pos"].shape[-2],
                    features["msa"].shape[-2]
                    if features.get("msa") is not None and features["msa"].ndim >= 2  # noqa: PLR2004 - tensor rank or format cardinality
                    else None,
                )
                if args.execution.bucketing
                else None
            )
            report = {
                "buckets": None if shape is None else vars(shape),
                "bucket_scope": (
                    "trunk, sampled MSA, confidence pairformer, denoiser; "
                    "sampler/output use real atoms"
                )
                if shape is not None
                else None,
                "ccd": database.describe(),
                "trunk_seed": args.trunk_seed,
                "diffusion_seed": args.diffusion_seed,
                "recycles": args.recycles,
                "steps": args.steps,
                "samples": args.samples,
                "checkpoint": model.foldforge_load_report,
                "elapsed_seconds": time.monotonic() - start,
                "confidence": summary_for_json(
                    prediction.get("summary_confidence", [])
                ),
                "logs": json_value(logs),
            }
            return Decoded(
                name,
                canonical,
                cifs,
                report,
                raw={
                    key: value
                    for key, value in prediction.items()
                    if key != "distogram_logits"
                },
                image_inputs=image_inputs,
                image_distogram=prediction.get("distogram_logits"),
            )

        yield Case(
            lambda features=features: model(
                copy_containers(features), None, None, mode="inference"
            ),
            decode,
            data["input_feature_dict"],
            f"features-{idx}.pt",
        )
