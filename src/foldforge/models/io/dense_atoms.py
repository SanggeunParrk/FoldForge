# Optional chemistry/GPU backends load at the selected execution boundary.
# ruff: noqa: PLC0415
"""Checkpoint tensor-layout conversion for the common inference runtime."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from foldforge.models.io.output import Decoded, numpy_tree, target_name
from foldforge.models.io.runtime import Case, Runtime

if TYPE_CHECKING:
    from collections.abc import Iterator

    from foldforge.data.ccd import CCDDatabase
    from foldforge.models.bucketing import BucketShape
    from foldforge.models.io.request import Request


def prepare(args: Request, database: CCDDatabase, runtime: Runtime) -> Iterator[Case]:
    """Prepare ."""
    from alphafold3.constants import chemical_component_sets

    with chemical_component_sets.use_ccd_sets(database.chemical_component_sets):
        yield from _prepare(args, database, runtime)


def _prepare(args: Request, database: CCDDatabase, runtime: Runtime) -> Iterator[Case]:  # noqa: PLR0915 - case tensors stay alive through the decode closure
    import numpy as np
    import torch
    from alphafold3.common import folding_input
    from alphafold3.data import featurisation
    from alphafold3.model import feat_batch
    from alphafold3.model.atom_layout import atom_layout

    from foldforge.models import load
    from foldforge.models.io.confidence import from_af3

    if args.out is None or args.input is None:
        message = "The checkpoint adapter requires input and output paths"
        raise ValueError(message)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = (
        args.resolved_input.af3(args.trunk_seed)
        if args.resolved_input is not None
        else json.loads(args.input.read_text())
    )
    # Input/feature seed is controlled by the request, including legacy JSON.
    payload["modelSeeds"] = [args.trunk_seed]
    fold_input = folding_input.Input.from_json(
        json.dumps(payload), json_path=args.input
    )
    ccd = database.af3_ccd(user_ccd=fold_input.user_ccd)
    from foldforge.models.bucketing import TOKEN_SHAPES, bucket_af3

    examples = featurisation.featurise_input(
        fold_input,
        ccd,
        buckets=TOKEN_SHAPES if args.execution.bucketing else None,
        verbose=True,
    )
    dtype = torch.float32 if args.precision == "fp32" else torch.bfloat16
    model = load(
        args.model,
        args.checkpoint,
        backend=args.backend,
        dtype=dtype,
        precision_policy=args.precision,
        recycles=args.recycles,
        samples=args.samples,
        steps=args.steps,
    )
    runtime.bind(model)

    print("Loaded", model.foldforge_load_report, flush=True)  # noqa: T201
    from foldforge.data.features import dense_conventions

    for seed, raw_example in zip(fold_input.rng_seeds, examples, strict=True):
        example = dense_conventions.apply(raw_example, model.spec)
        bucket_shape = None
        if args.execution.bucketing:
            example, bucket_shape = bucket_af3(example)
            model.evoformer.foldforge_msa_bucketing = True
        example = dense_conventions.apply_after_bucketing(example, model.spec)
        start = time.monotonic()
        tensors = {
            k: torch.from_numpy(v).to("cuda")
            for k, v in example.items()
            if isinstance(v, np.ndarray) and v.dtype.kind in "biuf"
        }
        tensors["deletion_mean"] = tensors["deletion_mean"].float()

        from foldforge.models.io.images import input_images

        image_inputs = input_images(example, args.output.image_names, gap_id=31)

        def decode(
            result: dict[str, Any],
            _measurements: dict[str, float],
            *,
            bucket_shape: BucketShape | None = bucket_shape,
            example: dict[str, Any] = example,
            seed: int = seed,
            start: float = start,
            tensors: dict[str, torch.Tensor] = tensors,
            image_inputs: dict[str, Any] = image_inputs,
        ) -> Decoded:
            # The optional full distogram is only used for its PNG.
            cpu = numpy_tree(
                {key: value for key, value in result.items() if key != "distogram"}
            )
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
            name = target_name(fold_input.name)
            canonical = from_af3(
                result, torch.from_numpy(coords), tensors["pred_dense_atom_mask"]
            )
            from dataclasses import replace

            n_tokens = int(np.asarray(example["seq_mask"]).sum())
            canonical = replace(
                canonical,
                plddt=None
                if canonical.plddt is None
                else canonical.plddt[..., :n_tokens],
                pae=None
                if canonical.pae is None
                else canonical.pae[..., :n_tokens, :n_tokens],
                pde=None
                if canonical.pde is None
                else canonical.pde[..., :n_tokens, :n_tokens],
            )
            cifs = []
            for i, xyz in enumerate(coords):
                structure = layout.empty_output_struc.copy_and_update_atoms(
                    atom_x=xyz[:, 0],
                    atom_y=xyz[:, 1],
                    atom_z=xyz[:, 2],
                    atom_b_factor=np.asarray(plddt[i]),
                    atom_occupancy=np.ones(len(xyz)),
                )
                cifs.append(structure.to_mmcif())
            arrays = dict(
                coordinates=coords,
                plddt=plddt,
                **{k: v for k, v in cpu.items() if isinstance(v, np.ndarray)},
            )
            report = {
                "ccd": database.describe(),
                "buckets": None if bucket_shape is None else vars(bucket_shape),
                "trunk_seed": seed,
                "diffusion_seed": args.diffusion_seed,
                "recycles": args.recycles,
                "trunk_passes": model.num_recycles + 1,
                "samples_per_denoiser_call": model.num_samples,
                "steps": args.steps,
                "samples": args.samples,
                "checkpoint": model.foldforge_load_report,
                "elapsed_seconds": time.monotonic() - start,
                "mean_plddt": float(plddt.mean()),
                "confidence_fields": [
                    k for k, v in cpu.items() if isinstance(v, np.ndarray)
                ],
            }
            logits = result.get("distogram", {}).get("logits")
            return Decoded(
                f"{name}-seed{seed}",
                canonical,
                cifs,
                report,
                arrays=arrays,
                image_inputs=image_inputs,
                image_distogram=None
                if logits is None
                else logits[:n_tokens, :n_tokens],
            )

        yield Case(lambda tensors=tensors: model(tensors), decode)
