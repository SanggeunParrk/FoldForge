# GPU-specific dependencies are loaded at this inference boundary.
# ruff: noqa: PLC0415
"""Run released ESMFold2, recording precision, execution and structure metrics."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from team_gm.modules.exceptions import ImplementationType

from foldforge.eval.structure import deposit_ca, score_against_deposit
from foldforge.models import checkpoints, load
from foldforge.models.io.output import Decoded
from foldforge.models.io.runtime import Case, Runtime
from foldforge.modules.esmc import compute_lm_hidden_states, load_esmc
from foldforge.modules.sequence.features import (
    ESMFold2InputBuilder,
    build_input,
    model_kwargs,
)
from foldforge.prediction import Prediction

if TYPE_CHECKING:
    from collections.abc import Iterator

    from foldforge.data.ccd import CCDDatabase
    from foldforge.models.architectures.esmfold2 import ESMFold2Output
    from foldforge.models.io.request import Request

logger = logging.getLogger("foldforge")

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


def prepare(args: Request, database: CCDDatabase, runtime: Runtime) -> Iterator[Case]:  # noqa: C901, PLR0915 - case tensors stay alive through the decode closure
    """Prepare ."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    checkpoint = (
        Path(args.checkpoint) if args.checkpoint else checkpoints.resolve("esmfold2")
    )
    sample = Path(args.data_root) / args.target

    if args.resolved_input is not None:
        target = args.resolved_input
        structure_input = target.esmfold2(args.msa_depth)
        manifest = target.manifest()
    elif args.input_spec:
        from foldforge.data.inputs.build import load as load_input

        target = load_input(args.input_spec)
        if target.spec.ccd_db.resolve() != args.ccd_db.resolve():
            message = "Spec and selected CCD database disagree"
            raise ValueError(message)
        structure_input = target.esmfold2(args.msa_depth)
        manifest = target.manifest()
    else:
        structure_input = build_input(sample, args.msa_depth)
        manifest = json.loads((sample / "target.json").read_text())
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
    if backend == ImplementationType.CUEQUIVARIANCE:
        from cuequivariance_ops_torch import init_triton_cache

        init_triton_cache()
    model = load(
        "esmfold2", checkpoint, backend=backend.value, dtype=dtype, device=device
    )
    runtime.bind(model)

    kwargs = model_kwargs(features, dtype)
    kwargs["lm_hidden_states"] = lm_hidden
    bucket_shape = None
    if args.execution.bucketing:
        from foldforge.models.bucketing import pad_esmfold2

        kwargs, bucket_shape = pad_esmfold2(kwargs)

    def fold() -> object:
        """Compute fold."""
        return model(
            **kwargs,
            num_diffusion_samples=args.samples,
            num_loops=args.recycles,
            num_sampling_steps=args.steps,
            generator=torch.Generator(device=device).manual_seed(args.seed),
        )

    def decode(out: ESMFold2Output, measurements: dict[str, float]) -> Decoded:
        """Decode ."""
        if bucket_shape is not None:
            from foldforge.models.bucketing import unpad_esmfold2

            out = unpad_esmfold2(out, bucket_shape)
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
            "buckets": None if bucket_shape is None else vars(bucket_shape),
            "device": torch.cuda.get_device_name(device),
            "dtype": args.dtype,
            "seed": args.seed,
            "msa_depth": args.msa_depth,
            "checkpoint": str(checkpoint.resolve()),
            "lm_source": args.lm_source,
            "lm_path": str(lm_path.resolve()),
            "seconds_lm_forward": lm_seconds,
            "scoring_atoms": "CA, matched by chain/residue/name",
            "num_diffusion_samples": args.samples,
            "num_sampling_steps": args.steps
            if args.steps is not None
            else model.config.structure_head.inference_num_steps,
            "num_loops": args.recycles
            if args.recycles is not None
            else model.config.num_loops,
            "seconds_first": round(first, 3),
            "seconds_warm": None if warm is None else round(warm, 3),
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
            Path(args.experimental)
            if args.experimental
            else sample / f"{args.target}.cif"
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
        cifs = [sample.complex.to_mmcif() for sample in decoded_samples]
        return Decoded(args.target, prediction, cifs, report, singleton_cif=True)

    yield Case(fold, decode, inference_mode=False)
