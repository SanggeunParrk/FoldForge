# Optional chemistry/GPU backends load at the selected execution boundary.
# ruff: noqa: PLC0415
"""Checkpoint tensor-layout conversion for the common inference runtime."""

from __future__ import annotations

import hashlib
import os
import json
import threading
import time
from typing import TYPE_CHECKING, Any

from foldforge.models.io.output import Decoded, numpy_tree, target_name
from foldforge.models.io.runtime import Case, Runtime

if TYPE_CHECKING:
    from collections.abc import Iterator

    from foldforge.data.ccd import CCDDatabase
    from foldforge.models.bucketing import BucketShape
    from foldforge.models.io.request import Request


# --- CPU featurisation that can run ahead of the GPU -------------------------------------------------------------------------------------------
# `featurise_input` (+ the layout / bond-order captures) is pure CPU and takes 40-50 % of a trunk call on a mid-size target. A staged caller that
# already knows the next trunk_seed calls `prefetch_features` while the GPU works on the current one; `_prepare` then takes the finished result
# instead of featurising itself. The result is used only if the payload it was built from is byte-identical to the one `_prepare` builds.
# `capture_layouts` / `bond_orders.capture` patch AF3 classmethods process-wide, so every featurisation (prefetched or not) holds one lock.
_FEATURISE_LOCK = threading.Lock()
_PREFETCH: dict[tuple, Any] = {}
_PREFETCH_LOCK = threading.Lock()
#: set by a staged caller; called once per `_prepare`, right after its featurisation (taken from a prefetch or done here) and before the GPU work
ON_FEATURISED: Any = None
PREFETCH_TIMES: dict[str, float] = {"wait": 0.0, "featurise_here": 0.0}   # seconds the GPU thread was blocked on a prefetch / featurised on its own
PREFETCH_STATS: dict[str, int] = {"hit": 0, "stale": 0, "failed": 0, "none": 0}   # how the prefetches were used (a caller can print it)


def _payload_key(payload: dict[str, Any], bucketing: bool) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode() + bytes([bool(bucketing)])).hexdigest()


# Features handed over by a DataLoader worker are matched by the CONTENT of the input they were made from, not by re-deriving the AF3 payload
# (`Input.af3()` rewrites every MSA as a3m text and costs 0.1-3 s per call on the GPU thread). The payload is a pure function of the loaded
# input (spec + resources), the trunk_seed and the bucketing flag, so a signature of those identifies it.
_PROVIDED: dict[str, tuple] = {}


def _spec_signature(target: Any, trunk_seed: int, bucketing: bool) -> str:
    digest = hashlib.sha1()
    digest.update(json.dumps(target.spec.model_dump(mode="json"), sort_keys=True, default=str).encode())
    for chain in target.chains:
        digest.update(repr((chain.kind, chain.sequence, chain.ccds, None if chain.alignment is None else chain.alignment.tokens.shape)).encode())
        if chain.alignment is not None:
            digest.update(chain.alignment.tokens[:: max(1, len(chain.alignment.tokens) // 64)].tobytes())   # a sample of the rows: a changed MSA changes the shape or these
    digest.update(repr((int(trunk_seed), bool(bucketing))).encode())
    return digest.hexdigest()


def _featurise(fold_input: Any, ccd: Any, database: Any, bucketing: bool, structural: bool, verbose: bool) -> tuple[list, list, list]:
    """AF3 featurisation of one input with the structural-layout and bond-order captures: (examples, captures, orders)."""
    import contextlib

    from alphafold3.constants import chemical_component_sets
    from alphafold3.data import featurisation

    from foldforge.data.features import bond_orders, structural_tokens
    from foldforge.models.bucketing import TOKEN_SHAPES

    _install_reference_date_fallback()
    # a prefetch thread runs outside `prepare`'s own scope, so it opens the glycan sets itself; a database object without them (a test double) adds none
    sets = getattr(database, "chemical_component_sets", None)
    with _FEATURISE_LOCK, chemical_component_sets.use_ccd_sets(sets) if sets is not None else contextlib.nullcontext():
        with (
            structural_tokens.capture_layouts() if structural else contextlib.nullcontext([]) as captures,
            bond_orders.capture(ccd) as orders,
        ):
            examples = list(featurisation.featurise_input(fold_input, ccd, buckets=TOKEN_SHAPES if bucketing else None, verbose=verbose))
    return examples, captures, orders


def _prefetch_job(target: Any, model: str, trunk_seed: int, ccd_db: str, bucketing: bool) -> tuple[str, tuple, Any]:
    """One prefetch, in whichever thread or forked process the executor runs it: payload -> fold input -> featurisation. Module-level so it pickles."""
    from alphafold3.common import folding_input

    from foldforge.data.ccd import CCDDatabase
    from foldforge.models import entry
    from foldforge.modules.dense.spec import SPECS

    payload = target.af3(trunk_seed)
    payload["modelSeeds"] = [trunk_seed]
    fold_input = folding_input.Input.from_json(json.dumps(payload), json_path=None)
    database = CCDDatabase(ccd_db)
    ccd = database.af3_ccd(user_ccd=fold_input.user_ccd)
    family = entry(model).family
    return _payload_key(payload, bucketing), _featurise(fold_input, ccd, database, bucketing, SPECS[family or "alphafold3"].structural_tokens, False), (payload if os.environ.get("FF_PREFETCH_DEBUG") else None)


def prefetch_features(executor: Any, inputs: Any, *, model: str, trunk_seed: int, ccd_db: Any, msa_depth: int, bucketing: bool = True) -> None:
    """Start featurising ``inputs`` for ``trunk_seed`` on ``executor`` (``submit(fn, *args)``: a thread pool or a forked process pool; a process
    pool keeps the work off the GIL the GPU-driving thread needs); the next matching `_prepare` collects it."""
    from foldforge.data.inputs.build import limit_msa

    target = limit_msa(inputs.target, int(msa_depth))      # the same cap `prepare_prediction` puts on the input `_prepare` featurises
    slot = (target.spec.name, int(trunk_seed), bool(bucketing))
    with _PREFETCH_LOCK:
        _PREFETCH[slot] = executor.submit(_prefetch_job, target, model, int(trunk_seed), str(ccd_db), bool(bucketing))


def _language_features(
    name: str,
    checkpoint: Any,
    tensors: dict[str, Any],
    dtype: Any,
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the family's language model, as a pair or as per-token embeddings."""
    from foldforge.models.io import language

    common = {
        "aatype": tensors["aatype"],
        "asym_id": tensors["asym_id"],
        "mask": tensors["seq_mask"],
        "dtype": dtype,
        "cache": cache,
    }
    if language.reads_pair(name):
        return {
            "lm_pair": language.pair_representation(
                name,
                checkpoint,
                residue_index=tensors["residue_index"],
                is_protein=tensors["is_protein"].bool(),
                **common,
            )
        }
    return {
        "lm_embeddings": language.token_embeddings(
            name, checkpoint, is_protein=tensors["is_protein"].bool(), **common
        )
    }


def _attach_chemistry(example: dict, orders: Any, spec: Any, ccd: Any) -> None:
    """Add what AF3's featurisation drops: bond orders, and chirality if read."""
    from foldforge.data.features import bond_orders, chirality

    example[bond_orders.KEY] = orders
    if spec.atom_chiral_features:
        example.update(chirality.chiral_features(example, ccd))


def prepare(args: Request, database: CCDDatabase, runtime: Runtime) -> Iterator[Case]:
    """Prepare ."""
    from alphafold3.constants import chemical_component_sets

    with chemical_component_sets.use_ccd_sets(database.chemical_component_sets):
        yield from _prepare(args, database, runtime)


def _log_checkpoint(report: Any, *, verbose: bool) -> None:
    """Keep adapter progress optional for library callers."""
    if verbose:
        print("Loaded", report, flush=True)  # noqa: T201 - CLI progress


def _prepare(args: Request, database: CCDDatabase, runtime: Runtime) -> Iterator[Case]:  # noqa: PLR0915 - case tensors stay alive through the decode closure
    import contextlib

    import numpy as np
    import torch
    from alphafold3.common import folding_input
    from alphafold3.data import featurisation
    from alphafold3.model import feat_batch
    from alphafold3.model.atom_layout import atom_layout

    from foldforge.models.io.confidence import from_af3

    if args.input is None:
        message = "The checkpoint adapter requires an input origin path"
        raise ValueError(message)

    from copy import deepcopy

    _tp = time.perf_counter()
    provided = None
    if args.resolved_input is not None:
        with _PREFETCH_LOCK:
            provided = _PROVIDED.pop(_spec_signature(args.resolved_input, args.trunk_seed, bool(args.execution.bucketing)), None)
    if provided is not None:       # a worker featurised exactly this input: no payload, no JSON round trip (specs carry no user CCD)
        import types

        payload = {"name": args.resolved_input.spec.name}
        fold_input = types.SimpleNamespace(rng_seeds=[args.trunk_seed], user_ccd=None, name=args.resolved_input.spec.name)
    else:
        payload = (
            args.resolved_input.af3(args.trunk_seed)
            if args.resolved_input is not None
            else deepcopy(args.input_data)
            if args.input_data is not None
            else json.loads(args.input.read_text())
        )
        # Input/feature seed is controlled by the request, including legacy JSON.
        payload["modelSeeds"] = [args.trunk_seed]
        fold_input = folding_input.Input.from_json(
            json.dumps(payload), json_path=args.input
        )
    ccd = database.af3_ccd(user_ccd=fold_input.user_ccd)
    from foldforge.data.features import bond_orders, structural_tokens
    from foldforge.models import entry
    from foldforge.models.bucketing import TOKEN_SHAPES, bucket_af3
    from foldforge.modules.dense.spec import SPECS

    # A family that folds on structural tokens needs the AtomLayouts the
    # featurisation keeps to itself, so take them from the one call that
    # already happens rather than featurising a second time. The family is
    # resolved without building the model: nothing here needs the weights yet.
    family = entry(args.model).family
    structural = SPECS[family or "alphafold3"].structural_tokens
    with _PREFETCH_LOCK:
        ahead = _PREFETCH.pop((payload.get("name"), int(args.trunk_seed), bool(args.execution.bucketing)), None) if provided is None else None
    PREFETCH_TIMES["payload_ccd"] = PREFETCH_TIMES.get("payload_ccd", 0.0) + time.perf_counter() - _tp
    _tk = time.perf_counter()
    key = _payload_key(payload, args.execution.bucketing) if provided is None else None
    PREFETCH_TIMES["key"] = PREFETCH_TIMES.get("key", 0.0) + time.perf_counter() - _tk
    featurised = provided
    if provided is not None:
        PREFETCH_STATS["hit"] += 1
    if ahead is not None:
        try:
            _w = time.perf_counter()
            ahead_key, ahead_result, ahead_payload = ahead.result()
            PREFETCH_TIMES["wait"] += time.perf_counter() - _w
        except Exception:   # noqa: BLE001 - a failed prefetch only costs the featurisation it would have saved
            ahead_key = None
            ahead_payload = None
            PREFETCH_STATS["failed"] += 1
        if ahead_key == key:       # the prefetch was built from exactly this payload
            featurised = ahead_result
            PREFETCH_STATS["hit"] += 1
    if ahead is None and provided is None:
        PREFETCH_STATS["none"] += 1
    elif featurised is None and ahead is not None:
        PREFETCH_STATS["stale"] += 1
        if ahead_payload is not None:   # FF_PREFETCH_DEBUG: say what differed
            diff = [k for k in set(payload) | set(ahead_payload) if payload.get(k) != ahead_payload.get(k)]
            for s_i, (a, b) in enumerate(zip(payload["sequences"], ahead_payload["sequences"], strict=False)):
                if a != b:
                    diff.append(f"sequences[{s_i}]:" + ",".join(k for k in set(next(iter(a.values()))) | set(next(iter(b.values()))) if next(iter(a.values())).get(k) != next(iter(b.values())).get(k)))
            for s_i, (a, b) in enumerate(zip(payload["sequences"], ahead_payload["sequences"], strict=False)):
                x, y = next(iter(a.values())), next(iter(b.values()))
                for k in ("unpairedMsa", "pairedMsa"):
                    if x.get(k) != y.get(k):
                        la, lb = x[k].splitlines(), y[k].splitlines()
                        j = next((n for n, (p, q) in enumerate(zip(la, lb)) if p != q), None)
                        print(f"PREFETCH_STALE_DETAIL seq{s_i} {k}: lines {len(la)} vs {len(lb)}, first diff at line {j}: main={la[j][:60] if j is not None else None!r} prefetch={lb[j][:60] if j is not None else None!r}", flush=True)
            print(f"PREFETCH_STALE {payload.get('name')} seed={args.trunk_seed}: differs in {diff}", flush=True)
    if featurised is None:
        _f = time.perf_counter()
        featurised = _featurise(fold_input, ccd, database, bool(args.execution.bucketing), structural, args.verbose)
        PREFETCH_TIMES["featurise_here"] += time.perf_counter() - _f
    examples, captures, orders = featurised
    _tq = time.perf_counter()
    if ON_FEATURISED is not None:      # the caller's cue to start the next trunk_seed's featurisation: the CPU is free again, the GPU is about to be busy
        ON_FEATURISED(args.msa_depth)   # the depth the caller must cap the next input at, as `prepare_prediction` capped this one
    dtype = torch.float32 if args.precision == "fp32" else torch.bfloat16
    model = runtime.load_model()

    runtime.bind(model)

    _log_checkpoint(model.foldforge_load_report, verbose=args.verbose)
    from foldforge.data.features import dense_conventions

    for example_index, (seed, raw_example) in enumerate(
        zip(fold_input.rng_seeds, examples, strict=True)
    ):
        example = dense_conventions.apply(raw_example, model.spec)
        _attach_chemistry(example, orders[example_index], model.spec, ccd)
        bucket_shape = None
        if args.execution.bucketing:
            example, bucket_shape = bucket_af3(example)
            evoformer: Any = model.evoformer
            evoformer.foldforge_msa_bucketing = True
            evoformer.foldforge_msa_presampled = True      # the MSA row draw happens before the (graphed) evoformer call, see Evoformer.sample_msa
        if model.spec.structural_tokens:
            # The structural tokens join the bucketed residues; the engine's token DiT
            # serves only a
            # multiple of 128 tokens, so a bucketed run pads them to the residue
            # bucket's own granularity.
            example = structural_tokens.attach(
                example,
                captures[example_index],
                pad_multiple=128 if args.execution.bucketing else 32,
            )
        example = dense_conventions.apply_after_bucketing(example, model.spec)
        start = time.monotonic()
        tensors = {
            k: torch.from_numpy(np.ascontiguousarray(v)).to(runtime.device)
            for k, v in example.items()
            if isinstance(v, np.ndarray) and v.dtype.kind in "biuf"
        }
        tensors["deletion_mean"] = tensors["deletion_mean"].float()

        if model.spec.language_model is not None:
            # Most of this family's token stream; absent, the fold still runs
            # and is merely much worse, so a missing tower is an error. Some
            # families read the tower as one embedding per token and some as a
            # pair, which is a different graph rather than a different setting.
            tensors.update(
                _language_features(
                    model.spec.language_model,
                    args.checkpoint,
                    tensors,
                    dtype,
                    runtime.language_models,
                )
            )

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
                result,
                torch.from_numpy(coords),
                tensors["pred_dense_atom_mask"],
                asym_id=torch.from_numpy(np.asarray(example["asym_id"])),
                token_mask=torch.from_numpy(np.asarray(example["seq_mask"])),
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
                "trunk_passes": model.num_recycles,
                "samples_per_denoiser_call": model.num_samples,
                "steps": args.steps,
                "samples": args.samples,
                "checkpoint": model.foldforge_load_report,
                "elapsed_seconds": time.monotonic() - start,
                "mean_plddt": float(plddt.mean()),
                # Whether this fold's coin put its recycles under MC dropout.
                "recycle_dropout_applied": getattr(
                    model, "recycle_dropout_applied", False
                ),
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

        # no_grad, not inference_mode: the engine caches its pair bias across solver
        # steps by tensor version, which an inference tensor does not have.
        PREFETCH_TIMES["after_features"] = PREFETCH_TIMES.get("after_features", 0.0) + time.perf_counter() - _tq   # conventions, chemistry, bucket padding, tensors to device
        yield Case(
            lambda tensors=tensors: model(tensors),
            decode,
            inference_mode=False,
            run_trunk=lambda tensors=tensors: model.run_trunk(tensors),
            sample_diffusion=model.sample_diffusion,
        )


def install_msa_feature_cache(max_entries: int = 24) -> None:
    """Memoise AF3's ``extract_msa_features`` by the content of its MSA.

    It is a pure function of (the MSA sequences, the polymer type) and the dominant CPU cost of featurising a target with deep MSAs (a few seconds for
    20k-row chains). One featurisation calls it several times with the same MSA, and every trunk_seed of a target calls it again with the same MSA.
    Returned arrays are copies, so nothing downstream can alter a cached entry. Call before forking featuriser processes so they inherit it."""
    import collections

    from alphafold3.data import msa_features

    if getattr(msa_features.extract_msa_features, "_ff_cached", False):
        return
    original = msa_features.extract_msa_features
    cache: collections.OrderedDict = collections.OrderedDict()
    lock = threading.Lock()

    def cached(msa_sequences: Any, chain_poly_type: str) -> tuple:
        digest = hashlib.blake2b(digest_size=16)
        for sequence in msa_sequences:
            digest.update(sequence.encode())
            digest.update(b"\n")
        key = (digest.digest(), chain_poly_type, len(msa_sequences))
        with lock:
            hit = cache.get(key)
            if hit is not None:
                cache.move_to_end(key)
        if hit is None:
            hit = original(msa_sequences=msa_sequences, chain_poly_type=chain_poly_type)
            with lock:
                cache[key] = hit
                while len(cache) > max_entries:
                    cache.popitem(last=False)
        return hit[0].copy(), hit[1].copy()

    cached._ff_cached = True  # type: ignore[attr-defined]
    msa_features.extract_msa_features = cached


def provide_features(slot: tuple, key: str, result: tuple) -> None:
    """Hand `_prepare` an already featurised input (e.g. from a DataLoader worker); it is used only if its payload key matches."""
    from concurrent.futures import Future

    done: Future = Future()
    done.set_result((key, result, None))
    with _PREFETCH_LOCK:
        _PREFETCH[slot] = done


def provide_features_by_signature(signature: str, result: tuple) -> None:
    """Hand `_prepare` features built from the input with this `_spec_signature` (a DataLoader worker's); `_prepare` recomputes the signature of its own
    input and takes them only on a match, without building the AF3 payload at all."""
    with _PREFETCH_LOCK:
        _PROVIDED[signature] = result


def _install_reference_date_fallback() -> None:
    """A CCD component whose ideal coordinates are '?' takes its reference from the experimental model coordinates if its entry is older than
    ``ref_max_modified_date`` (AF3's data-leakage guard); that date is the featuriser's ``max_template_date``, which is None for "no cutoff", and
    ``date < None`` raises. None means no restriction, i.e. every entry counts as old enough. (``max_template_date`` itself must stay None: it also
    filters templates by their release date, and explicitly supplied templates carry a far-future placeholder.)"""
    import datetime

    from alphafold3.model import features

    original = features._get_reference_positions_from_ccd_cif
    if getattr(original, "_ff_patched", False):
        return

    def patched(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("ref_max_modified_date", 0) is None:
            kwargs["ref_max_modified_date"] = datetime.date.max
        return original(*args, **kwargs)

    patched._ff_patched = True  # type: ignore[attr-defined]
    features._get_reference_positions_from_ccd_cif = patched
