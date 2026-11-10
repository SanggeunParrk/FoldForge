# ruff: noqa: PLC0415 - checkpoint format dependencies are imported lazily
"""One construction, checkpoint validation, precision and backend lifecycle.

Architecture classes own network topology. Weight readers own serialization and
key conversion. Neither of them chooses runtime precision or installs backends.
"""

from __future__ import annotations

import dataclasses
import json
import math
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from foldforge.models import entry
from foldforge.models.msa_policy import RECORD
from foldforge.modules.engine.backends import BACKENDS, BackendPlan

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from foldforge.models.architectures.af3 import AlphaFold3
    from foldforge.models.config import Config


def load_configured(
    name: str,
    checkpoint: Path,
    config: Config,
    *,
    samples: int = 1,
    device: str | torch.device = "cuda",
) -> AlphaFold3:
    """Build the network once from the shared public inference configuration."""
    return cast(
        "AlphaFold3",
        load_checkpoint(
            name,
            checkpoint,
            backend=config.backend,
            device=device,
            dtype=torch.float32 if config.precision == "fp32" else torch.bfloat16,
            precision_policy=config.precision,
            recycles=config.trunk.recycles if config.trunk.recycles is not None else 10,
            samples=samples,
            steps=config.diffusion.steps if config.diffusion.steps is not None else 200,
            mode=config.mode,
            engine_skip=config.engine.skip,
            engine_ops=config.engine.ops,
            unsupported=config.engine.unsupported,
            dense_spec_override=config.debug.dense_spec_override,
        ),
    )


def _resolved_spec(spec: Any) -> dict[str, Any]:
    """Serialize final model conventions, including sets and infinite bounds."""

    def value(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: value(part) for key, part in item.items()}
        if isinstance(item, (set, frozenset)):
            return [value(part) for part in sorted(item)]
        if isinstance(item, (tuple, list)):
            return [value(part) for part in item]
        if isinstance(item, float) and not math.isfinite(item):
            return str(item)
        return item

    return value(dataclasses.asdict(spec))


def reorder_esm_nucleic_columns(model: Any) -> int:
    """Give each AF3 nucleic class the ESMFold2 weights of ITS base.

    The converted trunk projections read ESMFold2's classes 23..31 onto AF3's
    nucleic columns 22..30 in order. The two orders differ -- ESMFold2 puts the
    unknown ribonucleotide between RNA and DNA -- so every DNA column held the
    weights of the base before it. Corrected here rather than in the locked blob.
    """
    import torch

    from foldforge.modules.dense.featurization import (
        AF3_NUCLEIC_IN_ESM_ORDER,
    )

    source = list(range(31))
    for rank, af3_class in enumerate(AF3_NUCLEIC_IN_ESM_ORDER):
        source[af3_class] = 22 + rank  # the column the converter gave ESM class 23+rank
    fixed = 0
    evoformer = model.evoformer
    # Restype and profile blocks of the 447-wide target feature; the one-hot of
    # the MSA rows.
    targets = {
        "left_single": (0, 31),
        "right_single": (0, 31),
        "single_activations": (0, 31),
        "extra_msa_target_feat": (0, 31),
        "msa_activations": (0,),
    }
    with torch.no_grad():
        for name, offsets in targets.items():
            layer = getattr(evoformer, name, None)
            if layer is None:
                continue
            weight = layer.weight
            for offset in offsets:
                block = weight[:, offset : offset + 31].clone()
                weight[:, offset : offset + 31] = block[:, source]
                fixed += 1
    return fixed


def load_checkpoint(  # noqa: C901, PLR0912, PLR0915 - one explicit checkpoint lifecycle
    name: str,
    checkpoint: str | Path | None = None,
    *,
    backend: str = "miniworld",
    dtype: torch.dtype | None = None,
    device: str | torch.device = "cuda",
    recycles: int = 10,
    samples: int = 5,
    steps: int = 200,
    implementation: str | None = None,
    precision_policy: str | None = None,
    mode: str = "fast",
    engine_skip: Collection[str] = (),
    engine_ops: Mapping[str, str] | None = None,
    unsupported: str = "fallback",
    dense_spec_override: Mapping[str, Any] | None = None,
) -> torch.nn.Module:
    """Strictly load weights, then apply the same declared inference contract.

    ``recycles`` counts total trunk passes, including the first, for every family.

    ``engine_skip`` and ``dense_spec_override`` are the config's debugging aids;
    both default to nothing. ``backend`` supplies every op, while ``engine_ops``
    pins individual ops elsewhere. ``unsupported`` determines whether a missing
    backend kernel runs the PyTorch reference (``fallback``) or raises (``error``).
    """
    from foldforge.models.config.runtime import reject_removed_environment

    reject_removed_environment()
    if implementation is not None:
        backend = getattr(implementation, "value", implementation)
    if backend == "miniworld_engine":
        backend = "miniworld"
    if backend not in BACKENDS:
        message = f"Unknown backend: {backend}"
        raise ValueError(message)
    if precision_policy == "model_default":
        precision_policy = "af3_default"
        dtype = torch.bfloat16
    dtype = torch.bfloat16 if dtype is None else dtype
    if precision_policy not in {None, "bf16", "fp32", "af3_default", "model_default"}:
        message = f"Unsupported precision policy: {precision_policy}"
        raise ValueError(message)
    if precision_policy == "af3_default" and dtype != torch.bfloat16:
        message = "af3_default requires a BF16 trunk"
        raise ValueError(message)

    plan = BackendPlan.make(backend, engine_ops, unsupported).resolve(
        torch.device(device)
    )
    spec = entry(name)
    if spec.architecture is None:
        raise NotImplementedError(name)
    family = spec.family
    from foldforge.models.checkpoints import miniworld
    from foldforge.modules.dense.spec import spec_for

    dense_spec = spec_for(family or "alphafold3", mode)
    if checkpoint is None:
        checkpoint = miniworld.resolve_native(name)
    checkpoint = Path(checkpoint)
    from foldforge.models.checkpoints.lock import check_size

    check_size(checkpoint)
    package, _, symbol = spec.architecture.rpartition(".")
    architecture = getattr(import_module(package), symbol)
    report = {"checkpoint": str(checkpoint), "strict": True}

    if spec.layout != "dense_atoms":
        message = f"{name} has no loader: every family is a row of the dense graph"
        raise NotImplementedError(message)
    if dense_spec_override:
        # Porting aid: flip forward conventions of a family to price each one.
        # Fields that change a parameter shape make the strict load fail loudly.
        dense_spec = dataclasses.replace(dense_spec, **dense_spec_override)
        report["dense_spec_override"] = json.dumps(dense_spec_override)
    from foldforge.modules.engine.install import skip_engine_init

    if not miniworld.is_native(checkpoint):
        message = (
            f"{name} loads the miniworld checkpoint format; {checkpoint.name} is not one (a released blob?). "
            f"Convert it once: foldforge weights convert {name}"
        )
        raise ValueError(message)
    # a load fills every engine parameter straight away: skip the engine modules' own initialisation
    with skip_engine_init():
        model = architecture(
            num_recycles=recycles,
            num_samples=samples,
            diffusion_steps=steps,
            spec=dense_spec,
        )
    report["family"] = dense_spec.family
    report["mode"] = mode
    # Record the resolved spec, after family inheritance, exact-mode and debug
    # overrides. These describe the loaded equation, not observed execution.
    report["resolved_spec"] = _resolved_spec(dense_spec)
    # Parameters stay fp32 (``master`` or plain ``fp32``): the released bf16 values load
    # into fp32 exactly, so the af3_default policy needs no native-bf16 parameters.
    report.update(miniworld.load_into(model, checkpoint, family=dense_spec.family))
    if dense_spec.single_cond_layout == "esm":
        report["esm_nucleic_columns"] = reorder_esm_nucleic_columns(model)

    setattr(model, "foldforge_load_report", report)  # noqa: B010 - runtime metadata
    from foldforge.modules.engine.configure import configure_model, set_tf32_policy
    from foldforge.modules.engine.install import install_engine

    source = model.get_submodule("diffusion_head.fourier_embeddings")
    fourier_fp32 = (source.weight.float().clone(), source.bias.float().clone())
    if precision_policy == "af3_default":
        model.reference_precision = True
    # ``dtype`` is the activation dtype: bf16 -> ``master`` (fp32 parameters under
    # autocast), fp32 -> ``fp32``. Every parameter keeps its loaded fp32 value.
    model = configure_model(model, backend, dtype, device)
    # Noise Fourier features stay FP32 whatever the parameter dtype.
    fourier = model.get_submodule("diffusion_head.fourier_embeddings")
    fourier.weight = fourier_fp32[0].to(device)
    fourier.bias = fourier_fp32[1].to(device)
    from foldforge.modules.engine.capture import clear_weight_caches

    clear_weight_caches()
    model = install_engine(
        model,
        device,
        modules=plan.uses("miniworld", "anthropic")
        and torch.device(device).type == "cuda",
        skip=engine_skip,
        plan=plan,
    )
    # The engine's FP32 kernels run on TF32 tensor cores: the diffusion head of
    # af3_default takes them as its
    # atom blocks always have; everything else waits for ``execution.tf32``
    # (models.execution.Execution).
    set_tf32_policy(model, tf32=False, diffusion_tf32=precision_policy == "af3_default")
    report.update(
        backend=backend,
        backends=plan.report(),
        parameter_dtype=str(torch.float32),
        precision=model.foldforge_precision,
        activation_dtype=str(dtype),
        device=str(device),
        msa_policy=RECORD,
    )
    if precision_policy == "af3_default":
        report.update(
            precision_policy="af3_default",
            # fp32 parameters holding the released values (AF3's trunk is released in
            # BF16, Protenix's checkpoint in FP32; both are exact in fp32)
            trunk_parameter_dtype=sorted(
                {str(p.dtype) for p in getattr(model, "evoformer", model).parameters()}
            ),
            diffusion_parameter_dtype=sorted(
                {str(p.dtype) for p in model.diffusion_head.parameters()}
            ),
        )
    setattr(model, "foldforge_load_report", report)  # noqa: B010 - runtime metadata
    return model
