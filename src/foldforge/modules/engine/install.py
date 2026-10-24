"""The order in which a loaded model gets its engine modules.

The pairformer wrappers come first (they create the modules the installer looks for), the engine
modules next (they read the released ``DiffusionTransition`` blocks, which the conditioning pass
turns into other classes), the conditioning pass last.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import torch

from foldforge.modules.engine.backends import OP_CLASSES, BackendPlan
from foldforge.modules.engine.conditioning import install_conditioning
from foldforge.modules.engine.configure import implementation_of
from foldforge.modules.engine.modules import install_engine_modules
from foldforge.modules.engine.pairformer import install_pairformers

if TYPE_CHECKING:
    from collections.abc import Collection

    from torch import nn


@contextlib.contextmanager
def skip_engine_init():
    """Engine modules built while this is active keep their uninitialised weights.

    Every engine module the installers build is filled from the loaded weights straight away; the
    truncated-normal initialiser it would otherwise run first (SciPy, ~40 ms per matrix) took
    30 of the 43 seconds of a load.
    """
    from miniworld_engine.modules import primitives

    original = primitives.InitType.apply
    primitives.InitType.apply = lambda _self, _data: None
    try:
        yield
    finally:
        primitives.InitType.apply = original


def apply_plan(model: nn.Module, plan: BackendPlan) -> None:
    """Give every module of an op the implementation the plan resolved for it.

    ``configure_model`` set one ``foldforge_implementation`` on every module (the model's ``backend``). An op the plan
    pins elsewhere (``engine.ops``), or that its backend has no kernel for and so runs the PyTorch reference, gets its
    own here, by the class that carries the op (``backends.OP_CLASSES``). With the default plan nothing changes.
    """
    owner = {name: op for op, names in OP_CLASSES.items() for name in names}
    for module in model.modules():
        op = owner.get(type(module).__name__)
        if op is None:
            continue
        backend = plan.of(op)
        if op == "sampler_ops":
            # the engine's fused rigid-motion / pooling kernels: a switch, read where the glue is chosen
            module.foldforge_sampler_ops = backend == "miniworld"
            continue
        module.foldforge_implementation = implementation_of(backend)
        module.foldforge_unsupported = plan.unsupported
        if op == "single_attention":
            module.attention.foldforge_implementation = implementation_of(backend)
            module.attention.foldforge_unsupported = plan.unsupported
    model.foldforge_sampler_ops = plan.of("sampler_ops") == "miniworld"
    # The engine's OPM / pair weighted averaging modules take their own H100 paths before the Anthropic payload, even when
    # built with implementation="anthropic"; this switches those off, so that the backend named is the kernel that runs.
    off = {
        **({"opm_train": False} if plan.of("opm") == "anthropic" else {}),
        **({"pwa_train": False, "pwa_infer": False} if plan.of("msa_pwa") == "anthropic" else {}),
    }
    if off:
        from miniworld_engine import settings

        settings.configure(**off)
    if plan.uses("cuequivariance"):
        from cuequivariance_ops_torch import init_triton_cache

        init_triton_cache()


def install_engine(
    model: nn.Module,
    dtype: torch.dtype,
    device: str | torch.device,
    *,
    modules: bool,
    skip: Collection[str] = (),
    plan: BackendPlan | None = None,
) -> nn.Module:
    """Wrap the pairformers, attach the engine modules (``modules``), convert the conditioning blocks.

    ``skip`` (``engine.skip`` of the config, a debugging aid) lists the parts left dense: ``pairformers``,
    ``conditioning`` and the ``engine_*`` module keys. ``plan`` is the backend of every op (``None``: all on
    ``miniworld``, which is what a model loaded without ``engine.ops`` gets).
    """
    skip = set(skip)
    plan = BackendPlan.make() if plan is None else plan
    with skip_engine_init():
        if "pairformers" not in skip:
            model = install_pairformers(model)
        apply_plan(model, plan)
        if modules:
            model.foldforge_load_report.update(
                install_engine_modules(
                    model, torch.empty(0, dtype=dtype, device=device), skip, plan
                )
            )
        return model if "conditioning" in skip else install_conditioning(model)
