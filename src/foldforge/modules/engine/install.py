"""Gives a loaded model's engine modules their backends.

FoldForge's graph holds engine classes as its own modules, so installing is only: apply the backend plan to every engine op module
(per class, with the guard that implements ``engine.unsupported``), and record what was chosen. The graph already holds the engine classes, so no
module is converted or wrapped.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from foldforge.modules.engine.backends import BackendPlan
from foldforge.modules.engine.policy import apply_plan

if TYPE_CHECKING:
    from collections.abc import Collection

    import torch
    from torch import nn


@contextlib.contextmanager
def skip_engine_init():
    """Engine modules built while this is active keep their uninitialised weights (a load fills every one of them straight away; the
    truncated-normal initialiser (SciPy) it would run first costs ~40 ms per matrix)."""
    from miniworld_engine.modules import primitives

    original = primitives.InitType.apply
    primitives.InitType.apply = lambda _self, _data: None
    try:
        yield
    finally:
        primitives.InitType.apply = original


def install_engine(
    model: nn.Module,
    device: str | torch.device,
    *,
    modules: bool = True,
    skip: Collection[str] = (),
    plan: BackendPlan | None = None,
) -> nn.Module:
    """Apply ``plan`` (default: ``miniworld`` everywhere) to ``model``'s engine modules.

    ``modules=False`` (a CPU load, ``engine.skip`` containing ``engine_modules``) puts every module on ``pytorch``. The model's load report
    gets the plan and the per-module choices (``engine_backends``). The dense sampler glue has no engine wiring yet, so the model's
    ``foldforge_sampler_ops`` is False."""
    import torch as _torch

    plan = BackendPlan.make() if plan is None else plan
    if not modules or "engine_modules" in set(skip):
        plan = BackendPlan.make("pytorch", None, plan.unsupported)
    choices = apply_plan(model, plan, _torch.device(device))
    model.foldforge_sampler_ops = False
    report: dict[str, Any] = getattr(model, "foldforge_load_report", {})
    report["engine_backends"] = {"plan": plan.report(), "modules": choices}
    model.foldforge_load_report = report
    return model
