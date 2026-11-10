"""FoldForge's side of the engine's backend contract: per-module backend choice, the guard that asks ``unsupported`` before a call, and
the fallback / error policy.

The engine never falls back (a call outside a backend's range raises ``BackendUnsupported``) and has no global switch. This module does
what the engine leaves to its caller:

1. :func:`apply_plan` walks a model, gives every engine op module the backend the plan chose for ITS class
   (:func:`~foldforge.modules.engine.backends.choose_backend`) and installs the guard. Composite blocks (``PairformerBlock``,
   ``Pairformer``) stay on ``pytorch`` themselves and their children are set one by one, so an op that has no kernel is refused alone
   instead of taking the whole block down.
2. The guard asks ``module.unsupported(*args)`` before the call. A reason under ``engine.unsupported: fallback`` runs the engine
   ``pytorch`` backend of that module for the call and logs the reason ONCE; under ``error`` it raises ``BackendUnsupportedError``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from foldforge.modules.engine.backends import BackendPlan, BackendUnsupportedError, op_of
from foldforge.modules.engine.serving import observe_call

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger("foldforge.backend")

_LOGGED: set[tuple[str, str, str]] = set()
_GUARDED = "_foldforge_guarded"
_COMPOSITES = ("PairformerBlock", "Pairformer")


def forget_logged_refusals() -> None:
    """Let the next refused call log again (tests; a long-lived server that wants a fresh line per request)."""
    _LOGGED.clear()


def refuse(module: nn.Module, reason: str) -> bool:
    """A call the module's backend refuses: True when the caller must run the pytorch backend, after ONE log line per (class, backend,
    reason); raises under ``engine.unsupported: error`` (the module's ``foldforge_unsupported``)."""
    cls, backend = type(module).__name__, module.backend
    if getattr(module, "foldforge_unsupported", "fallback") == "error":
        message = f"{cls} backend {backend!r} refuses a call (engine.unsupported: error): {reason}"
        raise BackendUnsupportedError(message)
    key = (cls, backend, reason)
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.warning("%s backend %s refuses a call: %s; running its pytorch backend for it", cls, backend, reason)
    return True


def guard(module: nn.Module, unsupported: str = "fallback") -> nn.Module:
    """Wrap ``module.forward`` so that a call the selected backend refuses follows ``unsupported`` (idempotent; the policy is read per
    call from ``module.foldforge_unsupported``)."""
    module.foldforge_unsupported = unsupported
    if module.__dict__.get(_GUARDED):
        return module
    original = module.forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        backend = module.backend
        if backend == "pytorch":
            observe_call(module, backend, None)
            return original(*args, **kwargs)
        reason = module.unsupported(*args, **kwargs)
        if reason is None:
            observe_call(module, backend, None)
            return getattr(module, f"_forward_{backend}")(*args, **kwargs)
        refuse(module, reason)
        observe_call(module, backend, reason)
        module.backend = "pytorch"
        try:
            return module._forward_pytorch(*args, **kwargs)
        finally:
            module.backend = backend

    module.forward = forward
    object.__setattr__(module, _GUARDED, True)
    return module


def apply_plan(model: nn.Module, plan: BackendPlan, device: object | None = None) -> dict[str, dict[str, str | None]]:
    """Give every engine op module of ``model`` its backend and guard; returns ``{module name: {"backend", "requested", "reason"}}``.

    Does not descend into an op module (its parts are not ops of their own: a ``Transition``'s ``LayerNorm`` is a parameter holder). The
    sampler glue (``AugmentNoise`` ...) is an op like the others."""
    report: dict[str, dict[str, str | None]] = {}

    def visit(name: str, module: nn.Module) -> None:
        if type(module).__name__ in _COMPOSITES and hasattr(module, "set_backend"):
            module.set_backend("pytorch")  # also puts every child on pytorch; the loop below gives each its own backend
            module.foldforge_unsupported = plan.unsupported
        elif op_of(module) is not None:
            backend, reason = plan.for_class(type(module), device)
            if reason is not None and plan.unsupported == "error":
                raise BackendUnsupportedError(f"{name or '<root>'} ({type(module).__name__}): {reason}")
            module.set_backend(backend)
            guard(module, plan.unsupported)
            op = op_of(module)
            report[name] = {"op": op, "backend": backend, "requested": plan.requested(op), "reason": reason}
            return
        for child_name, child in module.named_children():
            visit(f"{name}.{child_name}" if name else child_name, child)

    visit("", model)
    refused = {n: r for n, r in report.items() if r["reason"] is not None}
    if refused:
        lines = "; ".join(f"{n} ({r['requested']}: {r['reason']})" for n, r in sorted(refused.items())[:8])
        logger.warning("backend %s: %d module(s) run the engine pytorch backend: %s%s", plan.backend, len(refused), lines,
                       " ..." if len(refused) > 8 else "")
    return report
