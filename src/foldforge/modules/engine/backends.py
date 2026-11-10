"""Which backend serves which op: the one table behind ``backend``, ``engine.ops`` and ``engine.unsupported``.

The engine (miniworld-engine 2.5.0) names five backends: ``pytorch`` (the reference), ``cuda`` (H100 kernels), ``triton``,
``anthropic`` and ``cueq``. A class lists the ones it has in its ``backends`` tuple and never falls back: a call outside a backend's
range raises ``BackendUnsupported``. FoldForge owns the policy on top of that:

* ``Config.backend`` takes the engine names plus two policy aliases: ``miniworld`` = the engine's own kernels, per class ``cuda`` on an
  H100, else ``triton``, else ``pytorch``; ``cuequivariance`` = ``cueq``.
* ``engine.ops`` pins single ops (the keys of ``OP_CLASSES``) to another backend.
* ``engine.unsupported``: ``fallback`` (default) runs the engine ``pytorch`` backend of that module for a call the chosen backend
  refuses and says so in ONE log line (:mod:`foldforge.modules.engine.policy`); ``error`` raises.

``SERVES`` (op -> the backends some class of the op has) is derived from the engine classes' ``backends`` tuples, never written down
here. This module imports nothing heavy at import time (the config imports it); the engine is imported when a table is asked for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger("foldforge.backend")

#: The engine's backend names.
ENGINE_BACKENDS = ("pytorch", "cuda", "triton", "anthropic", "cueq")
#: Policy aliases of ``Config.backend`` -> what they mean. ``None``: decided per class by :func:`choose_backend`.
ALIASES: dict[str, str | None] = {"miniworld": None, "cuequivariance": "cueq"}
BACKENDS = ("miniworld", "cuequivariance", *ENGINE_BACKENDS)

#: op -> the engine classes (by name, in ``miniworld_engine.modules``) that carry it. The keys are the names ``engine.ops`` takes.
OP_CLASSES: dict[str, tuple[str, ...]] = {
    "trimul": ("TriangleMultiplication", "TriangleMultiplicationRF3", "BidirectionalTriangleMultiplication"),
    "triangle_attention": ("TriangleAttention", "TriangleAttentionAF3", "TriangleAttentionRF3", "TriangleAttentionChai1",
                           "TriangleAttentionChai1Confidence"),
    "opm": ("OuterProductMean",),
    "msa_pwa": ("MSAPairWeightedAveraging",),
    "single_attention": ("AttentionPairBias",),
    "transition": ("Transition",),
    "layernorm": ("LayerNorm",),
    "conditioning": ("ConditionedTransition", "AdaptiveLayerNorm", "IdentityAdaptiveLayerNorm", "AdaptiveRMSNorm"),
    "dit_blocks": ("TokenDiTProtenix", "TokenDiTMiniWorldBiasOnly"),
    "atom_blocks": ("AtomDiTProtenix", "AtomDiTOpenFold3", "AtomDiTBoltz2", "AtomDiTRoseTTAFold3", "AtomDiTChai1", "AtomDiTMiniWorld"),
    "sampler_ops": ("AugmentNoise", "AlignEuler", "PoolAtoms"),
}
OPS = tuple(OP_CLASSES)
UNSUPPORTED = ("fallback", "error")

_H100 = (9, 0)


class BackendUnsupportedError(RuntimeError):
    """A (backend, module) combination the engine does not serve, with ``engine.unsupported: error``."""


def op_classes(op: str) -> tuple[type, ...]:
    """The engine classes of ``op`` (imports the engine)."""
    import miniworld_engine.modules as modules

    return tuple(getattr(modules, name) for name in OP_CLASSES[op])


def op_of(module: object) -> str | None:
    """The op an engine module instance carries, or None (imports the engine)."""
    for op in OPS:
        if isinstance(module, op_classes(op)):
            return op
    return None


def serves(op: str) -> tuple[str, ...]:
    """The backends some class of ``op`` has, in the policy's names (``miniworld`` when any class has ``cuda`` or ``triton``)."""
    have = {b for cls in op_classes(op) for b in cls.backends}
    named = [b for b in ENGINE_BACKENDS if b in have]
    return tuple([*(["miniworld"] if have & {"cuda", "triton"} else []), *named, *(["cuequivariance"] if "cueq" in have else [])])


def __getattr__(name: str) -> object:
    """``SERVES``: op -> :func:`serves`, derived from the engine classes on first use."""
    if name == "SERVES":
        return {op: serves(op) for op in OPS}
    raise AttributeError(name)


def validate_ops(ops: Mapping[str, str]) -> dict[str, str]:
    """``engine.ops``: known op names, known backends."""
    unknown = sorted(set(ops) - set(OPS))
    if unknown:
        message = f"engine.ops names unknown op(s) {unknown}; known: {list(OPS)}"
        raise ValueError(message)
    bad = {op: b for op, b in ops.items() if b not in BACKENDS}
    if bad:
        message = f"engine.ops names unknown backend(s) {bad}; known: {list(BACKENDS)}"
        raise ValueError(message)
    return dict(ops)


def capability(device: object | None) -> tuple[int, int] | None:
    """The compute capability of a CUDA ``device`` (None without CUDA)."""
    import torch

    if not torch.cuda.is_available() or getattr(device, "type", "cuda") != "cuda":
        return None
    index = getattr(device, "index", None)
    return torch.cuda.get_device_capability(index)


def choose_backend(requested: str, cls: type, device: object | None = None) -> tuple[str, str | None]:
    """``(backend, reason)`` of one engine class: the backend it runs under ``requested`` and, when that is not what was asked for, why.

    ``miniworld`` is the policy of the engine's own kernels: ``cuda`` where the class has it and the card is an H100, else ``triton`` where
    the class has it and a CUDA device exists, else ``pytorch`` (not a refusal: nothing named was asked for). A named backend the class
    does not have, or ``cuda`` off an H100, is a refusal: ``pytorch`` runs and the reason says why (``engine.unsupported: error`` raises
    it). Whether a CALL is in the backend's range is the module's own ``unsupported()``, asked when it is made."""
    name = ALIASES.get(requested, requested)
    have = tuple(cls.backends)
    cc = capability(device)
    if name is None:
        if "cuda" in have and cc == _H100:
            return "cuda", None
        if "triton" in have and cc is not None:
            return "triton", None
        return "pytorch", None
    if name == "pytorch":
        return "pytorch", None
    if name not in have:
        return "pytorch", f"{cls.__name__} has no {name!r} backend (available: {', '.join(have)})"
    if name == "cuda" and cc != _H100:
        return "pytorch", "the cuda backend is H100 only (sm_90)"
    if name in ("cuda", "triton") and cc is None:
        return "pytorch", f"the {name} backend needs a CUDA device"
    return name, None


@dataclass(frozen=True)
class Resolved:
    """One op after the checks: what was asked, what runs, and why they differ."""

    op: str
    requested: str
    backend: str
    reason: str | None = None


@dataclass(frozen=True)
class BackendPlan:
    """The backend of every op for one loaded model."""

    backend: str = "miniworld"
    ops: tuple[tuple[str, str], ...] = ()
    unsupported: str = "fallback"
    resolved: dict[str, Resolved] = field(default_factory=dict, compare=False, hash=False)

    @classmethod
    def make(cls, backend: str = "miniworld", ops: Mapping[str, str] | None = None, unsupported: str = "fallback") -> BackendPlan:
        """A plan; ``backend`` and ``ops`` are validated here."""
        if backend not in BACKENDS:
            message = f"Unknown backend: {backend}"
            raise ValueError(message)
        if unsupported not in UNSUPPORTED:
            message = f"engine.unsupported must be one of {UNSUPPORTED}, got {unsupported!r}"
            raise ValueError(message)
        return cls(backend, tuple(sorted(validate_ops(ops or {}).items())), unsupported)

    def requested(self, op: str) -> str:
        """The backend asked for ``op``: its ``engine.ops`` entry, else ``backend``."""
        return dict(self.ops).get(op, self.backend)

    def of(self, op: str) -> str:
        """The policy name that runs ``op`` (after :meth:`resolve`: the engine backend of the op's first class)."""
        resolved = self.resolved.get(op)
        return resolved.backend if resolved is not None else self.requested(op)

    def uses(self, *backends: str) -> bool:
        """Whether any op is asked to run on one of ``backends`` (policy or engine names)."""
        names = {ALIASES.get(b, b) or b for b in backends} | set(backends)
        return any(self.requested(op) in names or ALIASES.get(self.requested(op)) in names for op in OPS)

    def for_class(self, cls: type, device: object | None = None) -> tuple[str, str | None]:
        """:func:`choose_backend` for ``cls`` under this plan's request for its op."""
        op = next(op for op in OPS if issubclass(cls, op_classes(op)))
        return choose_backend(self.requested(op), cls, device)

    def resolve(self, device: object | None = None) -> BackendPlan:
        """Check every op against the engine classes and the machine: fill ``resolved`` and apply the ``unsupported`` policy.

        An op is refused when NO class of it can run what was asked; a class that lacks the backend while a sibling has it is refused
        per module when the plan is applied (:func:`foldforge.modules.engine.policy.apply_plan`). Logs ONE line naming every refused op.
        """
        self.resolved.clear()
        for op in OPS:
            asked = self.requested(op)
            choices = [choose_backend(asked, cls, device) for cls in op_classes(op)]
            ok = [name for name, reason in choices if reason is None]
            if ok:
                self.resolved[op] = Resolved(op, asked, ok[0])
            else:
                self.resolved[op] = Resolved(op, asked, "pytorch", choices[0][1])
        missing = [r for r in self.resolved.values() if r.reason is not None]
        if missing and self.unsupported == "error":
            lines = "; ".join(f"{r.op} on {r.requested}: {r.reason}" for r in missing)
            message = f"backend ops no engine class serves (engine.unsupported: error): {lines}"
            raise BackendUnsupportedError(message)
        if missing:
            lines = "; ".join(f"{r.op} ({r.requested}: {r.reason})" for r in missing)
            logger.warning("backend %s: running the engine pytorch backend for %s", self.backend, lines)
        return self

    def report(self) -> dict[str, object]:
        """The load report's record: the backend of every op and each refusal's reason."""
        return {
            "backend": self.backend,
            "ops": {op: r.backend for op, r in self.resolved.items()},
            "fallbacks": {op: f"{r.requested}: {r.reason}" for op, r in self.resolved.items() if r.reason is not None},
            "unsupported": self.unsupported,
        }
