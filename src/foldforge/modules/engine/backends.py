"""Which backend serves which op: the one table behind ``backend``, ``engine.ops`` and ``engine.unsupported``.

The four backends are ``miniworld`` (the miniworld-engine kernels), ``cuequivariance`` (NVIDIA's vendor kernels),
``anthropic`` (the kernels of Anthropic's uplifting-biomolecular-modeling release) and ``pytorch`` (the dense
FoldForge modules, the reference every other backend is compared with). A model is loaded with one ``backend``; an
op can be pinned to another with ``engine.ops``. Every op the chosen backend has no kernel for runs the PyTorch
reference, and says so in ONE log line (``engine.unsupported: fallback``, the default) or raises
(``engine.unsupported: error``): nothing is substituted silently.

This module imports nothing heavy (the config imports it); the checks that need the machine (a payload on disk, a
device capability) run when ``BackendPlan.resolve`` is called at install time.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger("foldforge.backend")

BACKENDS = ("miniworld", "pytorch", "cuequivariance", "anthropic")

#: op -> the backends with a kernel for it (``pytorch`` serves every op). The keys are the names ``engine.ops`` takes.
SERVES: dict[str, tuple[str, ...]] = {
    "trimul": ("miniworld", "cuequivariance", "anthropic"),
    "triangle_attention": ("miniworld", "cuequivariance", "anthropic"),
    "opm": ("miniworld", "anthropic"),
    "msa_pwa": ("miniworld", "anthropic"),
    "single_attention": ("miniworld", "cuequivariance"),
    "transition": ("miniworld", "anthropic"),
    "layernorm": ("miniworld",),
    "conditioning": ("miniworld",),
    "dit_blocks": ("miniworld",),
    "atom_blocks": ("miniworld",),
    "sampler_ops": ("miniworld",),
}
OPS = tuple(SERVES)

#: The dense classes (by name) that carry each op; the module's ``foldforge_implementation`` is the op's backend.
OP_CLASSES: dict[str, tuple[str, ...]] = {
    "trimul": ("TriangleMultiplication",),
    "triangle_attention": ("GridSelfAttention",),
    "opm": ("OuterProductMean",),
    "msa_pwa": ("MSAAttention",),
    "single_attention": ("_SingleAttention",),
    "transition": ("Transition",),
    "layernorm": ("NativeLayerNorm",),
    "conditioning": ("DiffusionTransition", "AdaptiveLayerNorm"),
    "dit_blocks": ("DiffusionTransformer",),
    "atom_blocks": ("DiffusionCrossAttTransformer",),
    "sampler_ops": ("AtomCrossAttEncoder",),
}

#: Why a backend has no kernel for an op, where the reason is not the plain absence of a kernel.
_NO_KERNEL = {
    (
        "anthropic",
        "single_attention",
    ): "the release's pair-bias attention core has no FoldForge adapter for the Pairformer single attention",
    (
        "anthropic",
        "layernorm",
    ): "the release's LayerNorm refuses bf16 weights (an fp32-weight row only)",
}

UNSUPPORTED = ("fallback", "error")


class BackendUnsupportedError(RuntimeError):
    """A (backend, op) combination no kernel serves, with ``engine.unsupported: error``."""


@dataclass(frozen=True, kw_only=True)
class Support:
    """A support decision, separate from evidence that an operation executed."""

    refusal: str | None = None
    selection: str = "engine_internal"

    @property
    def accepted(self) -> bool:
        """Whether the inspected conditions permit this candidate."""
        return self.refusal is None


@dataclass(frozen=True)
class Resolved:
    """One op after the checks: what was asked, what runs, and why they differ."""

    op: str
    requested: str
    backend: str
    reason: str | None = None


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


@dataclass(frozen=True)
class BackendPlan:
    """The backend of every op for one loaded model."""

    backend: str = "miniworld"
    ops: tuple[tuple[str, str], ...] = ()
    unsupported: str = "fallback"
    resolved: dict[str, Resolved] = field(
        default_factory=dict, compare=False, hash=False
    )

    @classmethod
    def make(
        cls,
        backend: str = "miniworld",
        ops: Mapping[str, str] | None = None,
        unsupported: str = "fallback",
    ) -> BackendPlan:
        """A plan; ``backend`` and ``ops`` are validated here."""
        if backend not in BACKENDS:
            message = f"Unknown backend: {backend}"
            raise ValueError(message)
        if unsupported not in UNSUPPORTED:
            message = (
                f"engine.unsupported must be one of {UNSUPPORTED}, got {unsupported!r}"
            )
            raise ValueError(message)
        return cls(backend, tuple(sorted(validate_ops(ops or {}).items())), unsupported)

    def requested(self, op: str) -> str:
        """The backend asked for ``op``: its ``engine.ops`` entry, else ``backend``."""
        return dict(self.ops).get(op, self.backend)

    def of(self, op: str) -> str:
        """The backend that runs ``op`` (after :meth:`resolve`; the request before it)."""
        resolved = self.resolved.get(op)
        return resolved.backend if resolved is not None else self.requested(op)

    def uses(self, *backends: str) -> bool:
        """Whether any op runs on one of ``backends``."""
        return any(self.of(op) in backends for op in OPS)

    def resolve(self, device: object | None = None) -> BackendPlan:
        """Check every op against the machine: fill ``resolved`` and apply the ``unsupported`` policy.

        Logs ONE line naming every op that does not run on the backend it was asked for.
        """
        self.resolved.clear()
        for op in OPS:
            asked = self.requested(op)
            reason = refusal(op, asked, device)
            self.resolved[op] = Resolved(
                op, asked, asked if reason is None else "pytorch", reason
            )
        shared = [
            op
            for op in _PAYLOAD_OPS
            if self.of(op) == "miniworld" and os.environ.get(_PAYLOAD_OPS[op])
        ]
        if shared:
            names = ", ".join(f"{_PAYLOAD_OPS[op]} ({op})" for op in shared)
            logger.warning(
                "backend %s: %s is set, so the engine's miniworld modules run the Anthropic payload where it can serve; "
                "unset it for a miniworld run that uses the miniworld kernels only",
                self.backend,
                names,
            )
        missing = [r for r in self.resolved.values() if r.reason is not None]
        if missing and self.unsupported == "error":
            lines = "; ".join(f"{r.op} on {r.requested}: {r.reason}" for r in missing)
            message = (
                f"backend ops no kernel serves (engine.unsupported: error): {lines}"
            )
            raise BackendUnsupportedError(message)
        if missing:
            lines = "; ".join(f"{r.op} ({r.requested}: {r.reason})" for r in missing)
            logger.warning(
                "backend %s: running the PyTorch reference for %s", self.backend, lines
            )
        return self

    def report(self) -> dict[str, object]:
        """The load report's record: the backend of every op and each fallback's reason."""
        return {
            "backend": self.backend,
            "ops": {op: r.backend for op, r in self.resolved.items()},
            "fallbacks": {
                op: f"{r.requested}: {r.reason}"
                for op, r in self.resolved.items()
                if r.reason is not None
            },
            "unsupported": self.unsupported,
            **(
                {"anthropic_locations": payload_environment()}
                if self.uses("anthropic")
                else {}
            ),
        }


def refusal(op: str, backend: str, device: object | None = None) -> str | None:
    """Why ``backend`` cannot serve ``op`` on this machine (None when it can). Checks only what is known at install
    time: a call whose shape the kernel does not take is refused when it is made (:func:`call_refusal`)."""
    if backend == "pytorch":
        return None
    if backend not in SERVES[op]:
        return _NO_KERNEL.get((backend, op), "no kernel for this op")
    if backend == "cuequivariance":
        try:
            import cuequivariance_torch  # noqa: F401
        except ImportError as error:
            return f"cuequivariance_torch is not importable ({error})"
    if backend == "anthropic":
        return _anthropic_refusal(op, device)
    return None


def _capability(device: object | None) -> tuple[int, int] | None:
    import torch

    if not torch.cuda.is_available():
        return None
    index = (
        getattr(device, "index", None)
        if getattr(device, "type", None) == "cuda"
        else None
    )
    return torch.cuda.get_device_capability(index)


def _anthropic_refusal(op: str, device: object | None) -> str | None:
    """The Anthropic payloads are located by environment variables (locations, not behaviour) and built per card."""
    if _capability(device) is None:
        return "no CUDA device"
    if op == "trimul":
        from miniworld_engine.baselines import anthropic_trimul

        if not anthropic_trimul.payload_dir():
            return f"{anthropic_trimul.ENV} is not set (a payload's build/ directory)"
    elif op in {"opm", "msa_pwa"}:
        from miniworld_engine.baselines import anthropic_msa

        if not anthropic_msa.payload_dir():
            return f"{anthropic_msa.ENV} is not set (the release's common/opt_core)"
    else:  # triangle_attention, transition: the engine's adapter of the release's opt_core
        from miniworld_engine.baselines import anthropic

        try:
            anthropic.configure()
        except RuntimeError as error:
            return str(error)
    return None


#: The ops whose miniworld engine module consults an Anthropic payload named by the environment (an engine behaviour).
_PAYLOAD_OPS = {
    "trimul": "TRIMUL_NATIVE_BUILD_DIR",
    "opm": "OPT_CORE_DIR",
    "msa_pwa": "OPT_CORE_DIR",
}

_LOGGED: set[tuple[str, str, str]] = set()


def call_refusal(
    plan_policy: str, op: str, backend: str, reason: str | Support | None
) -> bool:
    """A call the backend's kernel refuses: True when the caller must run the PyTorch reference, after ONE log line per
    (op, reason); raises under ``engine.unsupported: error``. ``reason`` None means the kernel serves it."""
    if isinstance(reason, Support):
        reason = reason.refusal
    if reason is None:
        return False
    if plan_policy == "error":
        message = f"{backend} refuses a {op} call (engine.unsupported: error): {reason}"
        raise BackendUnsupportedError(message)
    key = (op, backend, reason)
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.warning(
            "backend %s refuses a %s call: %s; running the PyTorch reference for it",
            backend,
            op,
            reason,
        )
    return True


def forget_logged_refusals() -> None:
    """Let the next refused call log again (tests; a long-lived server that wants a fresh line per request)."""
    _LOGGED.clear()


def payload_environment() -> dict[str, str | None]:
    """The locations the Anthropic backend reads (recorded with a run; they are not behaviour switches)."""
    names = ("MINIWORLD_ANTHROPIC_ROOT", "OPT_CORE_DIR", "TRIMUL_NATIVE_BUILD_DIR")
    return {name: os.environ.get(name) for name in names}
