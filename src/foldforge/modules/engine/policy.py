"""Shared support predicates and the boundary that observes/enforces a decision.

Predicates never execute kernels. Only callers that already enforced backend
refusals supply an op/backend to accept_support; ordinary eligibility checks
keep their existing silent fallback, including engine.skip and precision rules.
"""

from __future__ import annotations

import torch
from torch import nn

from foldforge.layers.implementation import ImplementationType
from foldforge.modules.engine.backends import Support, call_refusal
from foldforge.modules.engine.serving import observe_support


def backend_support(
    module: nn.Module, backend: object, x: torch.Tensor | None = None
) -> Support:
    """Check backend selection and, when supplied, the existing CUDA requirement."""
    if getattr(module, "foldforge_implementation", None) != backend:
        return Support(refusal="backend_not_selected")
    if x is not None and not x.is_cuda:
        return Support(refusal="cuda_required")
    return Support()


def miniworld_support(
    module: nn.Module, x: torch.Tensor, *, fp32: bool = False
) -> Support:
    """The common BF16 or explicitly allowed FP32/TF32 activation policy."""
    selected = backend_support(module, ImplementationType.MINIWORLD_ENGINE, x)
    if not selected.accepted:
        return selected
    if x.dtype != (torch.float32 if fp32 else torch.bfloat16):
        return Support(refusal="activation_dtype")
    if fp32 and not getattr(module, "foldforge_tf32", False):
        return Support(refusal="tf32_not_allowed")
    return Support()


def accept_support(
    module: nn.Module,
    support: Support,
    *,
    op: str | None = None,
    backend: str | None = None,
) -> bool:
    """Observe a check; apply legacy refusal policy only at explicit boundaries."""
    observe_support(module, support)
    if op is not None and backend is not None:
        return not call_refusal(
            getattr(module, "foldforge_unsupported", "fallback"), op, backend, support
        )
    return support.accepted


def refused(
    module: nn.Module, op: str, backend: str, reason: str | Support | None
) -> bool:
    """Compatibility-shaped refusal check through the same observation/policy boundary."""
    support = reason if isinstance(reason, Support) else Support(refusal=reason)
    return not accept_support(module, support, op=op, backend=backend)
