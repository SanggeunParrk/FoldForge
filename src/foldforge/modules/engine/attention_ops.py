"""The vendor legs of the dense attention modules (cuequivariance, Anthropic's release).

Where an engine module serves an attention (``modules.py``: ``TriangleAttention``, ``MSAPairWeightedAveraging``,
``AttentionPairBias``, the DiT blocks), the dense module hands it the whole call (residual included). A call no engine
module serves runs the dense module's PyTorch statements; the only pieces routed from here are the attention cores of
the two vendor backends, which a user selects by name (``engine.ops``). No engine kernel function is called here.
"""

from __future__ import annotations

from typing import cast

import torch

from foldforge.modules.engine.backends import Support
from foldforge.modules.engine.policy import accept_support, backend_support, refused

__all__ = [
    "anthropic_attention",
    "serve_projected_triangle",
]


def anthropic_attention(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """The release's k2b attention core, preserving its explicit refusal policy."""
    why = (
        "the release's calls are inference-only (grad is enabled)"
        if torch.is_grad_enabled()
        else None
    )
    if refused(module, "triangle_attention", "anthropic", why):
        return None
    from miniworld_engine.baselines import anthropic

    vendor_mask = None if mask is None else mask[None, :, None, None, :].bool()
    head_dim = cast("int", module.qkv_dim)
    try:
        out = anthropic.triangle_attention(
            q[None].contiguous(),
            k[None].contiguous(),
            v[None].contiguous(),
            bias[None, None].float().contiguous(),
            vendor_mask,
            head_dim**-0.5,
            row="k2b",
        )
    except Exception as error:  # the release refuses by name (opt_core Refusal)
        if type(error).__name__ != "Refusal":
            raise
        refused(module, "triangle_attention", "anthropic", str(error))
        return None
    return out.squeeze(0)


def serve_projected_triangle(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """The triangle-attention core on a selected vendor backend; None requests the PyTorch statements.

    The engine's ``TriangleAttention`` is not reached from here: it takes the whole module call
    (``GridSelfAttention.residual`` via ``routing.served_by``).
    """
    head_dim = cast("int", module.qkv_dim)
    cueq = backend_support(module, "cuequivariance", q)
    if cueq.accepted:
        accept_support(module, cueq)
        from cuequivariance_torch import triangle_attention

        vendor_mask = None if mask is None else mask[None, :, None, None, :]
        out = triangle_attention(
            q[None].contiguous(),
            k[None].contiguous(),
            v[None].contiguous(),
            bias[None, None].float().contiguous(),
            mask=vendor_mask,
            scale=head_dim**-0.5,
        )
        if not isinstance(out, torch.Tensor):
            message = "Triangle attention returned auxiliary values unexpectedly"
            raise TypeError(message)
        return out.squeeze(0)
    if backend_support(module, "anthropic", q).accepted:
        return anthropic_attention(module, q, k, v, bias, mask)
    accept_support(module, Support(refusal="projected_attention_reference"))
    return None
