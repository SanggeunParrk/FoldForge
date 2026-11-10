"""Shared attention composition with explicit checkpoint precision policy."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    scale: float = 1.0,
    logits_float32: bool = True,
    efficient: bool = False,
    mask_value: float = -1e9,
) -> torch.Tensor:
    """Preserve bias-before-mask and probability/value precision boundaries.

    ``mask`` is broadcastable to [..., heads, query, key], True means keep.
    Fully masked finite-logit rows retain the released softmax convention.
    Callers own projections, window/gather layout, gating and residuals.
    """
    dtype = q.dtype
    if logits_float32:
        q, k = q.float(), k.float()
        if bias is not None:
            bias = bias.float()
    if efficient:
        if mask is not None:
            # Preserve replacement semantics; additive mask+bias differs on
            # fully-masked rows, so use the explicit equation for this case.
            raise ValueError(
                "Use explicit attention when a replacement mask is present"
            )
        return F.scaled_dot_product_attention(
            q, k, v.to(q.dtype), attn_mask=bias, scale=scale
        ).to(dtype)
    with torch.autocast(device_type=q.device.type, enabled=False):
        logits = (q * scale) @ k.transpose(-1, -2)
        if bias is not None:
            logits = logits + bias
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), mask_value)
        weights = logits.softmax(dim=-1)
        value = v.to(weights.dtype) if not logits_float32 else v.to(dtype)
        return (weights.to(value.dtype) @ value).to(dtype)


def af3_outer_product_mean(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """AF3's released contraction layout and post-projection normalization.

    Explicit layout preserves BF16 accumulation order and checkpoint [H,H,C]
    weights; the shared default OPM instead accepts a Linear projection.
    """
    outer = torch.einsum("acb,ade->dceb", left.permute(0, 2, 1), right)
    output = torch.einsum("dceb,cef->dbf", outer, weight) + bias
    norm = torch.einsum("abc,adc->bdc", mask, mask)
    return output.permute(1, 0, 2) / (eps + norm)


def dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention on [..., H, L, D]; a 3-D input gets a batch axis.

    ``mask`` is [L_k], [B, L_k] or already broadcastable; ``bias`` is added to the logits.
    """
    unbatched = q.dim() == 3
    if unbatched:
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    if mask is not None:
        if mask.ndim == 1:
            mask = mask[None, None, None, :]
        elif mask.ndim == 2:
            mask = mask[:, None, None, :]
    out = attention(
        q, k, v, bias=bias, mask=mask, scale=q.shape[-1] ** -0.5, logits_float32=False
    )
    return out.squeeze(0) if unbatched else out
