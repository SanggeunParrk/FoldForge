"""Native inference precision with norm affine parameters retained in FP32."""

import torch
from team_gm.modules.primitives import LayerNorm
from torch import nn


def _explicit_fp32_norms(model: nn.Module) -> None:
    # Plain PyTorch CUDA LayerNorm does not support every mixed-affine shape.
    # The shared reference norm explicitly reduces in FP32 and restores x.dtype.
    for name, child in list(model.named_children()):
        if type(child) is nn.LayerNorm:
            norm = LayerNorm(
                list(child.normalized_shape),
                eps=child.eps,
                elementwise_affine=child.elementwise_affine,
                bias=child.bias is not None,
            )
            norm.weight, norm.bias = child.weight, child.bias
            setattr(model, name, norm)
        else:
            _explicit_fp32_norms(child)


def inference_precision(
    model: nn.Module, device: torch.device, dtype: torch.dtype
) -> nn.Module:
    """Keep norm values exact and respect upstream device/buffer hooks."""
    _explicit_fp32_norms(model)
    norm_values = []
    for module in model.modules():
        is_norm = isinstance(module, (nn.LayerNorm, nn.RMSNorm))
        for name, parameter in module.named_parameters(recurse=False):
            if is_norm or name in {"layer_norm_weight", "layer_norm_bias"}:
                norm_values.append((module, name, parameter.detach().float().clone()))
    # ESMC RoPE's _apply materializes meta buffers and regenerates FP32 inv_freq.
    model.to(device=device, dtype=dtype)
    with torch.no_grad():
        for module, name, value in norm_values:
            getattr(module, name).data = value.to(device=device)
    model.requires_grad_(requires_grad=False)
    return model.eval()
