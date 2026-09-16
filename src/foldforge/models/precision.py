"""Native inference precision with norm affine parameters retained in FP32."""

from functools import wraps
from typing import Any

import torch
from team_gm.modules.primitives import LayerNorm
from torch import nn
from torch.utils._pytree import tree_map


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


def _amp_method(owner: nn.Module, name: str, *, float_output: bool = False) -> None:
    original = getattr(owner, name)

    @wraps(original)
    def forward(*args: Any, **kwargs: Any) -> Any:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = original(*args, **kwargs)
        if float_output:
            result = tree_map(
                lambda x: (
                    x.float()
                    if isinstance(x, torch.Tensor) and x.is_floating_point()
                    else x
                ),
                result,
            )
        return result

    setattr(owner, name, forward)


def reference_precision(model: Any, name: str) -> nn.Module:
    """Apply released CUDA precision scopes to FP32 reference parameters.

    Native BF16 modes never install these wrappers. Reference mode deliberately
    preserves FP32 parameter storage and uses the original selective autocast.
    """
    scopes = []
    if name == "protenix":
        model.configs.skip_amp.sample_diffusion = True
        model.configs.skip_amp.confidence_head = False
        _amp_method(model, "forward")
        scopes = ["forward (diffusion explicitly disables autocast)"]
    elif name == "esmfold2":
        for path in ("inputs_embedder", "pair_trunk", "confidence_head.folding_trunk"):
            _amp_method(model.get_submodule(path), "forward", float_output=True)
            scopes.append(path)
        conditioning = model.get_submodule(
            "structure_head.diffusion_module.conditioning"
        )
        for index, block in enumerate(conditioning.pair_transitions):
            _amp_method(block, "forward", float_output=True)
            scopes.append(
                f"structure_head.diffusion_module.conditioning.pair_transitions.{index}"
            )
    elif name != "opendde":
        message = f"No reference precision policy for {name}"
        raise ValueError(message)
    model.reference_autocast_scopes = scopes
    model.reference_tf32 = True
    return model
