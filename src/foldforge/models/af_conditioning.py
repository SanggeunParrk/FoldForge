"""Convert released AF-family conditioning weights into engine-owned modules."""

from __future__ import annotations

from copy import copy

import torch
from miniworld_engine.modules import AdaptiveLayerNorm, ConditionedTransition
from miniworld_engine.modules.dispatch import KernelBackend
from miniworld_engine.modules.exceptions import ImplementationType as EngineImpl
from team_gm.modules.exceptions import ImplementationType
from team_gm.modules.precision import NativeLinear
from torch import nn

_UNBATCHED_TOKEN_RANK = 2


def _implementation(source: nn.Module) -> EngineImpl:
    return (
        EngineImpl.MINIWORLD
        if source.foldforge_implementation == ImplementationType.MINIWORLD_ENGINE
        else EngineImpl.PYTORCH
    )


def _reference_copy(layer: nn.Module) -> nn.Module:
    """Share exact loaded parameters while selecting the engine's reference math."""
    reference = copy(layer)
    # A separate child registry prevents changing the optimized module's AdaLN.
    reference._modules = dict(layer._modules)  # noqa: SLF001 - isolate nn.Module child registry
    reference._backend = KernelBackend.PYTORCH  # noqa: SLF001 - pinned engine reference dispatch
    if isinstance(layer, ConditionedTransition):
        reference.ada_ln_in = _reference_copy(layer.ada_ln_in)
    return reference


def _compatible(layer: nn.Module, a: torch.Tensor, s: torch.Tensor) -> bool:
    weight = layer.to_scale.weight
    return a.is_cuda and a.dtype == weight.dtype and s.dtype == weight.dtype


def convert_adaln(source: nn.Module) -> AdaptiveLayerNorm:
    """Map sigmoid-scale AdaLN; keep exact FP32 norms and projection parameters."""
    af3 = hasattr(source, "single_cond_layer_norm")
    norm = source.single_cond_layer_norm if af3 else source.layernorm_s
    scale = source.single_cond_scale if af3 else source.linear_s
    bias = source.single_cond_bias if af3 else source.linear_nobias_s
    with torch.device("meta"):
        result = AdaptiveLayerNorm(
            scale.out_features,
            scale.in_features,
            implementation=_implementation(source),
            dtype=scale.weight.dtype,
        )
    result.ln_in = source.layer_norm if af3 else source.layernorm_a
    result.ln_cond = norm
    result.to_scale, result.to_bias = scale, bias
    return result.eval()


class ReleasedAdaLN(nn.Module):
    """Released argument names and conditioning broadcast around engine AdaLN."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.layer = convert_adaln(source)
        self.reference = _reference_copy(self.layer)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Apply the converted normalization with explicit broadcast dimensions."""
        cond = s.expand(*a.shape[:-1], s.shape[-1])
        # Coordinate-derived streams may stay FP32. Preserve normalization before
        # the native projection cast; fused GEMMs require matching operand dtypes.
        layer = self.layer if _compatible(self.layer, a, cond) else self.reference
        # Released AF3 passes [L,D]; engine shape keys require the semantic B axis.
        if a.ndim == _UNBATCHED_TOKEN_RANK:
            return layer(a.unsqueeze(0), cond.unsqueeze(0)).squeeze(0)
        return layer(a, cond)


class ReleasedConditionedTransition(nn.Module):
    """Checkpoint signature adapter; all transition computation belongs to engine."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        af3 = hasattr(source, "adaptive_layernorm")
        adaln = source.adaptive_layernorm if af3 else source.adaln
        scale = source.adaptive_zero_init.adaptive_zero_cond if af3 else source.linear_s
        n = source.num_intermediate_factor if af3 else source.n
        with torch.device("meta"):
            self.layer = ConditionedTransition(
                scale.out_features,
                scale.in_features,
                n=n,
                implementation=_implementation(source),
                dtype=scale.weight.dtype,
            )
        self.layer.ada_ln_in = convert_adaln(adaln)
        if af3:
            wa, wb = source.transition1.weight.chunk(2, dim=0)
            self.layer.expand_a.weight = nn.Parameter(wa, requires_grad=False)
            self.layer.expand_b.weight = nn.Parameter(wb, requires_grad=False)
            self.layer.expand_a = NativeLinear.from_linear(self.layer.expand_a)
            self.layer.expand_b = NativeLinear.from_linear(self.layer.expand_b)
            self.layer.squeeze = source.adaptive_zero_init.transition2
        else:
            self.layer.expand_a = source.linear_nobias_a1
            self.layer.expand_b = source.linear_nobias_a2
            self.layer.squeeze = source.linear_nobias_b
        self.layer.to_scale = scale
        self.reference = _reference_copy(self.layer)
        self.eval().requires_grad_(requires_grad=False)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Return a raw delta; the shared block owns the residual addition."""
        cond = s.expand(*a.shape[:-1], s.shape[-1])
        # Coordinate-derived streams may stay FP32. Preserve normalization before
        # the native projection cast; fused GEMMs require matching operand dtypes.
        layer = self.layer if _compatible(self.layer, a, cond) else self.reference
        # Released AF3 passes [L,D]; engine shape keys require the semantic B axis.
        if a.ndim == _UNBATCHED_TOKEN_RANK:
            return layer(a.unsqueeze(0), cond.unsqueeze(0)).squeeze(0)
        return layer(a, cond)


def install_conditioning(model: nn.Module) -> nn.Module:
    """Convert only mathematically matching AF-family modules after strict load."""
    counts = {"engine_adaln": 0, "engine_conditioned_transition": 0}

    def visit(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            cls = type(child)
            if not cls.__module__.startswith("foldforge.models."):
                visit(child)
                continue
            if cls.__name__ in {"DiffusionTransition", "ConditionedTransitionBlock"}:
                if not getattr(child, "use_single_cond", True):
                    continue
                setattr(parent, name, ReleasedConditionedTransition(child))
                counts["engine_conditioned_transition"] += 1
                counts["engine_adaln"] += 1
            elif cls.__name__ == "AdaptiveLayerNorm":
                if not getattr(child, "use_single_cond", True):
                    continue
                setattr(parent, name, ReleasedAdaLN(child))
                counts["engine_adaln"] += 1
            else:
                visit(child)

    visit(model)
    model.foldforge_load_report.update(counts)
    return model.eval().requires_grad_(requires_grad=False)
