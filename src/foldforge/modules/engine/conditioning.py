"""Convert released AF-family conditioning weights into engine-owned modules."""

from __future__ import annotations

from copy import copy

import torch
from miniworld_engine.modules import AdaptiveLayerNorm, ConditionedTransition
from miniworld_engine.modules.dispatch import KernelBackend
from miniworld_engine.modules.exceptions import ImplementationType as EngineImpl
from torch import nn

from foldforge.layers.implementation import ImplementationType
from foldforge.modules.engine.backends import Support
from foldforge.modules.engine.installation import BuildResult, record_phase
from foldforge.modules.engine.native_linear import NativeLinear
from foldforge.modules.engine.policy import accept_support
from foldforge.modules.engine.routing import enabled_fp32

_UNBATCHED_TOKEN_RANK = 2
#: The engine Transition kernels normalise with this eps (its LayerNorm default).
_ENGINE_TRANSITION_EPS = 1e-5


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
    reference._modules = dict(layer._modules)
    reference._backend = KernelBackend.PYTORCH
    if isinstance(layer, ConditionedTransition):
        reference.ada_ln_in = _reference_copy(layer.ada_ln_in)
    return reference


def conditioning_support(layer: nn.Module, a: torch.Tensor, s: torch.Tensor) -> Support:
    """Retain projection operand compatibility independently of internal backend selection."""
    weight = layer.to_scale.weight
    if not a.is_cuda:
        return Support(refusal="cuda_required")
    if a.dtype != weight.dtype or s.dtype != weight.dtype:
        return Support(refusal="conditioning_operand_dtype")
    return Support()


def convert_adaln(source: nn.Module) -> AdaptiveLayerNorm:
    """Map sigmoid-scale AdaLN; keep exact FP32 norms and projection parameters."""
    norm = source.single_cond_layer_norm
    scale = source.single_cond_scale
    bias = source.single_cond_bias
    with torch.device("meta"):
        result = AdaptiveLayerNorm(
            scale.out_features,
            scale.in_features,
            implementation=_implementation(source),
            dtype=scale.weight.dtype,
        )
    result.ln_in = source.layer_norm
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
        layer = (
            self.layer
            if accept_support(self, conditioning_support(self.layer, a, cond))
            else self.reference
        )
        # Released AF3 passes [L,D]; engine shape keys require the semantic B axis.
        if a.ndim == _UNBATCHED_TOKEN_RANK:
            return layer(a.unsqueeze(0), cond.unsqueeze(0)).squeeze(0)
        return layer(a, cond)


class ReleasedConditionedTransition(nn.Module):
    """Checkpoint signature adapter; all transition computation belongs to engine."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        adaln = source.adaptive_layernorm
        scale = source.adaptive_zero_init.adaptive_zero_cond
        n = source.num_intermediate_factor
        with torch.device("meta"):
            self.layer = ConditionedTransition(
                scale.out_features,
                scale.in_features,
                n=n,
                implementation=_implementation(source),
                dtype=scale.weight.dtype,
            )
        self.layer.ada_ln_in = convert_adaln(adaln)
        wa, wb = source.transition1.weight.chunk(2, dim=0)
        self.layer.expand_a.weight = nn.Parameter(wa, requires_grad=False)
        self.layer.expand_b.weight = nn.Parameter(wb, requires_grad=False)
        self.layer.expand_a = NativeLinear.from_linear(self.layer.expand_a)
        self.layer.expand_b = NativeLinear.from_linear(self.layer.expand_b)
        self.layer.squeeze = source.adaptive_zero_init.transition2
        self.layer.to_scale = scale
        self.reference = _reference_copy(self.layer)
        self.eval().requires_grad_(requires_grad=False)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Return a raw delta: the engine module adds its own residual, the shared block adds the released one."""
        cond = s.expand(*a.shape[:-1], s.shape[-1])
        # Coordinate-derived streams may stay FP32. Preserve normalization before
        # the native projection cast; fused GEMMs require matching operand dtypes.
        layer = (
            self.layer
            if accept_support(self, conditioning_support(self.layer, a, cond))
            else self.reference
        )
        # Released AF3 passes [L,D]; engine shape keys require the semantic B axis.
        if a.ndim == _UNBATCHED_TOKEN_RANK:
            return layer.delta(a.unsqueeze(0), cond.unsqueeze(0)).squeeze(0)
        return layer.delta(a, cond)


class ReleasedTransition(nn.Module):
    """An unconditioned ``DiffusionTransition`` on the engine.

    The diffusion head's pair / single conditioning transitions are LayerNorm ->
    SwiGLU -> output projection: the engine's plain ``Transition``, which returns
    ``x + transition(x)``. Callers take ``residual(x)`` so the sum is the
    engine's (no ``y - x`` round trip in bf16); ``forward`` keeps the released
    delta signature for any other caller.
    """

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        from miniworld_engine.modules import Transition

        norm = source.adaptive_layernorm.layer_norm
        out = source.adaptive_zero_init.transition2
        with torch.device("meta"):
            self.layer = Transition(
                source.c_x,
                source.num_intermediate_factor,
                implementation=_implementation(source),
            )
        # The engine pins the norm affine to FP32 (exact for a bf16 checkpoint).
        ln = self.layer.ln_in
        ln.weight = nn.Parameter(norm.weight.detach().float(), requires_grad=False)
        ln.bias = nn.Parameter(norm.bias.detach().float(), requires_grad=False)
        ln.eps = norm.eps
        wa, wb = source.transition1.weight.chunk(2, dim=0)
        self.layer.expand_a.weight = nn.Parameter(wa.contiguous(), requires_grad=False)
        self.layer.expand_b.weight = nn.Parameter(wb.contiguous(), requires_grad=False)
        squeeze = out.weight.detach()
        self.layer.squeeze.weight = nn.Parameter(squeeze, requires_grad=False)
        self.source = source
        self.eval().requires_grad_(requires_grad=False)

    def _serves(self, x: torch.Tensor) -> bool:
        return accept_support(self, self._support(x))

    def _support(self, x: torch.Tensor) -> Support:
        """The original BF16/native-weight or explicit FP32/TF32 condition."""
        weight = self.layer.expand_a.weight.dtype
        if x.dtype == torch.float32:
            # FP32 activations and weights: the engine's TF32 tensor core kernel, where TF32 is allowed.
            accepted = weight == torch.float32 and enabled_fp32(self.source, x)
        else:
            accepted = (
                x.is_cuda
                and x.dtype == torch.bfloat16
                and weight in (torch.bfloat16, torch.float32)
            )
        return Support(
            refusal=None if accepted else "transition_operand_or_tf32_policy"
        )

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """``x + transition(x)``: the engine's kernel, else the released module."""
        if not self._serves(x):
            return x + self.source(x)
        return self.layer(x.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The released delta ``transition(x)``."""
        return self.source(x)


def install_conditioning(model: nn.Module) -> nn.Module:
    """Convert only mathematically matching AF-family modules after strict load."""
    counts = {
        "engine_adaln": 0,
        "engine_conditioned_transition": 0,
        "engine_transition": 0,
    }
    decisions: dict[str, dict] = {}

    def decision(path: str, candidate: str, result: BuildResult) -> None:
        """Record conversion without retaining the newly constructed module."""
        decisions[path] = {
            "candidate": candidate,
            "status": "converted" if result.value is not None else "refused",
            "reason": result.refusal,
        }

    def identity_scale(module: nn.Module) -> bool:
        """Whether this block scales by (s + 1) rather than sigmoid(s).

        The engine's AdaLN computes the sigmoid form and normalises its
        conditioning; neither is a reparameterisation of the other, so a block
        trained on the identity-centred form keeps the reference implementation.
        """
        norm = getattr(module, "adaptive_layernorm", module)
        return bool(getattr(norm, "identity_scale", False))

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            cls = type(child)
            # Exact class names: subclasses (the gated transition) keep their reference path.
            released_transition = cls.__name__ == "DiffusionTransition"
            released_norm = cls.__name__ == "AdaptiveLayerNorm" and hasattr(
                child, "layer_norm"
            )
            if released_transition:
                if not getattr(child, "use_single_cond", True):
                    # The diffusion head's pair / single conditioning transitions:
                    # LayerNorm (affine, the engine's eps) -> SwiGLU -> projection,
                    # the engine's plain Transition.
                    # By its parameters: configure_model swaps nn.LayerNorm for
                    # NativeLayerNorm, which is not a subclass.
                    norm = child.adaptive_layernorm.layer_norm
                    affine = [getattr(norm, p, None) for p in ("weight", "bias")]
                    if (
                        all(
                            isinstance(t, torch.Tensor) and t.shape == (child.c_x,)
                            for t in affine
                        )
                        and getattr(norm, "eps", None) == _ENGINE_TRANSITION_EPS
                    ):
                        result = BuildResult(value=ReleasedTransition(child))
                        decision(path, "unconditioned_transition", result)
                        setattr(parent, name, result.value)
                        counts["engine_transition"] += 1
                    else:
                        decision(
                            path,
                            "unconditioned_transition",
                            BuildResult(refusal="affine_norm_or_eps_mismatch"),
                        )
                        visit(child, path)
                    continue
                if identity_scale(child):
                    decision(
                        path,
                        "conditioned_transition",
                        BuildResult(refusal="identity_scale_unsupported"),
                    )
                    visit(child, path)
                    continue
                result = BuildResult(value=ReleasedConditionedTransition(child))
                decision(path, "conditioned_transition", result)
                setattr(parent, name, result.value)
                counts["engine_conditioned_transition"] += 1
                counts["engine_adaln"] += 1
            elif released_norm:
                if not getattr(child, "use_single_cond", True) or identity_scale(child):
                    decision(
                        path,
                        "adaln",
                        BuildResult(refusal="unconditioned_or_identity_scale"),
                    )
                    continue
                result = BuildResult(value=ReleasedAdaLN(child))
                decision(path, "adaln", result)
                setattr(parent, name, result.value)
                counts["engine_adaln"] += 1
            else:
                visit(child, path)

    visit(model)
    record_phase(model, "conditioning", decisions)
    model.foldforge_load_report.update(counts)
    return model.eval().requires_grad_(requires_grad=False)
