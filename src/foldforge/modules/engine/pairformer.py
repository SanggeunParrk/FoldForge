"""Released-checkpoint interface around the engine's shared Pairformer composition.

Only the signature and the upstream raw-delta contracts live here: the ordering of
Pairformer operations is owned by the engine, the numerics by the wrapped dense modules.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, cast

import torch
from torch import nn

from foldforge.blocks.pairformer import PairformerBlock
from foldforge.modules.engine.installation import record_phase
from foldforge.modules.engine.modules import engine_of, token_mask
from foldforge.modules.engine.policy import accept_support, backend_support
from foldforge.modules.engine.routing import (
    cueq_single_attention,
    served_by,
    single_attention_support,
    transition_residual,
    triangle_residual,
)
from foldforge.modules.engine.serving import (
    ADAPTER_ATTR,
    Dispatch,
    EngineAdapter,
    observe,
)

#: A bias over every head that the stack supplying it builds (a statement about token
#: pairs, not about heads); scoped to one block call.
_EXTRA_PAIR_BIAS = ContextVar("released_pairformer_extra_pair_bias", default=None)


def _per_sample(single, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """B samples through the one-sample path: the fallback when no batched engine module serves."""
    return torch.stack(
        [single(x[i], None if mask is None else mask[i]) for i in range(x.shape[0])]
    )


class _Transition(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.source = source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return transition_residual(self.source, x)


class _TriangleMultiplication(nn.Module):
    """The engine block is batched; the released module is not."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.source = source

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if x.shape[0] == 1:
            return triangle_residual(
                self.source, x[0], None if mask is None else mask[0]
            )[None]
        # B > 1 (the template slots): one batched engine call, else B one-sample calls.
        built = served_by(self.source, x)
        if built is not None and not getattr(self.source, "divide_by_length", False):
            return built(x, token_mask(mask))
        return _per_sample(lambda v, m: triangle_residual(self.source, v, m), x, mask)


class _TriangleAttention(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.source = source

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if x.shape[0] == 1:
            return self.source.residual(x[0], None if mask is None else mask[0])[None]
        built = served_by(self.source, x)
        if built is not None:
            return built(x, token_mask(mask))
        return _per_sample(self.source.residual, x, mask)


class _SingleAttention(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.norm = source.single_pair_logits_norm
        self.projection = source.single_pair_logits_projection
        self.attention = cast("nn.Module", source.single_attention_)

    def forward(
        self, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        bias = _EXTRA_PAIR_BIAS.get()
        built = engine_of(self)
        supported = accept_support(self, single_attention_support(self, single, bias))
        if built is not None and supported:
            # The engine module owns the residual and reaches the hand-CUDA legs.
            return built(single, pair, None if mask is None else mask > 0)
        if bias is None and backend_support(self.attention, "cuequivariance").accepted:
            served = cueq_single_attention(self, single, pair, mask)
            if served is not None:
                return served
        if single.shape[0] > 1:
            # B samples (the confidence head's diffusion samples) without an engine module.
            return torch.stack(
                [
                    self(
                        single[i : i + 1],
                        pair[i : i + 1],
                        None if mask is None else mask[i : i + 1],
                    )
                    for i in range(single.shape[0])
                ]
            ).squeeze(1)
        logits = self.projection(self.norm(pair[0])).permute(2, 0, 1)
        if bias is not None:
            logits = logits + bias[None].to(logits.dtype)
        delta = self.attention(
            single[0], None if mask is None else mask[0], pair_logits=logits
        )
        return single + delta[None]


def _serve_pairformer(
    _built: nn.Module, module: AF3Pairformer, *inputs: Any
) -> Dispatch:
    """Run the existing composition; constituent kernels retain their own routing."""
    return Dispatch(module._forward(*inputs), selection="engine_composition")


class AF3Pairformer(nn.Module):
    """AF3 unbatched pair-first signature around the engine's ``PairformerBlock``.

    A pair-only block (no single track) also takes ``[B, L, L, C]``: B samples that share one
    ``[L, L]`` mask, which is how the template slots run as one batch.
    """

    #: The pair-only block accepts B > 1 samples (template slots) in one call.
    supports_template_batch = True
    #: The block with a single track accepts S > 1 samples (the confidence head's diffusion
    #: samples, one shared pair mask and token mask) in one call.
    supports_sample_batch = True

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        with_single = source.with_single
        attention_start = getattr(source, "pair_attention1", None)
        attention_end = getattr(source, "pair_attention2", None)
        self.block = PairformerBlock.from_components(
            triangle_outgoing=_TriangleMultiplication(
                source.triangle_multiplication_outgoing
            ),
            triangle_incoming=_TriangleMultiplication(
                source.triangle_multiplication_incoming
            ),
            attention_starting=None
            if attention_start is None
            else _TriangleAttention(attention_start),
            attention_ending=None
            if attention_end is None
            else _TriangleAttention(attention_end),
            transition_pair=_Transition(source.pair_transition),
            attention_single=_SingleAttention(source) if with_single else None,
            transition_single=_Transition(source.single_transition)
            if with_single
            else None,
            **({"parallel": True} if getattr(source, "parallel", False) else {}),
        )
        object.__setattr__(
            self,
            ADAPTER_ATTR,
            EngineAdapter(self.block, "pairformer", _serve_pairformer),
        )
        self.eval()

    def forward(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        single: torch.Tensor | None = None,
        seq_mask: torch.Tensor | None = None,
        extra_pair_bias: torch.Tensor | None = None,
    ) -> Any:
        """Use the shared adapter result and host observation contract."""
        adapter = self.__dict__[ADAPTER_ATTR]
        return observe(
            self,
            adapter.try_run(self, pair, pair_mask, single, seq_mask, extra_pair_bias),
        )

    def _forward(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        single: torch.Tensor | None = None,
        seq_mask: torch.Tensor | None = None,
        extra_pair_bias: torch.Tensor | None = None,
    ) -> Any:
        """The released signature; under bf16 autocast an fp32 stream is handed to the engine in bf16.

        A ``bf16-mixed`` model keeps its pair and single streams in fp32 between the blocks, while the engine modules
        serve bf16 only (they hold bf16 weights): left alone, the whole stack would fall back to the dense path. The
        blocks get the streams in bf16, as MiniWorld's training hands its confidence Pairformer (``z.to(bfloat16)``),
        and the result goes back to the dtype it came in.
        """
        if (
            pair.dtype == torch.float32
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.bfloat16
        ):
            out = self._run(
                pair.bfloat16(),
                pair_mask,
                None if single is None else single.bfloat16(),
                seq_mask,
                extra_pair_bias,
            )
            if single is None or not isinstance(out, tuple):
                return out.to(pair.dtype)
            return out[0].to(pair.dtype), out[1].to(single.dtype)
        return self._run(pair, pair_mask, single, seq_mask, extra_pair_bias)

    def _run(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        single: torch.Tensor | None = None,
        seq_mask: torch.Tensor | None = None,
        extra_pair_bias: torch.Tensor | None = None,
    ) -> Any:
        """Run the released AF3 signature through the common block.

        ``extra_pair_bias`` reaches the single attention through a scoped variable,
        which is what lets a fused block take a bias the released signature has no
        argument for.
        """
        if self.training:
            message = "Released Pairformer adapter is qualified for inference only"
            raise RuntimeError(message)
        if pair.ndim == 4:
            if single is not None:
                # [S, L, L, C] pair + [S, L, C] single: the diffusion samples of one fold.
                token = _EXTRA_PAIR_BIAS.set(extra_pair_bias)
                try:
                    batched_pair, batched_single = self.block(
                        pair,
                        single,
                        None
                        if seq_mask is None
                        else seq_mask.expand(pair.shape[0], -1),
                        pair_mask=None
                        if pair_mask is None
                        else pair_mask.bool().expand(pair.shape[0], -1, -1),
                    )
                finally:
                    _EXTRA_PAIR_BIAS.reset(token)
                return batched_pair, batched_single
            batched, _ = self.block(
                pair,
                None,
                None,
                pair_mask=None
                if pair_mask is None
                else pair_mask.bool().expand(pair.shape[0], -1, -1),
            )
            return batched
        token = _EXTRA_PAIR_BIAS.set(extra_pair_bias)
        try:
            pair, single = self.block(
                pair[None],
                None if single is None else single[None],
                None if seq_mask is None else seq_mask[None],
                pair_mask=None if pair_mask is None else pair_mask[None].bool(),
            )
        finally:
            _EXTRA_PAIR_BIAS.reset(token)
        return (pair[0], single[0]) if single is not None else pair[0]


def install_pairformers(model: nn.Module) -> nn.Module:
    """Replace released Pairformer compositions after strict checkpoint loading."""
    converted = 0
    decisions: dict[str, Any] = {}

    def visit(parent: nn.Module, prefix: str = "") -> None:
        nonlocal converted
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            # The shared block threads each update through the running
            # activation, or (``parallel``, Chai-1) has every update read the
            # representation ENTERING it: both schedules are the block's.
            if type(child).__name__ == "PairformerBlock" and hasattr(
                child, "triangle_multiplication_outgoing"
            ):
                decisions[path] = {
                    "candidate": "pairformer_composition",
                    "status": "converted",
                    "reason": None,
                }
                setattr(parent, name, AF3Pairformer(child))
                converted += 1
            else:
                visit(child, path)

    visit(model)
    record_phase(model, "pairformers", decisions)
    model.foldforge_load_report["engine_pairformer_blocks"] = converted
    return model
