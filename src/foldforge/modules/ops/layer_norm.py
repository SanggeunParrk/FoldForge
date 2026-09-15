import numbers

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch import Size
from torch.nn import functional as F

from foldforge.modules.ops import config as fastnn_config

_shape_t = int | list[int] | Size


# modified from triton/python/tutorials/05-layer-norm.py
@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    USE_WEIGHTS: tl.constexpr,
    USE_BIAS: tl.constexpr,
) -> None:
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * N
    X += row * N
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        if USE_WEIGHTS is True:
            w = tl.load(W + cols, mask=mask)
            if USE_BIAS is True:
                b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        if USE_WEIGHTS is True:
            y = x_hat * w
            if USE_BIAS is True:
                y += b
        else:
            y = x_hat
        # Write output
        tl.store(Y + cols, y, mask=mask)


class LayerNormTritonFunc(torch.autograd.Function):
    """Represent layer norm triton func."""

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):  # noqa: ARG004 - shared callback or fixture signature
        """Compute the module output."""
        x = x.contiguous()
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        N = x.shape[-1]
        M = x.numel() // N
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            msg = "This layer norm doesn't support feature dim >= 64KB."
            raise RuntimeError(msg)
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        use_weight = weight is not None
        use_bias = bias is not None
        _layer_norm_fwd_fused.run(
            x,
            y,
            weight,
            bias,
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            USE_WEIGHTS=use_weight,
            USE_BIAS=use_bias,
            num_warps=num_warps,
            num_ctas=1,
            grid=(M,),
            warmup=False,
        )
        return y


class LayerNorm(nn.Module):
    """Represent layer norm."""

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Compute reset parameters."""
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002 - external interface keyword
        """Compute the module output."""
        if fastnn_config.layer_norm_implementation == "torch":
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )
        if fastnn_config.layer_norm_implementation == "triton":
            fast_out = LayerNormTritonFunc.apply(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )
            if not isinstance(fast_out, torch.Tensor):
                message = "Layer norm must return a tensor"
                raise TypeError(message)
            return fast_out
        msg = f"fastnn_config must be 'torch' or 'triton', got {fastnn_config}"
        raise ValueError(msg)

    def extra_repr(self) -> str:
        """Compute extra repr."""
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )
