"""The AF3 atom attention GEMMs follow native parameter precision."""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
from team_gm.modules.checkpoints.af_family import configure_model
from torch.overrides import TorchFunctionMode

from foldforge.modules.dense.diffusion_transformer import CrossAttention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA precision validation"
)


class ObserveEinsum(TorchFunctionMode):
    def __init__(self) -> None:
        super().__init__()
        self.dtypes = []

    def __torch_function__(self, func, types, args=(), kwargs=None) -> Any:
        if func is torch.einsum:
            self.dtypes.append(
                tuple(x.dtype for x in args if isinstance(x, torch.Tensor))
            )
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cross_attention_native_gemm_and_fp32_reference(dtype):
    torch.manual_seed(14)
    source = CrossAttention()
    reference = configure_model(copy.deepcopy(source), "pytorch", dtype=torch.float32)
    model = configure_model(source, "pytorch", dtype=dtype)
    q = torch.randn(2, 4, 128, device="cuda").to(dtype)
    k = torch.randn(2, 8, 128, device="cuda").to(dtype)
    condq = torch.randn_like(q)
    condk = torch.randn_like(k)
    maskq = torch.ones(2, 4, device="cuda", dtype=torch.bool)
    maskk = torch.ones(2, 8, device="cuda", dtype=torch.bool)
    maskq[1, -1] = False
    maskk[1, -1] = False
    observed = ObserveEinsum()
    with torch.inference_mode():
        expected = reference(
            q.float(),
            k.float(),
            maskq,
            maskk,
            single_cond_q=condq.float(),
            single_cond_k=condk.float(),
        )
        with observed:
            actual = model(q, k, maskq, maskk, single_cond_q=condq, single_cond_k=condk)
    assert observed.dtypes == [(dtype, dtype), (dtype, dtype)]
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    assert not torch.is_autocast_enabled("cuda")
    torch.testing.assert_close(actual.float(), expected, atol=0.005, rtol=0.03)
