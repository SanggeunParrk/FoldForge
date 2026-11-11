"""``engine_activation``: an fp32 activation enters the engine's master modules as the autocast dtype, never as fp32.

The engine's ``MasterWeights`` contract is fp32 masters with bf16 activations. FoldForge handed fp32 tensors to TriMul / TriAttn /
Transition / AttentionPairBias / AtomDiT in a bf16 run; their cuda backends refuse fp32 and the unchunked pytorch fallback allocated
N^3 fp32 triangle-attention logits (the 5I28 out-of-memory cells at 1024+ tokens).
"""

from __future__ import annotations

import torch

from foldforge.modules.dense import fused_template
from foldforge.modules.engine.configure import engine_activation


def test_fp32_becomes_the_autocast_dtype_inside_autocast():
    x = torch.randn(2, 3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = engine_activation(x)
    assert y.dtype is torch.bfloat16
    torch.testing.assert_close(y.float(), x.to(torch.bfloat16).float(), rtol=0, atol=0)


def test_fp32_precision_runs_without_autocast_and_stays_fp32():
    x = torch.randn(2, 3)
    assert engine_activation(x) is x


def test_other_dtypes_masks_and_none_are_left_alone():
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert engine_activation(None) is None
        half = torch.randn(2, 3).bfloat16()
        assert engine_activation(half) is half
        mask = torch.ones(3, dtype=torch.bool)
        assert engine_activation(mask) is mask
        index = torch.arange(3)
        assert engine_activation(index) is index


def test_a_disabled_autocast_region_inside_an_enabled_one_keeps_fp32():
    x = torch.randn(2, 3)
    with torch.autocast("cpu", dtype=torch.bfloat16), torch.autocast("cpu", enabled=False):
        assert engine_activation(x) is x


def test_rf3_template_distogram_edges_are_the_released_ones():
    edges = fused_template._RF3_DISTOGRAM_EDGES  # noqa: SLF001
    released = torch.cat([torch.arange(1.0, 4.0, 0.1), torch.arange(4.0, 20.5, 0.5)])
    assert edges.shape == (63,)  # 64 distogram bins
    torch.testing.assert_close(edges, released, rtol=0, atol=0)
    assert edges.device.type == "cpu"  # moved to the device once, through on_device (cached), never inside a CUDA-graph capture
