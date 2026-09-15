"""Exercise the engine interfaces used by real ESMFold2 blocks."""

import pytest
import torch
from miniworld_engine.modules.swa_atom_attention import build_3d_rope
from team_gm.modules import ImplementationType

from foldforge.modules.sequence.atom_transformer import SWAAtomBlock
from foldforge.modules.sequence.msa_encoder import MSAEncoderBlock


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_msa_block_keeps_the_initialized_pair_residual(dtype):
    block = (
        MSAEncoderBlock(
            d_msa=16,
            d_pair=16,
            d_hidden=4,
            n_heads_msa=2,
            msa_head_width=8,
            is_final_block=True,
            implementation=ImplementationType.PYTORCH,
        )
        .to(dtype)
        .eval()
    )
    assert block.outer_product_mean.normalize_before_proj is False
    msa = torch.randn(1, 3, 4, 16, dtype=dtype)
    pair = torch.randn(1, 4, 4, 16, dtype=dtype)
    mask = torch.ones(1, 3, 4, dtype=torch.bool)
    with torch.no_grad():
        _, output = block(msa, pair, mask)
    torch.testing.assert_close(output, pair, atol=0, rtol=0)


def test_atom_block_uses_engine_attention_and_keeps_zero_gate_residual():
    block = SWAAtomBlock(64, 64, 2, half_window=2).eval()
    assert type(block.attn).__module__.startswith("miniworld_engine.")
    x, cond = torch.randn(1, 4, 64), torch.randn(1, 4, 64)
    cos, sin = build_3d_rope(
        torch.randn(1, 4, 3), torch.zeros(1, 4, dtype=torch.long), 32
    )
    valid = torch.ones(1, 4, dtype=torch.bool)
    params = (
        cos,
        sin,
        valid.sum(-1, dtype=torch.int32),
        torch.tensor([0, 4], dtype=torch.int32),
        4,
        valid,
    )
    with torch.no_grad():
        output = block(x, cond, params)
    torch.testing.assert_close(output, x, atol=0, rtol=0)
