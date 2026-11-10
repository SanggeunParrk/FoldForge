"""One H100: the ``miniworld`` policy gives the engine's own kernels and they agree with the engine's ``pytorch`` backend on master weights.

Master parameters (fp32) with bf16 activations: the engine modules run their kernels under bf16 autocast / bf16 GEMMs themselves; no model-level
cast is involved. Other cards: the policy gives ``triton`` and the test only checks that the guard serves or refuses by name.
"""

from __future__ import annotations

import copy

import pytest
import torch

from foldforge.modules.engine import backends, policy
from foldforge.modules.engine.pairformer import pairformer_block
from foldforge.modules.engine.serving import report_observations

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def rms(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-6)).item()


def test_pairformer_block_policy_matches_pytorch():
    torch.manual_seed(1)
    block = pairformer_block(d_pair=128, triangle_model="af3", d_single=384, n_head_single=16).cuda().eval()
    with torch.no_grad():
        for p in block.parameters():
            if p.ndim > 1:
                p.normal_(0, 0.05)
    reference = copy.deepcopy(block)
    plan = backends.BackendPlan.make("miniworld").resolve(torch.device("cuda"))
    choices = policy.apply_plan(torch.nn.ModuleList([block]), plan, torch.device("cuda"))
    assert choices, "no engine op module found"
    pair = torch.randn(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    single = torch.randn(1, 128, 384, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 128, device="cuda", dtype=torch.bool)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # master weights: bf16 activations under autocast
        expected = reference(pair.clone(), mask, single.clone())
        actual = block(pair.clone(), mask, single.clone())
    assert rms(actual[0], expected[0]) < 0.03 and rms(actual[1], expected[1]) < 0.03
    served = report_observations(block)["modules"]
    ran = {name: stats["served_host_calls"] for name, stats in served.items()}
    print(ran)
    major = torch.cuda.get_device_capability()[0]
    if major == 9:
        assert any("cuda" in calls for calls in ran.values()), ran
