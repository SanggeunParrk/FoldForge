"""Check real-checkpoint batched denoising against independent single calls."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from miniworld_engine.modules import AugmentedAttentionPairBias

from foldforge.models.inference import run
from foldforge.models.io import runtime


def main() -> int:
    assert os.environ.get("SLURM_JOB_ID")
    assert torch.cuda.is_available()
    report = {"calls": [], "attention_shapes": []}
    original_bind = runtime.Runtime.bind
    original_attention = AugmentedAttentionPairBias.forward
    recording = False

    def attention(
        module: AugmentedAttentionPairBias,
        single: torch.Tensor,
        cond: torch.Tensor,
        pair: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        # single [A, B, L, d], pair [B, L, L, d_pair]: A samples share one pair bias
        if recording:
            report["attention_shapes"].append([list(single.shape), list(pair.shape)])
        return original_attention(module, single, cond, pair, *args, **kwargs)

    def bind(owner: runtime.Runtime, model: Any) -> Any:
        assert not owner.request.execution.compile
        assert not owner.request.execution.cuda_graph
        original = model.diffusion_head.forward

        def denoise(**kwargs: Any) -> torch.Tensor:
            nonlocal recording
            coords = kwargs["positions_noisy"]
            report["calls"].append(list(coords.shape))
            recording = True
            actual = original(**kwargs)
            recording = False
            if "comparison" not in report:
                expected = torch.stack(
                    [
                        original(**{**kwargs, "positions_noisy": sample})
                        for sample in coords
                    ]
                )
                delta = actual.float() - expected.float()
                rms = (
                    delta.square().mean().sqrt()
                    / expected.float().square().mean().sqrt().clamp_min(1e-6)
                )
                report["comparison"] = {
                    "relative_rms": rms.item(),
                    "max_abs": delta.abs().max().item(),
                }
                assert torch.isfinite(actual).all()
                assert rms.item() < (
                    0.0005 if owner.request.precision == "af3_default" else 0.02
                )
            return actual

        model.diffusion_head.forward = denoise
        report["backend"] = owner.request.backend
        report["precision"] = owner.request.precision
        report["samples"] = model.num_samples
        report["steps"] = model.diffusion_steps
        return original_bind(owner, model)

    runtime.Runtime.bind = bind
    AugmentedAttentionPairBias.forward = attention
    try:
        status = run("af3", sys.argv[2:])
        assert len(report["calls"]) == report["steps"]
        assert all(s[0] == report["samples"] for s in report["calls"])
        if report["backend"] == "miniworld":
            assert report["attention_shapes"]
            assert all(
                q[0] == report["samples"] and b[0] == 1
                for q, b in report["attention_shapes"]
            )
        report["passed"] = True
        return status
    finally:
        runtime.Runtime.bind = original_bind
        AugmentedAttentionPairBias.forward = original_attention
        Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
