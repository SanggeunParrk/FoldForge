"""Observe real-checkpoint sample axes outside compilation and timed forwards."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from miniworld_engine import kernels, ops

from foldforge.models.inference import run
from foldforge.models.io import runtime
from foldforge.modules.sequence import atom_encoder


def main() -> int:
    assert os.environ.get("SLURM_JOB_ID")
    assert torch.cuda.is_available()
    output, family = Path(sys.argv[1]), sys.argv[2]
    report: dict[str, Any] = {
        "model": family,
        "calls": [],
        "token_attention": [],
        "swa": [],
    }
    recording = False
    old_bind = runtime.Runtime.bind
    old_dense = ops.augmented_attention_pair_bias
    old_aug = kernels.triton_augmented_attention_pair_bias
    old_swa = atom_encoder.build_attention_params

    def dense(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if recording:
            report["token_attention"].append(
                {"q": list(q.shape), "bias": list(bias.shape), "layout": "ABHLD"}
            )
        return old_dense(q, k, v, bias, mask)

    def augmented(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if recording:
            report["token_attention"].append(
                {"q": list(q.shape), "bias": list(bias.shape), "layout": "ABLHD"}
            )
        return old_aug(q, k, v, bias, mask, **kwargs)

    def swa(
        cos: torch.Tensor, sin: torch.Tensor, valid: torch.Tensor, num_aug: int
    ) -> tuple:
        if num_aug > 1:
            report["swa"].append(
                {"cos": list(cos.shape), "valid": list(valid.shape), "num_aug": num_aug}
            )
        return old_swa(cos, sin, valid, num_aug)

    def bind(owner: runtime.Runtime, model: Any) -> Any:
        assert owner.request.backend == "miniworld"
        assert not owner.request.execution.compile
        assert not owner.request.execution.cuda_graph
        report["samples"] = owner.request.samples
        if family == "esmfold2":
            denoiser, method = model.structure_head.diffusion_module, "denoise"
            coord_name = "scaled_coords"
        else:
            denoiser, method = model.diffusion_module, "forward"
            coord_name = "x_noisy"
        original = getattr(denoiser, method)

        def forward(*args: Any, **kwargs: Any) -> torch.Tensor:
            nonlocal recording
            coords = kwargs.get(coord_name, args[0] if args else None)
            report["calls"].append(list(coords.shape))
            recording = True
            try:
                result = original(*args, **kwargs)
                assert torch.isfinite(result).all()
                return result
            finally:
                recording = False

        setattr(denoiser, method, forward)
        return old_bind(owner, model)

    runtime.Runtime.bind = bind
    ops.augmented_attention_pair_bias = dense
    kernels.triton_augmented_attention_pair_bias = augmented
    atom_encoder.build_attention_params = swa
    try:
        status = run(family, sys.argv[3:])
        assert status == 0
        assert report["calls"]
        assert all(x[0] == report["samples"] for x in report["calls"])
        assert report["token_attention"]
        assert all(
            x["q"][0] == report["samples"] and x["bias"][0] == 1
            for x in report["token_attention"]
        )
        if family == "esmfold2":
            assert report["swa"]
            assert all(
                x["num_aug"] == report["samples"]
                and x["cos"][0] == 1
                and x["valid"][0] == report["samples"]
                for x in report["swa"]
            )
        report["passed"] = True
        return status
    finally:
        runtime.Runtime.bind = old_bind
        ops.augmented_attention_pair_bias = old_dense
        kernels.triton_augmented_attention_pair_bias = old_aug
        atom_encoder.build_attention_params = old_swa
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
