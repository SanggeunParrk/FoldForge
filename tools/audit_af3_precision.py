"""Audit native AF3 parameter and matrix operand dtypes outside timed runs.

Usage: audit_af3_precision.py REPORT --spec INPUT --config CONFIG --out RUN.
Use eager execution, one recycle and one diffusion step for this diagnostic;
measure latency separately with benchmark_end_to_end.py without these hooks.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_leaves

from foldforge.models.inference import run
from foldforge.models.io import runtime

if TYPE_CHECKING:
    from collections.abc import Callable


def signature(value: Any) -> tuple:
    return tuple(
        (str(x.dtype), tuple(x.shape))
        for x in tree_leaves(value)
        if isinstance(x, torch.Tensor)
    )


class PrecisionAudit(TorchFunctionMode):
    """Observe functional GEMM operands without changing values or selecting kernels."""

    def __init__(self, output: Path) -> None:
        super().__init__()
        self.family = "af3"
        self.output = output
        self.stack: list[str] = []
        self.operations: Counter = Counter()
        self.module_io: Counter = Counter()
        self.report: dict[str, Any] = {}
        self.handles: list[Any] = []

    def __torch_function__(
        self, func: Any, types: Any, args: tuple = (), kwargs: Any = None
    ) -> Any:
        name = getattr(func, "__name__", str(func))
        if name in {
            "linear",
            "einsum",
            "matmul",
            "__matmul__",
            "mm",
            "bmm",
            "addmm",
            "scaled_dot_product_attention",
        }:
            key = (
                self.stack[-1] if self.stack else "<functional>",
                name,
                signature(args),
                torch.is_autocast_enabled("cuda"),
            )
            self.operations[key] += 1
        return func(*args, **(kwargs or {}))

    def install(self, model: torch.nn.Module) -> None:
        self.report["load_report"] = getattr(model, "foldforge_load_report", {})
        self.report["reference_precision"] = getattr(
            model, "reference_precision", False
        )
        norm_ids = {
            id(p)
            for module in model.modules()
            if "LayerNorm" in type(module).__name__
            or "RMSNorm" in type(module).__name__
            for p in module.parameters(recurse=False)
        }
        self.report["parameters"] = [
            {
                "name": n,
                "dtype": str(p.dtype),
                "shape": list(p.shape),
                "numel": p.numel(),
                "norm": id(p) in norm_ids
                or n.endswith(("layer_norm_weight", "layer_norm_bias")),
            }
            for n, p in model.named_parameters()
        ]
        self.report["buffers"] = [
            {"name": n, "dtype": str(p.dtype), "shape": list(p.shape)}
            for n, p in model.named_buffers()
        ]
        self.report["linear_compute_overrides"] = [
            {"name": n, "compute_dtype": str(m.compute_dtype)}
            for n, m in model.named_modules()
            if getattr(m, "compute_dtype", None) is not None
        ]
        seen_hooks = set()
        for name, module in model.named_modules():
            # The reference conditioning adapters shallow-copy their modules;
            # registering twice on their shared hook registry duplicates callbacks.
            registry = id(module._forward_hooks)  # noqa: SLF001 - diagnostic deduplication
            if registry in seen_hooks:
                continue
            seen_hooks.add(registry)

            def before(_module: Any, _args: Any, name: str = name) -> None:
                self.stack.append(name)

            def after(module: Any, args: Any, value: Any, name: str = name) -> None:
                if isinstance(module, torch.nn.Linear) or name.startswith(
                    "diffusion_head"
                ):
                    self.module_io[name, signature(args), signature(value)] += 1
                assert self.stack.pop() == name

            self.handles.append(module.register_forward_pre_hook(before))
            self.handles.append(module.register_forward_hook(after))

    def save(self, *, completed: bool) -> None:
        self.report.update(
            completed=completed,
            timings_are_diagnostic=True,
            operations=[
                {
                    "module": k[0],
                    "op": k[1],
                    "inputs": k[2],
                    "autocast": k[3],
                    "calls": v,
                }
                for k, v in self.operations.items()
            ],
            module_io=[
                {"module": k[0], "inputs": k[1], "outputs": k[2], "calls": v}
                for k, v in self.module_io.items()
            ],
            autocast_observed=any(k[3] for k in self.operations),
            gpu=torch.cuda.get_device_name(),
            job=os.environ["SLURM_JOB_ID"],
        )
        diffusion_ops = [
            x
            for x in self.report["operations"]
            if x["module"].startswith("diffusion_head")
        ]
        checks = {
            "autocast_off": not self.report["autocast_observed"],
            "native_parameters": all(
                p["dtype"] == ("torch.float32" if p["norm"] else "torch.bfloat16")
                for p in self.report["parameters"]
            ),
            "no_compute_override": not self.report["linear_compute_overrides"],
            "diffusion_gemms_bf16": bool(diffusion_ops)
            and all(
                t[0] == "torch.bfloat16" for x in diffusion_ops for t in x["inputs"]
            ),
            "dit_residual_bf16": any(
                x["module"] == "diffusion_head.transformer"
                and bool(x["outputs"])
                and all(t[0] == "torch.bfloat16" for t in x["outputs"])
                for x in self.report["module_io"]
            ),
        }
        if self.report["reference_precision"]:
            params = self.report["parameters"]

            def expected(p: dict) -> str:
                name = p["name"]
                fp32 = (
                    p["norm"]
                    or name.startswith(
                        (
                            "diffusion_head.",
                            "evoformer_conditioning.",
                            "distogram_head.",
                        )
                    )
                    or any(
                        name.startswith("confidence_head." + head + ".")
                        for head in (
                            "plddt_logits",
                            "pae_logits",
                            "left_half_distance_logits",
                            "experimentally_resolved_logits",
                        )
                    )
                )
                return "torch.float32" if fp32 else "torch.bfloat16"

            checks = {
                "autocast_off": not self.report["autocast_observed"],
                "released_parameter_dtypes": all(
                    p["dtype"] == expected(p) for p in params
                ),
                "diffusion_gemms_fp32": bool(diffusion_ops)
                and all(
                    t[0] == "torch.float32" for x in diffusion_ops for t in x["inputs"]
                ),
                "trunk_linear_bf16": any(
                    x["module"].startswith("evoformer.") and x["op"] == "linear"
                    for x in self.report["operations"]
                )
                and all(
                    t[0] == "torch.bfloat16"
                    for x in self.report["operations"]
                    if x["module"].startswith("evoformer.") and x["op"] == "linear"
                    for t in x["inputs"]
                ),
            }
        if self.family != "af3":
            reference = (
                self.report["load_report"].get("precision_policy") == "model_default"
            )
            expect_amp = reference and self.family in {"esmfold2", "protenix"}
            linears = [
                x
                for x in self.report["operations"]
                if x["op"] == "linear"
                and x["module"].startswith(
                    ("diffusion_module.", "structure_head.diffusion_module.")
                )
                and "pair_transitions" not in x["module"]
            ]
            checks = {
                "parameter_policy": all(
                    p["dtype"]
                    == ("torch.float32" if reference or p["norm"] else "torch.bfloat16")
                    for p in self.report["parameters"]
                ),
                "autocast_policy": self.report["autocast_observed"] == expect_amp,
                "no_native_compute_override": reference
                or not self.report["linear_compute_overrides"],
                "denoiser_linears_observed": bool(linears),
                "denoiser_linear_precision": all(
                    not x["autocast"]
                    and all(
                        t[0] == ("torch.float32" if reference else "torch.bfloat16")
                        for t in x["inputs"]
                    )
                    for x in linears
                ),
            }
            self.report["model"] = self.family
        self.report["checks"] = checks
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(self.report, indent=2) + "\n")
        for handle in self.handles:
            handle.remove()
        if completed and not all(checks.values()):
            message = f"Precision audit failed: {checks}"
            raise AssertionError(message)


def main() -> int:
    assert os.environ.get("SLURM_JOB_ID")
    assert torch.cuda.is_available()
    audit = PrecisionAudit(Path(sys.argv[1]))
    if "--model" in sys.argv:
        index = sys.argv.index("--model")
        audit.family = sys.argv.pop(index + 1)
        sys.argv.pop(index)
    original_bind = runtime.Runtime.bind
    original_measure = runtime.measured_forward

    def bind(owner: runtime.Runtime, model: torch.nn.Module) -> Any:
        if owner.request.execution.compile or owner.request.execution.cuda_graph:
            message = "Audit must run eagerly, independently of latency measurements"
            raise ValueError(message)
        audit.install(model)
        return original_bind(owner, model)

    def measure(fn: Callable, repeats: int) -> Any:
        completed = False
        try:
            with audit:
                result = original_measure(fn, repeats)
            completed = True
            return result
        finally:
            audit.save(completed=completed)

    runtime.Runtime.bind = bind
    runtime.measured_forward = measure
    try:
        return run(audit.family, sys.argv[2:])
    finally:
        runtime.Runtime.bind = original_bind
        runtime.measured_forward = original_measure


if __name__ == "__main__":
    raise SystemExit(main())
