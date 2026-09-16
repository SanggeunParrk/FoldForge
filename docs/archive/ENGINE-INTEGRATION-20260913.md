# Engine integration — 2026-09-13

The root and `libs/team-gm` now pin engine main
`d2266a035de11384c46f8cc980e6460f60925413`.
The engine includes the A5000 cache qualification and consumer API repairs.

`libs/team-gm` is a uv workspace member. Its cu12/cu13 Torch index routing is
used directly; duplicated root Torch sources were invalid for transitive extras
and then conflicted with the member's sources. The root explicitly declares both
its own extra conflict and the member's conflict, allowing one universal lock.
See [uv workspace conflict rules](https://docs.astral.sh/uv/concepts/resolution/).
`uv lock --check --offline` passes. `uv.lock` is now provided.

Engine fixes restore the SWA geometry helper exports and support
`OuterProductMean(normalize_before_proj=False)`, including ESMFold2's normalized
projection bias. FP32 attention is no longer silently quantized to BF16.
Existing module ownership remains: triangle/ordinary transition modules own their
residuals; conditioned transition and augmented attention return deltas.

Validation: all 17 FoldForge CPU tests pass, including actual MSA and atom-block
construction/forward, FP32/native-BF16 MSA execution, and residual checks. These
used the engine Python 3.12 / Torch 2.10+cu128 environment with the existing
ESMFold2 dependencies. No full checkpoint inference or FoldForge-specific GPU
benchmark was rerun. The sibling team-gm consumer additionally passed 112 GPU
module tests on A5000 and 64 checkpoint/model/solver CPU tests.

The follow-up [environment record](ENVIRONMENT-20260913.md) defines the local
`.venv`, pinned Torch/FA2/Quack/CuTe versions, the Biohub Transformers source,
and the compute-node installation procedure. It supersedes the initial missing
local environment noted during this integration audit.

Changes are left for review in this working tree, including the engine-pin
change inside `libs/team-gm`; the original team-gm submodule revision is retained.
When publishing this consumer later, commit its member change before the parent.

Full-checkpoint follow-up: [MODEL-INTEGRATION.md](../guides/MODEL-INTEGRATION.md) records
the public CLI, live ESMC, normalization fix and corrected CA scoring. Its
validation supersedes the initial import/module-only scope above.
