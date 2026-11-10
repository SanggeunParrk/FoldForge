# Backends

FoldForge's graph holds miniworld-engine modules directly (a pairformer block IS `miniworld_engine.modules.PairformerBlock`), so a model's
backend is the engine's: every op module has a `backends` tuple and `module.set_backend(name)`. A model is loaded with one `backend` for every
op, single ops can be pinned elsewhere with `engine.ops`, and `engine.unsupported` says what happens to a call a backend cannot take.

```yaml
backend: miniworld          # miniworld | cuequivariance | cuda | triton | anthropic | cueq | pytorch
engine:
  ops: {trimul: anthropic}  # optional: single ops on another backend than `backend`
  unsupported: fallback     # fallback (default) | error
```

## Names

| `backend` | meaning |
|---|---|
| `pytorch` | the engine's reference backend of every module: runs anywhere, the baseline every other backend is compared with |
| `cuda` | the engine's hand-written H100 (sm_90) kernels; the module must have them |
| `triton` | the engine's portable Triton kernels; the module must have them |
| `anthropic` | the engine's connection to Anthropic's uplifting-biomolecular-modeling release (a baseline: connected, never gated on accuracy) |
| `cueq` | the engine's connection to NVIDIA cuEquivariance (a baseline) |
| `miniworld` (default, alias) | the engine's own kernels, chosen **per module class**: `cuda` where the class has it and the card is an H100, else `triton` where the class has it and a CUDA device exists, else `pytorch` (not a refusal) |
| `cuequivariance` (alias) | `cueq` |

`miniworld_engine` (the engine package name) is accepted by the loader as `miniworld`. The aliases are FoldForge policy (`modules/engine/backends.py::ALIASES`);
the engine itself knows only `pytorch | cuda | triton | anthropic | cueq`.

A named backend (`cuda`, `triton`, `anthropic`, `cueq`) the module class does not have, `cuda` off an H100, and `cuda` / `triton` without a CUDA
device are **refusals**: the module runs `pytorch` and the reason is recorded. Whether a particular CALL is inside a backend's range (shape, dtype,
mask, gradient) is the module's own `unsupported(*args)`, asked at the call.

## Ops

`engine.ops` takes these names (`backends.OPS`; the engine classes of each are `backends.OP_CLASSES`):

| op | engine classes |
|---|---|
| `trimul` | `TriangleMultiplication`, `TriangleMultiplicationRF3`, `BidirectionalTriangleMultiplication` |
| `triangle_attention` | `TriangleAttention`, `TriangleAttentionAF3` / `RF3` / `Chai1` / `Chai1Confidence` |
| `opm` | `OuterProductMean` |
| `msa_pwa` | `MSAPairWeightedAveraging` |
| `single_attention` | `AttentionPairBias` |
| `transition` | `Transition` |
| `layernorm` | `LayerNorm` |
| `conditioning` | `ConditionedTransition`, `AdaptiveLayerNorm`, `IdentityAdaptiveLayerNorm`, `AdaptiveRMSNorm` |
| `dit_blocks` | `TokenDiT*` blocks |
| `atom_blocks` | `AtomDiT*` blocks |
| `sampler_ops` | `AugmentNoise`, `AlignEuler`, `PoolAtoms` |

What each backend serves is **derived from the engine classes' `backends` tuples**, never written down here:
`foldforge.modules.engine.backends.SERVES` maps every op to the policy names some class of it has (`miniworld` when any class has `cuda` or
`triton`). The load report records the outcome (below). Composite blocks (`PairformerBlock`, `Pairformer`) stay on `pytorch` themselves and their
children are set one by one, so an op without a kernel is refused alone instead of taking the whole block down.

## `engine.unsupported`

- `fallback` (default): a module whose chosen backend refuses a call runs its engine `pytorch` backend for that call and says so in **one** log line
  per (class, backend, reason) (logger `foldforge.backend`). At load, an op that NO class of can serve what was asked logs one line naming every
  such op.
- `error`: `BackendUnsupportedError`, at load for ops and at the call for refusals. Meant for a per-op request (`backend: pytorch`,
  `engine.ops: {trimul: anthropic}`).

Nothing is substituted silently and never by another backend's kernel: the fallback is always the engine's `pytorch` backend of that module.
The guard lives in `modules/engine/policy.py`; what FoldForge asked for and what the engine refuses is catalogued in
[ENGINE-UNSUPPORTED.md](../ENGINE-UNSUPPORTED.md) (U-backend).

## What a run records

The load report (`model.foldforge_load_report`, and `checkpoint` in the run's JSON) holds `backend`, `backends` (the plan: `backend`, `ops` = the
backend that runs each op, `fallbacks` = op: reason, `unsupported`), `engine_backends.modules` (per module: class, op, backend, requested, reason),
`precision` (`master` / `fp32`), `activation_dtype`, `parameter_dtype` (always `float32`). Runtime results carry `engine_dispatch`: per guarded
module the selected backend and the **host-observed** calls (`served_host_calls` per backend, `refused_host_calls` per reason). These count Python
calls including graph capture; compiled regions and CUDA-graph replays do not update them (replay counts are in the execution report), so they are
not a kernel trace.

## Precision

`precision` is the engine's parameter precision, derived from the activation dtype: `bf16` (default) and `af3_default` run `master` (fp32 parameters,
bf16 activations under `activation_autocast`), `fp32` runs `fp32` (plain fp32). There are no native-bf16 parameters; the trunk is always `master`.
`af3_default` additionally lets the diffusion head use TF32 (`foldforge_tf32` through `set_tf32_policy`) while the trunk stays exact. See
[CONFIG.md](CONFIG.md) and [ENGINE-UNSUPPORTED.md](../ENGINE-UNSUPPORTED.md) (U-precision).

## External baselines

`anthropic` and `cueq` are connections to other people's kernels, kept as comparison baselines. They are located by the engine (the Anthropic
release is `MINIWORLD_ANTHROPIC_ROOT`, else the engine's `third_party/anthropic/upstream`, revision- and hash-checked by the engine's loader;
cuEquivariance is the `baselines` extra). FoldForge never gates them on accuracy and calls no vendor library itself.

## Other cards

The checks that decide whether a kernel runs ask the device (`torch.cuda.get_device_capability`); FoldForge holds no card list. On a card that is
not an H100, `miniworld` means the engine's `triton` kernels for classes that have them and `pytorch` for the rest; `cuda` is refused. Only an H100
was available to test this port: nothing was run on other cards.

## Comparing backends

```
python tools/compare_backends.py af3 --spec input.yaml --config runtime.yaml --out compare/af3
```

runs the same model, input and config once per backend, each in its own process, and prints one row per backend (warm median time, mean pLDDT, and
the displacement of the predicted coordinates from the `pytorch` run, raw and after the best rigid fit); a refusal, out-of-memory or crash is a row
too. It needs a GPU allocation like every other run. `tools/benchmark_end_to_end.py --backends ...` takes the same names for whole-process timings.

## Measurements

The measurement tables of the pre-2.5.0 engine (2.3.x, dense twins attached) are kept for the record in
[../archive/BACKENDS-engine-2.3.md](../archive/BACKENDS-engine-2.3.md); they do not describe this engine.
