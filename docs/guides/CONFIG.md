# Configuration

Every option that changes what a fold computes or how it runs is a key of the nested Pydantic `Config`
(`src/foldforge/models/config/runtime.py`), given with `foldforge fold <model> --spec input.yaml --config runtime.yaml`.
A key left out has its default, and the defaults are the behaviour without any override. The config written next to a
run (`model.config.json`) records every key.

```yaml
backend: miniworld          # miniworld | cuequivariance | cuda | triton | anthropic | cueq | pytorch   (BACKENDS.md)
precision: bf16             # bf16 | fp32 | af3_default | model_default
mode: fast                  # fast | exact
trunk: {recycles: null, msa_depth: null}
diffusion: {steps: null}
execution:
  compile: false
  cuda_graph: false
  scope: denoiser           # denoiser | trunk | model
  confidence_sample_chunk: null   # samples per confidence-head call; null = the head's rule, 1 = sequential
engine:
  skip: []                  # [engine_modules]: every engine module on its pytorch backend, see below
  ops: {}                   # {op: backend}: single ops on another backend than `backend`, e.g. {trimul: anthropic}
  unsupported: fallback     # fallback | error: an op the backend has no kernel for
debug:
  dense_spec_override: null # DenseSpec fields to replace, e.g. {confidence_reembed_pair: true}
```

## `trunk.recycles`

The **total number of trunk passes**, including the first, for every model.
`recycles: 10` runs the trunk exactly ten times; `recycles: 1` runs it once.
An omitted or null value defaults to ten. Zero and negative values are rejected.
The Python `load(..., recycles=...)` API and legacy `--recycles` flag use the same rule.
Output JSON records this total in both `recycles` and `trunk_passes`.

Before 2026-10-26, AF3, OpenFold3, Boltz-2, IntelliFold2 and ESMFold2 counted
additional passes: a setting of ten ran eleven. To reproduce those old workloads,
set eleven now. Protenix, OpenDDE, Chai-1 and RoseTTAFold3 already counted totals.
Historical benchmark records retain their original settings and timings.

## Resolved model and observed engine dispatch

The checkpoint report's `resolved_spec` records the final model conventions after family inheritance, `mode: exact` and
`debug.dense_spec_override`. Set-valued fields are sorted lists and infinite bounds are strings, so this is strict JSON. These describe the
loaded equation, not proof of kernel execution.

The load report also holds `backend`, `backends` (the plan: the backend of every op, the fallbacks and their reasons), `engine_backends.modules`
(per module: class, op, requested and chosen backend, reason), `precision` (`master` or `fp32`), `activation_dtype` and `parameter_dtype`
(always `float32`). Runtime results contain `engine_dispatch`: for every engine module that carries a backend guard, its backend and its
**host-observed** calls (`served_host_calls` per backend, `refused_host_calls` per reason; `host_observed: false` when none was seen). The counters
count Python calls including graph capture; compiled regions and CUDA-graph replays do not update them, so they are not a kernel trace and a
module with zero observations is not evidence that it did not run. Replay counts are the separate `cuda_graph_replays` /
`sampler_graph_replays` fields.

## `precision`

`bf16` (default) and `af3_default` run the engine's `master` precision: fp32 parameters, bf16 activations under autocast (`master`); `fp32` runs
plain fp32 (`fp32`); `model_default` is `af3_default`. There are no native-bf16 parameters. `af3_default` loads the released values exactly into
fp32 (a bf16 value is exact in fp32) and lets the diffusion head use TF32 while the trunk stays exact. The resulting engine precision is
`precision` in the load report.

## `backend`, `engine.ops`, `engine.unsupported`

`backend` is the kernel source of every op of the model: an engine backend (`pytorch`, `cuda`, `triton`, `anthropic`, `cueq`) or one of the two
policy aliases (`miniworld` = the engine's own kernels, per class `cuda` on an H100 else `triton` else `pytorch`; `cuequivariance` = `cueq`).
`engine.ops` pins single ops to another one; `engine.unsupported` says what an op does for a call its backend refuses (`fallback`: the engine's
`pytorch` backend of that module after one log line; `error`: raise). The ops, names and how to compare backends: [BACKENDS.md](BACKENDS.md).

## `engine.skip`

A debugging aid: `[engine_modules]` puts every engine module on its `pytorch` backend (what a CPU load does anyway), so the installed backends can
be ruled out at once. An unknown value is a config error. From Python: `load(..., engine_skip=("engine_modules",))`.

## `engine_cache_dir` (removed)

miniworld-engine 2.5.0 has no autotune cache (launch configs come from the engine's pretuned tables). `engine_cache_dir` in `load_model` /
`predict` / `--engine-cache-dir` is accepted and ignored with one `DeprecationWarning`; `tools/build_autotune_cache.py` and its guide are deleted.

## `debug.dense_spec_override`

Replaces fields of the family's `DenseSpec` after it is chosen, to flip one forward convention of a release and
measure it. A field that changes a parameter shape makes the strict weight load fail loudly.

## What stays an environment variable

Locations and the machine, not behaviour: `FOLDFORGE_HOME`, `FOLDFORGE_CHECKPOINT_DIR`, `FOLDFORGE_CCD_DB`,
`FOLDFORGE_RUNS`, `FOLDFORGE_RELEASE_ROOT`, `FOLDFORGE_BENCHMARKS`; `CUDA_VISIBLE_DEVICES`, `SLURM_*`, `XDG_CACHE_HOME`,
`TORCH_EXTENSIONS_DIR`, `CUTLASS_PATH`; the location of the Anthropic release for the `anthropic` backend, `MINIWORLD_ANTHROPIC_ROOT` (BACKENDS.md); `CUBLAS_WORKSPACE_CONFIG` (set by `utils.seed` for deterministic cuBLAS); the
test inputs `FOLDFORGE_TEST_CCD_DB` and `MINIWORLD_TEST_SOURCE`.

## Removed variables

These used to change behaviour and are now an error if they are set (`RuntimeError` from `Config(...)` and from the
loader), never silently ignored:

| removed | now |
|---|---|
| `FF_ENGINE_SKIP` | `engine.skip` |
| `FOLDFORGE_DENSE_SPEC_OVERRIDE` | `debug.dense_spec_override` |
| `FOLDFORGE_CONFIDENCE_SAMPLE_CHUNK` | `execution.confidence_sample_chunk` |
| `FF_TRUNK_BF16` | deleted: an experiment (bf16 trunk on an fp32 release under `af3_default`), concluded; the released dtypes stay |
