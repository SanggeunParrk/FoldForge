# Configuration

Every option that changes what a fold computes or how it runs is a key of the nested Pydantic `Config`
(`src/foldforge/models/config/runtime.py`), given with `foldforge fold <model> --spec input.yaml --config runtime.yaml`.
A key left out has its default, and the defaults are the behaviour without any override. The config written next to a
run (`model.config.json`) records every key.

```yaml
backend: miniworld          # miniworld | cuequivariance | anthropic | pytorch   (BACKENDS.md)
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
  skip: []                  # parts left dense, see below
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

The checkpoint report's `resolved_spec` records the final model conventions after
family inheritance, `mode: exact` and `debug.dense_spec_override`. Set-valued fields
are sorted lists and infinite bounds are strings, so this is strict JSON. These
describe the loaded equation, not proof of kernel execution.

Runtime results contain `engine_dispatch` for token/atom transformers, Pairformer
compositions, and operation support checks. The installer gives per-block,
whole-stack and Pairformer implementations the same `EngineAdapter` / `Dispatch`
contract. Each module reports attachment, implementation names, observed served
calls and named refusal counts (for example `not_installed`,
`inference_mode_has_no_tensor_versions`, `extra_pair_bias_unsupported`,
or `rotary_or_pair_mask_mismatch`). Layout-specific
view/padding/cache-key conventions stay inside their adapters.

`execution_scope: composition` and selection `engine_composition` mean that the
Pairformer composition ran, including its existing per-operation backend routing.
They do not mean that every constituent operation used an engine kernel.
`execution_scope: support_checks_only` reports candidate checks through the common
`policy.accept_support` boundary. This includes `served_by`, Pairformer single
attention, projected triangle attention, transition and conditioning operand checks:
`supported_host_checks` and `refused_host_checks` never increment
`served_host_calls`. A successful support check can still be followed by a
caller-specific fallback. `Support` carries the common support decision and
`call_refusal` retains the existing backend fallback/error policy and log deduplication.
This refactor does not extend that policy to previously silent fallback paths.

The shared policy owns backend/CUDA and BF16/FP32/TF32 eligibility; shape and
equation restrictions stay with their operation adapters. Projected triangle
attention selects miniworld, cuequivariance and Anthropic in the existing order
inside the engine integration layer; the dense module only supplies projected
tensors and runs its chunked reference when the adapter returns `None`.
`attached` describes an out-of-band engine twin or a registered `EngineAdapter`,
not every internal implementation a standalone conditioning wrapper could select.

These are **host observations**, including eager calls and graph capture. Compiled
regions and CUDA Graph replays do not update Python counters. Use the separate
`cuda_graph_replays` / `sampler_graph_replays` fields for replay counts. A variant's
`cuda_candidate` / `portable_candidate` reports the accepting support predicate;
`engine_internal` leaves the kernel choice to the per-block engine. None is a
per-kernel dispatch trace, and an attached module with zero observations is not
evidence that it executed. `host_observed: false` explicitly marks that case.

## Installation decisions

The checkpoint report contains `engine_installation`, separately from runtime
`engine_dispatch`. One ordered `Candidate` chain uses `BuildResult` for each
supported dense module class: per-block candidates precede variant stacks.
The public raw factory APIs still return an engine object or `None`.

Each module records requested/resolved backend, attempted candidates, named
refusals, the constructed candidate and attachment status. Reasons distinguish
missing engine API features, incompatible equations, conditioning width, FP32
variant exclusion, backend routes without a twin, explicit skip, and Anthropic
probe refusal. The engine is pinned (`miniworld-engine==2.4.3`), so the options
the released families need are always present and no capability probe is reported. Unexpected construction errors propagate.

`attached` and `attached_units` describe attachments made by this installation;
`preexisting_twin` / `present_after_install` also expose an older twin retained
when reinstalling with a skip or an ineligible candidate. Existing twins are not
silently removed. `built_candidate` describes construction and may be skipped.

The old `engine_*` counters are preserved as `legacy_counts` and at their existing
locations. In particular, old DiT counters include constructed-but-skipped blocks;
use the new attachment fields for installed counts. No installation field proves
that a runtime kernel executed. Pairformer/conditioning phase summaries report
processing or explicit skip. Their `decisions` maps also record registered-module
conversions and named exclusions such as an engine without parallel composition,
identity-scale normalization or incompatible affine/epsilon conventions.

## `backend`, `engine.ops`, `engine.unsupported`

`backend` is the kernel source of every op of the model; `engine.ops` pins single ops to another one;
`engine.unsupported` says what an op does that its backend has no kernel for (the PyTorch reference after one log
line, or an error). The ops, what each backend serves and how to compare backends: [BACKENDS.md](BACKENDS.md).
With no override the model runs on `miniworld` exactly as before.

## `engine.skip`

A debugging aid: the engine install leaves the named parts as the dense PyTorch modules, so one kernel family can be
priced or ruled out at a time. Allowed values: `pairformers`, `conditioning`, `engine_trimul`,
`engine_triangle_attention`, `engine_msa_pwa`, `engine_opm`, `engine_trunk_transition`, `engine_dit_blocks`, `engine_single_attention`,
`engine_atom_blocks`, `engine_dit_variant_blocks`, `engine_atom_variant_blocks` (the keys of the load report's engine
counts). An unknown value is a config error. From Python: `load(..., engine_skip=("engine_opm",))`.

## `debug.dense_spec_override`

Replaces fields of the family's `DenseSpec` after it is chosen, to flip one forward convention of a release and
measure it. A field that changes a parameter shape makes the strict weight load fail loudly.

## What stays an environment variable

Locations and the machine, not behaviour: `FOLDFORGE_HOME`, `FOLDFORGE_CHECKPOINT_DIR`, `FOLDFORGE_CCD_DB`,
`FOLDFORGE_RUNS`, `FOLDFORGE_RELEASE_ROOT`, `FOLDFORGE_BENCHMARKS`; `CUDA_VISIBLE_DEVICES`, `SLURM_*`, `XDG_CACHE_HOME`,
`TORCH_EXTENSIONS_DIR`, `CUTLASS_PATH`; the Anthropic payloads of the `anthropic` backend `MINIWORLD_ANTHROPIC_ROOT`,
`OPT_CORE_DIR` and `TRIMUL_NATIVE_BUILD_DIR` (BACKENDS.md); `CUBLAS_WORKSPACE_CONFIG` (set by `utils.seed` for deterministic cuBLAS); the
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
