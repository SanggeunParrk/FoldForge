# FoldForge code ownership

The runtime entry point is `foldforge.load(name, checkpoint, **options)`. All four
predictors use one lifecycle: configuration, architecture construction, strict
checkpoint import, native parameter precision, backend installation and load
report. `get_model(name)` remains a partial of this same function. There are no
per-model `load` functions or `from_checkpoint` methods.

```text
src/foldforge/
  __init__.py           public load, checkpoint discovery and Prediction API
  cli.py                command parsing and shared execution entry point
  prediction.py         common prediction result type
  models/
    architectures/     network assembly and checkpoint-specific topology
    config/            runtime schemas and released architecture defaults
    checkpoints/       discovery, Haiku/safetensors conversion and source licenses
    io/                input/output tensor adapters and common request runtime
    loading.py         one strict loading lifecycle
    execution.py       compile and CUDA graph ownership
    bucketing.py       inference padding policies
    sampling.py        team-gm sampler argument adaptation
    precision.py       parameter/normalization precision policy
  modules/             layout adapters, operators and ESMC language backbone
  data/                CCD, features, tokenization, MSA, templates and preparation
  eval/                structural scores and model confidence metrics
  training/            training losses
  utils/               geometry, permutation, logging and support utilities
```

## Where to make a change

| Change | Owner |
|---|---|
| Choose a checkpoint or backend, change loading precision | `models/loading.py` |
| Add a model variant or change released dimensions | `models/config/` |
| Change serialized weight names or projection packing | `models/checkpoints/` |
| Inspect a complete model's network topology | `models/architectures/` |
| Adapt input tensor axes to a checkpoint | `models/io/`, `data/` |
| Change common diffusion schedule, solver or augmentation | `team_gm.diffusion` |
| Change shared attention, triangle, MSA, transition equations | `team_gm.modules` |
| Change an accelerated kernel | `miniworld_engine` |

## What was consolidated

- Removed AF3/Protenix/OpenDDE `ported` trees and all four model packages.
- Removed per-model lazy loader wrappers and CLI wrappers. The CLI dispatches to
  the common request/runtime directly; installed CLI commands keep their flags.
- Strict state-dict reading, backend choice, dtype conversion and reporting now
  have one execution lifecycle. Haiku and safetensors remain distinct formats.
- Removed import-only chemistry and guidance wrappers: consumers import their
  actual shared owner. Both flat-atom parsers use `foldforge.data.ccd.components`.
- Shared identical geometry featurization, substructure permutation, scatter,
  download support, residue indices and logging implementations.
- Removed duplicated sampling-generator presets; arguments go to the shared
  sampler adapter and team-gm guidance implementation.

## Intentional differences

Architecture files remain because network topology and checkpoint keys differ.
Dense and sequence modules preserve different tensor layouts and conditioning
formulas; combining them must not change a released model's equation. Some flat
feature processors still have explicit Protenix/OpenDDE variants (for example,
OpenDDE structural tokens versus Protenix tokenization). This reorganization does
not claim that distinct input schemas have become identical algorithms.

Native BF16 still means BF16 learned projection parameters with norm values kept
in FP32; existing FP32 geometry/residual calculations are retained. Sampling order
is unchanged: AF3 still chunks at one sample, Protenix/OpenDDE default to five,
and ESMFold2 expands samples into the batch. Changing these policies is separate
from relocating code.

`tests/models/test_loading.py` verifies strict loading and native precision through the
common API. Existing numerical tests follow the new owners. The historical
upstream-relative names used by frozen equation oracles are mapped only inside
tests; production has no legacy import aliases. Validation: 180 CPU tests passed (164 GPU/resource-dependent tests skipped in
the CPU run), 37 targeted A6000 GPU tests passed, and all five architectures
retained their complete state keys/shapes/dtypes and module topology. The wheel
built offline and includes the source licenses without retired model packages.
See `layout-validation-20260915.json` for job IDs and exact scope.

## Top-level consolidation (2026-09-15)

The top-level package now has six directories instead of fourteen. Configuration
defaults and runtime schemas share `models/config`; checkpoint discovery and
format conversion share `models/checkpoints`. Confidence metrics live with
structural evaluation in `eval`. The ESMC implementation is `modules/esmc.py`,
and dense geometry helpers are `utils/dense_geometry.py`. The single-file CLI
is now `cli.py`. Original licenses are packaged under
`models/checkpoints/provenance`. There are no forwarding packages at the old paths.

The public `foldforge.load`, `foldforge.resolve`, `foldforge.available`,
`foldforge.Prediction` and `foldforge` CLI remain the entry points. Internal
imports in source, scripts and tests follow the new owners.

## Utility ownership

`utils` has eight files (including its initializer) and no model-prefixed files.

| Responsibility | Location |
|---|---|
| Tensor conversion, distance policy, autocast and device cleanup | `utils/tensor.py` |
| Process-group communication and aggregation | `utils/distributed.py` |
| Seeding and deterministic execution switches | `utils/seed.py` |
| NumPy coordinate transforms / tensor rigid frames | `utils/geometry.py`, `utils/dense_geometry.py` |
| Logging / scatter reductions | `utils/logging.py`, `utils/scatter.py` |
| JSON, pickle, hashed LMDB and structure serialization | `data/io.py` |
| Spatial and sequence cropping | `data/cropping.py` |
| Atom/chain symmetry matching and alignment | `eval/permutation/` |
| Metric aggregation | `eval/aggregation.py` |
| Optimizers and learning-rate schedules | `training/optim.py`, `training/schedules.py` |
| Runtime device/backend diagnostics | `models/environment.py` |
| Checkpoint and managed asset download | `models/checkpoints/download.py` |

The two geometry conventions are explicit arguments to one implementation:
`eps=1e-4` retains regularized angles, and `translation_before_rotation=True`
retains translation in the local frame. Training distance calculations explicitly
request `donot_use_mm_for_euclid_dist`; the common default keeps PyTorch's mode
selection. Container export and transfer use the shared non-mutating implementation.
Distributed gathering uses the actual initialized process-group size, and seeding
sets deterministic switches in both directions. Source copyright and licenses
remain recorded; retired import-only compatibility modules are not retained.

## Tests and run artifacts

`tests/{data,models,kernels,runtime,utils}` groups checks by responsibility.
`tests/references` contains frozen equation oracles; `tests/support.py` owns
repository paths and oracle lookup. Validation inputs live in `validation/inputs`;
historical work is grouped in `validation/archive`, and logs/manifests are in
`validation/reports`. Every prediction writer and both CLI routes use
`models/io/paths.py` to keep predictions inside the repository `runs/` directory.
