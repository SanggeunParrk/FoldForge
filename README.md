# FoldForge

**Many structure predictors, one environment, one set of kernels.**

AF3, Protenix v1/v2, ESMFold2 and OpenDDE share MiniWorld-format inputs,
one CCD LMDB, team-gm block compositions and miniworld-engine kernels.
Released model equations, checkpoint layouts and confidence heads remain explicit
in each adapter. Boltz-2 and Chai-1 are planned and are not implemented.

See [current qualification](docs/QUALIFICATION-20260914.md) for complex inputs,
actual compilation/CUDA graph boundaries and remaining deployment requirements.

## Where this sits

FoldForge is a **terminal** repo in a three-layer stack. The boundary rules —
which decide whether a piece of code belongs here or a layer down — are
canonical in [libs/team-gm/docs/ARCHITECTURE.md](libs/team-gm/docs/ARCHITECTURE.md).
The short version:

| layer | repo | holds |
|---|---|---|
| engine | miniworld-engine | fused kernels + the ops that wrap them |
| framework | team-gm | representative AF3 **blocks**, diffusion, training base |
| terminal | **FoldForge** | the **full models**, data pipeline, eval, CLI |

Nothing here defines an op or a general-purpose block. If you are writing one,
it goes a layer down.

## Install

```bash
git clone --recurse-submodules git@github.com:SanggeunParrk/FoldForge.git
cd FoldForge
uv sync --extra cu12    # CUDA 12.8   (or --extra cu13 for CUDA 13)
```

For an existing checkout: `git submodule update --init --recursive`.

The pinned Biohub ESMFold2 processor requires **Python 3.12**. For
A5000/A6000/A100, the complete environment includes FA2, Quack 0.5.0 and
CUTLASS DSL 4.5.2. On this cluster:

```bash
mkdir -p validation/logs
sbatch scripts/setup_env.sbatch
# After installation succeeds, before running Python directly:
source scripts/activate_env.sh
```

Use the setup script for updates too: it reuses the verified FA2 wheel when
available, otherwise builds the pinned release. Model runners activate `.venv`
with its required C++ runtime. See [the environment record](docs/ENVIRONMENT-20260913.md).


> [!NOTE]
> Resolve with **uv**, not plain pip. The CUDA index routing that picks a torch
> build matching `cuequivariance-ops-torch-cuNN` is uv-specific; pip ignores it
> and installs a mismatched torch. `cu12` and `cu13` are mutually exclusive.
>
> team-gm owns torch and the CUDA ops stack — never declare torch here.

### The engine pin

`miniworld-engine` is a git dependency, pinned by `rev` in **both**
`pyproject.toml` and `libs/team-gm/pyproject.toml`. team-gm is a uv workspace member;
the root source declaration governs the workspace. Keep the member's pin aligned
so its standalone environment uses the same engine revision.

See [the integration handoff](docs/ENGINE-INTEGRATION-20260913.md) for the pinned
revision, validation results, and FlashAttention setup required for GPU SWA.

## Layout

```
libs/team-gm/            submodule, tracks exp/miniworld
src/foldforge/
  checkpoints.py         where the weights are — asked, never hardcoded
  prediction.py          the one output type every predictor returns
  models/                one package per predictor + the name -> loader registry
    af3/ boltz2/ chai1/ esmfold2/ opendde/ protenix/
  modules/               blocks two+ predictors need that team-gm does not carry
  data/                  shared input features (sequence, MSA, templates)
  eval/                  RMSD / lDDT / TM against a deposit or another predictor
  cli/                   foldforge models, foldforge fold <model>
scripts/                 Slurm wrappers and measurement drivers
configs/  tests/  docs/
```

`model_checkpoints/`, `benchmark/` and `validation/` are gitignored: weights are
~44 GB, and measurements describe a machine and a checkpoint rather than the
source.

## Conventions

- **bf16 is the default.** The released checkpoints are fp32 on disk. Model
  loading selects the runtime precision; engine attention preserves its caller's
  dtype. Use fp32 when a task explicitly calls for it.
- **Never quote a speed number without its conditions** — device, dtype, MSA
  depth, warm or cold, and which backend *both* sides ran. Fold times on the
  same target and config have differed by 6% across processes purely from
  Triton autotune selection, so a before/after taken from two jobs is noise with
  a sign. A/B in one process, with a same-arm control that reports the noise
  floor.

### Where a block goes

| the unit | lives in |
|---|---|
| wraps a kernel, one PyTorch reference | miniworld-engine |
| representative AF3 block (Pairformer, MSA module, template, diffusion transformer) | team-gm |
| specific to one predictor | `models/<name>/` |
| needed by two or more predictors, no reason for team-gm to carry it | `modules/` |

A single-consumer block in `modules/` is the common mistake: it reads as shared,
and the next person changing it has to prove nothing else depends on it.

## Status

Every predictor has a package from the start — the port is where the boundary
decisions get made, and they belong next to the model rather than in an issue.
`known_models()` is the plan; `registered_models()` is what actually loads.

| predictor | state |
|---|---|
| ESMFold2 | ported; live ESMC → folding → CIF runs through the public CLI; see [integration verification](docs/MODEL-INTEGRATION.md) |
| AF3 | ported; strict released weights, full 1UBQ inference and PyTorch comparison verified |
| Protenix v1/v2 | ported; both variants passed full 1UBQ inference and PyTorch comparison |
| OpenDDE | ported; strict released weights, full 1UBQ inference and PyTorch comparison verified |
| Boltz-2, Chai-1 | not ported |

### Run ESMFold2

On an allocated GPU node:

```bash
source scripts/activate_env.sh
foldforge models
foldforge fold esmfold2 --target 1ubq --lm-source compute \
  --out validation/results/esmfold2-live
```

Targets use prepared `validation/data/<target>/target.json` and MSA files.
`--lm-source compute` runs the ESMC checkpoint; `cache` explicitly reuses an existing
embedding file. The command writes CIF and JSON with actual precision, sampling,
compile and CUDA-graph settings. See [the integration record](docs/MODEL-INTEGRATION.md)
for the additional models' source/checkpoint status and validation scope.
All four predictors use the MiniWorld BioMol LMDB at
`data/ccd/preprocessed_CCD.lmdb`. Use the same MiniWorld YAML with each model:

```bash
foldforge fold af3 --spec configs/inference/1ubq.yaml --out predictions/af3
foldforge fold protenix --spec configs/inference/1ubq.yaml --out predictions/protenix
foldforge fold opendde --spec configs/inference/1ubq.yaml --out predictions/opendde
foldforge fold esmfold2 --spec configs/inference/1ubq.yaml --out predictions/esmfold2
```

See [MiniWorld formats](docs/MINIWORLD-FORMAT.md) for database migration,
nested model settings, data conventions and checkpoint-specific capabilities.
