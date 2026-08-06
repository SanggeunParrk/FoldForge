# FoldForge

**Many structure predictors, one environment, one set of kernels.**

AF3, Boltz-2, Chai-1, Protenix v1/v2, ESMFold2 and OpenDDE, rebuilt on a shared
block library instead of six vendored upstream repos. A change to a triangle
multiplication is then a change to all of them at once, and a speed comparison
between two of them is a comparison of the models rather than of whose kernels
happen to be newer.

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

> [!NOTE]
> Resolve with **uv**, not plain pip. The CUDA index routing that picks a torch
> build matching `cuequivariance-ops-torch-cuNN` is uv-specific; pip ignores it
> and installs a mismatched torch. `cu12` and `cu13` are mutually exclusive.
>
> team-gm owns torch and the CUDA ops stack — never declare torch here.

### The engine pin

`miniworld-engine` is a git dependency, pinned by `rev` in **both**
`pyproject.toml` and `libs/team-gm/pyproject.toml`. uv does not inherit a path
dependency's `[tool.uv.sources]`, so the two are separate declarations that must
be bumped together. If they drift, team-gm's blocks and the engine ops they call
come from different builds.

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
  cli/                   foldforge fold, foldforge bench
scripts/                 Slurm wrappers and measurement drivers
configs/  tests/  docs/
```

`model_checkpoints/`, `benchmark/` and `validation/` are gitignored: weights are
~44 GB, and measurements describe a machine and a checkpoint rather than the
source.

## Conventions

- **bf16 is the default.** The released checkpoints are fp32 on disk, but the
  fused kernels are ~2x slower on fp32 input and the reference attention casts
  to bf16 regardless. fp32 only when a task explicitly calls for it.
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
| ESMFold2 | written and working on team-gm `exp/miniworld-integrated`; the move here is a rewire, see [docs/PORTING-esmfold2.md](docs/PORTING-esmfold2.md) |
| AF3, Boltz-2, Chai-1, Protenix v1/v2, OpenDDE | not started |
