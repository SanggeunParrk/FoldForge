# FoldForge

**Many structure predictors, one environment, one set of kernels.**

Eleven AF3-family predictors share MiniWorld-format inputs, one CCD LMDB,
team-gm block compositions and miniworld-engine kernels — and one graph. A
family is a `DenseSpec` row stating where its release disagrees with AF3;
where a release's equations, checkpoint layout or confidence head differ, the
difference is a field in that row rather than a second implementation.

Start with the [documentation index](docs/README.md) and
[benchmark results](docs/benchmark_results.md): 5I28 replicated to 512 tokens and
beyond on A100 and A6000, five execution modes at 512 tokens and a length ladder
to the out-of-memory point, under one AF3-style input policy (16384 prepared MSA rows, 1024 sampled
per recycle, four templates per chain).

See the [qualification record](docs/archive/QUALIFICATION-20260914.md) for complex inputs,
actual compilation/CUDA graph boundaries and remaining deployment requirements.
[The follow-up](docs/archive/CLOSEOUT-20260914.md) records operational DB reads, A5000
execution and the large-input AF3 compiler numerical regression that did not pass.

## Code and loading

Start with [the directory guide](docs/guides/CODE-STRUCTURE.md).
`foldforge.load(model, checkpoint, backend=..., dtype=..., device=...)` is the
single loading API. Model directories and `ported/` packages have been removed;
architecture assembly, configuration, weight mapping and feature processing each
have their own owner.

```python
import torch
from foldforge import load

model = load("protenix", "/path/to/protenix-v2.pt", variant="protenix-v2",
             backend="miniworld", dtype=torch.bfloat16, device="cuda")
```

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
The checked-in submodule revision is the validated dependency: team-gm
`52292ac` on branch `foldforge/dense-families`. Keep that revision when
reproducing the recorded results; newer `exp/miniworld` commits use a
different engine/patch set.

The pinned Biohub ESMFold2 processor requires **Python 3.12**. For
A5000/A6000/A100, the complete environment includes FA2, Quack 0.5.0 and
CUTLASS DSL 4.5.2. On this cluster:

```bash
mkdir -p validation/reports/logs
sbatch scripts/setup_env.sbatch
# After installation succeeds, before running Python directly:
source scripts/activate_env.sh
```

Use the setup script for updates too: it reuses the verified FA2 wheel when
available, otherwise builds the pinned release. Model runners activate `.venv`
with its required C++ runtime. See [the environment record](docs/archive/ENVIRONMENT-20260913.md).


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

See [the integration handoff](docs/archive/ENGINE-INTEGRATION-20260913.md) for the pinned
revision, validation results, and FlashAttention setup required for GPU SWA.

## Layout

```
libs/team-gm/            submodule, pinned; branch foldforge/dense-families
src/foldforge/
  cli.py                 foldforge models / ccd / fold <model>
  prediction.py          the one output type every predictor returns
  models/                architectures, config, checkpoints, io adapters and the
                         single loading / execution / sampling / precision lifecycle
  modules/               the dense AF3 graph's blocks, shared ops, the ESM-C backbone
  data/                  CCD, features, MSA, templates and MiniWorld input preparation
  eval/                  RMSD / lDDT / TM, confidence metrics, permutation matching
  training/  utils/      losses; geometry, seeding, logging and tensor helpers
scripts/                 Slurm wrappers and measurement drivers
configs/  tests/  docs/  typings/
```

`model_checkpoints/`, `benchmark/`, `runs/` and `validation/` are gitignored: weights are
~44 GB, and measurements describe a machine and a checkpoint rather than the
source.

## Conventions

- **bf16 is the default.** The released checkpoints are fp32 on disk. Model
  loading selects the runtime precision; engine attention preserves its caller's
  dtype. Use fp32 when a task explicitly calls for it.
- **One MSA policy for every predictor.** Inputs keep up to 16384 alignment
  rows and each trunk pass embeds a fresh random subset of 1024 valid rows,
  the official AF3 pipeline rule, applied to all four checkpoints at load time.
  Released defaults differed; see [MiniWorld formats](docs/guides/MINIWORLD-FORMAT.md).
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
| specific to one predictor | `models/architectures/<name>.py` |
| needed by two or more predictors, no reason for team-gm to carry it | `modules/` |

A single-consumer block in `modules/` is the common mistake: it reads as shared,
and the next person changing it has to prove nothing else depends on it.

## Status

Every family is a ROW of one graph, not a copy of it: `modules/dense/spec.py`
holds a `DenseSpec` per family and `models/architectures/af3.py` is the only
architecture in the tree. `known_models()` is the plan and `registered_models()`
is what actually loads; they are now the same set.

| family | what its row says |
|---|---|
| AF3 | the reference the other rows are stated against |
| Protenix v1 / v2 | `--variant` picks the release |
| OpenDDE | folds on structural tokens: the diffusion axis is re-tokenised, the trunk stays on residues |
| ESMFold2 | pair-only trunk, SSM recycle, ESM-C 6B into the pair track |
| ESMFold2-Fast | the same row at 24 trunk blocks and no MSA stack |
| Boltz-2, Chai-1, IntelliFold-v2, OpenFold3 (v0.5.0 / preview-2), RoseTTAFold3 | the AF3 graph with that release's conventions |

`foldforge models` lists them with their sources. Per-family checkpoint
evidence and validation scope are in
[the integration record](docs/guides/MODEL-INTEGRATION.md).

### Run a fold

On an allocated GPU node, every family takes the same MiniWorld YAML and the
same flags — there is one CLI, not one per predictor:

```bash
source scripts/activate_env.sh
foldforge models
foldforge fold af3      --spec configs/inference/1ubq.yaml --out runs/af3
foldforge fold protenix --spec configs/inference/1ubq.yaml --out runs/protenix
foldforge fold opendde  --spec configs/inference/1ubq.yaml --out runs/opendde
foldforge fold esmfold2 --spec configs/inference/1ubq.yaml --out runs/esmfold2
```

`--checkpoint` is optional when the release's default file is in
`model_checkpoints/<model>/`. The command writes CIF and JSON recording the
actual precision, sampling, compile and CUDA-graph settings. Every family
reads the MiniWorld BioMol LMDB at `data/ccd/preprocessed_CCD.lmdb`.

### Exact and fast modes

Every family folds in one of two modes, chosen with `mode:` in the config:

- **`fast`** (default) runs AF3's computation wherever a release's own
  computation folds the same -- every such convention was reset one at a time
  against its release and moved no fold beyond the release's own spread.
- **`exact`** keeps every one of those release conventions as well
  (`EXACT_CONVENTIONS` in `modules/dense/spec.py`): key-window end policies,
  the OR-form atom mask, outer-product-mean normalisations, template details,
  the empty MSA context, the atoms a release drops. Use it to compare a fold
  with its release computation for computation.

Conventions a release NEEDS -- its bond matrix, its bond orders, its MC
dropout, its ligand atom names and so on -- are in both modes.

### Accuracy against the released implementations

`outputs/` holds, for three reference targets (protein; protein + ligand +
ion; DNA + protein + ions), every family's structures from its **own released
code**, three seeds of five samples. `foldforge validate` judges a FoldForge
run against them -- structure, pLDDT, ligand bond geometry and ligand/ion
placement, each relative to the release's own seed-to-seed spread -- and the
current verdicts are in [docs/validation/](docs/validation/). See
[outputs/README.md](outputs/README.md) to regenerate or extend them.

Today 27 of 28 family x target rows pass in both modes. The one exception,
OpenDDE's pLDDT on 5I28 (68.4 against 71.1, structure passing), is the
reference conformer source, not the model: with the release's conformer it
reads 70.9. The [validation notes](docs/validation/README.md) list the rule for
each check and the defects the comparison found.

### Checkpoints

Converted blobs are pinned by SHA-256 in
`src/foldforge/models/checkpoints/checkpoints.lock`. Loading refuses a blob
whose size does not match (a blob from another converter version);
`foldforge checkpoints verify` checks the hashes. How to produce each blob from
its release file, and which conversions reproduce the lock bit for bit, is in
[the checkpoint guide](docs/guides/CHECKPOINTS.md).

See [MiniWorld formats](docs/guides/MINIWORLD-FORMAT.md) for database migration,
nested model settings, data conventions and checkpoint-specific capabilities.

## Tests, validation and prediction outputs

- `tests/`: maintained regression tests by responsibility; run `pytest tests`.
- `validation/inputs/`: prepared evaluation targets and model input specs.
- `validation/reports/`: validation logs and artifact relocation manifests.
- `validation/archive/`: historical scripts, temporary source copies and checks.
- `runs/`: all structure-prediction outputs, including benchmark predictions.

Both CLI forms enforce the same destination. `--out experiment` and
`--out runs/experiment` write to `<repo>/runs/experiment`, independent of the
working directory. Without `--out`, a unique `runs/<model>/<timestamp-id>/` is
used. Absolute paths must be within this runs tree; outside destinations fail
before input preparation or prediction writing. Prepared inputs, coordinates,
confidence files and the run report are kept together. Historical predictions
are preserved under `runs/archive/validation/`; relocation manifests map old
paths to their new locations.

Development checks: run `bash scripts/check_quality.sh` for Ruff lint/format and
Pyright. See [code quality](docs/guides/CODE-QUALITY.md) for scope, retained numerical-code
exceptions, and compute-node regression checks.
