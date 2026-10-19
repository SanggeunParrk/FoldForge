# FoldForge

**Many structure predictors, one environment, one set of kernels.**

Eleven AF3-family predictors share MiniWorld-format inputs, one CCD LMDB,
miniworld-engine blocks and kernels — and one graph. A
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

model = load("protenix2", backend="miniworld", dtype=torch.bfloat16, device="cuda")
```

## Where this sits

FoldForge is a **terminal** repo on top of one dependency, miniworld-engine
(team-gm was absorbed into the engine and into MiniWorld; FoldForge no longer
depends on it). The boundary rule decides whether a piece of code belongs here or
a layer down:

| layer | repo | holds |
|---|---|---|
| engine | miniworld-engine | fused kernels, the ops that wrap them, the representative AF3 **blocks**, diffusion and the checkpoint-attention helpers (`miniworld_engine.architecture`) |
| terminal | **FoldForge** | the **full models**, the checkpoint installers, data pipeline, eval, CLI |

Nothing here defines an op or a general-purpose block. If you are writing one,
it goes in the engine. What used to come from team-gm and has no engine home — the routing of dense
modules to engine ops (`modules/engine`), the reference equations of the released
checkpoints (`modules/reference`), compile/graph execution and bucketing
(`models/`), TFG guidance, the ESM-C model — now lives here; see [the directory guide](docs/guides/CODE-STRUCTURE.md).

## Install

One **pixi** environment, the same layout as MiniWorld and miniworld-engine
(torch 2.13.0+cu129, cuequivariance 0.12, FlashAttention-4; `-e cu13` for CUDA 13).
No GPU is needed to install, and nothing is run on a login node: use a CPU
allocation of the cluster.

```bash
git clone git@github.com:SanggeunParrk/FoldForge.git
cd FoldForge
srun -p h100 -A cssb --qos=normal_h100 --gres=none -c 8 --mem=32G pixi install
pixi run --frozen python -m pytest tests -q        # on a GPU allocation
```

The pinned Biohub ESMFold2 processor requires **Python 3.12**. On
A5000/A6000/A100 add FlashAttention-2 (`FOLDFORGE_FLASH2=1 sbatch tools/setup_env.sbatch`).
Before running Python directly: `source tools/activate_env.sh`.

### The engine pin

`miniworld-engine` is a git dependency pinned by `rev` in `pyproject.toml`
(`[tool.pixi.pypi-dependencies]`); `pixi.lock` records the resolved commit. Bump
both together, then re-run the tests: the engine changes contracts between
releases (for example, since the 2026-10 release every `x = x + f(x)` module
returns the residual itself — use `.delta` for the bare update).

## Layout

```
src/foldforge/        the package
  cli.py              foldforge models / ccd / checkpoints / validate / fold <model>
  paths.py            where weights, databases, release envs and runs live
  models/             registry, the one AF3 architecture, checkpoints, loading,
                      execution, sampling, precision, io
  modules/            the dense graph's blocks (dense/), engine routing (engine/),
                      reference equations (reference/), ESM-C (esmc/) and its glue
  guidance/           TFG guidance (not wired into the AF3 sampler yet)
  data/               CCD database, chemical constants, dense features and
                      family conventions, MiniWorld input specs
  eval/               the release-reference judge, confidence decoding, structures
  utils/              geometry, seeding, logging and tensor helpers
tests/                unit and GPU tests; release_references/ holds every release's
                      own structures and the inputs they were made from
docs/                 guides, validation verdicts, findings, bug reports, examples/
tools/                release runners, validation, conversion, benchmarks, Slurm
                      wrappers; typings/ (pyright stubs)
third_party/          the AF3 feature package
```

Nothing machine-specific lives in the checkout. Weights, the CCD database, the
release environments, runs and benchmark inputs sit under `$FOLDFORGE_HOME`
(default `~/.cache/foldforge`), each overridable on its own
(`src/foldforge/paths.py`):

```
$FOLDFORGE_HOME/
  checkpoints/   converted blobs, one directory per model   FOLDFORGE_CHECKPOINT_DIR
  ccd/           preprocessed_CCD.lmdb                       FOLDFORGE_CCD_DB
  runs/          where `--out <name>` goes                   FOLDFORGE_RUNS
  releases/      release environments (reference regen)      FOLDFORGE_RELEASE_ROOT
  benchmarks/    benchmark input sets                        FOLDFORGE_BENCHMARKS
```

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
| representative AF3 block (Pairformer, MSA module, template, diffusion transformer) | miniworld-engine (`architecture`) |
| specific to one predictor | `models/architectures/<name>.py` |
| needed by two or more predictors, no reason for the engine to carry it | `modules/` |

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
| Protenix v1 / v2 | two models, `protenix1` and `protenix2` |
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
source tools/activate_env.sh
foldforge models
foldforge fold af3      --spec docs/examples/inference/1ubq.yaml --out $FOLDFORGE_HOME/runs/af3
foldforge fold protenix2 --spec docs/examples/inference/1ubq.yaml --out $FOLDFORGE_HOME/runs/protenix2
foldforge fold opendde  --spec docs/examples/inference/1ubq.yaml --out $FOLDFORGE_HOME/runs/opendde
foldforge fold esmfold2 --spec docs/examples/inference/1ubq.yaml --out $FOLDFORGE_HOME/runs/esmfold2
```

`--checkpoint` is optional when the release's default file is in
`$FOLDFORGE_HOME/checkpoints/<model>/`. The command writes CIF and JSON recording the
actual precision, sampling, compile and CUDA-graph settings. Every family
reads the MiniWorld BioMol LMDB at `$FOLDFORGE_HOME/ccd/preprocessed_CCD.lmdb`.

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

`tests/release_references/` holds, for three reference targets (protein; protein + ligand +
ion; DNA + protein + ions), every family's structures from its **own released
code**, three seeds of five samples. `foldforge validate` judges a FoldForge
run against them -- structure, pLDDT, ligand bond geometry and ligand/ion
placement, each relative to the release's own seed-to-seed spread -- and the
current verdicts are in [docs/validation/](docs/validation/). See
[tests/release_references/README.md](tests/release_references/README.md) to regenerate or extend them.

Today 37 of 38 family x condition rows pass in both modes, the MSA condition
included. The one exception,
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
- `$FOLDFORGE_HOME/benchmarks/`: prepared evaluation targets and model input specs.
- `validation/reports/`: validation logs and artifact relocation manifests.
- `validation/archive/`: historical scripts, temporary source copies and checks.
- `$FOLDFORGE_HOME/runs/`: all structure-prediction outputs, including benchmark predictions.

Both CLI forms enforce the same destination. `--out experiment` and
`--out $FOLDFORGE_HOME/runs/experiment` write to `<repo>/runs/experiment`, independent of the
working directory. Without `--out`, a unique `$FOLDFORGE_HOME/runs/<model>/<timestamp-id>/` is
used. Absolute paths must be within this runs tree; outside destinations fail
before input preparation or prediction writing. Prepared inputs, coordinates,
confidence files and the run report are kept together. Historical predictions
are preserved under `$FOLDFORGE_HOME/runs/archive/validation/`; relocation manifests map old
paths to their new locations.

Development checks: run `bash tools/check_quality.sh` for Ruff lint/format and
Pyright. See [code quality](docs/guides/CODE-QUALITY.md) for scope, retained numerical-code
exceptions, and compute-node regression checks.
