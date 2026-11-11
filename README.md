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

Weight setup: `foldforge weights download --all-public`; see the
[weight download guide](docs/guides/WEIGHTS.md) for all models, language-model
weights and offline installation. Inference uses PyTorch and CPU input preparation
uses NumPy. FoldForge does not use JAX, Haiku or Chex in inference,
input preparation or weight conversion.

## Code and loading

Use the [Python prediction API](docs/guides/PYTHON-API.md) from notebooks,
services or another package:

```python
import foldforge as ff

inputs = ff.load_inputs("input.yaml")
with ff.load_model("boltz2", config="config.yaml", device="cuda:0") as model:
    result = model(inputs)
result.save("./results/my-run")
```

For test-time scaling, retain the trunk and sample diffusion repeatedly:

```python
import foldforge as ff

inputs = ff.load_inputs("input.yaml")
with ff.load_model("boltz2", config="config.yaml") as model:
    with model.run_trunk(inputs, recycles=10, seed=0) as trunk:
        for seed in range(10):
            result = model.sample_diffusion(trunk, samples=5, steps=200, seed=seed)
            result.save(f"./results/seed-{seed}")
```

The CLI calls this same API:
`foldforge fold boltz2 --spec input.yaml --config config.yaml --out my-run`.

Start with [the directory guide](docs/guides/CODE-STRUCTURE.md).
`foldforge.load(model, checkpoint, backend=..., dtype=..., device=...)` is the
lower-level network loading API. Model directories and `ported/` packages have been removed;
architecture assembly, configuration, weight mapping and feature processing each
have their own owner.

```python
import torch
import foldforge as ff

model = ff.load("protenix2", backend="miniworld", dtype=torch.bfloat16, device="cuda")
```

`backend` is `miniworld` (the default: the engine's own kernels, `cuda` on an H100 else `triton` else `pytorch`, per module),
`pytorch`, `cuda`, `triton`, `anthropic` or `cueq` (the engine's backend names; `cuequivariance` is an alias of `cueq`); single ops can be
pinned to another one with `engine.ops`, and `engine.unsupported: fallback | error` decides what a call a backend cannot take does. Precision is
`master` (fp32 parameters, bf16 activations; config `precision: bf16`) or `fp32`. What each backend serves and how to compare them:
[docs/guides/BACKENDS.md](docs/guides/BACKENDS.md). The weights are read from the **miniworld checkpoint format** (`<model>.miniworld.pt`,
converted once from the released blob with `foldforge weights convert <model>`; [docs/guides/CHECKPOINT-FORMAT.md](docs/guides/CHECKPOINT-FORMAT.md)).

## Where this sits

FoldForge is a **terminal** repo on top of one dependency, miniworld-engine
(team-gm was absorbed into the engine and into MiniWorld; FoldForge no longer
depends on it). The boundary rule decides whether a piece of code belongs here or
a layer down:

| layer | repo | holds |
|---|---|---|
| engine | miniworld-engine | fused kernels, the modules that wrap them (`miniworld_engine.modules`) and the single-op kernel entry points (`miniworld_engine.kernels`); since 2.4.2 no model architecture, diffusion process or checkpoint adapter |
| terminal | **FoldForge** | the **full models**, the checkpoint installers, data pipeline, eval, CLI |

Nothing here defines an op or a general-purpose block. If you are writing one,
it goes in the engine. What the engine has no class for — the FoldForge-own pieces of the graph (`modules/dense`: featurization, atom cross attention, templates, heads),
the backend plan and its guard (`modules/engine`), compile/graph execution and bucketing
(`models/`), TFG guidance, the ESM-C model — lives here (what the engine lacks is listed in [docs/ENGINE-UNSUPPORTED.md](docs/ENGINE-UNSUPPORTED.md),
requests in [docs/ENGINE-REQUESTS.md](docs/ENGINE-REQUESTS.md)); see [the directory guide](docs/guides/CODE-STRUCTURE.md).

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

### Standard Python installation

The package declares its runtime engine dependency. The native AF3 data component
is built as a separate wheel; it must be installed alongside FoldForge. From a
source checkout with Python 3.12 and a C++20 build toolchain:

```bash
python -m pip install ./third_party/af3-data
python -m pip install '.[language]'
```

The `language` extra supplies Transformers for the vendored ESM-C tower; models
without that tower do not require it. The Biohub `esm` fork is a Pixi-only reference
validation dependency and is not required to run FoldForge's inference API.

To build distributable artifacts, run `python -m build --wheel --outdir wheelhouse`
for both the root project and `third_party/af3-data`. Install the resulting wheels
with `python -m pip install --find-links wheelhouse 'foldforge[language]'`.
Checkpoints and CCD/MSA/template databases remain separate resources. Inference
requires a supported CUDA runtime; installing the Python package does not qualify
all model/backend combinations on a new GPU.

### The engine pin

`miniworld-engine` is pinned once in the package runtime dependencies (`==2.5.1`); Pixi
resolves that same requirement. The committed `pixi.lock` records the 2.5.1 wheels (`miniworld-engine`, `-kernels-sm90-cu129`; the `-cu130` kernel wheel
of 2.4.4 is gone from the lock because 2.5.1 does not ship one). An environment installed from an older lock must be refreshed (`pixi install --frozen`). 2.5.1 brings its own prebuilt H100 kernels as the wheel
`miniworld-engine-kernels-sm90-cu129==2.5.1` (CPython 3.12, linux x86_64, CUDA 12.9 / torch 2.13); there is no CUDA 13 kernel wheel, so the `cu13`
environment JIT-builds kernels on first use. Update the pin,
refresh the lock and re-run the tests: the engine changes contracts between
releases (for example, since the 2026-10 release every `x = x + f(x)` module
returns the residual itself — use `.delta` for the bare update).

## Layout

```
src/foldforge/        the package
  cli.py              foldforge models / ccd / checkpoints / validate / fold <model>
  paths.py            where weights, databases, release envs and runs live
  models/             registry, the one AF3 architecture, checkpoints, loading,
                      execution, sampling, precision, io
  modules/            the FoldForge-own graph pieces (dense/), the engine adapter: backend plan,
                      guard, converter rules (engine/), shared equations (reference/), ESM-C (esmc/)
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

- **bf16 is the default.** Parameters are always fp32 (the miniworld-format files hold fp32; a released bf16 value is exact in fp32). `bf16`
  runs them under bf16 autocast (engine precision `master`), `fp32` as they are. Use fp32 when a task explicitly calls for it.
- **One MSA policy for every predictor.** Inputs keep up to 16384 alignment
  rows and each trunk pass embeds a fresh random subset of 1024 valid rows,
  the official AF3 pipeline rule, applied to all four checkpoints at load time.
  Released defaults differed; see [MiniWorld formats](docs/guides/MINIWORLD-FORMAT.md).
- **Never quote a speed number without its conditions** — device, dtype, MSA
  depth, warm or cold, and which backend *both* sides ran. Fold times on the
  same target and config have differed by 6% across processes purely from
  run-to-run variation (clocks, allocator state, kernel selection), so a before/after taken from two jobs is noise with
  a sign. A/B in one process, with a same-arm control that reports the noise
  floor.

### Where a block goes

| the unit | lives in |
|---|---|
| wraps a kernel, one PyTorch reference | miniworld-engine |
| diffusion process (EDM, decoupled x-prediction, augmentation, schedules, samplers) | `foldforge.diffusion` (self-contained: movable into its own package) |
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

`--checkpoint` is optional when the release's converted file (`<model>.miniworld.pt`) is in
`$FOLDFORGE_HOME/checkpoints/<model>/`; if it is missing the command stops and prints the exact `foldforge weights convert <model>` to run
(the loader never converts at run time). The command writes CIF and JSON recording the
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

Two steps: `foldforge weights download <model>` produces the released blob (`<model>.bin.zst`, below), and `foldforge weights convert <model>` converts it
once, offline, to the miniworld checkpoint format that `load_model` reads ([CHECKPOINT-FORMAT.md](docs/guides/CHECKPOINT-FORMAT.md)).

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
