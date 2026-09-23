Current contract: [MiniWorld formats](MINIWORLD-FORMAT.md). Current execution qualification: [2026-09-14](../archive/QUALIFICATION-20260914.md). Dated results below are historical and do not override these documents.

Current format and shared-composition reference: [MiniWorld formats](MINIWORLD-FORMAT.md).
The dated validation sections below describe the state when those checks ran.

# FoldForge model integration

Working tree: `main`; shared team-gm member: branch `foldforge/dense-families`.
All predictors must use FoldForge's environment, team-gm blocks, and
miniworld-engine ops. A package name or a subprocess launching an untouched
upstream model is not a completed shared-engine port.

**As of 2026-09-23 there is one graph.** `models/architectures/af3.py` is the
only architecture in the tree, and a predictor is a `DenseSpec` row in
`modules/dense/spec.py` stating where its release disagrees with AF3. The flat
and sequence layouts, and the per-model architectures that lived on them, are
deleted. Sections dated before this describe the separate ports they replaced.

See **One graph — 2026-09-23** below for each family's evidence.

Source pins and original licenses are retained in each model's `SOURCE.json` and
`UPSTREAM-LICENSE`:

- AF3 PyTorch: Xfold `22bdeedfa309ef4ff6f9199910d8403915de69d6`.
- AF3 CPU feature package: official AlphaFold3 v3.0.1,
  `231efc9bb9c13b45cc59e43f7107869084ee9624`, in `libs/af3-data`.
- Protenix: `4c355be4553512f72453ecbfb65e69f4c35d1413`.
- OpenDDE: `ddfa1df8aff1babf1fddac4247b7d2351bd0ce9f`.
- OpenDDE checkpoint and CCD assets: Hugging Face revision
  `eddd563ce96571f784012edd8f045181c8f8627d`; checkpoint SHA256
  `7b826620390afad877ee2babc6a4d0df81b94d3a0be030959853d6a7da0807cc`.

The upstream source trees are retained for provenance. Unused upstream training,
web-service, search and optional ESM utilities are not qualified FoldForge APIs.
Only the packaged loaders and inference commands described here are supported.

Current CCD and module-boundary status is recorded in
[code unification](../archive/CODE-UNIFICATION-20260913.md). The original port validation
below establishes execution, not complete migration to team-gm blocks.

## Running a fold

Every family takes the same MiniWorld YAML and the same flags -- there is one
CLI, not one per predictor. On an allocated GPU node:

```bash
source scripts/activate_env.sh
foldforge models
foldforge fold af3      --spec configs/inference/1ubq.yaml --out runs/af3
foldforge fold esmfold2 --spec configs/inference/1ubq.yaml --out runs/esmfold2
```

`--checkpoint` is optional when the release's default blob sits in
`model_checkpoints/<model>/` or in its family's directory. `--config` takes a
YAML of backend, precision, seeds, recycles, steps and execution settings; see
`runs/*/tools/config.yaml` in any recorded run for the shape.

A family with no template stack (`template_layers == 0`, both ESMFold2
releases) refuses a templated input, and one with no MSA stack
(`msa_layers == 0`, ESMFold2-Fast) ignores the alignment. Both rules are read
from the family row, so a new release needs no code.

The command writes per-sample CIF and a JSON recording the LM source,
precision, seeds, MSA depth, sampling settings and the actual compile/CUDA-graph
settings. This correctness runner uses compile off and CUDA graphs off.

## One graph — 2026-09-23

Every registered family is a `DenseSpec` row on `models/architectures/af3.py`.
Two kinds of evidence are recorded, because the first does not imply the second.

**Parameter trees.** `scripts/diff_dense_checkpoint.py` compares what the graph
wants against what the blob supplies. Eleven of twelve rows are exact with no
discrepancy in either direction: alphafold3 404, boltz2 440, chai1 397,
esmfold2 343, esmfold2-fast 303, intellifold2 404, openbind0 404, openfold3
404, protenix1 402, protenix2 402, rosettafold3 444. OpenDDE's blob is a raw
`.pt` the tool does not read; it is verified by folding instead.

**Folds.** A tree diff cannot see a convention -- a function applied to a
tensor owns no parameter -- and this project has three cases of an exact tree
folding to garbage. So a family is accepted by folding it. 5I28, PyTorch
backend, `af3_default` precision, 10 recycles, 200 steps, seeds 0/0, with AF3
as the control on the identical input:

| family | peptide C-N | clashes | CA vs AF3 | pLDDT |
|---|---|---|---|---|
| af3 (control) | 0 / 127 | 0 | -- | 96.11 |
| boltz2 | 0 / 127 | 0 | 0.667 A | 97.18 |
| intellifold2 | 0 / 127 | 0 | 0.148 A | 95.78 |
| openfold3 | 0 / 127 | 0 | 0.203 A | 95.06 |
| openfold3-preview2 | 0 / 127 | 0 | 0.209 A | 84.95 |
| rosettafold3 | 0 / 127 | 0 | 0.208 A | 86.87 |

Chai-1 on the same input folds 0 / 127 with 31 clashes at pLDDT 41; its
remaining gap is recorded separately. OpenDDE folds 0 / 127 on 5I28 and
0 / 75, 0 / 222, 0 / 591 on 1UBQ, 3PTB and 4YX2.

The two ESMFold2 releases have no template stack, so they take 1UBQ without
templates, measured against the deposited structure:

| family | peptide C-N | clashes | CA vs deposited 1UBQ | pLDDT |
|---|---|---|---|---|
| esmfold2 | 0 / 75 | 0 | 1.517 A (1.065 A at the release's own 3 loops / 14 steps) | 78.6 |
| esmfold2-fast | 0 / 75 | 0 | **7.528 A** | 59.6 |

**ESMFold2-Fast is accepted, on 6MRR.** It was not accepted for most of a
session on the strength of 1UBQ, where it folds to 6.7-17.7 A, and that was the
wrong test. The reference measures this family on **6MRR**, and on 1UBQ the
RELEASED implementation does not fold at all: every one of five seeds dies
inside its own Kabsch align with a non-converging SVD, its coordinates gone
degenerate. A poor number on a target the release itself cannot fold says
nothing about the port.

On 6MRR, no alignment, five seeds -- against what the reference records for the
same target and the native implementation it was comparing itself to:

| | FoldForge best | FoldForge mean | reference best / mean | native mean |
|---|---|---|---|---|
| esmfold2 | 1.644 A | **1.668 A** | 1.494 / 1.742 | 1.739 |
| esmfold2-fast | 1.662 A | **1.684 A** | 1.243 / 1.699 | 1.646 |

Both land inside the band. The row differs from `esmfold2` in `trunk_layers`
(48 -> 24) and `msa_layers` (4 -> 0) and in nothing else, matching the released
configs and the reference's registry, and it folds like it.

### Running a released implementation as the control

Two of this document's findings rest on folding the same target with the
RELEASE, and neither was reachable from reasoning alone. Both are set up here
so the next person does not rediscover the setup.

**ESMFold2** ships inside `transformers`, so it needs nothing extra:

```python
ESMFold2Model.from_pretrained("model_checkpoints/esmfold2")
```

Use the LOCAL path. `from_pretrained("biohub/ESMFold2")` fails in this build --
`DiffusionStructureHeadConfig.__init__() got an unexpected keyword argument
'architectures'` -- because the published config parses differently. Call it
with `num_diffusion_samples=1` and `num_sampling_steps=200`; asking for five
samples in one call hits a degenerate SVD inside the model's own Kabsch align.

**Chai-1** needs its weights and a few packages:

- Source: `/public_data/thalkak_envs/chai-lab` (already on this machine).
- Weights: `models_v2/*.pt` (1.1 GB) and `conformers_v1.apkl` (119 MB) from
  `https://chaiassets.com/chai1-inference-depencencies/`, into
  `model_checkpoints/chai1/`. Point `CHAI_DOWNLOADS_DIR` at that directory.
- Missing packages: `gemmi`, `antipickle==0.2.0`, `modelcif`, `typer`,
  `pandera`, `numba`. Install them to a directory and reach them with
  `PYTHONPATH` rather than into `.venv` -- they are for a control, not for
  FoldForge.
- **Pin `antipickle==0.2.0`.** 0.2.2 ships a built-in `torch` adapter whose
  typestring collides with the one `chai_lab` registers, and the collision is a
  bare `assert`, not a warning.
- `run_inference` refuses a non-empty output directory.

`ScriptModule` refuses `register_forward_hook`, and the traced blocks expose no
callable `forward`. The call site is ordinary Python, so wrap
`chai_lab.chai1.ModuleWrapper.forward` to capture any component's inputs and
outputs. Discriminate by an argument only one component has --
`atom_within_token_index` for the confidence head, `msa_input_feats` for the
trunk -- because both take a `token_single_trunk_repr`.

### Chai-1's confidence, narrowed and still open

Chai-1 folds 5I28 to 0.89 A CA-RMSD with 0 broken bonds -- a good structure --
and reports **pLDDT 41** where AF3 on the identical input reports 96, with a
median PAE of 4.80 A against 1.65. Its pLDDT never exceeds 0.499 across 128
tokens.

Ruled out, each measured rather than reasoned about:

- **The structure the head reads.** Tapping `ConfidenceHead.forward` and
  matching its `dense_atom_positions` against the written mmCIF gives
  **0.0005 A** on dense slot 1, for Chai-1 and for AF3 alike. The head is
  reading exactly the structure that got written.
- **The 37-slot pLDDT gather.** Chai-1 predicts over ATOM37 and gathers per
  atom by name; ours matches the reference operation for operation, including
  the `argmax`-on-no-hit behaviour and the `take_along` axis.
- **The bin arithmetic.** 50 bins, `bin_width = 1/50`, centres
  `arange(0.5*w, 1.0, w)`, `sum(softmax * centres) * 100` -- the reference's
  own lines.
- **The confidence pairformer.** The reference gated Chai-1's `pae_logits` and
  `pde_logits` at corr 0.999938 and 0.999916 against captured native I/O.
- **The language model.** Alive: 128 distinct ESM2 embeddings for 128 tokens.

**The release was downloaded and run, and the gap is real.** `chai_lab` is at
`/public_data/thalkak_envs/chai-lab`; `models_v2/` (1.1 GB) and
`conformers_v1.apkl` come from `chaiassets.com`. Pin `antipickle==0.2.0`: 0.2.2
ships a built-in `torch` adapter whose typestring collides with the one
`chai_lab` registers, and the collision is an assertion, not a warning.

On 5I28, same sequence, 3 recycles and 200 steps:

| | pLDDT mean | range | vs deposited |
|---|---|---|---|
| released Chai-1 | **97.45** | 75.8 - 98.9 | 0.692 A |
| FoldForge Chai-1 | **42.03** | 23.0 - 50.6 | 0.919 A |
| FoldForge AF3 | 98.28 | 90.7 - 99.0 | 0.674 A |

The two structures agree to 0.581 A, so the fold is right and only the
confidence is wrong -- by more than a factor of two.

Also cleared, by weight comparison against the released TorchScript: the pLDDT
projection is **bit-identical**, `max|diff| = 0.0` and corr 1.00000000, in the
atom-major layout we use. Matching it bin-major instead reads corr 0.408, so
the `(n_atom n_bins)` reshape convention is confirmed too.

**Not the cause: the head norms.** The released `confidence_head.pt` carries
five non-block parameters -- all bare projection weights, no norm tensor -- and
the converter invents `scale=1/offset=0` for `logits_ln`, `pae_logits_ln` and
`plddt_logits_ln`. That looked decisive, since a scale-1/offset-0 LayerNorm
still centres and rescales, and the reference records exactly that reasoning
for boltz2 in `NO_HEAD_NORM` while leaving Chai-1 out of it. Removing the three
norms moved pLDDT from 42.03 to **37.78** -- slightly worse, structure
unchanged -- so the reading was wrong: **no parameters does not mean no
normalisation.** An affine-free LayerNorm has zero parameters and still
normalises, which is what the converter's ones and zeros faithfully represent.
The change was reverted.

**Located: the trunk hands the head a different representation.** Wrapping
`ModuleWrapper.forward` -- the call site is ordinary Python even though the
traced blocks have no callable `forward`, and a ScriptModule refuses
`register_forward_hook` -- captures the released head's inputs. Against ours on
the same target, over the 128 real tokens:

| tensor into the confidence head | corr | ours RMS | released RMS | best scale | residual |
|---|---|---|---|---|---|
| `single` | +0.922 | 137.1 | 217.5 | 0.58 | 0.387 |
| `pair` | +0.942 | 96.9 | 97.3 | 0.94 | 0.335 |

Close but not equal: the pair's scale matches and the single's is 0.58x, and
best-fit scaling still leaves 39% and 33% of the energy unexplained. The
reference's own gates aim at 0.999, so 0.92 is not agreement. The diffusion
head survives this -- it conditions through its own norms, and the fold is
0.919 A against the release's 0.692 -- while the confidence head turns it into
97 against 42.

**The 2x2 that settles it.** The released head IS callable -- `forward` is
undefined but the bucketed `forward_256` .. `forward_3072` are all there -- so
each side's single and pair can be crossed into each head. Ours by injecting
into `ConfidenceHead.forward` during a real fold; the release's offline, in
bf16, with a control confirming the crop-and-pad harness costs nothing (the
release's own tensors through it still read 95.14).

| single | pair | RELEASED head | OUR head |
|---|---|---|---|
| release | release | 95.14 | 97.07 |
| release | ours | 93.39 | 96.04 |
| **ours** | release | 79.39 | **44.66** |
| **ours** | ours | 76.71 | **41.87** |

Three things fall out, and two of them correct claims made earlier in this
document:

- **The pair is not the problem.** Swapping it moves either head by about two
  points.
- **The single is.** It costs the RELEASED head 16 points, so it is genuinely
  degraded and not merely different -- consistent with corr 0.922 at 0.58x
  scale against the release's.
- **Our head is correct, and amplifies.** On the release's single it agrees
  with the released head to within two points (97.07 against 95.14, 96.04
  against 93.39). On ours it loses 52 points where the released head loses 16.
  So the head is not the root cause, but it is about three times as sensitive
  to a degraded single -- which is worth understanding separately and is not
  what to fix first.

The earlier readings in this document -- first "the trunk, not the head", then
"the head is the larger cause" -- were each half right, and both were asserted
from one half of this table. **The work is the trunk's single track.** Its
INITIAL single already matches (corr 0.991 against the release's), so the
divergence accumulates across the 48 trunk blocks rather than starting before
them.

The reference **deliberately did not gate this**: it compared LOGITS rather
than derived scores so that no assumption about Chai-1's bin centres entered,
and excluded pLDDT because, unlike PAE and PDE, it is per atom and needs
atom-layout agreement.

### The one real defect this found

**The language model was dead.** `build_lm_inputs` selected protein tokens with
`mol_type == PROTEIN_MOL_TYPE`, which is 0, while the caller passed
`is_protein`, which is 1 on protein -- so it kept exactly the tokens it meant to
drop. On an all-protein input nothing was packed, the tower saw an empty
sequence, and the shim turned its zeros into ONE 256-vector repeated at all
16384 pair positions (1 distinct row of 16384, max deviation exactly 0). Every
ESMFold2 fold ran with no language model.

It was invisible because it still folded: `esmfold2` reached 0.99 A on 1UBQ
because its MSA encoder carried the structure by itself, and the constant even
measured like a contribution, 45% of the injection by RMS. What gave it away is
that the reference records disabling the LM dropout as worth ~18 A, and
disabling ours moved the fold by 0.1 A. **An input whose ablation changes
nothing is not connected, whatever it measures.**
`scripts/check_live_inputs.py` asks the two questions that catch this class:
does the tensor VARY, and does a linear read-out of a pair stream rank true
contacts above chance.

After the fix the LM pair carries 60 of 100 top-ranked long-range pairs as true
contacts, against 2 before and a 4.7% baseline, and the 6MRR numbers above are
what it buys.

The port also reproduces the release where the release runs. 1UBQ, no
alignment, five seeds: released ESMFold2 best 5.775 / mean 7.354 A, FoldForge
best 5.776 / mean 7.621.

## Acceptance per model

1. Pin model source, released configuration and weight conversion; require strict
   loading, with no unexplained missing or unexpected parameters.
2. Check masks, residual ownership, normalization and precision at module level.
3. Run sequence/features through the complete checkpoint, with coordinates and
   confidence compared against the corresponding reference under recorded settings.
4. Verify compilation, inference graph capture, cache coverage and timing on the
   complete model before reporting performance.

Only completed ports appear as implemented in `foldforge models`; the other
names remain explicitly unported.

## Verified results — 2026-09-13

A6000, 1UBQ (76 residues), released ESMFold2 weights, native BF16 with FP32 norm
parameters, no autocast, MSA depth 512, seed 0, 3 recurrence loops, 14 diffusion
steps, 1 sample. Compile and CUDA graphs are off in this correctness run.

| Folding backend / LM input | pLDDT (0–1) | CA RMSD to deposited 1UBQ | CA RMSD to saved original reference |
| --- | --- | --- | --- |
| MiniWorld / cached ESMC | 0.8008 | 0.699 Å | 0.665 Å |
| MiniWorld / live ESMC-6B | 0.7969 | 0.684 Å | 0.633 Å |
| PyTorch / same cached ESMC | 0.7969 | 0.677 Å | not measured |

MiniWorld versus PyTorch under the same cached-LM setup: **0.203 Å CA RMSD**.
Every comparison matches all 76 residues, with zero residue-name mismatches.
The saved original reference used FP32 folding; its comparison is structural
validation across precision settings, not an exact numerical parity test.
These results establish one monomer case; complexes, ligands, larger targets,
full-model compile/graph performance, and model-shape cache completeness remain
outside this qualification.

Jobs: 1682652 (MiniWorld/cached LM), 1682655 (public CLI/live ESMC),
1682656 (PyTorch/cached LM), 1682658 (final tests and corrected CA rescoring).
**23 tests pass**, including the actual GPU diffusion-conditioning path,
ESMC RoPE/precision, CB-vs-CA scoring, and occupancy-free prediction CIF parsing.
Ruff passes for the new CLI, precision, LM, inference and evaluation code and
new tests. The live ESMC forward ran for 2.471 s; that is not an end-to-end timing
or a comparative performance benchmark.

Artifacts are under `validation/results/connection-20260913/{cached,computed,pytorch}/`.
Each contains `1ubq.cif` and the corrected `1ubq.json`. Logs are retained under
`validation/logs/esm2-*.{log,err}`. Failed probes are retained separately.
All consumer edits remain local and uncommitted; existing unrelated changes
and the team-gm member branch are preserved.

## AF3, Protenix and OpenDDE commands

Run on an allocated GPU node after `scripts/setup_env.sbatch` completes.
Input JSON must carry prepared MSA/template data; these commands do not run an
MSA search. Protenix/OpenDDE use their upstream sequence/job JSON schema, whereas
AF3 uses the official AF3 JSON schema. Example input files are in
`validation/inputs/data/1ubq/{af3-full,af-family}.json`.

```bash
source scripts/activate_env.sh
# Once for all models, independently of their checkpoints:
foldforge ccd prepare \
  --components model_checkpoints/opendde/common/components.cif \
  --rdkit model_checkpoints/opendde/common/components.cif.rdkit_mol.pkl \
  --out data/ccd/preprocessed_CCD.lmdb
foldforge ccd verify --ccd-db data/ccd/preprocessed_CCD.lmdb

foldforge fold af3 --input validation/inputs/data/1ubq/af3-full.json \
  --checkpoint model_checkpoints/af3/af3.bin.zst \
  --ccd-db data/ccd/preprocessed_CCD.lmdb --samples 1 --out runs/af3
foldforge fold protenix --input validation/inputs/data/1ubq/af-family.json \
  --checkpoint model_checkpoints/protenix/protenix_base_default_v1.0.0.pt \
  --ccd-db data/ccd/preprocessed_CCD.lmdb --samples 1 --out runs/protenix
# For v2, change both --variant and --checkpoint:
foldforge fold protenix --variant protenix-v2 \
  --input validation/inputs/data/1ubq/af-family.json \
  --checkpoint model_checkpoints/protenix/protenix-v2.pt \
  --ccd-db data/ccd/preprocessed_CCD.lmdb --samples 1 --out runs/protenix-v2
foldforge fold opendde --input validation/inputs/data/1ubq/af-family.json \
  --checkpoint model_checkpoints/opendde/opendde.pt \
  --ccd-db data/ccd/preprocessed_CCD.lmdb --samples 1 --out runs/opendde
```

Defaults are MiniWorld, native BF16, 10 recycles, 200 diffusion steps and 5
samples. Use `--backend pytorch` for the corresponding ported reference equations.
Protenix/OpenDDE templates are opt-in (`--templates`). `--no-msa` is a smoke-test
option. All commands use `--ccd-db` (default `FOLDFORGE_CCD_DB`, otherwise `data/ccd/preprocessed_CCD.lmdb`).
The former model-specific `--assets` options are replaced. The database is MiniWorld-compatible BioMol LMDB. Per-record metadata preserves
full CCD categories and prepared references, with a lazy AF3 mapping over those
same records. There are no runtime CIF/pickle database fallbacks. See
[MiniWorld formats](MINIWORLD-FORMAT.md) for the standard YAML entry point;
the JSON commands below remain explicit checkpoint-compatibility interfaces.

Each command writes per-sample CIF, raw model confidence and JSON run settings.
`*.prediction.pt` contains the common `Prediction` fields as CPU tensors: flat
sample-major coordinates, token pLDDT in [0,1], and token PAE in Angstrom.
Protenix/OpenDDE decoded atom pLDDT is already in [0,1] and is averaged over
each original token. A redundant division by 100 in the initial common payload
was corrected during the unification audit; raw heads, summary pLDDT and CIF
coordinates were unaffected. Original payloads were backed up and regenerated. Undefined heads
remain `None`; AF3 pTM aggregation is not yet exposed in this common payload,
while its raw confidence output is retained. AF3 writes raw arrays as NPZ;
Protenix/OpenDDE use a tensor dictionary.

## Shared operations and precision

`team_gm.modules.checkpoints.af_family` maps plain residual transitions, compatible residual
TriMul and LayerNorm onto miniworld-engine. It preserves masks and consumes the
fused residual exactly once. AF3 interleaved projection weights and incoming
contraction order are mapped explicitly. Protenix template TriMul with pair
width 64 and hidden width 128 retains its original PyTorch equation because the
current engine requires equal widths. Triangle attention, MSA/outer-product,
conditioned composites and model-specific heads retain model/PyTorch equations;
shared normalization may still use the engine inside them.

Learned parameters are native BF16 except FP32 normalization parameters. Fixed
Fourier, geometry and confidence-bin buffers retain their original precision.
No autocast is used. Explicit FP32 attention/head calculations remain where the
model equations require them, with casts at projection boundaries. This is a
parameter-precision policy, not a claim that every intermediate is BF16.

Full inference found and fixed the Xfold sampler overwriting its sampled noise
field, mixed-dtype attention/head contractions, and obsolete Biotite private
bond-array mutation. Strict AF3 conversion also checks source keys, shapes and
stack cardinality. Tests exercise nonzero projection weights, arbitrary masks,
incoming/outgoing equations, single residual ownership, native BF16 attention,
normalization without affine parameters, and the random-noise regression.

## AF-family verification — 2026-09-13

A6000 on gpu03, Slurm job 1682660. Complete 1UBQ (76 residues), seed 0, prepared
MSA, templates disabled, 10 recycles, 200 diffusion steps, 1 sample. Both backends
use identical native BF16 policy, compile OFF and CUDA graphs OFF. All CA
comparisons match 76 residues with zero residue-name mismatches.

| Model | MiniWorld pLDDT (0–100) | PyTorch pLDDT | MiniWorld vs deposited CA RMSD | PyTorch vs deposited CA RMSD | MiniWorld vs PyTorch CA RMSD |
| --- | ---: | ---: | ---: | ---: | ---: |
| AF3 | 91.233 | 91.211 | 0.709 Å | 0.696 Å | 0.058 Å |
| Protenix v1 | 94.079 | 94.075 | 2.579 Å | 2.552 Å | 0.138 Å |
| Protenix v2 | 93.347 | 93.326 | 2.702 Å | 2.720 Å | 0.073 Å |
| OpenDDE | 94.816 | 94.832 | 1.025 Å | 1.019 Å | 0.049 Å |

These compare MiniWorld to the ported PyTorch equations, not to a bit-exact
original JAX/FP32 upstream run. Protenix's approximately 2.6–2.7 Å deposited
structure difference is present in both backends. One monomer does not qualify
complexes, ligands, training, large targets or other GPUs. The supported June
2025 Protenix v1 configuration has not received the full comparison shown above.

The locked environment was rebuilt and checked successfully, including its AF3
native feature extension. The public CLI then passed two-sample smoke inference
for AF3, Protenix v1 and OpenDDE. Full results and `comparison.json` are under
`validation/results/ports-20260913/`; logs are under `validation/logs/ports-*`.
Final full suite: **47 tests passed**, with five missing-shape autotune warnings
(`validation/logs/ports-final-tests2-20260913.log`). New integration code passes
Ruff; `uv lock --check` and `uv pip check` pass (209 resolved / 175 installed
packages). Public two-sample payloads have coordinates `(2, 602, 3)`, token
pLDDT `(2, 76)` and PAE `(2, 76, 76)`, all finite. Confidence averaging and
explicit AF3 checkpoint filename selection have regression tests. The residual
audit classifies upstream raw-delta adds by full model-relative path. Failed
probes are preserved.

Some new model shapes miss tuned autotune entries and use the engine's heuristic
candidate selection. Cache coverage, hot timing and full-model compile/graph
qualification remain separate work. Cold elapsed times include compilation and
are not speedup measurements.

The root team-gm branch is `exp/miniworld-integrated`; its engine connection is
in place. The single transition inside Pairformer now forwards the requested
backend in both team-gm copies, with a passing root regression. FoldForge stays
on `main` and its team-gm member stays on `exp/miniworld`. Consumer changes remain
local and uncommitted; existing unrelated edits are preserved.


## Proposed result storage contract (not implemented)

Resolve the output root independently of the source checkout. All permanent
prediction artifacts should live under the deployment's top-level `runs/`, even
when executing a source snapshot. Use one unique directory per run; refuse a
nonempty destination unless an explicit, validated resume is requested.

For grouped benchmarks or seed sweeps, use a single level of named cases:

```text
runs/<run-id>/
  run.json
  af3-miniworld_graph-t7-d19/
    sample-000.cif
    sample-001.cif
    confidence.npz
    result.json
  logs/
```

`run.json` indexes cases and records source/checkpoint/input hashes and resolved
configuration. `result.json` records the two seeds, sample indices, confidence
summaries, timing scope, compile/graph observations, artifact paths and completion
status. The sample index is not an additional seed. Writes should use temporary
files and publish the completion manifest last, so interrupted output is not
mistaken for a successful run.

Default artifacts are mmCIF coordinates, compact common confidence arrays and
JSON metadata. Full PAE/PDE/distogram matrices, raw model dictionaries, input
feature tensors and intermediate trajectories should be explicit diagnostics,
not unconditional duplicate `.pt`/`.npz` output. Use numeric arrays with documented
axes, units and masks; reading outputs should not require model-class pickles.

Compiler/autotune caches belong in a shared versioned cache location. Disposable
source snapshots belong in job-local scratch, with the exact source archive/hash
retained only as needed for provenance. Existing `.bench` artifacts must be
inventoried and migrated with their manifest paths intact before any deletion.

## Optional diagnostic images (implemented)

All four inference adapters use the same image writer. PNG export is disabled by
default. Both CLI forms (`--spec` and legacy `--input`/`--input-spec`) accept:

```bash
foldforge fold af3 --spec input.yaml --config runtime.yaml --save-images all
# Or select a subset:
foldforge fold af3 --spec input.yaml --save-images pae pde plddt msa
```

The equivalent runtime YAML (also supported as `output` in notebook requests) is:

```yaml
output:
  images: [pae, pde, distogram, plddt, msa, template]
```

CLI selection overrides the YAML selection. Empty `images: []` is the default.
Images are saved beside the existing CIF/JSON artifacts under
`runs/<run>/images/<target>/`; the result JSON records relative paths and reasons
for requested images that were unavailable. This addition does not implement the
proposed output migration above or change existing CIF filenames.

- `sample-000-pae.png`, `sample-000-pde.png`: token-pair errors in angstroms, one
  image per diffusion sample, with a common scale across samples for each head.
- `sample-000-plddt.png`: token pLDDT on the 0–100 scale, one image per sample.
- `distogram.png`: trunk distance distribution summarized as the **most likely
  distance-bin index**, not an angstrom-valued distance map. Bin definitions can
  differ between checkpoints; do not compare these colors as physical distances.
  Saved once per trunk run. ESMFold2's optional distogram head is enabled only
  when this image is requested; other models retain their already-computed logits.
- `input-msa.png`: match/mismatch to the prepared query row, with gaps and padding
  blank. Displays at most the first 1,024 valid rows without reordering them.
  `input-msa-coverage.png` counts non-gap entries across **all** prepared rows.
  These show the input alignment, before stochastic internal MSA row sampling.
- `input-template.png`: resolved-atom coverage on query tokens for nonempty input
  templates. Empty template slots are excluded. If no templates were supplied or
  the checkpoint has no template path (ESMFold2), JSON records the omission.

Input images exclude padded tokens/rows; confidence images use decoded unpadded
outputs. The two seeds continue to control model/input randomness: rendering does
not draw from those RNG streams. Input image preparation precedes the timed
forward, and CPU PNG rendering follows it. Requested head computation and retention
can add inference work/memory (especially ESMFold2 distogram); keep image options
identical when comparing runtimes. Full optional distogram logits are not added to
raw result archives merely to save their PNG. Matplotlib is a declared runtime
dependency and imported only when rendering.
