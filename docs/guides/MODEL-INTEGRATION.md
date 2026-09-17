Current contract: [MiniWorld formats](MINIWORLD-FORMAT.md). Current execution qualification: [2026-09-14](../archive/QUALIFICATION-20260914.md). Dated results below are historical and do not override these documents.

Current format and shared-composition reference: [MiniWorld formats](MINIWORLD-FORMAT.md).
The dated validation sections below describe the state when those checks ran.

# FoldForge model integration

Working tree: `main`; shared team-gm member: `exp/miniworld`.
All predictors must use FoldForge's environment, team-gm blocks, and
miniworld-engine ops. A package name or a subprocess launching an untouched
upstream model is not a completed shared-engine port.

| Predictor | Current implementation | Strict checkpoint evidence |
| --- | --- | --- |
| ESMFold2 | Live ESMC and folding through the public CLI | Released ESMFold2 and ESMC-6B weights |
| AF3 | Xfold PyTorch model, official AF3 features, shared engine ops | 405 source records: 404 parameters/buffers plus identifier metadata; 368,384,666 parameters |
| Protenix v1 / v2 | Pinned official models with shared engine ops | 4,174 state entries each; 368,484,735 / 464,442,431 parameters |
| OpenDDE | Pinned official model with shared engine ops | 4,482 state entries; 655,791,538 parameters |

These are in-process FoldForge ports, including the model-specific diffusion and
confidence heads. They preserve released weight names and load strictly. They do
not yet replace every upstream operation with an engine kernel.

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

## ESMFold2 command

On an allocated GPU node, after installing the environment:

```bash
source scripts/activate_env.sh
foldforge models
foldforge fold esmfold2 --target 1ubq --lm-source compute \
  --out runs/esmfold2-live
```

`compute` loads ESMC-6B and computes embeddings from the input sequence; it is the
default. `--lm-source cache` explicitly uses the existing target embedding file.
That legacy file has no source-version manifest and is not treated as proof that
the current LM was executed. `--checkpoint` and `--lm-checkpoint` override the
checkpoint resolver. Input targets currently use `validation/inputs/data/<target>/`
with `target.json` and prepared MSA files; this command does not run an MSA search.

The registry loader converts and strictly loads the released folding weights.
The command writes `<target>.cif` and `<target>.json`, recording the LM source,
precision, seed, MSA depth, sampling settings, and actual compile/graph settings.
Native BF16 retains norm parameters in FP32; no autocast is used.
This correctness runner uses compile off and CUDA graphs off. The earlier SWA
module graph check does not establish full-model graph coverage or performance.

During full-model verification, the noise-conditioning LayerNorm exposed a
missing semantic length dimension. The conditioning path now passes `[BS, 1, D]`
to LayerNorm. A GPU regression compares the complete conditioning module with
PyTorch. Feature preparation no longer injects fake `team_gm.core`, Lightning,
or cuEquivariance modules into `sys.modules`.

The original scoring helper incorrectly selected `distogram_atom_idx` (usually
CB) while comparing with CA coordinates. It now selects explicit protein CA
atoms and matches residue identity. CIF parsing also supports prediction files
without occupancy columns. The former CB-vs-CA numbers are invalid; the original
JSON files are preserved with `.before-ca-fix.json` suffixes.

The public ESMC loader retains its upstream RoPE `_apply` hook (including meta
buffer materialization and FP32 frequency regeneration). Its PyTorch fallback
normalizers explicitly reduce in FP32 before native-BF16 projections; tiny-model
regressions cover FP32 parity and BF16 loading.

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
