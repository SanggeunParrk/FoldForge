Current contract: [MiniWorld formats](../guides/MINIWORLD-FORMAT.md). Current execution qualification: [2026-09-14](QUALIFICATION-20260914.md). Dated results below are historical and do not override these documents.

# Code unification audit — 2026-09-13

This records the **first pass**. See [the continuation](CODE-UNIFICATION-CONTINUED-20260913.md)
for subsequent MSA, diffusion, template and precision changes and verification.

The initial AF-family ports passed checkpoint inference but were not fully
rebuilt in team-gm style. Their upstream block compositions, data-path globals
and configuration conventions remained. This audit separates functional
compatibility from code ownership; it does not relabel a vendored model as a
complete shared-framework port.

## Reference and boundaries

MiniWorld's `src/miniworld/models/af3_like/model.py` assembles imported
`team_gm.modules.{Pairformer,MSAModule,DiffusionTransformer}` and keeps its own
input embedding, heads and full model. Configurations are explicit nested
Pydantic `Config` classes. Its `data/inference/ccd.py:CCDLookup` receives a
`ccd_db` path and owns per-database residue/molecule caches. These ownership
rules, plus `libs/team-gm/docs/ARCHITECTURE.md`, are the reference here.

MiniWorld's existing CCD store is a preprocessed BioMol LMDB. This change does
not modify MiniWorld or claim that its LMDB contains all raw AF3 CCD categories.
FoldForge has one model-independent CCD database and explicit views; it does
not reconstruct missing CCD metadata or replace learned reference conventions.

## Changes made

| Concern | Before | Now |
| --- | --- | --- |
| CCD selection | Protenix globals, OpenDDE globals, AF3 assets, ESM download/cache | One `CCDDatabase`, `--ccd-db`, source manifest and shared file ownership |
| CCD operations | Two copies of atom lookup, conformer, permutation and bond helpers | Same function objects from `foldforge.data.ccd.components` |
| Cache lifetime | Global caches could survive path changes | Instance-owned caches selected by a nested, context-local activation |
| AF3 CCD sets | Import-time global glycan sets | Explicit scoped set source; nested databases restore their prior view |
| ESMFold2 CCD | Independent Biohub CCD/download | Lazy atom-property view of the same prepared RDKit references and raw CCD |
| Pairformer composition | Three independent operation orderings | `team_gm.PairformerBlock.from_components`, used by all three AF families |
| Backend state | Ad-hoc string on every source module | team-gm `ImplementationType` at the common model/framework boundary |
| Common pLDDT | Protenix/OpenDDE values divided by 100 twice | Keep decoded [0,1] units; regression uses the actual decoder/bin config |

The Pairformer adapters own only released signatures and operation contracts.
They retain projection layouts, incoming/outgoing direction, attention masks,
chunking and OpenDDE's extra attention bias. AF3 uses 54 common blocks,
Protenix v1/v2 use 58, and OpenDDE uses 62 (including template/confidence blocks).
Weights load strictly before composition conversion. `forward_source` in
OpenDDE also enters the same shared block. These adapters are currently
inference-only; Fold-CP/distributed and training are not qualified by this work.

The common block delegates self-residual ownership to its components. Existing
raw attention equations require a residual adapter; fused transition/TriMul
results are not added twice. This unifies composition, not all underlying
attention kernel implementations.

## One CCD input for every model

On an allocated compute node:

```bash
source scripts/activate_env.sh
foldforge ccd prepare \
  --components model_checkpoints/opendde/common/components.cif \
  --rdkit model_checkpoints/opendde/common/components.cif.rdkit_mol.pkl \
  --out data/ccd
foldforge ccd verify --ccd-db data/ccd
export FOLDFORGE_CCD_DB="$PWD/data/ccd"
```

The preparation inputs shown above are the existing verified assets; the output
is independent of OpenDDE and of every model checkpoint. `data/ccd` is ignored by
git. Every `foldforge fold <model>` and the FoldForge profiling/cache scripts use
`--ccd-db`, defaulting to that environment variable or `data/ccd`.

```
data/ccd/
  manifest.json
  components.cif
  molecules.pkl
  af3/ccd.pickle
  af3/chemical_component_sets.pickle
  af3/manifest.json
```

The root manifest fingerprints every file. AF3's derived manifest must match the
raw CCD source hash. Each queried RDKit molecule is checked against CCD atom
names/indices and its reference mask. There is no implicit model-dependent
network fallback. Each prediction JSON records the same CCD source identities.

The model views preserve atom ordering, significant-hydrogen/leaving-atom policy
and required coordinate representation. ESMFold2 atom properties are added to
copies of the prepared molecules; their selected conformer coordinates are not
regenerated. AF3 still constructs its official reference features from the
shared raw CCD/index. Shared assets do not imply identical tokenization or
identical atom counts across predictors.

## Module and layer audit

[The updated class inventory](model-boundaries-20260913.csv) contains 145
(first pass: 143)
forward-bearing classes, including retained reference/helper definitions.
The script records file/line, base class, nested Config, annotated forward and
imports. These are structural indicators, not a numerical correctness score;
OpenDDE also has external Pydantic schemas even where a module has no nested
Config. The inventory includes code not reached by the qualified inference path.

| Family | Forward-bearing definitions | Nested Config on that class | Fully annotated forward signature |
| --- | ---: | ---: | ---: |
| AF3 source | 26 | 0 | 22 |
| Protenix source | 52 | 0 | 49 |
| OpenDDE source | 37 | 0 | 34 |
| ESMFold2 model | 22 | 3 | 22 |

The first pass had six explicit shared checkpoint adapter definitions; the
continuation adds two conditioning adapters. Counts
are source-level, so a retained `PairformerBlock` definition still appears even
though the loader replaces that composition at runtime.

| Module / layer family | Current ownership | Remaining unification work |
| --- | --- | --- |
| Full model, feature embedding, recycle loop, confidence/distogram heads | Terminal model, as in MiniWorld | Keep model-specific dimensions, heads, structural tokens and conditioning |
| Pairformer | Common team-gm composition after strict loading | Convert compatible raw attention components to shared engine modules with verified bias/mask mappings |
| Plain transition / TriMul | Common checkpoint adapters call engine ops | Replace raw-layout compatibility modules with converted engine module state; asymmetric template widths still require their reference equation |
| LayerNorm / projection precision | Common native BF16 policy, FP32 norm parameters and fixed buffers | Remove unused legacy norm definitions; replace activation-casting hooks with explicit shared layer contracts |
| MSA / OuterProductMean | Upstream AF3/Protenix/OpenDDE compositions remain | Move equivalent block ordering to team-gm; explicitly map sequence masks, contraction weights and chunking |
| Template embedding | Model-specific features plus shared Pairformer inside | Separate model-specific features from common template block composition; preserve 64→128 asymmetric contractions |
| Token diffusion / AdaLN / conditioned transition | Upstream compositions remain | Convert released weights into shared conditioned blocks; keep bias placement, sigmoid gates, norm epsilon and super-block pair logits explicit |
| Atom attention | Upstream AF3-family windowed/cross attention remains | Preserve gather/scatter layouts and query/key window masks before sharing compositions |
| ESMFold2 atom DiT | Model-specific DiT modulation with shared engine SWA attention | Correctly remains distinct from AF3 sigmoid-scale AdaLN; do not substitute a different algorithm merely for matching names |
| Schedulers / samplers | Released model-specific implementations | Audit schedule and augmentation semantics before sharing team-gm diffusion utilities |

AF3 packed/interleaved projection layout, Protenix/OpenDDE attention biases,
OpenDDE structural conditioning and ESMFold2 DiT modulation are concrete
checkpoint differences. Their configuration should be explicit; duplicating
whole general-purpose modules is not the desired final architecture. Therefore
**the repository is not yet fully uniform in team-gm style**. The table above is
an actionable record of the remaining boundary work, not a claim that every
module was rewritten in this change.

## Verification

A6000 gpu03, Slurm 1682818. Native BF16 learned parameters, FP32 normalization,
no autocast; compile/graphs off. AF3/Protenix v1/v2/OpenDDE each ran full 1UBQ
inference (10 recycles, 200 diffusion steps, one sample, prepared MSA).
ESMFold2 ran its released setup (3 loops, 14 steps, cached ESMC, MSA 512).
All use the same CCD manifest.

| Model | CA RMSD vs pre-unification output | CA RMSD vs deposited 1UBQ | Summary pLDDT (0–100) |
| --- | ---: | ---: | ---: |
| AF3 | 0.050 Å | 0.701 Å | 91.224 |
| Protenix v1 | 0.093 Å | 2.576 Å | 94.070 |
| Protenix v2 | 0.106 Å | 2.695 Å | 93.335 |
| OpenDDE | 0.052 Å | 1.037 Å | 94.824 |
| ESMFold2 | 0.082 Å | 0.700 Å | 80.469 |

Every CA comparison matches 76 residues with no name mismatches. The ESMFold2
comparison additionally changes its old independent CCD source; it is a
structural regression, not bitwise equality. Per-model atom means and common
per-token pLDDT means have different weighting and need not be identical.

The common CCD view check covers 31 amino-acid, RNA/DNA and ligand entries,
checking atom identities and exact selected-conformer coordinates. Additional
ATP/NAG/HEM ligand-coordinate lookups preserve those coordinates exactly, and
ATP-only ESM input preparation succeeds (32 atoms). The ligand-specific cache
branch has its own nested-database regression; static name/import checks also
cover the adapted input code. This qualifies ligand feature construction, not
full complex/ligand structure prediction. A6000 tests
compare nonzero-weight Pairformer blocks with arbitrary pair masks, plus nested
CCD caches, AF3 glycan-source restoration and common lookup identity. Full
model runs caught OpenDDE's additional `forward_source` entry point.

The pLDDT unit bug affected only canonical Protenix/OpenDDE `.prediction.pt`
files. Fourteen payloads were regenerated from saved raw confidence; original
copies have `.before-unit-fix` suffixes. Existing structure coordinates and
original summary pLDDT were correct. Tests now use both models' actual decoder
configuration rather than an assumed synthetic unit convention.

Final verification: **56 FoldForge tests passed**, plus the root team-gm
backend regression (**1 passed**). Five expected missing-shape autotune warnings
remain. New common code passes Ruff; `uv lock --check`, `uv pip check` and
`git diff --check` pass. Environment pins remain Torch 2.10.0+cu128, Quack 0.5.0,
CUTLASS DSL 4.5.2 and FlashAttention 2.8.3.post1. Test log:
`validation/logs/unify-final-tests4.log`.

Artifacts: `validation/results/unify-20260913/`, including `comparison.json` and
`prediction-unit-repair.json`. Logs: `validation/logs/unify-*`. Pre-change source
copies and failed probes are retained. This is an input/composition correctness
check, not a performance or complete cache-coverage benchmark.

Root team-gm remains `exp/miniworld-integrated`; FoldForge remains `main`; its
team-gm member remains `exp/miniworld`. Changes are local and uncommitted.
