# MiniWorld format contract

FoldForge uses MiniWorld's inference YAML, FASTA grammar and CCDMol serialization.
`miniworld-engine` owns kernels and standalone operators; `team-gm` owns reusable
block compositions; FoldForge owns complete predictors, input packing, heads and
released-checkpoint conversion. Model-specific equations, vocabularies and weight
layouts remain explicit at those boundaries.

## CCD

The primary database is a directory LMDB, keyed by UTF-8 CCD ID. Each value is
`CCDMol.to_bytes()` with `atoms`, `residues`, `chains`, `index_table`, `metadata`.
MiniWorld's original `CCDMol.from_bytes()` reads these records directly. The core
atom view excludes H/D and keeps CCD **model** coordinates, charge, aromatic and
stereo flags. Ring/conjugation fields use the prepared sanitized molecular graph.
`CCDLookup[key]` and `CCDLookup.fragments(key)` follow MiniWorld's lookup and v2
fragmentation conventions.

Full CIF categories, leaving atoms, hydrogen-bearing topology, prepared RDKit
conformers and their masks are preserved in each record's metadata. These are
needed by released checkpoints and do not replace the canonical model coordinates.
AF3 and Biotite differ in multiline text whitespace; sparse AF3 text overrides
preserve the exact input of each parser. AF3's `data_` key is restored from the
record ID. Every model derives its chemistry view from the selected LMDB; runtime
access does not read a separate CIF, RDKit pickle or AF3 pickle database.

`manifest.json` records source hashes, LMDB hash, entry count and chemical-component
sets. It also lists components whose source RDKit chemistry cannot support
fragmentation: these remain present for ordinary CCD lookup, while fragmentation
raises an explicit error. No fictitious rings or bonds are substituted.

Prepare once on an allocated compute node:

```bash
source scripts/activate_env.sh
foldforge ccd prepare \
  --components data/ccd/components.cif \
  --rdkit data/ccd/molecules.pkl \
  --out data/ccd/preprocessed_CCD.lmdb
foldforge ccd verify --ccd-db data/ccd/preprocessed_CCD.lmdb
```

The old files above are migration inputs only. Preparation uses a temporary sibling
directory and renames it after success; an existing destination is never overwritten.
The environment pins remain unchanged; `lmdb` is now a direct locked dependency.
`FOLDFORGE_CCD_DB` overrides the CLI default. A plain external MiniWorld LMDB can
be opened with `CCDLookup`; released-model views additionally require the full
chemistry metadata produced by `ccd prepare`.

Readers are shared by path within a process, reopened after fork, and excluded
from pickle transport. Caches belong to the selected lookup/database instance.

## Inputs and settings

```bash
foldforge fold af3 --spec configs/inference/1ubq.yaml \
  --config configs/inference/native-bf16.yaml --out runs/af3
foldforge fold protenix --spec configs/inference/1ubq.yaml \
  --config configs/inference/protenix-v2.yaml --out runs/protenix-v2
```

The same `--spec` works with `af3`, `protenix`, `opendde` and `esmfold2`. Paths in
it are relative to the spec file. Numeric `chain_letters` keys determine chain
order; repeated letters share FASTA/MSA input. The adapter emits unique chain IDs
and records the mapping in `chain-map.json`. MiniWorld's typed FASTA headers support
protein, RNA, DNA and CCD ligands. No model chooses another CCD release implicitly.

`--config` is a nested Pydantic `Config` with `trunk` and `diffusion` sections.
The default is native BF16 learned parameters with FP32 normalization parameters,
explicit checkpoint-specific FP32 calculations, and no autocast. A null recycle or
step setting preserves the released default: AF-family 10/200, ESMFold2 3/14.
An explicit `trunk.msa_depth` caps input alignment rows before AF-family adapters,
and limits the ESM input MSA. Null preserves model defaults (ESM input limit 512).
Unknown config/spec keys fail validation rather than silently doing nothing.

Each output contains the resolved spec, model settings, chain mapping, actual run
settings and CCD hashes. AF-family adapter JSON is saved for inspection. ESM
embeddings are computed by default; `--lm-cache <file>` explicitly selects existing
embeddings. Compile and CUDA graphs remain off in these qualified inference runners.

MiniWorld-specific contact conditioning, fragment-resolution tokenization,
refinement/docking, chain-group diffusion and branched FASTA bonds
are not defined by these released checkpoint adapters. These options raise
at the adapter boundary rather than being ignored. This runner currently accepts
one trunk sample, unchunked diffusion samples, and `save_trajectory: false`.
Native checkpoint JSON remains available for model-specific templates and features;
it uses the same CCD database and precision/block plumbing.

## MiniWorld MSA-LMDB and template-LMDB

`foldforge fold MODEL --spec input.yaml --out results/target` reads the existing
StructCooker/MiniWorld serialized records directly. Fragment tokenization is not
required. Add the following fields to the normal FASTA/CCD spec:

```yaml
# MSA paths and lookup keys use FASTA chain letters, like a3m.
# Each letter names its exact shard; there is no scan over large DBs or keys.
msa_db:
  A: /path/to/msa-shard.lmdb
msa:
  A: sequence_lookup_key
# Templates retain MiniWorld's numeric chain-index mapping.
template_db: /path/to/template.lmdb
template:
  "0": template_lookup_key
template_n: 4
```

Paths can be relative to the spec. Directory and single-file LMDB environments
are supported, readonly. `msa` and `msa_db` must contain matching letters; `a3m`
and LMDB cannot both select the same letter. Homomer letters reuse one MSA.
Missing keys and query/length mismatches raise before checkpoint loading.
Template keys are used verbatim (PDB template records commonly use `pdb_labelchain`,
while MSA records use a deduplicated sequence ID). No PDB-name/key guessing.

| Input | AF3 | Protenix v1/v2 | OpenDDE | ESMFold2 |
|---|---|---|---|---|
| Protein MSA-LMDB | Yes | Yes | Yes | Yes |
| RNA MSA-LMDB | Yes | Yes | Yes | No checkpoint input path |
| Protein template-LMDB | Yes | Yes | Yes | No checkpoint input path |

An unsupported conditioning request raises rather than being ignored. DNA MSAs
are not consumed by these released input paths and are rejected; the generic
reader can decode the tokens but the adapters do not silently discard them. Templates for
non-protein chains and complex/docking templates remain outside these checkpoint
input contracts.

MSA records are BioMol zstd dictionaries with `msa_dict.sequences` and
`msa_dict.headers`. Protein 0..20, RNA 21..25, DNA 26..30 and gap 31 are decoded
using MiniWorld's alphabet. The query may be a character array (StructCooker) or
integer tokens. Alignment rows and insertion counts are retained: lowercase `x`
in the generated A3M carries the stored count because the original inserted
letters are absent from this DB schema. Original `profile` and `deletion_mean`
remain available on the decoded alignment; each checkpoint computes its own
profile/statistics with its native deduplication and cropping rules.

Full species strings are assigned collision-free aliases shared across the
LMDB chains in one input. These are transport identifiers, not biological
accessions. They allow the native AF3/P/O species pairing engines to consume the
same grouping; `N/A`/query/missing species are excluded. The alias map is saved in
`msa-species-aliases.json`. Plain `a3m` files retain their existing unpaired-input
contract. ESMFold2 uses its released MSA processor.

Templates read `template_mols` via the original `TemplateMol` views. Empty aligned
residue codes become gaps; finite N/CA/C/CB coordinates retain atom masks. A CA
copied into the fourth slot by StructCooker's pseudo-beta fallback is not labeled
as an observed CB. Native featurizers select glycine CA themselves. At least four
residues with a complete backbone are required, as in MiniWorld; the first valid
0..4 records in DB order are used deterministically. `template_n: 0` skips reading
the template DB. AF3 receives single-chain mmCIF and the exact alignment mapping;
P/O receive atom37 features at their existing template assembly boundary. There
is no search, realignment, remote fetch, or additional template database.

Known `metadata.release_date` is retained. When absent, the native template
transport uses `9999-12-31` (the P/O supplied-template convention) because AF3
requires a date; this is not an asserted historical date. Explicit supplied
hits are not subjected to a training-date filter. `input.resources.json` records
selected keys, actual MSA rows, template IDs, atom/backbone coverage and nullable
release dates. It also makes empty/filtered template inputs visible.

## Blocks and weights

All AF-family Pairformer, MSA update ordering, MSA row chunks, template aggregation
and conditioned residual ordering use shared team-gm composition. Loaded AdaLN and
conditioned transitions use the existing engine modules, including explicit mixed
precision handling. Plain residual transition/TriMul adapters retain the already
validated engine wiring. Shared attention math now owns the P/O attention equation
and AF3 torch attention, including scaling, mask replacement and precision rules.
OPM contraction/projection uses common code with an explicit AF3 layout to preserve
its reduction order; epsilon and post-projection bias normalization stay exact.

Strict checkpoint loading happens before layout conversion. Original source class
signatures remain compatibility boundaries; they are not evidence of different
CCD ownership or a separate generic block algorithm. Model-specific atom-window
indexing, schedulers, feature encodings and heads preserve their trained equations.
The present qualification covers single-GPU inference, not distributed training.

Source file identities and hashes for the MiniWorld formats are recorded in
`src/foldforge/data/MINIWORLD-SOURCE.json`. Frozen pre-migration forward/attention
oracles in `tests/references` support regression comparisons after promotion.

## Validation (2026-09-14)

A6000, allocated `gpu03`, Slurm `1684556`; native BF16, FP32 normalization,
no autocast, compile off, CUDA graphs off. All 158 tests passed in the final installed working tree (98.40 s). The full CCD audit
verified 48,965 entries: no category/reference-coordinate/glycan mismatches.
48,912 final records were also byte-identical to the fully audited database;
53 differed only in the separately verified AF3 multiline-text metadata.
248 source entries lack valid fragmentation chemistry and are explicitly listed.

Actual checkpoint inference used the same MiniWorld YAML for 1UBQ (76 residues),
prepared MSA, templates off, seed 0, one sample. AF-family used 10 recycles/200
steps; ESMFold2 used 3 loops/14 steps and explicit cached ESMC embeddings.
All five strict checkpoint runs produced finite outputs and 76 matched CAs.

| Model | CA RMSD to pre-format run (Å) | CA RMSD to deposited 1UBQ (Å) | pLDDT (0–100) |
|---|---:|---:|---:|
| AF3 | 0.0452 | 0.7005 | 91.251 |
| Protenix v1 | 0.0581 | 2.5917 | 94.071 |
| Protenix v2 | 0.2025 | 2.7138 | 93.329 |
| OpenDDE | 0.0495 | 1.0326 | 94.842 |
| ESMFold2 | 0.1016 | 0.6837 | 80.078 |

These are correctness observations, not performance measurements or a claim of
bitwise model determinism. The five full runs used the initial LMDB; final text-only
metadata changes were independently proven not to alter chemistry, conformers or
those target records. Primitive tests additionally cover masks, OPM bias/epsilon,
BF16 precision boundaries and gradients; reader tests cover fork/spawn and missing
fragmentation chemistry. MiniWorld's original CCDMol and fragmentation functions
were run against the new records, independently of the copied FoldForge helpers.

Self-contained validation evidence is archived under
`validation/miniworld-format-20260914/`; full raw run artifacts are retained in
the shared validation workspace. The final database hash is
`3636431697cf0875171b5e93350b95d77a6e02443d305b7bb49240308e9cd26b`.

The final file also includes the uncompressed size in every Zstd frame. BioMol
1.0.1 writes unsized streaming frames, whereas older MiniWorld environments call
`ZstdDecompressor.decompress()` and need that size. All 48,965 decompressed payloads
were verified identical during reframing. The historical BioMol reader body and
MiniWorld's original CCDMol class successfully read protein/nucleic/ligand/ion
records; 26 focused format tests passed after this compatibility change.

Actual installed-path validation additionally passed 32 tests and a full AF3
run (pLDDT 91.234). The ESMFold2 two-sample check exposed the pinned inference
TriMul front's B=1 limit in its confidence trunk. `FoldingTrunk` now evaluates
independent batch items in order at inference for the MiniWorld backend; sample
masks and the backend are preserved. A GPU regression compared a masked two-item
batch to two individual kernel calls and passed. The diffusion sampling axis is
retained, and every decoded sample is saved separately.

The actual ESMFold2 checkpoint also completed the two-sample configuration check
(1 loop, 2 sampling steps). Both CIFs, both pTM/ipTM values, and the common
`1ubq.prediction.pt` with two finite coordinate samples were saved. This short run
checks option wiring and sample preservation, not structural accuracy. P/O triangle
attention's multiple-bias path also uses the shared helper, with four exact-output
FP32/BF16 comparisons against frozen pre-migration functions.

The final installed working tree passed all 158 tests and `uv pip check` (175
compatible packages). Final CLI validation checked the selected 2 samples, 2 steps,
1 loop, both CIF files, finite common Prediction tensors, all 48,965 CCD keys,
and the sized-frame reader path. Existing code and the preceding CCD database
were backed up under `validation/miniworld-before-20260914/`.

## LMDB connection validation (2026-09-14)

Slurm job 1684557, allocated gpu03 RTX A6000: 181 regression tests passed,
including 23 LMDB tests. These cover single-file/directory LMDB, StructCooker
character queries, alphabet translation, exact insertion counts, shared species
aliases across chain order and depth limits, query/key/shape failures, blank
alignment columns, partial atom masks, CA pseudo-beta fallbacks, AF3 native
mmCIF feature alignment, and actual P/O dataset template/RNA-MSA activation.

The existing 1UBQ MiniWorld exports (16,177 MSA rows and 11 template hits) were
serialized back into the original BioMol LMDB schema for model validation. The
large original BioMol DB mount paths were unavailable on this host; no claim is
made that those whole databases were scanned or validated. Provenance and input
artifacts are in `validation/lmdb-inputs/`.

AF3, Protenix v1/v2 and OpenDDE completed strict-checkpoint inference with
MSA-LMDB and four selected templates (128 input MSA rows, one recycle, two
sampling steps). ESMFold2 completed protein MSA-LMDB inference both with that
smoke setup and its default three loops/fourteen steps (512 input rows, explicit
cached ESMC embeddings). All saved prediction tensors are finite. These are
connection checks, not model accuracy or timing benchmarks. Native BF16/FP32
norm, no autocast, compile off and graphs off are unchanged. Results and feature
coverage are in `validation/lmdb-results/`; the regression log is
`validation/lmdb-regression.log`.


## Qualified inference execution

Use `configs/inference/graph-bf16.yaml` for native BF16 and explicit denoiser
CUDA graphs. Set `execution.compile: true` to also compile that callable.
The trunk, host sampling and confidence remain outside capture; full-model graph
capture is rejected. Reports record actual graph/replay counts. The complete
numerical and complex-input qualification is in [the current execution record](QUALIFICATION-20260914.md).
