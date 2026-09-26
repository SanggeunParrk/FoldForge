# Reference outputs from the released implementations

Every structure under `outputs/<target>/<family>/` was produced by that
family's **own released code and weights**, not by FoldForge. They are the
ground truth FoldForge is judged against: `foldforge validate` folds the same
inputs under the same conditions and reports, per family and target, whether
FoldForge is inside the release's own seed-to-seed variation.

## Layout

```
outputs/
  inputs/<target>/            the same target in every release's input format
    foldforge.yaml  boltz.yaml  intellifold.yaml  protenix.json
    of3.json  rf3.json  chai.fasta  *.fasta
  <target>/<family>/seed<k>/
    sample<i>.cif.gz          the release's i-th diffusion sample for seed k
    manifest.json             source file names and each sample's mean pLDDT
```

## Targets

| target | contents | what it exercises |
|---|---|---|
| `5i28` | one protein chain (127 residues) | the trunk and sampler on a low-confidence, no-MSA fold |
| `3ptb` | trypsin + Ca2+ + benzamidine | ligand bond geometry, ion placement, atomised tokens |
| `1a1k` | two DNA strands + zinc finger + 3 Zn2+ | nucleic acids, identical ions, multi-chain relations |

## Conditions (identical for every family)

No MSA, no template, four trunk passes (Chai-1: its default three), 200
diffusion steps, five samples per seed, seeds 0, 1 and 2. Protenix and
IntelliFold run at fp32. Three seeds matter: Protenix flips an MC-dropout coin
per seed, so a single seed cannot represent it.

## Families and the released code used

| family | released implementation |
|---|---|
| `boltz2` | `boltz predict --model boltz2` |
| `chai1` | `chai_lab.chai1.run_inference` (with `rdkit==2024.9.5`) |
| `protenix1` | `protenix pred -n protenix_base_default_v1.0.0` |
| `protenix2` | `protenix pred -n protenix-v2` |
| `openfold3` | OpenFold3 v0.5.0 (OpenBind) `run_openfold.py predict` |
| `openfold3-preview2` | OpenFold3 preview-2 `run_openfold.py predict` |
| `rosettafold3` | RosettaCommons foundry `rf3 fold` |
| `intellifold2` | IntelliFold `runner/intellifold_inference.py --model v2` |
| `opendde` | OpenDDE `runner/batch_inference.py pred -n opendde_v1` |
| `esmfold2` | `transformers.ESMFold2Model` + `esm.models.esmfold2` |

The exact commands are `scripts/references/run_release.sh`. To regenerate
everything: `sbatch scripts/references/generate.sbatch` (each task runs one
family x target x seed and normalises it with `scripts/references/collect.py`).

The released environments are too large for the repository. The runner reads
`FOLDFORGE_RELEASE_ROOT` (the directory holding `of3-deps`, `rf3-deps`,
`if2-deps`, `chai-rdkit-deps`, `protenix-root` and `boltz-cache`, each an
isolated `pip install --target` of that release's pinned dependencies) and
`FOLDFORGE_RELEASE_ENVS` (the Boltz, Protenix and Chai-lab environments).

## Release quirks recorded here

- IntelliFold writes the same pLDDT for every sample of a seed.
- OpenDDE's release cannot featurise any input with a ligand or ion (its
  template featuriser raises `map_to_standard` even with templates off), so it
  has `5i28` alone.
- ESMFold2's release fails 1A1K seed 0 inside its own rigid align (the SVD does
  not converge); that seed is absent.
- Chai-lab with RDKit >= 2025 fails to tokenise every ligand and folds the
  protein alone without an error; pin `rdkit==2024.9.5`.
- Protenix reads identical ions as separate entities unless they are one entry
  with a count (`"count": 3`), unlike every other release.
- Protenix v1 and v2 apply MC dropout to the pair recycle on ~40% of seeds.

## Validating FoldForge

```
scripts/references/validate.sh fast
foldforge validate runs/validate-fast --report docs/validation/fast.md
```

The latest reports are in `docs/validation/`.
