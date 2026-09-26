# Checkpoints: from a release file to a locked blob

FoldForge runs every family from one converted blob per model
(`<model>.bin.zst`, AF3's record format). The blobs are not in git (~17 GB) and
must not be redistributed casually: several releases (AF3 in particular) restrict
sharing their weights. This page is the path from each release's own file to the
exact blob `checkpoints.lock` pins.

## Layout and lookup

Blobs live under `model_checkpoints/<dir>/` (see `DEFAULT_FILES` in
`src/foldforge/models/checkpoints/__init__.py`), or under
`$FOLDFORGE_CHECKPOINT_DIR/<dir>/` to keep them on shared or larger storage.
`foldforge checkpoints verify` checks every blob against the lock; loading
refuses a blob whose size is not the locked one.

## Converting

The converter is sokrypton/alphafold3, branch `af3-any-model`, at
`301430b3cbf9122912663c67eb19628fd6feec3e` (public), driven by FoldForge's
`scripts/convert_dense_checkpoint.py`, which runs in FoldForge's environment:

```bash
git clone -b af3-any-model https://github.com/sokrypton/alphafold3 ../refs/alphafold3-any-model
git -C ../refs/alphafold3-any-model checkout 301430b3cbf9122912663c67eb19628fd6feec3e
source scripts/activate_env.sh
python scripts/convert_dense_checkpoint.py --model <model> --checkpoint <release file> \
  --reference ../refs/alphafold3-any-model --out model_checkpoints/<dir>
foldforge checkpoints verify
```

## Per model

"Reproduces" is the result of converting the listed release file on
2026-09-26 and comparing with `checkpoints.lock`.

| `--model` | release file (sha256, first 12) | output under `model_checkpoints/` | reproduces |
|---|---|---|---|
| `esmfold2` | `biohub/ESMFold2` `model.safetensors` (HF) | `esmfold2/esmfold2.bin.zst` + `.lm.npz` | bit-identical |
| `esmfold2_fast` | `biohub/ESMFold2-Fast` `model.safetensors` (`60ca19f28981`) | `esmfold2-fast/esmfold2_fast.bin.zst` + `.lm.npz` | bit-identical |
| `opendde` | `opendde.pt` from the OpenDDE release bucket (`7b826620390a`) | `opendde/opendde.bin.zst` | bit-identical |
| `rosettafold3` | `rf3_foundry_01_24_latest_remapped.ckpt` (`364ef592fd80`) | `rosettafold3/rosettafold3.bin.zst` | bit-identical |
| `openfold3` | `of3-p2-155k.pt`, https://openfold.s3.amazonaws.com/staging/of3-p2-155k.pt (`3ae79a701f55`) | `openfold3/openfold3.bin.zst` | bit-identical |
| `openbind0` | `of3-ob-2025-06-30-174k.pt` (`bd43301c011d`) | `openfold3/openbind0.bin.zst` | bit-identical |
| `boltz2` | `boltz2_conf.ckpt`, `boltz-community/boltz-2` (HF) | `boltz2/boltz2.bin.zst` | bit-identical |
| `intellifold2` | `intellifold_v2.pt` (`8ee1c03344a9`) | `intellifold2/intellifold2.bin.zst` | weights identical; the current converter writes a content hash as the blob identifier where the locked blob has the name |
| `chai1` | chai-lab `models_v2/` (`trunk.pt`, `token_embedder.pt`, `diffusion_module.pt`, `confidence_head.pt`, `feature_embedding.pt`) plus `distogram_head.pt` from sokrypton/chai-lab `dgram` @ `849f990db241f4ea400be287ae12e5217bb70d26` (`6ba2e198e421`) placed in `models_v2/` | `chai1/chai1.bin.zst` | one record differs by 6e-6 (fp32 rounding in the chain/entity unfold) |
| `protenix2` | `protenix-v2.pt` (`8f931f9774a3`) | `protenix/protenix2.bin.zst` | two records differ by 2e-5 (fp32 rounding) |
| `protenix1` | `protenix_base_default_v1.0.0.pt` (`2b7d5a8b3049`) | `protenix/protenix1.bin.zst` | needs the reference's JAX environment (its guard imports haiku) |
| AF3 | DeepMind's `af3.bin.zst`, on request | `af3/af3.bin.zst` | the release file itself |

ESM-C 6B (`biohub/ESMC-6B`) is read directly from its release directory
`model_checkpoints/esmc-6b/`; do not rely on the Hugging Face cache copy (see
`docs/bug-reports/esmfold2-hf-esmc-random-weights.md`).

Where a conversion reproduces only to fp32 rounding, the freshly converted blob
folds the same but is refused by the lock's size check: copy the locked blob from
a machine that has it (verify with `foldforge checkpoints verify`), or, after
re-validating (`scripts/references/validate.sh`), update the lock.
