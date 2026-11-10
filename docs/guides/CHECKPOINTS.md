# Checkpoints: from a release file to a locked blob

For the download CLI, model-by-model setup and offline import, see
[Weight downloads](WEIGHTS.md): `foldforge weights download boltz2 opendde`.

FoldForge runs every family from one converted blob per model
(`<model>.bin.zst`, AF3's record format). That blob is the released layout; at load time FoldForge reads the **miniworld checkpoint format**
(`<model>.miniworld.pt`, [CHECKPOINT-FORMAT.md](CHECKPOINT-FORMAT.md)) converted once from it with `foldforge weights convert <model>`. The blobs are not in git (~17 GB) and
must not be redistributed casually: several releases (AF3 in particular) restrict
sharing their weights. This page is the path from each release's own file to the
exact blob `checkpoints.lock` pins.

## Current qualified conversion releases (2026-10-28)

`foldforge weights download` now supports IntelliFold2, Chai1 and Protenix1
through the reproducible `cpu-v1` profile, without JAX or Haiku. The original
`checkpoints.lock` is preserved; additional validated hashes are recorded in
`conversions.lock`. All three passed the existing release-reference criteria
on 5I28, 3PTB and 1A1K (three seeds, five samples each). The per-model conversion
results below are the historical 2026-09-26 results, not current blockers.
Protenix2 remains an authorized local-weight exception. AF3 is now available
directly from Google; the current official blob has its own separately pinned
hash and is installed without conversion.

## Layout and lookup

Blobs live under `$FOLDFORGE_HOME/checkpoints/<dir>/` (see `DEFAULT_FILES` in
`src/foldforge/models/checkpoints/__init__.py`), or under
`$FOLDFORGE_CHECKPOINT_DIR/<dir>/` to keep them on shared or larger storage.
`foldforge checkpoints verify` checks every blob against the lock; loading
refuses a blob whose size is not the locked one.

## Converting

The converter is sokrypton/alphafold3, branch `af3-any-model`, at
`301430b3cbf9122912663c67eb19628fd6feec3e` (public), driven by FoldForge's
`tools/convert_dense_checkpoint.py`, which runs in FoldForge's environment:

```bash
git clone -b af3-any-model https://github.com/sokrypton/alphafold3 ../refs/alphafold3-any-model
git -C ../refs/alphafold3-any-model checkout 301430b3cbf9122912663c67eb19628fd6feec3e
source tools/activate_env.sh
python tools/convert_dense_checkpoint.py --model <model> --checkpoint <release file> \
  --reference ../refs/alphafold3-any-model --out $FOLDFORGE_HOME/checkpoints/<dir>
foldforge checkpoints verify
```

## Per model

"Reproduces" is the result of converting the listed release file on
2026-09-26 and comparing with `checkpoints.lock`.

| `--model` | release file (sha256, first 12) | output under `$FOLDFORGE_HOME/checkpoints/` | reproduces |
|---|---|---|---|
| `esmfold2` | `biohub/ESMFold2` `model.safetensors` (HF), **revision `8fc3ff4710`** (sha256 `138fd435...`, 939,505,228 bytes) | `esmfold2/esmfold2.bin.zst` + `.lm.npz` | bit-identical |
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

**Pin the Hugging Face revisions.** On 2026-09-14 `biohub/ESMFold2` and `biohub/ESMC-6B` were both restructured (the single
`model.safetensors` was removed from `ESMFold2`'s HEAD; the key names changed in both). The converter and the vendored `ESMCModel`
read the old layout: ESM-C `esmc.embed.weight` / `esmc.transformer.blocks.N.attn...` (revision `45b0fa5d7f`, 2026-06-03; HEAD has
`esmc.embed_tokens.weight` / `esmc.layers.N.input_layernorm...`), ESMFold2 `z_init_1.weight` and friends (revision `8fc3ff4710`).
With HEAD's ESM-C nothing matches, the language model keeps uninitialised weights, its hidden states are all NaN and every ESMFold2
fold ends with `non-finite AF3 coordinates`; with HEAD's ESMFold2 the converter stops at `KeyError: 'z_init_1.weight'`.

ESM-C 6B (`biohub/ESMC-6B`, revision `45b0fa5d7f`) is read directly from its release directory
`$FOLDFORGE_HOME/checkpoints/esmc-6b/`; do not rely on the Hugging Face cache copy (see
`docs/bug-reports/esmfold2-hf-esmc-random-weights.md`).

The historical mismatches above are handled by the separately qualified CPU
conversion releases described at the top of this page. Use `foldforge weights
download MODEL` and `foldforge checkpoints verify`; do not edit lock entries to
make an unqualified conversion pass.
