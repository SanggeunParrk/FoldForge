# The miniworld checkpoint format

Every model's graph is built from miniworld-engine classes (`PairformerBlock`, `TriangleMultiplication`, `TokenDiT*`, `AtomDiT*` ...) plus
FoldForge-own modules, so its parameters have the engine's names and layouts. Its checkpoint is that model's `state_dict`:

```text
torch.save({"format": "miniworld", "version": 1, "family": "alphafold3", "source": "af3.bin.zst",
            "state_dict": {name: fp32 tensor}}, "af3.miniworld.pt")
```

* Parameters are **fp32** (a released bf16 value is exact in fp32). Precision `master` runs them under bf16 autocast, `fp32` as they are; there
  are no native-bf16 parameters.
* The file is read with `torch.load(weights_only=True)`; the format tag, the version and the family are checked before anything is loaded.
* **Loading** is `model.load_state_dict(state_dict, strict=True)` (`models/checkpoints/miniworld.py::load_into`). The Fourier noise features
  (`diffusion_head.fourier_embeddings.{weight,bias}`, fixed constants or trained, depending on the family) are buffers registered at load time and
  are part of the file.
* One file serves both modes (`fast` / `exact` differ in forward conventions, not in parameters).

## Files

All twelve registered models have one (`miniworld.NATIVE_FILES`), next to their released blob in the model's checkpoint directory:

| model | file | released blob it is converted from |
|---|---|---|
| `af3` | `af3.miniworld.pt` | `af3.bin.zst` |
| `intellifold2` | `intellifold2.miniworld.pt` | `intellifold2.bin.zst` |
| `openfold3` | `openbind0.miniworld.pt` | `openbind0.bin.zst` |
| `openfold3-preview2` | `openfold3.miniworld.pt` | `openfold3.bin.zst` |
| `boltz2` | `boltz2.miniworld.pt` | `boltz2.bin.zst` |
| `rosettafold3` | `rosettafold3.miniworld.pt` | `rosettafold3.bin.zst` |
| `chai1` | `chai1.miniworld.pt` | `chai1.bin.zst` |
| `esmfold2` | `esmfold2.miniworld.pt` | `esmfold2.bin.zst` (+ `esmfold2.lm.npz`) |
| `esmfold2-fast` | `esmfold2_fast.miniworld.pt` | `esmfold2_fast.bin.zst` (+ `.lm.npz`) |
| `protenix1` | `protenix1.miniworld.pt` | `protenix1.bin.zst` |
| `protenix2` | `protenix2.miniworld.pt` | `protenix2.bin.zst` |
| `opendde` | `opendde.miniworld.pt` | `opendde.bin.zst` |

The language-model files (`esmfold2*.lm.npz`, the ESM-C directory) are not part of this format.

## Finding the file (`load_model` / `foldforge fold`)

`load_model("<model>")` (and `foldforge fold <model>` without `--checkpoint`) resolves `<model>.miniworld.pt` in the usual checkpoint locations
(`$FOLDFORGE_CHECKPOINT_DIR` / `$FOLDFORGE_HOME/checkpoints/<dir>/`, `registry.yaml`). **It never converts at run time.** A missing file raises
`FileNotFoundError` with the exact command, for example:

```text
no miniworld-format checkpoint (af3.miniworld.pt) for 'af3': the released blob is present; convert it once: foldforge weights convert af3
(same as: python -m foldforge.models.checkpoints.miniworld convert af3). The loader never converts at run time.
```

(`first fetch the released blob: foldforge weights download af3; then ...` when the blob is missing too.) An explicit `checkpoint=` that is a Haiku
blob is refused with the same command.

## Converting a released blob

A released Haiku blob is **converted once, offline**:

```bash
foldforge weights convert af3 boltz2            # <model>.bin.zst -> <model>.miniworld.pt next to it
foldforge weights convert --all                 # every model whose blob is installed (skips the others)
foldforge weights convert openfold3 --blob PATH --out PATH
python -m foldforge.models.checkpoints.miniworld convert af3     # the same, without the click CLI
```

Run it on a CPU allocation (it needs no GPU; the slurm jobs of the port took about half a minute per model). The converter is the only place the released
layout is dealt with, and it lives in one package, `src/foldforge/models/checkpoints/legacy/` (the dense released-layout modules, the Haiku record
codec `haiku.py`, their reference equations). `convert_haiku` (`miniworld.py`):

1. builds the dense released-layout graph of the family (`legacy.dense_model`, the runtime `AlphaFold3` class with the legacy blocks installed through
   `engine.graph.use_dense_blocks`) and reads the blob into it (`haiku.import_jax_weights_`);
2. converts every module the engine graph holds as an engine class with the rules of `models/checkpoints/legacy/released.py` (split `transition1` / the TriMul
   `projection` and `gate` / the shared DiT pair projection, broadcast the shared DiT pair LayerNorm, discard the softmax-invariant pair-LayerNorm
   offsets); FoldForge-own modules are copied through under their own names;
3. writes the file only if every released key was read, copied or a known discard (`released.DISCARDED`) and the result has exactly the keys and
   shapes of the engine graph. Anything else raises `ConversionError`.

Checked discards of particular families: the released `k_adaptive_layernorm` of the shared-KV atom blocks (Boltz-2, RoseTTAFold3, Chai-1) must equal
`q_adaptive_layernorm`; RoseTTAFold3's `q_projection.bias` must be all zero (the engine classes have neither). A release that violates one raises
instead of being converted. Protenix1 / 2 and OpenDDE release the DiT pair norm per block (nothing sliced) and keep the K / V AdaLN (`chained`);
OpenDDE's `structural_token_expander.*` is FoldForge-own.

Nothing at inference or training imports the legacy package (`tests/models/test_miniworld_resolution.py` checks that `foldforge.api`, `loading`,
`execution`, the architecture and `miniworld` import none of it). Only the converter CLI does, lazily, inside `convert_haiku`.

How to produce the released blobs themselves (`<model>.bin.zst`) from each release's own file is in [CHECKPOINTS.md](CHECKPOINTS.md) and
[WEIGHTS.md](WEIGHTS.md) (`foldforge weights download`); this page is the second step, blob to miniworld file.
