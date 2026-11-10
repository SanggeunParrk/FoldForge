# Backend measurements on the pre-2.5.0 engine (historical)

The policy of this page (backends `miniworld` / `cuequivariance` / `anthropic` / `pytorch`, the `engine_skip` parts, the `variant_serves` seam, the engine settings) was replaced by the 2.5.0 backend policy in [guides/BACKENDS.md](../guides/BACKENDS.md). The tables below are measurements on engine 2.3.9 to 2.3.11 with the dense twins attached; they are not comparable to a 2.5.0 run and are kept as a record only.

---

# Backends

A model is loaded with one `backend`, the source of the kernels of every op in it:

| `backend` | kernels | works on |
|---|---|---|
| `miniworld` (default) | the miniworld-engine: hand-written CUDA where the engine has it for the card, the engine's own other legs elsewhere | any CUDA card; tuned for H100 / B200 |
| `cuequivariance` | NVIDIA cuEquivariance 0.12 (`triangle_multiplicative_update`, `triangle_attention`, `attention_pair_bias`) | any card the library supports |
| `anthropic` | the kernels of Anthropic's uplifting-biomolecular-modeling release (the TriMul payload, the `opt_core` cells) | where its payload builds and loads (sm_90 here) |
| `pytorch` | the dense FoldForge modules: the reference every other backend is compared with | anywhere |

```yaml
backend: miniworld          # or: cuequivariance | anthropic | pytorch
engine:
  ops: {trimul: anthropic}  # optional: single ops on another backend than `backend`
  unsupported: fallback     # fallback (default) | error
```

Nothing else changes between backends: the same model classes, the same weights, the same call sites. The choice is
made once, when the model is loaded, in `modules/engine/backends.py` (`BackendPlan`), and read by the one install path
(`engine.install.install_engine`, `engine.modules.install_engine_modules`) and the routing functions
(`engine.routing`). With no override every op is `miniworld` and the model runs exactly as it did before backends
existed (checked below: same pLDDT, same times, bit-identical coordinates).

## Ops

`engine.ops` takes these names (the keys of `backends.SERVES`):

| op | what it is |
|---|---|
| `trimul` | triangle multiplication (Pairformer, MSA module, template stacks) |
| `triangle_attention` | triangle attention, starting and ending node |
| `opm` | MSA outer product mean |
| `msa_pwa` | MSA pair weighted averaging |
| `single_attention` | the Pairformer's single attention with pair bias |
| `transition` | the SwiGLU transitions of the Pairformer and the MSA module |
| `layernorm` | the LayerNorms of the model (FP32 parameters and reduction) |
| `conditioning` | AdaLN, conditioned transitions and the diffusion head's conditioning transitions |
| `dit_blocks` | the diffusion transformer's token blocks (per-block and the Boltz-2 / RoseTTAFold3 / Chai-1 stacks) |
| `atom_blocks` | the atom-window transformer blocks of the diffusion head |
| `sampler_ops` | the engine's fused rigid-motion glue of the sampler step and the atom-to-token pooling |

## What each backend serves

`served`: the backend's kernel runs the op. `ref`: the op has no kernel on that backend and runs the PyTorch reference
(`unsupported: fallback`: one log line at load; `unsupported: error`: the load raises). A served op can still refuse a
call its kernel does not take; the conditions are the second table.

| op | miniworld | cuequivariance | anthropic | pytorch |
|---|---|---|---|---|
| `trimul` | served | served | served | served |
| `triangle_attention` | served | served | served | served |
| `opm` | served | ref | served | served |
| `msa_pwa` | served | ref | served | served |
| `single_attention` | served | served | ref | served |
| `transition` | served | ref | served | served |
| `layernorm` | served | ref | ref | served |
| `conditioning` | served | ref | ref | served |
| `dit_blocks` | served | ref | ref | served |
| `atom_blocks` | served | ref | ref | served |
| `sampler_ops` | served | ref | ref | served (the torch ops) |

Conditions of a served op (a call outside them runs the PyTorch reference after one log line per op and reason, or
raises with `unsupported: error`; a module the machine or its widths exclude for good is decided at load):

| backend, op | the kernel takes |
|---|---|
| miniworld, any | whatever the engine's dispatch takes on the card; what it cannot serve, the engine runs as the PyTorch reference (on H100 / B200 deliberately, see the engine's `docs/gpus/h100/dispatch.md`). Dense FP32 models install the diffusion transformers only |
| miniworld, `dit_blocks`, `atom_blocks` | on H100 / B200 the engine's CUDA blocks and variant stacks (Boltz-2, RoseTTAFold3, Chai-1 inference); on another card the engine's portable blocks where the engine holds a record for that card: the AF3-style and ESMFold2 atom blocks and the three families' token and atom stacks (head width up to 64, up to 256 keys, atom stacks with a token count that is a multiple of 32). A card without a record, or a call outside these limits, keeps FoldForge's dense blocks (the engine says so once) |
| cuequivariance, `trimul` | bf16 CUDA, pair width a multiple of 32 and equal to the contraction width, no length division (RoseTTAFold3's) |
| cuequivariance, `triangle_attention` | bf16 CUDA (any head count the library takes) |
| cuequivariance, `single_attention` | bf16 CUDA, AF3's form: no single conditioning, k/q norm or gate bias (so not Chai-1's) |
| anthropic, `trimul` | `TRIMUL_NATIVE_BUILD_DIR` names a payload; bf16, no grad, one square pair plane per call, sm_90 (sm_100 with the rebuilt member), a payload unit for (pair width, hidden width): 128/128 and 128/256; not RoseTTAFold3's length-divided contraction (it scales a weight per call, the payload packs weights once per version) |
| anthropic, `triangle_attention` | the release's row `k2b` through `integrations.anthropic`; bf16, no grad; the release refuses by name what it has no cell for |
| anthropic, `opm` | `OPT_CORE_DIR`; 32-wide outer channels into a 128-wide pair, tokens a multiple of 16, bf16, sm_90, and the bias added after the division by the pair count (the OpenFold-lineage convention: AF3's adds it before and is refused) |
| anthropic, `msa_pwa` | `OPT_CORE_DIR`; the cell is (64-wide MSA, 128-wide pair, 8 heads of width 32): AF3's heads are 8 wide and are refused |
| anthropic, `transition` | `MINIWORLD_ANTHROPIC_ROOT` (the release's `opt_core`); bf16, no grad, row `v2` |

The Anthropic payloads are located by environment variables, which are locations and not behaviour (CONFIG.md):
`MINIWORLD_ANTHROPIC_ROOT` (the release checkout), `OPT_CORE_DIR` (its `common/opt_core`) and
`TRIMUL_NATIVE_BUILD_DIR` (a TriMul payload built for the local CUDA stack). The engine's own `miniworld` modules also
use a named payload where it can serve; so a `miniworld` run is a run of the miniworld kernels only when these are unset
(the loader logs a line when they are set next to a `miniworld` op). `tools/compare_backends.py` unsets them for every
backend but `anthropic`.

Anthropic's kits also carry pair-bias attention (`apb`, atom attention), the diffusion transformer rows (`dit_exact`,
`ditfast`) and a LayerNorm (`fastln`). They are not connected: FoldForge's single attention and diffusion blocks are
the engine's fused blocks and the release's pieces would replace parts of them, not whole ops; its LayerNorm refuses
bf16 weights. Those ops are `ref` on `anthropic`.

## Unsupported combinations

`engine.unsupported: fallback` (default) runs the PyTorch reference for an op the backend has no kernel for, and says so
in one log line (logger `foldforge.backend`) at load, listing the ops and the reason of each. The same list is in the
run's JSON, `checkpoint.backends`: `ops` (the backend that ran each op), `fallbacks` (op: reason), `unsupported`.
A call a served kernel refuses (shape, dtype, a width without a unit) logs one line per (op, reason) the first time.
`engine.unsupported: error` raises `BackendUnsupportedError` instead, at load for the ops and at the call for the
refusals. It is meant for a per-op request (`backend: pytorch`, `engine.ops: {trimul: anthropic}`), because no backend
but `miniworld` and `pytorch` serves every op.

Nothing is substituted silently, and nothing is substituted by another backend's kernel: the fallback is always the
PyTorch reference.

## Other cards

The checks that decide whether a kernel runs ask the device (`torch.cuda.get_device_capability`, in the engine's
dispatch and in the Anthropic payloads' own `refusal` predicates) or the payload (a unit for the widths); FoldForge
holds no card list. On a card that is not an H100 or B200:

- `miniworld` runs what the engine runs there (the engine owns that): its hand-written CUDA where it serves the call,
  else its portable kernels on a card it holds a record for (`docs/dispatch.md` of the engine), else one
  `UnrecordedCardWarning` and the PyTorch reference. The diffusion transformers follow the same rule through the one
  seam, `modules.engine.variant_serves` (the stacks of Boltz-2, RoseTTAFold3, Chai-1 and ESMFold2's token stack: the H100
  question first, then the engine's portable one) and the engine's per-block atom/token blocks, which route themselves.
  Where the kernels do not serve, the engine stack runs its own PyTorch statements (`serving() == "pytorch"` counts as served);
- `cuequivariance` and `pytorch` do not depend on the card;
- `anthropic` serves what its payload serves on that card and refuses the rest with the reason from the payload (sm_90a
  units for TriMul and the MSA cells; an sm_80 member exists for TriMul): everything else is the PyTorch reference.

Only an H100 was available. Every cell of the tables above was run there; the other cards are untested (A100, A6000,
L40 and B200: nothing of this was run on them), and so is a `miniworld` run of these four ops off an H100. The engine's
per-card records (`docs/dispatch.md` of the engine) decide what `miniworld` runs on a card; at engine 2.3.10 only the
token stacks of the three variant families have a record on A100 / A5000 / A6000, the atom blocks and atom stacks warn once
and run the dense path there until the engine's `run_all` has been run on the card.

## Comparing backends

```
python tools/compare_backends.py af3 --spec input.yaml --config runtime.yaml --out compare/af3
```

runs the same model, input and config once per backend, each in its own process, and prints one row per backend: warm
median time, mean pLDDT, and the difference of the predicted coordinates from the `pytorch` run (max and rms of the
per-atom displacement, raw and after the best rigid fit). A refusal, an out-of-memory or a crash is a row too. The rows
are appended to `<out>/rows.jsonl`. It needs a Slurm GPU allocation like every other run.
`tools/benchmark_end_to_end.py --backends ...` takes the same four names for the end-to-end timing of whole processes.

The measurements below come from the forward-only harness (`ffbench/ff_fwd.py`, scratch, outside the repository), which
now records the `backend`, the backend of every op (`ops`), the fallbacks, the peak allocated memory and the
coordinate file next to the earlier fields.

## Results (H100 80GB, 2026-10-24)

Forward-only harness, the same as `final_af3.jsonl` / `final_esm.jsonl`: `compile_graph_trunk_bucket` (compile and CUDA
graphs on the denoiser, graphs on the trunk and heads, bucketing), real MSA (and templates where the family has them)
for 5I28 replicated n times, `mode: exact`, 5 diffusion samples, 200 steps, the released recycles per family; warm median
of 3 forwards (n <= 2), 2 (n <= 8) or 1. One backend per process, one process per GPU, on the engine tree FoldForge main
runs on (the PyPI wheel `miniworld-engine==2.3.9`; the numbers below were measured on the unreleased tree 2.3.8 + `integ/all-20261023`, whose content 2.3.9 contains, and re-checked on the wheel, see "Re-check on the 2.3.9 wheel"). FoldForge `6d5b7b6`, except RoseTTAFold3 on `anthropic`
(`91fb117`: its length-divided TriMul is a stated reference there, see below). `anthropic` has the payloads of
`~/ext/uplifting-biomolecular-modeling` and `trimul_native_cu129`.

Warm median forward time, seconds:

| model | n | miniworld | cuequivariance | anthropic | pytorch |
|---|---|---|---|---|---|
| af3 | 1 | 0.51 | 0.89 | 0.86 | 0.99 |
| openfold3 | 1 | 0.51 | 0.89 | 0.87 | 1.00 |
| protenix1 | 1 | 0.43 | 0.93 | 0.90 | 1.04 |
| boltz2 | 1 | 0.45 | 1.01 | 0.97 | 1.16 |
| rosettafold3 | 1 | 0.47 | 1.04 | 1.01 | 1.07 |
| esmfold2 | 1 | 0.25 | 0.79 | 0.65 | 0.78 |
| chai1 | 1 | 0.46 | 0.79 | 0.69 | 0.82 |
| opendde | 1 | 0.61 | 1.36 | 1.37 | 1.59 |
| af3 | 4 | 1.38 | 4.06 | 3.42 | 8.03 |
| openfold3 | 4 | 1.38 | 4.06 | 3.43 | 8.05 |
| protenix1 | 4 | 1.28 | 4.43 | 3.60 | 9.04 |
| boltz2 | 4 | 1.60 | 5.03 | 4.09 | 11.26 |
| rosettafold3 | 4 | 1.41 | 5.54 | 4.98 | 8.00 |
| esmfold2 | 4 | 0.93 | 3.24 | 2.21 | 4.51 |
| chai1 | 4 | 2.22 | 4.80 | 4.04 | 8.05 |
| opendde | 4 | 3.09 | 10.90 | 9.86 | 21.02 |

Mean pLDDT:

| model | n | miniworld | cuequivariance | anthropic | pytorch |
|---|---|---|---|---|---|
| af3 | 1 | 96.20 | 96.15 | 96.15 | 96.19 |
| openfold3 | 1 | 95.13 | 95.09 | 95.08 | 95.07 |
| protenix1 | 1 | 96.14 | 96.14 | 96.10 | 96.13 |
| boltz2 | 1 | 97.11 | 97.07 | 97.09 | 97.05 |
| rosettafold3 | 1 | 86.65 | 86.64 | 86.63 | 86.62 |
| esmfold2 | 1 | 92.36 | 92.36 | 92.42 | 92.36 |
| chai1 | 1 | 95.54 | 95.53 | 95.54 | 95.54 |
| opendde | 1 | 94.86 | 94.86 | 94.84 | 94.87 |
| af3 | 4 | 89.19 | 89.11 | 88.85 | 89.08 |
| openfold3 | 4 | 83.89 | 83.73 | 83.80 | 83.73 |
| protenix1 | 4 | 83.30 | 83.25 | 83.35 | 83.40 |
| boltz2 | 4 | 91.21 | 91.19 | 91.15 | 91.02 |
| rosettafold3 | 4 | 83.88 | 83.84 | 83.87 | 83.88 |
| esmfold2 | 4 | 91.11 | 89.67 | 89.80 | 89.81 |
| chai1 | 4 | 89.45 | 89.27 | 89.06 | 89.40 |
| opendde | 4 | 89.71 | 89.65 | 89.56 | 89.64 |

Predicted coordinates against the `pytorch` backend, rms / max displacement of the atoms as they come out, and rms after
the best rigid fit, Angstrom (the five samples together):

| model | n | miniworld | cuequivariance | anthropic |
|---|---|---|---|---|
| af3 | 1 | 0.72 / 4.08 / 0.19 | 0.31 / 2.54 / 0.13 | 0.37 / 2.37 / 0.11 |
| openfold3 | 1 | 0.87 / 3.94 / 0.13 | 0.52 / 2.48 / 0.08 | 0.30 / 2.87 / 0.08 |
| protenix1 | 1 | 0.65 / 3.43 / 0.14 | 0.37 / 2.02 / 0.09 | 0.27 / 2.62 / 0.10 |
| boltz2 | 1 | 1.08 / 2.88 / 0.14 | 0.77 / 2.44 / 0.12 | 0.47 / 1.61 / 0.08 |
| rosettafold3 | 1 | 0.56 / 2.17 / 0.07 | 0.32 / 3.16 / 0.10 | 0.46 / 2.97 / 0.17 |
| esmfold2 | 1 | 4.71 / 20.66 / 0.25 | 0.00 / 0.00 / 0.00 | 1.32 / 5.06 / 0.14 |
| chai1 | 1 | 0.67 / 3.58 / 0.15 | 0.37 / 3.55 / 0.12 | 0.45 / 2.95 / 0.15 |
| opendde | 1 | 1.31 / 5.28 / 0.16 | 0.28 / 2.39 / 0.12 | 0.51 / 2.19 / 0.11 |
| af3 | 4 | 1.47 / 6.64 / 0.59 | 0.58 / 3.41 / 0.29 | 1.16 / 4.68 / 0.41 |
| openfold3 | 4 | 16.16 / 63.12 / 15.57 | 2.46 / 14.37 / 2.21 | 9.51 / 34.97 / 9.43 |
| protenix1 | 4 | 8.92 / 43.52 / 8.14 | 17.55 / 78.40 / 5.45 | 2.53 / 12.30 / 2.48 |
| boltz2 | 4 | 15.66 / 70.37 / 10.01 | 8.26 / 45.87 / 7.24 | 14.10 / 51.37 / 13.72 |
| rosettafold3 | 4 | 14.76 / 79.55 / 14.14 | 8.75 / 44.47 / 7.69 | 15.55 / 80.41 / 12.05 |
| esmfold2 | 4 | 1.63 / 7.40 / 1.40 | 0.62 / 4.08 / 0.43 | 0.65 / 4.05 / 0.30 |
| chai1 | 4 | 0.27 / 2.89 / 0.22 | 0.55 / 3.20 / 0.14 | 1.19 / 4.45 / 0.27 |
| opendde | 4 | 1.26 / 4.94 / 0.86 | 0.44 / 3.15 / 0.20 | 0.81 / 3.72 / 0.14 |

At n = 1 the fits agree to 0.07-0.25 A and pLDDT to 0.1 on every backend and family. At n = 4 the chains are placed
differently by every run, so the displacement says nothing about the kernels there (a different arrangement of the same
chains: up to 80 A); pLDDT is the figure to compare. The ESMFold2 n = 1 rms of 4.7 A is a different global frame
(its fit is 0.25 A), and its `cuequivariance` column is identical to `pytorch` because that family contains none of the ops
cuequivariance serves. RoseTTAFold3 is not deterministic from run to run even on the
tree before backends existed (two runs of it differ by up to 2.5 A), so its columns carry that noise.

ESMFold2, n = 16 (the largest size of the recorded series; FoldForge `9bd2e11`, which has the same source as `6d5b7b6`;
no row failed):

| backend | warm median s | pLDDT | peak allocated GiB |
|---|---|---|---|
| miniworld | 10.53 | 73.83 | 33.0 |
| anthropic | 23.24 | 73.02 | 34.9 |
| cuequivariance | 39.33 | 72.99 | 58.6 |
| pytorch | 80.94 | 73.01 | 55.8 |

`miniworld` against the recorded final numbers (`~/foldforge-scratch/ffbench/final_af3.jsonl`, `final_esm.jsonl`):

| | recorded | now (`backend: miniworld`) |
|---|---|---|
| AF3 n=4 warm / pLDDT | 1.39 s / 89.19 | 1.38 s / 89.19 |
| ESMFold2 n = 1, 2, 4, 8, 12, 16 warm, seconds | 0.25, 0.42, 0.93, 2.81, 6.11, 10.68 | 0.25, 0.41, 0.93, 2.77, 6.03, 10.53 |

Unchanged by the backend work: the tree before it (`2af6213`) and the tree after it were run back to back on one GPU with
the same caches and `backend: miniworld`. All nine arrays of the output (coordinates, pLDDT, PAE, PDE, ...) are bit-identical
for AF3 at n = 4 and for AF3, Boltz-2, Chai-1, ESMFold2, OpenDDE, OpenFold3 and Protenix at n = 1; RoseTTAFold3 is not
bit-reproducible by itself (above), and its pLDDT is the same (86.65). The tables use a copy of the kernel cache of the recorded runs (a fresh cache gives pLDDT 89.17 for the same AF3 fold: the engine's tuned configurations are part of the numbers).

Each op through the seam, milliseconds per call at L = 256 / 384 tokens (bf16, AF3 widths, MSA of 512 rows; the
Anthropic pair weighted averaging cell needs 32-wide heads, so that row uses them on every backend; its OPM epilogue
needs the OpenFold-lineage bias convention, so there are two OPM rows):

| op | miniworld | cuequivariance | anthropic | pytorch |
|---|---|---|---|---|
| TriMul (outgoing) | 0.51 / 0.52 | 0.57 / 0.57 | 0.34 / 0.34 | 0.76 / 1.69 |
| triangle attention (starting) | 0.15 / 0.28 | 0.51 / 1.21 | 0.45 / 1.02 | 1.16 / 3.34 |
| OPM, OpenFold convention | 0.90 / 1.88 | ref | 0.29 / 0.43 | 0.75 / 1.42 |
| OPM, AF3 convention | 0.53 / 1.07 | ref | refused | 0.75 / 1.42 |
| pair weighted averaging | 0.59 / 0.85 | ref | 0.55 / 0.56 | 0.95 / 1.49 |
| Transition (pair) | 0.08 / 0.11 | ref | 0.11 / 0.16 | 0.46 / 0.99 |
| single attention | 0.14 / 0.14 | 0.33 / 0.35 | ref | 0.37 / 0.45 |

What a backend refused or could not express, in the folds above (each is one log line; the op then ran the PyTorch
reference):

| family | `anthropic` | `cuequivariance` |
|---|---|---|
| AF3, OpenFold3, Protenix | pair weighted averaging (heads 8 wide), OPM (AF3 convention) | Protenix: a TriMul whose contraction is wider than the pair |
| Boltz-2 | none | none |
| RoseTTAFold3 | TriMul (length-divided contraction) | TriMul (length-divided contraction) |
| ESMFold2 | pair weighted averaging (128/256 wide), OPM | none (no triangle op in the family) |
| Chai-1 | pair weighted averaging, OPM, Transition (256 x 512 and 384 x 768 cells) | single attention (gate bias) |
| OpenDDE | pair weighted averaging, OPM, Transition (384 x 768) | none |

Out of memory: nothing on 80 GB at these sizes, once RoseTTAFold3's TriMul was made a stated reference on `anthropic`
(before, that combination ran out of memory capturing the trunk graph at n = 4: the payload repacked a weight on every
call).

Not measured: any card but the H100 (see Other cards); n = 8 and 12 for other backends than `miniworld` on ESMFold2;
fp32 and `af3_default` precision (the engine kernels of the pair stack are bf16; `precision: fp32` installs the diffusion
transformers only on `miniworld`, and the other backends run the reference).

## Re-check on the 2.3.9 wheel

FoldForge pins `miniworld-engine==2.3.9` from PyPI (2.3.9 contains the engine work the tables above were measured on:
`serves_extra_bias`, `sampler_ops`, `dit_variants`, the sm90 triangle attention fp16 core, `local_dit`, PWA up to N <= 2048).
Nothing is put on `PYTHONPATH`: the engine is the installed wheel, FoldForge the editable checkout. Test suite
(`FOLDFORGE_TEST_ALLOW_SIZE_MISMATCH=1`, empty kernel cache, `CUTLASS_PATH=/nonexistent`): 604 passed, 1 failed (the
missing checkpoints of `test_every_registered_family_resolves_its_checkpoint`); the backend tests with the Anthropic
payload variables set: 52 passed. Warm median forward (empty caches, `compile_graph_trunk_bucket`, real MSA), seconds,
and mean pLDDT in brackets:

| model | n | miniworld | cuequivariance | anthropic | pytorch |
|---|---|---|---|---|---|
| af3 | 1 | 0.51 (96.22) | 0.89 (96.16) | 0.86 (96.16) | 0.99 (96.19) |
| af3 | 4 | 1.38 (89.17) | 4.08 (89.11) | 3.42 (88.85) | 8.08 (89.08) |
| esmfold2 | 1 | 0.25 (92.36) | 0.79 (92.36) | 0.65 (92.46) | 0.78 (92.36) |

All within 1% of the tables above. AF3 n=4 on `miniworld` gives pLDDT 89.17 with an empty cache (89.19 with the recorded
cache, see above). No engine kernel was compiled in the four-backend folds. The pytest run built one engine extension,
`miniworld_opm_epilogue` (the Anthropic-backend OPM epilogue of `integrations/anthropic_msa.py`), which the wheel does not
ship prebuilt.

## Diffusion transformers off H100 / B200 (engine 2.3.10, H100 measurement of the other cards' path)

On a card that is not H100 / B200, `miniworld` now sends the diffusion transformers of every family to the engine
where the engine can serve them (see "Other cards"): `variant_serves` is the one seam, the engine routes the call. Before,
Boltz-2, RoseTTAFold3 and Chai-1 (token and atom) and ESMFold2's token stack ran FoldForge's dense blocks there, and the
AF3-style and ESMFold2 atom blocks the engine's reference core.

No other card was available, so this was measured on the **H100 with the engine's internal A/B switch**
(`miniworld_engine.settings.configure(force_portable_kernels=True)`, set by the scratch harness `ff_stage.py` and by the tests only: FoldForge
has no key for it), which sends the engine to the kernels a recorded card would run, for the whole model. Harness
`ffbench/ff_fwd.py`, `compile_graph_trunk_bucket`, real MSA and templates, `mode: exact`, 5 samples, 200 steps, warm median
of 3 (n = 1) or 2 (n = 4) forwards, wheel `miniworld-engine==2.3.10` and no overlay. Columns: `default` = `miniworld` on the
H100 without the switch (CUDA paths); `portable` = switch on; `dense` = switch on and the diffusion blocks left dense
(`engine.skip: [engine_atom_blocks, engine_dit_variant_blocks, engine_atom_variant_blocks]`: what a recorded card ran before
the seam knew the engine's portable path; for AF3 only the atom blocks, its token blocks are the engine's in both).

| model | n | warm s: default / portable / dense | portable vs dense | pLDDT default / portable / dense | rigid-fit rms A, portable-dense |
|---|---|---|---|---|---|
| boltz2 | 1 | 0.45 / 0.77 / 1.03 | 1.34x | 97.10 / 97.10 / 97.09 | 0.08 |
| boltz2 | 4 | 1.60 / 2.47 / 3.42 | 1.38x | 91.41 / 90.96 / 91.49 | (chains placed differently) |
| rosettafold3 | 1 | 0.46 / 0.72 / 1.09 | 1.51x | 86.65 / 86.64 / 86.63 | 0.05 |
| rosettafold3 | 4 | 1.42 / 2.12 / 3.20 | 1.51x | 83.88 / 83.88 / 83.86 | (chains placed differently) |
| chai1 | 1 | 0.46 / 0.59 / 0.70 | 1.19x | 95.53 / 95.51 / 95.51 | 0.09 |
| chai1 | 4 | 2.23 / 2.49 / 3.06 | 1.23x | 89.39 / 89.06 / 89.62 | 0.29 |
| af3 | 1 | 0.51 / 0.91 / 0.95 | 1.04x | 96.22 / 96.19 / 96.19 | 0.08 |
| af3 | 4 | 1.38 / 2.80 / 3.00 | 1.07x | 89.17 / 89.20 / 89.14 | 0.29 |
| esmfold2 | 1 | 0.25 / 0.47 / 0.62 | 1.32x | 92.36 / 92.28 / 92.37 | 0.09 |
| esmfold2 | 4 | 0.93 / 1.60 / 2.13 | 1.33x | 91.11 / 91.14 / 91.15 | 0.20 |

The `portable` column is not a speed claim for the card: it is the H100 running the engine's portable kernels, which on H100
are slower than the CUDA paths by design (1.1-2.0x here); it is the gain of a card without CUDA paths over the dense blocks. The fits
at n = 1 and at n = 4 for Chai-1, AF3 and ESMFold2 agree to 0.05-0.3 A, the pLDDT to 0.1 (n = 1) and 0.6 (n = 4); at n = 4 the
chains of Boltz-2 and RoseTTAFold3 are placed differently by every run (as in the tables above), so only the pLDDT compares.
The same switch gave: the AF3-style (AlphaFold 3, Protenix) and ESMFold2 atom blocks and the three variant stacks' token and
atom forwards equal to the dense blocks and to the FP32 truth within the existing bf16 tolerances (`tests/kernels/test_engine_modules.py`, the
`portable` cases and `test_variant_*_portable`); the engine's gaps A (pair weighted averaging with 8-wide values, outer product mean
groups / projection bias, triangle attention gate / output bias / untransposed / dual) are inside the engine modules the
families already call and ran in these forced folds without a fallback line.

Without the switch nothing changed on the H100: `default` above gives AF3 n = 4 1.38 s / pLDDT 89.17, ESMFold2 n = 1 0.25 s,
Boltz-2 n = 1 0.45 s, RoseTTAFold3 0.46 s, Chai-1 0.46 s (the tables above: 1.38 s / 89.17-89.19, 0.25, 0.45, 0.47, 0.46).
Test suite on the wheel (`FOLDFORGE_TEST_ALLOW_SIZE_MISMATCH=1`): 641 passed, 1 failed (the missing checkpoints of
`test_every_registered_family_resolves_its_checkpoint`), 14 skipped (Anthropic payloads not named); 37 of them are new.

## Re-check on the 2.3.11 wheel

FoldForge pins `miniworld-engine==2.3.11` from PyPI (the La-Proteina kernels and the window-attention backward bands on top of
2.3.10). `pixi.lock` resolves the cp312 wheel with sha256 `834835eb...eec7` (the file on PyPI); the `default` and `cu13`
environments import 2.3.11 from FoldForge's own `.pixi` with an empty `PYTHONPATH`. Test suite
(`FOLDFORGE_TEST_ALLOW_SIZE_MISMATCH=1`, empty kernel cache, `CUTLASS_PATH=/nonexistent`): 641 passed, 1 failed (the missing
checkpoints of `test_every_registered_family_resolves_its_checkpoint`), 14 skipped; the backend tests with the Anthropic payload
variables set: 52 passed. No engine extension was compiled in either run (empty `TORCH_EXTENSIONS_DIR`; the only `.so` is
Triton's `cuda_utils` stub), including `miniworld_opm_epilogue`, which 2.3.9 still built in the pytest run. Warm median forward
(n = 1, empty caches, `compile_graph_trunk_bucket`, real MSA), seconds, mean pLDDT in brackets:

| model | n | miniworld | cuequivariance | anthropic | pytorch |
|---|---|---|---|---|---|
| af3 | 1 | 0.51 (96.22) | 0.89 (96.15) | 0.86 (96.17) | 1.00 (96.17) |
| esmfold2 | 1 | 0.25 (92.36) | 0.78 (92.36) | 0.65 (92.42) | 0.78 (92.36) |

Identical to the 2.3.9 / 2.3.10 numbers above within noise (miniworld AF3 0.51 s, ESMFold2 0.25 s; no deviation over 5%).

## Engine-side changes this needs

None was made (the engine is not touched from here). In order of value:

1. `modules.dispatch.resolve` raises for `anthropic` on any op but triangle multiplication ("the anthropic payload serves
   triangle_multiplication only"), although `TriangleAttention` and `Transition` have `anthropic` branches. FoldForge calls the
   release's adapters (`integrations.anthropic.triangle_attention`, `.transition`) itself instead of the engine modules.
2. `integrations.opm_train.wanted` and `pwa_train.wanted` accept `ImplementationType.ANTHROPIC`, so an engine OPM / pair
   weighted averaging built with `implementation="anthropic"` runs the engine's own H100 path whenever it serves.
   `install_engine` therefore sets `settings.opm_train`, `pwa_train` and `pwa_infer` to False when those ops are on
   `anthropic`. `wanted` should exclude `ANTHROPIC`.
3. `anthropic_msa.opm_refusal` does not know `normalize_before_proj`: the release's epilogue adds the bias after the division
   by the pair count, AF3 before; the engine would serve AF3's OPM with the wrong bias. FoldForge refuses that case itself.
4. `ImplementationType.ANTHROPIC` is missing from `architecture.exceptions` (FoldForge keeps the name as a string).
5. `_CUEQ_OPS` has no `attention_pair_bias` although `cuequivariance_torch` has the op (FoldForge calls it directly).
6. The engine's OPM in the OpenFold-lineage convention (`normalize_before_proj`) is slower than the PyTorch reference at
   S = 512, L = 384 on H100 (1.9 ms against 1.4; the Anthropic path: 0.43 ms). The AF3 convention is 1.1 ms.

7. `integrations/anthropic_msa._ext()` JIT-compiles `miniworld_opm_epilogue` on first use; the 2.3.9 wheel ships the `native` variants of that
   extension prebuilt but not this one (seen on the `anthropic` OPM path of the test suite).

8. The window kernels of the portable atom paths (`augmented_attention_*_window_triton`) have no `ok` row in any recorded
   card's manifest at engine 2.3.10 (A100, RTX A5000, RTX A6000), so on those cards the AF3-style and ESMFold2 atom blocks and the
   atom stacks of Boltz-2 / RoseTTAFold3 / Chai-1 warn once (`UnrecordedCardWarning`) and run the dense path. `python -m
   miniworld_engine.autotune.run_all` on each card, and the manifest committed, turns them on without a FoldForge change.
9. The portable stacks refuse a head wider than 64 or more than 256 keys (one `KernelRefusedWarning`, then the reference) and, for the
   atom stacks, a token count that is not a multiple of 32; FoldForge's seam only asks, so those calls keep the dense blocks.
