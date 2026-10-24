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
| cuequivariance, `trimul` | bf16 CUDA, pair width a multiple of 32, no checkpoint-specific length division |
| cuequivariance, `triangle_attention` | bf16 CUDA (any head count the library takes) |
| cuequivariance, `single_attention` | bf16 CUDA, AF3's form: no single conditioning, k/q norm or gate bias (so not Chai-1's) |
| anthropic, `trimul` | `TRIMUL_NATIVE_BUILD_DIR` names a payload; bf16, no grad, one square pair plane per call, sm_90 (sm_100 with the rebuilt member), a payload unit for (pair width, hidden width): 128/128 and 128/256 |
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

- `miniworld` runs what the engine runs there (the engine owns that);
- `cuequivariance` and `pytorch` do not depend on the card;
- `anthropic` serves what its payload serves on that card and refuses the rest with the reason from the payload (sm_90a
  units for TriMul and the MSA cells; an sm_80 member exists for TriMul): everything else is the PyTorch reference.

Only an H100 was available. Every cell of the tables above was run there; the other cards are untested (A100, A6000,
L40 and B200: nothing of this was run on them), and so is a `miniworld` run of these four ops off an H100.

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

RESULTS_PLACEHOLDER
