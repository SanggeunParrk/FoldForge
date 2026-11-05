# Execution scopes: what compile and CUDA graphs cover

`execution.scope` selects the part of the forward that `compile` and `cuda_graph` wrap
(`src/foldforge/models/execution.py`). Each wrapped callable keeps one CUDA graph per input
signature (tensor shapes, dtypes, strides and Python scalars/bools); replay copies the inputs into
static buffers.

`max_graphs` bounds each wrapper's resident graph cache. When a new input signature
fills the cache, the least recently used graph is evicted and the new signature is
captured. A loaded predictor can therefore serve many input shapes without growing
the graph cache indefinitely. Compiled callables survive between requests; graph
replay is retained only where the existing shape and conditioning lifetime permit it.

| scope | wrapped | CUDA graph |
|---|---|---|
| `denoiser` (default) | `diffusion_head.forward`, called once per sampler step | yes |
| `trunk` | the denoiser, plus `evoformer.forward` (one call per trunk pass), `confidence_head.forward` and `distogram_head.forward` | yes |
| `model` | the whole `forward` | no: host-side sampling cannot be captured; compile only, experimental |

With `trunk`, `compile` still applies to the denoiser only; the trunk and head sites are graph-only.
Compiling the trunk gained no time over the graph and changed the result (Protenix1 5I28: pLDDT
73.86 compiled vs 70.34 eager and graph-only).

`trunk_graph_max_tokens` (default 512) bounds the trunk and head graphs: above that many tokens
(the largest `[N, N, ...]` input) those sites run uncaptured with the same kernels. There the GPU is
already full (Protenix1 n=8, 1,024 tokens: 97% busy) so the graph gains nothing, and its memory pools
made the denoiser capture run out of memory at 2,048 tokens. `uncaptured_calls` in the execution
report counts them.

`trunk` runs a trunk pass as three graphs (first / middle / last pass differ in their Python flags),
so `max_graphs` should be at least 8 with it.

## Known limit: trunk graphs on three families

`DenseSpec.trunk_graph` is False for Boltz2, Chai-1 and RosettaFold3. For
those, `scope: trunk` does not wrap the trunk pass and the heads; the denoiser and the sampler step still
run as graphs. A trunk graph of these families is captured without error and replays correctly right after
the capture, but replaying it on the NEXT fold raises `illegal memory access` (Boltz2, reproduced with
the evoformer alone or the confidence head alone; recapturing per fold avoids it, so the graph reads
state that does not survive the fold; the cause is not found). The other families are unaffected.

ESMFold2 (and ESMFold2-fast) is on the trunk graph. Its pass was not capturable only because the first pass
drew the recurrent state with `trunc_normal_` inside the pass, whose rejection loop reads the device on the
host; the state is now drawn in `AlphaFold3.forward`, before the passes (`_ssm_state`). At 128 tokens the
eager trunk pass was bound by the host (the pass launches ~850 kernels for ~10 ms of GPU work): 0.24 s for
three passes against 0.03 s as graphs. A second, different fold (the same chains with each sequence reversed)
replays the graphs captured on the first without error and with the pLDDT of an eager run.

## Bucketing (default on)

`execution.bucketing` (CLI `--bucketing` / `--no-bucketing`) pads tokens (multiples of 128), atoms and MSA
rows up the ladders of `models/bucketing.py`. It is on by default: the dense atom layout reserves 24 atom
slots per token, so without bucketing the atom blocks compute about three times the real atoms (Protenix1,
8 chains: 24,576 slots for 7,784 atoms), and the sampler is slower by that much. The padded shape also
fixes the shape of the sampler's random draws, so a result differs from an unbucketed one by sampling
noise, not by accuracy; use `--no-bucketing` to reproduce an unpadded run.

## Rules for code inside a captured scope

A capture runs the callable twice outside the capture (warmup), then once inside it. Inside the
capture nothing may synchronise with the host or copy from pageable host memory:

- No `.item()`, `bool(tensor)`, `float(tensor)`, `.tolist()` or data-dependent Python branching on
  tensor values. The capture raises on a sync; a branch on a value that is constant per fold is baked
  into the graph and would be wrong for another input of the same shape.
- No `torch.tensor(...)`, `new_tensor(...)`, `torch.as_tensor(...)` or `table.to(device)` of a host
  value per call, and no `tensor[idx] = 1.0` with a Python scalar (the scalar is a CPU tensor).
  Use a device scalar (`torch.ones((), device=...)`, `torch.full_like`), `x.zero_()`, or a constant
  table placed once by `foldforge.utils.device_constants.on_device` / `built_on_device`. The warmup
  calls fill that cache before the capture starts.
- Host randomness (the MC-dropout coin in `AlphaFold3.forward`, the sampler's noise) stays outside
  the wrapped callables.

## Out of scope: the ESM-C language model

FoldForge does not optimise ESM-C (the language model of ESMFold2). Its speed work is taken from
Anthropic's ESMFold2 optimisation kit as shipped: CUDA-graph capture of the language model and MSA
encoders (lever `eg`), ESM-C MLP through TransformerEngine (`esmc_mlp=te`), and the LM block
streaming above 1,500 tokens (`x4`). Nothing in `scope: trunk` wraps the language model.


## Repeated compiled predictions

A 2026-10-28 H100 check with a reusable Boltz2 predictor, 5I28 real MSA/templates,
BF16 fast mode, total trunk passes 10, diffusion steps 200, five samples and
`benchmark_repeats: 1` reproduced coordinates exactly with eager execution and
with CUDA graphs without compilation. With `compile: true`, the second request
returned coordinates differing from the first by up to **2.84283 angstroms**.
Prepared feature hashes, trunk tensors and the initial diffusion RNG states were
identical. The difference begins during diffusion; Dynamo also reports hitting
its recompilation limit and falling back for engine-wrapper specializations.
The warning's causal role has not been established. Small four-step checks were
exact in all tested modes. No numerical tolerance was relaxed to call the longer
compiled repeat reproducible.

The reusable API preserves the explicit compile policy. For reproducibility under
this measured workload, keep `compile: false`; graphs may stay enabled. Resolving
this compiled diffusion behavior is separate from input/model lifetime ownership.
Raw evidence: `/home/psk6950/foldforge-scratch/reusable_api_20261028/REPORT.md`.
