# Checkpoint backend dispatch and CUDA graph experiment — 2026-09-15

The AF-family MiniWorld selection previously left triangle and token attention
cores on the PyTorch path. Supported square BF16 attention now calls the engine's
public `ops.augmented_attention_pair_bias`; triangle rows use the augmentation
axis, retaining the shared pair bias and a separate key mask per row. Projection
weights, AF3 ending-direction bias order, residual ownership, and padded-row
semantics remain checkpoint-owned. This is an inference adapter.

A backend name identifies the requested implementation family, not a promise that
that vendor implements every model operation. The explicit coverage is:

| Core operation | PyTorch selection | MiniWorld selection | cuEquiv selection |
|---|---|---|---|
| Main/MSA TriMul, equal pair/hidden widths | Shared PyTorch | Engine whole op | Vendor TriMul |
| Template TriMul, pair 64 / hidden 128 | Shared PyTorch | Shared PyTorch: engine width contract unsupported | Shared PyTorch: adapter/vendor width contract unsupported |
| AF3/Protenix/OpenDDE triangle attention | Shared PyTorch | Engine projected attention, CUDA BF16, head dimension ≤64 | Vendor triangle attention, including stacks outside Pairformer |
| AF3/Protenix/OpenDDE square token attention | Shared PyTorch | Engine projected attention, CUDA BF16, head dimension ≤64 | Shared PyTorch: no matching vendor op |
| Released rectangular atom attention (typically 32×128) | Shared PyTorch SDPA/checkpoint math | Same shared path: public engine core is square-only | Same shared path |
| Transition | Shared PyTorch | Engine whole op for compatible native BF16 | Shared PyTorch |
| AdaLN / conditioned transition | Engine's PyTorch reference with original weights | Engine fused path for compatible BF16; FP32 coordinate streams retain reference normalization before projection | PyTorch reference |
| LayerNorm | Shared PyTorch, FP32 parameters/reduction | Engine LayerNorm for compatible BF16 | Shared PyTorch |
| Pair-bias LN→linear | Shared PyTorch | Fused public engine op when both affine biases are absent and output width is supported; otherwise separate operations | Shared PyTorch |
| OPM contraction / MSA weighted averaging | Shared PyTorch | Shared PyTorch core, selected normalization; no general public fused engine op | Shared PyTorch |
| ESMFold2 SWA | Permitted FA2 plus PyTorch preprocessing | Existing engine Q/K preprocessing, RoPE/RMSNorm and gate/projection dispatch plus FA2 | PyTorch preprocessing plus permitted FA2 |
| ESMFold2 token DiT | Existing PyTorch reference | Existing engine augmented attention / conditioning | Existing PyTorch reference |
| General linear projections, embedding, geometry, sampler, confidence math | PyTorch | PyTorch/compiler unless included in a supported whole op | PyTorch |

The shared Protenix/OpenDDE triangle adapter retains its existing PyTorch path for
length ≤16. Training, CPU, FP32 and unsupported head sizes do not enter the new
checkpoint attention path. The engine's existing length-key mapping is unchanged;
physical token bucketing still rounds up to a multiple of 128.

OPM is not a missing call to a hidden fused OPM kernel: the engine's own module
also uses PyTorch contractions. Substituting that module would additionally change
some released checkpoints' normalization epsilon/order. The private bias-only
attention kernel assumes a row count equal to token length and does not directly
represent general MSA depth. These remaining PyTorch paths need new compatible
kernel capabilities, not a backend flag relabeling.

Graph experiment (before this wiring change): cssb3 A100 80GB PCIe, AF3 4YX2,
10 recycles, 200 diffusion steps, 5 samples, native BF16 parameters with FP32 norms,
no autocast, compile on, graph scope `denoiser`. Same loaded model and compiled
callable per backend; identical RNG restored before each forward; synchronized
whole-forward wall timing. Order: off/on warmups, then off/on/on/off/on/off.
Three measured observations per state; no measured recaptures. Graph on replayed
1,000 times per forward; graph off replayed zero times. Compile's automatic graph
option was disabled in both states, so the off state is genuinely off.

| Original wiring | Graph off median (s) | Graph on median (s) | Speedup | Wall-time reduction |
|---|---:|---:|---:|---:|
| PyTorch | 68.2783 | 67.4419 | 1.0124× | 1.22% |
| MiniWorld | 52.5087 | 51.0212 | 1.0292× | 2.83% |

This is a complete-forward result, not a denoiser-only GPU-kernel timing. Trunk,
Python sampling, graph input copies and output clones remain outside the captured
callable. The data does not establish the isolated cost of each of those parts.
One model/input/GPU is not evidence of the same benefit for every model or length.

Raw experiment sources/results are preserved on cssb3 under
`/home/psk6950/practice/FoldForge/validation/graph-ab-20260915/` (job 43907),
with an immutable source snapshot. The post-wiring snapshot, numerical checks,
full-checkpoint smoke tests and repeated timing live under
`validation/backend-wiring-20260915/`.

After the wiring fix, the identical AF3 configuration (job 43917) measured:

| Final MiniWorld wiring | Graph off median (s) | Graph on median (s) | Speedup | Wall-time reduction |
|---|---:|---:|---:|---:|
| AF3 4YX2 | 32.2754 | 26.2397 | 1.2300× | 18.70% |

With graph on in both cases, the new path is 2.5702× as fast as the verified
PyTorch baseline and 1.9444× as fast as the prior MiniWorld wiring. The experiment
changes triangle/token attention and eligible LN+projection wiring together;
it does not attribute the speedup to one isolated kernel. All six measured new
forwards had zero recaptures, and each graph-on forward had 1,000 replays.
The first off warmup included 234.82 s of compilation/autotuning plus inference;
that observation is excluded. Tuned-cache warnings remain for known missing or
stale entries, so these measurements are not a claim of a fully populated cache.

Validation records below are updated from completed Slurm results. Numerical
checks use nonzero projection weights, irregular/full masks, both triangle
directions, batch and unbatched layouts, supported/unsupported head widths,
CPU/reference fallbacks, actual custom-op observations, compile fullgraph and
manual replay. GPU validation checks inference semantics; it is not a scientific
qualification of full sampled structures or a new training qualification.

The unit-scale-normalization numerical run (43923) passed all 19 tests. Across
15 reported numerical comparisons, the largest relative RMS error was 0.6901%
(AF3 token attention with irregular masking), and the largest absolute element
error was 0.015625 (fused pair-bias LN→linear). These are BF16 module errors, not
coordinate errors in angstroms. The direct compiled/graph replay comparison is
recorded separately in the JSON. Unsupported rectangular/head-96/FP32 cases
matched their unchanged reference exactly.

The initial 15-test run was 43911. The final regression job 43919 passed 19 backend
tests and 89 existing inference/bucketing/I/O tests. One I/O test initially could
not find `scripts/verify_prediction_artifacts.py` in the isolated source snapshot;
its unchanged helper was copied into the snapshot and only that test was rerun
in job 43937. This was a validation-snapshot omission, not a model failure.

Full-checkpoint smoke configuration: 3PTB, one recycle, two diffusion steps,
one sample, native BF16, no autocast, denoiser compile and manual graph enabled.
Each run must produce prediction/confidence artifacts and record actual compiled
graphs and replays. CPU operator counts include tracing/warmup/capture, not
individual launches inside subsequent GPU graph replay. Protenix runtime checks
in this record use v1; v2 uses shared source but was not requalified here.

Final smoke results: all eight combinations (four models × MiniWorld/cuEquiv)
passed. Each recorded one capture and four replays. Job 43937 passed the remaining
I/O test, completing 90 existing regression tests. Full evidence:
[BACKEND-WIRING-20260915.json](BACKEND-WIRING-20260915.json).
