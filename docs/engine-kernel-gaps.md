# Engine work still needed to connect every family (2026-10-19, updated 2026-10-21 for miniworld-engine 2.3.3)

Source: the module-by-module wiring table of FoldForge at `69f7bd6` (engine pin `b4680179`, loader order
`install_pairformers` -> `install_engine_modules` -> `install_conditioning`). Every item below was read off a
rejected builder or a model attribute, not guessed. Counts are modules on the engine / modules in the model.

Already connected (no kernel work): AF3, IntelliFold2, OpenBind0, OpenFold3 (all columns), Protenix1/2 and OpenDDE
(everything, atom included since FoldForge `69f7bd6`), ESMFold2 token DiT / TriMul / MSA / OPM / single attention.

## 0. FIXED in miniworld-engine 2.3.3: H100 bf16 triangle attention overflowed on trained weights

The forward core now raises the softmax offset to the row's running maximum (lazy rescale; engine `cad85074`, test
`tests/gpu/sm90/test_triattn_large_logits_gpu.py`). Kept below as the record of what it was.

`kernels/triangle_attention/cuda/h100_sources/triattn_sm90.cu` uses a max-free softmax: the offset `m_r` is the maximum of the
row's FIRST 32 keys, fixed per task; a later logit more than ~126 log2 units (~87 nats) above it overflows to inf -> NaN. The
epilogue counts such rows in `FLAGS` but nothing reads the flag (`h100_sm90.py::inference` drops it), so the NaN is silent.
Reproduced with synthetic weights (`TriangleAttention(128, 4)`, bf16, L=128 and 256, random x): NaN fraction 0 up to logits ~100,
0.3% at ~190, 49-73% at ~500-580; fp32 is clean at every scale. OpenFold3/OpenDDE trained weights have logits up to ~100 with
q/k projection std 0.5, and the real `5i28` fold ends with `non-finite AF3 coordinates` the first time the engine module is attached
(bisected: only the triangle attention group triggers it). Independent of `transposed_bias` and of the mask. Needs: either an exact row
max (online softmax / a max pre-pass) or a flag-driven fallback to a safe path. Repro: `~/foldforge-scratch/ta_big.py`,
captured failing case `~/foldforge-scratch/ffbench/nan_case.pt`.

## 1. Boltz2 transition up-gate (unlocks Boltz2 token DiT 1, atom DiT 3, conditioned transitions 37)

`GatedDiffusionTransition`: `hidden = silu(g) * v * (W_up @ norm(x))` where `[g, v] = W_expand @ norm(x)`.
`ConditionedTransition` has the SwiGLU only, no third operand. Needed: a conditioned-transition variant with a third GEMM
(`a_to_b`, `n * d` wide) multiplied into the SwiGLU output, in the DiT block and in the local (atom) DiT block, inference first.
Boltz2's token DiT is otherwise plain (sigmoid AdaLN, per-block pair LN, gating, no kq-norm); its atom spec differs from
AF3 only in this transition and `atom_features_bias`.

## 2. RosettaFold3

- Token DiT and atom DiT: `kq_norm` (LayerNorm over the projected q and k, key width), and the **parallel schedule**
  `x + attention(x) + transition(x)` from one `x` (needs a delta-returning form of the blocks; only `ConditionedTransition`
  and `AugmentedAttentionPairBias` have `.delta` today).
- Triangle attention 0/116: `gate_bias` and `output_projection` bias on every `GridSelfAttention` (engine front/core/tail have none).
- Outer product mean 0/4: bias on the left/right projections (`projection_bias=True`).

## 3. Chai-1

- Token and atom DiT: parallel schedule, **no gating query**, **no output projection** (atom), identity-centred AdaLN scale
  (`(s + 1)` instead of `sigmoid(s)`), per-block pair LN; atom adds a same-conformer pair mask and (exact mode) the circular key window.
- Triangle attention 64/116 unwired: `untransposed_output` (ending node, 52 modules) and `dual_output` (confidence head, 8).
- Outer product mean 0/4: `groups=8` with a product LayerNorm.

## 3b. Chai-1 pair-bias attention (AttentionPairBias column 0/52)

Chai's `PairformerBlock` is not converted by `install_pairformers`, so its single attention (pair-bias attention with
`single_attention_gate_bias=1.0`) never reaches the engine's `AttentionPairBias`.

## 4. ESMFold2 / ESMFold2-fast atom DiT (0/3)

All four differ from the engine's 32x128 LayerNorm-AdaLN block:
RMS-norm + SiLU conditioned AdaLN with raw (ungated) output gate; identity-centred scale and no zero bias;
q/k RMS-normalised then rotary (`atom_rope`: 2 spatial axes + 10 uid axes, bases 20 / 10000);
192 keys per 32 queries with a +-64 window mask (`atom_keys_subset=192`, `atom_window_half=64`).
Conditioned transitions 12/25: 9 of the 13 unwired are the atom blocks' (same RMS / identity form), 4 are the diffusion-head ones of item 6.

## 5. Template pair stacks (width 64 with 4x32 heads) -- WIRED 2026-10-21

The builders pass the hidden width as the engine's `d_hidden` (it was never tied in the engine, only here): Protenix1
TriMul and TriAttn 116/116, Boltz2 TriAttn 156/156 (RF3 / Chai template TriAttn stay on their own other gaps).
TriAttn pair 64 / 4 x 32 heads runs the H100 CUDA kernels already; TriMul pair 64 / hidden 128 runs the engine's PyTorch
path on H100 until the engine ports B200's hidden = 2 x pair kernels (engine branch feat/trimul-h100-hidden2x, in progress).

## 6. Not kernel work

- DONE (2026-10-21): the 4 diffusion-head conditioning transitions (`use_single_cond=False`) run the engine's plain `Transition`
  (`engine.conditioning.ReleasedTransition`; every family 4/4).
- DONE in 2.3.3 (engine `909d7257`): the slide window's real-atom count is device-side, so the AF3 atom blocks run inside a CUDA graph capture
  and below 128 real atoms.

## 7. Open items on what exists

- AF3 atom variant (`kv_norm="parallel"`, `key_window="slide"`): inference only, speed not measured; the first two and the
  last windows are recomputed in torch. No backward.
- FP32 master weights under bf16 activations in the diffusion blocks: kept FP32 since `69f7bd6`; accuracy vs the reference
  not measured.
- Engine branch `feat/foldforge-af3` (b4680179): landed on engine main (544e758d) and released as miniworld-engine 2.3.3 on PyPI,
  which FoldForge now depends on (no git pin).
