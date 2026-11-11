# Requests to the miniworld-engine team (2.5.0)

Requests only: what FoldForge would like the engine to provide. What the engine does not support and what FoldForge does meanwhile is in
[ENGINE-UNSUPPORTED.md](ENGINE-UNSUPPORTED.md); each request names the rows there it would retire (`U-<section>`).

Priority: **P1** blocks a clean integration or costs most of a fold's speed on the main path (H100, bf16); **P2** removes a FoldForge workaround
or a per-family slow path; **P3** convenience, documentation, or a wish for exactness in a corner.

Numbering is stable (R1 ... R19); a request that is done is struck out here, not renumbered. Earlier lists were merged: the former items 1 / 5
are R1, 4 / 8 are R2, 3 / F4 are R6, 6 is R5, 9 / D2 are R4, 7 / 11 are R10, D1 is R7, F1 is R8, F2 is R9, F3 is R11.

| # | Priority | Request | Retires |
| --- | --- | --- | --- |
| R1 | P1 | Public CUDA-graph capture and weight-cache API | U-removed (capture), U-layout (caches) |
| R2 | P1 | Kernel-served `OuterProductMean` with `count_offset` (AF3's `n + 1e-3` divisor) | U-backend (OPM) |
| R3 | P2 | `AttentionPairBias` `cuda` / `triton` with an additive `extra_pair_bias` | U-backend (single attention) |
| R4 | P2 | `protenix_v2` key (or documented alias) on `TriangleAttention` / `TriangleMultiplication` | U-class (protenix2 keys) |
| R5 | P2 | Separate model keys for the triangle attention and the triangle multiplication of a `PairformerBlock` | U-layout (RoseTTAFold3 fast) |
| R6 | P2 | ESMFold2 atom block with the atom pair (and rotary keyed by `uid`), or a documented non-equivalence | U-class (ESMFold2 atom stack) |
| R7 | P2 | Chai-1 `TokenDiTChai1` / `AtomDiTChai1` on `cuda` with FoldForge's mixed dtypes (bf16 `single`, fp32 `cond` / `pair`) | U-backend (Chai-1 DiT) |
| R8 | P2 | fp32 `TriangleMultiplication` / `TriangleAttention` / `Transition` kernels (`precision: fp32` runs them on `pytorch`) | U-backend (fp32) |
| R9 | P3 | Chai-1 `circular` atom key window, or a documented equivalence of `pad` + `uid` | U-backend (Chai-1 key window) |
| R10 | P3 | Stack classes for the DiT blocks (`TokenDiTStack` / `AtomDiTStack`) and a `from_shared_pair(...)` helper for the released shared pair norm | U-layout (DiT stacks, shared pair norm) |
| R11 | P3 | A Chai-1 MSA block (`outer_product_mean(groups=8)` into parallel MSA / pair stages) and a general MSA block with the AF3 / Boltz orders | U-class (MSA block) |
| R12 | P3 | `parallel_residual`: a fused residual add of parallel branches | U-removed (parallel_residual) |
| R13 | P3 | An option on `AtomDiTOpenFold3` for AF3's mask form under 128 real atoms | U-equivalence (atom window < 128 atoms) |
| R14 | P3 | Class-docstring invariants for the shared-KV atom AdaLN and RoseTTAFold3's zero query bias (or a `query_bias` constructor option) | U-layout (shared KV, zero query bias) |
| R15 | P2 | Export the dead-module loader: `mark_dead_modules(model, threshold=1e-30)` / `clear_dead_modules(model)` from the package root | U-removed (dead-module loader) |
| R16 | P1 (training) | `AtomDiT` blocks send a gradient to the atom-pair (pair-bias) input | U-backend (training: atom-pair conditioning) |
| R17 | P2 (training) | An autograd-capable `PoolAtoms` (or a backward for the `sampler_ops` module) | U-backend (training: pooling) |
| R18 | P2 | A public multi-fold (trunk_seed x diffusion_seed) contract for the DiT blocks | U-class (grouped 5x5 sampler) |
| R19 | P2 | Smaller temporaries for the 2.5.0 `PairformerBlock` batched over samples (the confidence head's CUDA-graph pool) | U-class (confidence-head graph memory) |

## P1

**R1. Public capture API.** FoldForge's CUDA-graph wrapper (`models/execution.py::_static_inputs`) enters
`miniworld_engine.kernels._capture.static_weights()` / `static_inputs()` around every capture (via `modules/engine/capture.py`), and `modules/engine/capture.py::
clear_weight_caches` empties `kernels._capture._FORMS`, `kernels._tf32._CACHE` and the `_PACKS*` dicts of `kernels.integrations.*`
before weights are replaced. All of that is private API, confined to `modules/engine/capture.py`. Please export `static_weights`, `static_inputs` and `clear_weight_caches` from the
package root with the same semantics (a model that builds a second set of weights, or a test that builds many models, needs the clear).

**R2. OPM `count_offset` on kernels.** AF3's divisor is `n + 1e-3`; the `cuda` / `triton` backends serve only `clamp(n, 1)` and refuse a call that
sets `count_offset`, so AF3-family rows choose between the kernels (clamp divisor, a relative difference of 1e-3 / n) and exactness (`pytorch`).
FoldForge builds `count_offset=None` by default and runs the exact form on `pytorch` only where the mode asks for it
(`modules.OPM_COUNT_OFFSET`). A kernel with the offset removes the choice.

**R16. Gradient into the atom-pair conditioning (training).** In a training step of Protenix v1 the parameters that build the atom pair (the input embedder and diffusion head `embed_pair_distances`, `embed_pair_offsets`, `single_to_pair_cond_row` / `_col`: 8 tensors) receive no gradient at all: the engine `AtomDiT` blocks appear not to send one back through their pair-bias input. Observed, root cause not confirmed. Without it those parameters never train. Please check, and if it is missing, provide backward for the pair input of the `cuda` / `triton` / `pytorch` backends of `AtomDiT*`.

## P2

**R3. `AttentionPairBias` with `extra_pair_bias`.** OpenDDE's structural-token refiner adds an `[L, L]` bias to every head of the single
attention. The `cuda` backend refuses it ("the kernels compute the bias from pair inside the pass"), so those calls (4 per fold) run
`pytorch`. An additive logit input, or folding the bias into the hoisted pair bias as `TokenDiTProtenix` does, would serve them.

**R4. `protenix_v2` key.** `TokenDiT` / `AtomDiT` have `protenix_v2`; `TriangleAttention` / `TriangleMultiplication` do not, so protenix2 builds
them with `protenix` (same math: transposed ending bias). A documented alias lets the family key map to one name.

**R5. Separate triangle keys.** A `PairformerBlock` has one `model` key for both the triangle attention and the multiplication. RoseTTAFold3's
fast mode wants RF3 attention with the plain multiplication, so FoldForge replaces the block's two multiplications after building it
(`modules/engine/pairformer.py`). A second key (for example `trimul_model=`) removes the replaced sub-modules.

**R6. ESMFold2 atom stack with the pair.** `AtomDiT("esmfold2")` is the pair-free sliding-window block (fused `Wqkv`, six-way modulation, 3D RoPE
inside the block, no `to_bias`). FoldForge's ESMFold2 atom stack is the AF3 block over a `[windows, 32, 192, d_pair]` atom pair with the
family's rotary and RMS / SiLU AdaLN, so it stays FoldForge-own. Either an option that takes the pair bias and the rotary keyed by `uid`, or a
statement that the two are different functions, tells FoldForge whether the stack could ever move.

**R7. Chai-1 DiT on `cuda`.** FoldForge hands the token block bf16 `single` and fp32 `cond` / `pair` (the conditioning is built in fp32).
`TokenDiTProtenix` serves that on `cuda`; the Chai-1 classes refuse ("all bfloat16 or all float32"), so Chai-1's 64 token-DiT and 27 atom-DiT
calls per fold run `pytorch` (observed on H100). The atom block also refuses under the fp32 sampler glue ("bfloat16 activations under bfloat16
autocast only"). Accepting fp32 `cond` / `pair` with a bf16 `single` (or casting inside the class) would put them on `cuda`.

**R8. fp32 pair-stack kernels.** An fp32-activation (`precision: fp32`) fold of the Protenix lineage runs `TriangleMultiplication`,
`TriangleAttention` and `Transition` on `pytorch` (the kernels are bf16-only). Documented behaviour; listed because every fp32 reference run is
mostly `pytorch`.

**R15. Dead-module loader.** 2.5.0 keeps the plain `dead` flag on `Transition`, `AttentionPairBias`, `TriangleAttention`, `TriangleMultiplication`, `OuterProductMean` and `MSAPairWeightedAveraging` (a flagged module returns its residual input without grad outside training) but no longer ships the function that sets it. Please export `mark_dead_modules(model, threshold=1e-30)` and `clear_dead_modules(model)` from the package root with the 2.4.x semantics (all parameters of a module below the threshold). FoldForge keeps `modules/dense/dead.py` meanwhile, which scans every class that declares `dead`; on Protenix v1 it skips 76 modules (the confidence head: 8 triangle attentions, 2 triangle multiplications, a pair transition; 34 attention-pair-bias and 31 transition modules of the trunk) with bit-identical output.

**R17. `PoolAtoms` backward (training).** The `sampler_ops` modules (`AugmentNoise`, `AlignEuler`, `PoolAtoms`) are inference-only by design (no autograd), so a training step cannot use `PoolAtoms` and FoldForge switches every module to the torch pooling (`foldforge_sampler_ops=False`, `train/prepare.py`). The same op (gather + relu + masked mean over the atoms of a token) with a backward would remove the switch and the slower torch gather in the training graph (`indexing_backward_kernel` is the largest PyTorch backward kernel of a step).

**R18. Multi-fold sampling contract.** FoldForge's trunk_seed x diffusion_seed sampler (`models/grouped.py`) runs the DiT blocks once over `[A, NT, ...]` inputs where each of the NT folds has its own pair, conditioning and atom pair; the engine's grouped-bias attention reads sample s's pair from fold `s % nbg`. That behaviour is not documented, and FoldForge reaches it through `stack.blocks` of the DiT stacks. Please state the contract (for example a public `pair_groups` argument) so it can be relied on. Measured end to end on FoldBench (1,522 assemblies): 1.5x for assemblies up to 256 tokens, 1.08x over the whole set, 1.0x above 1,024 tokens.

**R19. Confidence-head graph memory.** With one CUDA-graph pool per captured graph, the 2.5.0 `PairformerBlock` batched over samples (`engine/graph.py::make_pairformer_block(..., batched="sample")`) needs 2.4x the pool of the 2.4.0 fused `AF3Pairformer` in the confidence head: summed over the captures of 70 ascending-size FoldBench assemblies, 26.9 GiB against 11.3 GiB (largest bucket 5.67 against 3.04 GiB). Together with the evoformer pools (about 37-41 GiB) the private pools reached 80 GiB at the ~400-token bucket and a capture failed (five assemblies; `release-graphs target` passed). FoldForge now evicts the callable's oldest graphs before a capture when memory is short (`models/execution.py`), but the size of the temporaries is the engine's: please check whether the batched block can use a fused path or state a workspace-size contract. The temporaries were estimated from pool sizes, not profiled.

## P3

**R9. Chai-1 `circular` window.** `AtomDiTChai1(key_window="pad")` + `uid` equals the old exact mode unless a conformer spans both ends of the atom
list (or the system has fewer than 128 atoms). If the release's `get_qkv_indices_for_blocks` really wraps, an engine `circular` window makes
FoldForge's `exact` mode exact again.

**R10. DiT stacks and the shared pair norm.** FoldForge assembles `ModuleList`s of `TokenDiT*` / `AtomDiT*` blocks itself
(`modules/engine/transformers.py`, with the `extra_pair_bias` / `uid` plumbing), and converts the released shared pair LayerNorm / projection
once, offline, into per-block copies (`models/checkpoints/legacy/released.py`). Reference stack classes and a `from_shared_pair(...)` helper would let
every caller of a released AF3-family checkpoint use one engine implementation.

**R11. MSA block.** The engine has no MSA block; FoldForge composes `modules/engine/graph.py::EvoformerBlock` from engine parts, including
Chai-1's parallel schedule (each part owns its residual, so a stage's update is the sum of the parts' deltas `part(x) - x`).

**R12. `parallel_residual`.** `EvoformerBlock._forward_parallel` (Chai-1's MSA block) sums parallel branches in torch (`x + sum_i (y_i - x)`) because the
parts own their residuals; the 2.5.0 modules expose no fused equivalent. (The pairformer blocks use the engine's own `parallel=True`.)

**R13. AF3 mask under 128 atoms.** `AtomDiT*` switch every masked key off; AF3's own form (`+1e9` only where both query and key are padding) lets a
real query read the padded keys of its slid window below 128 real atoms. Identical from 128 atoms (tested at 130 and 200).

**R14. Documented invariants.** `AtomDiTBoltz2` / `AtomDiTRoseTTAFold3` / `AtomDiTChai1` share one AdaLN for q and k / v; FoldForge checks the
released `k_adaptive_layernorm` equals the query one before discarding it. RoseTTAFold3's `FlatQKNormAugmentedAttention` has no query bias; FoldForge
drops the released zero bias after checking it. If the releases are documented to do this, say so in the class docstrings so the checks are
confirmations, not inferences.
