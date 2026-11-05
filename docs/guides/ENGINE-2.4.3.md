# miniworld-engine 2.4.3: released-checkpoint parity

FoldForge pins `miniworld-engine==2.4.3`. The engine now builds its modules from the released (FoldForge dense) layout:
`<Module>.from_released(state_dict, ...)` (engine `docs/kernels/released-checkpoints.md`).

| FoldForge adapter (`modules/engine/modules.py`) | engine call |
|---|---|
| `_build_triangle_multiplication` | `TriangleMultiplication.from_released(..., outgoing=...)` |
| `_build_triangle_attention` | `TriangleAttention.from_released(..., transposed_bias=..., untransposed_output=...)` (gate / out / dual biases follow from the weights) |
| `_build_pair_weighted_averaging` | `MSAPairWeightedAveraging.from_released` |
| `_build_outer_product_mean` | `OuterProductMean.from_released(..., clamped_norm=True)` (the kernels refuse `count_offset`; see DEDUP) |
| `_build_single_attention` | `AttentionPairBias.from_released` |
| `_build_diffusion_variant_stack` / `_build_atom_variant_stack` | `DiTVariantStack` / `AtomDiTVariantStack.from_released` |

`variant_serves` counts `stack.serving(...) == "pytorch"` as served: the engine's PyTorch statements, not the dense blocks, are the
fallback of a built variant stack. The hand-written weight-copy code, the `_ENGINE_HAS_*` capability probes (and the `capabilities`
report key) and the ten `Compatibility API` wrappers are gone.
