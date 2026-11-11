# miniworld-engine 2.4.2: what moved here

> Historical migration record for that engine release. The current engine integration (2.5.1) is described in [BACKENDS.md](BACKENDS.md), [../ENGINE-UNSUPPORTED.md](../ENGINE-UNSUPPORTED.md) and [CHECKPOINT-FORMAT.md](CHECKPOINT-FORMAT.md).

miniworld-engine 2.4.2 removed `miniworld_engine.architecture` and `miniworld_engine.ops`
(engine `docs/CHANGELOG.md`, 2.4.2 "Removed"). FoldForge pins `miniworld-engine==2.4.2`.

| was | now |
|---|---|
| `architecture.diffusion.*` (base, edm, decoupled_xpred, augmentation, schedules, samplers) | `foldforge.diffusion` (same layout) |
| `architecture.utils.align` / `se3` / `transform` | `foldforge.diffusion.geometry.alignment` / `se3` / `kabsch` |
| `architecture.tensor_layout` | `foldforge.diffusion.tensor_layout` |
| `architecture.typing.typecheck` | `miniworld_engine.utils.typecheck.typecheck` (inside `foldforge.diffusion`: its own `_typecheck`) |
| `architecture.exceptions` / `_engine_impl` (`ImplementationType`, `to_engine_impl`) | `foldforge.layers.implementation` |
| `architecture.blocks.pairformer` | `foldforge.blocks.pairformer` |
| `architecture.checkpoints.backend_attention` | `foldforge.checkpoints.backend_attention` |
| `architecture.primitives.LayerNorm` | `miniworld_engine.modules.primitives.LayerNorm` |
| `ops.layer_norm` / `ops.swiglu_ffn` / `ops.triangle_multiplicative_update` | `miniworld_engine.kernels.layer_norm` / `swiglu_ffn` / `triangle_multiplicative_update` |
| `ops.transition` | `miniworld_engine.kernels.transition_layer` |
| `ops.augmented_attention_pair_bias` | `miniworld_engine.kernels.augmented_attention.whole_op.augmented_attention_pair_bias` |
| `ops.layer_norm_linear` | `miniworld_engine.kernels.layernorm_linear.interface.layer_norm_linear` |
| `ops.gated_linear` | removed; `linear(sigmoid(gate) * value)` (no CUDA gate-row kernel any more) |
| `miniworld_engine.integrations.anthropic*` | `miniworld_engine.baselines.anthropic*` |
| `miniworld_engine.integrations.token_dit` | `miniworld_engine.kernels.integrations.token_dit` |

`foldforge.diffusion` imports torch, numpy, scipy, pydantic and jaxtyping only, so the folder can be moved into its own package.

## Update: engine modules only (branch `migrate/engine-modules`)

FoldForge no longer calls engine kernel functions for model computation. A dense module the engine has a module for gets that
module at install time (`modules/engine/modules.py`, now including `Transition` -> `miniworld_engine.modules.Transition`), and
`NativeLayerNorm` hands its BF16 calls to `miniworld_engine.modules.primitives.LayerNorm`. Everything else runs its PyTorch
statements: the rows above that map to `kernels.layer_norm`, `swiglu_ffn`, `triangle_multiplicative_update`, `transition_layer`,
`augmented_attention_pair_bias` and `layer_norm_linear` are gone, as is `foldforge.checkpoints` (`backend_attention`,
`gate_projection`). Remaining infrastructure imports from `miniworld_engine.kernels`: `_capture` (graph capture scopes),
`sampler_ops.interface` (sampler glue), `transition.cuda._weights` (weight re-layout cache clear),
`integrations.token_dit.serves_extra_bias`, `dit_variants.cuda.parallel_residual`.
