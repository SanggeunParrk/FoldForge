# Porting ESMFold2 from team-gm into FoldForge

**Status:** PLANNED. The working ESMFold2 port lives in team-gm's
`exp/miniworld-integrated` branch, **uncommitted**, at
`src/team_gm/models/esmfold2/` plus ~17 driver scripts. It was written against
team-gm at `63cc244`. `origin/exp/miniworld` has since moved 10 commits ahead and
changed three things that the port depends on, so this is a rewire, not a
`git mv`.

## What changed underneath it

`origin/exp/miniworld` moved the ops out of team-gm into miniworld-engine, and
made the residual unconditional. Both are in
[ARCHITECTURE.md](../libs/team-gm/docs/ARCHITECTURE.md).

> [!NOTE]
> **Resolved upstream.** Engine `478cb72` (RoPE dtype) and `79c819d`
> (ConditionedTransition owns its AdaLN) landed the two correctness fixes, and
> team-gm `7978a0b` bumped its pin to `79c819d`. Upstream put the AdaLN in the
> **engine module**, not in team-gm's block — a better placement than the local
> fix this doc first described, because no caller can forget it. That also means
> a team-gm-side AdaLN would now **double-apply**; the local branch that did it
> has been deleted. Neither upstream fix carries a regression test, so
> `tests/test_swa_rope_dtype.py` is still worth offering.

### 1. The residual contract — the one that breaks silently

Engine modules now **always apply their own self-residual**, fused into the
kernel epilogue where possible, with no runtime toggle (rule 2). The port's
blocks add it themselves:

```python
# src/team_gm/models/esmfold2/trunk.py, as ported — WRONG under the new contract
pair = pair + self.tri_mul_out(pair, mask)
pair = pair + self.tri_mul_in(pair, mask)
return pair + self.pair_transition(pair)
```

Against the current engine every one of those double-adds. It will not raise —
it produces a plausible, wrong structure. Correct form is a bare call:

```python
pair = self.tri_mul_out(pair, mask)
pair = self.tri_mul_in(pair, mask)
return self.pair_transition(pair)
```

Cross-tensor residuals are the opposite: `OuterProductMean`, MSA pair-weighted
averaging and attention-pair-bias take `residual=` and the *model* owns it.
`msa_encoder.py`'s `pair = pair + self.outer_product_mean(...)` becomes
`pair = self.outer_product_mean(msa, msa_mask, residual=pair)`.

**Verify by structure, not by inspection.** Fold one target before and after and
compare coordinates against a same-code repeat: the fold is not bit-deterministic
(cuBLAS split-k, Triton atomics), and on 4yx2 two identical runs differ by
~0.38 Å. A real double-add is far outside that; eyeballing the diff is not.

### 2. Imports that moved

| the port imports | now |
|---|---|
| `team_gm.modules.layers` → `Transition`, `TriangleMultiplication` | `miniworld_engine.modules` |
| `team_gm.modules.layers.swa_atom_attention` → `build_attention_params` | `miniworld_engine.modules.swa_atom_attention` |
| `team_gm.modules.exceptions.ImplementationType.MINIWORLD_KERNELS` | `…MINIWORLD_ENGINE` |
| `team_gm.checkpoints` | `foldforge.checkpoints` (added here; it was never upstream) |
| `team_gm.modules.blocks` → `DiffusionTransformer` | unchanged |
| `team_gm.diffusion`, `team_gm.modules.primitives`, `team_gm.utils.transform`, `team_gm.typecheck` | unchanged |

`team_gm.modules.layers` now contains only `embeddings.py` and `ops.py`.

### 3. `SWAAtomTransformer` — DECIDED: FoldForge owns a copy

ESMFold2's atom encoder and decoder both need it. It is currently in
`team_gm.modules.blocks`, but team-gm's
[migration plan](../libs/team-gm/docs/MIGRATION-model-specific-to-terminals.md)
schedules it to move to **MiniWorld**, on the grounds that it is model-specific.

FoldForge cannot import from MiniWorld. Three options:

1. **FoldForge owns a copy** in `models/esmfold2/atom_transformer.py`. The block
   *is* ESMFold2's (Algorithm 8) and the architecture rule says model-specific
   blocks live next to the model that needs them — so two terminals each holding
   their own is what the layering prescribes, not a duplication bug. The shared
   part, `SWA3DRoPEAttention`, is already an engine op.
2. **It stays in team-gm.** Two terminals need it, which is the usual argument
   for hub code — but it would be the only model-specific block left there.
3. **It moves to the engine.** Wrong layer: it composes ops, it is not one.

**Option 1 was taken.** It unblocks FoldForge without waiting on MiniWorld's
migration, which the plan lists as blocked on MiniWorld adopting the rename
first. The file is now `models/esmfold2/atom_transformer.py`, copied from
`origin/exp/miniworld` with three changes: the ownership note in the docstring,
a corrected cross-reference (`AugmentedAttentionPairBias` moved to the engine),
and ruff's import sort. It is excluded from the formatter and carries a
per-file lint ignore so it keeps diffing cleanly against team-gm's copy and
MiniWorld's future one.

It sits in the model package rather than `foldforge.modules` because ESMFold2 is
its only consumer here; promote it if a second predictor needs it.

**Not yet verified.** It has been copied and lints, but nothing has imported it
— `miniworld_engine` is not installed in this environment, so the first real
check is an import smoke on a compute node after `uv sync`.

### 4. `OuterProductMean` — the engine could not express ESMFold2's convention

Found by the port, fixed in the engine rather than worked around here.

The engine computed `to_out(outer / n)` only. ESMFold2 needs `to_out(outer) / n`,
which scales `to_out`'s bias by `1/n` as well. The two differ by `b * (1 - 1/n)`
with `n` a count of valid MSA pairs in the hundreds, so the engine's ordering
discards essentially the whole trained bias — silently, since both produce
finite, plausible pair features.

Where each convention comes from:

| | divide vs `to_out` | denominator | bias scaled |
|---|---|---|---|
| AF3 (Algorithm 9: mean, then project) | before | `N`, unmasked in the reference impl | no |
| miniworld-engine | before (default) | per-pair valid count, `clamp(min=1)` | no |
| ESMFold2 (released) | **after** | per-pair valid count, `clamp(min=1)` | **yes** |

So the engine matched AF3 and ESMFold2 was the outlier. It is a genuine
divergence, not a porting slip: the reference implementation carries a
`divide_outer_before_proj` switch precisely because different ESMFold2
checkpoints were trained with different orderings.

Fixed on the engine branch `feat/opm-normalize-before-proj` by restoring a
`normalize_before_proj` flag (default `True`, so AF3 behaviour is unchanged),
with `tests/test_outer_product_normalization.py` covering both orderings, the
residual interaction, and — because the projection is zero-initialised and would
otherwise make both orderings agree at zero — a guard that the two actually
differ. **Still the only one of the three findings not upstream** (as of engine
`79c819d`), so FoldForge runs against that branch via PYTHONPATH.

Still missing from the engine's version: the `row_chunk` streaming team-gm's copy
had. The outer product materialises `[B, L, L, d_hidden**2]` before the
projection — ~0.7 GiB at L=594, ~16 GiB at L=2048.

## What comes with it

Beyond `models/esmfold2/`, the branch has work worth carrying over rather than
rewriting:

- **Drivers** (`scripts/`): end-to-end fold vs the reference, module-by-module
  parity on the real checkpoint, RMSD against a deposit with explicit chain
  mapping, op-dispatch audit, backend matrix, profile, optimisation sweep.
- **Measurements** (`benchmark/esmfold2/`): the provenance for every number in
  the docs. Gitignored, so copy the tree if it is wanted.
- **Two findings worth keeping as guardrails**, both currently enforced in the
  port's code: `lm_encoder` cannot be disabled (doing so takes 4YX2 from pLDDT
  0.842 / RMSD 4.4 Å to 0.550 / 11.3 Å for a ~6% saving), and the MSA encoder is
  hoisted out of the recurrence because it is loop-invariant.

## Residual fusion work — superseded, do not port

The branch also carries a residual-fusion implementation in the engine
(`gate_elem.py`, `transition/triton/main.py`, both `whole_op.py`) exposing a
`residual=` kwarg. **Drop it.** The engine already has this, done better:
`8faf495` (transition squeeze epilogue), `e196f90` (triton gate store),
`3469c48` (cute inference back), `a519b90` (bidirectional), and `ae99292` made it
unconditional — explicitly rejecting the runtime toggle the branch added, because
a toggle forces the slow separate-add path.

What is *not* superseded is the measurement, which was taken on **A6000 (sm_86)**
while the engine's fused paths were tuned on H100:

| | 4yx2, 594 tokens, bf16, A6000 |
|---|---|
| residual adds | 0.510 s of a 5.48 s fold (9.3%), 678 calls |
| the add is bandwidth-bound | 710 GB/s, identical to a pure copy |
| fusion recovers | 2/3 (measured 0.666 transition / 0.696 trimul) |
| whole fold | 5.493 → 5.177 s, **1.061x**, noise floor 0.003 s |
| structure | unchanged: 0.383 Å between arms vs 0.381 Å same-arm |

Worth reporting to the engine as sm_86 evidence.
