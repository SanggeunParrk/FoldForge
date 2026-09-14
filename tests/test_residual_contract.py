"""No model may re-add a residual an engine op already owns.

Engine ops apply their own self-residual and fuse it into the kernel epilogue
(`libs/team-gm/docs/ARCHITECTURE.md` rule 2). Writing ``x = x + op(x)`` on top of
that doubles it, and **nothing raises** — it produces a plausible, wrong
structure. It cost one 3PTB fold pLDDT 0.973 -> 0.750 via a tripled conditioned
pair (|x| 15.1 -> 46.1) and an over-compacted structure (Rg 15.9 -> 14.9 A).

The grep that was supposed to catch it, ``= \\w+ \\+ self\\.``, missed both
instances because they call through a loop variable::

    for block in self.pair_transitions:
        conditioned = conditioned + block(conditioned)   # no `self.` to match

So this walks the AST instead of the text, and finds the *shape* of a
self-residual regardless of how the callee is spelled: an assignment
``name = name + call(...)`` whose call takes ``name`` as its first argument.

Every such site must be listed in ALLOWED with a reason. That is the point: a new
one fails until somebody classifies it, rather than passing until somebody
notices the structure is subtly wrong.
"""

from __future__ import annotations

import ast
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "src" / "foldforge" / "models"

#: Self-residual sites that are model-level and correct, with why. Keyed by
#: "<model-relative-path>:<callee>" — not by line number, which every edit invalidates.
ALLOWED = {
    # AF3 Evoformer keeps GridSelfAttention raw deltas; these are not engine
    # TriangleAttention modules (which already own their residual).
    "af3/ported/nn/pairformer.py:pair_attention1",
    "af3/ported/nn/pairformer.py:pair_attention2",
    # A projection add, not an op residual: single features broadcast into the
    # token track (AF3 diffusion conditioning).
    "esmfold2/diffusion.py:single_to_token",
    # Whole-stack residuals around a FoldingTrunk. The blocks apply their own
    # residuals internally; this outer one is the released architecture (the
    # confidence head and the LM-side pair encoder both have it).
    "esmfold2/confidence.py:folding_trunk",
    "esmfold2/pair_trunk.py:lm_encoder",
    # Pair-track feature adds in the confidence head's conditioning: row/col
    # projections of `single` and the distance-bin embedding, none of them ops.
    "esmfold2/confidence.py:single_to_pair_row",
    "esmfold2/confidence.py:single_to_pair_col",
    "esmfold2/confidence.py:single_to_pair_prod_out",
    "esmfold2/confidence.py:distance_bin_embed",
    # Reference-conformer coordinates projected into the atom track.
    "esmfold2/atom_encoder.py:coords_linear",
    # Cross-tensor: OuterProductMean reads msa and adds into pair, so the engine
    # returns a raw delta and the model supplies the target. Passed as
    # `residual=`, so it never appears as `x = x + ...` — listed for the reader.
    # These are the upstream raw-delta modules, not shared residual wrappers.
    # Fold-CP calls Transition.forward directly (linear SwiGLU output only).
    "opendde/ported/distributed/foldcp/real_pairformer.py:single_transition",
    # TriangleAttention returns projected attention only; Pairformer owns the add.
    "opendde/ported/model/modules/pairformer.py:tri_att_start",
    "opendde/ported/model/modules/pairformer.py:tri_att_end",
    # Sequential Linear/ReLU atom-pair MLP has no internal residual.
    "protenix/ported/model/modules/transformer.py:small_mlp",
    # Guidance projection returns an accumulated coordinate delta (zero if disabled).
    "protenix/ported/tfg/engine.py:_project",
    "opendde/ported/tfg/engine.py:_project",
}


def _callee_name(call: ast.Call) -> str | None:
    """Rightmost name of the thing being called, however it is spelled."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Subscript):  # self.blocks[i](x)
        inner = func.value
        return inner.attr if isinstance(inner, ast.Attribute) else None
    return None


def _target_name(node: ast.Assign) -> str | None:
    if len(node.targets) != 1:
        return None
    target = node.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _first_arg_name(call: ast.Call) -> str | None:
    if not call.args:
        return None
    first = call.args[0]
    return first.id if isinstance(first, ast.Name) else None


def find_self_residuals(source: str) -> list[tuple[int, str]]:
    """Every ``name = name + call(name, ...)`` in ``source``, as (line, callee)."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.BinOp):
            continue
        if not isinstance(node.value.op, ast.Add):
            continue
        name = _target_name(node)
        if name is None:
            continue
        for side in (node.value.left, node.value.right):
            other = node.value.right if side is node.value.left else node.value.left
            if not (isinstance(side, ast.Name) and side.id == name):
                continue
            if not isinstance(other, ast.Call):
                continue
            if _first_arg_name(other) != name:
                continue
            callee = _callee_name(other)
            if callee:
                found.append((node.lineno, callee))
    return found


def test_every_self_residual_is_classified():
    unlisted = []
    for path in sorted(MODELS.rglob("*.py")):
        for line, callee in find_self_residuals(path.read_text()):
            if f"{path.relative_to(MODELS)}:{callee}" not in ALLOWED:
                unlisted.append(f"{path.relative_to(MODELS)}:{line} -> {callee}")
    assert not unlisted, (
        "Unclassified self-residual(s). If the callee is an engine op, DELETE the "
        "`x = x +` — the op owns its residual and adding it again doubles it "
        "silently. If it is a model-level add, list it in ALLOWED with a reason:\n  "
        + "\n  ".join(unlisted)
    )


def test_the_detector_catches_the_bug_it_was_written_for():
    # Verbatim shape of the two sites that broke 3PTB, including the loop
    # variable the original grep could not see.
    source = (
        "for block in self.pair_transitions:\n"
        "    conditioned = conditioned + block(conditioned)\n"
    )
    assert find_self_residuals(source) == [(2, "block")]


def test_the_detector_sees_through_indexing_and_attributes():
    assert find_self_residuals("x = x + self.op(x)\n") == [(1, "op")]
    assert find_self_residuals("x = x + self.blocks[0](x)\n") == [(1, "blocks")]
    # Reversed operand order is the same residual.
    assert find_self_residuals("x = self.op(x) + x\n") == [(1, "op")]


def test_the_detector_does_not_flag_a_projection_add():
    # `f(y)` does not take x, so it is a feature add, not a self-residual.
    assert find_self_residuals("x = x + self.proj(y)\n") == []
