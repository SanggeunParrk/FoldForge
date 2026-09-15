"""Inventory model classes and their framework boundaries without importing Torch."""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path


def inventory(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "src/foldforge").rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if not any(f"/{part}/" in relative for part in ("models", "modules")):
            continue
        tree = ast.parse(path.read_text())
        imports = [
            n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
        ]
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            forward = next(
                (
                    n
                    for n in node.body
                    if isinstance(n, ast.FunctionDef) and n.name == "forward"
                ),
                None,
            )
            if forward is None:
                continue
            args = [
                a
                for a in forward.args.args + forward.args.kwonlyargs
                if a.arg != "self"
            ]
            config = any(
                isinstance(n, ast.ClassDef) and n.name == "Config" for n in node.body
            )
            bases = ",".join(ast.unparse(b) for b in node.bases)
            name = node.name
            if "/architectures/" in relative:
                status = "terminal model or explicit checkpoint adapter"
            elif name == "PairformerBlock":
                status = "reference only; loader replaces composition with team-gm"
            elif name in {"ConditionedTransitionBlock", "DiffusionTransition"}:
                status = (
                    "conditioned instances converted to engine after strict loading"
                )
            elif name == "AdaptiveLayerNorm":
                status = "conditioned instances converted; unconditioned norm retained"
            elif any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id
                in {
                    "conditioned_residual",
                    "msa_pair_update",
                    "msa_row_update",
                    "template_embedding_mean",
                    "outer_product_projection",
                    "af3_outer_product_mean",
                }
                for n in ast.walk(node)
            ):
                status = "team-gm composition; released signatures/layouts retained"
            elif any(
                k in name
                for k in [
                    "DiffusionTransformer",
                    "ConditionedTransition",
                    "AdaptiveLayerNorm",
                    "Triangle",
                    "Attention",
                    "OuterProduct",
                    "Transition",
                    "MSA",
                ]
            ):
                status = "shared candidate; upstream composition/equation remains"
            else:
                status = "model-specific or source utility; retained"
            rows.append(
                {
                    "path": relative,
                    "line": node.lineno,
                    "class": name,
                    "bases": bases,
                    "nested_config": config,
                    "typed_forward": bool(
                        forward.returns is not None
                        and all(a.annotation is not None for a in args)
                    ),
                    "imports_team_gm": any(v.startswith("team_gm") for v in imports),
                    "imports_engine": any(
                        v.startswith("miniworld_engine") for v in imports
                    ),
                    "status": status,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = inventory(root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} forward-bearing classes inventoried")  # noqa: T201


if __name__ == "__main__":
    main()
