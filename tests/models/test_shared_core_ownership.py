"""Model adapters must not bring back independent kernel or FoldCP implementations."""

import ast
from pathlib import Path

from team_gm.modules.checkpoints import stacks, transformer

import foldforge

ROOT = Path(foldforge.__file__).parent


def test_removed_model_layer_packages_stay_removed():
    for family in ("protenix", "opendde"):
        model = ROOT / "models" / family / "ported" / "model"
        assert not (model / "triangular" / "layers.py").exists()
        for name in (
            "primitives",
            "transformer",
            "pairformer",
            "diffusion",
            "confidence",
            "embedders",
            "head",
        ):
            assert not (model / "modules" / f"{name}.py").exists()
    assert not (ROOT / "models/opendde/ported/distributed").exists()


def test_adapter_source_has_no_retired_execution_hooks():
    for path in ROOT.rglob("*.py"):
        source = path.read_text()
        assert "foldcp" not in source.lower(), path
        assert "fold-cp" not in source.lower(), path
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ClassDef) and any(
                f"models/{family}/" in path.as_posix()
                for family in ("protenix", "opendde")
            ):
                assert node.name not in {
                    "AttentionPairBias",
                    "DiffusionTransformer",
                    "ConditionedTransitionBlock",
                    "PairformerStack",
                    "MSABlock",
                }, path


def test_shared_checkpoint_classes_have_one_owner():
    assert (
        transformer.AttentionPairBias.__module__
        == "team_gm.modules.checkpoints.transformer"
    )
    assert stacks.PairformerStack.__module__ == "team_gm.modules.checkpoints.stacks"


def test_common_runtime_has_no_distributed_model_hooks():
    import inspect

    from team_gm.diffusion.edm.sampling import EulerSampler
    from team_gm.diffusion.guidance.compat import TFGEngine
    from team_gm.diffusion.guidance.engine import GuidanceEngine

    for implementation in (EulerSampler, GuidanceEngine, TFGEngine):
        source = inspect.getsource(implementation).lower()
        assert "foldcp" not in source
        assert "fold-cp" not in source
        assert "rank_action_synchronizer" not in source
        assert "synchronize_action" not in source
