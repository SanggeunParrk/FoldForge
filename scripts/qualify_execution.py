"""Check graph replays against the same callable with capture disabled.

On an allocated GPU node, activate the environment, then run::

    python scripts/qualify_execution.py af3 --spec target.yaml \
        --config configs/inference/graph-bf16.yaml --out validation/af3

The first two denoiser calls also evaluate the uncaptured callable. With compile
on, that reference is compiled too: compiler-vs-eager differences need a separate
whole-prediction comparison. Cold timings include these diagnostic calls.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import torch
from team_gm.modules.execution import ExecutedCallable, _flatten, _map

from foldforge.models.inference import run


class ReplayCheck:
    """Bounded verification without changing model arithmetic or replay policy."""

    def __init__(self) -> None:
        self.original = ExecutedCallable.__call__
        self.counts: dict[int, int] = {}

    def call(
        self, wrapper: ExecutedCallable, *args: object, **kwargs: object
    ) -> object:
        """Compare outputs from identical tensors at the installed boundary."""
        result = self.original(wrapper, *args, **kwargs)
        if not wrapper.config.cuda_graph:
            message = "Replay qualification requires execution.cuda_graph=true"
            raise ValueError(message)
        count = self.counts.get(id(wrapper), 0)
        if count < 2:
            ref_args, ref_kwargs = _map((args, kwargs), lambda tensor: tensor.clone())
            reference = wrapper.fn(*ref_args, **ref_kwargs)
            _, actual = _flatten(result)
            _, expected = _flatten(reference)
            errors = []
            for value, ref in zip(actual, expected, strict=True):
                torch.testing.assert_close(value, ref, atol=1e-4, rtol=1e-4)
                errors.append((value.float() - ref.float()).abs().max().item())
            print(  # noqa: T201 - explicit diagnostic result
                f"GRAPH CONSISTENCY {wrapper.label} compile={wrapper.config.compile} "
                f"max_abs={max(errors, default=0)}",
                flush=True,
            )
            self.counts[id(wrapper)] = count + 1
        return result


def main() -> int:
    """Run the ordinary CLI with a temporary diagnostic wrapper."""
    if len(sys.argv) < 2:
        message = "usage: qualify_execution.py MODEL [ordinary fold arguments]"
        raise SystemExit(message)
    check = ReplayCheck()

    def checked(wrapper: ExecutedCallable, *args: object, **kwargs: object) -> object:
        return check.call(wrapper, *args, **kwargs)

    with patch.object(ExecutedCallable, "__call__", checked):
        return run(sys.argv[1], sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
