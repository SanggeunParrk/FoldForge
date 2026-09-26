"""List predictors and run their model-specific inference commands."""

import argparse
from importlib import import_module

from foldforge.models import describe, known_models, registered_models


def main(argv: list[str] | None = None) -> int:
    """Dispatch lazily so listing models never imports the GPU stack."""
    parser = argparse.ArgumentParser(prog="foldforge")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("models", help="List implemented and planned predictors")
    ccd = commands.add_parser("ccd", help="Prepare or verify the shared CCD database")
    ccd.add_argument("arguments", nargs=argparse.REMAINDER)
    checkpoints = commands.add_parser(
        "checkpoints", help="Verify converted checkpoints against checkpoints.lock"
    )
    checkpoints.add_argument("arguments", nargs=argparse.REMAINDER)
    validate = commands.add_parser(
        "validate", help="Judge FoldForge folds against the releases' outputs/"
    )
    validate.add_argument("arguments", nargs=argparse.REMAINDER)
    fold = commands.add_parser("fold", help="Run a predictor")
    fold.add_argument("model", choices=known_models())
    fold.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "ccd":
        return import_module("foldforge.data.ccd.prepare").main(args.arguments)
    if args.command == "checkpoints":
        return import_module("foldforge.models.checkpoints.lock").main(args.arguments)
    if args.command == "validate":
        return import_module("foldforge.eval.references").main(args.arguments)
    if args.command == "models":
        available = set(registered_models())
        for name in known_models():
            status = "implemented" if name in available else "not ported"
            print(f"{name}\t{status}\t{describe(name)}")  # noqa: T201
        return 0
    if args.model not in registered_models():
        parser.error(f"{args.model} is not ported yet")
    if "--spec" in args.arguments or any(
        x.startswith("--spec=") for x in args.arguments
    ):
        return import_module("foldforge.models.inference").run(
            args.model, args.arguments
        )
    return import_module("foldforge.models.io.cli").run(args.model, args.arguments)


if __name__ == "__main__":
    # `python -m foldforge.cli` must do what the `foldforge` script does.
    # Without this it exits 0 having run nothing, which reads as a fold that
    # produced no output rather than as a command that never started.
    raise SystemExit(main())
