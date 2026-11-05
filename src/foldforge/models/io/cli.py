"""Compatibility entry points backed by the single public fold parser."""
# ruff: noqa: PLC0415 - compatibility entry points share lazy execution

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foldforge.models.io.request import Request


def parse(model: str, argv: list[str] | None = None) -> Request:
    """Resolve a legacy JSON request without accessing inputs or checkpoints."""
    import sys

    import rich_click as click

    from foldforge.data.ccd import default_path
    from foldforge.models.inference import resolve_cli_config
    from foldforge.models.io.options import parse as parse_options
    from foldforge.models.io.request import Request

    arguments = parse_options(model, list(sys.argv[1:] if argv is None else argv))
    if arguments.input is None or arguments.checkpoint is None:
        failure = click.UsageError(
            "This compatibility parser requires --input and --checkpoint."
        )
        failure.show()
        raise SystemExit(failure.exit_code)
    try:
        return Request.from_config(
            resolve_cli_config(arguments),
            model=model,
            ccd_db=arguments.ccd_db or default_path(),
            input=arguments.input,
            checkpoint=arguments.checkpoint,
            device=arguments.device,
            out=arguments.out,
            samples=arguments.samples if arguments.samples is not None else 5,
        )
    except ValueError as error:
        failure = click.UsageError(str(error))
        failure.show()
        raise SystemExit(failure.exit_code) from error


def run(model: str, argv: list[str] | None = None) -> int:
    """Compatibility execution uses the shared CLI/library path."""
    import sys

    from foldforge.models.inference import run as predict

    return predict(model, list(sys.argv[1:] if argv is None else argv))
