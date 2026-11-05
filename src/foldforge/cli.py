"""Human-friendly commands over FoldForge's public Python prediction API."""
# ruff: noqa: PLC0415 - optional commands remain lazy

from __future__ import annotations

import json
from importlib import import_module
from typing import Any

import rich_click as click

from foldforge.models import describe, known_models, registered_models
from foldforge.models.checkpoints.commands import weights
from foldforge.models.io.options import FoldArguments, fold_command


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]}, no_args_is_help=True
)
@click.version_option(package_name="foldforge")
def app() -> None:
    """FoldForge — one prediction API, many structure models.

    Predict: foldforge fold MODEL --spec input.yaml --config config.yaml
    Python:  import foldforge as ff; model = ff.load_model(MODEL)
    """


app.add_command(fold_command)
app.add_command(weights)


@app.result_callback()
def dispatch(result: Any, **_: Any) -> int:
    """Separate parsing from library execution."""
    if isinstance(result, FoldArguments):
        import yaml

        from foldforge.models.inference import execute

        try:
            return execute(result)
        except (OSError, ValueError, yaml.YAMLError) as error:
            raise click.ClickException(str(error)) from error
    return result if isinstance(result, int) else 0


@app.command("models")
@click.option(
    "--json", "as_json", is_flag=True, help="Print structured model metadata."
)
@click.option("--plain", is_flag=True, help="Print the original tab-separated listing.")
def models(*, as_json: bool, plain: bool) -> int:
    """List models without loading Torch or weights."""
    if as_json and plain:
        message = "Choose --json or --plain."
        raise click.UsageError(message)
    registered = set(registered_models())
    rows = [
        {
            "model": name,
            "implemented": name in registered,
            "description": describe(name),
        }
        for name in known_models()
    ]
    if as_json:
        click.echo(json.dumps(rows, indent=2))
    elif plain:
        for row in rows:
            status = "implemented" if row["implemented"] else "not ported"
            click.echo(f"{row['model']}\t{status}\t{row['description']}")
    else:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="FoldForge models", header_style="bold cyan")
        table.add_column("Model", style="bold")
        table.add_column("Status")
        table.add_column("Description")
        for row in rows:
            table.add_row(
                row["model"],
                "implemented" if row["implemented"] else "not ported",
                row["description"],
            )
        Console().print(table)
    return 0


def _utility(name: str, module: str, help_text: str) -> None:
    """Forward admin commands without reparsing their flags."""

    @app.command(
        name,
        help=help_text,
        add_help_option=False,
        context_settings={"ignore_unknown_options": True},
    )
    @click.argument("arguments", nargs=-1, type=click.UNPROCESSED)
    def utility(arguments: tuple[str, ...]) -> int:
        return import_module(module).main(list(arguments))


_utility("ccd", "foldforge.data.ccd.prepare", "Prepare or verify the CCD database.")
_utility(
    "checkpoints", "foldforge.models.checkpoints.lock", "Verify converted checkpoints."
)
_utility(
    "validate",
    "foldforge.eval.references",
    "Compare predictions with release references.",
)


def main(argv: list[str] | None = None) -> int:
    """Console and module entry point with concise user errors."""
    try:
        result = app.main(args=argv, prog_name="foldforge", standalone_mode=False)
        return result if isinstance(result, int) else 0
    except click.ClickException as error:
        error.show()
        return error.exit_code
    except click.Abort:
        click.echo("Aborted.", err=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
