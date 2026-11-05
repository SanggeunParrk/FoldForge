"""Lazy command-line weight management; help never imports Torch."""
# ruff: noqa: PLC0415 - keep command discovery lightweight

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import rich_click as click


@click.group()
def weights() -> None:
    """Download, convert and verify model weights explicitly."""


@weights.command("list")
@click.option("--json", "as_json", is_flag=True, help="Show pinned sources and hashes.")
def list_weights(*, as_json: bool) -> None:
    """List automatic and local-import models and required language models."""
    from foldforge.models.checkpoints.download import catalog, model_pin

    data = catalog()
    if as_json:
        click.echo(json.dumps(data, indent=2))
        return
    from rich.console import Console
    from rich.table import Table

    table = Table(title="FoldForge weight sources")
    for name in ("Model", "Provisioning", "Blob GB", "Additional weights"):
        table.add_column(name)
    for model, spec in data["models"].items():
        extra = [*spec["aux"], *(item["name"] for item in spec.get("extras", []))]
        table.add_row(
            model,
            "local import"
            if spec["reason"]
            else (
                "download"
                if spec.get("format") == "converted"
                else "download + convert"
            ),
            f"{model_pin(model).size / 1e9:.2f}",
            ", ".join(extra) or "—",
        )
    Console().print(table)
    click.echo("See docs/guides/WEIGHTS.md for access and conversion requirements.")


@weights.command("download")
@click.argument("models", nargs=-1)
@click.option("--all", "all_models", is_flag=True, help="Attempt all twelve models.")
@click.option(
    "--all-public", is_flag=True, help="Select all automatically available releases."
)
@click.option(
    "--dry-run", is_flag=True, help="Print the plan without I/O or downloads."
)
@click.option(
    "--root",
    type=click.Path(path_type=Path, file_okay=False),
    help="Checkpoint destination; default: FOLDFORGE_CHECKPOINT_DIR or cache.",
)
@click.option(
    "--from-dir",
    "source_dir",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    help="Import an authorized locked checkpoint tree without network access.",
)
def download(
    models: tuple[str, ...],
    *,
    all_models: bool,
    all_public: bool,
    dry_run: bool,
    root: Path | None,
    source_dir: Path | None,
) -> None:
    """Fetch MODELS and their language models, convert and check SHA-256.

    Examples: foldforge weights download boltz2 opendde

    foldforge weights download --all --dry-run
    """
    from foldforge import paths
    from foldforge.models.checkpoints.download import (
        catalog,
        download_weights,
    )

    data = catalog()
    selected = _select_models(
        data, models, all_models=all_models, all_public=all_public
    )
    destination = (root or paths.checkpoints()).expanduser().resolve()
    click.echo(f"Checkpoint root: {destination}")
    if root is not None:
        click.echo(f"For inference, set FOLDFORGE_CHECKPOINT_DIR={destination}")
    if dry_run:
        _print_plan(selected, data, source_dir=source_dir)
        return
    try:
        results = download_weights(
            selected, root=destination, source_dir=source_dir, report=click.echo
        )
    except OSError as error:
        raise click.ClickException(str(error)) from error
    for result in results:
        click.echo(
            f"{'OK' if result.installed else 'FAILED'} {result.model}: {result.message}"
        )
    failures = sum(not result.installed for result in results)
    if failures:
        message = f"{failures}/{len(results)} model(s) could not be provisioned."
        raise click.ClickException(message)


def _select_models(
    data: dict[str, Any], models: tuple[str, ...], *, all_models: bool, all_public: bool
) -> list[str]:
    if (all_models and all_public) or (models and (all_models or all_public)):
        message = "Choose model names, --all or --all-public."
        raise click.UsageError(message)
    selected = list(dict.fromkeys(models))
    if all_models:
        selected = list(data["models"])
    elif all_public:
        selected = [name for name, spec in data["models"].items() if not spec["reason"]]
    if not selected:
        message = (
            "Provide model names, --all or --all-public; see foldforge weights list."
        )
        raise click.UsageError(message)
    unknown = set(selected) - data["models"].keys()
    if unknown:
        message = (
            f"Unknown models: {', '.join(sorted(unknown))}; see foldforge weights list."
        )
        raise click.UsageError(message)
    return selected


def _print_plan(
    selected: list[str], data: dict[str, Any], *, source_dir: Path | None
) -> None:
    from foldforge.models.checkpoints.download import model_pin

    artifacts = {}
    for model in selected:
        spec = data["models"][model]
        action = _plan_action(spec, source_dir=source_dir)
        click.echo(f"{model}: {action} -> {model_pin(model).path}")
        if not spec["reason"]:
            for artifact in spec["files"]:
                artifacts[artifact["sha256"]] = artifact
        for artifact in spec.get("extras", []):
            if artifact.get("url") or artifact.get("repo"):
                artifacts[artifact["sha256"]] = artifact
        for dependency in spec["aux"]:
            click.echo(f"  requires shared {dependency}")
            for artifact in data["aux"][dependency]:
                artifacts[artifact["sha256"]] = artifact
    size = sum(item["size"] for item in artifacts.values()) / 1e9
    if source_dir is None:
        click.echo(f"Pinned source files: {size:.2f} GB (before cache reuse).")
        click.echo("Allow extra space for caches, conversion and installed weights.")


def _plan_action(spec: dict[str, Any], *, source_dir: Path | None) -> str:
    if source_dir is not None:
        return "local import"
    if spec["reason"]:
        return f"BLOCKED: {spec['reason']}"
    return (
        "download (already converted)"
        if spec.get("format") == "converted"
        else "download + convert"
    )
