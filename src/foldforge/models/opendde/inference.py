"""opendde inference command."""

from __future__ import annotations

from foldforge.models.af_inference import run


def main(argv: list[str] | None = None) -> int:
    """Run the model-specific command."""
    return run("opendde", argv)


if __name__ == "__main__":
    raise SystemExit(main())
