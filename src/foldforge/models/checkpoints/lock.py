"""Pin every converted checkpoint the code was validated with.

``checkpoints.lock`` (beside this file, tracked in git) lists each converted
blob's SHA-256, size and path under ``model_checkpoints/``. The code and the
blobs move together: when a converter changes a blob's layout -- Chai-1's
relative encoding grew nine columns on 2026-09-25 -- a machine that still
holds the old blob must fail at load with a message that says so, not deep in
a strict load with a shape error, or silently if the shapes happen to agree.

Loading checks the size, which is free; ``foldforge checkpoints verify``
checks the hashes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

LOCK = Path(__file__).with_name("checkpoints.lock")


@dataclass(frozen=True)
class Pin:
    """One locked blob."""

    sha256: str
    size: int
    path: str  # relative to model_checkpoints/


def pins() -> dict[str, Pin]:
    """Every locked blob, by its file name."""
    out = {}
    for line in LOCK.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        sha, size, path = line.split()
        out[Path(path).name] = Pin(sha, int(size), path)
    return out


class CheckpointMismatchError(RuntimeError):
    """A blob that is not the one this code was validated with."""


def check_size(path: Path) -> None:
    """Refuse a locked blob whose size is not the locked one."""
    path = path.resolve()  # a release alias is a symlink to its blob
    pin = pins().get(path.name)
    if pin is None or not path.is_file():
        return
    size = path.stat().st_size
    if size != pin.size:
        message = (
            f"{path} is {size} bytes; this code was validated with the "
            f"{pin.size}-byte blob (sha256 {pin.sha256[:16]}...). The blob is "
            "from another version of the converter: regenerate it with "
            "scripts/convert_dense_checkpoint.py or copy the matching one."
        )
        raise CheckpointMismatchError(message)


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    """Stream a file's SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path, *, quick: bool = False) -> list[str]:
    """Problems with the blobs under ``root``: missing, resized or rehashed."""
    problems = []
    for pin in pins().values():
        path = root / pin.path
        if not path.is_file():
            problems.append(f"missing  {pin.path}")
        elif path.stat().st_size != pin.size:
            problems.append(f"size     {pin.path}")
        elif not quick and sha256(path) != pin.sha256:
            problems.append(f"sha256   {pin.path}")
    return problems


def main(argv: list[str]) -> int:
    """``foldforge checkpoints verify [--quick] [--root DIR]``."""
    import argparse  # noqa: PLC0415

    from foldforge.models.checkpoints import DEFAULT_DIR  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="foldforge checkpoints")
    parser.add_argument("action", choices=["verify"])
    parser.add_argument("--quick", action="store_true", help="sizes only")
    parser.add_argument("--root", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args(argv)
    problems = verify(args.root, quick=args.quick)
    for problem in problems:
        print(problem)  # noqa: T201 - CLI output contract
    if not problems:
        print(f"{len(pins())} checkpoints match checkpoints.lock")  # noqa: T201
    return 1 if problems else 0
