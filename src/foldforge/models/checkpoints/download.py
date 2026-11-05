"""Explicit, hash-checked weight provisioning, independent of inference.

Original releases stay in a cache; only fully verified conversion products are
published into the checkpoint tree. Converters run in a separate interpreter.
"""
# ruff: noqa: PLC0415 - downloading never imports Torch in the caller

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from foldforge import paths
from foldforge.models.checkpoints import DEFAULT_FILES
from foldforge.models.checkpoints.lock import (
    CheckpointMismatchError,
    Pin,
    accepted_pins,
    sha256,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

REFERENCE_URL = "https://github.com/sokrypton/alphafold3.git"
REFERENCE_REVISION = "301430b3cbf9122912663c67eb19628fd6feec3e"
_CATALOG = Path(__file__).with_name("sources.json")


@dataclass(frozen=True)
class Artifact:
    """Immutable source file, with its expected byte count and SHA-256."""

    name: str
    sha256: str
    size: int
    url: str = ""
    repo: str | None = None
    revision: str | None = None


def catalog() -> dict[str, Any]:
    """Read the packaged, revision-pinned source manifest without network I/O."""
    return json.loads(_CATALOG.read_text())


def model_pin(model: str) -> Pin:
    """Resolve a public model name to its locked converted file."""
    return accepted_pins(DEFAULT_FILES[model])[-1]


def _verify(path: Path, artifact: Artifact | Pin) -> None:
    if path.stat().st_size != artifact.size or sha256(path) != artifact.sha256:
        message = f"Checksum mismatch: {path}; expected SHA-256 {artifact.sha256}"
        raise CheckpointMismatchError(message)


def _verify_existing(path: Path, artifact: Artifact | Pin) -> None:
    if isinstance(artifact, Pin) and artifact in accepted_pins(path.name):
        alternatives = [p for p in accepted_pins(path.name) if p.path == artifact.path]
        for candidate in alternatives:
            if (
                path.stat().st_size == candidate.size
                and sha256(path) == candidate.sha256
            ):
                return
    _verify(path, artifact)


@contextlib.contextmanager
def _locked(cache: Path) -> Iterator[None]:
    """Serialize provisioning across processes sharing this checkpoint root."""
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "provision.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _copy(source: Path, destination: Path, artifact: Artifact | Pin) -> None:
    """Publish a checked copy without replacing an existing unknown file."""
    if destination.exists():
        _verify_existing(destination, artifact)
        return
    if isinstance(artifact, Pin):
        for version in accepted_pins(Path(artifact.path).name):
            if (
                version.path == artifact.path
                and source.stat().st_size == version.size
                and sha256(source) == version.sha256
            ):
                artifact = version
                break
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".weights-", dir=destination.parent)
    staging = Path(temporary)
    os.close(descriptor)
    try:
        shutil.copyfile(source, staging)
        _verify(staging, artifact)
        # link is atomic and refuses an existing destination, including a race.
        os.link(staging, destination)
    finally:
        staging.unlink(missing_ok=True)


def _cache_copy(source: Path, destination: Path, artifact: Artifact) -> None:
    """Share immutable cached release bytes where the filesystem permits it."""
    if destination.exists():
        _verify(destination, artifact)
        return
    _verify(source, artifact)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source.resolve(), destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        _copy(source, destination, artifact)


def _fetch(artifact: Artifact, cache: Path, report: Callable[[str], None]) -> Path:
    destination = cache / "downloads" / artifact.sha256 / Path(artifact.name).name
    if destination.exists():
        _verify(destination, artifact)
        return destination
    report(f"Downloading {artifact.name} ({artifact.size / 1e9:.2f} GB)")
    if artifact.repo:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            message = "Install download dependencies: pip install 'foldforge[weights]'"
            raise RuntimeError(message) from error
        source = Path(
            hf_hub_download(
                artifact.repo,
                artifact.name,
                revision=artifact.revision,
                cache_dir=cache / "huggingface",
            )
        )
        _cache_copy(source, destination, artifact)
    else:
        if not artifact.url.startswith("https://"):
            message = "Weight sources must use HTTPS."
            raise ValueError(message)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".partial")
        # Public HTTPS sources receive no Hugging Face credentials. A failed
        # transfer stays uninstalled; retry restarts only this incomplete file.
        request = urllib.request.Request(  # noqa: S310 - HTTPS checked above
            artifact.url, headers={"User-Agent": "foldforge/0.1.0"}
        )
        with (
            urllib.request.urlopen(request, timeout=60) as incoming,  # noqa: S310
            partial.open("wb") as outgoing,
        ):
            shutil.copyfileobj(incoming, outgoing, length=1 << 20)
        _verify(partial, artifact)
        partial.replace(destination)
    return destination


def _reference(cache: Path) -> Path:
    checkout = cache / "converters" / REFERENCE_REVISION
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=checkout.parent) as temporary:
            staging = Path(temporary) / "reference"
            _git("init", str(staging))
            _git(
                "-C",
                str(staging),
                "fetch",
                "--depth=1",
                REFERENCE_URL,
                REFERENCE_REVISION,
            )
            _git("-C", str(staging), "checkout", "--detach", "FETCH_HEAD")
            staging.rename(checkout)
    if (
        _git("-C", str(checkout), "rev-parse", "HEAD").strip() != REFERENCE_REVISION
        or _git(
            "-C", str(checkout), "status", "--porcelain", "--untracked-files=normal"
        ).strip()
    ):
        message = f"Converter checkout is modified: {checkout}; remove it and retry."
        raise RuntimeError(message)
    return checkout


def _git(*arguments: str) -> str:
    return subprocess.run(  # noqa: S603 - argument list, no shell
        ["git", *arguments],  # noqa: S607 - user's Git installation
        check=True,
        text=True,
        capture_output=True,
        timeout=180,
    ).stdout


def _convert(model: str, source: Path, output: Path, cache: Path) -> None:
    with (output / "conversion.log").open("w") as log:
        try:
            _run_converter(model, source, output, cache, log)
        except subprocess.CalledProcessError as error:
            message = (
                f"Conversion failed for {model}; see {output / 'conversion.log'}. "
                "Check dependencies: pip install 'foldforge[weights]'."
            )
            raise RuntimeError(message) from error


def _run_converter(
    model: str, source: Path, output: Path, cache: Path, log: Any
) -> None:
    subprocess.run(  # noqa: S603 - fixed module and explicit arguments
        [
            sys.executable,
            "-m",
            "foldforge.models.checkpoints.convert",
            "--model",
            model,
            "--checkpoint",
            str(source),
            "--reference",
            str(_reference(cache)),
            "--out",
            str(output),
        ],
        check=True,
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def _extras(model: str) -> list[Artifact]:
    return [Artifact(**item) for item in catalog()["models"][model].get("extras", [])]


def _auxiliary(
    model: str,
    root: Path,
    cache: Path,
    source_dir: Path | None,
    report: Callable[[str], None],
) -> None:
    data = catalog()
    for dependency in data["models"][model]["aux"]:
        for item in data["aux"][dependency]:
            artifact = Artifact(**item)
            destination = root / dependency / artifact.name
            if destination.exists():
                _verify(destination, artifact)
                continue
            source = (
                source_dir / dependency / artifact.name
                if source_dir is not None
                else _fetch(artifact, cache, report)
            )
            _copy(source, destination, artifact)


def _from_release(
    model: str, root: Path, cache: Path, report: Callable[[str], None]
) -> None:
    spec = catalog()["models"][model]
    pin = model_pin(model)
    target = root / pin.path
    extras = _extras(model)
    raw = cache / "releases" / model
    for item in spec["files"]:
        artifact = Artifact(**item)
        _cache_copy(_fetch(artifact, cache, report), raw / artifact.name, artifact)
    if spec.get("format") == "converted":
        # First-party releases already in the target record format need no key map.
        source = raw / spec["checkpoint"]
        _verify(source, pin)
        _copy(source, target, pin)
        return
    # Preserve a failed candidate for diagnosis, without exposing it to load_model.
    output = cache / "converted" / model
    output.mkdir(parents=True, exist_ok=True)
    report(f"Converting {model} on CPU; log: {output / 'conversion.log'}")
    _convert(spec["converter"], raw / spec["checkpoint"], output, cache)
    _verify(output / Path(pin.path).name, pin)
    for artifact in extras:
        if (target.parent / artifact.name).exists():
            _verify(target.parent / artifact.name, artifact)
            continue
        source = (
            _fetch(artifact, cache, report)
            if artifact.url or artifact.repo
            else output / artifact.name
        )
        _copy(source, target.parent / artifact.name, artifact)
    _copy(output / Path(pin.path).name, target, pin)


def _install(
    model: str,
    root: Path,
    cache: Path,
    source_dir: Path | None,
    report: Callable[[str], None],
) -> None:
    spec = catalog()["models"][model]
    pin = model_pin(model)
    target = root / pin.path
    extras = _extras(model)
    if target.exists():
        _verify_existing(target, pin)
    complete = target.exists() and all(
        (target.parent / item.name).exists() or item.url or item.repo for item in extras
    )
    if complete:
        pass
    elif source_dir is not None:
        for artifact in extras:
            _copy(
                source_dir / target.parent.name / artifact.name,
                target.parent / artifact.name,
                artifact,
            )
        _copy(source_dir / pin.path, target, pin)
    elif spec["reason"]:
        raise RuntimeError(spec["reason"])
    else:
        _from_release(model, root, cache, report)
    # Existing blobs can still need their LM sidecar or shared language model.
    for artifact in extras:
        destination = target.parent / artifact.name
        if destination.exists():
            _verify(destination, artifact)
        elif source_dir is not None:
            _copy(
                source_dir / target.parent.name / artifact.name, destination, artifact
            )
        elif artifact.url or artifact.repo:
            _copy(_fetch(artifact, cache, report), destination, artifact)
        else:
            message = f"Missing sidecar {destination}; supply it with --from-dir."
            raise FileNotFoundError(message)
    _auxiliary(model, root, cache, source_dir, report)


@dataclass(frozen=True)
class DownloadResult:
    """Per-model completion, including dependency checks and useful errors."""

    model: str
    installed: bool
    message: str


def download_weights(
    models: Sequence[str],
    *,
    root: Path | None = None,
    source_dir: Path | None = None,
    report: Callable[[str], None] = lambda _: None,
) -> list[DownloadResult]:
    """Provision models explicitly, continuing after per-model failures.

    ``source_dir`` is an authorized local checkpoint tree in the locked layout;
    it is copied and verified without accessing the network. No existing weight
    is replaced. Inference never calls this function implicitly.
    """
    names = list(dict.fromkeys(models))
    unknown = set(names) - catalog()["models"].keys()
    if unknown:
        message = f"Unknown models: {', '.join(sorted(unknown))}"
        raise ValueError(message)
    root = (root or paths.checkpoints()).expanduser().resolve()
    source_dir = source_dir.expanduser().resolve() if source_dir is not None else None
    results = []
    with _locked(root / ".weights"):
        for model in names:
            report(f"Checking {model}")
            try:
                _install(model, root, root / ".weights", source_dir, report)
            except (
                OSError,
                RuntimeError,
                ValueError,
                subprocess.SubprocessError,
            ) as error:
                results.append(
                    DownloadResult(model, installed=False, message=str(error))
                )
            else:
                results.append(
                    DownloadResult(
                        model,
                        installed=True,
                        message="Verified weights and dependencies",
                    )
                )
    return results
