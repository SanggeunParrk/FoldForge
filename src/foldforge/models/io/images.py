"""Common, opt-in diagnostic PNGs from decoded predictions and input features.

No model imports or global pyplot state. Rendering runs after measured inference.
Distance-bin indices are deliberately not mislabeled as distances in angstroms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from foldforge.models.io.output import Decoded

MAX_MSA_ROWS = 1024


def as_array(value: Any) -> np.ndarray:
    """Copy GPU/BF16 tensors to a NumPy-compatible host representation."""
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def input_images(
    features: dict[str, Any],
    selected: tuple[str, ...],
    *,
    gap_id: int,
    batched: bool = False,
) -> dict[str, Any]:
    """Snapshot input coverage before forward, excluding gaps and padding.

    MSA is the prepared input alignment, before any model-internal row sampling.
    Template coverage marks query tokens with any resolved atom;
    ligand reference conformers are not treated as structural templates.
    """
    result: dict[str, Any] = {}
    if not {"msa", "template"}.intersection(selected):
        return result

    def array(key: str) -> np.ndarray | None:
        value = features.get(key)
        if value is None:
            return None
        value = as_array(value)
        return value[0] if batched else value

    token_mask = array("seq_mask")
    if token_mask is None:
        token_mask = array("token_attention_mask")
    if token_mask is not None:
        token_mask = token_mask.astype(bool)

    if "msa" in selected:
        msa = array("msa")
        valid = array("msa_mask")
        if valid is None:
            valid = array("msa_attention_mask")
        result.update(_msa_images(msa, valid, token_mask, gap_id))
    if "template" in selected:
        mask = array("template_atom_mask")
        result.update(_template_images(mask, token_mask))
    return result


def _template_images(
    mask: np.ndarray | None, token_mask: np.ndarray | None
) -> dict[str, Any]:
    if mask is None or mask.ndim != 3:  # noqa: PLR2004
        return {}
    coverage = mask.astype(bool).any(axis=-1)
    if token_mask is not None:
        coverage = coverage[:, token_mask]
    coverage = coverage[coverage.any(axis=1)]
    return {"template": coverage.astype(float)} if coverage.size else {}


def _msa_images(
    msa: np.ndarray | None,
    valid: np.ndarray | None,
    token_mask: np.ndarray | None,
    gap_id: int,
) -> dict[str, Any]:
    if msa is None or msa.ndim != 2 or not msa.size:  # noqa: PLR2004
        return {}
    valid = np.ones_like(msa, dtype=bool) if valid is None else valid.astype(bool)
    if token_mask is not None:
        msa, valid = msa[:, token_mask], valid[:, token_mask]
    rows = valid.any(axis=1)
    msa, valid = msa[rows], valid[rows]
    if not msa.size:
        return {}
    covered = valid & (msa != gap_id)
    return {
        "msa_coverage": covered.sum(axis=0),
        "msa_rows": int(msa.shape[0]),
        "msa": np.where(
            covered[:MAX_MSA_ROWS],
            (msa[:MAX_MSA_ROWS] == msa[0]).astype(float),
            np.nan,
        ),
    }


@dataclass(frozen=True)
class Plot:
    """One model-independent chart with explicit units and axes."""

    values: np.ndarray
    title: str
    ylabel: str
    units: str = ""
    vmax: float | None = None


def _render(plot: Plot, path: Path) -> None:
    # Import only on opt-in: headless figures do not consume model RNG state.
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: PLC0415
    from matplotlib.figure import Figure  # noqa: PLC0415

    figure = Figure(figsize=(8, 6), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    values = plot.values
    if values.ndim == 1:
        axes.plot(np.arange(1, len(values) + 1), values, linewidth=1)
        axes.set_ylim(bottom=0, top=plot.vmax)
        axes.grid(alpha=0.2)
    else:
        cmap = "viridis" if plot.units != "angstrom" else "viridis_r"
        im = axes.imshow(
            np.ma.masked_invalid(values),
            origin="upper",
            interpolation="nearest",
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=plot.vmax,
            extent=(0.5, values.shape[1] + 0.5, values.shape[0] + 0.5, 0.5),
        )
        figure.colorbar(im, ax=axes, label=plot.units)
    axes.set(xlabel="Token index (1-based)", ylabel=plot.ylabel, title=plot.title)
    figure.savefig(path, dpi=160)
    figure.clear()


def _prediction_plots(result: Decoded, name: str) -> list[tuple[str, Plot]]:
    if name == "distogram":
        logits = result.image_distogram
        if logits is None:
            logits = result.prediction.distogram_logits
        if logits is None:
            return []
        if logits.ndim != 3 or logits.shape[0] != logits.shape[1]:  # noqa: PLR2004
            message = "Distogram image requires (tokens, tokens, bins) logits"
            raise ValueError(message)
        finite = (
            bool(logits.isfinite().all())
            if hasattr(logits, "isfinite")
            else bool(np.isfinite(logits).all())
        )
        if not finite:
            message = "Non-finite distogram logits"
            raise FloatingPointError(message)
        # Reduce on the originating device before copying: no N*N*bins host copy.
        values = as_array(logits.argmax(-1))
        return [
            (
                "distogram",
                Plot(
                    values,
                    "Distogram: most likely distance bin",
                    "Token index (1-based)",
                    "Bin index (0-based)",
                ),
            )
        ]
    value = getattr(result.prediction, name, None)
    if value is None:
        return []
    values = as_array(value)
    plots = []
    for sample, sample_values in enumerate(values):
        if name == "plddt":
            plot = Plot(
                sample_values * 100,
                f"pLDDT — sample {sample}",
                "pLDDT (0-100)",
                vmax=100,
            )
        else:
            plot = Plot(
                sample_values,
                f"{name.upper()} — sample {sample}",
                "Token index (1-based)",
                "angstrom",
                vmax=float(values.max()),
            )
        plots.append((f"sample-{sample:03d}-{name}", plot))
    return plots


def write_images(
    result: Decoded, directory: Path, selected: tuple[str, ...]
) -> dict[str, Any]:
    """Save requested plots and enumerate unavailable heads/inputs in JSON."""
    from foldforge.models.config.runtime import IMAGE_KINDS  # noqa: PLC0415

    requested = IMAGE_KINDS if "all" in selected else tuple(dict.fromkeys(selected))
    if set(requested) - set(IMAGE_KINDS):
        message = "Unknown diagnostic image selection"
        raise ValueError(message)
    root = directory / "images" / result.name
    # Resolve before mkdir so an existing symlink cannot redirect output.
    if not root.resolve().is_relative_to(directory.resolve()):
        message = "Image destination must remain inside the output directory"
        raise ValueError(message)
    saved: dict[str, list[str]] = {}
    skipped: dict[str, str] = {}
    for name in requested:
        plots = []
        if name in {"msa", "template"}:
            values = result.image_inputs.get(name)
            if values is not None:
                title = (
                    "Template coverage (resolved atoms)"
                    if name == "template"
                    else "Input MSA: query identity (gaps/padding blank)"
                )
                plots.append(
                    (
                        f"input-{name}",
                        Plot(
                            values,
                            title,
                            "Template row" if name == "template" else "MSA row",
                            "Coverage" if name == "template" else "Match to query",
                            vmax=1,
                        ),
                    )
                )
                if name == "msa":
                    total = result.image_inputs["msa_rows"]
                    plots.append(
                        (
                            "input-msa-coverage",
                            Plot(
                                result.image_inputs["msa_coverage"],
                                f"Input MSA coverage — {total} rows",
                                "Non-gap sequences",
                            ),
                        )
                    )
        else:
            plots = _prediction_plots(result, name)
        if not plots:
            skipped[name] = (
                "No valid input templates"
                if name == "template"
                else "Not provided by this prediction/input"
            )
            continue
        root.mkdir(parents=True, exist_ok=True)
        saved[name] = []
        for filename, plot in plots:
            path = root / f"{filename}.png"
            _render(plot, path)
            saved[name].append(str(path.relative_to(directory)))
    return {
        "requested": list(requested),
        "saved": saved,
        "skipped": skipped,
        "msa_total_rows": result.image_inputs.get("msa_rows"),
        "msa_display_limit": MAX_MSA_ROWS,
        "msa_stage": "prepared input, before model-internal sampling",
        "distogram_representation": "argmax distance-bin index (not angstroms)",
    }
