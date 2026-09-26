"""Judge FoldForge folds against the released implementations' own outputs.

``outputs/`` holds, for a few reference targets, each family's structures
from its RELEASED code (``scripts/references``): three seeds of five samples
under shared conditions. A FoldForge run of the same family, target and
seeds is judged against them on four things, each relative to the release's
own seed-to-seed variation rather than to a fixed number, because a fold with
no MSA can be as chaotic in the release as in any port:

- structure: mean pairwise polymer RMSD (CA and C1'), ours to release, against
  the release's own spread;
- confidence: mean pLDDT, against the spread of the release's seed means;
- ligand geometry: each ligand's pairwise-distance spectrum against its CCD
  ideal coordinates, name-free (a release may name ligand atoms its own way);
- ligand and ion placement: centroids after superposing the polymers, matched
  across identical components by the best permutation.

Protein metrics alone missed five real faults on 2026-09-25; ligand geometry
is judged separately for that reason.
"""

from __future__ import annotations

import gzip
import itertools
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = ROOT / "outputs"
TARGETS = ("5i28", "3ptb", "1a1k")
SEEDS = (0, 1, 2)

#: How FoldForge folds each family under the references' conditions: the CLI
#: model, its trunk passes as the config counts them, and the release variant.
FAMILIES: dict[str, dict[str, Any]] = {
    "boltz2": {"model": "boltz2", "recycles": 3},
    "chai1": {"model": "chai1", "recycles": 3},
    "protenix1": {
        "model": "protenix",
        "recycles": 4,
        "variant": "protenix_base_default_v1.0.0",
    },
    "protenix2": {"model": "protenix", "recycles": 4, "variant": "protenix-v2"},
    "openfold3": {"model": "openfold3", "recycles": 3},
    "openfold3-preview2": {"model": "openfold3-preview2", "recycles": 3},
    "rosettafold3": {"model": "rosettafold3", "recycles": 4},
    "intellifold2": {"model": "intellifold2", "recycles": 3},
    "opendde": {"model": "opendde", "recycles": 4},
    "esmfold2": {"model": "esmfold2", "recycles": 3},
}

POLYMER_ATOMS = ("CA", "C1'")


@dataclass
class Structure:
    """What the judgement reads from one CIF."""

    polymer: dict[str, np.ndarray] = field(default_factory=dict)
    #: chain -> (elements, coordinates) for non-polymer chains
    ligands: dict[str, tuple[tuple[str, ...], np.ndarray]] = field(default_factory=dict)
    ligand_names: dict[str, str] = field(default_factory=dict)
    plddt: float = float("nan")


def read_cif(path: Path, plddt: float | None = None) -> Structure:
    """Parse the first model's atom_site loop, tolerant of short rows."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        text = fh.read()
    cols: list[str] = []
    polymer: dict[str, list] = {}
    other: dict[str, list] = {}
    names: dict[str, str] = {}
    bfactors: list[float] = []
    for line in text.splitlines():
        if line.startswith("_atom_site."):
            cols.append(line.strip().split(".", 1)[1])
            continue
        if not cols or not line.startswith(("ATOM", "HETATM")):
            continue
        row = shlex.split(line)
        ix = {c: i for i, c in enumerate(cols)}
        model = ix.get("pdbx_PDB_model_num")
        if model is not None and len(row) > model and row[model] != "1":
            continue
        chain = row[ix.get("label_asym_id", ix.get("auth_asym_id"))]
        atom = row[ix["label_atom_id"]].strip('"')
        element = row[ix["type_symbol"]].upper() if "type_symbol" in ix else atom[0]
        xyz = [float(row[ix[k]]) for k in ("Cartn_x", "Cartn_y", "Cartn_z")]
        if len(row) > ix["B_iso_or_equiv"]:  # Chai-lab writes short ion rows
            bfactors.append(float(row[ix["B_iso_or_equiv"]]))
        if element == "H":
            continue
        is_polymer = row[0] == "ATOM" and not (atom == "CA" and element == "CA")
        if is_polymer:
            if atom in POLYMER_ATOMS:
                polymer.setdefault(chain, []).append(xyz)
        else:
            other.setdefault(chain, []).append((element, xyz))
            names[chain] = row[ix["label_comp_id"]]
    b = np.asarray(bfactors)
    structure = Structure(
        polymer={c: np.asarray(v) for c, v in polymer.items()},
        ligands={
            c: (tuple(e for e, _ in v), np.asarray([x for _, x in v]))
            for c, v in other.items()
            if c not in polymer
        },
        ligand_names={c: n for c, n in names.items() if c not in polymer},
    )
    if plddt is not None:
        structure.plddt = plddt
    elif b.size:
        structure.plddt = float(b.mean() * (100 if b.max() <= 1 else 1))
    return structure


def _kabsch(mobile: np.ndarray, target: np.ndarray):
    mc, tc = mobile.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((mobile - mc).T @ (target - tc))
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1, 1, d]) @ vt
    return lambda x: (x - mc) @ rotation + tc


def _polymer_coords(a: Structure, b: Structure) -> tuple[np.ndarray, np.ndarray]:
    chains = [c for c in a.polymer if c in b.polymer]
    pa, pb = [], []
    for c in chains:
        n = min(len(a.polymer[c]), len(b.polymer[c]))
        pa.append(a.polymer[c][:n])
        pb.append(b.polymer[c][:n])
    return np.concatenate(pa), np.concatenate(pb)


def polymer_rmsd(a: Structure, b: Structure) -> float:
    """Polymer CA/C1' RMSD after superposing ``a`` onto ``b``."""
    pa, pb = _polymer_coords(a, b)
    moved = _kabsch(pa, pb)(pa)
    return float(np.sqrt(((moved - pb) ** 2).sum(1).mean()))


def placement(a: Structure, b: Structure) -> float | None:
    """Mean ligand/ion centroid distance after polymer superposition.

    Components are matched only within identical element compositions, by the
    permutation that minimises the mean -- identical ions are interchangeable.
    """
    if not a.ligands or not b.ligands:
        return None
    pa, pb = _polymer_coords(a, b)
    onto = _kabsch(pa, pb)
    groups: dict[tuple[str, ...], tuple[list, list]] = {}
    for elements, xyz in a.ligands.values():
        groups.setdefault(tuple(sorted(elements)), ([], []))[0].append(
            onto(xyz).mean(0)
        )
    for elements, xyz in b.ligands.values():
        key = tuple(sorted(elements))
        if key in groups:
            groups[key][1].append(xyz.mean(0))
    distances = []
    for mine, theirs in groups.values():
        n = min(len(mine), len(theirs))
        if n == 0:
            continue
        mine_a, theirs_a = np.asarray(mine), np.asarray(theirs)
        best = min(
            np.linalg.norm(mine_a[list(p)][:n] - theirs_a[:n], axis=1).mean()
            for p in itertools.permutations(range(len(mine)), n)
        )
        distances.append(best)
    return float(np.mean(distances)) if distances else None


def _spectrum(xyz: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    return np.sort(d[np.triu_indices(len(xyz), 1)])


def ligand_geometry(structure: Structure, ideal: dict[str, np.ndarray]) -> float | None:
    """RMS gap between each ligand's distance spectrum and its CCD ideal's."""
    errors = []
    for chain, (elements, xyz) in structure.ligands.items():
        if len(elements) < 3:
            continue
        reference = ideal.get(chain)
        if reference is None or len(reference) != len(xyz):
            continue
        errors.append(np.sqrt(np.mean((_spectrum(xyz) - _spectrum(reference)) ** 2)))
    return float(np.mean(errors)) if errors else None


def load_reference(target: str, family: str) -> dict[int, list[Structure]]:
    """The release's structures, by seed, with its recorded pLDDTs."""
    out: dict[int, list[Structure]] = {}
    for seed_dir in sorted((OUTPUTS / target / family).glob("seed*")):
        manifest = json.loads((seed_dir / "manifest.json").read_text())
        out[int(manifest["seed"])] = [
            read_cif(seed_dir / s["file"], plddt=s["mean_plddt"])
            for s in manifest["samples"]
        ]
    return out


def load_ours(run_dir: Path) -> dict[int, list[Structure]]:
    """A FoldForge run tree: <run_dir>/seed<k>/*-<i>.cif."""
    out: dict[int, list[Structure]] = {}
    for seed_dir in sorted(run_dir.glob("seed*")):
        cifs = sorted(seed_dir.glob("*-[0-9].cif"))
        if cifs:
            out[int(seed_dir.name.removeprefix("seed"))] = [read_cif(c) for c in cifs]
    return out


def ideal_ligands(target: str) -> dict[str, np.ndarray]:
    """CCD ideal heavy-atom coordinates of every multi-atom ligand chain."""
    from foldforge.data.ccd.database import CCDDatabase  # noqa: PLC0415 - heavy
    from foldforge.data.inputs.build import load  # noqa: PLC0415 - heavy

    spec = load(OUTPUTS / "inputs" / target / "foldforge.yaml")
    database = CCDDatabase(spec.spec.ccd_db)
    ideal = {}
    with database.activate():
        ccd = database.af3_ccd()
        for chain in spec.chains:
            if len(chain.ccds) != 1:
                continue
            record = ccd.get(chain.ccds[0])
            if (
                record is None
                or "_chem_comp_atom.pdbx_model_Cartn_x_ideal" not in record
            ):
                continue
            elements = record["_chem_comp_atom.type_symbol"]
            xyz = np.asarray(
                [
                    [
                        float(record[f"_chem_comp_atom.pdbx_model_Cartn_{k}_ideal"][i])
                        for k in "xyz"
                    ]
                    for i in range(len(elements))
                    if elements[i].upper() != "H"
                ]
            )
            if len(xyz) >= 3:
                ideal[chain.letter] = xyz
    return ideal


def _pairwise(a: list[Structure], b: list[Structure] | None, fn) -> float:
    values = []
    if b is None:
        for i, x in enumerate(a):
            values.extend(v for y in a[i + 1 :] if (v := fn(x, y)) is not None)
    else:
        values.extend(v for x in a for y in b if (v := fn(x, y)) is not None)
    return float(np.mean(values)) if values else float("nan")


def _within(ours_rel: float, rel_rel: float) -> bool:
    if np.isnan(ours_rel):
        return True
    return ours_rel <= rel_rel + max(0.5, 0.5 * rel_rel)


def judge(
    target: str, family: str, run_dir: Path, ideal: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Every metric and verdict for one family on one target."""
    reference = load_reference(target, family)
    ours = load_ours(run_dir)
    rel = [s for seed in reference.values() for s in seed]
    mine = [s for seed in ours.values() for s in seed]
    if not rel or not mine:
        return {
            "target": target,
            "family": family,
            "missing": {"release": len(rel), "ours": len(mine)},
        }
    rel_seed_means = [np.mean([s.plddt for s in v]) for v in reference.values()]
    our_seed_means = [np.mean([s.plddt for s in v]) for v in ours.values()]
    rel_plddt = float(np.mean([s.plddt for s in rel]))
    our_plddt = float(np.mean([s.plddt for s in mine]))
    # Both arms' seed-to-seed variation, with a floor of two points: three
    # seeds of a low-confidence fold with no MSA cannot resolve less.
    plddt_tolerance = max(
        2.0, 2 * float(np.hypot(np.std(rel_seed_means), np.std(our_seed_means)))
    )
    structure = {
        "rel_rel": _pairwise(rel, None, polymer_rmsd),
        "ours_rel": _pairwise(mine, rel, polymer_rmsd),
        "ours_ours": _pairwise(mine, None, polymer_rmsd),
    }
    place = {
        "rel_rel": _pairwise(rel, None, placement),
        "ours_rel": _pairwise(mine, rel, placement),
    }
    geometry_rel = [g for s in rel if (g := ligand_geometry(s, ideal)) is not None]
    geometry_ours = [g for s in mine if (g := ligand_geometry(s, ideal)) is not None]
    geometry = {
        "rel": float(np.mean(geometry_rel)) if geometry_rel else None,
        "ours": float(np.mean(geometry_ours)) if geometry_ours else None,
    }
    checks = {
        "structure": _within(structure["ours_rel"], structure["rel_rel"]),
        "plddt": abs(our_plddt - rel_plddt) <= plddt_tolerance,
        "placement": _within(place["ours_rel"], place["rel_rel"]),
        "ligand_geometry": geometry["ours"] is None
        or geometry["rel"] is None
        or geometry["ours"] <= geometry["rel"] + 0.05,
    }
    return {
        "target": target,
        "family": family,
        "samples": {"release": len(rel), "ours": len(mine)},
        "plddt": {
            "release": rel_plddt,
            "ours": our_plddt,
            "tolerance": plddt_tolerance,
        },
        "structure": structure,
        "placement": place,
        "ligand_geometry": geometry,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _fmt(value: float | None, digits: int = 2) -> str:
    return (
        "-"
        if value is None or (isinstance(value, float) and np.isnan(value))
        else f"{value:.{digits}f}"
    )


def report(results: list[dict[str, Any]]) -> str:
    """A Markdown table of every judgement."""
    lines = [
        "| target | family | pLDDT rel / ours (tol) | RMSD ours-rel (rel-rel) | placement ours-rel (rel-rel) | ligand geometry rel / ours | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if "missing" in r:
            lines.append(
                f"| {r['target']} | {r['family']} | missing {r['missing']} | | | | - |"
            )
            continue
        p, s, pl, g = r["plddt"], r["structure"], r["placement"], r["ligand_geometry"]
        failed = [k for k, ok in r["checks"].items() if not ok]
        lines.append(
            f"| {r['target']} | {r['family']} | {_fmt(p['release'])} / {_fmt(p['ours'])} ({_fmt(p['tolerance'], 1)}) "
            f"| {_fmt(s['ours_rel'])} ({_fmt(s['rel_rel'])}) | {_fmt(pl['ours_rel'])} ({_fmt(pl['rel_rel'])}) "
            f"| {_fmt(g['rel'], 3)} / {_fmt(g['ours'], 3)} | {'PASS' if r['pass'] else 'FAIL: ' + ', '.join(failed)} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    """``foldforge validate RUNS [--report FILE] [--families ...] [--targets ...]``."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="foldforge validate")
    parser.add_argument(
        "runs", type=Path, help="tree of <family>/<target>/seed<k>/ FoldForge folds"
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--families", nargs="+", default=list(FAMILIES))
    parser.add_argument("--targets", nargs="+", default=list(TARGETS))
    args = parser.parse_args(argv)
    results = []
    for target in args.targets:
        ideal = ideal_ligands(target)
        results.extend(
            judge(target, family, args.runs / family / target, ideal)
            for family in args.families
            # A release that cannot fold a target has no reference for it.
            if (OUTPUTS / target / family).is_dir()
        )
    text = report(results)
    print(text)  # noqa: T201 - CLI output contract
    if args.report:
        args.report.write_text(text)
        args.report.with_suffix(".json").write_text(
            json.dumps(results, indent=1, default=float) + "\n"
        )
    return 0 if all(r.get("pass", False) for r in results) else 1
