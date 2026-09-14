"""Build ESMFold2's input features, using the released implementation's builder.

Adapted from team-gm's ``scripts/esmfold2_e2e.py``. It lives in the model package
rather than ``foldforge.data`` because it is ESMFold2's own contract: the feature
dict comes from the upstream ``ESMFold2InputBuilder`` so that a comparison
against the released implementation differs in the model and not in the
plumbing. A shared FoldForge feature pipeline, when there is one, adapts to this
rather than replacing it.
"""

import json
from pathlib import Path

import torch
from foldforge.models.esmfold2.input.processor import ESMFold2InputBuilder
from esm.utils.msa.msa import MSA
from esm.utils.parsing import FastaEntry
from esm.utils.structure.input_builder import (
    DNAInput,
    LigandInput,
    ProteinInput,
    RNAInput,
    StructurePredictionInput,
)

__all__ = [
    "ESMFold2InputBuilder",
    "build_input",
    "ca_coords_from_cif",
    "kabsch_rmsd",
    "model_kwargs",
    "read_a3m",
]

# The reference stores token_bonds as [B, L, L, 1]; team-gm takes [B, L, L].
BONDS_WITH_CHANNEL = 4
POLYMER_INPUT = {"protein": ProteinInput, "dna": DNAInput, "rna": RNAInput}


def read_a3m(path: Path, depth: int) -> list[FastaEntry]:
    """Read up to ``depth`` alignment rows, query first."""
    entries: list[FastaEntry] = []
    header, chunks = None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                entries.append(FastaEntry(header, "".join(chunks)))
                if len(entries) >= depth:
                    return entries
            header, chunks = line[1:], []
        elif header is not None:
            chunks.append(line.strip())
    if header is not None:
        entries.append(FastaEntry(header, "".join(chunks)))
    return entries


def build_input(sample: Path, msa_depth: int) -> StructurePredictionInput:
    """Assemble the model input from a target's ``target.json``."""
    manifest = json.loads((sample / "target.json").read_text())
    sequences = []
    for index, chain in enumerate(manifest["chains"]):
        # Fresh sequential ids: several entities can share one auth chain
        # (3PTB's ligands sit on chain A alongside the protein).
        chain_id = chr(ord("A") + index)
        kind = chain["type"]
        if kind == "ligand":
            codes = chain["ccd"]
            sequences.append(LigandInput(id=[chain_id], ccd=[codes] if isinstance(codes, str) else codes))
            continue
        cls = POLYMER_INPUT[kind]
        if kind == "protein":
            a3m = sample / "msa" / f"{chain['chain']}.a3m"
            msa = MSA(read_a3m(a3m, msa_depth)) if a3m.exists() else None
            sequences.append(cls(id=[chain_id], sequence=chain["sequence"], msa=msa))
        else:
            sequences.append(cls(id=[chain_id], sequence=chain["sequence"]))
    return StructurePredictionInput(sequences=sequences)


def ca_coords_from_cif(path: Path) -> torch.Tensor:
    """Pull every CA coordinate out of an mmCIF ``_atom_site`` loop."""
    columns: dict[str, int] = {}
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("_atom_site."):
            columns[stripped.split(".", 1)[1]] = len(columns)
            continue
        if not columns or not stripped or stripped.startswith(("#", "loop_", "_")):
            continue
        fields = stripped.split()
        if len(fields) < len(columns) or fields[columns["label_atom_id"]] != "CA":
            continue
        rows.append(
            [float(fields[columns[a]]) for a in ("Cartn_x", "Cartn_y", "Cartn_z")]
        )
    return torch.tensor(rows) if rows else torch.zeros(0, 3)


def kabsch_rmsd(mobile: torch.Tensor, target: torch.Tensor) -> float:
    """RMSD after optimal superposition."""
    mobile = mobile - mobile.mean(0, keepdim=True)
    target = target - target.mean(0, keepdim=True)
    u, _, vh = torch.linalg.svd(target.T @ mobile)
    sign = torch.sign(torch.linalg.det(u @ vh))
    rotation = u @ torch.diag(torch.tensor([1.0, 1.0, sign], dtype=mobile.dtype)) @ vh
    return float((((mobile @ rotation.T) - target) ** 2).sum(-1).mean().sqrt())


def model_kwargs(features: dict, dtype: torch.dtype = torch.float32) -> dict:
    """Map the reference feature dict onto the model's forward signature.

    ``dtype`` applies to the real-valued features only; indices stay integral and
    masks stay boolean. It has to be threaded through: hardcoding ``.float()``
    here silently feeds fp32 into a bf16 model, which either raises at a fused
    kernel boundary or upcasts the whole track back to fp32 and forfeits the 2.1x
    the bf16 kernels are worth.
    """
    token_bonds = features["token_bonds"]
    if token_bonds.dim() == BONDS_WITH_CHANNEL:
        token_bonds = token_bonds[..., 0]
    kwargs = {
        "residue_type": features["res_type"].long(),
        "residue_index": features["residue_index"].long(),
        "asym_id": features["asym_id"].long(),
        "sym_id": features["sym_id"].long(),
        "entity_id": features["entity_id"].long(),
        "token_index": features["token_index"].long(),
        "mol_type": features["mol_type"].long(),
        "token_bonds": token_bonds.to(dtype),
        "mask": features["token_attention_mask"].bool(),
        "ref_pos": features["ref_pos"].to(dtype),
        "ref_charge": features["ref_charge"].to(dtype),
        "ref_element": features["ref_element"].long(),
        "ref_atom_name_chars": features["ref_atom_name_chars"].long(),
        "ref_space_uid": features["ref_space_uid"].long(),
        "atom_mask": features["atom_attention_mask"].bool(),
        "atom_to_token": features["atom_to_token"].long(),
        "representative_atom_index": features["distogram_atom_idx"].long(),
    }
    for name in ("msa", "has_deletion", "deletion_value", "deletion_mean"):
        if features.get(name) is not None:
            kwargs[name] = (
                features[name].long() if name == "msa" else features[name].to(dtype)
            )
    if features.get("msa_attention_mask") is not None:
        kwargs["msa_mask"] = features["msa_attention_mask"].bool()
    return kwargs
