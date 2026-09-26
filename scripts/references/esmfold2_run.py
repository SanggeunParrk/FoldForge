"""Run the released ESMFold2 (transformers + esm) on a reference input.

Usage: esmfold2_run.py BOLTZ_YAML OUT_DIR SEED. Reads the Boltz-style YAML the
other references use, folds it with three loops (the checkpoint's own
default), 200 steps and five samples, and writes sample_<i>.cif with pLDDT
in the B-factors.
"""

import sys
from pathlib import Path

import torch
import yaml
from esm.models.esmfold2 import ESMFold2InputBuilder
from esm.utils.structure.input_builder import (
    DNAInput,
    LigandInput,
    ProteinInput,
    RNAInput,
    StructurePredictionInput,
)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

ROOT = Path(__file__).resolve().parents[2]
spec_path, out, seed = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
kinds = {"protein": ProteinInput, "dna": DNAInput, "rna": RNAInput}
sequences = []
for entry in yaml.safe_load(spec_path.read_text())["sequences"]:
    ((kind, body),) = entry.items()
    if kind == "ligand":
        sequences.append(LigandInput(id=body["id"], ccd=[body["ccd"]]))
    else:
        sequences.append(kinds[kind](id=body["id"], sequence=body["sequence"]))
model = (
    ESMFold2Model.from_pretrained(str(ROOT / "model_checkpoints/esmfold2"))
    .cuda()
    .eval()
)
builder = ESMFold2InputBuilder()
with torch.no_grad():
    results = builder.fold(
        model,
        StructurePredictionInput(sequences=sequences),
        num_loops=3,
        num_sampling_steps=200,
        num_diffusion_samples=5,
        seed=seed,
    )
for i, result in enumerate(results):
    (out / f"sample_{i}.cif").write_text(result.complex.to_mmcif())
