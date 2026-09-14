"""Resolve one MiniWorld input into model-specific tensor preparation boundaries."""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path

import yaml

from .fasta import EntityType, index_to_lowercase_letter, parse_fasta_file
from .lmdb import Alignment, Template, load_templates, species_aliases
from .spec import InferenceSpec


@dataclass(frozen=True)
class Chain:
    index: int
    id: str
    letter: str
    kind: str
    sequence: str
    ccds: tuple[str, ...]
    a3m: Path | None
    alignment: Alignment | None = None
    templates: tuple[Template, ...] = ()


@dataclass(frozen=True)
class Input:
    """Resolved target; chain order and chemistry selection are model-independent."""

    spec: InferenceSpec
    chains: tuple[Chain, ...]
    msa_species: dict[str, str] = dataclass_field(default_factory=dict)

    def manifest(self):
        return {
            "name": self.spec.name,
            "chains": [
                {
                    "chain": c.id,
                    "type": c.kind,
                    "sequence": c.sequence,
                    **({"ccd": c.ccds[0]} if c.kind == "ligand" else {}),
                }
                for c in self.chains
            ],
        }

    def resources(self):
        """Describe the resolved keys and coverage alongside saved predictions."""
        return [
            {
                "chain_index": c.index,
                "msa_db": str(self.spec.msa_db[c.letter])
                if c.alignment is not None
                else None,
                "msa_key": self.spec.msa.get(c.letter),
                "msa_rows": len(c.alignment.tokens)
                if c.alignment is not None
                else None,
                "template_db": str(self.spec.template_db)
                if str(c.index) in self.spec.template
                else None,
                "template_key": self.spec.template.get(str(c.index)),
                "templates": [
                    {
                        "id": t.id,
                        "backbone_residues": int(t.mask[:, :3].all(-1).sum()),
                        "observed_atoms": int(t.mask.sum()),
                        "release_date": t.release_date,
                    }
                    for t in c.templates
                ],
            }
            for c in self.chains
        ]

    def msa_text(self, chain):
        if chain.alignment is not None:
            return chain.alignment.a3m(self.msa_species)
        return chain.a3m.read_text() if chain.a3m else ""

    def af3(self, seed=0):
        sequences = []
        for c in self.chains:
            body = {"id": c.id}
            if c.kind == "ligand":
                body["ccdCodes"] = list(c.ccds)
            else:
                body["sequence"] = c.sequence
                body["modifications"] = []
                if c.kind in {"protein", "rna"}:
                    body["unpairedMsa"] = self.msa_text(c)
                if c.kind == "protein":
                    body["pairedMsa"] = (
                        self.msa_text(c) if c.alignment is not None else ""
                    )
                    body["templates"] = [t.af3() for t in c.templates]
            sequences.append({c.kind: body})
        return {
            "name": self.spec.name,
            "modelSeeds": [seed],
            "sequences": sequences,
            "dialect": "alphafold3",
            "version": 2,
        }

    def af_family(self):
        keys = {
            "protein": "proteinChain",
            "dna": "dnaSequence",
            "rna": "rnaSequence",
            "ligand": "ligand",
        }
        sequences = []
        for c in self.chains:
            body = {"count": 1}
            if c.kind == "ligand":
                body["ligand"] = "CCD_" + "_".join(c.ccds)
            else:
                body["sequence"] = c.sequence
                if c.alignment is not None:
                    body["unpairedMsa"] = self.msa_text(c)
                elif c.a3m:
                    body["unpairedMsaPath"] = str(c.a3m)
                body["pairedMsa"] = (
                    self.msa_text(c)
                    if c.alignment is not None and c.kind == "protein"
                    else ""
                )
                if c.templates:
                    body["miniworldTemplates"] = [t.payload() for t in c.templates]
            sequences.append({keys[c.kind]: body})
        return [{"name": self.spec.name, "sequences": sequences}]

    def esmfold2(self, msa_depth):
        if self.spec.template:
            raise ValueError("ESMFold2 checkpoint has no template conditioning path")
        if any(c.alignment is not None and c.kind != "protein" for c in self.chains):
            raise ValueError("ESMFold2 checkpoint consumes protein MSAs only")
        from esm.utils.msa.msa import MSA
        from esm.utils.parsing import FastaEntry
        from esm.utils.structure.input_builder import (
            DNAInput,
            LigandInput,
            ProteinInput,
            RNAInput,
            StructurePredictionInput,
        )

        from foldforge.models.esmfold2.features import read_a3m

        classes = {"protein": ProteinInput, "rna": RNAInput, "dna": DNAInput}
        sequences = []
        for c in self.chains:
            if c.kind == "ligand":
                sequences.append(LigandInput(id=[c.id], ccd=list(c.ccds)))
            elif c.kind == "protein":
                if c.alignment is not None:
                    lines = c.alignment.a3m(self.msa_species, msa_depth).splitlines()
                    msa = MSA(
                        [
                            FastaEntry(lines[i][1:], lines[i + 1])
                            for i in range(0, len(lines), 2)
                        ]
                    )
                else:
                    msa = MSA(read_a3m(c.a3m, msa_depth)) if c.a3m else None
                sequences.append(ProteinInput(id=[c.id], sequence=c.sequence, msa=msa))
            else:
                sequences.append(classes[c.kind](id=[c.id], sequence=c.sequence))
        return StructurePredictionInput(sequences=sequences)


def load(path: Path) -> Input:
    """Use MiniWorld's YAML fields and FASTA grammar, paths relative to the spec."""
    path = path.expanduser().resolve()
    data = yaml.safe_load(path.read_text())
    for field in ("fasta", "a3m", "msa_db"):
        data[field] = {
            str(k): str((path.parent / Path(v)).resolve())
            for k, v in data.get(field, {}).items()
        }
    for field in ("ccd_db", "template_db", "cif_db", "tokenization", "contacts"):
        if isinstance(data.get(field), str):
            data[field] = str((path.parent / data[field]).resolve())
    spec = InferenceSpec.model_validate(data)
    if spec.name is None:
        spec.name = path.stem
    if Path(spec.name).name != spec.name or spec.name in {"", ".", ".."}:
        raise ValueError("Input name must be a target name, not a path")
    # A released checkpoint cannot acquire MiniWorld-specific conditioning by
    # changing its input syntax. Reject requests it cannot mathematically honor.
    unsupported = [
        field
        for field in (
            "complex_templates",
            "diffusion_groups",
            "flexible_docking",
            "refinement",
            "condition_groups",
            "residue_indices",
            "tokenization",
        )
        if getattr(spec, field)
    ]
    if spec.contacts.positive or spec.contacts.negative:
        unsupported.append("contacts")
    if spec.paired_msa_only or spec.no_pairing_msa:
        unsupported.append("msa_pairing_mode")
    if unsupported:
        raise ValueError(
            "Released checkpoint adapters do not support these conditioning fields: "
            + ", ".join(unsupported)
        )
    if set(spec.msa) != set(spec.msa_db):
        raise ValueError("msa and msa_db must provide the same chain letters")
    if set(spec.msa) & set(spec.a3m):
        raise ValueError("Choose a3m or msa/msa_db for each chain letter")
    if (set(spec.msa) | set(spec.a3m)) - set(spec.chain_letters.values()):
        raise ValueError("MSA references an unknown chain letter")
    if set(spec.template) - set(spec.chain_letters):
        raise ValueError("template references an unknown chain index")
    if spec.template and spec.template_db is None:
        raise ValueError("template requires template_db")
    if spec.template_as_contact:
        raise ValueError("Released checkpoints do not support template_as_contact")
    indices = spec.chain_indices()
    if indices != list(range(len(indices))):
        raise ValueError("Chain indices must be consecutive from zero")
    kinds = {
        EntityType.PROTEIN: "protein",
        EntityType.RNA: "rna",
        EntityType.DNA: "dna",
        EntityType.NON_POLYMER: "ligand",
    }
    chains = []
    alignments = {}
    for i in indices:
        letter = spec.chain_letters[str(i)]
        parsed = parse_fasta_file(spec.fasta[letter], i)
        if parsed.entity_type == EntityType.BRANCHED:
            raise ValueError(
                "Branched FASTA needs explicit atom-level bonds for released checkpoint adapters"
            )
        kind = kinds[parsed.entity_type]
        sequence = "".join(parsed.one_letter_seq)
        if kind == "dna" and (letter in spec.msa or letter in spec.a3m):
            raise ValueError("Released checkpoint adapters do not consume DNA MSAs")
        if letter in spec.msa and letter not in alignments:
            alignments[letter] = Alignment.load(
                spec.msa_db[letter], spec.msa[letter], kind, sequence
            )
        templates = ()
        if str(i) in spec.template:
            if kind != "protein":
                raise ValueError("Released checkpoints support protein templates only")
            templates = load_templates(
                spec.template_db, spec.template[str(i)], len(sequence), spec.template_n
            )
        chains.append(
            Chain(
                i,
                index_to_lowercase_letter(i).upper(),
                letter,
                kinds[parsed.entity_type],
                "".join(parsed.one_letter_seq),
                tuple(parsed.chemcomp_ids),
                spec.a3m.get(letter),
                alignments.get(letter),
                templates,
            )
        )
    return Input(spec, tuple(chains), species_aliases(alignments.values()))


def write_adapter_input(target: Input, model: str, destination: Path, seed=0):
    """Persist the exact translated input for review and reproduction."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = target.af3(seed) if model == "af3" else target.af_family()
    destination.write_text(json.dumps(payload, indent=2) + "\n")


def limit_msa(target: Input, depth: int, directory: Path) -> Input:
    """Apply an explicit input-row limit identically before model preparation."""
    from dataclasses import replace

    if depth < 1:
        raise ValueError("MSA depth must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for chain in target.chains:
        if chain.a3m is None or chain.a3m in paths:
            continue
        rows = 0
        lines = []
        for line in chain.a3m.read_text().splitlines(keepends=True):
            if line.startswith(">"):
                rows += 1
                if rows > depth:
                    break
            lines.append(line)
        if rows == 0:
            raise ValueError(f"MSA file has no FASTA records: {chain.a3m}")
        path = (directory / f"chain-{chain.index}.a3m").resolve()
        path.write_text("".join(lines))
        paths[chain.a3m] = path
    chains = tuple(
        replace(
            chain,
            a3m=paths.get(chain.a3m),
            alignment=(
                replace(
                    chain.alignment,
                    tokens=chain.alignment.tokens[:depth],
                    deletions=chain.alignment.deletions[:depth],
                    species=chain.alignment.species[:depth],
                )
                if chain.alignment is not None
                else None
            ),
        )
        for chain in target.chains
    )
    spec = target.spec.model_copy(
        update={"a3m": {key: paths[value] for key, value in target.spec.a3m.items()}}
    )
    return Input(spec, chains, target.msa_species)
