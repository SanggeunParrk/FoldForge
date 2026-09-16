# Data and evaluation ownership

Operational code is organized by responsibility. Model names belong at the
architecture, released configuration, and checkpoint boundaries.

| Responsibility | Implementation |
| --- | --- |
| Chemical constants | `data/constants/` |
| CCD access and molecular representations | `data/ccd/` |
| Tokens, atom/geometry features, constraints, embeddings | `data/features/` |
| Input specifications, JSON/FASTA entities, sequence preparation, LMDB resources | `data/inputs/` |
| MSA parsing, pairing, alignment/search tools, features | `data/msa/` |
| Template parsing and features | `data/template/` |
| Notebook requests, remote service UI, visualization | `data/web/` |
| mmCIF parsing, structure compatibility, filtering, statistics | `data/parser.py`, `structure.py`, `filter.py`, `statistics.py` |
| Training datasets, loaders, preparation pipeline | `data/dataset.py`, `loader.py`, `pipeline.py` |
| Confidence and clashes | `eval/confidence.py`, `clash.py` |
| RMSD, lDDT, permutation | `eval/rmsd.py`, `lddt.py`, `permutation/` |
| Training loss and shape complementarity | `training/loss.py`, `shape_complementarity.py` |

The data tree has seven subpackages. The former `core`, `constraint`, `esm`,
`pipeline`, `sequence`, and `tools` directories are removed, and `inference` is
renamed `inputs` because those contracts are consumed before model execution.
Feature implementations live together in `features`; alignment/search tools live
with MSA processing. There are no compatibility wrappers at the retired paths.

`Token`, `TokenArray`, `RawMsa`, `MSAPairingEngine`, chemical tables, template
parser records/exceptions, Kalign, structure compatibility, and clash detection
have one implementation. Shared feature/parser methods live in common bases in
the same responsibility module.

`Residue*` and `Structural*` identify token-layout policies. The structural
branch splits protein backbone/sidechain and nucleic-acid components, retaining
parent-residue and twin-token indices; residue tokens are still used by its
trunk. These are not interchangeable token layouts. Their policy-specific
feature assembly remains explicit, alongside the shared implementation.

`detailed_*` and `compact_*` confidence assembly retain released configuration
fields, output fields, and CPU-offload behavior. Both call the common team-gm
confidence mathematics. Clash detail dtype is an explicit constructor option:
legacy detailed output uses boolean details and compact output uses float32
counts. No thresholds or ranking equations change in this cleanup.

There are no compatibility modules at the retired model-prefixed data/eval
paths. Test oracle identities remain frozen; `tests/references/current_modules.json`
and `current_symbols.json` resolve those identities to current implementations.
Retired namespace-only and unused extension entries are omitted from that lookup.

Prediction output remains under repository `runs/`, enforced by
`models/io/paths.py` for CLI and direct output writers.
