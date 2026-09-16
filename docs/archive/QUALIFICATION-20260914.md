# FoldForge inference qualification — 2026-09-14

Follow-up operational DB discovery and cache-builder repairs are recorded in
[CLOSEOUT-20260914.md](CLOSEOUT-20260914.md). The absent-mount notes below describe
the original qualification; the follow-up directly reads the operational database.

This is an inference integration qualification. It does not qualify training,
new GPU architectures, or structure-prediction accuracy from two-step smoke runs.
Pinned environments, checkpoint files and the CCD database were preserved.

## Shared code

MiniWorld YAML/typed FASTA, CCDMol LMDB, MSA-LMDB and TemplateMol LMDB readers
are common to all adapters. Protein/DNA/RNA/ligand features are packed at the
released-model boundary; unsupported conditioning is rejected explicitly.
ESMFold2 accepts nucleic-acid sequences but does not consume RNA MSA or templates.
Fragment tokenization is optional and is not required for ordinary inference.

Thirteen identical token/atom/window layout functions now live in
`team_gm.modules.tensor_layout` (see `shared-layout-migration.json`). Their
original equations are frozen in regression fixtures. A fourteenth shared
function owns FP32 atom sum/mean accumulation, including ESMFold2 masking. Shared attention,
MSA/template composition, Pairformer and conditioning mappings from the preceding
integration remain in team-gm/engine. Model-specific parameter shells and
sampling/confidence equations remain next to the checkpoint adapter.

Reusable compile/capture logic lives in `team_gm.modules.execution`; FoldForge
only selects the boundary and measures the complete forward. OpenDDE's lazy
relative-position inputs expose their tensor fields as a dataclass. AF3 atom
layout sizes are Python integer tuples at the compute boundary, with an explicit
layout compatibility check. This check exposed and fixed AF3's key-side reference
space gather, which used raw token-atom IDs instead of query-layout IDs.
Protenix/OpenDDE CIF writers now preserve atom pLDDT as B factors on the 0-100 scale;
the common confidence tensor had already been populated correctly.

The ESMFold2 adapter previously passed CCD strings into a list field. `BEN`
was read as separate components and `ZN` expanded into unrelated components.
Both entry points now pass CCD code lists. Cached ESMC token dimensions must
match the tokenized input; a mismatch fails before loading folding weights.

## Execution contract

Use `--config configs/inference/graph-bf16.yaml` for inference graph execution.
Native BF16 means non-normalization learned parameters are BF16; normalization
parameters remain FP32. These runners do not use autocast.

`execution.compile` applies real `torch.compile(backend="inductor")` to the
selected callable. Inductor internal CUDA graphs are disabled; explicit capture
owns buffers and replay. Low-precision rounding and division-rounding options
preserve the intended boundaries; FP32 geometry matmuls use `highest` precision.
Compiler arithmetic still differs from eager: the initial pointwise denoiser
atol/rtol 0.025 comparison did not pass and is not relabeled as a pass.

The numerical checks distinguish three questions: repeated eager evaluation,
replay vs the same uncaptured callable, and compiled vs eager final predictions.
BF16 atom scatter reductions made even eager repetition differ (Protenix max abs
0.515625; ESMFold2 0.05078125 in the diagnostic case). The shared atom mean now
accumulates in FP32 and casts once. Eager repetitions and manual replays then
matched exactly in those probes. This does not change parameter dtype or enable
autocast. Norm and projection unit tests preserve the checkpoint equations.

Compiled-vs-eager differences were then assessed on complete released schedules,
under predeclared limits of 0.25 A aligned protein CA RMSD and one pLDDT point.
Confidence comes from the common `.prediction.pt` token scores, converted from
[0,1] to [0,100]; an absent CIF B factor is never treated as zero confidence.
All five checkpoints passed this end-to-end numerical regression. It establishes
neither bitwise compiler equivalence nor biological accuracy on arbitrary targets.

`execution.cuda_graph` captures the denoiser: ESMFold2 `denoise`, AF3 diffusion
head, and Protenix/OpenDDE diffusion-module `forward`. Noise generation,
augmentation, the trunk and confidence/output processing remain outside this
graph. Full-model capture is rejected because host-side sampling would otherwise
be frozen or misrepresented. Compilation with scope `model` is experimental;
only the denoiser scope is qualified here.

Warmup/autotuning is enclosed in an RNG fork so setup does not consume the
sampler's random stream; replay owns the RNG advance for captured random ops.
A regression tests both the returned random tensor and the following random draw.

Graph keys include tensor shape/dtype/device/stride and scalar/metadata values.
A change in tensor values copies into static buffers; outputs are cloned before
returning so subsequent replay cannot overwrite an earlier result. The graph
count is bounded; exceeding it or failing capture raises instead of falling back
silently. The runner rejects gradient-enabled execution.

Reports separate requested flags from actual unique compiled graph and replay
counts. A requested compile that executes no compiled graph raises. Timing covers
complete model forward, excluding featurization and output decoding; the first
forward includes cold compile/capture. Warm repeats reset Python/NumPy/Torch RNG
and restore input containers that model code consumes. These are measurements of
the same released predictor, not a PyTorch-vs-engine speedup claim.

## Assets and deployment boundary

CCD: 48,965 BioMol records, SHA256
`3636431697cf0875171b5e93350b95d77a6e02443d305b7bb49240308e9cd26b`.
Operational MSA/template mount paths configured in MiniWorld were absent on this
server. Qualification serialized the existing MiniWorld export arrays into actual
LMDB records and used the ordinary readers. This establishes format compatibility;
it is not an audit of an unavailable production database. Select existing paths
in the input YAML before production use.

Model shape coverage differs from the original MiniWorld tuning sweep. Cache-miss
records in `qualification-cache-misses.csv` are shape-specific tuning gaps; the
runners used the reported fallback candidate search. Full cache coverage is not
claimed. Cache tuning on another GPU remains a separate GPU-specific operation.

## Validation results

Slurm A6000 allocations: gpu03 `1684558`, gpu04 `1684559`, and gpu01
`1684601`. Python 3.12.14, Torch 2.10.0+cu128, Quack 0.5.0, CUTLASS DSL 4.5.2,
FA2 2.8.3.post1; engine commit d2266a035de11384c46f8cc980e6460f60925413.

1UBQ, 76 tokens, one sample, MSA 128. AF-family: 10 recycles, 200 sampling steps,
four prepared templates; ESMFold2: configured 3 loops / 14 pre-clipping steps,
10 actual denoiser calls after schedule clipping, cached ESMC, no templates.
Full forward timings below are one warm repeat in separate processes, including
trunk and confidence, excluding feature preparation, decoding and ESMC. Cold
calls include compilation/capture and two additional diagnostic reference calls.
These are reproducibility measurements, not a controlled same-process speedup
benchmark or a PyTorch-vs-engine comparison.

| Checkpoint | Graph forward (s) | Compile + graph (s) | Compiled graphs | CA difference (A) | Token pLDDT difference (0-100) |
| --- | ---: | ---: | ---: | ---: | ---: |
| af3 | 2.928 | 2.517 | 1 | 0.045 | 0.003 |
| esmfold2 | 0.275 | 0.273 | 13 | 0.103 | 0.080 |
| opendde | 3.449 | 2.822 | 1 | 0.054 | 0.011 |
| protenix | 3.072 | 2.361 | 1 | 0.056 | 0.008 |
| protenix-v2 | 4.011 | 3.308 | 1 | 0.066 | 0.004 |

Manual replays were compared against the same callable with capture disabled
(against the compiled callable when compile was enabled), at atol/rtol 0.0001,
and observed max absolute difference was zero at both checked denoising calls.

Reproduce on an allocated GPU node:

```bash
source scripts/activate_env.sh
python scripts/qualify_execution.py af3 --spec target.yaml \
  --config configs/inference/graph-bf16.yaml --out runs/af3/graph
# For the second run use the same config with execution.compile: true.
python scripts/compare_execution_structures.py validation/comparison \
  --output validation/comparison.json
```

The structure comparison expects one sample and one target per model/mode.
Diagnostics assert on replay mismatches; runtime reports use observed compile
and replay counts, never flags alone. Failed early candidates remain in local
validation logs and are excluded from the accepted tables.

Full regression before the final CIF writer change: **199 passed, zero skipped**.
The added CIF roundtrip check and its neighboring confidence checks: **7 passed**.
Final installed-tree validation: **200 passed, zero skipped**, 101.40 s.
Imports resolved to FoldForge `src/` and its actual `libs/team-gm/src/` member;
all 37 initially promoted file hashes matched the reviewed candidate.
The installed public `qualify_execution.py` also passed OpenDDE protein/RNA
inference, with both replay differences exactly zero.

| Complex | AF3 | Protenix v1 | OpenDDE | ESMFold2 |
| --- | --- | --- | --- | --- |
| 3PTB, protein + calcium/benzamidine | pass | pass | pass | pass |
| 1A1K, protein + DNA + zinc | pass | pass | pass | pass |
| 4YX2, three protein chains (594 residues) | pass | pass | pass | pass |
| Synthetic protein 16 aa + RNA 8 nt | pass | pass | pass | pass |

All 16 complex cases used 1 recycle, 2 sampling steps, 1 sample and MSA 128;
these are input/execution smoke tests, not structure-accuracy results. They
compare the first two graph denoiser calls to ordinary execution. The exact
observed differences are in `qualification-complex-results.json`. The synthetic
RNA case used actual protein/RNA MSA LMDB records for AF-family models. ESMFold2
received the RNA sequence without unsupported RNA MSA conditioning and computed
ESMC from its live checkpoint. PDB cases used prepared MiniWorld exports serialized
into MSA/template LMDB records; ESMFold2 explicitly omitted templates and used
cached ESMC embeddings.

The separately configured production database mount remains the only untested
input deployment resource. This is not a claim that every architecture or every
new model shape has a tuned engine cache; those gaps are enumerated separately.
