Current contract: [MiniWorld formats](MINIWORLD-FORMAT.md). Current execution qualification: [2026-09-14](../archive/QUALIFICATION-20260914.md). Dated results below are historical and do not override these documents.

Current format and shared-composition reference: [MiniWorld formats](MINIWORLD-FORMAT.md).
The dated validation sections below describe the state when those checks ran.

# FoldForge model integration

Working tree: `main`; shared team-gm member: branch `foldforge/dense-families`.
All predictors must use FoldForge's environment, team-gm blocks, and
miniworld-engine ops. A package name or a subprocess launching an untouched
upstream model is not a completed shared-engine port.

**As of 2026-09-23 there is one graph.** `models/architectures/af3.py` is the
only architecture in the tree, and a predictor is a `DenseSpec` row in
`modules/dense/spec.py` stating where its release disagrees with AF3. The flat
and sequence layouts, and the per-model architectures that lived on them, are
deleted. Sections dated before this describe the separate ports they replaced.

See **One graph — 2026-09-23** below for each family's evidence.

Source pins and original licenses are retained in each model's `SOURCE.json` and
`UPSTREAM-LICENSE`:

- AF3 PyTorch: Xfold `22bdeedfa309ef4ff6f9199910d8403915de69d6`.
- AF3 CPU feature package: official AlphaFold3 v3.0.1,
  `231efc9bb9c13b45cc59e43f7107869084ee9624`, in `libs/af3-data`.
- Protenix: `4c355be4553512f72453ecbfb65e69f4c35d1413`.
- OpenDDE: `ddfa1df8aff1babf1fddac4247b7d2351bd0ce9f`.
- OpenDDE checkpoint and CCD assets: Hugging Face revision
  `eddd563ce96571f784012edd8f045181c8f8627d`; checkpoint SHA256
  `7b826620390afad877ee2babc6a4d0df81b94d3a0be030959853d6a7da0807cc`.

The upstream source trees are retained for provenance. Unused upstream training,
web-service, search and optional ESM utilities are not qualified FoldForge APIs.
Only the packaged loaders and inference commands described here are supported.

Current CCD and module-boundary status is recorded in
[code unification](../archive/CODE-UNIFICATION-20260913.md). The original port validation
below establishes execution, not complete migration to team-gm blocks.

## Running a fold

Every family takes the same MiniWorld YAML and the same flags -- there is one
CLI, not one per predictor. On an allocated GPU node:

```bash
source scripts/activate_env.sh
foldforge models
foldforge fold af3      --spec configs/inference/1ubq.yaml --out runs/af3
foldforge fold esmfold2 --spec configs/inference/1ubq.yaml --out runs/esmfold2
```

`--checkpoint` is optional when the release's default blob sits in
`model_checkpoints/<model>/` or in its family's directory. `--config` takes a
YAML of backend, precision, seeds, recycles, steps and execution settings; see
`runs/*/tools/config.yaml` in any recorded run for the shape.

A family with no template stack (`template_layers == 0`, both ESMFold2
releases) refuses a templated input, and one with no MSA stack
(`msa_layers == 0`, ESMFold2-Fast) ignores the alignment. Both rules are read
from the family row, so a new release needs no code.

The command writes per-sample CIF and a JSON recording the LM source,
precision, seeds, MSA depth, sampling settings and the actual compile/CUDA-graph
settings. This correctness runner uses compile off and CUDA graphs off.

## One graph — 2026-09-23

Every registered family is a `DenseSpec` row on `models/architectures/af3.py`.
Two kinds of evidence are recorded, because the first does not imply the second.

**Parameter trees.** `scripts/diff_dense_checkpoint.py` compares what the graph
wants against what the blob supplies. Eleven of twelve rows are exact with no
discrepancy in either direction: alphafold3 404, boltz2 440, chai1 397,
esmfold2 343, esmfold2-fast 303, intellifold2 404, openbind0 404, openfold3
404, protenix1 402, protenix2 402, rosettafold3 444. OpenDDE's blob is a raw
`.pt` the tool does not read; it is verified by folding instead.

**Folds.** A tree diff cannot see a convention -- a function applied to a
tensor owns no parameter -- and this project has three cases of an exact tree
folding to garbage. So a family is accepted by folding it. 5I28, PyTorch
backend, `af3_default` precision, 10 recycles, 200 steps, seeds 0/0, with AF3
as the control on the identical input:

| family | peptide C-N | clashes | CA vs AF3 | pLDDT |
|---|---|---|---|---|
| af3 (control) | 0 / 127 | 0 | -- | 96.11 |
| boltz2 | 0 / 127 | 0 | 0.667 A | 97.18 |
| intellifold2 | 0 / 127 | 0 | 0.148 A | 95.78 |
| openfold3 | 0 / 127 | 0 | 0.203 A | 95.06 |
| openfold3-preview2 | 0 / 127 | 0 | 0.209 A | 84.95 |
| rosettafold3 | 0 / 127 | 0 | 0.208 A | 86.87 |

Agreement with AF3 is NOT acceptance: it passed Chai-1 while it was 50 pLDDT
off and Protenix while its template term was missing. Every family has since
been folded against its OWN release -- see "Every family against its own
release" below. Chai-1 matches its release on this input -- pLDDT 95.5 against
95.0-95.2, CA within 0.25 A -- recorded separately below. OpenDDE folds 0 / 127 on 5I28 and
0 / 75, 0 / 222, 0 / 591 on 1UBQ, 3PTB and 4YX2.

The two ESMFold2 releases have no template stack, so they take 1UBQ without
templates, measured against the deposited structure:

| family | peptide C-N | clashes | CA vs deposited 1UBQ | pLDDT |
|---|---|---|---|---|
| esmfold2 | 0 / 75 | 0 | 1.517 A (1.065 A at the release's own 3 loops / 14 steps) | 78.6 |
| esmfold2-fast | 0 / 75 | 0 | **7.528 A** | 59.6 |

**ESMFold2-Fast is accepted, on 6MRR.** It was not accepted for most of a
session on the strength of 1UBQ, where it folds to 6.7-17.7 A, and that was the
wrong test. The reference measures this family on **6MRR**, and on 1UBQ the
RELEASED implementation does not fold at all: every one of five seeds dies
inside its own Kabsch align with a non-converging SVD, its coordinates gone
degenerate. A poor number on a target the release itself cannot fold says
nothing about the port.

On 6MRR, no alignment, five seeds -- against what the reference records for the
same target and the native implementation it was comparing itself to:

| | FoldForge best | FoldForge mean | reference best / mean | native mean |
|---|---|---|---|---|
| esmfold2 | 1.644 A | **1.668 A** | 1.494 / 1.742 | 1.739 |
| esmfold2-fast | 1.662 A | **1.684 A** | 1.243 / 1.699 | 1.646 |

Both land inside the band. The row differs from `esmfold2` in `trunk_layers`
(48 -> 24) and `msa_layers` (4 -> 0) and in nothing else, matching the released
configs and the reference's registry, and it folds like it.

### Running a released implementation as the control

Two of this document's findings rest on folding the same target with the
RELEASE, and neither was reachable from reasoning alone. Both are set up here
so the next person does not rediscover the setup.

**ESMFold2** ships inside `transformers`, so it needs nothing extra:

```python
ESMFold2Model.from_pretrained("model_checkpoints/esmfold2")
```

Use the LOCAL path. `from_pretrained("biohub/ESMFold2")` fails in this build --
`DiffusionStructureHeadConfig.__init__() got an unexpected keyword argument
'architectures'` -- because the published config parses differently. Call it
with `num_diffusion_samples=1` and `num_sampling_steps=200`; asking for five
samples in one call hits a degenerate SVD inside the model's own Kabsch align.

**Chai-1** needs its weights and a few packages:

- Source: `/public_data/thalkak_envs/chai-lab` (already on this machine).
- Weights: `models_v2/*.pt` (1.1 GB) and `conformers_v1.apkl` (119 MB) from
  `https://chaiassets.com/chai1-inference-depencencies/`, into
  `model_checkpoints/chai1/`. Point `CHAI_DOWNLOADS_DIR` at that directory.
- Missing packages: `gemmi`, `antipickle==0.2.0`, `modelcif`, `typer`,
  `pandera`, `numba`. Install them to a directory and reach them with
  `PYTHONPATH` rather than into `.venv` -- they are for a control, not for
  FoldForge.
- **Pin `antipickle==0.2.0`.** 0.2.2 ships a built-in `torch` adapter whose
  typestring collides with the one `chai_lab` registers, and the collision is a
  bare `assert`, not a warning.
- `run_inference` refuses a non-empty output directory.

`ScriptModule` refuses `register_forward_hook`, and the traced blocks expose no
callable `forward`. The call site is ordinary Python, so wrap
`chai_lab.chai1.ModuleWrapper.forward` to capture any component's inputs and
outputs. Discriminate by an argument only one component has --
`atom_within_token_index` for the confidence head, `msa_input_feats` for the
trunk -- because both take a `token_single_trunk_repr`.


The other releases, all runnable from this machine (details in
`runs/release-compare-20260924/tools/*.sbatch`):

- **Boltz-2**: `/public_data/thalkak_envs/boltz/bin/boltz predict`, cache = a
  directory with `boltz2_conf.ckpt` and `mols` linked from
  `/public_data/thalkak_checkpoints/boltz`. `msa: empty` for no alignment.
- **Protenix**: `/public_data/thalkak_envs/protenix{,_v2}/bin/protenix pred`
  with `PROTENIX_ROOT_DIR` holding `checkpoint/` (the params) and a WRITABLE
  `common/` of links to `/public_data/alphaworld02/protenix_root/common` -- it
  downloads missing files there. `LAYERNORM_TYPE=torch` avoids compiling its
  CUDA LayerNorm; `-c` is total passes.
- **OpenFold3**: source at `/home/hwlee/project/openfold3` (preview-2 weights
  `/home/hwlee/.openfold3/of3-p2-155k.pt`) or the v0.5.0 tree under
  `refs/uplifting-biomolecular-modeling/openfold3_ob0/stock/src` with
  `of3-ob-2025-06-30-174k.pt` from `openfold3-data.s3.amazonaws.com`. Our venv
  plus `pdbeccdutils kalign-python ijson memory_profiler func_timeout awscrt
  gemmi` on `PYTHONPATH`. A capture script must guard its CLI call with
  `__main__`: the data loader SPAWNS workers that re-import it.
- **RoseTTAFold3**: foundry `4010e3e` (tarball under the reference's
  `rosettafold3/stock/`), `rf3_foundry_01_24_latest_remapped.ckpt` from
  `files.ipd.uw.edu`, plus `atomworks==2.2.1` and **`biotite==1.4.0`** on
  `PYTHONPATH` -- atomworks needs the older biotite.
- **IntelliFold-v2**: `pip install intellifold==2.0.4` to a directory,
  `intellifold_v2.pt` and `ccd_v2.pkl` from HF `intelligenAI/intellifold`, and
  its PyTorch runner `runner/intellifold_inference.py --model v2`. Its summary
  writer passes numpy scalars to `json.dump`; wrap the encoder, not the model.

### Chai-1 against its release -- resolved 2026-09-24

On 5I28 with no alignment, 200 steps, five samples, both measured the same way
(mean of the CIF B-factors):

| | pLDDT per sample | CA vs release sample 0 |
|---|---|---|
| released Chai-1 | 95.00 - 95.17 | 0.18 - 0.69 A (its own spread) |
| FoldForge Chai-1 | **95.51 - 95.55** | **0.195 - 0.243 A** |

An earlier "97.45" for the release came from a different aggregation and was
compared against our B-factor mean; it overstated the gap by two points.

**Seven defects, none visible to the tree diff.** In the order found, with the
5I28 pLDDT each left behind (ours started at 41.87):

| defect | fix | pLDDT |
|---|---|---|
| single read the block's own updated pair | parallel block reads the pair ENTERING it | |
| single-attention gate had no offset | `single_attention_gate_bias = 1.0` (`sigmoid(g + 1)`) | |
| single-attention residual unmasked | `mask_single_attention_residual` | 54.93 |
| MSA stack projected the raw recycle carry | it reads the single AFTER its recycle add | 64.26 |
| MSA transition read the post-attention MSA | the row update is parallel too | 64.29 |
| with no alignment the query still formed the MSA, so the profile was a one-hot of the sequence | `empty_msa_without_alignment`: every row masked, profile and deletion mean zero | 77.66 |
| **the loader swapped every parallel block for the shared SEQUENTIAL one** | `install_pairformers` keeps a block that declares its own schedule | **95.53** |

Two of these were later REMOVED (2026-09-25). Once the loader bug was fixed,
resetting `mask_single_attention_residual` changed no output bit on any input
(padded rows are masked downstream), and `empty_msa_without_alignment` moved
Chai-1 by 0.4 pLDDT -- its 13 points above were measured on a trunk the loader
had already broken. See "Unified conventions" below.

The last is the one to remember. The first three fixes were written into
`PairformerBlock.forward`, and `install_pairformers` then replaced that block
with team-gm's sequential composition in every real fold -- only the gate
offset survived, because it lives inside `SelfAttention`. A harness that built
the model WITHOUT the loader matched the release to 0.9% after 48 blocks and
three recycles, while the real fold read corr 0.93. **A comparison harness
must load the model exactly as the CLI does**, or it validates a graph that
never runs.

The reference conformer is the one input still different: chai caches its own
(`conformers_v1.apkl`, deterministic for standard residues), and against ours
the intra-residue distances differ by 1.1 A median. It accounts for the whole
of the atom encoder's input difference -- swapping the release's positions in
takes the conditioning from 16% error to 0.09% -- but is worth 0.03 pLDDT
(95.53 -> 95.56), so it is left.

**How it was found, and what to reuse.**

- **The traced release is readable.** `torch.jit.load(...).forward_256.code`
  is the whole inlined graph, 22,644 lines for the trunk; stripping the
  attribute fetches leaves the arithmetic. `code_with_constants` gives the
  literals (`CONSTANTS.c0` is the gate's 1). Every convention above was read
  from there, not guessed.
- **Load it as `chai_lab` does**: `torch.jit.set_fusion_strategy([("STATIC", 0),
  ("DYNAMIC", 0)])`, `torch.jit.load(path).to(device)` -- `map_location` leaves
  traced CPU constants behind -- and feed floating inputs in bf16.
- **Bisect by weight surgery.** The traced trunk has no per-block entry point,
  so zero every residual output projection of blocks >= K
  (`transition_pair.linear_out`, `triangle_multiplication.linear_z_out`,
  `triangle_attention.linear_out`, `transition_single.linear_out`,
  `attention_pair_bias.attention.output_proj`) and compare with our first K
  blocks on the release's captured inputs.
  `runs/chai-native-20260923/tools/trunk_bisect.py` does K = 0..48 and three
  recycles.
- **Compare values, not names.** Cutting every tensor into vectors along each
  axis (rounded to bf16) and looking them up on the other side matches through
  splits, merges and transposes; only fused tensors stay unmatched
  (`out_scalers * linear_out`, `query_bias` as a q bias).
- **Invert a projection to reach an unexported tensor.** The initial single is
  `proj_in_trunk(cat[pooled, TOKEN])`; with `TOKEN` captured, a least-squares
  solve recovers the release's pooled atom-encoder output.
- **A separable error is inherited.** 98% of the initial pair's error was
  row + column, i.e. `left_single + right_single` of a wrong single, which sent
  the search upstream instead of into the pair.

### Every family against its own release -- 2026-09-25

5I28, no alignment and no template, the same number of trunk passes (4) and
200 steps on both sides, 5 samples. pLDDT is the per-atom mean; "CA to release"
is the mean pairwise CA RMSD from each of our samples to each release sample,
set against the release's own sample-to-sample spread. Scripts and outputs are
under `runs/release-compare-20260924/`.

| family | release pLDDT | ours | CA to release | release spread | verdict |
|---|---|---|---|---|---|
| boltz2 | 65.52 | 66.56 | 2.97 A | 3.00 A | matches |
| boltz2, same MSA | 97.04 | 96.93 | 0.63 A | 0.67 A | matches |
| protenix v1 | 68.43 (fp32) | 68.32 | 1.67 A | 1.46 A | **fixed**, matches |
| protenix v2 | 62.52 (fp32) | 62.28 | 2.70 A | 2.57 A | **fixed**, matches |
| rosettafold3 | 72.93 | 72.90 | 2.15 A | 2.23 A | matches |
| intellifold2 | 61.39 (fp32) | 60.59 | 3.00 A | 2.90 A | matches |
| openfold3 (v0.5.0 OpenBind) | 65.94 | 68.01 (66.79 on the release's conformer) | 2.78 A | 2.59 A | matches; conformer |
| openfold3-preview2 | 53.36 | 48.42 (53.97 on the release's conformer) | 3.95 A | 2.44 A | model matches; conformer |
| chai1 | 95.0-95.2 | 95.5 | 0.20-0.24 A | 0.18-0.69 A | **fixed**, matches |

**Found and fixed on the way, both invisible to the tree diff:**

- **Protenix's template term was missing.** The fused template embedder
  skipped absent slots and divided by the present count, so a template-free
  fold got exactly zero. The release runs every slot -- the normed query pair
  still drives the stack -- and divides by the slot count, adding a pair term
  of RMS 17 (v1) / 12 (v2) on every recycle. v1 went from CA 4.25 A off the
  release to inside its spread.
- **Recycle counting.** Protenix (`range(N_cycle)`), OpenDDE, which inherits
  it, and RF3 (`range(n_recycles)`) count TOTAL trunk passes, like Chai-1;
  AF3, Boltz, OpenFold3 (`num_recycles + 1`) and IntelliFold
  (`recycling_iters + 1`) count additional ones. `recycles_are_total` now says
  so for each, so one setting means one number of passes.

**Three things that look like defects and are not:**

- **Precision.** Protenix, IntelliFold and Chai-1 default to bf16. Switching
  the RELEASE alone between bf16 and fp32 moves its pLDDT by 1.5-4 points
  (Protenix v1 69.93 -> 68.43, v2 58.35 -> 62.52, IntelliFold 65.50 -> 61.39).
  Compare against the precision we run, not the release's default.
- **Reference conformers.** OpenFold3 generates a fresh RDKit conformer per run
  and rotates it at random; the atom encoder reads the offsets, so its input is
  stochastic. Ours is one cached ETKDGv3 conformer per residue, ~1.2 A off the
  release's intra-residue distances. Swapping the release's conformer into our
  batch closes both OpenFold3 gaps (preview-2 48.42 -> 53.97 against 53.36);
  across five release seeds preview-2 itself ranges 53.2-57.1. Chai-1 and
  IntelliFold do not move on the same swap.
- **Release reporting.** IntelliFold's CIF B-factor is not its per-atom
  pLDDT (64.39 against 61.39 in its own JSON), and it reports one pLDDT for
  all five samples. RF3 defaults to 50 steps and abandons a fold below pLDDT
  0.5 (`early_stopping_plddt_threshold`); pass `num_steps=200
  early_stopping_plddt_threshold=0` to compare.

**How each was localised** -- the same three moves as Chai-1, and they
generalise: capture the release's stage outputs (a forward pre-hook copies the
INPUTS, since Protenix and OpenFold3 update in place -- read after the call, an
input is already the output), feed the release's own inputs to our model LOADED
BY THE CLI (`runs/release-compare-20260924/tools/ours_blocks.py` does a
teacher-forced per-block pass inside a real `inference.run`), and swap one input
at a time into a real fold.

### One architecture, implemented six ways -- why the graph is shared

Every family here is AF3's architecture. What differs between them is mostly
not design but HOW each team turned AF3's description into code -- and the
description disagrees with itself. The AF3 Supplementary Information's
pseudocode and AF3's released code differ in several places, the SI has at
least one outright typo, and each team resolved those points its own way.
Their weights were then trained on their resolution, so inference must
reproduce it exactly.

Measured by resetting one convention at a time to AF3's value and folding with
the same seed (bit-identical baselines, so every difference is that
convention's):

| convention | families | where it comes from | reset to AF3 |
|---|---|---|---|
| `msa_double_add` | Boltz-2, Chai-1 | **SI typo, implemented literally.** SI Alg. 1 line 10 `{z} += MsaModule(...)`, but Alg. 8 returns the UPDATED pair, so the input is counted twice. AF3's code assigns. The Protenix report lists it as an erratum (Table 1). Boltz-2: `z = z + msa_module(z)`; Chai-1: traced trunk. Neither report mentions it. | Boltz-2 -9 pLDDT |
| `msa_update_before_opm` | Boltz-2, OpenDDE | **Documented design change.** Boltz-1 report section 3.1 reorders SI Alg. 8 to PairWeightedAveraging -> MSATransition -> OuterProductMean; OpenDDE copies it ("Boltz-style MSA block"). | Boltz-2 -46, OpenDDE -18 |
| `transposed_column_pair_bias` | Protenix v1/v2, OpenDDE, OF3 preview-2, Boltz-2 | **SI vs AF3 code.** SI Alg. 15 (and AF2/OpenFold) use bias b_ki; AF3's code projects before transposing, b_ik. | -2 to -28 |
| `pre_trunk_atom_query` | Boltz-2, Chai-1, RF3 | **SI vs AF3 code.** SI Alg. 5 copies q = c (line 7) before adding the trunk single (line 9); AF3's code copies after. | Boltz-2 -43 / 19 A |
| `parallel_attention_transition` | Chai-1, RF3 | **SI as written.** Alg. 23 `a <- b + Transition(a)`, which the Boltz-1, Protenix and RF3 reports all call a problem. RF3's report says it switched to sequential residuals; its released config (`rf3_net.yaml`) keeps the parallel form. | Chai-1 5.5 A |
| `untransposed_column_pair_output`, `parallel_pairformer_block`, `atom_cond_norm`, `same_conformer_atom_attention`, `single_attention_gate_bias` | Chai-1 | **Undocumented**, read from the traced release. | up to -47 |
| reference-conformer pose | OpenFold3 (random), Chai-1, IntelliFold (fixed) | **SI vs AF3 code.** SI Table 5: ref_pos has "a random rotation and translation applied"; AF3's inference code poses nothing. | see below |

Three consequences, and they are the case for one graph:

- **Each of these is a reading of AF3, not a new architecture.** Written as a
  separate model per family, every one of them would be a hidden line in its
  own copy of the network, and nobody would see that six copies disagree about
  the same algorithm. As DenseSpec rows they are named, documented flags with
  the family and the source side by side, and a reset measures each one.
- **The bugs cannot be fixed in inference.** `msa_double_add` is an SI typo,
  and the weights learned around it; removing it costs Boltz-2 9 pLDDT. They
  are implemented as the releases have them, commented with their source, and
  worth reporting upstream.
- **Paper and code disagree even within one project** (RF3's diffusion
  transformer). Only the released code decides what the weights saw.

### Unified conventions -- 2026-09-25

Every convention that owns no parameter was reset to AF3's value one at a time,
per family, and folded with the same seed on three inputs: 5I28 (no MSA, low
confidence, chaotic), 3PTB (protein + Ca + benzamidine, MSA and template) and
1A1K (two DNA strands + protein). Baselines repeated bit-identically, so every
difference is that convention's. Runs: `runs/release-compare-20260924/abl/`,
scores `convention_table.txt`.

**Removed -- no output bit changed on any input**, and the code shows why each
is redundant: `key_masked_offsets` (7 families), `mask_single_attention_residual`,
`mask_atom_act_per_block`, `msa_pair_mask_logits`, `opm_sum_without_norm`
(Chai-1's grouped OPM never read it). Padded atoms, tokens and masked MSA rows
are masked again downstream, and masked MSA rows reach the pair only through
the outer product mean, which masks them itself.

**Unified to AF3 -- effect below the noise on all three inputs** (at most 0.2
pLDDT / 0.3 A on the stable inputs): `atom_key_window` (7 families; the pad,
q-block and circular windows are gone), `opm_clamped_norm` and `opm_bias_after_norm` (OPM is AF3's everywhere),
`triangle_mul_divide_by_length` (RF3), `template_stack_outer_residual` and
`template_visibility_by_coverage` (Boltz-2), `template_gap_uncovered` and
`template_mask_class` (Chai-1), `empty_msa_without_alignment` and
`adaptive_norm_eps` (Chai-1), and Boltz-2's sampler constants (gamma_0,
gamma_min, noise_scale, step_scale). `symmetric_bonds` was on this list too and
was wrong to be: see "Ligand geometry" below.

A same-seed reset measures how far ONE sample moves, not how widely the samples
spread, so every family was re-folded against its release after the change
(5I28, no alignment, 5 samples). All stayed inside the release's spread except
two sampler findings:

- **Chai-1's churn window and variance floor are kept.** Without them its
  samples spread 0.55 A against the release's 0.39 A.
- **Boltz-2 had Boltz-1's sampler.** The constants on its row (gamma_0 0.605,
  gamma_min 1.107, noise_scale 0.901, rho 8, step_scale 1.638) are boltz's
  `BoltzDiffusionParams`; Boltz-2 inference uses `Boltz2DiffusionParams`
  (boltz `main.py`): AF3's four constants and rho 7, with sigma_min 1e-4. With
  its MSA our samples were 0.245 A apart against the release's 0.665 A; on the
  Boltz-2 values they are 0.595 A apart and 0.560 A from the release (0.634
  before).

Chai-1 after the change: pLDDT 95.44 against the release's 95.11 (95.07
before), CA 0.43 A from the release against its 0.39 A spread -- the 0.4 pLDDT
is the unified `empty_msa_without_alignment`.

**Second round, judged on the distribution** (1A1K, 4 seeds x 5 samples per
arm, on vs off compared across seeds): `drop_atoms` (Boltz-2, Chai-1,
IntelliFold2, both OpenFold3, RF3) and `key_masked_atom_attention` (6 families)
left the distribution unchanged and were unified -- OXT is kept everywhere but
ESMFold2, whose fixed atom table has none. `empty_template_gap` also looked
inert on 1A1K but moved protenix1 on 5I28 by 1.6 pLDDT away from its release,
so it stays; `chained_atom_key_norm` moves the distribution (cross/within 1.2
to 1.7) and stays. The confidence selector turned out to be a label -- only
"boltz2" was ever read -- and is now the boolean `confidence_reembed_pair`.

**Kept -- a reset moved folds** (the table in "One architecture, implemented six
ways", plus conventions that only fire on some inputs: on 1A1K, Chai-1's
`parallel_msa_block` is worth 25 pLDDT, `template_coverage_mask` 24 and
`recycle_from_initial` 9). Also kept without a measurement: conventions that
act only on outputs these folds did not score (`distogram_mean_symmetrised`,
`confidence`, `pde_symmetrise`), and `template_present_denominator`, which a
fully populated 3PTB template stack never triggers.

### Reference conformers -- AF3's rule kept

MiniWorld's rule -- CCD model coordinates, centred per residue, randomly posed
-- was tried against every release. The GEOMETRY is harmless everywhere (CCD
model coordinates in AF3's pose change nothing measurable). The random POSE is
family-specific: OpenFold3 preview-2 gains 4.4 pLDDT, Chai-1 loses 2.2 and
IntelliFold moves 7 away from its release, because the first was trained with
random rotations and the others with none. AF3's rule stays.

MiniWorld also fills missing CCD model coordinates with 0.0 in training
(`convert.py`, `to_reference_features`) and inference (`inference/ccd.py`),
and sets `ref_mask` after the fill, so those atoms reach the model as valid
atoms at the raw origin -- 12 to 38 A from their residue (e.g. `DRP` O3P,
`Y7G` O1-O3; `4TJ` loses 45 of 72). About 4% of CCD components are partly
missing, 0.2% wholly; all sampled cases have ideal coordinates to fall back on.
Worth reporting to MiniWorld.

### Ligand geometry -- what the protein metrics missed (2026-09-25)

Every convention above was judged by pLDDT and CA RMSD. Both are PROTEIN
metrics: a ligand's own geometry can be wrong while neither moves. Scored by
3PTB's benzamidine bond lengths against each release (no MSA, no template,
5 samples; releases 0.01-0.03 A RMS off ideal), four faults came out, each
invisible on protein and each a different family's:

| Fault | Family | Benzamidine before -> after | Cause |
|---|---|---|---|
| `symmetric_bonds` unified to AF3 | Protenix, Boltz-2, RF3, OF3, OpenDDE | 0.80 -> 0.02 A (protenix2) | The OpenFold3 lineage trained on a token bond matrix with both [i, j] and [j, i]; AF3 has one direction. Bisected to 7b245fc; restored. |
| Every ligand bond written as single | Boltz-2 | 0.11 -> 0.01 A | Boltz-2 embeds bond ORDER; AF3's featurisation drops it, so benzene read as cyclohexane (1.52 A ring bonds). `bond_orders` now records it from the CCD component's sanitised RDKit molecule, as the release does. |
| Atom attention "within a token" | Chai-1 | 0.3 -> 0.04 A | The traced mask is within one reference CONFORMER. On a protein the two are the same; on a ligand each atom attended to itself alone. Now `same_conformer_atom_attention`. |
| Chain and entity relations folded into a bias | Chai-1 | (complexes) | The converter folded RelativeChain and RelativeEntity at single-chain values, so every cross-chain pair read as intra-chain. The Chai-1 blob now carries both as input columns (`scripts/convert_dense_checkpoint.py` unfolds them; the old blob is kept as `chai1.bin.zst.pre-chain-relations`). |

A fifth fault came from the same audit: ESM2 ran on every chain, so Chai-1's
ligand and nucleic tokens got a "protein of unknown residues" embedding where
the release gives zero rows. Only protein chains reach the tower now.

The Chai-1 RELEASE had silently dropped both of 3PTB's ligands (RDKit 2026 no
longer accepts `useChirality` on ETKDG parameters; the entity fails to tokenise
and the fold goes ahead without it). Pin `rdkit==2024.9.5`
(`runs/release-compare-20260924/chai-rdkit-deps`) or the control is protein-only.

RoseTTAFold3 stays 0.15-0.19 A against a release that itself spreads
0.01-0.16 A; its extra atom channels (`ref_pos_ground_truth`,
`has_atom_level_embedding`, the 384-d atom-level embedding) are all zero at
release inference, so they are not the cause.

After the fixes, against each release (no MSA, no template, 4 trunk passes,
200 steps, 5 samples; spread = mean pairwise CA RMSD):

| Family | 3PTB pLDDT rel / ours | 3PTB CA ours-rel (rel-rel) | Benzamidine bonds rel / ours | 1A1K pLDDT rel / ours | 1A1K CA ours-rel (rel-rel) |
|---|---|---|---|---|---|
| Boltz-2 | 98.81 / 98.85 | 0.07 (0.07) | 0.01 / 0.01 | 97.43 / 97.28 | 0.51 (0.55) |
| Chai-1 | 98.06 / 98.21 | 0.15 (0.10) | 0.02 / 0.03 | 95.51 / 96.36 | 0.56 (0.44) |
| OpenFold3 p2 | 29.83 / 29.42 | 16.5 (16.5) | 0.01 / 0.01 | 86.21 / 85.83 | 0.84 (0.88) |
| OpenFold3 v0.5 | 34.21 / 33.97 | 15.5 (14.9) | 0.02 / 0.02 | 77.61 / 77.55 | 0.72 (0.60) |
| IntelliFold2 | 31.11 / 30.73 | 12.3 (12.6) | 0.03 / 0.02 | 85.03 / 84.80 | 1.31 (1.00) |
| RoseTTAFold3 | 69.95 / 69.85 | 14.8 (14.8) | 0.02 / 0.02 | 84.63 / 84.54 | 0.72 (0.31) |
| Protenix v1 | 32.59 / 33.27 | 11.4 (12.3) | 0.03 / 0.04 | seed-dependent, see below | 12.6 (11.0) |
| Protenix v2 | 37.71 / 37.42 | 10.6 (8.6) | 0.14 / 0.07 | seed-dependent, see below | 6.75 (5.85) |

Without an MSA 3PTB is low-confidence for every family but Boltz-2 and Chai-1,
in the release as in ours, so the protein columns there only say the two arms
are equally lost.

Still open:

- ~~RoseTTAFold3's benzamidine~~ -- **resolved**: 0.11 -> 0.02 A (release
  0.02), 3PTB pLDDT 69.85 against 69.95. Its atomworks featurisation names an
  atomised token's atoms by ELEMENT (`use_element_for_atom_names_of_atomized_
  tokens: true`), so a ligand reads as C, C, ..., N, N; fed the CCD names
  C1..C6, C, N1, N2, the atom-name embedding saw names it never trained on.
  Now `DenseSpec.element_ligand_names`. Located by feeding our diffusion the
  release's own trunk (still 0.11, so the diffusion), then one denoiser call on
  the release's exact input, where the release's atom names would not match
  ours. It also writes a ligand's MSA column as UNK in the query row only
  (`nonprotein_msa_as_query="query"`, vs the Protenix lineage's "rows");
  that one moved pLDDT 0.3 and not the bonds.
- **RoseTTAFold3's chiral-centre input is dead here.** The release feeds 669
  protein chiral rows on 3PTB; FoldForge featurises none, so
  `atom_chiral_features` embeds zeros.
- ~~Protenix on 1A1K~~ -- **resolved**, see "Protenix on 1A1K: two inference
  conventions" below.
- The Protenix control must write identical ions as ONE entry with a count.
  Three `count: 1` entries are three entities to Protenix, one to AF3's
  featurisation; that alone moved the v1 release by 8 pLDDT on 1A1K.

### Protenix on 1A1K: two inference conventions (2026-09-25)

The Protenix pLDDT gap on 1A1K (78 against the release's 86 for v2, 81 against
73 for v1, uniform over every chain) was settled in three steps:

1. **Same structures, our head** (`tools/score_release.py` swaps the diffusion
   samples for the release's coordinates before the confidence head): our
   head still gave OUR numbers. Boltz-2, the control, gave the release's to the
   decimal. So the gap was not the structures.
2. **The release's trunk too**: fed the release's final single and pair, our
   head reproduced the release's per-chain pLDDT exactly. So the head is right
   and the TRUNK differed -- by rel 0.20 at the template's input.
3. **Clean z_init decomposition**, with a pre-hook copy (the release updates
   its pair in place): two causes.

- **MC dropout at inference.** Protenix (v1 and v2; not OpenDDE) flips one coin
  per seed, `random.random() < mc_dropout_apply_rate` (0.4), and when it lands
  every pass adds the pair recycle under `F.dropout(p=mc_dropout_rate=0.4)` --
  the dropped fraction was exactly 0.40 and the kept elements exactly 1/0.6.
  Seed 101, the one compared, had it on. Now `DenseSpec.recycle_mc_dropout`;
  the fold report records `recycle_dropout_applied`.
- **Non-protein MSA columns.** AF3 writes a ligand's MSA column as the gap,
  query row included; the Protenix lineage fills a non-protein chain's all-gap
  rows with its query, UNK for a ligand, and its profile follows. The zinc
  ions' profile was the gap where the release's is UNK, which moved every
  token's initial single through the outer sum (s_init rel 0.29 on the ions,
  0.002 elsewhere). Now `DenseSpec.nonprotein_msa_as_query` (Protenix v1, v2,
  OpenDDE).

After both, fp32, a seed with the dropout off: s_init rel 0.003, template input
0.001, trunk pair after four passes 0.0013. Across eight seeds each:

| 1A1K, v2 | dropout off (5 seeds) | dropout on (3 seeds) |
|---|---|---|
| release (101-108) | 80.2-81.0, mean 80.7 | 85.6, 87.6, 79.6 |
| FoldForge (0-7) | 80.64-80.68 | 79.8, 85.6, 79.0 |

A single-seed Protenix comparison is therefore a coin toss: compare seeds
with the dropout in the same state, or distributions.

**Score every entity by its own geometry, not by the protein around it.**
`runs/release-compare-20260924/tools/complex_compare.py` reports it per family.

### The one real defect this found

**The language model was dead.** `build_lm_inputs` selected protein tokens with
`mol_type == PROTEIN_MOL_TYPE`, which is 0, while the caller passed
`is_protein`, which is 1 on protein -- so it kept exactly the tokens it meant to
drop. On an all-protein input nothing was packed, the tower saw an empty
sequence, and the shim turned its zeros into ONE 256-vector repeated at all
16384 pair positions (1 distinct row of 16384, max deviation exactly 0). Every
ESMFold2 fold ran with no language model.

It was invisible because it still folded: `esmfold2` reached 0.99 A on 1UBQ
because its MSA encoder carried the structure by itself, and the constant even
measured like a contribution, 45% of the injection by RMS. What gave it away is
that the reference records disabling the LM dropout as worth ~18 A, and
disabling ours moved the fold by 0.1 A. **An input whose ablation changes
nothing is not connected, whatever it measures.**
`scripts/check_live_inputs.py` asks the two questions that catch this class:
does the tensor VARY, and does a linear read-out of a pair stream rank true
contacts above chance.

After the fix the LM pair carries 60 of 100 top-ranked long-range pairs as true
contacts, against 2 before and a 4.7% baseline, and the 6MRR numbers above are
what it buys.

The port also reproduces the release where the release runs. 1UBQ, no
alignment, five seeds: released ESMFold2 best 5.775 / mean 7.354 A, FoldForge
best 5.776 / mean 7.621.

## Acceptance per model

1. Pin model source, released configuration and weight conversion; require strict
   loading, with no unexplained missing or unexpected parameters.
2. Check masks, residual ownership, normalization and precision at module level.
3. Run sequence/features through the complete checkpoint, with coordinates and
   confidence compared against the corresponding reference under recorded settings.
4. Verify compilation, inference graph capture, cache coverage and timing on the
   complete model before reporting performance.

Only completed ports appear as implemented in `foldforge models`; the other
names remain explicitly unported.

## Verified results — 2026-09-13

A6000, 1UBQ (76 residues), released ESMFold2 weights, native BF16 with FP32 norm
parameters, no autocast, MSA depth 512, seed 0, 3 recurrence loops, 14 diffusion
steps, 1 sample. Compile and CUDA graphs are off in this correctness run.

| Folding backend / LM input | pLDDT (0–1) | CA RMSD to deposited 1UBQ | CA RMSD to saved original reference |
| --- | --- | --- | --- |
| MiniWorld / cached ESMC | 0.8008 | 0.699 Å | 0.665 Å |
| MiniWorld / live ESMC-6B | 0.7969 | 0.684 Å | 0.633 Å |
| PyTorch / same cached ESMC | 0.7969 | 0.677 Å | not measured |

MiniWorld versus PyTorch under the same cached-LM setup: **0.203 Å CA RMSD**.
Every comparison matches all 76 residues, with zero residue-name mismatches.
The saved original reference used FP32 folding; its comparison is structural
validation across precision settings, not an exact numerical parity test.
These results establish one monomer case; complexes, ligands, larger targets,
full-model compile/graph performance, and model-shape cache completeness remain
outside this qualification.

Jobs: 1682652 (MiniWorld/cached LM), 1682655 (public CLI/live ESMC),
1682656 (PyTorch/cached LM), 1682658 (final tests and corrected CA rescoring).
**23 tests pass**, including the actual GPU diffusion-conditioning path,
ESMC RoPE/precision, CB-vs-CA scoring, and occupancy-free prediction CIF parsing.
Ruff passes for the new CLI, precision, LM, inference and evaluation code and
new tests. The live ESMC forward ran for 2.471 s; that is not an end-to-end timing
or a comparative performance benchmark.

Artifacts are under `validation/results/connection-20260913/{cached,computed,pytorch}/`.
Each contains `1ubq.cif` and the corrected `1ubq.json`. Logs are retained under
`validation/logs/esm2-*.{log,err}`. Failed probes are retained separately.
All consumer edits remain local and uncommitted; existing unrelated changes
and the team-gm member branch are preserved.

## AF3, Protenix and OpenDDE commands

Run on an allocated GPU node after `scripts/setup_env.sbatch` completes.
Input JSON must carry prepared MSA/template data; these commands do not run an
MSA search. Protenix/OpenDDE use their upstream sequence/job JSON schema, whereas
AF3 uses the official AF3 JSON schema. Example input files are in
`validation/inputs/data/1ubq/{af3-full,af-family}.json`.

```bash
source scripts/activate_env.sh
# Once for all models, independently of their checkpoints:
foldforge ccd prepare \
  --components model_checkpoints/opendde/common/components.cif \
  --rdkit model_checkpoints/opendde/common/components.cif.rdkit_mol.pkl \
  --out data/ccd/preprocessed_CCD.lmdb
foldforge ccd verify --ccd-db data/ccd/preprocessed_CCD.lmdb

foldforge fold af3 --input validation/inputs/data/1ubq/af3-full.json \
  --checkpoint model_checkpoints/af3/af3.bin.zst \
  --ccd-db data/ccd/preprocessed_CCD.lmdb --samples 1 --out runs/af3
foldforge fold protenix --input validation/inputs/data/1ubq/af-family.json \
  --checkpoint model_checkpoints/protenix/protenix_base_default_v1.0.0.pt \
  --ccd-db data/ccd/preprocessed_CCD.lmdb --samples 1 --out runs/protenix
# For v2, change both --variant and --checkpoint:
foldforge fold protenix --variant protenix-v2 \
  --input validation/inputs/data/1ubq/af-family.json \
  --checkpoint model_checkpoints/protenix/protenix-v2.pt \
  --ccd-db data/ccd/preprocessed_CCD.lmdb --samples 1 --out runs/protenix-v2
foldforge fold opendde --input validation/inputs/data/1ubq/af-family.json \
  --checkpoint model_checkpoints/opendde/opendde.pt \
  --ccd-db data/ccd/preprocessed_CCD.lmdb --samples 1 --out runs/opendde
```

Defaults are MiniWorld, native BF16, 10 recycles, 200 diffusion steps and 5
samples. Use `--backend pytorch` for the corresponding ported reference equations.
Protenix/OpenDDE templates are opt-in (`--templates`). `--no-msa` is a smoke-test
option. All commands use `--ccd-db` (default `FOLDFORGE_CCD_DB`, otherwise `data/ccd/preprocessed_CCD.lmdb`).
The former model-specific `--assets` options are replaced. The database is MiniWorld-compatible BioMol LMDB. Per-record metadata preserves
full CCD categories and prepared references, with a lazy AF3 mapping over those
same records. There are no runtime CIF/pickle database fallbacks. See
[MiniWorld formats](MINIWORLD-FORMAT.md) for the standard YAML entry point;
the JSON commands below remain explicit checkpoint-compatibility interfaces.

Each command writes per-sample CIF, raw model confidence and JSON run settings.
`*.prediction.pt` contains the common `Prediction` fields as CPU tensors: flat
sample-major coordinates, token pLDDT in [0,1], and token PAE in Angstrom.
Protenix/OpenDDE decoded atom pLDDT is already in [0,1] and is averaged over
each original token. A redundant division by 100 in the initial common payload
was corrected during the unification audit; raw heads, summary pLDDT and CIF
coordinates were unaffected. Original payloads were backed up and regenerated. Undefined heads
remain `None`; AF3 pTM aggregation is not yet exposed in this common payload,
while its raw confidence output is retained. AF3 writes raw arrays as NPZ;
Protenix/OpenDDE use a tensor dictionary.

## Shared operations and precision

`team_gm.modules.checkpoints.af_family` maps plain residual transitions, compatible residual
TriMul and LayerNorm onto miniworld-engine. It preserves masks and consumes the
fused residual exactly once. AF3 interleaved projection weights and incoming
contraction order are mapped explicitly. Protenix template TriMul with pair
width 64 and hidden width 128 retains its original PyTorch equation because the
current engine requires equal widths. Triangle attention, MSA/outer-product,
conditioned composites and model-specific heads retain model/PyTorch equations;
shared normalization may still use the engine inside them.

Learned parameters are native BF16 except FP32 normalization parameters. Fixed
Fourier, geometry and confidence-bin buffers retain their original precision.
No autocast is used. Explicit FP32 attention/head calculations remain where the
model equations require them, with casts at projection boundaries. This is a
parameter-precision policy, not a claim that every intermediate is BF16.

Full inference found and fixed the Xfold sampler overwriting its sampled noise
field, mixed-dtype attention/head contractions, and obsolete Biotite private
bond-array mutation. Strict AF3 conversion also checks source keys, shapes and
stack cardinality. Tests exercise nonzero projection weights, arbitrary masks,
incoming/outgoing equations, single residual ownership, native BF16 attention,
normalization without affine parameters, and the random-noise regression.

## AF-family verification — 2026-09-13

A6000 on gpu03, Slurm job 1682660. Complete 1UBQ (76 residues), seed 0, prepared
MSA, templates disabled, 10 recycles, 200 diffusion steps, 1 sample. Both backends
use identical native BF16 policy, compile OFF and CUDA graphs OFF. All CA
comparisons match 76 residues with zero residue-name mismatches.

| Model | MiniWorld pLDDT (0–100) | PyTorch pLDDT | MiniWorld vs deposited CA RMSD | PyTorch vs deposited CA RMSD | MiniWorld vs PyTorch CA RMSD |
| --- | ---: | ---: | ---: | ---: | ---: |
| AF3 | 91.233 | 91.211 | 0.709 Å | 0.696 Å | 0.058 Å |
| Protenix v1 | 94.079 | 94.075 | 2.579 Å | 2.552 Å | 0.138 Å |
| Protenix v2 | 93.347 | 93.326 | 2.702 Å | 2.720 Å | 0.073 Å |
| OpenDDE | 94.816 | 94.832 | 1.025 Å | 1.019 Å | 0.049 Å |

These compare MiniWorld to the ported PyTorch equations, not to a bit-exact
original JAX/FP32 upstream run. Protenix's approximately 2.6–2.7 Å deposited
structure difference is present in both backends. One monomer does not qualify
complexes, ligands, training, large targets or other GPUs. The supported June
2025 Protenix v1 configuration has not received the full comparison shown above.

The locked environment was rebuilt and checked successfully, including its AF3
native feature extension. The public CLI then passed two-sample smoke inference
for AF3, Protenix v1 and OpenDDE. Full results and `comparison.json` are under
`validation/results/ports-20260913/`; logs are under `validation/logs/ports-*`.
Final full suite: **47 tests passed**, with five missing-shape autotune warnings
(`validation/logs/ports-final-tests2-20260913.log`). New integration code passes
Ruff; `uv lock --check` and `uv pip check` pass (209 resolved / 175 installed
packages). Public two-sample payloads have coordinates `(2, 602, 3)`, token
pLDDT `(2, 76)` and PAE `(2, 76, 76)`, all finite. Confidence averaging and
explicit AF3 checkpoint filename selection have regression tests. The residual
audit classifies upstream raw-delta adds by full model-relative path. Failed
probes are preserved.

Some new model shapes miss tuned autotune entries and use the engine's heuristic
candidate selection. Cache coverage, hot timing and full-model compile/graph
qualification remain separate work. Cold elapsed times include compilation and
are not speedup measurements.

The root team-gm branch is `exp/miniworld-integrated`; its engine connection is
in place. The single transition inside Pairformer now forwards the requested
backend in both team-gm copies, with a passing root regression. FoldForge stays
on `main` and its team-gm member stays on `exp/miniworld`. Consumer changes remain
local and uncommitted; existing unrelated edits are preserved.


## Proposed result storage contract (not implemented)

Resolve the output root independently of the source checkout. All permanent
prediction artifacts should live under the deployment's top-level `runs/`, even
when executing a source snapshot. Use one unique directory per run; refuse a
nonempty destination unless an explicit, validated resume is requested.

For grouped benchmarks or seed sweeps, use a single level of named cases:

```text
runs/<run-id>/
  run.json
  af3-miniworld_graph-t7-d19/
    sample-000.cif
    sample-001.cif
    confidence.npz
    result.json
  logs/
```

`run.json` indexes cases and records source/checkpoint/input hashes and resolved
configuration. `result.json` records the two seeds, sample indices, confidence
summaries, timing scope, compile/graph observations, artifact paths and completion
status. The sample index is not an additional seed. Writes should use temporary
files and publish the completion manifest last, so interrupted output is not
mistaken for a successful run.

Default artifacts are mmCIF coordinates, compact common confidence arrays and
JSON metadata. Full PAE/PDE/distogram matrices, raw model dictionaries, input
feature tensors and intermediate trajectories should be explicit diagnostics,
not unconditional duplicate `.pt`/`.npz` output. Use numeric arrays with documented
axes, units and masks; reading outputs should not require model-class pickles.

Compiler/autotune caches belong in a shared versioned cache location. Disposable
source snapshots belong in job-local scratch, with the exact source archive/hash
retained only as needed for provenance. Existing `.bench` artifacts must be
inventoried and migrated with their manifest paths intact before any deletion.

## Optional diagnostic images (implemented)

All four inference adapters use the same image writer. PNG export is disabled by
default. Both CLI forms (`--spec` and legacy `--input`/`--input-spec`) accept:

```bash
foldforge fold af3 --spec input.yaml --config runtime.yaml --save-images all
# Or select a subset:
foldforge fold af3 --spec input.yaml --save-images pae pde plddt msa
```

The equivalent runtime YAML (also supported as `output` in notebook requests) is:

```yaml
output:
  images: [pae, pde, distogram, plddt, msa, template]
```

CLI selection overrides the YAML selection. Empty `images: []` is the default.
Images are saved beside the existing CIF/JSON artifacts under
`runs/<run>/images/<target>/`; the result JSON records relative paths and reasons
for requested images that were unavailable. This addition does not implement the
proposed output migration above or change existing CIF filenames.

- `sample-000-pae.png`, `sample-000-pde.png`: token-pair errors in angstroms, one
  image per diffusion sample, with a common scale across samples for each head.
- `sample-000-plddt.png`: token pLDDT on the 0–100 scale, one image per sample.
- `distogram.png`: trunk distance distribution summarized as the **most likely
  distance-bin index**, not an angstrom-valued distance map. Bin definitions can
  differ between checkpoints; do not compare these colors as physical distances.
  Saved once per trunk run. ESMFold2's optional distogram head is enabled only
  when this image is requested; other models retain their already-computed logits.
- `input-msa.png`: match/mismatch to the prepared query row, with gaps and padding
  blank. Displays at most the first 1,024 valid rows without reordering them.
  `input-msa-coverage.png` counts non-gap entries across **all** prepared rows.
  These show the input alignment, before stochastic internal MSA row sampling.
- `input-template.png`: resolved-atom coverage on query tokens for nonempty input
  templates. Empty template slots are excluded. If no templates were supplied or
  the checkpoint has no template path (ESMFold2), JSON records the omission.

Input images exclude padded tokens/rows; confidence images use decoded unpadded
outputs. The two seeds continue to control model/input randomness: rendering does
not draw from those RNG streams. Input image preparation precedes the timed
forward, and CPU PNG rendering follows it. Requested head computation and retention
can add inference work/memory (especially ESMFold2 distogram); keep image options
identical when comparing runtimes. Full optional distogram logits are not added to
raw result archives merely to save their PNG. Matplotlib is a declared runtime
dependency and imported only when rendering.
