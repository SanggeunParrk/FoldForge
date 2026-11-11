# The 5I28 benchmark

The model-level speed benchmark of FoldForge: every supported model on the engine against its Anthropic kit (and the vendor's stock code),
on **5I28 (azurin, 128 residues)**, one H100 80GB.

**Ownership.** The protocol was first written in miniworld-engine (`benchmarks/5i28.md`, commit `71c800225`, 2026-10-30; removed from the
engine tree by the 2.5.0 redesign). FoldForge owns it from 2026-11-11: this page is the protocol, FoldForge runs it, and the results
are recorded here. The engine commit keeps the earlier records (all nine models, Protenix v1/v2 at 10 trunk passes, the kit
forward-definition audit): `docs/records/5i28_*.md` and `anthropic_kits_fwd_definition_audit_20261029.md` in that commit.

## Why 5I28

5I28 is 128 residues long, exactly one token-bucket unit. FoldForge rounds tokens up to multiples of 128
(`models/bucketing.py::TOKEN_BUCKET_SIZE = 128`), so n identical chains give 128 n tokens, a whole number of buckets for every n:
the benchmark measures the model at 128, 256, ... 2048 tokens with **no padding rows** in the timed forward, and the size axis is
plain "n chains". A target that is not a multiple of 128 would time padded tokens and make the sizes of different targets
incomparable. One fixed target, one known sequence and one real MSA / template set also make every model and kit see the same input.

## Inputs (both sides, every model, every n)

- **Size:** n identical chains, n = 1, 2, 4, 8, 12, 16 (128 n tokens).
- **MSA:** the real MSA, always. FoldForge: `msa_db: ~/data/BioMolDB_20260224/a3m_16k.lmdb`, key `P0113359` (6790 rows); kits: the same
  rows as an a3m. There is no run without the MSA.
- **Templates:** 4 distinct real templates, always: `5SYD_A`, `6KOL_A`, `5YT7_A`, `6TFD_A`.
  - FoldForge: `template_db: .../template.lmdb`, key `P0113359`, `template_n: 4`. The LMDB stores every chain of an entry (20 records,
    6 distinct sequences). **Keep `template_dedup: true` (the default)**: `load_templates` then keeps one record per hit sequence,
    which gives these four. (FoldBench specs set `template_dedup: false`; a 5I28 spec must not.)
  - Kits: those 4, in each kit's own input form. For the Protenix-family kits (Protenix v1/v2, OpenDDE) use the whole 20-hit a3m;
    their search keeps one hit per sequence. Never pass a list that a kit's dedupe would shrink.
  - A kit run counts only when its log shows 4 real templates (`real=4/4`, `Found 4 templates`).
  - RoseTTAFold3 takes one coordinate set per chain: the same single template on both sides. ESMFold2 has no template input.
- **Sampling:** 4 recycles, 200 steps, 5 samples, bf16. The same seeds where the kit allows.

## Timing

One complete model forward (trunk passes, diffusion sampler, confidence heads), CUDA-synchronized before and after, excluding
featurization, loading and output writing. Report the median of the warm forwards; the first forward is cold.
- FoldForge: `model_seconds_warm`.
- Kits: their `PHASE ... fwd_s` (or the kit's equivalent span, per the kit forward-definition audit). Not the kits' wider
  "Model forward time" lines.
- FoldForge runs on prebuilt kernels and warm caches: no JIT compile inside a timed run.

## Reporting

- Show the ratio as **kit / FoldForge** (> 1 means FoldForge is faster), never FoldForge / kit.
- A number taken any other way (no MSA, no templates, fewer than 4 real templates, a wider timing span) is not a 5I28 number.
- Records are dated files (`docs/records/`, created with the first FoldForge record) and are not edited afterwards.

## Preflight before any sweep

Run n = 1 of every model on BOTH sides first and check: the checkpoint loads; the trunk passes equal the kit's; the templates and
MSA rows are the real ones; zero compiles in the timed region; every setting matched to the kit (never a FoldForge default).
Never edit a script a running queue executes.
