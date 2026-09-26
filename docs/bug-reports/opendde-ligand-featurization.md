# OpenDDE: inference fails on any input containing a ligand or ion

**Upstream:** https://github.com/aurekaresearch/OpenDDE (checked at
`a72e9f655231660f8af0072dbdb8b2a54f3fbd3c`, 2026-07-05)

## Summary

`runner/batch_inference.py pred` raises in the template featuriser for every
input that has a non-polymer entity (ligand or ion), **even with templates and
MSA disabled**. Protein-only inputs work.

## Reproduce

Input: 1A1K (two DNA strands, a zinc finger, three Zn2+) or 3PTB (trypsin +
benzamidine + Ca2+), in the Protenix JSON format. Both fail; 5I28 (protein
only) runs.

```
python runner/batch_inference.py pred -i input.json -o out -s 0 -c 4 -p 200 -e 5 \
  -n opendde_v1 --use_msa false -d fp32 \
  --trimul_kernel torch --triatt_kernel torch --load_checkpoint_path opendde.pt
```

```
File "opendde/data/inference/infer_dataloader.py", line 151, in process_one
File "opendde/data/template/template_featurizer.py", line 326, in make_template_feature
    std_idxs = map_to_standard(ca.asym_id_int, ca.res_id, template_meta_infos)
File "opendde/data/msa/msa_utils.py", line 464, in map_to_standard
ValueError: map_to_standard could not map residues to the standardized coordinate
system (unmapped ids: ['3-1', '4-1', '5-1'], known asym_ids: ['0', '1', '2']).
```

The unmapped ids are exactly the non-polymer chains (asym 3, 4, 5 = the ions).

## Expected

Ligand and ion chains are skipped by the template featuriser (they have no
template rows), as in Protenix, from which this code path is derived.

## Likely cause

`make_template_feature` builds `ca` over every chain, including non-polymer
ones, but `template_meta_infos` only knows polymer chains, so
`map_to_standard` has nothing to map the ligand residues to. The featuriser is
also reached when templates are off.

## Workaround

None found short of patching; we validate OpenDDE on protein-only inputs.
