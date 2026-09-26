# ESMFold2: `from_pretrained` can run with a randomly initialised ESM-C, silently

**Upstream:** https://github.com/Biohub/esm (`esm` 3.3.0) and the ESMFold2
model in the Biohub `transformers` fork (4.57.6)

## Summary

`ESMFold2Model.from_pretrained(path)` loads its ESM-C language model from
`config.esmc_id` (`biohub/ESMC-6B`) through the Hugging Face cache. With the
cached snapshot `af1602ba7406f521b11bf8f81d52af378cde09e4`, **none** of the
checkpoint keys match `ESMCModel`, every weight is freshly initialised, and the
only signal is a transformers warning:

```
Some weights of ESMCModel were not initialized from the model checkpoint at
biohub/ESMC-6B and are newly initialized: ['embed.weight',
'transformer.blocks.0.attn.k_ln.weight', ...]
```

Folding then proceeds normally on a random language model.

## Impact

On a 127-residue protein (PDB 5I28), three seeds x five samples:

| ESM-C | mean pLDDT |
|---|---|
| random (default `from_pretrained`) | ~29 |
| real weights (`load_esmc=False`, then `model.load_esmc(local_dir)`) | ~92.5 |

With the random LM the rigid-align SVD in `modeling_esmfold2_common.sample`
also fails to converge on some seeds. Nothing else distinguishes the broken
run: the structures and confidences are written and look like a hard target.

## Reproduce

```python
m = ESMFold2Model.from_pretrained("ESMFold2")          # warning only
m2 = ESMFold2Model.from_pretrained("ESMFold2", load_esmc=False)
m2.load_esmc("path/to/ESMC-6B")                         # correct
```

Compare the language-model hidden states (a forward pre-hook on
`language_model`): the first is uncorrelated with the second.

## Expected

A mismatch between the ESM-C checkpoint and `ESMCModel` should raise, not warn:
a folding model without its language model is a different model.

## Workaround

Load ESM-C explicitly from a verified local directory, as above.
