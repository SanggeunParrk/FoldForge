# af3-any-model ESMFold2 converter: nucleic classes mapped in the wrong order

**Upstream:** https://github.com/sokrypton/alphafold3, branch `af3-any-model`
(the ESMFold2 key map used by the checkpoint converter)

## Summary

ESMFold2 numbers its polymer classes

```
0 unused, 1 gap, 2..21 amino acids, 22 UNK,
23 A, 24 G, 25 C, 26 U, 27 N, 28 DA, 29 DG, 30 DC, 31 DT, 32 DN
```

AF3 numbers them

```
0..19 amino acids, 20 UNK, 21 gap,
22 A, 23 G, 24 C, 25 U, 26 DA, 27 DG, 28 DC, 29 DT, 30 N
```

The converter handles the gap correctly (ESM 1 -> AF3 21) but lays ESM classes
23..31 onto AF3 22..30 **in order**. The unknown ribonucleotide sits between RNA
and DNA in ESMFold2 and last in AF3, so every DNA class receives the weights
of the class before it:

| AF3 class | receives ESM class | should receive |
|---|---|---|
| DA (26) | 27 N | 28 DA |
| DG (27) | 28 DA | 29 DG |
| DC (28) | 29 DG | 30 DC |
| DT (29) | 30 DC | 31 DT |
| N (30) | 31 DT | 27 N |

Affected: the restype and profile columns of `left_single`, `right_single`,
`single_activations`, `extra_msa_target_feat`, and the MSA one-hot columns of
`msa_activations` (verified column by column against the release safetensors).

## Impact

Protein-only folds are unaffected. On a DNA duplex bound by a zinc finger (PDB
1A1K), the converted model folds the duplex 9.9 A (CA/C1' RMSD) from the
released ESMFold2, against the release's own seed-to-seed spread of 0.9 A; with
the columns reordered it reads 0.96 A and matches the release's pLDDT
(94.40 vs 94.38).

## Fix

Map ESM `[23, 24, 25, 26, 28, 29, 30, 31, 27]` onto AF3 `22..30`, i.e. AF3
nucleic class `c` takes ESM class `[23, 24, 25, 26, 28, 29, 30, 31, 27][c - 22]`.
Any code that widens AF3 31-class features to ESMFold2's 33 needs the same
permutation.
