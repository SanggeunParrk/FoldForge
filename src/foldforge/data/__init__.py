"""Input features: sequence, MSA, templates, reference conformers.

One tokenisation and one feature contract for every predictor, so a target is
prepared once and folded by all of them. Where an upstream predictor ships its
own builder (ESMFold2 has ``ESMFold2InputBuilder``), the adapter that maps it
onto this contract lives in that model's package, not here — this package holds
only what is genuinely shared.
"""
