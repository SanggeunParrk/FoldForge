"""Blocks FoldForge owns — the ones team-gm deliberately does not carry.

Three layers hold code here (see
``libs/team-gm/docs/ARCHITECTURE.md``), and this package is the narrowest of
them. Deciding what belongs:

* an **op** — wraps a kernel, has one PyTorch reference → **miniworld-engine**
* a **representative AF3 block** — Pairformer, MSA module, template, diffusion
  transformer → **team-gm**, where every terminal shares one copy
* a block that is **specific to one model** → that model's package,
  ``foldforge.models.<name>/``, next to the thing that needs it
* a block **two or more FoldForge models need** that team-gm has no reason to
  carry → **here**

The last case is the only reason this package exists, so it should stay small.
A single-consumer block living here instead of in its model's package is the
common mistake: it reads as shared, and the next person changing it has to prove
to themselves that nothing else depends on it.

If this grows past a handful of files, mirror team-gm's ``blocks/`` vs
``layers/`` split rather than letting it become a flat drawer.
"""
