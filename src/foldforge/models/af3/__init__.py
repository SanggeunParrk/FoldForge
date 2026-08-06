"""AlphaFold 3 (DeepMind).

**Not ported.** Weights are access-gated by DeepMind and have to be requested
per-user, so this is the one predictor whose port cannot be unblocked by a
download.

Record here, as the port is made: which of team-gm's blocks map onto the
upstream stack unchanged, where the input featurisation diverges from
``foldforge.data``, and any place the released weights disagree with the paper.
AF3 is the architecture team-gm's representative blocks were written against, so
this port is the one that tests whether the shared-block premise holds — treat a
block that needs a FoldForge-local variant as a finding, not a workaround.
"""
