"""Checkpoint layout adapters built from team-gm blocks and engine operations.

``dense`` handles token-by-atom AF3 layouts; ``sequence`` handles the ESMFold2
sequence/atom topology. Shared equations and sampling are owned by team-gm.
The architecture assemblies and checkpoint readers live outside this package.
"""
