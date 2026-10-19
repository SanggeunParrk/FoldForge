"""Checkpoint layout adapters built from engine blocks and engine operations.

``dense`` handles token-by-atom AF3 layouts; ``sequence`` handles the ESMFold2
sequence/atom topology. ``install`` converts released weights onto the engine blocks;
``math`` holds the shared equations. Sampling is owned by the engine.
The architecture assemblies and checkpoint readers live outside this package.
"""
