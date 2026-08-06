"""Scoring a prediction: against a deposit, and against another predictor.

Matching is on ``(chain, residue number)`` after an explicit chain map, never on
count equality — a crystal structure is missing its disordered residues, so a
count check silently reports nothing on anything but a fully ordered monomer.
"""
