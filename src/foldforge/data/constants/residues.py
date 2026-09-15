# Copyright 2024 ByteDance and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Shared protein, RNA and DNA residue indices for flat-atom features."""

PRO_STD_RESIDUES = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "UNK": 20,
}

RNA_STD_RESIDUES = {
    "A": 21,
    "G": 22,
    "C": 23,
    "U": 24,
    "N": 25,
}

DNA_STD_RESIDUES = {
    "DA": 26,
    "DG": 27,
    "DC": 28,
    "DT": 29,
    "DN": 30,
}

STD_RESIDUES = PRO_STD_RESIDUES | RNA_STD_RESIDUES | DNA_STD_RESIDUES
