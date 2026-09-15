"""Shared data preparation for every predictor.

CCD access, chemistry tables, feature construction, input contracts, MSA search,
and templates each have one package. Dataset loading and pipeline orchestration
live directly in this package; notebook integrations are isolated in ``web``.
"""
