"""HierE2E (Rangapuram et al., ICML 2021) ported to PyTorch, plus a DSF variant.

Self-contained: imports the core (data loading, splits, CRPS, scoring, the DSF
transform) and is never imported by it. See NOTES.md for what was ported and
where the port departs from GluonTS.
"""
