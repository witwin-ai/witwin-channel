# Copyright Xingyu Chen.
# Removed import path; importing it raises ImportError naming its replacement.

"""Removed import path; importing it raises ImportError naming its replacement."""

raise ImportError(
    "witwin.channel.deterministic was removed in Channel 0.6.0; use witwin.channel.radiomap: "
    "Config -> radiomap.Coherent (coherent=True/False -> combine='field'/'power'), "
    "Result -> radiomap.CoherentRadioMap, PathTable -> radiomap.PathTable, and "
    "solve -> radiomap.solve. See docs/migration-0.6.md."
)
