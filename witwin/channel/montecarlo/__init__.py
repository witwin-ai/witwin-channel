# Copyright Xingyu Chen.
# Removed import path; importing it raises ImportError naming its replacement.

"""Removed import path; importing it raises ImportError naming its replacement."""

raise ImportError(
    "witwin.channel.montecarlo was removed in Channel 0.6.0: montecarlo.basic is now "
    "witwin.channel.radiomap (Config -> radiomap.Stochastic, Result -> radiomap.RadioMap, "
    "solve -> radiomap.solve), and montecarlo.bdpt is now witwin.channel.experimental.bdpt. "
    "See docs/migration-0.6.md."
)
