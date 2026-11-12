# Copyright Xingyu Chen.
# One table of the solver entry points, so tests can parametrize over every solver by label.

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class SolverEntry:
    """A solver as tests see it: its config and result types, ``solve`` and owner module."""

    entry: str
    owner: ModuleType
    Config: type
    Metadata: Any
    solve: Any


_TABLE = {
    # label: (entry module, owner module, config, metadata type)
    "deterministic": (
        "witwin.channel.radiomap", "witwin.channel.radiomap._coherent", "Coherent",
        "CoherentRadioMapMetadata",
    ),
    "path": ("witwin.channel.path", "witwin.channel.path", "Config", "PathMetadata"),
    "montecarlo.basic": (
        "witwin.channel.radiomap", "witwin.channel.radiomap._stochastic", "Stochastic",
        "RadioMapMetadata",
    ),
    "montecarlo.bdpt": (
        "witwin.channel.experimental.bdpt", "witwin.channel.experimental.bdpt", "Config",
        "Metadata",
    ),
}
_ALIASES = {"det": "deterministic", "basic": "montecarlo.basic", "mc": "montecarlo.basic",
            "bdpt": "montecarlo.bdpt"}


def solver(label: str) -> SolverEntry:
    """The solver a test label names (``deterministic`` / ``det`` is ``radiomap.Coherent``)."""

    entry, owner, config, metadata = _TABLE[_ALIASES.get(label, label)]
    module = import_module(entry)
    return SolverEntry(
        entry=entry, owner=import_module(owner), Config=getattr(module, config),
        Metadata=getattr(module, metadata), solve=module.solve,
    )
