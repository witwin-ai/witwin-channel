# Copyright Xingyu Chen.
# Native kernel facades for every Channel domain.

"""Native kernel facades for every Channel domain.

This package is the one place a reader finds every native facade. Each module
here validates a typed contract, requests the required native symbol through
:mod:`witwin.channel.runtime`, dispatches the native operation, and converts
its result into a named typed contract. The facades own no physics: a Torch
expression that reconstructs a numerical operation is forbidden by the compute
policy in ``CLAUDE.md``.

One module per domain, and each is the single owner of its facades:

``deterministic``
    Deterministic accumulation and its AD companions.
``fields``
    RF field evaluation, transport, source amplitude, and AD companions.
``geometry``
    RayD geometry bridges, segment penetration, and their AD companions.
``materials``
    Material contracts and EM layer-stack evaluation.
``montecarlo``
    Monte Carlo Basic and BDPT maps, paths, sampling, and transmission.
``scattering``
    Scattering tables, ensembles, chains, and their AD companions.
``topology``
    Discrete path blocks, candidates, compaction, and construction.

This package sits above the RF domains, so a facade may import
:mod:`witwin.channel.runtime` and the shared row contracts, but never a solver
and never a domain that imports it back. Importing a facade must not be a way
to reach a domain package.

The package root deliberately re-exports nothing. Every consumer spells the
owning module (``from witwin.channel.kernels import topology`` or
``from witwin.channel.kernels.topology import path_merge_blocks``), so a
facade name has exactly one import path and no second spelling.
"""

from __future__ import annotations