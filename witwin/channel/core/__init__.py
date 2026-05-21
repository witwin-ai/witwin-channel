"""Shared, solver-independent utilities for channel simulation packages.

This package collects grouped numerics, geometry, physics, runtime, result,
scene, and kernel helpers shared across ``witwin.channel.deterministic``,
``witwin.channel.montecarlo``, and ``witwin.channel.path``. Modules here must
not depend on any sibling solver package.

Submodules
----------
* :mod:`.numerics` - constants, DrJit array helpers, and tensor adapters
* :mod:`.geometry` - primitives, mesh buffers, diffraction geometry, raygen
* :mod:`.grid` - shared axis-aligned receiver grid (``GridSpec``, ``Grid``)
* :mod:`.physics` - material, wave, polarization, and boundary-policy helpers
* :mod:`.runtime` - solve-time context and scene-aware runtime helpers
* :mod:`.results` - shared public result containers and result-level controls
* :mod:`.kernels` - shared native-kernel Python wrappers (e.g. shadow
  boundary) so neither solver imports the other's kernel namespace
"""

from .results import (
    RadioMapCoordinates,
    RadioMapFieldPayload,
    RadioMapPowerPayload,
    RadioMapResult,
    coordinates_from_grid,
)
from .grid import Grid, GridSpec

__all__ = [
    "Grid",
    "GridSpec",
    "RadioMapCoordinates",
    "RadioMapFieldPayload",
    "RadioMapPowerPayload",
    "RadioMapResult",
    "coordinates_from_grid",
]
