"""Compatibility imports for scene object models.

The canonical implementation lives in :mod:`witwin.channel_native.scene.models`.
This module intentionally has no ``__all__`` so legacy wildcard imports keep
their historical behavior.
"""

from witwin.channel_native.scene.models import (  # noqa: F401
    AntennaArray,
    AntennaPattern,
    Material,
    Protocol,
    ReceiverGrid,
    ReceiverPoint,
    Structure,
    Transmitter,
    _as_array,
    _as_orientation,
    _as_pattern,
    _as_polarization,
    _as_vector3,
    _as_weights,
    dataclass,
    orientation_matrix,
    planar_uv,
    replace,
    torch,
)
