"""Compatibility import for the canonical compiled scene owner."""

from .assignments import AssignmentStore  # noqa: F401 - annotation compatibility
from .geometry import GeometryStore  # noqa: F401 - annotation compatibility
from .material_store import MaterialStore  # noqa: F401 - annotation compatibility
from .raydn import RayDNScene  # noqa: F401 - annotation compatibility
from witwin.channel_native.scene.compiled import CompiledScene

__all__ = ["CompiledScene"]
