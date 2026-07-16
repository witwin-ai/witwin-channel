"""Compatibility import for the canonical compiled scene owner."""

from .geometry import GeometryStore  # noqa: F401 - annotation compatibility
from witwin.channel_native.scene.kernels.rayd_scene import (  # noqa: F401
    RayDNScene,
)
from witwin.channel_native.scene.stores.assignments import (  # noqa: F401
    AssignmentStore,
)
from witwin.channel_native.scene.stores.materials import (  # noqa: F401
    MaterialStore,
)
from witwin.channel_native.scene.compiled import CompiledScene

__all__ = ["CompiledScene"]
