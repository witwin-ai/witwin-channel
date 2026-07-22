"""Compatibility import for the canonical compiled scene owner."""

from .geometry import GeometryStore  # noqa: F401 - annotation compatibility
from witwin.channel.scene.kernels.rayd_scene import (  # noqa: F401
    RayDSceneResource,
)
from witwin.channel.scene.stores.assignments import (  # noqa: F401
    AssignmentStore,
)
from witwin.channel.scene.stores.materials import (  # noqa: F401
    MaterialStore,
)
from witwin.channel.scene.scattering_resources import (  # noqa: F401
    KirchhoffRuntimeResources,
    PhaseScreenRuntimeResources,
)
from witwin.channel.scene.compiled import CompiledScene

__all__ = ["CompiledScene"]
