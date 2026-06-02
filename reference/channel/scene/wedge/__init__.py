"""Backend-neutral wedge building and packing utilities."""

from .runtime import RUNTIME_REGISTRY, WedgeRuntime, create_wedge_backend, get_scene_wedge_runtime
from .types import (
    HeightPlaneAnchorSpec,
    TriangleWedgeMap,
    WedgeAnchorView,
    WedgeGeometry,
    WedgeGeometryConfig,
    WedgePack,
    WedgeSelection,
    WedgeSelectionConfig,
)

__all__ = [
    "HeightPlaneAnchorSpec",
    "RUNTIME_REGISTRY",
    "TriangleWedgeMap",
    "WedgeAnchorView",
    "WedgeGeometry",
    "WedgeGeometryConfig",
    "WedgePack",
    "WedgeRuntime",
    "WedgeSelection",
    "WedgeSelectionConfig",
    "create_wedge_backend",
    "get_scene_wedge_runtime",
]
