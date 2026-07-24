"""GPU-accelerated differentiable wireless propagation simulation."""

from .core.kernels.extension import build_info
from .deployment import pipeline_cache_key, runtime_diagnostics
from .capabilities import capabilities
from witwin.core import (
    AntennaPattern,
    AntennaState,
    Material,
    MaterialLayer,
    PhaseScreen,
    PhysicalMaterial,
    ReceiverGrid,
    Scene,
    SceneSnapshot,
    Structure,
    SurfaceRoughness,
)
from .core.field_state import Complex3State, JonesState

__all__ = [
    "AntennaPattern",
    "AntennaState",
    "Material",
    "MaterialLayer",
    "PhaseScreen",
    "PhysicalMaterial",
    "ReceiverGrid",
    "Scene",
    "SceneSnapshot",
    "Structure",
    "SurfaceRoughness",
    "Complex3State",
    "JonesState",
    "build_info",
    "pipeline_cache_key",
    "runtime_diagnostics",
    "capabilities",
]
