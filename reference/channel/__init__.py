"""Witwin Channel - Wireless channel simulation (RadioMap & Path/CIR)."""

from __future__ import annotations

from importlib import import_module

try:
    import witwin.core
except ImportError:
    raise ImportError(
        "witwin-core is required. Install with: pip install witwin-core"
    )

from witwin.core import (
    Box,
    Cone,
    Cylinder,
    Ellipsoid,
    GeometryBase,
    HollowBox,
    Material,
    Mesh,
    Prism,
    Pyramid,
    Sphere,
    Structure,
    Torus,
)

from .utils.constants import DEFAULT_VARIANT, POWER_DB_FLOOR


_LAZY_EXPORTS = {
    "ChannelConfig": "witwin.channel.config",
    "TraceConfig": "witwin.channel.config",
    "DiffractionExecutionConfig": "witwin.channel.config",
    "Tracer": "witwin.channel.trace",
    "Scene": "witwin.channel.scene",
    "DrJitMesh": "witwin.channel.scene",
    "Field": "witwin.channel.monitors.field",
    "FieldMonitor": "witwin.channel.monitors",
    "PathMonitor": "witwin.channel.monitors",
    "RadioMapMonitor": "witwin.channel.monitors",
    "InteractionType": "witwin.channel.types",
    "MonitorResult": "witwin.channel.monitors.field.result",
    "RadioMapResult": "witwin.channel.monitors.radio_map",
    "PathResult": "witwin.channel.monitors.path.result",
    "draw_edges": "witwin.channel.scene",
    "draw_edges_with_normals": "witwin.channel.scene",
    "draw_corners": "witwin.channel.scene",
    "draw_tx": "witwin.channel.scene",
    "draw_scene": "witwin.channel.scene",
    "plot_field_with_edges": "witwin.channel.scene",
    "plot_gradient_with_edges": "witwin.channel.scene",
    "generate_circle_directions": "witwin.channel.utils.raygen",
    "generate_sphere_directions": "witwin.channel.utils.raygen",
    "compute_los_field": "witwin.channel.trace",
    "compute_reflection_field": "witwin.channel.trace",
    "compute_diffraction_field": "witwin.channel.trace.diffraction",
    "compute_diffraction_order_breakdown": "witwin.channel.trace.diffraction",
    "Edge2D": "witwin.channel.scene",
    "Corner2D": "witwin.channel.scene",
    "VerticalEdge": "witwin.channel.scene",
    "DiffractionPoint": "witwin.channel.scene",
    "edge_xy": "witwin.channel.utils",
    "corner_xy": "witwin.channel.utils",
    "scalar": "witwin.channel.utils",
    "to_power_db": "witwin.channel.utils",
    "to_numpy": "witwin.channel.utils",
    "to_numpy_2d": "witwin.channel.utils",
    "runtime": "witwin.channel.runtime",
    "cuda_runtime_version": "witwin.channel._native",
    "run_cuda_noop": "witwin.channel._native",
    "sample_add_one": "witwin.channel._native",
    "native_extension_available": "witwin.channel._native",
    "SionnaAdaptor": "witwin.channel.scene",
}

_MODULE_EXPORTS = {"runtime"}

__version__ = "0.1.0"

__all__ = [
    "Tracer",
    "ChannelConfig",
    "TraceConfig",
    "DiffractionExecutionConfig",
    "Scene",
    "DrJitMesh",
    "Field",
    "FieldMonitor",
    "PathMonitor",
    "RadioMapMonitor",
    "InteractionType",
    "MonitorResult",
    "RadioMapResult",
    "PathResult",
    "Material",
    "Structure",
    "GeometryBase",
    "Mesh",
    "Box",
    "Sphere",
    "Cylinder",
    "Cone",
    "Ellipsoid",
    "Pyramid",
    "Prism",
    "Torus",
    "HollowBox",
    "draw_edges",
    "draw_edges_with_normals",
    "draw_corners",
    "draw_tx",
    "draw_scene",
    "plot_field_with_edges",
    "plot_gradient_with_edges",
    "generate_circle_directions",
    "generate_sphere_directions",
    "compute_los_field",
    "compute_reflection_field",
    "compute_diffraction_field",
    "compute_diffraction_order_breakdown",
    "Edge2D",
    "Corner2D",
    "VerticalEdge",
    "DiffractionPoint",
    "edge_xy",
    "corner_xy",
    "scalar",
    "to_power_db",
    "to_numpy",
    "to_numpy_2d",
    "DEFAULT_VARIANT",
    "POWER_DB_FLOOR",
    "runtime",
    "cuda_runtime_version",
    "run_cuda_noop",
    "sample_add_one",
    "native_extension_available",
    "SionnaAdaptor",
]


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = module if name in _MODULE_EXPORTS else getattr(module, name)
    globals()[name] = value
    return value
