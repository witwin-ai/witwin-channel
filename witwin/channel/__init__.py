"""User-facing channel API umbrella."""

from __future__ import annotations

import inspect
from importlib import import_module

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]

from witwin.channel.types import (
    Bool,
    Complex2f,
    Float,
    Int32,
    Matrix4f,
    Point2f,
    Point3f,
    UInt32,
    Vector2f,
    Vector3f,
    Vector3u,
)
from witwin.core import Box, Material, Structure
from witwin.channel.core import Grid, GridSpec, RadioMapResult
from witwin.channel.core.scene import (
    AntennaArray,
    EdgePolicy,
    PlanarArray,
    Receiver,
    ReceiverGrid,
    Scene,
    Transmitter,
    ULA,
    UPA,
)
from witwin.channel.core.scene.material_presets import install_material_from_itu

install_material_from_itu(Material)

_LAZY_NAMESPACES = {
    "path": "witwin.channel.path",
    "deterministic": "witwin.channel.deterministic",
    "montecarlo": "witwin.channel.montecarlo",
}


def __getattr__(name: str):
    if name in _LAZY_NAMESPACES:
        module = import_module(_LAZY_NAMESPACES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Bool",
    "AntennaArray",
    "Box",
    "Complex2f",
    "EdgePolicy",
    "Float",
    "Grid",
    "GridSpec",
    "Int32",
    "Material",
    "Matrix4f",
    "PlanarArray",
    "Point2f",
    "Point3f",
    "RadioMapResult",
    "Receiver",
    "ReceiverGrid",
    "Scene",
    "Structure",
    "Transmitter",
    "ULA",
    "UPA",
    "UInt32",
    "Vector2f",
    "Vector3f",
    "Vector3u",
    "deterministic",
    "montecarlo",
    "path",
]
