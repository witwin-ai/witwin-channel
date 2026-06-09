from __future__ import annotations

import torch as _torch  # noqa: F401

try:
    from . import _raydtorch as _C
except ImportError as exc:
    _C = None
    _EXTENSION_IMPORT_ERROR = exc
else:
    _EXTENSION_IMPORT_ERROR = None

from .camera import Camera
from .mesh import Mesh
from .scene import Scene
from .types import (
    DfrAccum,
    DfrGrid,
    DfrMaterial,
    DfrPaths,
    DfrStates,
    Intersection,
    NearestPointEdge,
    NearestRayEdge,
    Ray,
    RayFlags,
    ReflEpcField,
    ReflectionChain,
    SceneGlobalGeometry,
)

__all__ = [
    "DfrAccum",
    "DfrGrid",
    "DfrMaterial",
    "DfrPaths",
    "DfrStates",
    "Camera",
    "Intersection",
    "Mesh",
    "NearestPointEdge",
    "NearestRayEdge",
    "Ray",
    "RayFlags",
    "ReflEpcField",
    "ReflectionChain",
    "Scene",
    "SceneGlobalGeometry",
]
