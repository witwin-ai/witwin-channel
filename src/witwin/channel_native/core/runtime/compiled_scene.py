from __future__ import annotations

from dataclasses import dataclass

from .assignments import AssignmentStore
from .geometry import GeometryStore
from .material_store import MaterialStore
from .raydn import RayDNScene


@dataclass(slots=True)
class CompiledScene:
    geometry: GeometryStore
    materials: MaterialStore
    assignments: AssignmentStore
    raydn: RayDNScene
    workspace: object | None
    geometry_version: int
    material_version: int
    assignment_version: int
