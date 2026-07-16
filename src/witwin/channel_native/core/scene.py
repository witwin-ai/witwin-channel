from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

from witwin.channel_native.scene.models import ReceiverGrid, ReceiverPoint, Structure, Transmitter
from witwin.channel_native.scene.compiled import CompiledScene
from witwin.channel_native.scene.compile import (
    _abi_v3_layer_view,  # noqa: F401 - compatibility re-export
    _compile_assignments,  # noqa: F401 - compatibility re-export
    _compile_geometry,  # noqa: F401 - compatibility re-export
    _compile_materials,  # noqa: F401 - compatibility re-export
    _frequency_dependent_material_keys,  # noqa: F401 - compatibility re-export
    _material_records,  # noqa: F401 - compatibility re-export
    _phase_screen_descriptor,  # noqa: F401 - compatibility re-export
    compile_scene,
)
from witwin.channel_native.scene.stores.assignments import AssignmentStore  # noqa: F401
from witwin.channel_native.scene.stores.geometry import GeometryStore  # noqa: F401
from witwin.channel_native.scene.stores.materials import MaterialStore  # noqa: F401
from witwin.channel_native.scene.kernels.rayd_scene import (
    RayDNScene,
    build_scene_from_structures,
)
from .edge_policy import DEFAULT_EDGE_POLICY, EdgePolicy
from .edge_selection import resolve_scene_edge_policy
from witwin.channel_native.propagation.geometry.kernels import (
    primitives as geometry_primitives,
)


Receiver = ReceiverPoint | ReceiverGrid

_RAYD_EDGE_INFO_PLANE_TOL = 1.34e-5


@dataclass(frozen=True, slots=True)
class Scene:
    structures: tuple[Structure, ...]
    transmitters: tuple[Transmitter, ...]
    receivers: tuple[Receiver, ...]
    frequency: float | torch.Tensor
    metadata: dict[str, object]
    _geometry_version: int = 0
    _material_version: int = 0
    _assignment_version: int = 0
    _compiled_cache: CompiledScene | None = field(
        default=None, init=False, compare=False, repr=False
    )
    _raydn_cache: RayDNScene | None = field(
        default=None, init=False, compare=False, repr=False
    )

    def __init__(
        self,
        *,
        structures: list[Structure] | tuple[Structure, ...],
        transmitters: list[Transmitter] | tuple[Transmitter, ...],
        receivers: list[Receiver] | tuple[Receiver, ...],
        frequency: float | torch.Tensor,
        metadata: dict[str, object] | None = None,
        _geometry_version: int = 0,
        _material_version: int = 0,
        _assignment_version: int = 0,
    ) -> None:
        # The carrier frequency may be a 0-d torch tensor so it can carry
        # requires_grad / forward-mode tangents into the differentiable field
        # kernels (plan 07 AD-1). Every non-AD consumer reads it through
        # float(scene.frequency), which detaches by contract.
        if isinstance(frequency, torch.Tensor):
            if frequency.ndim != 0:
                raise ValueError("tensor frequency must be a 0-d tensor")
            if float(frequency.detach()) <= 0.0:
                raise ValueError("frequency must be positive")
        else:
            if frequency <= 0.0:
                raise ValueError("frequency must be positive")
            frequency = float(frequency)
        object.__setattr__(self, "structures", tuple(structures))
        object.__setattr__(self, "transmitters", tuple(transmitters))
        object.__setattr__(self, "receivers", tuple(receivers))
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "metadata", dict(metadata or {}))
        object.__setattr__(self, "_geometry_version", _geometry_version)
        object.__setattr__(self, "_material_version", _material_version)
        object.__setattr__(self, "_assignment_version", _assignment_version)
        object.__setattr__(self, "_compiled_cache", None)
        object.__setattr__(self, "_raydn_cache", None)

    @classmethod
    def load_mitsuba(cls, filename: str, **kwargs) -> Scene:
        from witwin.channel_native.scene.loader import load_mitsuba

        return load_mitsuba(filename, scene_cls=cls, **kwargs)

    def add(self, obj: Transmitter | Receiver) -> Scene:
        if isinstance(obj, Transmitter):
            object.__setattr__(self, "transmitters", self.transmitters + (obj,))
            return self
        if isinstance(obj, (ReceiverPoint, ReceiverGrid)):
            object.__setattr__(self, "receivers", self.receivers + (obj,))
            return self
        raise TypeError(f"scene object type is not accepted: {type(obj).__name__}")

    def with_structure_vertices(self, index: int, vertices: torch.Tensor) -> Scene:
        structures = list(self.structures)
        structures[index] = structures[index].with_vertices(vertices)
        return replace(
            self,
            structures=tuple(structures),
            _geometry_version=self._geometry_version + 1,
        )

    def with_structure_material(self, index: int, material: object) -> Scene:
        structures = list(self.structures)
        structures[index] = structures[index].with_material(material)  # type: ignore[arg-type]
        return replace(
            self,
            structures=tuple(structures),
            _material_version=self._material_version + 1,
        )

    def with_frequency(self, frequency: float) -> Scene:
        return replace(
            self, frequency=frequency, _material_version=self._material_version + 1
        )

    def diffraction_edge_count(self, edge_policy: EdgePolicy | None = None) -> int:
        policy = DEFAULT_EDGE_POLICY if edge_policy is None else edge_policy
        if not policy.edge_diffraction:
            return 0
        if len(self.structures) == 0:
            return 0
        raydn_scene = self.raydn_scene()
        if not raydn_scene.available:
            raise RuntimeError(
                "diffraction edge counting requires RayDN native scene capability"
            )
        return _diffraction_edge_count_from_raydn_scene(raydn_scene, policy)

    @property
    def n_diffraction_edges(self) -> int:
        return self.diffraction_edge_count()

    def raydn_scene(self) -> RayDNScene:
        cached = self._raydn_cache
        if cached is not None:
            return cached
        raydn = build_scene_from_structures(self.structures)
        # Expose the scene's edge policy to the diffraction edge-geometry
        # builders so path generation honors it (audit DF-4).
        raydn.runtime_cache["edge_policy"] = resolve_scene_edge_policy(self)
        object.__setattr__(self, "_raydn_cache", raydn)
        return raydn

    def compile(self) -> CompiledScene:
        return compile_scene(self)


def _diffraction_edge_count_from_raydn_scene(
    raydn_scene: RayDNScene, edge_policy: EdgePolicy
) -> int:
    records = raydn_scene.edge_records()
    return geometry_primitives.core_diffraction_edge_count(
        vertices=records.vertices,
        faces=records.faces,
        face_normals=records.face_normals,
        edge_v0=records.edge_v0,
        edge_v1=records.edge_v1,
        face0=records.face0,
        face1=records.face1,
        vertical_only=edge_policy.vertical_only,
        vertical_ratio=float(edge_policy.vertical_ratio),
        boundary_half_plane=edge_policy.boundary_edge_policy == "half_plane",
        plane_tol=_RAYD_EDGE_INFO_PLANE_TOL,
    )
