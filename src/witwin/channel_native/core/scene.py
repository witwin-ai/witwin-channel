from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

from .objects import ReceiverGrid, ReceiverPoint, Structure, Transmitter
from .runtime.assignments import AssignmentStore
from .runtime.compiled_scene import CompiledScene
from .runtime.geometry import GeometryStore
from .runtime.material_store import MaterialStore
from .runtime.raydn import RayDNScene, build_scene_from_structures
from .edge_policy import DEFAULT_EDGE_POLICY, EdgePolicy
from .kernels.ops import bdpt_zero_matrix, core_diffraction_edge_count, core_pack_int2


Receiver = ReceiverPoint | ReceiverGrid

_RAYD_EDGE_INFO_PLANE_TOL = 1.34e-5


@dataclass(frozen=True, slots=True)
class Scene:
    structures: tuple[Structure, ...]
    transmitters: tuple[Transmitter, ...]
    receivers: tuple[Receiver, ...]
    frequency: float
    metadata: dict[str, object]
    _geometry_version: int = 0
    _material_version: int = 0
    _assignment_version: int = 0
    _compiled_cache: CompiledScene | None = field(default=None, init=False, compare=False, repr=False)
    _raydn_cache: RayDNScene | None = field(default=None, init=False, compare=False, repr=False)

    def __init__(
        self,
        *,
        structures: list[Structure] | tuple[Structure, ...],
        transmitters: list[Transmitter] | tuple[Transmitter, ...],
        receivers: list[Receiver] | tuple[Receiver, ...],
        frequency: float,
        metadata: dict[str, object] | None = None,
        _geometry_version: int = 0,
        _material_version: int = 0,
        _assignment_version: int = 0,
    ) -> None:
        if frequency <= 0.0:
            raise ValueError("frequency must be positive")
        object.__setattr__(self, "structures", tuple(structures))
        object.__setattr__(self, "transmitters", tuple(transmitters))
        object.__setattr__(self, "receivers", tuple(receivers))
        object.__setattr__(self, "frequency", float(frequency))
        object.__setattr__(self, "metadata", dict(metadata or {}))
        object.__setattr__(self, "_geometry_version", _geometry_version)
        object.__setattr__(self, "_material_version", _material_version)
        object.__setattr__(self, "_assignment_version", _assignment_version)
        object.__setattr__(self, "_compiled_cache", None)
        object.__setattr__(self, "_raydn_cache", None)

    @classmethod
    def load_mitsuba(cls, filename: str, **kwargs) -> Scene:
        from .scene_loader import load_mitsuba

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
        return replace(self, structures=tuple(structures), _geometry_version=self._geometry_version + 1)

    def with_structure_material(self, index: int, material: object) -> Scene:
        structures = list(self.structures)
        structures[index] = structures[index].with_material(material)  # type: ignore[arg-type]
        return replace(self, structures=tuple(structures), _material_version=self._material_version + 1)

    def with_frequency(self, frequency: float) -> Scene:
        return replace(self, frequency=frequency, _material_version=self._material_version + 1)

    def diffraction_edge_count(self, edge_policy: EdgePolicy | None = None) -> int:
        policy = DEFAULT_EDGE_POLICY if edge_policy is None else edge_policy
        if not policy.edge_diffraction:
            return 0
        if len(self.structures) == 0:
            return 0
        raydn_scene = self.raydn_scene()
        if not raydn_scene.available:
            raise RuntimeError("diffraction edge counting requires RayDN native scene capability")
        return _diffraction_edge_count_from_raydn_scene(raydn_scene, policy)

    @property
    def n_diffraction_edges(self) -> int:
        return self.diffraction_edge_count()

    def raydn_scene(self) -> RayDNScene:
        cached = self._raydn_cache
        if cached is not None:
            return cached
        raydn = build_scene_from_structures(self.structures)
        object.__setattr__(self, "_raydn_cache", raydn)
        return raydn

    def compile(self) -> CompiledScene:
        cached = self._compiled_cache
        if (
            cached is not None
            and cached.geometry_version == self._geometry_version
            and cached.material_version == self._material_version
            and cached.assignment_version == self._assignment_version
        ):
            return cached
        raydn = self.raydn_scene()
        geometry = _compile_geometry(self.structures, self._geometry_version, raydn=raydn)
        materials = _compile_materials(self.structures, self.frequency, self._material_version)
        assignments = _compile_assignments(
            self.structures,
            num_faces=geometry.faces.shape[0],
            num_edges=geometry.edges.shape[0],
            version=self._assignment_version,
        )
        compiled = CompiledScene(
            geometry=geometry,
            materials=materials,
            assignments=assignments,
            raydn=raydn,
            workspace=None,
            geometry_version=geometry.version,
            material_version=materials.version,
            assignment_version=assignments.version,
        )
        object.__setattr__(self, "_compiled_cache", compiled)
        return compiled


def _diffraction_edge_count_from_raydn_scene(raydn_scene: RayDNScene, edge_policy: EdgePolicy) -> int:
    records = raydn_scene.edge_records()
    return core_diffraction_edge_count(
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


def _compile_geometry(structures: tuple[Structure, ...], version: int, *, raydn: RayDNScene) -> GeometryStore:
    if not structures:
        empty_vertices = torch.empty((0, 3), dtype=torch.float32)
        empty_faces = torch.empty((0, 3), dtype=torch.int32)
        empty_edges = torch.empty((0, 2), dtype=torch.int32)
        return GeometryStore(
            vertices=empty_vertices,
            faces=empty_faces,
            face_normals=empty_vertices,
            edges=empty_edges,
            edge_adj_faces=empty_edges,
            edge_param_range=torch.empty((0, 2), dtype=torch.float32),
            face_structure_id=torch.empty((0,), dtype=torch.int32),
            face_surface_id=torch.empty((0,), dtype=torch.int32),
            version=version,
        )

    if not raydn.available:
        raydn.require_handle()
    records = raydn.edge_records()
    face_structure_id = []
    face_surface_id = []
    for structure_id, structure in enumerate(structures):
        face_structure_id.extend([structure_id] * structure.faces.shape[0])
        face_surface_id.extend([structure.surface_id] * structure.faces.shape[0])

    return GeometryStore(
        vertices=records.vertices,
        faces=records.faces,
        face_normals=records.face_normals,
        edges=core_pack_int2(records.edge_v0, records.edge_v1),
        edge_adj_faces=core_pack_int2(records.face0, records.face1),
        edge_param_range=bdpt_zero_matrix(records.vertices, rows=records.edge_v0.shape[0], cols=2),
        face_structure_id=torch.tensor(face_structure_id, dtype=torch.int32),
        face_surface_id=torch.tensor(face_surface_id, dtype=torch.int32),
        version=version,
    )


def _compile_materials(
    structures: tuple[Structure, ...], frequency_hz: float, version: int
) -> MaterialStore:
    params = [structure.material.parameters() for structure in structures]
    if not params:
        params = [{"eps_r": 1.0, "mu_r": 1.0, "sigma_e": 0.0, "gain": 1.0, "model_id": 1}]

    return MaterialStore(
        eps_r=torch.tensor([float(p["eps_r"]) for p in params], dtype=torch.float32),
        mu_r=torch.tensor([float(p["mu_r"]) for p in params], dtype=torch.float32),
        sigma_e=torch.tensor([float(p["sigma_e"]) for p in params], dtype=torch.float32),
        gain=torch.tensor([float(p["gain"]) for p in params], dtype=torch.float32),
        model_id=torch.tensor([int(p["model_id"]) for p in params], dtype=torch.int32),
        model_params=torch.tensor([[0.0, 0.0, 0.0, 0.0] for _ in params], dtype=torch.float32),
        frequency_hz=frequency_hz,
        version=version,
    )


def _compile_assignments(
    structures: tuple[Structure, ...], *, num_faces: int, num_edges: int, version: int
) -> AssignmentStore:
    face_material_ids = []
    for material_id, structure in enumerate(structures):
        face_material_ids.extend([material_id] * structure.faces.shape[0])
    return AssignmentStore(
        face_material_id=torch.tensor(face_material_ids, dtype=torch.int32),
        edge_material_id0=torch.tensor([0] * num_edges, dtype=torch.int32),
        edge_material_id1=torch.tensor([0] * num_edges, dtype=torch.int32),
        surface_material_id=torch.tensor(list(range(len(structures))), dtype=torch.int32),
        structure_material_id=torch.tensor(list(range(len(structures))), dtype=torch.int32),
        num_faces=num_faces,
        num_edges=num_edges,
        version=version,
    )


