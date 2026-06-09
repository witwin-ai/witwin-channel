from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations

import torch

from .objects import ReceiverGrid, ReceiverPoint, Structure, Transmitter
from .runtime.assignments import AssignmentStore
from .runtime.compiled_scene import CompiledScene
from .runtime.geometry import GeometryStore
from .runtime.material_store import MaterialStore
from .runtime.raydn import RayDNEdgeRecords, build_scene_from_structures
from .edge_policy import DEFAULT_EDGE_POLICY, EdgePolicy


Receiver = ReceiverPoint | ReceiverGrid

_RAYD_EDGE_EPSILON = 1.0e-6
_RAYD_NORMAL_COS_TOL = 1.0 - 1.0e-5
_RAYD_EDGE_INFO_PLANE_TOL = 1.313e-5


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
        raise TypeError(f"unsupported scene object: {type(obj).__name__}")

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
        raydn_scene = build_scene_from_structures(self.structures)
        if raydn_scene.available:
            return _diffraction_edge_count_from_raydn_records(raydn_scene.edge_records(), policy)
        geometry = _compile_geometry(self.structures, self._geometry_version)
        return _diffraction_edge_count_from_geometry(geometry, policy)

    @property
    def n_diffraction_edges(self) -> int:
        return self.diffraction_edge_count()

    def compile(self) -> CompiledScene:
        cached = self._compiled_cache
        if (
            cached is not None
            and cached.geometry_version == self._geometry_version
            and cached.material_version == self._material_version
            and cached.assignment_version == self._assignment_version
        ):
            return cached
        geometry = _compile_geometry(self.structures, self._geometry_version)
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
            raydn=build_scene_from_structures(self.structures),
            workspace=None,
            geometry_version=geometry.version,
            material_version=materials.version,
            assignment_version=assignments.version,
        )
        object.__setattr__(self, "_compiled_cache", compiled)
        return compiled


def _opposite_vertex(face: torch.Tensor, shared0: torch.Tensor, shared1: torch.Tensor) -> torch.Tensor:
    face = face.to(dtype=torch.long)
    x_other = (face[:, 0] != shared0) & (face[:, 0] != shared1)
    y_other = (face[:, 1] != shared0) & (face[:, 1] != shared1)
    return torch.where(x_other, face[:, 0], torch.where(y_other, face[:, 1], face[:, 2]))


def _selected_diffraction_edges(
    *,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_normals: torch.Tensor,
    edge_v0: torch.Tensor,
    edge_v1: torch.Tensor,
    face0: torch.Tensor,
    face1: torch.Tensor,
    edge_policy: EdgePolicy,
    plane_tol: float,
) -> torch.Tensor:
    if edge_v0.numel() == 0:
        return torch.empty((0,), dtype=torch.bool, device=vertices.device)

    edge_v0 = edge_v0.to(dtype=torch.long)
    edge_v1 = edge_v1.to(dtype=torch.long)
    face0 = face0.to(dtype=torch.long)
    face1 = face1.to(dtype=torch.long)
    vectors = vertices[edge_v1] - vertices[edge_v0]
    lengths = torch.linalg.vector_norm(vectors, dim=1).clamp_min(1.0e-12)
    valid0 = face0 >= 0
    valid1 = face1 >= 0
    boundary = valid0 & ~valid1
    interior = valid0 & valid1

    safe0 = face0.clamp_min(0)
    safe1 = face1.clamp_min(0)
    n0 = torch.nn.functional.normalize(face_normals[safe0], dim=1, eps=_RAYD_EDGE_EPSILON)
    n1 = torch.nn.functional.normalize(face_normals[safe1], dim=1, eps=_RAYD_EDGE_EPSILON)

    face_a = faces[safe0]
    face_b = faces[safe1]
    plane_point = vertices[edge_v0]
    point_a = vertices[_opposite_vertex(face_a, edge_v0, edge_v1)]
    point_b = vertices[_opposite_vertex(face_b, edge_v0, edge_v1)]
    normal_dot = (n0 * n1).sum(dim=1)
    aligned = normal_dot.abs() >= _RAYD_NORMAL_COS_TOL
    plane_dist_a = ((point_a - plane_point) * n0).sum(dim=1).abs()
    plane_dist_b = ((point_b - plane_point) * n0).sum(dim=1).abs()
    coplanar = (
        interior
        & aligned
        & (plane_dist_a <= float(plane_tol))
        & (plane_dist_b <= float(plane_tol))
    )

    interior_angle = torch.acos(torch.clamp(-normal_dot, -1.0, 1.0))
    exterior_angle = torch.where(interior, 2.0 * torch.pi - interior_angle, torch.zeros_like(interior_angle))
    if edge_policy.boundary_edge_policy == "half_plane":
        exterior_angle = torch.where(
            boundary,
            torch.full_like(exterior_angle, 2.0 * torch.pi),
            exterior_angle,
        )
    wedge_n = exterior_angle / torch.pi
    selected = (
        (interior | boundary)
        & ~coplanar
        & (lengths > _RAYD_EDGE_EPSILON)
        & (wedge_n > 1.0 + _RAYD_EDGE_EPSILON)
    )
    if edge_policy.vertical_only:
        vertical_ratio = vectors[:, 2].abs() / lengths
        selected = selected & (vertical_ratio > float(edge_policy.vertical_ratio))
    return selected


def _diffraction_edge_count_from_raydn_records(
    records: RayDNEdgeRecords,
    edge_policy: EdgePolicy,
) -> int:
    selected = _selected_diffraction_edges(
        vertices=records.vertices,
        faces=records.faces,
        face_normals=records.face_normals,
        edge_v0=records.edge_v0,
        edge_v1=records.edge_v1,
        face0=records.face0,
        face1=records.face1,
        edge_policy=edge_policy,
        plane_tol=_RAYD_EDGE_INFO_PLANE_TOL,
    )
    return int(selected.sum().item())


def _diffraction_edge_count_from_geometry(geometry: GeometryStore, edge_policy: EdgePolicy) -> int:
    selected = _selected_diffraction_edges(
        vertices=geometry.vertices,
        faces=geometry.faces,
        face_normals=geometry.face_normals,
        edge_v0=geometry.edges[:, 0],
        edge_v1=geometry.edges[:, 1],
        face0=geometry.edge_adj_faces[:, 0],
        face1=geometry.edge_adj_faces[:, 1],
        edge_policy=edge_policy,
        plane_tol=1.0e-5,
    )
    return int(selected.sum().item())


def _compile_geometry(structures: tuple[Structure, ...], version: int) -> GeometryStore:
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

    vertices = []
    faces = []
    face_structure_id = []
    face_surface_id = []
    offset = 0
    for structure_id, structure in enumerate(structures):
        vertices.append(structure.vertices)
        faces.append(structure.faces + offset)
        face_structure_id.extend([structure_id] * structure.faces.shape[0])
        face_surface_id.extend([structure.surface_id] * structure.faces.shape[0])
        offset += structure.vertices.shape[0]

    all_vertices = torch.cat(vertices, dim=0).contiguous()
    all_faces = torch.cat(faces, dim=0).contiguous()
    face_normals = _face_normals(all_vertices, all_faces)
    edges, edge_adj_faces = _mesh_edges(all_faces)

    return GeometryStore(
        vertices=all_vertices,
        faces=all_faces,
        face_normals=face_normals,
        edges=edges,
        edge_adj_faces=edge_adj_faces,
        edge_param_range=torch.zeros((edges.shape[0], 2), dtype=torch.float32),
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
        model_params=torch.zeros((len(params), 4), dtype=torch.float32),
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
        edge_material_id0=torch.zeros((num_edges,), dtype=torch.int32),
        edge_material_id1=torch.zeros((num_edges,), dtype=torch.int32),
        surface_material_id=torch.arange(len(structures), dtype=torch.int32),
        structure_material_id=torch.arange(len(structures), dtype=torch.int32),
        num_faces=num_faces,
        num_edges=num_edges,
        version=version,
    )


def _face_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    if faces.numel() == 0:
        return torch.empty((0, 3), dtype=torch.float32)
    tri = vertices[faces.to(dtype=torch.long)]
    normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    return torch.nn.functional.normalize(normals, dim=1).to(dtype=torch.float32).contiguous()


def _mesh_edges(faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, face in enumerate(faces.tolist()):
        for a, b in combinations(face, 2):
            key = tuple(sorted((int(a), int(b))))
            edge_to_faces.setdefault(key, []).append(face_id)

    edges = []
    adj_faces = []
    for edge, face_ids in edge_to_faces.items():
        edges.append(edge)
        padded = (face_ids + [-1, -1])[:2]
        adj_faces.append(padded)

    if not edges:
        return torch.empty((0, 2), dtype=torch.int32), torch.empty((0, 2), dtype=torch.int32)
    return torch.tensor(edges, dtype=torch.int32), torch.tensor(adj_faces, dtype=torch.int32)
