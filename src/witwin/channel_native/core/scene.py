from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

import torch

from .objects import ReceiverGrid, ReceiverPoint, Structure, Transmitter
from .runtime.assignments import AssignmentStore
from .runtime.compiled_scene import CompiledScene
from .runtime.geometry import GeometryStore
from .runtime.material_store import MaterialStore
from .runtime.raydn import RayDNScene


Receiver = ReceiverPoint | ReceiverGrid


@dataclass(frozen=True, slots=True)
class Scene:
    structures: tuple[Structure, ...]
    transmitters: tuple[Transmitter, ...]
    receivers: tuple[Receiver, ...]
    frequency: float
    _geometry_version: int = 0
    _material_version: int = 0
    _assignment_version: int = 0

    def __init__(
        self,
        *,
        structures: list[Structure] | tuple[Structure, ...],
        transmitters: list[Transmitter] | tuple[Transmitter, ...],
        receivers: list[Receiver] | tuple[Receiver, ...],
        frequency: float,
        _geometry_version: int = 0,
        _material_version: int = 0,
        _assignment_version: int = 0,
    ) -> None:
        if not receivers:
            raise ValueError("Scene requires at least one receiver")
        if frequency <= 0.0:
            raise ValueError("frequency must be positive")
        object.__setattr__(self, "structures", tuple(structures))
        object.__setattr__(self, "transmitters", tuple(transmitters))
        object.__setattr__(self, "receivers", tuple(receivers))
        object.__setattr__(self, "frequency", float(frequency))
        object.__setattr__(self, "_geometry_version", _geometry_version)
        object.__setattr__(self, "_material_version", _material_version)
        object.__setattr__(self, "_assignment_version", _assignment_version)

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

    def compile(self) -> CompiledScene:
        geometry = _compile_geometry(self.structures, self._geometry_version)
        materials = _compile_materials(self.structures, self.frequency, self._material_version)
        assignments = _compile_assignments(
            self.structures,
            num_faces=geometry.faces.shape[0],
            num_edges=geometry.edges.shape[0],
            version=self._assignment_version,
        )
        return CompiledScene(
            geometry=geometry,
            materials=materials,
            assignments=assignments,
            raydn=RayDNScene(),
            workspace=None,
            geometry_version=geometry.version,
            material_version=materials.version,
            assignment_version=assignments.version,
        )


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
