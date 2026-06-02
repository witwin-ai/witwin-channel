from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import drjit as dr
import rayd
import witwin as wt
import torch

from witwin.core import Mesh

from ..utils.constants import EDGE_2D_EPS, EPS, SMALL_EPS
from ..utils import scalar
from ..utils.drjit_ops import ArrayInit, Concat
from ..utils.geometry import (
    compute_face_normals,
    extract_edges_with_adjacency,
    triangles_are_coplanar,
)
from ..utils.mesh_buffers import to_point3f, to_vector3u
from .mesh import compute_edge_geometry, filter_diffraction_edges


SURFACE_GROUP_NORMAL_COS_TOL = 1.0 - 1e-5
SURFACE_GROUP_PLANE_TOL = 1e-5
MATERIAL_EPS_R_DEFAULT = 1.0
MATERIAL_SIGMA_E_DEFAULT = 0.0


def _torch_runtime_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None:
        return torch.device("cpu")
    return torch.device(device)


def _ray_width_from_ray(ray) -> int:
    try:
        return max(1, int(dr.width(ray.o.x)))
    except AttributeError:
        pass
    try:
        return max(1, int(dr.width(ray.origins.x)))
    except AttributeError:
        return 1


def _invalid_intersection(width: int):
    zeros = dr.zeros(wt.Float, width)
    false_mask = dr.full(wt.Bool, False, width)
    neg_one = dr.full(wt.Int32, -1, width)
    return SimpleNamespace(
        is_valid=lambda: false_mask,
        t=dr.full(wt.Float, float("inf"), width),
        p=wt.Point3f(zeros, zeros, zeros),
        n=wt.Vector3f(zeros, zeros, zeros),
        geo_n=wt.Vector3f(zeros, zeros, zeros),
        prim_id=neg_one,
        shape_id=neg_one,
    )


@dataclass
class _EmptyRayDScene:
    device: torch.device
    _version: int
    _edge_mask: torch.Tensor | None = None

    def __post_init__(self):
        if self._edge_mask is None:
            self._edge_mask = torch.zeros((0,), dtype=torch.bool, device=self.device)

    def version(self) -> int:
        return int(self._version)

    def edge_version(self) -> int:
        return int(self._version)

    def edge_info(self):
        empty_vec3 = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        empty_bool = torch.zeros((0,), dtype=torch.bool, device=self.device)
        empty_i32 = torch.zeros((0,), dtype=torch.int32, device=self.device)
        return SimpleNamespace(
            start=empty_vec3,
            edge=empty_vec3,
            end=empty_vec3,
            length=torch.zeros((0,), dtype=torch.float32, device=self.device),
            normal0=empty_vec3,
            normal1=empty_vec3,
            is_boundary=empty_bool,
            shape_id=empty_i32,
            local_edge_id=empty_i32,
            global_edge_id=empty_i32,
        )

    def edge_topology(self):
        empty_i32 = torch.zeros((0,), dtype=torch.int32, device=self.device)
        return SimpleNamespace(
            v0=empty_i32,
            v1=empty_i32,
            face0_local=empty_i32,
            face1_local=empty_i32,
            face0_global=empty_i32,
            face1_global=empty_i32,
            opposite_vertex0=empty_i32,
            opposite_vertex1=empty_i32,
        )

    def triangle_edge_indices(self, prim_id, global_=True):
        del global_
        prim_id = torch.as_tensor(prim_id, dtype=torch.int32, device=self.device).reshape(-1)
        empty = torch.full_like(prim_id, -1)
        return empty, empty, empty

    def edge_adjacent_faces(self, edge_id, global_=True):
        del global_
        edge_id = torch.as_tensor(edge_id, dtype=torch.int32, device=self.device).reshape(-1)
        empty = torch.full_like(edge_id, -1)
        return empty, empty

    def mesh_face_offsets(self):
        return torch.tensor([0, 0], dtype=torch.int32, device=self.device)

    def mesh_edge_offsets(self):
        return torch.tensor([0, 0], dtype=torch.int32, device=self.device)

    def intersect(self, ray, active=True, flags=None):
        del active, flags
        return _invalid_intersection(_ray_width_from_ray(ray))

    def shadow_test(self, ray, active=True):
        del active
        return dr.full(wt.Bool, False, _ray_width_from_ray(ray))

    def nearest_edge(self, query, active=True):
        del active
        width = _ray_width_from_ray(query)
        zeros = dr.zeros(wt.Float, width)
        false_mask = dr.full(wt.Bool, False, width)
        neg_one = dr.full(wt.Int32, -1, width)
        return SimpleNamespace(
            distance=dr.full(wt.Float, float("inf"), width),
            ray_t=dr.full(wt.Float, float("inf"), width),
            point=wt.Point3f(zeros, zeros, zeros),
            edge_t=zeros,
            edge_point=wt.Point3f(zeros, zeros, zeros),
            shape_id=neg_one,
            edge_id=neg_one,
            global_edge_id=neg_one,
            is_boundary=false_mask,
            is_valid=lambda: false_mask,
        )

    def set_edge_mask(self, mask):
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device).reshape(-1).contiguous()
        if int(mask_tensor.numel()) != 0:
            raise RuntimeError("Scene.set_edge_mask(): mask size must match the scene edge count.")
        self._edge_mask = mask_tensor

    def edge_mask(self):
        return self._edge_mask.clone()

    def sync(self):
        return None


def empty_edge_selection_summary() -> dict[str, int]:
    return {
        "total_vertical_edges": 0,
        "interior_vertical_edges": 0,
        "boundary_vertical_edges": 0,
        "included_vertical_edges": 0,
        "total_selected_edges": 0,
        "interior_selected_edges": 0,
        "boundary_selected_edges": 0,
        "included_selected_edges": 0,
        "included_boundary_edges": 0,
        "excluded_boundary_edges": 0,
        "excluded_endpoint_anchors": 0,
    }


def build_structure_meshes(scene):
    built_meshes = []
    for structure_idx, structure in enumerate(scene.structures):
        if not structure.enabled:
            continue
        geometry = structure.geometry
        if isinstance(geometry, Mesh) or geometry.kind == "drjit_mesh":
            vertices, faces = geometry.to_mesh(device=scene.device)
        else:
            vertices, faces = geometry.to_mesh()
        vertices_dr = to_point3f(vertices)
        faces_dr = to_vector3u(faces)
        n_triangles = int(dr.width(faces_dr))
        material_sample = structure.material.evaluate_static()
        material_specified = _material_sample_is_specified(material_sample)
        built_meshes.append(
            {
                "vertices": vertices_dr,
                "faces": faces_dr,
                "n_triangles": n_triangles,
                "material_eps_r": _material_value_array(
                    material_sample.eps_r,
                    n_triangles,
                    name=f"{structure.name}.material.eps_r",
                ),
                "material_sigma_e": _material_value_array(
                    material_sample.sigma_e,
                    n_triangles,
                    name=f"{structure.name}.material.sigma_e",
                ),
                "material_specified": dr.full(wt.Bool, material_specified, n_triangles),
                "material_structure_idx": dr.full(wt.Int32, structure_idx, n_triangles),
                "material_specified_value": bool(material_specified),
            }
        )
    return built_meshes


def _material_value_array(value, n_values: int, *, name: str):
    if n_values <= 0:
        return dr.zeros(wt.Float, 0)
    if isinstance(value, wt.Float):
        width = int(dr.width(value))
        if width == n_values:
            return value
        if width == 1:
            return dr.repeat(value, n_values)
        raise ValueError(
            f"{name} must be scalar or length {n_values}, got DrJit width {width}."
        )
    if isinstance(value, torch.Tensor):
        flat = value.reshape(-1).contiguous()
        if flat.numel() == n_values:
            return wt.Float(flat)
        if flat.numel() == 1:
            return dr.full(wt.Float, float(flat.detach().cpu()[0]), n_values)
        raise ValueError(f"{name} must be scalar or length {n_values}, got tensor shape {tuple(value.shape)}.")
    try:
        scalar_value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a float-compatible scalar or DrJit Float.") from exc
    return dr.full(wt.Float, scalar_value, n_values)


def _material_sample_is_specified(material_sample) -> bool:
    try:
        eps_r = float(material_sample.eps_r)
        sigma_e = float(material_sample.sigma_e)
    except (TypeError, ValueError):
        return True
    return not (
        abs(eps_r - MATERIAL_EPS_R_DEFAULT) <= 1e-9
        and abs(sigma_e - MATERIAL_SIGMA_E_DEFAULT) <= 1e-12
    )


def _build_triangle_material_data(built_meshes):
    if len(built_meshes) == 0:
        return None
    total_triangles = sum(int(mesh["n_triangles"]) for mesh in built_meshes)
    if total_triangles <= 0:
        return None

    material_eps_r = Concat.arrays(wt.Float, [mesh["material_eps_r"] for mesh in built_meshes])
    material_sigma_e = Concat.arrays(wt.Float, [mesh["material_sigma_e"] for mesh in built_meshes])
    material_specified = Concat.arrays(wt.Bool, [mesh["material_specified"] for mesh in built_meshes])
    material_structure_idx = Concat.arrays(wt.Int32, [mesh["material_structure_idx"] for mesh in built_meshes])
    n_specified_triangles = sum(
        int(mesh["n_triangles"]) for mesh in built_meshes if mesh["material_specified_value"]
    )
    return {
        "eps_r": material_eps_r,
        "sigma_e": material_sigma_e,
        "specified": material_specified,
        "structure_idx": material_structure_idx,
        "has_specified_materials": bool(n_specified_triangles > 0),
        "n_specified_triangles": int(n_specified_triangles),
        "n_default_material_triangles": int(total_triangles - n_specified_triangles),
    }


def compute_mesh_centers(vertices):
    n_verts = dr.width(vertices)
    mesh_center_3d = wt.Point3f(
        dr.sum(vertices.x) / n_verts,
        dr.sum(vertices.y) / n_verts,
        dr.sum(vertices.z) / n_verts,
    )
    mesh_center_2d = wt.Vector2f(
        dr.sum(vertices.x) / n_verts,
        dr.sum(vertices.y) / n_verts,
    )
    return n_verts, mesh_center_3d, mesh_center_2d


def _count_mask(mask) -> int:
    if int(dr.width(mask)) <= 0:
        return 0
    return int(scalar(dr.sum(dr.select(mask, wt.Int32(1), wt.Int32(0)))))


def _clear_edge_runtime(scene) -> None:
    scene.vertical_edges = []
    scene.edge_selection_summary = empty_edge_selection_summary()
    scene._vertical_edge_id_to_index = {}
    scene._global_diffraction_edge_indices = []
    scene._tri_edge_indices = None
    scene._diffraction_edge_gpu = None
    triangle_surface_data = scene._triangle_surface_data
    if triangle_surface_data is not None:
        triangle_surface_data["surface_edge_size"] = dr.zeros(wt.UInt32, 0)
        triangle_surface_data["surface_edge_indices"] = dr.zeros(wt.Int32, 0)
        triangle_surface_data["max_surface_edge_count"] = 0

def _attach_triangle_surface_data(scene) -> None:
    if scene.tri_data_gpu is None:
        return

    triangle_surface_data = scene._triangle_surface_data
    if triangle_surface_data is None:
        scene.tri_data_gpu["surface_group_id"] = dr.zeros(wt.Int32, 0)
        scene.tri_data_gpu["surface_canonical_prim"] = dr.zeros(wt.Int32, 0)
        scene.tri_data_gpu["surface_group_size"] = dr.zeros(wt.UInt32, 0)
        scene.tri_data_gpu["surface_group_members"] = dr.zeros(wt.Int32, 0)
        scene.tri_data_gpu["surface_max_group_size"] = 0
        scene.tri_data_gpu["surface_edge_size"] = dr.zeros(wt.UInt32, 0)
        scene.tri_data_gpu["surface_edge_indices"] = dr.zeros(wt.Int32, 0)
        scene.tri_data_gpu["surface_max_edge_count"] = 0
        return

    scene.tri_data_gpu["surface_group_id"] = triangle_surface_data["group_id"]
    scene.tri_data_gpu["surface_canonical_prim"] = triangle_surface_data["canonical_prim"]
    scene.tri_data_gpu["surface_group_size"] = triangle_surface_data["group_size"]
    scene.tri_data_gpu["surface_group_members"] = triangle_surface_data["group_members"]
    scene.tri_data_gpu["surface_max_group_size"] = int(triangle_surface_data["max_group_size"])
    scene.tri_data_gpu["surface_edge_size"] = triangle_surface_data["surface_edge_size"]
    scene.tri_data_gpu["surface_edge_indices"] = triangle_surface_data["surface_edge_indices"]
    scene.tri_data_gpu["surface_max_edge_count"] = int(triangle_surface_data["max_surface_edge_count"])


def _attach_triangle_material_data(scene) -> None:
    if scene.tri_data_gpu is None:
        return

    n_triangles = int(scene.tri_data_gpu["n_triangles"])
    triangle_material_data = scene._triangle_material_data
    if triangle_material_data is None:
        scene.tri_data_gpu["material_eps_r"] = dr.full(wt.Float, MATERIAL_EPS_R_DEFAULT, n_triangles)
        scene.tri_data_gpu["material_sigma_e"] = dr.full(wt.Float, MATERIAL_SIGMA_E_DEFAULT, n_triangles)
        scene.tri_data_gpu["material_specified"] = dr.full(wt.Bool, False, n_triangles)
        scene.tri_data_gpu["material_structure_idx"] = dr.full(wt.Int32, -1, n_triangles)
        scene.tri_data_gpu["material_has_specified_materials"] = False
        scene.tri_data_gpu["material_n_specified_triangles"] = 0
        scene.tri_data_gpu["material_n_default_material_triangles"] = int(n_triangles)
        return

    scene.tri_data_gpu["material_eps_r"] = triangle_material_data["eps_r"]
    scene.tri_data_gpu["material_sigma_e"] = triangle_material_data["sigma_e"]
    scene.tri_data_gpu["material_specified"] = triangle_material_data["specified"]
    scene.tri_data_gpu["material_structure_idx"] = triangle_material_data["structure_idx"]
    scene.tri_data_gpu["material_has_specified_materials"] = bool(
        triangle_material_data["has_specified_materials"]
    )
    scene.tri_data_gpu["material_n_specified_triangles"] = int(
        triangle_material_data["n_specified_triangles"]
    )
    scene.tri_data_gpu["material_n_default_material_triangles"] = int(
        triangle_material_data["n_default_material_triangles"]
    )


def refresh_triangle_surface_data(scene) -> None:
    n_triangles = dr.width(scene.faces.x)
    if n_triangles == 0:
        scene._face_normals = None
        scene._triangle_surface_groups = ()
        scene._triangle_surface_group_by_triangle = ()
        scene._triangle_surface_edge_groups = ()
        scene._triangle_surface_data = None
        _attach_triangle_surface_data(scene)
        return

    face_normals = compute_face_normals(
        scene.vertices,
        scene.faces,
        mesh_center_3d=scene._mesh_center_3d,
        n_verts=scene._n_verts,
    )
    scene._face_normals = face_normals

    parent = list(range(int(n_triangles)))

    def _find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def _union(index_a: int, index_b: int) -> None:
        root_a = _find(index_a)
        root_b = _find(index_b)
        if root_a != root_b:
            parent[root_b] = root_a

    for edge_key, face_list in scene._edge_topology:
        del edge_key
        if len(face_list) != 2:
            continue
        face_idx_a = int(face_list[0])
        face_idx_b = int(face_list[1])
        if triangles_are_coplanar(
            face_idx_a,
            face_idx_b,
            face_normals,
            scene.vertices,
            scene.faces,
            normal_cos_tol=SURFACE_GROUP_NORMAL_COS_TOL,
            plane_tol=SURFACE_GROUP_PLANE_TOL,
        ):
            _union(face_idx_a, face_idx_b)

    grouped_faces: dict[int, list[int]] = {}
    for face_idx in range(int(n_triangles)):
        root = _find(face_idx)
        grouped_faces.setdefault(root, []).append(face_idx)

    groups = [tuple(sorted(face_indices)) for face_indices in grouped_faces.values()]
    groups.sort(key=lambda item: item[0])
    group_by_triangle = [0] * int(n_triangles)
    for group_idx, members in enumerate(groups):
        for face_idx in members:
            group_by_triangle[face_idx] = group_idx

    max_group_size = max((len(members) for members in groups), default=0)
    flattened_members: list[int] = []
    group_id_values = []
    canonical_values = []
    group_size_values = []
    for face_idx in range(int(n_triangles)):
        members = groups[group_by_triangle[face_idx]]
        group_id_values.append(group_by_triangle[face_idx])
        canonical_values.append(min(members))
        group_size_values.append(len(members))
        padded_members = list(members) + [-1] * (max_group_size - len(members))
        flattened_members.extend(padded_members)

    scene._triangle_surface_groups = tuple(groups)
    scene._triangle_surface_group_by_triangle = tuple(group_by_triangle)
    scene._triangle_surface_edge_groups = tuple(() for _ in groups)
    scene._triangle_surface_data = {
        "group_id": wt.Int32(*group_id_values),
        "canonical_prim": wt.Int32(*canonical_values),
        "group_size": wt.UInt32(*group_size_values),
        "group_members": wt.Int32(*flattened_members) if max_group_size > 0 else dr.zeros(wt.Int32, 0),
        "max_group_size": int(max_group_size),
        "surface_edge_size": dr.zeros(wt.UInt32, 0),
        "surface_edge_indices": dr.zeros(wt.Int32, 0),
        "max_surface_edge_count": 0,
    }
    _attach_triangle_surface_data(scene)


def preload_minimal_diffraction_edge_runtime(scene) -> None:
    rayd_scene = scene._rayd_scene
    if rayd_scene is None:
        _clear_edge_runtime(scene)
        scene._edge_runtime_dirty = False
        return

    edge_info = rayd_scene.edge_info()
    edge_topology = rayd_scene.edge_topology()
    edge_length_source = edge_info.length
    total_mesh_edges = (
        int(edge_length_source.numel())
        if isinstance(edge_length_source, torch.Tensor)
        else int(dr.width(edge_length_source))
    )
    if total_mesh_edges <= 0:
        _clear_edge_runtime(scene)
        scene._edge_runtime_dirty = False
        return

    edge_vec = wt.Vector3f(edge_info.edge)
    edge_length = wt.Float(edge_length_source)
    edge_length_eps = edge_length + wt.Float(EPS)
    is_boundary = wt.Bool(edge_info.is_boundary)
    valid_length = edge_length > wt.Float(SMALL_EPS)
    vertical_match = valid_length & (
        (dr.abs(edge_vec.z) / edge_length_eps) > wt.Float(scene.vertical_ratio)
    )
    candidate_mask = (
        vertical_match if scene.edge_selection_mode == "vertical_only" else valid_length
    )
    include_mask = (
        candidate_mask & ~is_boundary
        if scene.boundary_edge_policy == "exclude"
        else candidate_mask
    )
    selected_idx = dr.compress(include_mask)
    n_selected = int(dr.width(selected_idx))
    scene.edge_selection_summary = {
        **empty_edge_selection_summary(),
        "selection_mode": scene.edge_selection_mode,
        "total_mesh_edges": int(total_mesh_edges),
        "included_edges": int(n_selected),
    }

    scene.vertical_edges = []
    scene._vertical_edge_id_to_index = {}
    scene._global_diffraction_edge_indices = []
    scene._tri_edge_indices = None

    if n_selected <= 0:
        scene._diffraction_edge_gpu = None
        scene._edge_runtime_dirty = False
        return

    p0 = dr.gather(wt.Point3f, edge_info.start, selected_idx)
    p1 = dr.gather(wt.Point3f, edge_info.end, selected_idx)
    edge_vec = dr.gather(wt.Vector3f, edge_vec, selected_idx)
    edge_length = dr.gather(wt.Float, edge_length, selected_idx)
    edge_dir = edge_vec / (dr.norm(edge_vec) + wt.Float(EPS))
    is_boundary = dr.gather(wt.Bool, is_boundary, selected_idx)
    adjacent_face0 = wt.Int32(
        dr.gather(type(edge_topology.face0_global), edge_topology.face0_global, selected_idx)
    )
    adjacent_face1 = wt.Int32(
        dr.gather(type(edge_topology.face1_global), edge_topology.face1_global, selected_idx)
    )
    triangle_surface_data = scene._triangle_surface_data
    if triangle_surface_data is not None and "group_id" in triangle_surface_data:
        valid_face0 = adjacent_face0 >= 0
        valid_face1 = adjacent_face1 >= 0
        safe_face0 = wt.UInt32(dr.select(valid_face0, adjacent_face0, wt.Int32(0)))
        safe_face1 = wt.UInt32(dr.select(valid_face1, adjacent_face1, wt.Int32(0)))
        adjacent_surface_group0 = dr.select(
            valid_face0,
            dr.gather(wt.Int32, triangle_surface_data["group_id"], safe_face0),
            wt.Int32(-1),
        )
        adjacent_surface_group1 = dr.select(
            valid_face1,
            dr.gather(wt.Int32, triangle_surface_data["group_id"], safe_face1),
            wt.Int32(-1),
        )
    else:
        adjacent_surface_group0 = dr.full(wt.Int32, -1, n_selected)
        adjacent_surface_group1 = dr.full(wt.Int32, -1, n_selected)
    n0_candidate = dr.gather(wt.Vector3f, edge_info.normal0, selected_idx)
    n1_candidate = dr.gather(wt.Vector3f, edge_info.normal1, selected_idx)

    safe_n1_candidate = dr.select(is_boundary, -n0_candidate, n1_candidate)

    to_hat_1 = dr.cross(n0_candidate, edge_dir)
    tn_hat_1 = dr.cross(safe_n1_candidate, edge_dir)
    to_hat_2 = dr.cross(safe_n1_candidate, edge_dir)
    tn_hat_2 = dr.cross(n0_candidate, edge_dir)

    to_hat_1 = to_hat_1 / (dr.norm(to_hat_1) + wt.Float(EPS))
    tn_hat_1 = tn_hat_1 / (dr.norm(tn_hat_1) + wt.Float(EPS))
    to_hat_2 = to_hat_2 / (dr.norm(to_hat_2) + wt.Float(EPS))
    tn_hat_2 = tn_hat_2 / (dr.norm(tn_hat_2) + wt.Float(EPS))

    cross_1 = dr.cross(to_hat_1, tn_hat_1)
    dot_1 = dr.dot(to_hat_1, tn_hat_1)
    sign_1 = dr.sign(dr.dot(cross_1, edge_dir))
    angle_1 = dr.atan2(sign_1 * dr.norm(cross_1), dot_1)
    angle_1 = dr.select(angle_1 < 0, angle_1 + 2 * dr.pi, angle_1)

    cross_2 = dr.cross(to_hat_2, tn_hat_2)
    dot_2 = dr.dot(to_hat_2, tn_hat_2)
    sign_2 = dr.sign(dr.dot(cross_2, edge_dir))
    angle_2 = dr.atan2(sign_2 * dr.norm(cross_2), dot_2)
    angle_2 = dr.select(angle_2 < 0, angle_2 + 2 * dr.pi, angle_2)

    choose_first = angle_1 < angle_2
    interior_n0 = dr.select(choose_first, n0_candidate, safe_n1_candidate)
    interior_n1 = dr.select(choose_first, safe_n1_candidate, n0_candidate)
    n0 = dr.select(is_boundary, n0_candidate, interior_n0)
    n1 = (
        dr.select(is_boundary, -n0_candidate, interior_n1)
        if scene.boundary_edge_policy == "half_plane"
        else interior_n1
    )

    dot_product = dr.clip(-dr.dot(n0, n1), -1.0, 1.0)
    interior_angle = dr.acos(dot_product)
    exterior_angle = 2 * dr.pi - interior_angle
    wedge_n = dr.select(
        is_boundary,
        wt.Float(2.0),
        exterior_angle / dr.pi,
    )

    half_length = wt.Float(0.5) * edge_length
    scene._diffraction_edge_gpu = {
        "pos": (p0 + p1) * wt.Float(0.5),
        "edge_dir": edge_dir,
        "n0": n0,
        "n_face_n": n1,
        "wedge_n": wedge_n,
        "length": edge_length,
        "line_min": -half_length,
        "line_max": half_length,
        "adjacent_face0": adjacent_face0,
        "adjacent_face1": adjacent_face1,
        "adjacent_surface_group0": adjacent_surface_group0,
        "adjacent_surface_group1": adjacent_surface_group1,
        "n_edges": int(n_selected),
    }
    scene._edge_runtime_dirty = True


def _ensure_edge_runtime(scene) -> None:
    if not scene._edge_runtime_dirty:
        return
    build_vertical_edges(scene)
    build_triangle_edge_mapping(scene)
    scene._edge_runtime_dirty = False

def build_rayd_scene(vertices, faces, *, device: str | torch.device | None = None, version: int = 0):
    if dr.width(vertices) == 0 or dr.width(faces) == 0:
        return _EmptyRayDScene(device=_torch_runtime_device(device), _version=int(version))

    mesh = rayd.Mesh(vertices, faces)
    rayd_scene = rayd.Scene()
    rayd_scene.add_mesh(mesh, dynamic=True)
    rayd_scene.build()
    rayd_scene.sync()
    return rayd_scene


def configure_runtime_backends(scene, *, update_rayd_vertices: bool = False) -> None:
    scene._wedge_backend_error = None

    try:
        has_geometry = dr.width(scene.vertices) > 0 and dr.width(scene.faces) > 0
        can_update_rayd = (
            update_rayd_vertices
            and has_geometry
            and scene._rayd_scene is not None
            and not isinstance(scene._rayd_scene, _EmptyRayDScene)
        )
        if can_update_rayd:
            scene._rayd_scene.update_mesh_vertices(0, scene.vertices)
            scene._rayd_scene.sync()
        else:
            scene._rayd_scene = build_rayd_scene(
                scene.vertices,
                scene.faces,
                device=scene.device,
                version=scene._mesh_version,
            )
    except Exception as exc:  # pragma: no cover - exercised in environment-dependent paths
        scene._rayd_scene = None
        raise RuntimeError("Failed to build the RayD runtime scene.") from exc

    scene._wedge_backend_source = scene._rayd_scene
    scene._wedge_backend_kind = "rayd"


def _merge_meshes(*meshes):
    """Merge multiple ``(vertices, faces)`` pairs into one mesh."""
    if len(meshes) == 1:
        return meshes[0]
    all_vx, all_vy, all_vz = [], [], []
    all_f0, all_f1, all_f2 = [], [], []
    vertex_offset = 0
    for vertices, faces in meshes:
        all_vx.append(vertices.x)
        all_vy.append(vertices.y)
        all_vz.append(vertices.z)
        all_f0.append(faces.x + wt.UInt32(vertex_offset))
        all_f1.append(faces.y + wt.UInt32(vertex_offset))
        all_f2.append(faces.z + wt.UInt32(vertex_offset))
        vertex_offset += dr.width(vertices)
    return (
        wt.Point3f(
            Concat.arrays(wt.Float, all_vx),
            Concat.arrays(wt.Float, all_vy),
            Concat.arrays(wt.Float, all_vz),
        ),
        wt.Vector3u(
            Concat.arrays(wt.UInt32, all_f0),
            Concat.arrays(wt.UInt32, all_f1),
            Concat.arrays(wt.UInt32, all_f2),
        ),
    )


def rebuild_runtime(scene):
    from .wedge.runtime import RUNTIME_REGISTRY

    RUNTIME_REGISTRY.discard(scene)
    built_meshes = build_structure_meshes(scene)
    if built_meshes:
        scene.vertices, scene.faces = _merge_meshes(
            *((mesh["vertices"], mesh["faces"]) for mesh in built_meshes)
        )
        scene._triangle_material_data = _build_triangle_material_data(built_meshes)
        edge_to_faces = extract_edges_with_adjacency(scene.vertices, scene.faces)
        scene._edge_topology = sorted(edge_to_faces.items(), key=lambda item: item[0])
    else:
        scene.vertices = ArrayInit.empty_point3()
        scene.faces = ArrayInit.empty_vector3u()
        scene._edge_topology = []
        scene._triangle_material_data = None

    configure_runtime_backends(scene)
    scene._mesh_version += 1
    scene._edge_cache.clear()
    if dr.width(scene.vertices) > 0:
        scene._n_verts, scene._mesh_center_3d, scene._mesh_center_2d = compute_mesh_centers(scene.vertices)
    else:
        scene._n_verts = 0
        scene._mesh_center_3d = wt.Point3f(0.0, 0.0, 0.0)
        scene._mesh_center_2d = wt.Vector2f(0.0, 0.0)
    preload_triangle_data(scene)
    refresh_triangle_surface_data(scene)
    preload_minimal_diffraction_edge_runtime(scene)


def build_vertical_edges(scene):
    if dr.width(scene.vertices) == 0 or dr.width(scene.faces) == 0:
        scene._n_verts = 0
        scene._mesh_center_3d = wt.Point3f(0.0, 0.0, 0.0)
        scene._mesh_center_2d = wt.Vector2f(0.0, 0.0)
        _clear_edge_runtime(scene)
        scene._edge_runtime_dirty = False
        return

    scene._n_verts, scene._mesh_center_3d, scene._mesh_center_2d = compute_mesh_centers(scene.vertices)
    face_normals = scene._face_normals
    if face_normals is None:
        face_normals = compute_face_normals(
            scene.vertices,
            scene.faces,
            mesh_center_3d=scene._mesh_center_3d,
            n_verts=scene._n_verts,
        )
        scene._face_normals = face_normals
    vertical_edges_raw, edge_selection_summary = filter_diffraction_edges(
        scene.vertices,
        scene._edge_topology,
        scene.vertical_ratio,
        edge_selection_mode=scene.edge_selection_mode,
        boundary_edge_policy=scene.boundary_edge_policy,
    )
    edge_selection_summary.setdefault("excluded_endpoint_anchors", 0)
    scene.edge_selection_summary = edge_selection_summary

    scene.vertical_edges = []
    for edge_info in vertical_edges_raw:
        compute_edge_geometry(
            edge_info,
            scene.vertices,
            scene.faces,
            mesh_center_3d=scene._mesh_center_3d,
            mesh_center_2d=scene._mesh_center_2d,
            n_verts=scene._n_verts,
            face_normals=face_normals,
            boundary_edge_policy=scene.boundary_edge_policy,
        )
        scene.vertical_edges.append(edge_info)
    scene._vertical_edge_id_to_index = {id(edge): idx for idx, edge in enumerate(scene.vertical_edges)}
    scene._edge_runtime_dirty = False


def preload_triangle_data(scene):
    n_triangles = dr.width(scene.faces)
    if n_triangles == 0:
        scene.tri_data_gpu = None
        return

    v0_idx = scene.faces.x
    v1_idx = scene.faces.y
    v2_idx = scene.faces.z
    scene.tri_data_gpu = {
        "v0": dr.gather(wt.Point3f, scene.vertices, v0_idx),
        "v1": dr.gather(wt.Point3f, scene.vertices, v1_idx),
        "v2": dr.gather(wt.Point3f, scene.vertices, v2_idx),
        "n_triangles": n_triangles,
    }
    dr.eval(
        scene.tri_data_gpu["v0"].x,
        scene.tri_data_gpu["v0"].y,
        scene.tri_data_gpu["v0"].z,
        scene.tri_data_gpu["v1"].x,
        scene.tri_data_gpu["v1"].y,
        scene.tri_data_gpu["v1"].z,
        scene.tri_data_gpu["v2"].x,
        scene.tri_data_gpu["v2"].y,
        scene.tri_data_gpu["v2"].z,
    )
    _attach_triangle_material_data(scene)
    _attach_triangle_surface_data(scene)


def build_triangle_edge_mapping(scene):
    n_triangles = dr.width(scene.faces)
    if n_triangles == 0:
        _clear_edge_runtime(scene)
        scene._edge_runtime_dirty = False
        return

    diffraction_edge_keys = {}
    scene._global_diffraction_edge_indices = []
    for idx, edge_info in enumerate(scene.vertical_edges):
        if edge_info.wedge_n is None:
            continue
        wedge_n_val = scalar(edge_info.wedge_n)
        if wedge_n_val <= 1.0 + SMALL_EPS:
            continue
        v0_idx, v1_idx = edge_info.vertex_indices
        edge_key = (min(v0_idx, v1_idx), max(v0_idx, v1_idx))
        diffraction_edge_keys[edge_key] = idx
        scene._global_diffraction_edge_indices.append(idx)

    f0 = scene.faces.x
    f1 = scene.faces.y
    f2 = scene.faces.z
    tri_edge_indices = []

    for tri_idx in range(n_triangles):
        v0 = int(f0[tri_idx])
        v1 = int(f1[tri_idx])
        v2 = int(f2[tri_idx])
        edges_of_tri = [
            (min(v0, v1), max(v0, v1)),
            (min(v1, v2), max(v1, v2)),
            (min(v2, v0), max(v2, v0)),
        ]
        edge_indices = [diffraction_edge_keys[edge_key] for edge_key in edges_of_tri if edge_key in diffraction_edge_keys]
        while len(edge_indices) < 3:
            edge_indices.append(-1)
        tri_edge_indices.append(tuple(edge_indices[:3]))

    edge0_list = [tri[0] for tri in tri_edge_indices]
    edge1_list = [tri[1] for tri in tri_edge_indices]
    edge2_list = [tri[2] for tri in tri_edge_indices]
    scene._tri_edge_indices = {
        "edge0": wt.Int32(*edge0_list),
        "edge1": wt.Int32(*edge1_list),
        "edge2": wt.Int32(*edge2_list),
        "n_triangles": n_triangles,
        "n_diffraction_edges": len(scene.vertical_edges),
    }

    triangle_surface_data = scene._triangle_surface_data
    if triangle_surface_data is not None:
        surface_edge_groups = []
        for members in scene._triangle_surface_groups:
            ordered_edges = []
            seen_edges = set()
            for tri_idx in members:
                for edge_idx in tri_edge_indices[int(tri_idx)]:
                    if edge_idx >= 0 and edge_idx not in seen_edges:
                        seen_edges.add(edge_idx)
                        ordered_edges.append(edge_idx)
            surface_edge_groups.append(tuple(ordered_edges))

        scene._triangle_surface_edge_groups = tuple(surface_edge_groups)
        max_surface_edge_count = max((len(edges) for edges in surface_edge_groups), default=0)
        surface_edge_size_values = []
        flattened_surface_edges = []
        for face_idx in range(int(n_triangles)):
            group_idx = scene._triangle_surface_group_by_triangle[face_idx]
            edges = surface_edge_groups[group_idx]
            surface_edge_size_values.append(len(edges))
            flattened_surface_edges.extend(list(edges) + [-1] * (max_surface_edge_count - len(edges)))

        triangle_surface_data["surface_edge_size"] = (
            wt.UInt32(*surface_edge_size_values) if len(surface_edge_size_values) > 0 else dr.zeros(wt.UInt32, 0)
        )
        triangle_surface_data["surface_edge_indices"] = (
            wt.Int32(*flattened_surface_edges) if max_surface_edge_count > 0 else dr.zeros(wt.Int32, 0)
        )
        triangle_surface_data["max_surface_edge_count"] = int(max_surface_edge_count)
        _attach_triangle_surface_data(scene)

    preload_diffraction_edge_geometry(scene)


def preload_diffraction_edge_geometry(scene):
    n_edges = len(scene.vertical_edges)
    if n_edges == 0:
        scene._diffraction_edge_gpu = None
        return

    pos_x, pos_y, pos_z = [], [], []
    edge_dir_x, edge_dir_y, edge_dir_z = [], [], []
    n0_x, n0_y, n0_z = [], [], []
    nn_x, nn_y, nn_z = [], [], []
    wedge_n_list = []
    length_list = []
    line_min_list = []
    line_max_list = []
    adjacent_face0_list = []
    adjacent_face1_list = []
    adjacent_surface_group0_list = []
    adjacent_surface_group1_list = []

    for edge_info in scene.vertical_edges:
        pos_x.append((edge_info.p0.x + edge_info.p1.x) * wt.Float(0.5))
        pos_y.append((edge_info.p0.y + edge_info.p1.y) * wt.Float(0.5))
        pos_z.append((edge_info.p0.z + edge_info.p1.z) * wt.Float(0.5))

        e_vec = edge_info.edge_vector
        e_len = dr.norm(e_vec) + wt.Float(EPS)
        edge_dir_x.append(e_vec.x / e_len)
        edge_dir_y.append(e_vec.y / e_len)
        edge_dir_z.append(e_vec.z / e_len)

        if edge_info.face_normals_3d and len(edge_info.face_normals_3d) >= 2:
            n0 = edge_info.face_normals_3d[0]
            nn = edge_info.face_normals_3d[1]
            n0_x.append(n0.x)
            n0_y.append(n0.y)
            n0_z.append(n0.z)
            nn_x.append(nn.x)
            nn_y.append(nn.y)
            nn_z.append(nn.z)
        else:
            n0_x.append(wt.Float(0.0))
            n0_y.append(wt.Float(0.0))
            n0_z.append(wt.Float(1.0))
            nn_x.append(wt.Float(0.0))
            nn_y.append(wt.Float(0.0))
            nn_z.append(wt.Float(-1.0))

        wedge_n_list.append(
            wt.Float(edge_info.wedge_n) if edge_info.wedge_n is not None else wt.Float(1.5)
        )
        edge_length = wt.Float(edge_info.length)
        half_length = wt.Float(0.5) * edge_length
        length_list.append(edge_length)
        line_min_list.append(-half_length)
        line_max_list.append(half_length)
        adjacent_faces = tuple(int(face) for face in (edge_info.adjacent_faces or ()))
        adjacent_face0_list.append(adjacent_faces[0] if len(adjacent_faces) > 0 else -1)
        adjacent_face1_list.append(adjacent_faces[1] if len(adjacent_faces) > 1 else -1)
        if scene._triangle_surface_data is not None and "group_id" in scene._triangle_surface_data:
            group_ids = scene._triangle_surface_data["group_id"]
            adjacent_surface_group0_list.append(
                int(group_ids[adjacent_faces[0]]) if len(adjacent_faces) > 0 else -1
            )
            adjacent_surface_group1_list.append(
                int(group_ids[adjacent_faces[1]]) if len(adjacent_faces) > 1 else -1
            )
        else:
            adjacent_surface_group0_list.append(-1)
            adjacent_surface_group1_list.append(-1)

    scene._diffraction_edge_gpu = {
        "pos": wt.Point3f(
            Concat.arrays(wt.Float, pos_x),
            Concat.arrays(wt.Float, pos_y),
            Concat.arrays(wt.Float, pos_z),
        ),
        "edge_dir": wt.Vector3f(
            Concat.arrays(wt.Float, edge_dir_x),
            Concat.arrays(wt.Float, edge_dir_y),
            Concat.arrays(wt.Float, edge_dir_z),
        ),
        "n0": wt.Vector3f(
            Concat.arrays(wt.Float, n0_x),
            Concat.arrays(wt.Float, n0_y),
            Concat.arrays(wt.Float, n0_z),
        ),
        "n_face_n": wt.Vector3f(
            Concat.arrays(wt.Float, nn_x),
            Concat.arrays(wt.Float, nn_y),
            Concat.arrays(wt.Float, nn_z),
        ),
        "wedge_n": Concat.arrays(wt.Float, wedge_n_list),
        "length": Concat.arrays(wt.Float, length_list),
        "line_min": Concat.arrays(wt.Float, line_min_list),
        "line_max": Concat.arrays(wt.Float, line_max_list),
        "adjacent_face0": wt.Int32(*adjacent_face0_list),
        "adjacent_face1": wt.Int32(*adjacent_face1_list),
        "adjacent_surface_group0": wt.Int32(*adjacent_surface_group0_list),
        "adjacent_surface_group1": wt.Int32(*adjacent_surface_group1_list),
        "n_edges": n_edges,
    }


def update_vertices(scene, vertices, recompute_edges: bool = True):
    from .wedge.runtime import RUNTIME_REGISTRY

    RUNTIME_REGISTRY.discard(scene)
    vertices = to_point3f(vertices)
    if dr.width(vertices) != scene._n_verts:
        raise ValueError(
            f"Expected {scene._n_verts} vertices when updating the scene, got {dr.width(vertices)}."
        )

    scene.vertices = vertices
    scene._runtime_vertex_override = vertices
    configure_runtime_backends(scene, update_rayd_vertices=True)
    scene._mesh_version += 1
    scene._edge_cache.clear()
    if dr.width(scene.vertices) > 0:
        scene._n_verts, scene._mesh_center_3d, scene._mesh_center_2d = compute_mesh_centers(scene.vertices)
    else:
        scene._n_verts = 0
        scene._mesh_center_3d = wt.Point3f(0.0, 0.0, 0.0)
        scene._mesh_center_2d = wt.Vector2f(0.0, 0.0)
    preload_triangle_data(scene)
    refresh_triangle_surface_data(scene)
    if recompute_edges:
        preload_minimal_diffraction_edge_runtime(scene)
    else:
        _clear_edge_runtime(scene)
        scene._edge_runtime_dirty = True
