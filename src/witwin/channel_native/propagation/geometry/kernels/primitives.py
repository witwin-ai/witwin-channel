from __future__ import annotations

import torch

from witwin.channel_native.runtime.symbols import (
    native_extension,
    required_symbol as _required_native_op,
)
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor


def core_diffraction_edge_count(
    *,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_normals: torch.Tensor,
    edge_v0: torch.Tensor,
    edge_v1: torch.Tensor,
    face0: torch.Tensor,
    face1: torch.Tensor,
    vertical_only: bool,
    vertical_ratio: float,
    boundary_half_plane: bool,
    plane_tol: float,
) -> int:
    validate_cuda_tensor(
        "vertices", vertices, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("faces", faces, dtype=torch.int32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor(
        "face_normals", face_normals, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("edge_v0", edge_v0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("edge_v1", edge_v1, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face0", face0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face1", face1, dtype=torch.int32, ndim=1)
    if faces.shape[0] != face_normals.shape[0]:
        raise ValueError("face_normals must match faces")
    for name, tensor in {"edge_v1": edge_v1, "face0": face0, "face1": face1}.items():
        if tensor.shape != edge_v0.shape:
            raise ValueError(f"{name} must match edge_v0")
    value = _required_native_op("core_diffraction_edge_count")(
        vertices,
        faces,
        face_normals,
        edge_v0,
        edge_v1,
        face0,
        face1,
        bool(vertical_only),
        float(vertical_ratio),
        bool(boundary_half_plane),
        float(plane_tol),
    )
    if not isinstance(value, int):
        raise TypeError(
            "_channel_native.core_diffraction_edge_count must return an int"
        )
    return value


def deterministic_normalize_vec3(
    values: torch.Tensor, *, eps: float = 1.0e-6
) -> torch.Tensor:
    validate_cuda_tensor(
        "values", values, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    out = _required_native_op("deterministic_normalize_vec3")(values, float(eps))
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.deterministic_normalize_vec3 must return a tensor"
        )
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if out.shape != values.shape:
        raise ValueError(
            "_channel_native.deterministic_normalize_vec3 returned bad shape"
        )
    return out


def deterministic_reflect_points(
    points: torch.Tensor,
    plane_points: torch.Tensor,
    normals: torch.Tensor,
) -> torch.Tensor:
    validate_cuda_tensor(
        "points", points, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "plane_points", plane_points, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "normals", normals, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if plane_points.shape != points.shape or normals.shape != points.shape:
        raise ValueError("points, plane_points, and normals must have matching shapes")
    out = _required_native_op("deterministic_reflect_points")(
        points, plane_points, normals
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.deterministic_reflect_points must return a tensor"
        )
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if out.shape != points.shape:
        raise ValueError(
            "_channel_native.deterministic_reflect_points returned bad shape"
        )
    return out


def deterministic_face_groups(
    tri_a: torch.Tensor,
    normals: torch.Tensor,
    surface_ids: torch.Tensor,
    *,
    quantization: float,
) -> dict[str, torch.Tensor | int]:
    validate_cuda_tensor(
        "tri_a", tri_a, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "normals", normals, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("surface_ids", surface_ids, dtype=torch.int64, ndim=1)
    if normals.shape != tri_a.shape:
        raise ValueError("normals must match tri_a")
    if surface_ids.shape != (tri_a.shape[0],):
        raise ValueError("surface_ids must match tri_a")
    if quantization <= 0.0:
        raise ValueError("quantization must be positive")
    exported = _required_native_op("deterministic_face_groups")(
        tri_a,
        normals,
        surface_ids,
        float(quantization),
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.deterministic_face_groups must return a dict")
    expected_fields = {
        "face_group_id",
        "representative_faces",
        "surface_group_id",
        "surface_group_size",
        "surface_group_members",
        "group_count",
    }
    if set(exported) != expected_fields:
        raise ValueError(
            "_channel_native.deterministic_face_groups returned unexpected fields"
        )
    face_count = int(tri_a.shape[0])
    validate_cuda_tensor(
        "face_group_id", exported["face_group_id"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "representative_faces",
        exported["representative_faces"],
        dtype=torch.int64,
        ndim=1,
    )
    validate_cuda_tensor(
        "surface_group_id", exported["surface_group_id"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "surface_group_size", exported["surface_group_size"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "surface_group_members",
        exported["surface_group_members"],
        dtype=torch.int32,
        ndim=1,
    )
    group_count = exported["group_count"]
    if not isinstance(group_count, int):
        raise TypeError(
            "_channel_native.deterministic_face_groups returned non-int group_count"
        )
    if exported["face_group_id"].shape != (face_count,) or exported[
        "surface_group_id"
    ].shape != (face_count,):
        raise ValueError(
            "_channel_native.deterministic_face_groups returned bad face group shape"
        )
    if exported["representative_faces"].shape != (group_count,) or exported[
        "surface_group_size"
    ].shape != (group_count,):
        raise ValueError(
            "_channel_native.deterministic_face_groups returned bad group shape"
        )
    return exported


def deterministic_surface_face_groups(
    surface_ids: torch.Tensor,
) -> dict[str, torch.Tensor | int]:
    validate_cuda_tensor("surface_ids", surface_ids, dtype=torch.int64, ndim=1)
    exported = _required_native_op("deterministic_surface_face_groups")(surface_ids)
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.deterministic_surface_face_groups must return a dict"
        )
    expected_fields = {
        "face_group_id",
        "representative_faces",
        "surface_group_id",
        "surface_group_size",
        "surface_group_members",
        "group_count",
    }
    if set(exported) != expected_fields:
        raise ValueError(
            "_channel_native.deterministic_surface_face_groups returned unexpected fields"
        )
    face_count = int(surface_ids.shape[0])
    validate_cuda_tensor(
        "face_group_id", exported["face_group_id"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "representative_faces",
        exported["representative_faces"],
        dtype=torch.int64,
        ndim=1,
    )
    validate_cuda_tensor(
        "surface_group_id", exported["surface_group_id"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "surface_group_size", exported["surface_group_size"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "surface_group_members",
        exported["surface_group_members"],
        dtype=torch.int32,
        ndim=1,
    )
    group_count = exported["group_count"]
    if not isinstance(group_count, int):
        raise TypeError(
            "_channel_native.deterministic_surface_face_groups returned non-int group_count"
        )
    if exported["face_group_id"].shape != (face_count,) or exported[
        "surface_group_id"
    ].shape != (face_count,):
        raise ValueError(
            "_channel_native.deterministic_surface_face_groups returned bad face group shape"
        )
    if exported["representative_faces"].shape != (group_count,) or exported[
        "surface_group_size"
    ].shape != (group_count,):
        raise ValueError(
            "_channel_native.deterministic_surface_face_groups returned bad group shape"
        )
    return exported


def mc_diffraction_edge_geometry(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_normals: torch.Tensor,
    edge_v0: torch.Tensor,
    edge_v1: torch.Tensor,
    face0: torch.Tensor,
    face1: torch.Tensor,
    *,
    plane_tol: float,
) -> tuple[torch.Tensor, ...]:
    validate_cuda_tensor(
        "vertices", vertices, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("faces", faces, dtype=torch.int32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor(
        "face_normals", face_normals, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("edge_v0", edge_v0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("edge_v1", edge_v1, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face0", face0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face1", face1, dtype=torch.int32, ndim=1)
    if (
        edge_v1.shape != edge_v0.shape
        or face0.shape != edge_v0.shape
        or face1.shape != edge_v0.shape
    ):
        raise ValueError("edge_v1, face0, and face1 must match edge_v0 shape")
    native = native_extension()
    if native is None or not hasattr(native, "mc_diffraction_edge_geometry"):
        raise RuntimeError(
            "_channel_native.mc_diffraction_edge_geometry CUDA kernel is required"
        )
    geometry = native.mc_diffraction_edge_geometry(
        vertices,
        faces,
        face_normals,
        edge_v0,
        edge_v1,
        face0,
        face1,
        float(plane_tol),
    )
    if not isinstance(geometry, tuple) or len(geometry) != 11:
        raise TypeError(
            "_channel_native.mc_diffraction_edge_geometry must return 11 tensors"
        )
    validate_cuda_tensor("selected", geometry[0], dtype=torch.bool, ndim=1)
    validate_cuda_tensor(
        "edge_pos", geometry[1], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "edge_dir", geometry[2], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("lengths", geometry[3], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("line_min", geometry[4], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("line_max", geometry[5], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "n0", geometry[6], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "n1", geometry[7], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("face0_out", geometry[8], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face1_out", geometry[9], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("exterior_angle", geometry[10], dtype=torch.float32, ndim=1)
    return geometry


def mc_surface_group_edge_candidates(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_normals: torch.Tensor,
    edge_v0: torch.Tensor,
    edge_v1: torch.Tensor,
    face0: torch.Tensor,
    face1: torch.Tensor,
    selected: torch.Tensor,
    *,
    plane_tol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_cuda_tensor(
        "vertices", vertices, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("faces", faces, dtype=torch.int32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor(
        "face_normals", face_normals, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("edge_v0", edge_v0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("edge_v1", edge_v1, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face0", face0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face1", face1, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("selected", selected, dtype=torch.bool, ndim=1)
    if (
        edge_v1.shape != edge_v0.shape
        or face0.shape != edge_v0.shape
        or face1.shape != edge_v0.shape
    ):
        raise ValueError("edge_v1, face0, and face1 must match edge_v0 shape")
    if selected.shape != edge_v0.shape:
        raise ValueError("selected must match edge_v0 shape")
    native = native_extension()
    if native is None or not hasattr(native, "mc_surface_group_edge_candidates"):
        raise RuntimeError(
            "_channel_native.mc_surface_group_edge_candidates CUDA kernel is required"
        )
    candidates = native.mc_surface_group_edge_candidates(
        vertices,
        faces,
        face_normals,
        edge_v0,
        edge_v1,
        face0,
        face1,
        selected,
        float(plane_tol),
    )
    if not isinstance(candidates, tuple) or len(candidates) != 2:
        raise TypeError(
            "_channel_native.mc_surface_group_edge_candidates must return 2 tensors"
        )
    counts, indices = candidates
    validate_cuda_tensor("counts", counts, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("indices", indices, dtype=torch.int32, ndim=2)
    if counts.shape[0] != faces.shape[0] or indices.shape[0] != faces.shape[0]:
        raise ValueError(
            "_channel_native.mc_surface_group_edge_candidates returned unexpected shapes"
        )
    return counts, indices
