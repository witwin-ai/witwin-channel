"""Shared declarative scene builders for channel tests and demos."""

from __future__ import annotations

import math

import witwin as wt
import numpy as np
import torch
from witwin.channel import DrJitMesh, Material, Scene, Structure
from witwin.channel.utils import to_point3f, to_vector3u
from witwin.core import Box, GeometryBase, Mesh, Prism
_LEGACY_PRISM_YAW_OFFSET = -math.pi / 2.0


def _scalar(value):
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError("Expected a scalar tensor.")
        return value.to(dtype=torch.float32)
    if isinstance(value, (int, float)):
        return float(value)
    return float(np.asarray(value, dtype=np.float32).reshape(-1)[0])


def _vec3(value):
    if isinstance(value, torch.Tensor):
        tensor = value.to(dtype=torch.float32)
        if tensor.ndim != 1 or tensor.shape[0] != 3:
            raise ValueError("Expected a tensor with shape (3,).")
        return tensor
    if isinstance(value, wt.Point3f):
        return tuple(_scalar(component) for component in (value.x, value.y, value.z))
    return tuple(float(component) for component in value)


def _cube_size(size):
    if isinstance(size, torch.Tensor):
        tensor = size.to(dtype=torch.float32)
        if tensor.ndim == 0:
            return torch.stack([tensor, tensor, tensor])
        if tensor.ndim == 1 and tensor.shape[0] == 3:
            return tensor
        raise ValueError("Cube size tensor must be scalar or shape (3,).")
    if isinstance(size, (int, float)):
        scalar = float(size)
        return (scalar, scalar, scalar)
    return tuple(float(component) for component in size)


def _z_rotation(rotation):
    if rotation is None:
        return None
    if isinstance(rotation, torch.Tensor):
        tensor = rotation.to(dtype=torch.float32)
        if tensor.ndim == 0:
            zero = torch.zeros((), device=tensor.device, dtype=tensor.dtype)
            return torch.stack([zero, zero, tensor])
        if tensor.ndim == 1 and tensor.shape[0] == 3:
            return tensor
        raise ValueError("Rotation tensor must be scalar or shape (3,).")
    if isinstance(rotation, (tuple, list)):
        if len(rotation) != 3:
            raise ValueError("Rotation sequences must have three elements.")
        return tuple(float(component) for component in rotation)
    angle = _scalar(rotation)
    return (0.0, 0.0, float(angle))


def _legacy_prism_rotation(rotation):
    if rotation is None:
        return _z_rotation(_LEGACY_PRISM_YAW_OFFSET)
    if isinstance(rotation, torch.Tensor):
        tensor = rotation.to(dtype=torch.float32)
        if tensor.ndim == 0:
            return _z_rotation(tensor + tensor.new_tensor(_LEGACY_PRISM_YAW_OFFSET))
        if tensor.ndim == 1 and tensor.shape[0] == 3:
            adjusted = tensor.clone()
            adjusted[2] = adjusted[2] + adjusted.new_tensor(_LEGACY_PRISM_YAW_OFFSET)
            return adjusted
        raise ValueError("Rotation tensor must be scalar or shape (3,).")
    if isinstance(rotation, (tuple, list)):
        if len(rotation) != 3:
            raise ValueError("Rotation sequences must have three elements.")
        adjusted = [float(component) for component in rotation]
        adjusted[2] += _LEGACY_PRISM_YAW_OFFSET
        return tuple(adjusted)
    return _z_rotation(_scalar(rotation) + _LEGACY_PRISM_YAW_OFFSET)


def box_geometry(*, center, size, rotation=None, device: str | None = "cuda") -> Box:
    return Box(
        position=_vec3(center),
        size=_cube_size(size),
        rotation=_z_rotation(rotation),
        device=device,
    )


def prism_geometry(
    *,
    n_sides: int = 5,
    center,
    radius,
    height,
    rotation=None,
    device: str | None = "cuda",
) -> Prism:
    return Prism(
        position=_vec3(center),
        radius=radius,
        height=height,
        num_sides=n_sides,
        rotation=_legacy_prism_rotation(rotation),
        device=device,
    )


def pentagonal_prism_geometry(*, center, radius, height, rotation=None, device: str | None = "cuda") -> Prism:
    return prism_geometry(
        n_sides=5,
        center=center,
        radius=radius,
        height=height,
        rotation=rotation,
        device=device,
    )


def mesh_from_geometry(geometry: GeometryBase, *, device: str | None = "cuda") -> Mesh:
    vertices, faces = geometry.to_mesh()
    return Mesh(vertices, faces, position=(0.0, 0.0, 0.0), recenter=False, device=device)


def drjit_mesh_from_geometry(geometry: GeometryBase) -> DrJitMesh:
    vertices, faces = geometry.to_mesh()
    return DrJitMesh(to_point3f(vertices), to_vector3u(faces))


def _to_bk_float(value):
    if isinstance(value, wt.Float):
        return value
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError("Expected a scalar tensor.")
        return wt.Float(float(value.detach().cpu()))
    return wt.Float(float(value))


def _to_bk_point3f(value) -> wt.Point3f:
    if isinstance(value, wt.Point3f):
        return value
    if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
        return wt.Point3f(
            _to_bk_float(value.x),
            _to_bk_float(value.y),
            _to_bk_float(value.z),
        )
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(dtype=torch.float32, device="cpu")
        if tensor.ndim != 1 or tensor.shape[0] != 3:
            raise ValueError("Expected a tensor center with shape (3,).")
        return wt.Point3f(
            _to_bk_float(tensor[0]),
            _to_bk_float(tensor[1]),
            _to_bk_float(tensor[2]),
        )
    return wt.Point3f(
        _to_bk_float(value[0]),
        _to_bk_float(value[1]),
        _to_bk_float(value[2]),
    )


def _to_optional_bk_float(value):
    if value is None:
        return None
    return _to_bk_float(value)


def _transform_base_mesh(base_vertices, base_faces, *, center, rotation=None) -> DrJitMesh:
    center_point = _to_bk_point3f(center)
    translate_vec = wt.Vector3f(center_point.x, center_point.y, center_point.z)
    trafo = wt.Transform4f().translate(translate_vec)
    angle = _to_optional_bk_float(rotation)
    if angle is not None:
        trafo = trafo.rotate(wt.Vector3f(0.0, 0.0, 1.0), angle * wt.Float(180.0 / math.pi))
    return DrJitMesh(trafo @ base_vertices, base_faces)


def box_drjit_geometry(*, center, size, rotation=None, device: str | None = "cuda") -> DrJitMesh:
    base_vertices, base_faces = box_geometry(
        center=(0.0, 0.0, 0.0),
        size=size,
        rotation=None,
        device=device,
    ).to_mesh()
    return _transform_base_mesh(
        to_point3f(base_vertices),
        to_vector3u(base_faces),
        center=center,
        rotation=rotation,
    )


def prism_drjit_geometry(
    *,
    n_sides: int = 5,
    center,
    radius,
    height,
    rotation=None,
    device: str | None = "cuda",
) -> DrJitMesh:
    base_vertices, base_faces = prism_geometry(
        n_sides=n_sides,
        center=(0.0, 0.0, 0.0),
        radius=radius,
        height=height,
        rotation=None,
        device=device,
    ).to_mesh()
    return _transform_base_mesh(
        to_point3f(base_vertices),
        to_vector3u(base_faces),
        center=center,
        rotation=rotation,
    )


def pentagonal_prism_drjit_geometry(*, center, radius, height, rotation=None, device: str | None = "cuda") -> DrJitMesh:
    return prism_drjit_geometry(
        n_sides=5,
        center=center,
        radius=radius,
        height=height,
        rotation=rotation,
        device=device,
    )


def mesh_structure(
    mesh_like,
    *,
    name: str,
    material: Material | None = None,
    metadata=None,
) -> Structure:
    if isinstance(mesh_like, Structure):
        return mesh_like
    if isinstance(mesh_like, GeometryBase):
        geometry = mesh_like
    else:
        vertices, faces = mesh_like
        geometry = DrJitMesh(vertices, faces)
    return Structure(
        geometry=geometry,
        material=Material() if material is None else material,
        name=name,
        metadata=metadata,
    )


def build_scene(*mesh_likes, device: str = "cuda", material: Material | None = None, **scene_kwargs) -> Scene:
    structures = [
        mesh_structure(mesh_like, name=f"structure_{index}", material=material)
        for index, mesh_like in enumerate(mesh_likes)
    ]
    return Scene(structures=structures, device=device, **scene_kwargs)

