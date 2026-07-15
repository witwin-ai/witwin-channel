from __future__ import annotations

import torch

from witwin.channel_native.runtime.symbols import required_symbol as _required_native_op


def _raydn_module_handle() -> int:
    # Kept temporarily in the internal call signature while the C++ bridge is
    # converted to direct linkage. RayD no longer has a separately loaded
    # Python extension, so there is no OS module handle to pass.
    return 0


def _raydn_scene_handle_id(handle: object) -> int:
    if isinstance(handle, int):
        return handle
    value = getattr(handle, "handle", None)
    if isinstance(value, int):
        return value
    handle_fn = getattr(handle, "handle", None)
    if callable(handle_fn):
        value = handle_fn()
        if isinstance(value, int):
            return value
    raise TypeError("RayDN scene handle must be an int or expose handle() -> int")


def raydn_scene_create(
    vertices: list[torch.Tensor],
    faces: list[torch.Tensor],
    uv: list[torch.Tensor],
    face_uv: list[torch.Tensor],
    to_world_left: list[torch.Tensor],
    to_world_right: list[torch.Tensor],
    mesh_flags: list[int],
) -> tuple[int, object]:
    out = _required_native_op("raydn_scene_create")(
        vertices,
        faces,
        uv,
        face_uv,
        to_world_left,
        to_world_right,
        mesh_flags,
        _raydn_module_handle(),
    )
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise TypeError(
            "_channel_native.raydn_scene_create must return (handle, owner)"
        )
    handle = out[0]
    if not isinstance(handle, int) or handle == 0:
        raise RuntimeError(
            "_channel_native.raydn_scene_create returned an invalid handle"
        )
    return handle, out[1]


def raydn_scene_edge_records(handle: int) -> tuple[torch.Tensor, ...]:
    out = _required_native_op("raydn_scene_edge_records")(
        _raydn_scene_handle_id(handle),
        _raydn_module_handle(),
    )
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "_channel_native.raydn_scene_edge_records must return a tensor sequence"
        )
    return tuple(out)


__all__ = [
    "_raydn_module_handle",
    "_raydn_scene_handle_id",
    "raydn_scene_create",
    "raydn_scene_edge_records",
]
