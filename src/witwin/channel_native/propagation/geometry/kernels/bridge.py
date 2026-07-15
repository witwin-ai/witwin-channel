from __future__ import annotations

import torch

from witwin.channel_native.runtime.symbols import required_symbol as _required_native_op
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor
from witwin.channel_native.scene.native_handles import (
    _raydn_module_handle,
    _raydn_scene_handle_id,
)


def bdpt_visibility_forward(
    handle: int,
    start: torch.Tensor,
    end: torch.Tensor,
    active: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    validate_cuda_tensor(
        "start", start, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("end", end, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if active is not None:
        validate_cuda_tensor("active", active, dtype=torch.bool, ndim=1)
    out = _required_native_op("bdpt_visibility_forward")(
        _raydn_scene_handle_id(handle),
        start,
        end,
        active,
        _raydn_module_handle(),
    )
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "_channel_native.bdpt_visibility_forward must return a tensor sequence"
        )
    return tuple(out)


_BDPT_INTERSECTION_FIELDS = (
    "t",
    "p",
    "n",
    "geo_n",
    "uv",
    "barycentric",
    "shape_id",
    "prim_id",
    "local_prim_id",
    "global_prim_id",
)


def bdpt_intersect_forward(
    handle: object,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    ray_tmax: torch.Tensor,
    active: torch.Tensor | None,
    *,
    flags: int = 7,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "ray_o", ray_o, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "ray_d", ray_d, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("ray_tmax", ray_tmax, dtype=torch.float32, ndim=1)
    if ray_d.shape != ray_o.shape:
        raise ValueError("ray_d must match ray_o")
    if ray_tmax.shape not in ((0,), (ray_o.shape[0],)):
        raise ValueError("ray_tmax must be empty or match ray_o")
    if active is not None:
        validate_cuda_tensor("active", active, dtype=torch.bool, ndim=1)
        if active.shape not in ((0,), (ray_o.shape[0],)):
            raise ValueError("active must be empty or match ray_o")
        if active.get_device() != ray_o.get_device():
            raise ValueError("active must share ray_o device")
    if (
        ray_d.get_device() != ray_o.get_device()
        or ray_tmax.get_device() != ray_o.get_device()
    ):
        raise ValueError("intersection tensors must share one CUDA device")
    if flags < 0:
        raise ValueError("flags must be non-negative")
    out = _required_native_op("bdpt_intersect_forward")(
        _raydn_scene_handle_id(handle),
        ray_o,
        ray_d,
        ray_tmax,
        active,
        int(flags),
        _raydn_module_handle(),
    )
    if not isinstance(out, (tuple, list)) or len(out) != len(_BDPT_INTERSECTION_FIELDS):
        raise TypeError("_channel_native.bdpt_intersect_forward must return 10 tensors")
    exported = dict(zip(_BDPT_INTERSECTION_FIELDS, out, strict=True))
    validate_cuda_tensor("t", exported["t"], dtype=torch.float32, ndim=1)
    if exported["t"].shape != (ray_o.shape[0],):
        raise ValueError("_channel_native.bdpt_intersect_forward returned bad t shape")
    for name in ("p", "n", "geo_n", "barycentric"):
        validate_cuda_tensor(
            name, exported[name], dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
    validate_cuda_tensor(
        "uv", exported["uv"], dtype=torch.float32, ndim=2, trailing_shape=(2,)
    )
    for name in ("shape_id", "prim_id", "local_prim_id", "global_prim_id"):
        validate_cuda_tensor(name, exported[name], dtype=torch.int32, ndim=1)
    return exported


def bdpt_reflection_accumulation_forward(*args: object) -> tuple[torch.Tensor, ...]:
    if not args:
        raise TypeError(
            "bdpt_reflection_accumulation_forward requires a RayDN scene handle"
        )
    native_args = (_raydn_scene_handle_id(args[0]), *args[1:])
    out = _required_native_op("bdpt_reflection_accumulation_forward")(
        *native_args,
        _raydn_module_handle(),
    )
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "_channel_native.bdpt_reflection_accumulation_forward must return a tensor sequence"
        )
    return tuple(out)


def bdpt_diffraction_discover_edges(*args: object) -> torch.Tensor:
    out = _required_native_op("bdpt_diffraction_discover_edges")(
        *args,
        _raydn_module_handle(),
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.bdpt_diffraction_discover_edges must return a tensor"
        )
    return out


def bdpt_diffraction_discover_edges_counted(*args: object) -> torch.Tensor:
    out = _required_native_op("bdpt_diffraction_discover_edges_counted")(
        *args,
        _raydn_module_handle(),
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.bdpt_diffraction_discover_edges_counted must return a tensor"
        )
    return out


def bdpt_diffraction_accumulation_forward(*args: object) -> tuple[torch.Tensor, ...]:
    if not args:
        raise TypeError(
            "bdpt_diffraction_accumulation_forward requires a RayDN scene handle"
        )
    native_args = (_raydn_scene_handle_id(args[0]), *args[1:])
    out = _required_native_op("bdpt_diffraction_accumulation_forward")(
        *native_args,
        _raydn_module_handle(),
    )
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "_channel_native.bdpt_diffraction_accumulation_forward must return a tensor sequence"
        )
    return tuple(out)


raydn_visibility_forward = bdpt_visibility_forward
raydn_reflection_accumulation_forward = bdpt_reflection_accumulation_forward
raydn_diffraction_discover_edges = bdpt_diffraction_discover_edges
raydn_diffraction_discover_edges_counted = bdpt_diffraction_discover_edges_counted
raydn_diffraction_accumulation_forward = bdpt_diffraction_accumulation_forward


def raydn_trace_reflections_forward(*args: object) -> tuple[torch.Tensor, ...]:
    if not args:
        raise TypeError("raydn_trace_reflections_forward requires a RayDN scene handle")
    native_args = (_raydn_scene_handle_id(args[0]), *args[1:])
    out = _required_native_op("raydn_trace_reflections_forward")(
        *native_args,
        _raydn_module_handle(),
    )
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "_channel_native.raydn_trace_reflections_forward must return a tensor sequence"
        )
    return tuple(out)


def raydn_reflection_epc_paths_forward(*args: object) -> tuple[torch.Tensor, ...]:
    if not args:
        raise TypeError(
            "raydn_reflection_epc_paths_forward requires a RayDN scene handle"
        )
    native_args = (_raydn_scene_handle_id(args[0]), *args[1:])
    out = _required_native_op("raydn_reflection_epc_paths_forward")(
        *native_args,
        _raydn_module_handle(),
    )
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "_channel_native.raydn_reflection_epc_paths_forward must return a tensor sequence"
        )
    return tuple(out)


__all__ = [
    "bdpt_diffraction_accumulation_forward",
    "bdpt_diffraction_discover_edges",
    "bdpt_diffraction_discover_edges_counted",
    "bdpt_intersect_forward",
    "bdpt_reflection_accumulation_forward",
    "bdpt_visibility_forward",
    "raydn_diffraction_accumulation_forward",
    "raydn_diffraction_discover_edges",
    "raydn_diffraction_discover_edges_counted",
    "raydn_reflection_accumulation_forward",
    "raydn_reflection_epc_paths_forward",
    "raydn_trace_reflections_forward",
    "raydn_visibility_forward",
]
