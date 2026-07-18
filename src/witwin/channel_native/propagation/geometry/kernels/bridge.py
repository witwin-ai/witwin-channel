from __future__ import annotations

import torch

from witwin.channel_native.runtime.symbols import required_symbol as _required_native_op
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor
from witwin.channel_native.runtime.native_handles import _raydn_scene_handle_id


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
    )
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "_channel_native.bdpt_reflection_accumulation_forward must return a tensor sequence"
        )
    return tuple(out)


def bdpt_diffraction_discover_edges(*args: object) -> torch.Tensor:
    out = _required_native_op("bdpt_diffraction_discover_edges")(
        *args,
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.bdpt_diffraction_discover_edges must return a tensor"
        )
    return out


def bdpt_diffraction_discover_edges_counted(*args: object) -> torch.Tensor:
    out = _required_native_op("bdpt_diffraction_discover_edges_counted")(
        *args,
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
    )
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "_channel_native.raydn_reflection_epc_paths_forward must return a tensor sequence"
        )
    return tuple(out)


def raydn_coupled_rd_geometry_forward(*args: object) -> dict[str, torch.Tensor]:
    """Construct reciprocal 1R+1D geometry without evaluating a coefficient.

    The native operation uses image-source edge stationarity, RayDN reflection
    EPC, and RayDN segment visibility. ``reverse=True`` constructs D->R by
    exchanging endpoints and reversing the interaction sequence. The returned
    dictionary intentionally has no ``path_gain`` or ``field`` entry; coupled
    complex/Jones transport belongs to the unified field phase.
    """

    if not args:
        raise TypeError(
            "raydn_coupled_rd_geometry_forward requires a RayDN scene handle"
        )
    native_args = (_raydn_scene_handle_id(args[0]), *args[1:])
    out = _required_native_op("raydn_coupled_rd_geometry_forward")(
        *native_args,
    )
    if not isinstance(out, dict):
        raise TypeError(
            "_channel_native.raydn_coupled_rd_geometry_forward must return a dict"
        )
    required = {
        "valid": (torch.bool, 1),
        "interaction_type_sequence": (torch.int32, 2),
        "primitive_sequence": (torch.int32, 2),
        "edge_sequence": (torch.int32, 2),
        "face_id": (torch.int32, 1),
        "edge_id": (torch.int32, 1),
        "interaction_positions": (torch.float32, 3),
        "interaction_normals": (torch.float32, 3),
        "reflection_position": (torch.float32, 2),
        "reflection_normal": (torch.float32, 2),
        "edge_position": (torch.float32, 2),
        "edge_direction": (torch.float32, 2),
        "path_length_m": (torch.float32, 1),
        "delay_s": (torch.float32, 1),
    }
    for name, (dtype, ndim) in required.items():
        value = out.get(name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"coupled geometry output {name!r} must be a tensor")
        validate_cuda_tensor(name, value, dtype=dtype, ndim=ndim)
    if "path_gain" in out or "field" in out or "path_field" in out:
        raise ValueError(
            "coupled geometry must not expose placeholder physical coefficients"
        )
    count = int(out["valid"].shape[0])
    if out["interaction_type_sequence"].shape != (count, 2):
        raise ValueError("interaction_type_sequence must have shape (N, 2)")
    if out["primitive_sequence"].shape != (count, 2) or out["edge_sequence"].shape != (
        count,
        2,
    ):
        raise ValueError("coupled primitive/edge sequences must have shape (N, 2)")
    if out["interaction_positions"].shape != (count, 2, 3):
        raise ValueError("interaction_positions must have shape (N, 2, 3)")
    if out["interaction_normals"].shape != (count, 2, 3):
        raise ValueError("interaction_normals must have shape (N, 2, 3)")
    return out


def raydn_coupled_dd_geometry_forward(*args: object) -> dict[str, torch.Tensor]:
    """Construct two-edge (double) diffraction geometry without a coefficient.

    The native operation runs an alternating-projection Fermat solve for the
    two-edge Keller point pair (Q1 on e1, Q2 on e2) and three RayDN segment
    visibility queries (tx->Q1, Q1->Q2, Q2->rx). Both edge ids are recoverable
    from ``edge_sequence`` (slot 0 = e1, slot 1 = e2); ``primitive_sequence`` is
    fully ``-1`` because a double-diffraction row touches no face. The returned
    dictionary intentionally carries no ``path_gain``/``field`` entry; complex
    transport belongs to the unified field phase.
    """

    if not args:
        raise TypeError(
            "raydn_coupled_dd_geometry_forward requires a RayDN scene handle"
        )
    native_args = (_raydn_scene_handle_id(args[0]), *args[1:])
    out = _required_native_op("raydn_coupled_dd_geometry_forward")(
        *native_args,
    )
    if not isinstance(out, dict):
        raise TypeError(
            "_channel_native.raydn_coupled_dd_geometry_forward must return a dict"
        )
    required = {
        "valid": (torch.bool, 1),
        "interaction_type_sequence": (torch.int32, 2),
        "primitive_sequence": (torch.int32, 2),
        "edge_sequence": (torch.int32, 2),
        "edge1_id": (torch.int32, 1),
        "edge2_id": (torch.int32, 1),
        "interaction_positions": (torch.float32, 3),
        "interaction_normals": (torch.float32, 3),
        "edge1_position": (torch.float32, 2),
        "edge2_position": (torch.float32, 2),
        "path_length_m": (torch.float32, 1),
        "delay_s": (torch.float32, 1),
    }
    for name, (dtype, ndim) in required.items():
        value = out.get(name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"coupled DD geometry output {name!r} must be a tensor")
        validate_cuda_tensor(name, value, dtype=dtype, ndim=ndim)
    if "path_gain" in out or "field" in out or "path_field" in out:
        raise ValueError(
            "coupled DD geometry must not expose placeholder physical coefficients"
        )
    count = int(out["valid"].shape[0])
    if out["interaction_type_sequence"].shape != (count, 2):
        raise ValueError("interaction_type_sequence must have shape (N, 2)")
    if out["primitive_sequence"].shape != (count, 2) or out["edge_sequence"].shape != (
        count,
        2,
    ):
        raise ValueError("coupled DD primitive/edge sequences must have shape (N, 2)")
    if out["interaction_positions"].shape != (count, 2, 3):
        raise ValueError("interaction_positions must have shape (N, 2, 3)")
    if out["interaction_normals"].shape != (count, 2, 3):
        raise ValueError("interaction_normals must have shape (N, 2, 3)")
    return out


def raydn_diffraction_paths_order1_forward(*args: object) -> tuple[torch.Tensor, ...]:
    if not args:
        raise TypeError(
            "raydn_diffraction_paths_order1_forward requires a RayDN scene handle"
        )
    native_args = (_raydn_scene_handle_id(args[0]), *args[1:])
    out = _required_native_op("raydn_diffraction_paths_order1_forward")(
        *native_args,
    )
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "_channel_native.raydn_diffraction_paths_order1_forward must return a tensor sequence"
        )
    return tuple(out)


__all__ = [
    "bdpt_diffraction_accumulation_forward",
    "bdpt_diffraction_discover_edges",
    "bdpt_diffraction_discover_edges_counted",
    "bdpt_intersect_forward",
    "bdpt_reflection_accumulation_forward",
    "bdpt_visibility_forward",
    "raydn_coupled_dd_geometry_forward",
    "raydn_coupled_rd_geometry_forward",
    "raydn_diffraction_accumulation_forward",
    "raydn_diffraction_discover_edges",
    "raydn_diffraction_discover_edges_counted",
    "raydn_diffraction_paths_order1_forward",
    "raydn_reflection_accumulation_forward",
    "raydn_reflection_epc_paths_forward",
    "raydn_trace_reflections_forward",
    "raydn_visibility_forward",
]
