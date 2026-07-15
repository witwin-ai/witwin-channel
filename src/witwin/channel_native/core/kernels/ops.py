from __future__ import annotations

import torch

from witwin.channel_native.materials.kernels.contracts import (  # noqa: F401
    _validate_layer_csr,
)
from witwin.channel_native.propagation.geometry.kernels.bridge import (  # noqa: F401
    _BDPT_INTERSECTION_FIELDS,
    bdpt_diffraction_accumulation_forward,
    bdpt_diffraction_discover_edges,
    bdpt_diffraction_discover_edges_counted,
    bdpt_intersect_forward,
    bdpt_reflection_accumulation_forward,
    bdpt_visibility_forward,
    raydn_coupled_rd_geometry_forward,
    raydn_diffraction_accumulation_forward,
    raydn_diffraction_discover_edges,
    raydn_diffraction_discover_edges_counted,
    raydn_diffraction_paths_order1_forward,
    raydn_reflection_accumulation_forward,
    raydn_reflection_epc_paths_forward,
    raydn_trace_reflections_forward,
    raydn_visibility_forward,
)
from witwin.channel_native.propagation.geometry.kernels.autograd import (  # noqa: F401
    _RaydnFaceNormalsAdFunction,
    _RaydnIntersectAdFunction,
    _RaydnReflectionEpcPathsAdFunction,
    _RaydnTraceReflectionsAdFunction,
    _epc_paths_frozen_winner_checks,
    raydn_face_normals_ad,
    raydn_intersect_ad,
    raydn_intersect_backward,
    raydn_intersect_jvp,
    raydn_reflection_epc_paths_ad,
    raydn_reflection_epc_paths_backward,
    raydn_reflection_epc_paths_jvp,
    raydn_scene_face_normals_backward,
    raydn_scene_face_normals_jvp,
    raydn_trace_reflections_ad,
    raydn_trace_reflections_backward,
    raydn_trace_reflections_forward_tape,
    raydn_trace_reflections_jvp,
)
from witwin.channel_native.propagation.geometry.kernels.primitives import (  # noqa: F401
    bdpt_diffraction_edge_geometry,
    bdpt_surface_group_edge_candidates,
    core_diffraction_edge_count,
    deterministic_face_groups,
    deterministic_normalize_vec3,
    deterministic_reflect_points,
    deterministic_surface_face_groups,
    mc_diffraction_edge_geometry,
    mc_surface_group_edge_candidates,
)
from witwin.channel_native.runtime import symbols as _native_symbols
from witwin.channel_native.runtime import torch_compat
from witwin.channel_native.runtime.autograd_contracts import (  # noqa: F401
    _ad_active_ctx,
    _ad_check_active,
    _ad_check_optional_grad,
    _ad_check_rows,
    _ad_check_tangent_vec3,
    _ad_checked_tangent,
    _ad_frequency_grad,
    _ad_frequency_tangent,
    _ad_frequency_value,
    _ad_geometry_live,
    _ad_geometry_tangent,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    _ad_raise_composed_transforms,
    _ad_reject_fixed_inputs,
    _ad_reject_fixed_tangents,
    _ad_still_wrapped,
)
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor
from witwin.channel_native.scene.kernels.rayd_scene import (  # noqa: F401
    _raydn_module_handle,
    _raydn_scene_handle_id,
    raydn_scene_create,
    raydn_scene_edge_records,
)

from .metadata import make_metadata, noop_metadata, validate_metadata  # noqa: F401


native_extension = _native_symbols.native_extension


def _required_native_op(name: str):
    return _native_symbols._required_symbol(native_extension(), name)


def bdpt_launch_state(
    reference: torch.Tensor,
    *,
    tx_count: int,
    samples: int,
    sample_streams: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    if tx_count < 0:
        raise ValueError("tx_count must be non-negative")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if sample_streams <= 0:
        raise ValueError("sample_streams must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    native = native_extension()
    if native is None or not hasattr(native, "bdpt_launch_state"):
        raise RuntimeError("_channel_native.bdpt_launch_state CUDA kernel is required")
    exported = native.bdpt_launch_state(
        reference,
        int(tx_count),
        int(samples),
        int(sample_streams),
        int(seed),
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.bdpt_launch_state must return a dict")
    expected = int(tx_count) * int(samples) * int(sample_streams)
    for name in ("tx_id", "sample_id", "stream_id"):
        validate_cuda_tensor(name, exported[name], dtype=torch.int32, ndim=1)
        if exported[name].shape != (expected,):
            raise ValueError(
                f"_channel_native.bdpt_launch_state returned bad {name} shape"
            )
    for name in ("light_seed",):
        validate_cuda_tensor(name, exported[name], dtype=torch.int64, ndim=1)
        if exported[name].shape != (expected,):
            raise ValueError(
                f"_channel_native.bdpt_launch_state returned bad {name} shape"
            )
    return exported


_BDPT_SUBPATH_SCHEMA: dict[str, tuple[torch.dtype, tuple[int | None, ...]]] = {
    "origin": (torch.float32, (None, 3)),
    "direction": (torch.float32, (None, 3)),
    "throughput_real": (torch.float32, (None,)),
    "throughput_imag": (torch.float32, (None,)),
    "pdf_forward": (torch.float32, (None,)),
    "pdf_reverse": (torch.float32, (None,)),
    "depth": (torch.int32, (None,)),
    "component_mask": (torch.int32, (None,)),
    "primitive_id": (torch.int32, (None,)),
    "edge_id": (torch.int32, (None,)),
    "tx_id": (torch.int32, (None,)),
    "rx_id": (torch.int32, (None,)),
    "grid_linear_id": (torch.int32, (None,)),
    "valid": (torch.bool, (None,)),
    "path_length": (torch.float32, (None,)),
    "field_real": (torch.float32, (None, 3)),
    "field_imag": (torch.float32, (None, 3)),
    "source_power": (torch.float32, (None,)),
    "event_type": (torch.int32, (None,)),
}


def _validate_bdpt_subpath_state(
    name: str, exported: dict[str, torch.Tensor], expected_count: int | None
) -> None:
    if not isinstance(exported, dict):
        raise TypeError(f"{name} must be a dict")
    if set(exported) != set(_BDPT_SUBPATH_SCHEMA):
        raise ValueError(f"{name} returned unexpected fields")
    inferred_count: int | None = expected_count
    for field, (dtype, shape_spec) in _BDPT_SUBPATH_SCHEMA.items():
        tensor = exported[field]
        validate_cuda_tensor(field, tensor, dtype=dtype, ndim=len(shape_spec))
        if inferred_count is None:
            inferred_count = int(tensor.shape[0])
        expected_shape = tuple(
            inferred_count if dim is None else dim for dim in shape_spec
        )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} returned bad {field} shape")


_BDPT_CONNECTION_SCHEMA: dict[str, tuple[torch.dtype, tuple[int | None, ...]]] = {
    "topology": (torch.int32, (None, 4)),
    "contribution": (torch.float32, (None,)),
    "pdf": (torch.float32, (None,)),
    "mis_weight": (torch.float32, (None,)),
    "component_id": (torch.int32, (None,)),
    "valid": (torch.bool, (None,)),
    "tx_id": (torch.int32, (None,)),
    "rx_id": (torch.int32, (None,)),
    "grid_linear_id": (torch.int32, (None,)),
    "light_depth": (torch.int32, (None,)),
    "sensor_depth": (torch.int32, (None,)),
    "path_length_m": (torch.float32, (None,)),
}


def _validate_bdpt_connection_samples(
    name: str,
    exported: dict[str, torch.Tensor],
    expected_count: int | None,
) -> None:
    if not isinstance(exported, dict):
        raise TypeError(f"{name} must be a dict")
    if set(exported) != set(_BDPT_CONNECTION_SCHEMA):
        raise ValueError(f"{name} returned unexpected fields")
    inferred_count = expected_count
    for field, (dtype, shape_spec) in _BDPT_CONNECTION_SCHEMA.items():
        tensor = exported[field]
        validate_cuda_tensor(
            f"{name}.{field}", tensor, dtype=dtype, ndim=len(shape_spec)
        )
        if inferred_count is None:
            inferred_count = int(tensor.shape[0])
        expected_shape = tuple(
            inferred_count if dim is None else dim for dim in shape_spec
        )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} returned bad {field} shape")


def _bdpt_mis_mode_id(mis: str) -> int:
    if mis == "none":
        return 0
    if mis == "balance":
        return 1
    if mis == "power_heuristic":
        return 2
    raise ValueError("mis is not supported")


def bdpt_empty_subpath_state(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    exported = _required_native_op("bdpt_empty_subpath_state")(reference)
    _validate_bdpt_subpath_state(
        "_channel_native.bdpt_empty_subpath_state", exported, 0
    )
    return exported


def bdpt_endpoint_subpath_state(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_positions: torch.Tensor,
    rx_polarization: torch.Tensor,
    launch_tx_id: torch.Tensor,
    light_seed: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "tx_polarization", tx_polarization, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "rx_polarization", rx_polarization, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("launch_tx_id", launch_tx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("light_seed", light_seed, dtype=torch.int64, ndim=1)
    if tx_power.shape != (tx_positions.shape[0],):
        raise ValueError("tx_power must match tx_positions")
    if tx_polarization.shape != tx_positions.shape:
        raise ValueError("tx_polarization must match tx_positions")
    if rx_polarization.shape != rx_positions.shape:
        raise ValueError("rx_polarization must match rx_positions")
    if light_seed.shape != launch_tx_id.shape:
        raise ValueError("light_seed must match launch_tx_id")
    device = tx_positions.get_device()
    if (
        tx_power.get_device() != device
        or tx_polarization.get_device() != device
        or rx_positions.get_device() != device
        or rx_polarization.get_device() != device
        or launch_tx_id.get_device() != device
        or light_seed.get_device() != device
    ):
        raise ValueError("BDPT endpoint tensors must share one CUDA device")
    exported = _required_native_op("bdpt_endpoint_subpath_state")(
        tx_positions,
        tx_power,
        tx_polarization,
        rx_positions,
        rx_polarization,
        launch_tx_id,
        light_seed,
    )
    if not isinstance(exported, dict) or set(exported) != {"light", "sensor"}:
        raise TypeError(
            "_channel_native.bdpt_endpoint_subpath_state must return light/sensor dicts"
        )
    light = exported["light"]
    sensor = exported["sensor"]
    _validate_bdpt_subpath_state(
        "_channel_native.bdpt_endpoint_subpath_state.light",
        light,
        int(launch_tx_id.shape[0]),
    )
    _validate_bdpt_subpath_state(
        "_channel_native.bdpt_endpoint_subpath_state.sensor",
        sensor,
        int(rx_positions.shape[0]),
    )
    return {"light": light, "sensor": sensor}


def bdpt_subpath_intersection_inputs(
    subpath: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("subpath", subpath, None)
    exported = _required_native_op("bdpt_subpath_intersection_inputs")(subpath)
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_subpath_intersection_inputs must return a dict"
        )
    if set(exported) != {"ray_o", "ray_d", "ray_tmax", "active"}:
        raise ValueError(
            "_channel_native.bdpt_subpath_intersection_inputs returned unexpected fields"
        )
    validate_cuda_tensor(
        "ray_o", exported["ray_o"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "ray_d", exported["ray_d"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("ray_tmax", exported["ray_tmax"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    if (
        exported["ray_o"].shape != subpath["origin"].shape
        or exported["ray_d"].shape != subpath["direction"].shape
    ):
        raise ValueError(
            "_channel_native.bdpt_subpath_intersection_inputs returned bad ray shape"
        )
    if exported["active"].shape != subpath["valid"].shape:
        raise ValueError(
            "_channel_native.bdpt_subpath_intersection_inputs returned bad active shape"
        )
    if exported["ray_tmax"].shape != (0,):
        raise ValueError(
            "_channel_native.bdpt_subpath_intersection_inputs returned bad ray_tmax shape"
        )
    return exported


def bdpt_reflected_light_subpath_state(
    light: dict[str, torch.Tensor],
    intersection: dict[str, torch.Tensor],
    *,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_eps_r: torch.Tensor,
    material_sigma_e: torch.Tensor,
    material_mu_r: torch.Tensor,
    material_thickness: torch.Tensor,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("light", light, None)
    if not isinstance(intersection, dict) or set(intersection) != set(
        _BDPT_INTERSECTION_FIELDS
    ):
        raise ValueError("intersection returned unexpected fields")
    count = int(light["origin"].shape[0])
    validate_cuda_tensor(
        "intersection.t", intersection["t"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "intersection.p",
        intersection["p"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "intersection.n",
        intersection["n"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "intersection.global_prim_id",
        intersection["global_prim_id"],
        dtype=torch.int32,
        ndim=1,
    )
    validate_cuda_tensor("material_gain", material_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_valid", material_valid, dtype=torch.bool, ndim=1)
    validate_cuda_tensor("material_eps_r", material_eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "material_sigma_e", material_sigma_e, dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor("material_mu_r", material_mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "material_thickness", material_thickness, dtype=torch.float32, ndim=1
    )
    if int(material_gain.shape[0]) != int(material_valid.shape[0]):
        raise ValueError("material_gain and material_valid must have matching length")
    for name, tensor in (
        ("material_eps_r", material_eps_r),
        ("material_sigma_e", material_sigma_e),
        ("material_mu_r", material_mu_r),
        ("material_thickness", material_thickness),
    ):
        if int(tensor.shape[0]) != int(material_gain.shape[0]):
            raise ValueError(f"{name} must match material_gain length")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    if (
        material_gain.get_device() != light["origin"].get_device()
        or material_valid.get_device() != light["origin"].get_device()
        or material_eps_r.get_device() != light["origin"].get_device()
        or material_sigma_e.get_device() != light["origin"].get_device()
        or material_mu_r.get_device() != light["origin"].get_device()
        or material_thickness.get_device() != light["origin"].get_device()
    ):
        raise ValueError("material tensors must share light device")
    for name in ("t", "p", "n", "global_prim_id"):
        if int(intersection[name].shape[0]) != count:
            raise ValueError("intersection must match light subpath count")
        if intersection[name].get_device() != light["origin"].get_device():
            raise ValueError("intersection tensors must share light device")
    exported = _required_native_op("bdpt_reflected_light_subpath_state")(
        light,
        intersection,
        material_gain,
        material_valid,
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        material_thickness,
        float(frequency_hz),
    )
    _validate_bdpt_subpath_state(
        "_channel_native.bdpt_reflected_light_subpath_state", exported, count
    )
    return exported


def bdpt_transmitted_light_subpath_state(
    light: dict[str, torch.Tensor],
    intersection: dict[str, torch.Tensor],
    *,
    face_material_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("light", light, None)
    if not isinstance(intersection, dict) or set(intersection) != set(
        _BDPT_INTERSECTION_FIELDS
    ):
        raise ValueError("intersection returned unexpected fields")
    count = int(light["origin"].shape[0])
    validate_cuda_tensor(
        "intersection.t", intersection["t"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "intersection.p",
        intersection["p"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "intersection.n",
        intersection["n"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "intersection.global_prim_id",
        intersection["global_prim_id"],
        dtype=torch.int32,
        ndim=1,
    )
    validate_cuda_tensor(
        "face_material_id", face_material_id, dtype=torch.int32, ndim=1
    )
    device = light["origin"].get_device()
    if face_material_id.get_device() != device:
        raise ValueError("face_material_id must share light device")
    _validate_layer_csr(
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        device,
    )
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    for name in ("t", "p", "n", "global_prim_id"):
        if int(intersection[name].shape[0]) != count:
            raise ValueError("intersection must match light subpath count")
        if intersection[name].get_device() != device:
            raise ValueError("intersection tensors must share light device")
    exported = _required_native_op("bdpt_transmitted_light_subpath_state")(
        light,
        intersection,
        face_material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        float(frequency_hz),
    )
    _validate_bdpt_subpath_state(
        "_channel_native.bdpt_transmitted_light_subpath_state", exported, count
    )
    return exported


def bdpt_endpoint_connection_samples(
    light: dict[str, torch.Tensor],
    sensor: dict[str, torch.Tensor],
    *,
    frequency_hz: float,
    samples_per_tx: int,
    max_paths: int | None = None,
    mis: str = "power_heuristic",
    beta: float = 2.0,
    strategy_count: int = 1,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("light", light, None)
    _validate_bdpt_subpath_state("sensor", sensor, None)
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    if samples_per_tx <= 0:
        raise ValueError("samples_per_tx must be positive")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if strategy_count != 1:
        raise ValueError("endpoint connections support exactly one strategy")
    max_paths_value = -1 if max_paths is None else int(max_paths)
    if max_paths_value < -1:
        raise ValueError("max_paths must be non-negative")
    mode_id = _bdpt_mis_mode_id(mis)
    expected_total = int(light["origin"].shape[0]) * int(sensor["origin"].shape[0])
    expected_count = (
        expected_total if max_paths is None else min(int(max_paths), expected_total)
    )
    exported = _required_native_op("bdpt_endpoint_connection_samples")(
        light,
        sensor,
        float(frequency_hz),
        int(samples_per_tx),
        int(mode_id),
        float(beta),
        int(strategy_count),
        int(max_paths_value),
    )
    _validate_bdpt_connection_samples(
        "_channel_native.bdpt_endpoint_connection_samples", exported, expected_count
    )
    return exported


def bdpt_endpoint_connection_visibility_inputs(
    light: dict[str, torch.Tensor],
    sensor: dict[str, torch.Tensor],
    *,
    sample_count: int,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_subpath_state("light", light, None)
    _validate_bdpt_subpath_state("sensor", sensor, None)
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    expected_total = int(light["origin"].shape[0]) * int(sensor["origin"].shape[0])
    if int(sample_count) > expected_total:
        raise ValueError("sample_count exceeds endpoint pair count")
    exported = _required_native_op("bdpt_endpoint_connection_visibility_inputs")(
        light,
        sensor,
        int(sample_count),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_endpoint_connection_visibility_inputs must return a dict"
        )
    if set(exported) != {"start", "end", "active"}:
        raise ValueError(
            "_channel_native.bdpt_endpoint_connection_visibility_inputs returned unexpected fields"
        )
    validate_cuda_tensor(
        "start", exported["start"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "end", exported["end"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    if tuple(exported["start"].shape) != (int(sample_count), 3):
        raise ValueError(
            "_channel_native.bdpt_endpoint_connection_visibility_inputs returned bad start shape"
        )
    if exported["end"].shape != exported["start"].shape or exported["active"].shape != (
        int(sample_count),
    ):
        raise ValueError(
            "_channel_native.bdpt_endpoint_connection_visibility_inputs returned bad visibility shape"
        )
    return exported


def bdpt_accumulate_connection_samples(
    samples: dict[str, torch.Tensor],
    *,
    tx_count: int,
    rx_count: int,
    accumulation_strategy: str = "atomic",
) -> dict[str, torch.Tensor]:
    _validate_bdpt_connection_samples("samples", samples, None)
    if tx_count < 0 or rx_count < 0:
        raise ValueError("tx_count and rx_count must be non-negative")
    strategy_ids = {"atomic": 0, "staged": 1, "compact": 2}
    if accumulation_strategy not in strategy_ids:
        raise ValueError(
            "accumulation_strategy must be 'atomic', 'staged', or 'compact'"
        )
    exported = _required_native_op("bdpt_accumulate_connection_samples")(
        samples,
        int(tx_count),
        int(rx_count),
        int(strategy_ids[accumulation_strategy]),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_accumulate_connection_samples must return a dict"
        )
    if set(exported) != {
        "path_gain",
        "los",
        "reflection",
        "diffraction",
        "transmission",
        "scattering",
    }:
        raise ValueError(
            "_channel_native.bdpt_accumulate_connection_samples returned unexpected fields"
        )
    for name, tensor in exported.items():
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=2)
        if tuple(tensor.shape) != (int(tx_count), int(rx_count)):
            raise ValueError(
                f"_channel_native.bdpt_accumulate_connection_samples returned bad {name} shape"
            )
    return exported


def bdpt_filter_connection_samples(
    samples: dict[str, torch.Tensor],
    visible: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_connection_samples("samples", samples, None)
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    if visible.shape != samples["valid"].shape:
        raise ValueError("visible must match samples")
    if visible.get_device() != samples["valid"].get_device():
        raise ValueError("visible must share samples device")
    exported = _required_native_op("bdpt_filter_connection_samples")(samples, visible)
    _validate_bdpt_connection_samples(
        "_channel_native.bdpt_filter_connection_samples", exported, None
    )
    return exported


def bdpt_count_valid_connection_samples(samples: dict[str, torch.Tensor]) -> int:
    _validate_bdpt_connection_samples("samples", samples, None)
    count = _required_native_op("bdpt_count_valid_connection_samples")(samples)
    if not isinstance(count, int):
        raise TypeError(
            "_channel_native.bdpt_count_valid_connection_samples must return an int"
        )
    if count < 0 or count > int(samples["valid"].shape[0]):
        raise ValueError(
            "_channel_native.bdpt_count_valid_connection_samples returned bad count"
        )
    return count


def bdpt_compact_connection_samples(
    samples: dict[str, torch.Tensor],
    *,
    max_paths: int | None = None,
) -> dict[str, torch.Tensor]:
    _validate_bdpt_connection_samples("samples", samples, None)
    max_paths_value = -1 if max_paths is None else int(max_paths)
    if max_paths_value < -1:
        raise ValueError("max_paths must be non-negative")
    exported = _required_native_op("bdpt_compact_connection_samples")(
        samples, int(max_paths_value)
    )
    _validate_bdpt_connection_samples(
        "_channel_native.bdpt_compact_connection_samples", exported, None
    )
    if max_paths is not None and int(exported["valid"].shape[0]) > int(max_paths):
        raise ValueError(
            "_channel_native.bdpt_compact_connection_samples exceeded max_paths"
        )
    return exported


def bdpt_concat_connection_samples(
    samples: tuple[dict[str, torch.Tensor], ...] | list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("samples must not be empty")
    expected_count = 0
    for index, block in enumerate(samples):
        _validate_bdpt_connection_samples(f"samples[{index}]", block, None)
        expected_count += int(block["valid"].shape[0])
    exported = _required_native_op("bdpt_concat_connection_samples")(tuple(samples))
    _validate_bdpt_connection_samples(
        "_channel_native.bdpt_concat_connection_samples", exported, expected_count
    )
    return exported


def bdpt_connection_variance(
    samples: dict[str, torch.Tensor],
    *,
    tx_count: int,
    rx_count: int,
    samples_per_tx: int,
) -> torch.Tensor:
    _validate_bdpt_connection_samples("samples", samples, None)
    if tx_count < 0 or rx_count < 0:
        raise ValueError("tx_count and rx_count must be non-negative")
    if samples_per_tx <= 0:
        raise ValueError("samples_per_tx must be positive")
    variance = _required_native_op("bdpt_connection_variance")(
        samples,
        int(tx_count),
        int(rx_count),
        int(samples_per_tx),
    )
    if not isinstance(variance, torch.Tensor):
        raise TypeError("_channel_native.bdpt_connection_variance must return a tensor")
    validate_cuda_tensor("variance", variance, dtype=torch.float32, ndim=2)
    if tuple(variance.shape) != (int(tx_count), int(rx_count)):
        raise ValueError("_channel_native.bdpt_connection_variance returned bad shape")
    return variance


def deterministic_component_counts(component_id: torch.Tensor) -> dict[str, int]:
    validate_cuda_tensor("component_id", component_id, dtype=torch.int32, ndim=1)
    exported = _required_native_op("deterministic_component_counts")(component_id)
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.deterministic_component_counts must return a dict"
        )
    counts: dict[str, int] = {}
    for name in ("los", "reflection", "diffraction"):
        value = exported[name]
        if not isinstance(value, int):
            raise TypeError(
                f"_channel_native.deterministic_component_counts returned non-int {name}"
            )
        counts[name] = value
    return counts


def deterministic_selected_edge_count(edge_id: torch.Tensor) -> int:
    validate_cuda_tensor("edge_id", edge_id, dtype=torch.int32, ndim=1)
    value = _required_native_op("deterministic_selected_edge_count")(edge_id)
    if not isinstance(value, int):
        raise TypeError(
            "_channel_native.deterministic_selected_edge_count must return an int"
        )
    return value


def core_pack_int2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("x", x, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("y", y, dtype=torch.int32, ndim=1)
    if y.shape != x.shape:
        raise ValueError("x and y must have the same shape")
    out = _required_native_op("core_pack_int2")(x, y)
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel_native.core_pack_int2 must return a tensor")
    validate_cuda_tensor("out", out, dtype=torch.int32, ndim=2, trailing_shape=(2,))
    if out.shape != (x.shape[0], 2):
        raise ValueError("_channel_native.core_pack_int2 returned an unexpected shape")
    return out


def bdpt_mis_weights(
    pdf: torch.Tensor,
    strategy_pdf_sum: torch.Tensor,
    *,
    mis: str,
    beta: float = 2.0,
) -> torch.Tensor:
    validate_cuda_tensor("pdf", pdf, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "strategy_pdf_sum", strategy_pdf_sum, dtype=torch.float32, ndim=0
    )
    mode_id = _bdpt_mis_mode_id(mis)
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    native = native_extension()
    if native is None or not hasattr(native, "bdpt_mis_weights"):
        raise RuntimeError("_channel_native.bdpt_mis_weights CUDA kernel is required")
    weights = native.bdpt_mis_weights(pdf, strategy_pdf_sum, int(mode_id), float(beta))
    if not isinstance(weights, torch.Tensor):
        raise TypeError("_channel_native.bdpt_mis_weights must return a tensor")
    validate_cuda_tensor("weights", weights, dtype=torch.float32, ndim=1)
    if weights.shape != pdf.shape:
        raise ValueError(
            "_channel_native.bdpt_mis_weights returned an unexpected shape"
        )
    return weights


def bdpt_diffraction_connection_samples_from_tape(
    tape: dict[str, torch.Tensor],
    states: tuple[torch.Tensor, ...],
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    *,
    tx_index: int,
    state_count: int,
    grid_axis: int,
    grid_position: float,
    grid_coord0_min: float,
    grid_coord0_max: float,
    grid_coord1_min: float,
    grid_coord1_max: float,
    grid_resolution0: int,
    grid_resolution1: int,
    grid_cell_area: float,
    wavelength: float,
    direct_samples: int,
    keller_samples: int,
    mis: str = "power_heuristic",
    beta: float = 2.0,
    strategy_count: int = 1,
) -> dict[str, torch.Tensor]:
    expected_tape = {
        "active": torch.bool,
        "state_idx": torch.int32,
        "cell": torch.int32,
        "material_idx": torch.int32,
        "edge_u": torch.float32,
    }
    if set(tape) != set(expected_tape):
        raise ValueError("diffraction tape returned unexpected fields")
    inferred_count: int | None = None
    for field, dtype in expected_tape.items():
        tensor = tape[field]
        validate_cuda_tensor(f"tape.{field}", tensor, dtype=dtype, ndim=1)
        if inferred_count is None:
            inferred_count = int(tensor.shape[0])
        if int(tensor.shape[0]) != inferred_count:
            raise ValueError("diffraction tape fields must share shape")
    if len(states) != 12:
        raise ValueError("states must contain 12 tensors")
    validate_cuda_tensor("material_gain", material_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_valid", material_valid, dtype=torch.bool, ndim=1)
    if material_gain.shape != material_valid.shape:
        raise ValueError("material_gain and material_valid must have matching shape")
    if state_count < 0:
        raise ValueError("state_count must be non-negative")
    if direct_samples < 0 or keller_samples < 0:
        raise ValueError("sample counts must be non-negative")
    if strategy_count <= 0:
        raise ValueError("strategy_count must be positive")
    actual_strategy_count = int(direct_samples > 0) + int(keller_samples > 0)
    if strategy_count != actual_strategy_count:
        raise ValueError("strategy_count must match enabled direct/Keller proposals")
    if mis == "none" and actual_strategy_count != 1:
        raise ValueError("mis='none' requires exactly one diffraction proposal")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    mode_id = _bdpt_mis_mode_id(mis)
    exported = _required_native_op("bdpt_diffraction_connection_samples_from_tape")(
        tape,
        tuple(states),
        material_gain,
        material_valid,
        int(tx_index),
        int(state_count),
        int(grid_axis),
        float(grid_position),
        float(grid_coord0_min),
        float(grid_coord0_max),
        float(grid_coord1_min),
        float(grid_coord1_max),
        int(grid_resolution0),
        int(grid_resolution1),
        float(grid_cell_area),
        float(wavelength),
        int(direct_samples),
        int(keller_samples),
        int(mode_id),
        float(beta),
        int(strategy_count),
    )
    _validate_bdpt_connection_samples(
        "_channel_native.bdpt_diffraction_connection_samples_from_tape",
        exported,
        inferred_count,
    )
    return exported


def bdpt_diffraction_point_connection_samples(
    rx_positions: torch.Tensor,
    states: tuple[torch.Tensor, ...],
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    *,
    tx_index: int,
    state_count: int,
    direct_samples: int,
    keller_samples: int,
    seed: int,
    wavelength: float,
    mis: str = "power_heuristic",
    beta: float = 2.0,
    strategy_count: int = 1,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if len(states) != 12:
        raise ValueError("states must contain 12 tensors")
    validate_cuda_tensor("material_gain", material_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_valid", material_valid, dtype=torch.bool, ndim=1)
    if material_gain.shape != material_valid.shape:
        raise ValueError("material_gain and material_valid must have matching shape")
    if state_count < 0:
        raise ValueError("state_count must be non-negative")
    if direct_samples < 0 or keller_samples < 0:
        raise ValueError("sample counts must be non-negative")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if wavelength <= 0.0:
        raise ValueError("wavelength must be positive")
    if strategy_count <= 0:
        raise ValueError("strategy_count must be positive")
    actual_strategy_count = int(direct_samples > 0) + int(keller_samples > 0)
    if strategy_count != actual_strategy_count:
        raise ValueError("strategy_count must match enabled direct/Keller proposals")
    if mis == "none" and actual_strategy_count != 1:
        raise ValueError("mis='none' requires exactly one diffraction proposal")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    mode_id = _bdpt_mis_mode_id(mis)
    exported = _required_native_op("bdpt_diffraction_point_connection_samples")(
        rx_positions,
        tuple(states),
        material_gain,
        material_valid,
        int(tx_index),
        int(state_count),
        int(direct_samples),
        int(keller_samples),
        int(seed),
        float(wavelength),
        int(mode_id),
        float(beta),
        int(strategy_count),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_diffraction_point_connection_samples must return a dict"
        )
    expected = {
        "samples",
        "source_start",
        "source_end",
        "target_start",
        "target_end",
        "visibility_active",
    }
    if set(exported) != expected:
        raise ValueError(
            "_channel_native.bdpt_diffraction_point_connection_samples returned unexpected fields"
        )
    sample_count = int(rx_positions.shape[0]) * (
        int(direct_samples) + int(keller_samples)
    )
    if state_count == 0:
        sample_count = 0
    samples = exported["samples"]
    if not isinstance(samples, dict):
        raise TypeError(
            "_channel_native.bdpt_diffraction_point_connection_samples samples must be a dict"
        )
    _validate_bdpt_connection_samples(
        "_channel_native.bdpt_diffraction_point_connection_samples.samples",
        samples,
        sample_count,
    )
    for name in ("source_start", "source_end", "target_start", "target_end"):
        tensor = exported[name]
        validate_cuda_tensor(
            name, tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
        if tuple(tensor.shape) != (sample_count, 3):
            raise ValueError(
                f"_channel_native.bdpt_diffraction_point_connection_samples returned bad {name} shape"
            )
    active = exported["visibility_active"]
    validate_cuda_tensor("visibility_active", active, dtype=torch.bool, ndim=1)
    if tuple(active.shape) != (sample_count,):
        raise ValueError(
            "_channel_native.bdpt_diffraction_point_connection_samples returned bad visibility_active shape"
        )
    return exported


def bdpt_zero_matrix(reference: torch.Tensor, *, rows: int, cols: int) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    if rows < 0 or cols < 0:
        raise ValueError("rows and cols must be non-negative")
    native = native_extension()
    if native is None or not hasattr(native, "bdpt_zero_matrix"):
        raise RuntimeError("_channel_native.bdpt_zero_matrix CUDA kernel is required")
    out = native.bdpt_zero_matrix(reference, int(rows), int(cols))
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel_native.bdpt_zero_matrix must return a tensor")
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2)
    if out.shape != (int(rows), int(cols)):
        raise ValueError(
            "_channel_native.bdpt_zero_matrix returned an unexpected shape"
        )
    return out


def bdpt_store_point_component_column(
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    rx_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("target", target, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("source", source, dtype=torch.float32, ndim=3)
    if source.shape[0] != target.shape[0] or source.shape[1:] != (1, 1):
        raise ValueError("source must have shape (tx, 1, 1)")
    if rx_index < 0 or rx_index >= target.shape[1]:
        raise ValueError("rx_index is out of range")
    native = _required_native_op("bdpt_store_point_component_column")
    out = native(target, source, int(rx_index))
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.bdpt_store_point_component_column must return a tensor"
        )
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2)
    if out.shape != target.shape:
        raise ValueError(
            "_channel_native.bdpt_store_point_component_column returned an unexpected shape"
        )
    return out


def bdpt_finalize_point_components(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("reflection", reflection, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("diffraction", diffraction, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("transmission", transmission, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("scattering", scattering, dtype=torch.float32, ndim=2)
    if (
        reflection.shape != los.shape
        or diffraction.shape != los.shape
        or transmission.shape != los.shape
        or scattering.shape != los.shape
    ):
        raise ValueError("point component matrices must have matching shapes")
    exported = _required_native_op("bdpt_finalize_point_components")(
        los, reflection, diffraction, transmission, scattering
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_finalize_point_components must return a dict"
        )
    validate_cuda_tensor(
        "path_gain", exported["path_gain"], dtype=torch.float32, ndim=2
    )
    if exported["path_gain"].shape != los.shape:
        raise ValueError(
            "_channel_native.bdpt_finalize_point_components returned bad path_gain shape"
        )
    for name in (
        "los_power",
        "reflection_power",
        "diffraction_power",
        "transmission_power",
        "scattering_power",
    ):
        validate_cuda_tensor(name, exported[name], dtype=torch.float32, ndim=0)
    return exported


def bdpt_point_component_power(
    path_gain: torch.Tensor, *, include_los: bool
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=2)
    native = native_extension()
    if native is None or not hasattr(native, "bdpt_point_component_power"):
        raise RuntimeError(
            "_channel_native.bdpt_point_component_power CUDA kernel is required"
        )
    exported = native.bdpt_point_component_power(path_gain, bool(include_los))
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.bdpt_point_component_power must return a dict")
    for name in ("los", "reflection", "diffraction"):
        validate_cuda_tensor(name, exported[name], dtype=torch.float32, ndim=0)
    return exported


def bdpt_transmitter_tensors(
    flat_positions: tuple[float, ...],
    powers: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    if len(flat_positions) % 3 != 0:
        raise ValueError("flat_positions must contain xyz triples")
    if len(flat_positions) // 3 != len(powers):
        raise ValueError("powers must match flat_positions")
    exported = _required_native_op("bdpt_transmitter_tensors")(flat_positions, powers)
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.bdpt_transmitter_tensors must return a dict")
    validate_cuda_tensor(
        "positions",
        exported["positions"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor("power", exported["power"], dtype=torch.float32, ndim=1)
    return exported


def bdpt_host_vec3_tensor(flat_positions: tuple[float, ...]) -> torch.Tensor:
    if len(flat_positions) % 3 != 0:
        raise ValueError("flat_positions must contain xyz triples")
    powers = tuple(1.0 for _ in range(len(flat_positions) // 3))
    return bdpt_transmitter_tensors(flat_positions, powers)["positions"]


def bdpt_receiver_grid_points(
    reference: torch.Tensor,
    *,
    origin: tuple[float, float, float],
    x_axis: tuple[float, float, float],
    y_axis: tuple[float, float, float],
    shape: tuple[int, int],
    spacing: tuple[float, float],
) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    rows, cols = shape
    if rows < 0 or cols < 0:
        raise ValueError("shape entries must be non-negative")
    if spacing[0] <= 0.0 or spacing[1] <= 0.0:
        raise ValueError("spacing entries must be positive")
    points = _required_native_op("bdpt_receiver_grid_points")(
        reference,
        int(rows),
        int(cols),
        float(origin[0]),
        float(origin[1]),
        float(origin[2]),
        float(x_axis[0]),
        float(x_axis[1]),
        float(x_axis[2]),
        float(y_axis[0]),
        float(y_axis[1]),
        float(y_axis[2]),
        float(spacing[0]),
        float(spacing[1]),
    )
    if not isinstance(points, torch.Tensor):
        raise TypeError(
            "_channel_native.bdpt_receiver_grid_points must return a tensor"
        )
    validate_cuda_tensor(
        "points", points, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if points.shape[0] != rows * cols:
        raise ValueError(
            "_channel_native.bdpt_receiver_grid_points returned an unexpected shape"
        )
    return points


def bdpt_los_export(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if tx_power.shape[0] != tx_positions.shape[0]:
        raise ValueError("tx_power must have one value per transmitter")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    exported = _required_native_op("bdpt_los_export")(
        tx_positions,
        tx_power,
        rx_positions,
        float(frequency_hz),
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.bdpt_los_export must return a dict")
    validate_cuda_tensor(
        "path_gain_matrix", exported["path_gain_matrix"], dtype=torch.float32, ndim=2
    )
    return exported


def bdpt_los_component_maps(los: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=3)
    maps = _required_native_op("bdpt_los_component_maps")(los)
    if not isinstance(maps, torch.Tensor):
        raise TypeError("_channel_native.bdpt_los_component_maps must return a tensor")
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    if maps.shape != los.shape:
        raise ValueError(
            "_channel_native.bdpt_los_component_maps returned an unexpected shape"
        )
    return maps


def bdpt_los_component_maps_from_matrix(
    los: torch.Tensor, *, rows: int, cols: int
) -> torch.Tensor:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=2)
    if rows < 0 or cols < 0:
        raise ValueError("rows and cols must be non-negative")
    if los.shape[1] != int(rows) * int(cols):
        raise ValueError("los columns must match rows * cols")
    maps = _required_native_op("bdpt_los_component_maps_from_matrix")(
        los, int(rows), int(cols)
    )
    if not isinstance(maps, torch.Tensor):
        raise TypeError(
            "_channel_native.bdpt_los_component_maps_from_matrix must return a tensor"
        )
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    if maps.shape != (los.shape[0], int(cols), int(rows)):
        raise ValueError(
            "_channel_native.bdpt_los_component_maps_from_matrix returned an unexpected shape"
        )
    return maps


def bdpt_los_visibility_inputs(
    tx_positions: torch.Tensor,
    *,
    tx_index: int,
    rx_count: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if rx_count < 0:
        raise ValueError("rx_count must be non-negative")
    exported = _required_native_op("bdpt_los_visibility_inputs")(
        tx_positions, int(tx_index), int(rx_count)
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.bdpt_los_visibility_inputs must return a dict")
    validate_cuda_tensor(
        "start", exported["start"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    return exported


def bdpt_apply_los_visibility(
    maps: torch.Tensor,
    los: torch.Tensor,
    visible: torch.Tensor,
    *,
    tx_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    if maps.shape[0] != los.shape[0] or los.shape[1] != maps.shape[1] * maps.shape[2]:
        raise ValueError("los must match maps flattened grid layout")
    out = _required_native_op("bdpt_apply_los_visibility")(
        maps, los, visible, int(tx_index)
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.bdpt_apply_los_visibility must return a tensor"
        )
    validate_cuda_tensor("maps", out, dtype=torch.float32, ndim=3)
    return out


def bdpt_component_map_buffer(
    reference: torch.Tensor,
    *,
    tx_count: int,
    dim0: int,
    dim1: int,
) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    if tx_count < 0 or dim0 < 0 or dim1 < 0:
        raise ValueError("tx_count, dim0, and dim1 must be non-negative")
    maps = _required_native_op("bdpt_component_map_buffer")(
        reference, int(tx_count), int(dim0), int(dim1)
    )
    if not isinstance(maps, torch.Tensor):
        raise TypeError(
            "_channel_native.bdpt_component_map_buffer must return a tensor"
        )
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    if maps.shape != (tx_count, dim0, dim1):
        raise ValueError(
            "_channel_native.bdpt_component_map_buffer returned an unexpected shape"
        )
    return maps


def bdpt_store_component_map(
    maps: torch.Tensor,
    source: torch.Tensor,
    *,
    tx_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("source", source, dtype=torch.float32, ndim=2)
    out = _required_native_op("bdpt_store_component_map")(maps, source, int(tx_index))
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel_native.bdpt_store_component_map must return a tensor")
    validate_cuda_tensor("maps", out, dtype=torch.float32, ndim=3)
    return out


def bdpt_store_scaled_component_map(
    maps: torch.Tensor,
    source: torch.Tensor,
    scale_values: torch.Tensor,
    *,
    tx_index: int,
    scale_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("source", source, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("scale_values", scale_values, dtype=torch.float32, ndim=1)
    out = _required_native_op("bdpt_store_scaled_component_map")(
        maps,
        source,
        scale_values,
        int(tx_index),
        int(scale_index),
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.bdpt_store_scaled_component_map must return a tensor"
        )
    validate_cuda_tensor("maps", out, dtype=torch.float32, ndim=3)
    return out


def bdpt_finalize_component_maps(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("reflection", reflection, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("diffraction", diffraction, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("transmission", transmission, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("scattering", scattering, dtype=torch.float32, ndim=3)
    if (
        reflection.shape != los.shape
        or diffraction.shape != los.shape
        or transmission.shape != los.shape
        or scattering.shape != los.shape
    ):
        raise ValueError("component maps must share shape")
    exported = _required_native_op("bdpt_finalize_component_maps")(
        los, reflection, diffraction, transmission, scattering
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_finalize_component_maps must return a dict"
        )
    validate_cuda_tensor(
        "path_gain", exported["path_gain"], dtype=torch.float32, ndim=3
    )
    if exported["path_gain"].shape != los.shape:
        raise ValueError(
            "_channel_native.bdpt_finalize_component_maps returned bad path_gain shape"
        )
    for name in (
        "los_power",
        "reflection_power",
        "diffraction_power",
        "transmission_power",
        "scattering_power",
    ):
        validate_cuda_tensor(name, exported[name], dtype=torch.float32, ndim=0)
    return exported


def bdpt_sample_directions(
    count: int, reference: torch.Tensor, *, seed: int
) -> torch.Tensor:
    if count < 0:
        raise ValueError("count must be non-negative")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    directions = _required_native_op("bdpt_sample_directions")(
        int(count), reference, int(seed)
    )
    if not isinstance(directions, torch.Tensor):
        raise TypeError("_channel_native.bdpt_sample_directions must return a tensor")
    validate_cuda_tensor(
        "directions", directions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    return directions


def bdpt_reflection_launch_inputs(
    tx_positions: torch.Tensor,
    *,
    tx_index: int,
    sample_count: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    exported = _required_native_op("bdpt_reflection_launch_inputs")(
        tx_positions, int(tx_index), int(sample_count)
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_reflection_launch_inputs must return a dict"
        )
    validate_cuda_tensor(
        "ray_o", exported["ray_o"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("ray_tmax", exported["ray_tmax"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    validate_cuda_tensor(
        "tx_pol", exported["tx_pol"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_id", exported["tx_id"], dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "light_seed", exported["light_seed"], dtype=torch.int64, ndim=1
    )
    return exported


def bdpt_diffraction_state_wi(
    state_edge_pos: torch.Tensor, state_src: torch.Tensor
) -> torch.Tensor:
    validate_cuda_tensor(
        "state_edge_pos",
        state_edge_pos,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "state_src", state_src, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    state_wi = _required_native_op("bdpt_diffraction_state_wi")(
        state_edge_pos, state_src
    )
    if not isinstance(state_wi, torch.Tensor):
        raise TypeError(
            "_channel_native.bdpt_diffraction_state_wi must return a tensor"
        )
    validate_cuda_tensor(
        "state_wi", state_wi, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    return state_wi


def bdpt_selected_edge_indices(selected: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("selected", selected, dtype=torch.bool, ndim=1)
    indices = _required_native_op("bdpt_selected_edge_indices")(selected)
    if not isinstance(indices, torch.Tensor):
        raise TypeError(
            "_channel_native.bdpt_selected_edge_indices must return a tensor"
        )
    validate_cuda_tensor("indices", indices, dtype=torch.int32, ndim=1)
    return indices


def bdpt_diffraction_state_pack(
    edge_indices: torch.Tensor,
    edge_pos: torch.Tensor,
    edge_dir: torch.Tensor,
    line_min: torch.Tensor,
    line_max: torch.Tensor,
    n0: torch.Tensor,
    n1: torch.Tensor,
    face0: torch.Tensor,
    face1: torch.Tensor,
    exterior_angle: torch.Tensor,
    tx: torch.Tensor,
    tx_power: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    validate_cuda_tensor("edge_indices", edge_indices, dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "edge_pos", edge_pos, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "edge_dir", edge_dir, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("line_min", line_min, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("line_max", line_max, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("n0", n0, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("n1", n1, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("face0", face0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face1", face1, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("exterior_angle", exterior_angle, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("tx", tx, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=0)
    states = _required_native_op("bdpt_diffraction_state_pack")(
        edge_indices,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
    )
    if not isinstance(states, tuple) or len(states) != 12:
        raise TypeError(
            "_channel_native.bdpt_diffraction_state_pack must return 12 tensors"
        )
    validate_cuda_tensor("state_edge_index", states[0], dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "state_edge_pos", states[1], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "state_edge_dir", states[2], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_line_min", states[3], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("state_line_max", states[4], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "state_n0", states[5], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "state_n1", states[6], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_face0", states[7], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_face1", states[8], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_exterior_angle", states[9], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "state_src", states[10], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_tx_power", states[11], dtype=torch.float32, ndim=1)
    return states


def deterministic_diffraction_state_pack(
    edge_indices: torch.Tensor,
    edge_pos: torch.Tensor,
    edge_dir: torch.Tensor,
    line_min: torch.Tensor,
    line_max: torch.Tensor,
    n0: torch.Tensor,
    n1: torch.Tensor,
    face0: torch.Tensor,
    face1: torch.Tensor,
    exterior_angle: torch.Tensor,
    tx: torch.Tensor,
    tx_power: torch.Tensor,
    tx_power_index: int,
) -> tuple[torch.Tensor, ...]:
    validate_cuda_tensor("edge_indices", edge_indices, dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "edge_pos", edge_pos, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "edge_dir", edge_dir, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("line_min", line_min, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("line_max", line_max, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("n0", n0, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("n1", n1, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("face0", face0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face1", face1, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("exterior_angle", exterior_angle, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("tx", tx, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    if not 0 <= int(tx_power_index) < int(tx_power.shape[0]):
        raise ValueError("tx_power_index is out of range")
    states = _required_native_op("deterministic_diffraction_state_pack")(
        edge_indices,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
        int(tx_power_index),
    )
    if not isinstance(states, tuple) or len(states) != 12:
        raise TypeError(
            "_channel_native.deterministic_diffraction_state_pack must return 12 tensors"
        )
    validate_cuda_tensor("state_edge_index", states[0], dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "state_edge_pos", states[1], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "state_edge_dir", states[2], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_line_min", states[3], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("state_line_max", states[4], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "state_n0", states[5], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "state_n1", states[6], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_face0", states[7], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_face1", states[8], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_exterior_angle", states[9], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "state_src", states[10], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_tx_power", states[11], dtype=torch.float32, ndim=1)
    return states


def deterministic_diffraction_state_pack_selected(
    selected: torch.Tensor,
    edge_pos: torch.Tensor,
    edge_dir: torch.Tensor,
    line_min: torch.Tensor,
    line_max: torch.Tensor,
    n0: torch.Tensor,
    n1: torch.Tensor,
    face0: torch.Tensor,
    face1: torch.Tensor,
    exterior_angle: torch.Tensor,
    tx: torch.Tensor,
    tx_power: torch.Tensor,
    tx_power_index: int,
) -> tuple[torch.Tensor, ...]:
    validate_cuda_tensor("selected", selected, dtype=torch.bool, ndim=1)
    validate_cuda_tensor(
        "edge_pos", edge_pos, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "edge_dir", edge_dir, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("line_min", line_min, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("line_max", line_max, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("n0", n0, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("n1", n1, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("face0", face0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face1", face1, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("exterior_angle", exterior_angle, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("tx", tx, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    if not 0 <= int(tx_power_index) < int(tx_power.shape[0]):
        raise ValueError("tx_power_index is out of range")
    states = _required_native_op("deterministic_diffraction_state_pack_selected")(
        selected,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
        int(tx_power_index),
    )
    if not isinstance(states, tuple) or len(states) != 12:
        raise TypeError(
            "_channel_native.deterministic_diffraction_state_pack_selected must return 12 tensors"
        )
    validate_cuda_tensor("state_edge_index", states[0], dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "state_edge_pos", states[1], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "state_edge_dir", states[2], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_line_min", states[3], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("state_line_max", states[4], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "state_n0", states[5], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "state_n1", states[6], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_face0", states[7], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_face1", states[8], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_exterior_angle", states[9], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "state_src", states[10], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_tx_power", states[11], dtype=torch.float32, ndim=1)
    return states


def bdpt_pack_vec3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("x", x, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("y", y, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("z", z, dtype=torch.float32, ndim=1)
    packed = _required_native_op("bdpt_pack_vec3")(x, y, z)
    if not isinstance(packed, torch.Tensor):
        raise TypeError("_channel_native.bdpt_pack_vec3 must return a tensor")
    validate_cuda_tensor(
        "packed", packed, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    return packed


def bdpt_face_material_tensors(
    material_eps_r: torch.Tensor,
    material_sigma_e: torch.Tensor,
    material_mu_r: torch.Tensor,
    face_material_id: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("material_eps_r", material_eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "material_sigma_e", material_sigma_e, dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor("material_mu_r", material_mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "face_material_id", face_material_id, dtype=torch.int32, ndim=1
    )
    exported = _required_native_op("bdpt_face_material_tensors")(
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        face_material_id,
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.bdpt_face_material_tensors must return a dict")
    validate_cuda_tensor("eps_r", exported["eps_r"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("sigma_e", exported["sigma_e"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("mu_r", exported["mu_r"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("gain", exported["gain"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("valid", exported["valid"], dtype=torch.bool, ndim=1)
    return exported


def bdpt_face_material_tensors_from_host(
    material_eps_r: tuple[float, ...],
    material_sigma_e: tuple[float, ...],
    material_mu_r: tuple[float, ...],
    face_material_id: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    if not material_eps_r:
        raise ValueError("material_eps_r must not be empty")
    if len(material_sigma_e) != len(material_eps_r):
        raise ValueError("material_sigma_e must match material_eps_r")
    if len(material_mu_r) != len(material_eps_r):
        raise ValueError("material_mu_r must match material_eps_r")
    if any(index < 0 or index >= len(material_eps_r) for index in face_material_id):
        raise ValueError("face_material_id entries must reference a material")
    exported = _required_native_op("bdpt_face_material_tensors_from_host")(
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        face_material_id,
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_face_material_tensors_from_host must return a dict"
        )
    validate_cuda_tensor("eps_r", exported["eps_r"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("sigma_e", exported["sigma_e"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("mu_r", exported["mu_r"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("gain", exported["gain"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("valid", exported["valid"], dtype=torch.bool, ndim=1)
    return exported


# ---------------------------------------------------------------------------
# RayDN differentiable geometry (AD-A0).
#
# These entry points expose RayD's fixed-winner differentiable geometry
# kernels across the source-linked C bridge (no rayd.torch import, no torch
# dispatcher). Discrete winner records (prim ids, barycentrics, hit points,
# normals) are detached tape constants; gradients flow only through the
# continuous outputs (t, p, n, image sources, EPC field, path length) with
# respect to mesh vertices, ray origin/direction, and EPC source/receiver.
# They are opt-in AD entry points: no public solver calls them and the
# ad_mode="none" forward paths are untouched.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RayDN reflection EPC paths geometry AD (plan 07 AD-2 layer 1).
#
# The reflection EPC path export (direct-plane mode) is the discovery entry
# the deterministic/path solvers use. Its geometry companions differentiate
# the continuous specular chain (hit points, emitted unit normals, path
# length) with respect to the scene vertex table and the per-row endpoints,
# under a frozen winner: the face sequence, the validity mask, the
# containment resolution and the visibility casts are detached discovery
# records. RayD chains each bounce's plane cotangents to the winner
# triangle's vertices itself, so no hit geometry is ever re-derived here.
# ---------------------------------------------------------------------------


def field_free_space(
    source: torch.Tensor,
    target: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    for name, value in (
        ("source", source),
        ("target", target),
        ("tx_polarization", tx_polarization),
        ("rx_polarization", rx_polarization),
    ):
        validate_cuda_tensor(
            name, value, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    count = int(source.shape[0])
    if any(
        int(value.shape[0]) != count
        for value in (target, tx_power, tx_polarization, rx_polarization)
    ):
        raise ValueError("free-space field tensors must have matching rows")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    out = _required_native_op("field_free_space")(
        source,
        target,
        tx_power,
        tx_polarization,
        rx_polarization,
        float(frequency_hz),
    )
    if not isinstance(out, dict):
        raise TypeError("_channel_native.field_free_space must return a dict")
    schema = {
        "field_vector": (torch.complex64, 2, (count, 3)),
        "coefficient": (torch.complex64, 1, (count,)),
        "path_field": (torch.complex64, 1, (count,)),
        "path_gain": (torch.float32, 1, (count,)),
        "path_length_m": (torch.float32, 1, (count,)),
        "delay_s": (torch.float32, 1, (count,)),
        "direction": (torch.float32, 2, (count, 3)),
    }
    if set(out) != set(schema):
        raise ValueError("_channel_native.field_free_space returned unexpected fields")
    for name, (dtype, ndim, shape) in schema.items():
        validate_cuda_tensor(name, out[name], dtype=dtype, ndim=ndim)
        if tuple(out[name].shape) != shape:
            raise ValueError(f"_channel_native.field_free_space returned bad {name} shape")
    return out


def field_project_complex3(
    field_vector: torch.Tensor,
    direction: torch.Tensor,
    rx_polarization: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("field_vector", field_vector, dtype=torch.complex64, ndim=2)
    validate_cuda_tensor(
        "direction", direction, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "rx_polarization",
        rx_polarization,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    count = int(field_vector.shape[0])
    if field_vector.shape != (count, 3):
        raise ValueError("field_vector must have shape (N, 3)")
    if direction.shape != (count, 3) or rx_polarization.shape != (count, 3):
        raise ValueError("direction and rx_polarization must match field_vector rows")
    out = _required_native_op("field_project_complex3")(
        field_vector, direction, rx_polarization
    )
    if not isinstance(out, dict) or set(out) != {"coefficient", "path_gain"}:
        raise TypeError("_channel_native.field_project_complex3 returned invalid fields")
    validate_cuda_tensor("coefficient", out["coefficient"], dtype=torch.complex64, ndim=1)
    validate_cuda_tensor("path_gain", out["path_gain"], dtype=torch.float32, ndim=1)
    if out["coefficient"].shape != (count,) or out["path_gain"].shape != (count,):
        raise ValueError("field projection returned invalid shapes")
    return out


def field_reflection_sequence(
    source: torch.Tensor,
    target: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    thickness: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    for name, value in (
        ("source", source),
        ("target", target),
        ("tx_polarization", tx_polarization),
        ("rx_polarization", rx_polarization),
    ):
        validate_cuda_tensor(
            name, value, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
    for name, value in (
        ("interaction_positions", interaction_positions),
        ("interaction_normals", interaction_normals),
    ):
        validate_cuda_tensor(
            name, value, dtype=torch.float32, ndim=3, trailing_shape=(3,)
        )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    count = int(source.shape[0])
    depth = int(interaction_positions.shape[1])
    if interaction_positions.shape != (count, depth, 3) or depth <= 0:
        raise ValueError("interaction_positions must have shape (N, D, 3), D > 0")
    if interaction_normals.shape != interaction_positions.shape:
        raise ValueError("interaction_normals must match interaction_positions")
    for name, value in (
        ("eps_r", eps_r),
        ("sigma_e", sigma_e),
        ("mu_r", mu_r),
        ("gain", gain),
        ("thickness", thickness),
    ):
        validate_cuda_tensor(name, value, dtype=torch.float32, ndim=2)
        if value.shape != (count, depth):
            raise ValueError(f"{name} must have shape (N, D)")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    out = _required_native_op("field_reflection_sequence")(
        source,
        target,
        interaction_positions,
        interaction_normals,
        tx_power,
        tx_polarization,
        rx_polarization,
        eps_r,
        sigma_e,
        mu_r,
        gain,
        thickness,
        float(frequency_hz),
    )
    if not isinstance(out, dict):
        raise TypeError("_channel_native.field_reflection_sequence must return a dict")
    schema = {
        "field_vector": (torch.complex64, 2, (count, 3)),
        "coefficient": (torch.complex64, 1, (count,)),
        "path_field": (torch.complex64, 1, (count,)),
        "path_gain": (torch.float32, 1, (count,)),
        "path_length_m": (torch.float32, 1, (count,)),
        "delay_s": (torch.float32, 1, (count,)),
        "direction": (torch.float32, 2, (count, 3)),
    }
    if set(out) != set(schema):
        raise ValueError("field_reflection_sequence returned unexpected fields")
    for name, (dtype, ndim, shape) in schema.items():
        validate_cuda_tensor(name, out[name], dtype=dtype, ndim=ndim)
        if tuple(out[name].shape) != shape:
            raise ValueError(f"field_reflection_sequence returned bad {name} shape")
    return out


def field_transmission_sequence(
    source: torch.Tensor,
    target: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    interaction_material_id: torch.Tensor,
    interaction_valid: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    for name, value in (
        ("source", source),
        ("target", target),
        ("tx_polarization", tx_polarization),
        ("rx_polarization", rx_polarization),
    ):
        validate_cuda_tensor(
            name, value, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
    for name, value in (
        ("interaction_positions", interaction_positions),
        ("interaction_normals", interaction_normals),
    ):
        validate_cuda_tensor(
            name, value, dtype=torch.float32, ndim=3, trailing_shape=(3,)
        )
    validate_cuda_tensor(
        "interaction_material_id", interaction_material_id, dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "interaction_valid", interaction_valid, dtype=torch.bool, ndim=2
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    count = int(source.shape[0])
    depth = int(interaction_positions.shape[1])
    if interaction_positions.shape != (count, depth, 3) or depth <= 0:
        raise ValueError("interaction_positions must have shape (N, D, 3), D > 0")
    if interaction_normals.shape != interaction_positions.shape:
        raise ValueError("interaction_normals must match interaction_positions")
    if interaction_material_id.shape != (count, depth):
        raise ValueError("interaction_material_id must have shape (N, D)")
    if interaction_valid.shape != (count, depth):
        raise ValueError("interaction_valid must have shape (N, D)")
    if any(
        int(value.shape[0]) != count
        for value in (target, tx_power, tx_polarization, rx_polarization)
    ):
        raise ValueError("transmission endpoint tensors must have matching rows")
    _validate_layer_csr(
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        source.get_device(),
    )
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    out = _required_native_op("field_transmission_sequence")(
        source,
        target,
        interaction_positions,
        interaction_normals,
        interaction_material_id,
        interaction_valid,
        tx_power,
        tx_polarization,
        rx_polarization,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        float(frequency_hz),
    )
    if not isinstance(out, dict):
        raise TypeError("_channel_native.field_transmission_sequence must return a dict")
    schema = {
        "field_vector": (torch.complex64, 2, (count, 3)),
        "coefficient": (torch.complex64, 1, (count,)),
        "path_field": (torch.complex64, 1, (count,)),
        "path_gain": (torch.float32, 1, (count,)),
        "path_length_m": (torch.float32, 1, (count,)),
        "delay_s": (torch.float32, 1, (count,)),
        "direction": (torch.float32, 2, (count, 3)),
    }
    if set(out) != set(schema):
        raise ValueError("field_transmission_sequence returned unexpected fields")
    for name, (dtype, ndim, shape) in schema.items():
        validate_cuda_tensor(name, out[name], dtype=dtype, ndim=ndim)
        if tuple(out[name].shape) != shape:
            raise ValueError(f"field_transmission_sequence returned bad {name} shape")
    return out


_EM_LAYER_STACK_FIELDS = (
    "r_te_real",
    "r_te_imag",
    "r_tm_real",
    "r_tm_imag",
    "t_te_real",
    "t_te_imag",
    "t_tm_real",
    "t_tm_imag",
    "cap_R_te",
    "cap_R_tm",
    "cap_T_te",
    "cap_T_tm",
)


def em_layer_stack_eval(
    cos_theta: torch.Tensor,
    material_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("cos_theta", cos_theta, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_id", material_id, dtype=torch.int32, ndim=1)
    count = int(cos_theta.shape[0])
    if material_id.shape != (count,):
        raise ValueError("material_id must match cos_theta length")
    if material_id.get_device() != cos_theta.get_device():
        raise ValueError("material_id must share cos_theta device")
    _validate_layer_csr(
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        cos_theta.get_device(),
    )
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    out = _required_native_op("em_layer_stack_eval")(
        cos_theta,
        material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        float(frequency_hz),
    )
    if not isinstance(out, dict):
        raise TypeError("_channel_native.em_layer_stack_eval must return a dict")
    if set(out) != set(_EM_LAYER_STACK_FIELDS):
        raise ValueError("em_layer_stack_eval returned unexpected fields")
    for name in _EM_LAYER_STACK_FIELDS:
        validate_cuda_tensor(name, out[name], dtype=torch.float32, ndim=1)
        if out[name].shape != (count,):
            raise ValueError(f"em_layer_stack_eval returned bad {name} shape")
    return out


def em_layer_stack_backward(
    cos_theta: torch.Tensor,
    material_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    grad_outputs: tuple[torch.Tensor | None, ...],
    *,
    frequency_hz: float,
    need_cos_theta: bool,
    need_layers: bool,
    need_frequency: bool,
) -> dict[str, torch.Tensor]:
    if len(grad_outputs) != len(_EM_LAYER_STACK_FIELDS):
        raise ValueError(
            "grad_outputs must carry one cotangent slot per stack output"
        )
    out = _required_native_op("em_layer_stack_backward")(
        cos_theta,
        material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        float(frequency_hz),
        list(grad_outputs),
        bool(need_cos_theta),
        bool(need_layers),
        bool(need_frequency),
    )
    if not isinstance(out, dict):
        raise TypeError("_channel_native.em_layer_stack_backward must return a dict")
    return out


def em_layer_stack_jvp(
    cos_theta: torch.Tensor,
    material_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    frequency_hz: float,
    tangent_cos_theta: torch.Tensor | None,
    tangent_layer_thickness: torch.Tensor | None,
    tangent_layer_eps_r: torch.Tensor | None,
    tangent_layer_sigma_e: torch.Tensor | None,
    tangent_frequency: float,
) -> dict[str, torch.Tensor]:
    out = _required_native_op("em_layer_stack_jvp")(
        cos_theta,
        material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        float(frequency_hz),
        tangent_cos_theta,
        tangent_layer_thickness,
        tangent_layer_eps_r,
        tangent_layer_sigma_e,
        float(tangent_frequency),
    )
    if not isinstance(out, dict):
        raise TypeError("_channel_native.em_layer_stack_jvp must return a dict")
    return out


class _EmLayerStackAdFunction(torch.autograd.Function):
    """Differentiable layer-stack r/t coefficients and power budgets.

    Differentiable inputs: cos_theta (per row), the CSR layer thickness /
    eps_r / sigma_e and the carrier frequency. layer_mu_r and the CSR
    topology stay fixed under the plan 07 contract; requesting the mu_r
    gradient fails loudly. Layer gradients accumulate atomically because the
    CSR store is shared by every row.
    """

    @staticmethod
    def forward(
        cos_theta,
        material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        frequency,
        frequency_value,
    ):
        out = em_layer_stack_eval(
            cos_theta,
            material_id,
            layer_offset,
            layer_count,
            layer_thickness_m,
            layer_eps_r,
            layer_sigma_e,
            layer_mu_r,
            frequency_hz=frequency_value,
        )
        return tuple(out[name] for name in _EM_LAYER_STACK_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        frequency = inputs[8]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:8]
        )
        ctx.frequency_value = inputs[9]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        none_grads = (None,) * 10
        _ad_reject_fixed_inputs(
            "em_layer_stack_ad",
            ctx.needs_input_grad,
            ((7, "layer_mu_r"),),
        )
        need_cos = bool(ctx.needs_input_grad[0])
        need_layers = any(bool(ctx.needs_input_grad[i]) for i in (4, 5, 6))
        need_frequency = bool(ctx.needs_input_grad[8])
        if not (need_cos or need_layers or need_frequency) or all(
            value is None for value in grad_outputs
        ):
            return none_grads
        saved = ctx.saved_tensors
        out = em_layer_stack_backward(
            *saved,
            grad_outputs,
            frequency_hz=ctx.frequency_value,
            need_cos_theta=need_cos,
            need_layers=need_layers,
            need_frequency=need_frequency,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            out["grad_cos_theta"] if need_cos else None,
            None,
            None,
            None,
            out["grad_layer_thickness_m"] if ctx.needs_input_grad[4] else None,
            out["grad_layer_eps_r"] if ctx.needs_input_grad[5] else None,
            out["grad_layer_sigma_e"] if ctx.needs_input_grad[6] else None,
            None,
            grad_frequency,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        t_cos_theta,
        _t_material_id,
        _t_layer_offset,
        _t_layer_count,
        t_layer_thickness,
        t_layer_eps_r,
        t_layer_sigma_e,
        t_layer_mu_r,
        t_frequency,
        _t_frequency_value,
    ):
        _ad_reject_fixed_tangents(
            "em_layer_stack_ad", ((t_layer_mu_r, "layer_mu_r"),)
        )
        saved = ctx.saved_tensors
        tangent_cos = _ad_checked_tangent(
            "em_layer_stack_ad tangent_cos_theta",
            _ad_native_tangent_or_none(t_cos_theta),
            tuple(saved[0].shape),
        )
        layer_shape = tuple(saved[4].shape)
        tangent_thickness = _ad_checked_tangent(
            "em_layer_stack_ad tangent_layer_thickness_m",
            _ad_native_tangent_or_none(t_layer_thickness),
            layer_shape,
        )
        tangent_eps = _ad_checked_tangent(
            "em_layer_stack_ad tangent_layer_eps_r",
            _ad_native_tangent_or_none(t_layer_eps_r),
            layer_shape,
        )
        tangent_sigma = _ad_checked_tangent(
            "em_layer_stack_ad tangent_layer_sigma_e",
            _ad_native_tangent_or_none(t_layer_sigma_e),
            layer_shape,
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        if (
            tangent_cos is None
            and tangent_thickness is None
            and tangent_eps is None
            and tangent_sigma is None
            and tangent_frequency == 0.0
        ):
            return (None,) * len(_EM_LAYER_STACK_FIELDS)
        with torch_compat.disable_functorch():
            out = em_layer_stack_jvp(
                *(_ad_native_tensor(value) for value in saved),
                frequency_hz=ctx.frequency_value,
                tangent_cos_theta=tangent_cos,
                tangent_layer_thickness=tangent_thickness,
                tangent_layer_eps_r=tangent_eps,
                tangent_layer_sigma_e=tangent_sigma,
                tangent_frequency=tangent_frequency,
            )
        return tuple(out[name] for name in _EM_LAYER_STACK_FIELDS)


def em_layer_stack_ad(
    cos_theta: torch.Tensor,
    material_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`em_layer_stack_eval` (plan 07 AD-3).

    ``frequency_value`` is the precomputed host scalar of ``frequency``; a
    seam that applies several Functions per solve reads the 0-d tensor once
    and threads the float here so no Function re-reads it (audit M3). When
    not supplied it is read here, exactly once per apply.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _EmLayerStackAdFunction.apply(
        cos_theta,
        material_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        frequency,
        float(frequency_value),
    )
    return dict(zip(_EM_LAYER_STACK_FIELDS, values, strict=True))


def scattering_table_eval(
    wi: torch.Tensor,
    wo: torch.Tensor,
    f_te: torch.Tensor,
    f_tm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native CUDA multilinear Kirchhoff-table evaluation; required op."""

    validate_cuda_tensor("wi", wi, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("wo", wo, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("f_te", f_te, dtype=torch.float32, ndim=4)
    validate_cuda_tensor("f_tm", f_tm, dtype=torch.float32, ndim=4)
    if wi.shape != wo.shape or wi.shape[1:] != (3,):
        raise ValueError("wi and wo must have matching shape (N, 3)")
    out = _required_native_op("scattering_table_eval")(wi, wo, f_te, f_tm)
    if not isinstance(out, dict) or set(out) != {"f_te", "f_tm"}:
        raise TypeError("_channel_native.scattering_table_eval returned invalid fields")
    return out["f_te"], out["f_tm"]


def scattering_table_pdf(
    wi: torch.Tensor,
    wo: torch.Tensor,
    sample_density: torch.Tensor,
    *,
    reverse: bool = False,
) -> torch.Tensor:
    """Native CUDA piecewise-constant Kirchhoff PDF; required op."""

    validate_cuda_tensor("wi", wi, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("wo", wo, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("sample_density", sample_density, dtype=torch.float32, ndim=4)
    return _required_native_op("scattering_table_pdf")(
        wi, wo, sample_density, bool(reverse)
    )


def scattering_table_sample(
    wi: torch.Tensor,
    uniforms: torch.Tensor,
    marginal_cdf: torch.Tensor,
    conditional_cdf: torch.Tensor,
    sample_density: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Native CUDA CDF inversion plus forward/reverse PDFs; required op."""

    validate_cuda_tensor("wi", wi, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("uniforms", uniforms, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("marginal_cdf", marginal_cdf, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("conditional_cdf", conditional_cdf, dtype=torch.float32, ndim=4)
    validate_cuda_tensor("sample_density", sample_density, dtype=torch.float32, ndim=4)
    out = _required_native_op("scattering_table_sample")(
        wi, uniforms, marginal_cdf, conditional_cdf, sample_density
    )
    expected = {"wo", "pdf_forward", "pdf_reverse"}
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError("_channel_native.scattering_table_sample returned invalid fields")
    return out


def scattering_event_probabilities(
    cos_theta: torch.Tensor,
    material_id: torch.Tensor,
    cap_r_te: torch.Tensor,
    cap_r_tm: torch.Tensor,
    cap_t_te: torch.Tensor,
    cap_t_tm: torch.Tensor,
    rough_sigma_h_m: torch.Tensor,
    scatter_model_id: torch.Tensor,
    *,
    frequency_hz: float,
    probability_floor: float,
) -> dict[str, torch.Tensor]:
    """Fused native CUDA rough-event budgets and probabilities."""

    validate_cuda_tensor("cos_theta", cos_theta, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_id", material_id, dtype=torch.int32, ndim=1)
    out = _required_native_op("scattering_event_probabilities")(
        cos_theta,
        material_id,
        cap_r_te,
        cap_r_tm,
        cap_t_te,
        cap_t_tm,
        rough_sigma_h_m,
        scatter_model_id,
        float(frequency_hz),
        float(probability_floor),
    )
    expected = {"p_scatter", "p_transmit", "r_coh_amplitude", "rough"}
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError("_channel_native.scattering_event_probabilities returned invalid fields")
    return out


def field_coupled_rd(
    source: torch.Tensor,
    target: torch.Tensor,
    reflection_position: torch.Tensor,
    reflection_normal: torch.Tensor,
    edge_position: torch.Tensor,
    edge_direction: torch.Tensor,
    edge_n0: torch.Tensor,
    edge_n1: torch.Tensor,
    exterior_angle: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    reflection_material: tuple[torch.Tensor, ...],
    wedge_material0: tuple[torch.Tensor, ...],
    wedge_material1: tuple[torch.Tensor, ...],
    *,
    frequency_hz: float,
    reverse: bool,
) -> dict[str, torch.Tensor]:
    vectors = (
        source,
        target,
        reflection_position,
        reflection_normal,
        edge_position,
        edge_direction,
        edge_n0,
        edge_n1,
        tx_polarization,
        rx_polarization,
    )
    count = int(source.shape[0])
    for value in vectors:
        validate_cuda_tensor(
            "coupled_vector", value, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
        if value.shape != (count, 3):
            raise ValueError("coupled field vector tensors must have shape (N, 3)")
    if any(len(bundle) != 5 for bundle in (reflection_material, wedge_material0, wedge_material1)):
        raise ValueError("coupled material bundles must contain eps/sigma/mu/gain/thickness")
    scalars = (
        exterior_angle,
        tx_power,
        *reflection_material,
        *wedge_material0,
        *wedge_material1,
    )
    for value in scalars:
        validate_cuda_tensor("coupled_scalar", value, dtype=torch.float32, ndim=1)
        if value.shape != (count,):
            raise ValueError("coupled field scalar tensors must have shape (N,)")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    out = _required_native_op("field_coupled_rd")(
        *vectors[:8],
        exterior_angle,
        tx_power,
        tx_polarization,
        rx_polarization,
        *reflection_material,
        *wedge_material0,
        *wedge_material1,
        float(frequency_hz),
        bool(reverse),
    )
    if not isinstance(out, dict):
        raise TypeError("_channel_native.field_coupled_rd must return a dict")
    schema = {
        "field_vector": (torch.complex64, 2, (count, 3)),
        "coefficient": (torch.complex64, 1, (count,)),
        "path_field": (torch.complex64, 1, (count,)),
        "path_gain": (torch.float32, 1, (count,)),
        "direction": (torch.float32, 2, (count, 3)),
    }
    if set(out) != set(schema):
        raise ValueError("field_coupled_rd returned unexpected fields")
    for name, (dtype, ndim, shape) in schema.items():
        validate_cuda_tensor(name, out[name], dtype=dtype, ndim=ndim)
        if tuple(out[name].shape) != shape:
            raise ValueError(f"field_coupled_rd returned bad {name} shape")
    return out


# ---------------------------------------------------------------------------
# Fixed-topology EM-response AD for the field transport kernels (plan 07
# AD-1 materials/frequency, AD-2 geometry). Differentiable inputs:
# per-bounce/per-layer eps_r, sigma_e, gain, thickness, the carrier frequency
# and the continuous hit geometry (source, target, interaction_positions,
# interaction_normals). The discrete winner (polarizations, tx_power, mu_r,
# material ids, valid masks) stays fixed and fails loudly. path_length_m /
# delay_s become differentiable outputs exactly when a geometry input is on
# the graph; they carry zero cotangent into materials and frequency.
# ---------------------------------------------------------------------------

_FIELD_AD_OUTPUT_FIELDS = (
    "field_vector",
    "coefficient",
    "path_field",
    "path_gain",
    "path_length_m",
    "delay_s",
    "direction",
)
_FIELD_AD_TANGENT_FIELDS = (
    "field_vector",
    "coefficient",
    "path_field",
    "path_gain",
    "path_length_m",
    "delay_s",
)


def field_free_space_backward(
    source: torch.Tensor,
    target: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    *,
    frequency_hz: float,
    grad_field_vector: torch.Tensor | None = None,
    grad_coefficient: torch.Tensor | None = None,
    grad_path_field: torch.Tensor | None = None,
    grad_path_gain: torch.Tensor | None = None,
    grad_path_length: torch.Tensor | None = None,
    grad_delay: torch.Tensor | None = None,
    need_grad_frequency: bool = True,
    need_grad_geometry: bool = False,
) -> dict[str, torch.Tensor | None]:
    out = _required_native_op("field_free_space_backward")(
        source,
        target,
        tx_power,
        tx_polarization,
        rx_polarization,
        float(frequency_hz),
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
        grad_path_length,
        grad_delay,
        bool(need_grad_frequency),
        bool(need_grad_geometry),
    )
    expected = {"grad_frequency", "grad_source", "grad_target"}
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError("_channel_native.field_free_space_backward returned invalid fields")
    return out


def field_free_space_jvp(
    source: torch.Tensor,
    target: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    *,
    frequency_hz: float,
    tangent_frequency: float,
    tangent_source: torch.Tensor | None = None,
    tangent_target: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    out = _required_native_op("field_free_space_jvp")(
        source,
        target,
        tx_power,
        tx_polarization,
        rx_polarization,
        float(frequency_hz),
        float(tangent_frequency),
        tangent_source,
        tangent_target,
    )
    if not isinstance(out, dict) or set(out) != set(_FIELD_AD_TANGENT_FIELDS):
        raise TypeError("_channel_native.field_free_space_jvp returned invalid fields")
    return out


def field_reflection_sequence_backward(
    source: torch.Tensor,
    target: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    thickness: torch.Tensor,
    *,
    frequency_hz: float,
    grad_field_vector: torch.Tensor | None = None,
    grad_coefficient: torch.Tensor | None = None,
    grad_path_field: torch.Tensor | None = None,
    grad_path_gain: torch.Tensor | None = None,
    grad_path_length: torch.Tensor | None = None,
    grad_delay: torch.Tensor | None = None,
    need_grad_eps_r: bool = True,
    need_grad_sigma_e: bool = True,
    need_grad_gain: bool = False,
    need_grad_thickness: bool = True,
    need_grad_frequency: bool = True,
    need_grad_geometry: bool = False,
) -> dict[str, torch.Tensor | None]:
    out = _required_native_op("field_reflection_sequence_backward")(
        source,
        target,
        interaction_positions,
        interaction_normals,
        tx_power,
        tx_polarization,
        rx_polarization,
        eps_r,
        sigma_e,
        mu_r,
        gain,
        thickness,
        float(frequency_hz),
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
        grad_path_length,
        grad_delay,
        bool(need_grad_eps_r),
        bool(need_grad_sigma_e),
        bool(need_grad_gain),
        bool(need_grad_thickness),
        bool(need_grad_frequency),
        bool(need_grad_geometry),
    )
    expected = {
        "grad_eps_r",
        "grad_sigma_e",
        "grad_gain",
        "grad_thickness",
        "grad_frequency",
        "grad_source",
        "grad_target",
        "grad_interaction_positions",
        "grad_interaction_normals",
    }
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError(
            "_channel_native.field_reflection_sequence_backward returned invalid fields"
        )
    return out


def field_reflection_sequence_jvp(
    source: torch.Tensor,
    target: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    thickness: torch.Tensor,
    *,
    frequency_hz: float,
    tangent_eps_r: torch.Tensor | None = None,
    tangent_sigma_e: torch.Tensor | None = None,
    tangent_gain: torch.Tensor | None = None,
    tangent_thickness: torch.Tensor | None = None,
    tangent_frequency: float = 0.0,
    tangent_source: torch.Tensor | None = None,
    tangent_target: torch.Tensor | None = None,
    tangent_interaction_positions: torch.Tensor | None = None,
    tangent_interaction_normals: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    out = _required_native_op("field_reflection_sequence_jvp")(
        source,
        target,
        interaction_positions,
        interaction_normals,
        tx_power,
        tx_polarization,
        rx_polarization,
        eps_r,
        sigma_e,
        mu_r,
        gain,
        thickness,
        float(frequency_hz),
        tangent_eps_r,
        tangent_sigma_e,
        tangent_gain,
        tangent_thickness,
        float(tangent_frequency),
        tangent_source,
        tangent_target,
        tangent_interaction_positions,
        tangent_interaction_normals,
    )
    if not isinstance(out, dict) or set(out) != set(_FIELD_AD_TANGENT_FIELDS):
        raise TypeError(
            "_channel_native.field_reflection_sequence_jvp returned invalid fields"
        )
    return out


def field_transmission_sequence_backward(
    source: torch.Tensor,
    target: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    interaction_material_id: torch.Tensor,
    interaction_valid: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    frequency_hz: float,
    grad_field_vector: torch.Tensor | None = None,
    grad_coefficient: torch.Tensor | None = None,
    grad_path_field: torch.Tensor | None = None,
    grad_path_gain: torch.Tensor | None = None,
    grad_path_length: torch.Tensor | None = None,
    grad_delay: torch.Tensor | None = None,
    need_grad_layer_thickness: bool = True,
    need_grad_layer_eps_r: bool = True,
    need_grad_layer_sigma_e: bool = True,
    need_grad_frequency: bool = True,
    need_grad_geometry: bool = False,
) -> dict[str, torch.Tensor | None]:
    out = _required_native_op("field_transmission_sequence_backward")(
        source,
        target,
        interaction_positions,
        interaction_normals,
        interaction_material_id,
        interaction_valid,
        tx_power,
        tx_polarization,
        rx_polarization,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        float(frequency_hz),
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
        grad_path_length,
        grad_delay,
        bool(need_grad_layer_thickness),
        bool(need_grad_layer_eps_r),
        bool(need_grad_layer_sigma_e),
        bool(need_grad_frequency),
        bool(need_grad_geometry),
    )
    expected = {
        "grad_layer_thickness_m",
        "grad_layer_eps_r",
        "grad_layer_sigma_e",
        "grad_frequency",
        "grad_source",
        "grad_target",
        "grad_interaction_positions",
        "grad_interaction_normals",
    }
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError(
            "_channel_native.field_transmission_sequence_backward returned invalid fields"
        )
    return out


def field_transmission_sequence_jvp(
    source: torch.Tensor,
    target: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    interaction_material_id: torch.Tensor,
    interaction_valid: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    frequency_hz: float,
    tangent_layer_thickness_m: torch.Tensor | None = None,
    tangent_layer_eps_r: torch.Tensor | None = None,
    tangent_layer_sigma_e: torch.Tensor | None = None,
    tangent_frequency: float = 0.0,
    tangent_source: torch.Tensor | None = None,
    tangent_target: torch.Tensor | None = None,
    tangent_interaction_positions: torch.Tensor | None = None,
    tangent_interaction_normals: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    out = _required_native_op("field_transmission_sequence_jvp")(
        source,
        target,
        interaction_positions,
        interaction_normals,
        interaction_material_id,
        interaction_valid,
        tx_power,
        tx_polarization,
        rx_polarization,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        float(frequency_hz),
        tangent_layer_thickness_m,
        tangent_layer_eps_r,
        tangent_layer_sigma_e,
        float(tangent_frequency),
        tangent_source,
        tangent_target,
        tangent_interaction_positions,
        tangent_interaction_normals,
    )
    if not isinstance(out, dict) or set(out) != set(_FIELD_AD_TANGENT_FIELDS):
        raise TypeError(
            "_channel_native.field_transmission_sequence_jvp returned invalid fields"
        )
    return out


class _FieldFreeSpaceAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable free-space transport (plan 07 AD-1/AD-2).

    Differentiable inputs: frequency (0-d tensor) and the source / target
    endpoints. tx_power and the polarizations stay fixed; requesting their
    gradient fails loudly. path_length_m / delay_s are differentiable exactly
    when an endpoint is on the graph. A float64 input batch routes through
    the float64 companion forward so torch.autograd.gradcheck can run in
    strict double precision.
    """

    @staticmethod
    def forward(
        source,
        target,
        tx_power,
        tx_polarization,
        rx_polarization,
        frequency,
        frequency_value,
    ):
        op_name = (
            "field_free_space_fwd64"
            if source.dtype == torch.float64
            else "field_free_space"
        )
        out = _required_native_op(op_name)(
            source,
            target,
            tx_power,
            tx_polarization,
            rx_polarization,
            frequency_value,
        )
        return tuple(out[name] for name in _FIELD_AD_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        (
            source,
            target,
            tx_power,
            tx_polarization,
            rx_polarization,
            frequency,
            frequency_value,
        ) = inputs
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (source, target, tx_power, tx_polarization, rx_polarization)
        )
        ctx.frequency_value = frequency_value
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.geometry_live = _ad_geometry_live(source, target)
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        if ctx.geometry_live:
            ctx.mark_non_differentiable(output[6])
        else:
            ctx.mark_non_differentiable(output[4], output[5], output[6])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx,
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
        grad_path_length,
        grad_delay,
        _grad_direction,
    ):
        none_grads = (None,) * 7
        _ad_reject_fixed_inputs(
            "field_free_space_ad",
            ctx.needs_input_grad,
            (
                (2, "tx_power"),
                (3, "tx_polarization"),
                (4, "rx_polarization"),
            ),
        )
        need_geometry = bool(ctx.needs_input_grad[0]) or bool(
            ctx.needs_input_grad[1]
        )
        need_frequency = bool(ctx.needs_input_grad[5])
        grads = (
            grad_field_vector,
            grad_coefficient,
            grad_path_field,
            grad_path_gain,
            grad_path_length,
            grad_delay,
        )
        if not (need_geometry or need_frequency) or all(
            value is None for value in grads
        ):
            return none_grads
        source, target, tx_power, tx_polarization, rx_polarization = ctx.saved_tensors
        out = field_free_space_backward(
            source,
            target,
            tx_power,
            tx_polarization,
            rx_polarization,
            frequency_hz=ctx.frequency_value,
            grad_field_vector=grad_field_vector,
            grad_coefficient=grad_coefficient,
            grad_path_field=grad_path_field,
            grad_path_gain=grad_path_gain,
            grad_path_length=grad_path_length,
            grad_delay=grad_delay,
            need_grad_frequency=need_frequency,
            need_grad_geometry=need_geometry,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            out["grad_source"] if ctx.needs_input_grad[0] else None,
            out["grad_target"] if ctx.needs_input_grad[1] else None,
            None,
            None,
            None,
            grad_frequency,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        t_source,
        t_target,
        t_tx_power,
        t_tx_pol,
        t_rx_pol,
        t_frequency,
        _t_frequency_value,
    ):
        _ad_reject_fixed_tangents(
            "field_free_space_ad",
            (
                (t_tx_power, "tx_power"),
                (t_tx_pol, "tx_polarization"),
                (t_rx_pol, "rx_polarization"),
            ),
        )
        saved = ctx.saved_tensors
        tangent_source = _ad_geometry_tangent(
            "field_free_space_ad tangent_source", t_source, saved[0]
        )
        tangent_target = _ad_geometry_tangent(
            "field_free_space_ad tangent_target", t_target, saved[1]
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        if (
            tangent_frequency == 0.0
            and tangent_source is None
            and tangent_target is None
        ):
            return (None,) * 7
        source, target, tx_power, tx_polarization, rx_polarization = saved
        with torch_compat.disable_functorch():
            out = field_free_space_jvp(
                _ad_native_tensor(source),
                _ad_native_tensor(target),
                _ad_native_tensor(tx_power),
                _ad_native_tensor(tx_polarization),
                _ad_native_tensor(rx_polarization),
                frequency_hz=ctx.frequency_value,
                tangent_frequency=tangent_frequency,
                tangent_source=tangent_source,
                tangent_target=tangent_target,
            )
        tangents = tuple(out[name] for name in _FIELD_AD_TANGENT_FIELDS)
        if not ctx.geometry_live:
            return (*tangents[:4], None, None, None)
        return (*tangents, None)


def field_free_space_ad(
    source: torch.Tensor,
    target: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_free_space` (frequency only in AD-1).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam, audit M3); when not
    supplied it is read here, exactly once per apply.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldFreeSpaceAdFunction.apply(
        source,
        target,
        tx_power,
        tx_polarization,
        rx_polarization,
        frequency,
        float(frequency_value),
    )
    return dict(zip(_FIELD_AD_OUTPUT_FIELDS, values, strict=True))


class _FieldReflectionSequenceAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable reflection transport (plan 07 AD-1/AD-2).

    Differentiable inputs: per-bounce eps_r / sigma_e / gain / thickness,
    frequency, and the hit geometry (source, target, interaction_positions,
    interaction_normals). tx_power, the polarizations and mu_r are fixed;
    requesting their gradient fails loudly instead of silently returning
    zeros. path_length_m / delay_s are differentiable exactly when a geometry
    input is on the graph.
    """

    @staticmethod
    def forward(
        source,
        target,
        interaction_positions,
        interaction_normals,
        tx_power,
        tx_polarization,
        rx_polarization,
        eps_r,
        sigma_e,
        mu_r,
        gain,
        thickness,
        frequency,
        frequency_value,
    ):
        out = _required_native_op("field_reflection_sequence")(
            source,
            target,
            interaction_positions,
            interaction_normals,
            tx_power,
            tx_polarization,
            rx_polarization,
            eps_r,
            sigma_e,
            mu_r,
            gain,
            thickness,
            frequency_value,
        )
        return tuple(out[name] for name in _FIELD_AD_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        frequency = inputs[12]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:12]
        )
        ctx.frequency_value = inputs[13]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.geometry_live = _ad_geometry_live(*inputs[:4])
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        if ctx.geometry_live:
            ctx.mark_non_differentiable(output[6])
        else:
            ctx.mark_non_differentiable(output[4], output[5], output[6])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx,
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
        grad_path_length,
        grad_delay,
        _grad_direction,
    ):
        none_grads = (None,) * 14
        _ad_reject_fixed_inputs(
            "field_reflection_sequence_ad",
            ctx.needs_input_grad,
            (
                (4, "tx_power"),
                (5, "tx_polarization"),
                (6, "rx_polarization"),
                (9, "mu_r"),
            ),
        )
        need_geometry = any(bool(ctx.needs_input_grad[i]) for i in range(4))
        need_eps = bool(ctx.needs_input_grad[7])
        need_sigma = bool(ctx.needs_input_grad[8])
        need_gain = bool(ctx.needs_input_grad[10])
        need_thickness = bool(ctx.needs_input_grad[11])
        need_frequency = bool(ctx.needs_input_grad[12])
        grads = (
            grad_field_vector,
            grad_coefficient,
            grad_path_field,
            grad_path_gain,
            grad_path_length,
            grad_delay,
        )
        if not (
            need_geometry
            or need_eps
            or need_sigma
            or need_gain
            or need_thickness
            or need_frequency
        ) or all(value is None for value in grads):
            return none_grads
        (
            source,
            target,
            interaction_positions,
            interaction_normals,
            tx_power,
            tx_polarization,
            rx_polarization,
            eps_r,
            sigma_e,
            mu_r,
            gain,
            thickness,
        ) = ctx.saved_tensors
        out = field_reflection_sequence_backward(
            source,
            target,
            interaction_positions,
            interaction_normals,
            tx_power,
            tx_polarization,
            rx_polarization,
            eps_r,
            sigma_e,
            mu_r,
            gain,
            thickness,
            frequency_hz=ctx.frequency_value,
            grad_field_vector=grad_field_vector,
            grad_coefficient=grad_coefficient,
            grad_path_field=grad_path_field,
            grad_path_gain=grad_path_gain,
            grad_path_length=grad_path_length,
            grad_delay=grad_delay,
            need_grad_eps_r=need_eps,
            need_grad_sigma_e=need_sigma,
            need_grad_gain=need_gain,
            need_grad_thickness=need_thickness,
            need_grad_frequency=need_frequency,
            need_grad_geometry=need_geometry,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            out["grad_source"] if ctx.needs_input_grad[0] else None,
            out["grad_target"] if ctx.needs_input_grad[1] else None,
            out["grad_interaction_positions"] if ctx.needs_input_grad[2] else None,
            out["grad_interaction_normals"] if ctx.needs_input_grad[3] else None,
            None,
            None,
            None,
            out["grad_eps_r"] if need_eps else None,
            out["grad_sigma_e"] if need_sigma else None,
            None,
            out["grad_gain"] if need_gain else None,
            out["grad_thickness"] if need_thickness else None,
            grad_frequency,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        t_source,
        t_target,
        t_positions,
        t_normals,
        t_tx_power,
        t_tx_pol,
        t_rx_pol,
        t_eps_r,
        t_sigma_e,
        t_mu_r,
        t_gain,
        t_thickness,
        t_frequency,
        _t_frequency_value,
    ):
        _ad_reject_fixed_tangents(
            "field_reflection_sequence_ad",
            (
                (t_tx_power, "tx_power"),
                (t_tx_pol, "tx_polarization"),
                (t_rx_pol, "rx_polarization"),
                (t_mu_r, "mu_r"),
            ),
        )
        saved = ctx.saved_tensors
        eps_shape = tuple(saved[7].shape)
        tangent_source = _ad_geometry_tangent(
            "field_reflection_sequence_ad tangent_source", t_source, saved[0]
        )
        tangent_target = _ad_geometry_tangent(
            "field_reflection_sequence_ad tangent_target", t_target, saved[1]
        )
        tangent_positions = _ad_geometry_tangent(
            "field_reflection_sequence_ad tangent_interaction_positions",
            t_positions,
            saved[2],
        )
        tangent_normals = _ad_geometry_tangent(
            "field_reflection_sequence_ad tangent_interaction_normals",
            t_normals,
            saved[3],
        )
        tangent_eps = _ad_checked_tangent(
            "field_reflection_sequence_ad tangent_eps_r",
            _ad_native_tangent_or_none(t_eps_r),
            eps_shape,
        )
        tangent_sigma = _ad_checked_tangent(
            "field_reflection_sequence_ad tangent_sigma_e",
            _ad_native_tangent_or_none(t_sigma_e),
            eps_shape,
        )
        tangent_gain = _ad_checked_tangent(
            "field_reflection_sequence_ad tangent_gain",
            _ad_native_tangent_or_none(t_gain),
            eps_shape,
        )
        tangent_thickness = _ad_checked_tangent(
            "field_reflection_sequence_ad tangent_thickness",
            _ad_native_tangent_or_none(t_thickness),
            eps_shape,
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        if (
            tangent_eps is None
            and tangent_sigma is None
            and tangent_gain is None
            and tangent_thickness is None
            and tangent_source is None
            and tangent_target is None
            and tangent_positions is None
            and tangent_normals is None
            and tangent_frequency == 0.0
        ):
            return (None,) * 7
        with torch_compat.disable_functorch():
            out = field_reflection_sequence_jvp(
                *(_ad_native_tensor(value) for value in saved),
                frequency_hz=ctx.frequency_value,
                tangent_eps_r=tangent_eps,
                tangent_sigma_e=tangent_sigma,
                tangent_gain=tangent_gain,
                tangent_thickness=tangent_thickness,
                tangent_frequency=tangent_frequency,
                tangent_source=tangent_source,
                tangent_target=tangent_target,
                tangent_interaction_positions=tangent_positions,
                tangent_interaction_normals=tangent_normals,
            )
        tangents = tuple(out[name] for name in _FIELD_AD_TANGENT_FIELDS)
        if not ctx.geometry_live:
            return (*tangents[:4], None, None, None)
        return (*tangents, None)


def field_reflection_sequence_ad(
    source: torch.Tensor,
    target: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    thickness: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_reflection_sequence` (materials + frequency).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam, audit M3); when not
    supplied it is read here, exactly once per apply.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldReflectionSequenceAdFunction.apply(
        source,
        target,
        interaction_positions,
        interaction_normals,
        tx_power,
        tx_polarization,
        rx_polarization,
        eps_r,
        sigma_e,
        mu_r,
        gain,
        thickness,
        frequency,
        float(frequency_value),
    )
    return dict(zip(_FIELD_AD_OUTPUT_FIELDS, values, strict=True))


class _FieldTransmissionSequenceAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable transmission transport (plan 07 AD-1/AD-2).

    Differentiable inputs: CSR layer thickness / eps_r / sigma_e, frequency,
    and the hit geometry (source, target, interaction_normals). The straight
    transmission field is independent of the crossing points themselves, so
    interaction_positions receives an exact zero gradient (None). tx_power,
    the polarizations, layer_mu_r, material ids and valid masks are fixed;
    requesting their gradient fails loudly. Layer gradients accumulate
    atomically across paths because the CSR store is shared by every wall
    crossing.
    """

    @staticmethod
    def forward(
        source,
        target,
        interaction_positions,
        interaction_normals,
        interaction_material_id,
        interaction_valid,
        tx_power,
        tx_polarization,
        rx_polarization,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        frequency,
        frequency_value,
    ):
        out = _required_native_op("field_transmission_sequence")(
            source,
            target,
            interaction_positions,
            interaction_normals,
            interaction_material_id,
            interaction_valid,
            tx_power,
            tx_polarization,
            rx_polarization,
            layer_offset,
            layer_count,
            layer_thickness_m,
            layer_eps_r,
            layer_sigma_e,
            layer_mu_r,
            frequency_value,
        )
        return tuple(out[name] for name in _FIELD_AD_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        frequency = inputs[15]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:15]
        )
        ctx.frequency_value = inputs[16]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.geometry_live = _ad_geometry_live(*inputs[:4])
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        if ctx.geometry_live:
            ctx.mark_non_differentiable(output[6])
        else:
            ctx.mark_non_differentiable(output[4], output[5], output[6])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx,
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
        grad_path_length,
        grad_delay,
        _grad_direction,
    ):
        none_grads = (None,) * 17
        _ad_reject_fixed_inputs(
            "field_transmission_sequence_ad",
            ctx.needs_input_grad,
            (
                (6, "tx_power"),
                (7, "tx_polarization"),
                (8, "rx_polarization"),
                (14, "layer_mu_r"),
            ),
        )
        # interaction_positions (index 2) never enters the straight-path
        # field: its gradient is exactly zero, so it does not drive a launch.
        need_geometry = (
            bool(ctx.needs_input_grad[0])
            or bool(ctx.needs_input_grad[1])
            or bool(ctx.needs_input_grad[3])
        )
        need_thickness = bool(ctx.needs_input_grad[11])
        need_eps = bool(ctx.needs_input_grad[12])
        need_sigma = bool(ctx.needs_input_grad[13])
        need_frequency = bool(ctx.needs_input_grad[15])
        grads = (
            grad_field_vector,
            grad_coefficient,
            grad_path_field,
            grad_path_gain,
            grad_path_length,
            grad_delay,
        )
        if not (
            need_geometry or need_thickness or need_eps or need_sigma
            or need_frequency
        ) or all(value is None for value in grads):
            return none_grads
        saved = ctx.saved_tensors
        out = field_transmission_sequence_backward(
            *saved,
            frequency_hz=ctx.frequency_value,
            grad_field_vector=grad_field_vector,
            grad_coefficient=grad_coefficient,
            grad_path_field=grad_path_field,
            grad_path_gain=grad_path_gain,
            grad_path_length=grad_path_length,
            grad_delay=grad_delay,
            need_grad_layer_thickness=need_thickness,
            need_grad_layer_eps_r=need_eps,
            need_grad_layer_sigma_e=need_sigma,
            need_grad_frequency=need_frequency,
            need_grad_geometry=need_geometry,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            out["grad_source"] if ctx.needs_input_grad[0] else None,
            out["grad_target"] if ctx.needs_input_grad[1] else None,
            None,
            out["grad_interaction_normals"] if ctx.needs_input_grad[3] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            out["grad_layer_thickness_m"] if need_thickness else None,
            out["grad_layer_eps_r"] if need_eps else None,
            out["grad_layer_sigma_e"] if need_sigma else None,
            None,
            grad_frequency,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        t_source,
        t_target,
        t_positions,
        t_normals,
        _t_material_id,
        _t_valid,
        t_tx_power,
        t_tx_pol,
        t_rx_pol,
        _t_layer_offset,
        _t_layer_count,
        t_layer_thickness,
        t_layer_eps_r,
        t_layer_sigma_e,
        t_layer_mu_r,
        t_frequency,
        _t_frequency_value,
    ):
        _ad_reject_fixed_tangents(
            "field_transmission_sequence_ad",
            (
                (t_tx_power, "tx_power"),
                (t_tx_pol, "tx_polarization"),
                (t_rx_pol, "rx_polarization"),
                (t_layer_mu_r, "layer_mu_r"),
            ),
        )
        saved = ctx.saved_tensors
        layer_shape = tuple(saved[11].shape)
        tangent_source = _ad_geometry_tangent(
            "field_transmission_sequence_ad tangent_source", t_source, saved[0]
        )
        tangent_target = _ad_geometry_tangent(
            "field_transmission_sequence_ad tangent_target", t_target, saved[1]
        )
        tangent_positions = _ad_geometry_tangent(
            "field_transmission_sequence_ad tangent_interaction_positions",
            t_positions,
            saved[2],
        )
        tangent_normals = _ad_geometry_tangent(
            "field_transmission_sequence_ad tangent_interaction_normals",
            t_normals,
            saved[3],
        )
        tangent_thickness = _ad_checked_tangent(
            "field_transmission_sequence_ad tangent_layer_thickness_m",
            _ad_native_tangent_or_none(t_layer_thickness),
            layer_shape,
        )
        tangent_eps = _ad_checked_tangent(
            "field_transmission_sequence_ad tangent_layer_eps_r",
            _ad_native_tangent_or_none(t_layer_eps_r),
            layer_shape,
        )
        tangent_sigma = _ad_checked_tangent(
            "field_transmission_sequence_ad tangent_layer_sigma_e",
            _ad_native_tangent_or_none(t_layer_sigma_e),
            layer_shape,
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        if (
            tangent_thickness is None
            and tangent_eps is None
            and tangent_sigma is None
            and tangent_source is None
            and tangent_target is None
            and tangent_positions is None
            and tangent_normals is None
            and tangent_frequency == 0.0
        ):
            return (None,) * 7
        with torch_compat.disable_functorch():
            out = field_transmission_sequence_jvp(
                *(_ad_native_tensor(value) for value in saved),
                frequency_hz=ctx.frequency_value,
                tangent_layer_thickness_m=tangent_thickness,
                tangent_layer_eps_r=tangent_eps,
                tangent_layer_sigma_e=tangent_sigma,
                tangent_frequency=tangent_frequency,
                tangent_source=tangent_source,
                tangent_target=tangent_target,
                tangent_interaction_positions=tangent_positions,
                tangent_interaction_normals=tangent_normals,
            )
        tangents = tuple(out[name] for name in _FIELD_AD_TANGENT_FIELDS)
        if not ctx.geometry_live:
            return (*tangents[:4], None, None, None)
        return (*tangents, None)


def field_transmission_sequence_ad(
    source: torch.Tensor,
    target: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    interaction_material_id: torch.Tensor,
    interaction_valid: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_transmission_sequence` (layers + frequency).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam, audit M3); when not
    supplied it is read here, exactly once per apply.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldTransmissionSequenceAdFunction.apply(
        source,
        target,
        interaction_positions,
        interaction_normals,
        interaction_material_id,
        interaction_valid,
        tx_power,
        tx_polarization,
        rx_polarization,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        frequency,
        float(frequency_value),
    )
    return dict(zip(_FIELD_AD_OUTPUT_FIELDS, values, strict=True))


# ---------------------------------------------------------------------------
# Plan 07 AD-4: differentiable UTD wedge diffraction (component 2 re-evaluated
# from the frozen topology), receiver projection, coupled R-D transport
# (components 3/4) and the coupled stationary-geometry re-solve. Each
# torch.autograd.Function below is dispatch only; the math lives in
# kernels/field_wedge_ad.cu (RayD's templated dual forward).
# ---------------------------------------------------------------------------

_WEDGE_OUTPUT_FIELDS = ("field_vector", "direction")
_COUPLED_OUTPUT_FIELDS = (
    "field_vector",
    "coefficient",
    "path_field",
    "path_gain",
    "direction",
)


def field_diffraction_wedge(
    source: torch.Tensor,
    target: torch.Tensor,
    edge_position: torch.Tensor,
    edge_direction: torch.Tensor,
    edge_t_min: torch.Tensor,
    edge_t_max: torch.Tensor,
    edge_n0: torch.Tensor,
    edge_n1: torch.Tensor,
    exterior_angle: torch.Tensor,
    face0_valid: torch.Tensor,
    face0_eps_r: torch.Tensor,
    face0_sigma_e: torch.Tensor,
    face0_mu_r: torch.Tensor,
    face0_gain: torch.Tensor,
    face1_valid: torch.Tensor,
    face1_eps_r: torch.Tensor,
    face1_sigma_e: torch.Tensor,
    face1_mu_r: torch.Tensor,
    face1_gain: torch.Tensor,
    tx_power: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    """Re-evaluate RayD's order-1 UTD wedge export from the frozen topology."""

    out = _required_native_op("field_diffraction_wedge")(
        source,
        target,
        edge_position,
        edge_direction,
        edge_t_min,
        edge_t_max,
        edge_n0,
        edge_n1,
        exterior_angle,
        face0_valid,
        face0_eps_r,
        face0_sigma_e,
        face0_mu_r,
        face0_gain,
        face1_valid,
        face1_eps_r,
        face1_sigma_e,
        face1_mu_r,
        face1_gain,
        tx_power,
        float(frequency_hz),
        None,
        None,
        None,
        None,
        None,
    )
    if not isinstance(out, dict) or set(out) != set(_WEDGE_OUTPUT_FIELDS):
        raise TypeError(
            "_channel_native.field_diffraction_wedge returned invalid fields"
        )
    return out


class _FieldDiffractionWedgeAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable UTD wedge field (plan 07 AD-4).

    Differentiable inputs: face eps_r / sigma_e / gain for both wedge faces,
    frequency, the endpoints, and (when the caller supplies the winner
    vertices) the per-row edge vertices v0/v1 plus the opposite vertex of
    each wedge face, from which the kernel rebuilds the edge tables so mesh
    vertex gradients reach the edge geometry. The frozen edge tables (anchor,
    direction, bounds, face normals, exterior angle), the valid masks, mu_r
    and tx_power stay fixed; requesting their gradient fails loudly. The
    stationary point on the edge is re-solved inside the kernel, so endpoint
    and vertex gradients include the diffraction-point motion.
    """

    @staticmethod
    def forward(
        source,
        target,
        edge_position,
        edge_direction,
        edge_t_min,
        edge_t_max,
        edge_n0,
        edge_n1,
        exterior_angle,
        face0_valid,
        face0_eps_r,
        face0_sigma_e,
        face0_mu_r,
        face0_gain,
        face1_valid,
        face1_eps_r,
        face1_sigma_e,
        face1_mu_r,
        face1_gain,
        tx_power,
        frequency,
        vertex_v0,
        vertex_v1,
        vertex_opp0,
        vertex_opp1,
        edge_boundary,
        frequency_value,
    ):
        out = _required_native_op("field_diffraction_wedge")(
            source,
            target,
            edge_position,
            edge_direction,
            edge_t_min,
            edge_t_max,
            edge_n0,
            edge_n1,
            exterior_angle,
            face0_valid,
            face0_eps_r,
            face0_sigma_e,
            face0_mu_r,
            face0_gain,
            face1_valid,
            face1_eps_r,
            face1_sigma_e,
            face1_mu_r,
            face1_gain,
            tx_power,
            frequency_value,
            vertex_v0,
            vertex_v1,
            vertex_opp0,
            vertex_opp1,
            edge_boundary,
        )
        return tuple(out[name] for name in _WEDGE_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        frequency = inputs[20]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:20]
        )
        vertex_primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            if isinstance(value, torch.Tensor)
            else value
            for value in inputs[21:26]
        )
        ctx.has_vertices = vertex_primals[0] is not None
        ctx.frequency_value = inputs[26]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.geometry_live = _ad_geometry_live(inputs[0], inputs[1])
        saved = primals + tuple(
            value for value in vertex_primals if value is not None
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)

    @staticmethod
    def _unpack_saved(ctx):
        saved = ctx.saved_tensors
        primals = saved[:20]
        vertices = saved[20:25] if ctx.has_vertices else (None,) * 5
        return primals, vertices

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_field_vector, grad_direction):
        none_grads = (None,) * 27
        _ad_reject_fixed_inputs(
            "field_diffraction_wedge_ad",
            ctx.needs_input_grad,
            (
                (2, "edge_position"),
                (3, "edge_direction"),
                (4, "edge_t_min"),
                (5, "edge_t_max"),
                (6, "edge_n0"),
                (7, "edge_n1"),
                (8, "exterior_angle"),
                (9, "face0_valid"),
                (12, "face0_mu_r"),
                (14, "face1_valid"),
                (17, "face1_mu_r"),
                (19, "tx_power"),
                (25, "edge_boundary"),
            ),
        )
        need_geometry = bool(ctx.needs_input_grad[0]) or bool(
            ctx.needs_input_grad[1]
        )
        need_material = any(
            bool(ctx.needs_input_grad[index]) for index in (10, 11, 13, 15, 16, 18)
        )
        need_frequency = bool(ctx.needs_input_grad[20])
        need_vertices = any(
            bool(ctx.needs_input_grad[index]) for index in (21, 22, 23, 24)
        )
        if not (need_geometry or need_material or need_frequency or need_vertices) or (
            grad_field_vector is None and grad_direction is None
        ):
            return none_grads
        primals, vertices = _FieldDiffractionWedgeAdFunction._unpack_saved(ctx)
        out = _required_native_op("field_diffraction_wedge_backward")(
            *primals,
            ctx.frequency_value,
            *vertices,
            grad_field_vector,
            grad_direction,
            need_material,
            need_frequency,
            need_geometry,
            need_vertices,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            out["grad_source"] if ctx.needs_input_grad[0] else None,
            out["grad_target"] if ctx.needs_input_grad[1] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            out["grad_face0_eps_r"] if ctx.needs_input_grad[10] else None,
            out["grad_face0_sigma_e"] if ctx.needs_input_grad[11] else None,
            None,
            out["grad_face0_gain"] if ctx.needs_input_grad[13] else None,
            None,
            out["grad_face1_eps_r"] if ctx.needs_input_grad[15] else None,
            out["grad_face1_sigma_e"] if ctx.needs_input_grad[16] else None,
            None,
            out["grad_face1_gain"] if ctx.needs_input_grad[18] else None,
            None,
            grad_frequency,
            out["grad_vertex_v0"] if ctx.needs_input_grad[21] else None,
            out["grad_vertex_v1"] if ctx.needs_input_grad[22] else None,
            out["grad_vertex_opp0"] if ctx.needs_input_grad[23] else None,
            out["grad_vertex_opp1"] if ctx.needs_input_grad[24] else None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        _ad_reject_fixed_tangents(
            "field_diffraction_wedge_ad",
            (
                (tangents[2], "edge_position"),
                (tangents[3], "edge_direction"),
                (tangents[4], "edge_t_min"),
                (tangents[5], "edge_t_max"),
                (tangents[6], "edge_n0"),
                (tangents[7], "edge_n1"),
                (tangents[8], "exterior_angle"),
                (tangents[12], "face0_mu_r"),
                (tangents[17], "face1_mu_r"),
                (tangents[19], "tx_power"),
            ),
        )
        primals, vertices = _FieldDiffractionWedgeAdFunction._unpack_saved(ctx)
        scalar_shape = tuple(primals[10].shape)
        tangent_source = _ad_geometry_tangent(
            "field_diffraction_wedge_ad tangent_source", tangents[0], primals[0]
        )
        tangent_target = _ad_geometry_tangent(
            "field_diffraction_wedge_ad tangent_target", tangents[1], primals[1]
        )
        material_tangents = {}
        for index, name in (
            (10, "face0_eps_r"),
            (11, "face0_sigma_e"),
            (13, "face0_gain"),
            (15, "face1_eps_r"),
            (16, "face1_sigma_e"),
            (18, "face1_gain"),
        ):
            material_tangents[name] = _ad_checked_tangent(
                f"field_diffraction_wedge_ad tangent_{name}",
                _ad_native_tangent_or_none(tangents[index]),
                scalar_shape,
            )
        vertex_tangents = []
        for index, name in (
            (21, "vertex_v0"),
            (22, "vertex_v1"),
            (23, "vertex_opp0"),
            (24, "vertex_opp1"),
        ):
            tangent = tangents[index] if index < len(tangents) else None
            vertex_tangents.append(
                _ad_native_tangent_or_none(
                    tangent if isinstance(tangent, torch.Tensor) else None
                )
            )
        tangent_frequency = _ad_frequency_tangent(tangents[20])
        if (
            tangent_source is None
            and tangent_target is None
            and tangent_frequency == 0.0
            and all(value is None for value in material_tangents.values())
            and all(value is None for value in vertex_tangents)
        ):
            return (None, None)
        with torch_compat.disable_functorch():
            out = _required_native_op("field_diffraction_wedge_jvp")(
                *(_ad_native_tensor(value) for value in primals),
                ctx.frequency_value,
                *(
                    _ad_native_tensor(value) if isinstance(value, torch.Tensor)
                    else value
                    for value in vertices
                ),
                tangent_source,
                tangent_target,
                material_tangents["face0_eps_r"],
                material_tangents["face0_sigma_e"],
                material_tangents["face0_gain"],
                material_tangents["face1_eps_r"],
                material_tangents["face1_sigma_e"],
                material_tangents["face1_gain"],
                float(tangent_frequency),
                *vertex_tangents,
            )
        return (out["tangent_field_vector"], out["tangent_direction"])


def field_diffraction_wedge_ad(
    source: torch.Tensor,
    target: torch.Tensor,
    edge_position: torch.Tensor,
    edge_direction: torch.Tensor,
    edge_t_min: torch.Tensor,
    edge_t_max: torch.Tensor,
    edge_n0: torch.Tensor,
    edge_n1: torch.Tensor,
    exterior_angle: torch.Tensor,
    face0_valid: torch.Tensor,
    face0_eps_r: torch.Tensor,
    face0_sigma_e: torch.Tensor,
    face0_mu_r: torch.Tensor,
    face0_gain: torch.Tensor,
    face1_valid: torch.Tensor,
    face1_eps_r: torch.Tensor,
    face1_sigma_e: torch.Tensor,
    face1_mu_r: torch.Tensor,
    face1_gain: torch.Tensor,
    tx_power: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
    vertices: tuple[torch.Tensor, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_diffraction_wedge`.

    ``vertices`` optionally supplies the winner edge vertices as
    ``(v0, v1, opp0, opp1, edge_boundary)`` per row; the kernel then rebuilds
    the edge tables from them so mesh-vertex gradients exist (plan 07 section
    9.3 mesh-vertex x diffraction). ``frequency_value`` optionally carries
    the precomputed host scalar of ``frequency`` (one read per solve at the
    seam, audit M3); when not supplied it is read here, exactly once per
    apply.
    """

    if vertices is not None and len(vertices) != 5:
        raise ValueError(
            "vertices must hold (v0, v1, opp0, opp1, edge_boundary) per row"
        )
    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    vertex_args = vertices if vertices is not None else (None,) * 5
    values = _FieldDiffractionWedgeAdFunction.apply(
        source,
        target,
        edge_position,
        edge_direction,
        edge_t_min,
        edge_t_max,
        edge_n0,
        edge_n1,
        exterior_angle,
        face0_valid,
        face0_eps_r,
        face0_sigma_e,
        face0_mu_r,
        face0_gain,
        face1_valid,
        face1_eps_r,
        face1_sigma_e,
        face1_mu_r,
        face1_gain,
        tx_power,
        frequency,
        *vertex_args,
        float(frequency_value),
    )
    return dict(zip(_WEDGE_OUTPUT_FIELDS, values, strict=True))


class _FieldProjectComplex3AdFunction(torch.autograd.Function):
    """Differentiable receiver projection on a frozen polarization basis."""

    @staticmethod
    def forward(field_vector, direction, rx_polarization):
        out = _required_native_op("field_project_complex3")(
            field_vector, direction, rx_polarization
        )
        return (out["coefficient"], out["path_gain"])

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal for value in inputs
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_coefficient, grad_path_gain):
        _ad_reject_fixed_inputs(
            "field_project_complex3_ad",
            ctx.needs_input_grad,
            ((2, "rx_polarization"),),
        )
        need_field = bool(ctx.needs_input_grad[0])
        need_direction = bool(ctx.needs_input_grad[1])
        if not (need_field or need_direction) or (
            grad_coefficient is None and grad_path_gain is None
        ):
            return (None, None, None)
        field_vector, direction, rx_polarization = ctx.saved_tensors
        out = _required_native_op("field_project_complex3_backward")(
            field_vector,
            direction,
            rx_polarization,
            grad_coefficient,
            grad_path_gain,
            need_field,
            need_direction,
        )
        return (
            out["grad_field_vector"] if need_field else None,
            out["grad_direction"] if need_direction else None,
            None,
        )

    @staticmethod
    def jvp(ctx, t_field_vector, t_direction, t_rx_polarization):
        _ad_reject_fixed_tangents(
            "field_project_complex3_ad",
            ((t_rx_polarization, "rx_polarization"),),
        )
        saved = ctx.saved_tensors
        tangent_field = _ad_native_tangent_or_none(t_field_vector)
        tangent_direction = _ad_geometry_tangent(
            "field_project_complex3_ad tangent_direction", t_direction, saved[1]
        )
        if tangent_field is None and tangent_direction is None:
            return (None, None)
        with torch_compat.disable_functorch():
            out = _required_native_op("field_project_complex3_jvp")(
                *(_ad_native_tensor(value) for value in saved),
                tangent_field,
                tangent_direction,
            )
        return (out["tangent_coefficient"], out["tangent_path_gain"])


def field_project_complex3_ad(
    field_vector: torch.Tensor,
    direction: torch.Tensor,
    rx_polarization: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_project_complex3` (field vector + direction)."""

    coefficient, path_gain = _FieldProjectComplex3AdFunction.apply(
        field_vector, direction, rx_polarization
    )
    return {"coefficient": coefficient, "path_gain": path_gain}


class _FieldCoupledRdAdFunction(torch.autograd.Function):
    """Fixed-topology differentiable coupled R-D transport (plan 07 AD-4).

    Differentiable inputs: eps_r / sigma_e / gain / thickness for the
    reflection wall and both wedge faces (12 scalars per path), frequency,
    and the continuous geometry (source, target, reflection_position,
    edge_position). The wedge axis/normals/exterior angle, the reflection
    normal (a frozen wall plane), mu_r, tx_power and the polarizations stay
    fixed. The pseudo-infinite edge truncation factor is a frozen regularizer
    of the differentiation (see kernels/field_wedge_ad.cu).
    """

    @staticmethod
    def forward(
        source,
        target,
        reflection_position,
        reflection_normal,
        edge_position,
        edge_direction,
        edge_n0,
        edge_n1,
        exterior_angle,
        tx_power,
        tx_polarization,
        rx_polarization,
        refl_eps_r,
        refl_sigma_e,
        refl_mu_r,
        refl_gain,
        refl_thickness,
        w0_eps_r,
        w0_sigma_e,
        w0_mu_r,
        w0_gain,
        w0_thickness,
        w1_eps_r,
        w1_sigma_e,
        w1_mu_r,
        w1_gain,
        w1_thickness,
        frequency,
        reverse,
        frequency_value,
    ):
        out = _required_native_op("field_coupled_rd")(
            source,
            target,
            reflection_position,
            reflection_normal,
            edge_position,
            edge_direction,
            edge_n0,
            edge_n1,
            exterior_angle,
            tx_power,
            tx_polarization,
            rx_polarization,
            refl_eps_r,
            refl_sigma_e,
            refl_mu_r,
            refl_gain,
            refl_thickness,
            w0_eps_r,
            w0_sigma_e,
            w0_mu_r,
            w0_gain,
            w0_thickness,
            w1_eps_r,
            w1_sigma_e,
            w1_mu_r,
            w1_gain,
            w1_thickness,
            frequency_value,
            bool(reverse),
        )
        return tuple(out[name] for name in _COUPLED_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        frequency = inputs[27]
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:27]
        )
        ctx.frequency_value = inputs[29]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.reverse = bool(inputs[28])
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        ctx.mark_non_differentiable(output[4])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx,
        grad_field_vector,
        grad_coefficient,
        grad_path_field,
        grad_path_gain,
        _grad_direction,
    ):
        none_grads = (None,) * 30
        _ad_reject_fixed_inputs(
            "field_coupled_rd_ad",
            ctx.needs_input_grad,
            (
                (3, "reflection_normal"),
                (5, "edge_direction"),
                (6, "edge_n0"),
                (7, "edge_n1"),
                (8, "exterior_angle"),
                (9, "tx_power"),
                (10, "tx_polarization"),
                (11, "rx_polarization"),
                (14, "reflection_mu_r"),
                (19, "wedge_mu_r0"),
                (24, "wedge_mu_r1"),
            ),
        )
        need_geometry = any(
            bool(ctx.needs_input_grad[index]) for index in (0, 1, 2, 4)
        )
        need_eps = any(bool(ctx.needs_input_grad[index]) for index in (12, 17, 22))
        need_sigma = any(bool(ctx.needs_input_grad[index]) for index in (13, 18, 23))
        need_gain = any(bool(ctx.needs_input_grad[index]) for index in (15, 20, 25))
        need_thickness = any(
            bool(ctx.needs_input_grad[index]) for index in (16, 21, 26)
        )
        need_frequency = bool(ctx.needs_input_grad[27])
        grads = (grad_field_vector, grad_coefficient, grad_path_field, grad_path_gain)
        if not (
            need_geometry
            or need_eps
            or need_sigma
            or need_gain
            or need_thickness
            or need_frequency
        ) or all(value is None for value in grads):
            return none_grads
        saved = ctx.saved_tensors
        out = _required_native_op("field_coupled_rd_backward")(
            *saved,
            ctx.frequency_value,
            ctx.reverse,
            grad_field_vector,
            grad_coefficient,
            grad_path_field,
            grad_path_gain,
            need_eps,
            need_sigma,
            need_gain,
            need_thickness,
            need_frequency,
            need_geometry,
        )
        grad_frequency = (
            _ad_frequency_grad(out["grad_frequency"], ctx.frequency_meta)
            if need_frequency
            else None
        )

        def material_column(name: str, column: int, index: int):
            if not ctx.needs_input_grad[index]:
                return None
            return out[name][:, column]

        return (
            out["grad_source"] if ctx.needs_input_grad[0] else None,
            out["grad_target"] if ctx.needs_input_grad[1] else None,
            out["grad_reflection_position"] if ctx.needs_input_grad[2] else None,
            None,
            out["grad_edge_position"] if ctx.needs_input_grad[4] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            material_column("grad_eps_r", 0, 12),
            material_column("grad_sigma_e", 0, 13),
            None,
            material_column("grad_gain", 0, 15),
            material_column("grad_thickness", 0, 16),
            material_column("grad_eps_r", 1, 17),
            material_column("grad_sigma_e", 1, 18),
            None,
            material_column("grad_gain", 1, 20),
            material_column("grad_thickness", 1, 21),
            material_column("grad_eps_r", 2, 22),
            material_column("grad_sigma_e", 2, 23),
            None,
            material_column("grad_gain", 2, 25),
            material_column("grad_thickness", 2, 26),
            grad_frequency,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        _ad_reject_fixed_tangents(
            "field_coupled_rd_ad",
            (
                (tangents[3], "reflection_normal"),
                (tangents[5], "edge_direction"),
                (tangents[6], "edge_n0"),
                (tangents[7], "edge_n1"),
                (tangents[8], "exterior_angle"),
                (tangents[9], "tx_power"),
                (tangents[10], "tx_polarization"),
                (tangents[11], "rx_polarization"),
                (tangents[14], "reflection_mu_r"),
                (tangents[19], "wedge_mu_r0"),
                (tangents[24], "wedge_mu_r1"),
            ),
        )
        saved = ctx.saved_tensors
        scalar_shape = tuple(saved[12].shape)
        tangent_source = _ad_geometry_tangent(
            "field_coupled_rd_ad tangent_source", tangents[0], saved[0]
        )
        tangent_target = _ad_geometry_tangent(
            "field_coupled_rd_ad tangent_target", tangents[1], saved[1]
        )
        tangent_hit = _ad_geometry_tangent(
            "field_coupled_rd_ad tangent_reflection_position",
            tangents[2],
            saved[2],
        )
        tangent_edge = _ad_geometry_tangent(
            "field_coupled_rd_ad tangent_edge_position", tangents[4], saved[4]
        )

        def material_pack(indices: tuple[int, int, int], name: str):
            columns = tuple(
                _ad_checked_tangent(
                    f"field_coupled_rd_ad tangent_{name}",
                    _ad_native_tangent_or_none(tangents[index]),
                    scalar_shape,
                )
                for index in indices
            )
            if all(column is None for column in columns):
                return None
            zero = torch.zeros(
                scalar_shape, device=saved[12].device, dtype=torch.float32
            )
            return torch.stack(
                [zero if column is None else column for column in columns], dim=1
            )

        tangent_eps = material_pack((12, 17, 22), "eps_r")
        tangent_sigma = material_pack((13, 18, 23), "sigma_e")
        tangent_gain = material_pack((15, 20, 25), "gain")
        tangent_thickness = material_pack((16, 21, 26), "thickness")
        tangent_frequency = _ad_frequency_tangent(tangents[27])
        if (
            tangent_source is None
            and tangent_target is None
            and tangent_hit is None
            and tangent_edge is None
            and tangent_eps is None
            and tangent_sigma is None
            and tangent_gain is None
            and tangent_thickness is None
            and tangent_frequency == 0.0
        ):
            return (None,) * 5
        with torch_compat.disable_functorch():
            out = _required_native_op("field_coupled_rd_jvp")(
                *(_ad_native_tensor(value) for value in saved),
                ctx.frequency_value,
                ctx.reverse,
                tangent_source,
                tangent_target,
                tangent_hit,
                tangent_edge,
                tangent_eps,
                tangent_sigma,
                tangent_gain,
                tangent_thickness,
                float(tangent_frequency),
            )
        return (
            out["tangent_field_vector"],
            out["tangent_coefficient"],
            out["tangent_path_field"],
            out["tangent_path_gain"],
            None,
        )


def field_coupled_rd_ad(
    source: torch.Tensor,
    target: torch.Tensor,
    reflection_position: torch.Tensor,
    reflection_normal: torch.Tensor,
    edge_position: torch.Tensor,
    edge_direction: torch.Tensor,
    edge_n0: torch.Tensor,
    edge_n1: torch.Tensor,
    exterior_angle: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_polarization: torch.Tensor,
    reflection_material: tuple[torch.Tensor, ...],
    wedge_material0: tuple[torch.Tensor, ...],
    wedge_material1: tuple[torch.Tensor, ...],
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
    reverse: bool,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`field_coupled_rd` (12 material scalars + frequency + geometry).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam, audit M3); when not
    supplied it is read here, exactly once per apply.
    """

    if any(
        len(bundle) != 5
        for bundle in (reflection_material, wedge_material0, wedge_material1)
    ):
        raise ValueError(
            "coupled material bundles must contain eps/sigma/mu/gain/thickness"
        )
    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _FieldCoupledRdAdFunction.apply(
        source,
        target,
        reflection_position,
        reflection_normal,
        edge_position,
        edge_direction,
        edge_n0,
        edge_n1,
        exterior_angle,
        tx_power,
        tx_polarization,
        rx_polarization,
        *reflection_material,
        *wedge_material0,
        *wedge_material1,
        frequency,
        bool(reverse),
        float(frequency_value),
    )
    return dict(zip(_COUPLED_OUTPUT_FIELDS, values, strict=True))


class _CoupledRdPrepareAdFunction(torch.autograd.Function):
    """Fixed-winner coupled stationary geometry (plan 07 AD-4).

    Re-solves the image source, the stationary diffraction point on the edge
    and the predicted wall crossing for the frozen winner (wall plane + edge
    line), so the coupled interaction points move with the endpoints on the
    autograd graph. Differentiable inputs: source and receiver.
    """

    @staticmethod
    def forward(
        source,
        receiver,
        plane_point,
        plane_normal,
        edge_pos,
        edge_dir,
        edge_t_min,
        edge_t_max,
    ):
        out = _required_native_op("coupled_rd_prepare")(
            source,
            receiver,
            plane_point,
            plane_normal,
            edge_pos,
            edge_dir,
            edge_t_min,
            edge_t_max,
        )
        active, edge_point, _virtual_source, reflection_point = out
        return (edge_point, reflection_point, active)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:7]
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)
        ctx.mark_non_differentiable(output[2])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_edge_point, grad_reflection_point, _grad_active):
        none_grads = (None,) * 8
        _ad_reject_fixed_inputs(
            "coupled_rd_prepare_ad",
            ctx.needs_input_grad,
            (
                (2, "plane_point"),
                (3, "plane_normal"),
                (4, "edge_pos"),
                (5, "edge_dir"),
                (6, "edge_t_min"),
                (7, "edge_t_max"),
            ),
        )
        need_source = bool(ctx.needs_input_grad[0])
        need_receiver = bool(ctx.needs_input_grad[1])
        if not (need_source or need_receiver) or (
            grad_edge_point is None and grad_reflection_point is None
        ):
            return none_grads
        out = _required_native_op("coupled_rd_prepare_backward")(
            *ctx.saved_tensors,
            grad_edge_point,
            grad_reflection_point,
            need_source,
            need_receiver,
        )
        return (
            out["grad_source"] if need_source else None,
            out["grad_receiver"] if need_receiver else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, t_source, t_receiver, *t_rest):
        _ad_reject_fixed_tangents(
            "coupled_rd_prepare_ad",
            (
                (t_rest[0], "plane_point"),
                (t_rest[1], "plane_normal"),
                (t_rest[2], "edge_pos"),
                (t_rest[3], "edge_dir"),
                (t_rest[4], "edge_t_min"),
                (t_rest[5], "edge_t_max"),
            ),
        )
        saved = ctx.saved_tensors
        tangent_source = _ad_geometry_tangent(
            "coupled_rd_prepare_ad tangent_source", t_source, saved[0]
        )
        tangent_receiver = _ad_geometry_tangent(
            "coupled_rd_prepare_ad tangent_receiver", t_receiver, saved[1]
        )
        if tangent_source is None and tangent_receiver is None:
            return (None, None, None)
        with torch_compat.disable_functorch():
            out = _required_native_op("coupled_rd_prepare_jvp")(
                *(_ad_native_tensor(value) for value in saved),
                tangent_source,
                tangent_receiver,
            )
        return (
            out["tangent_edge_point"],
            out["tangent_reflection_point"],
            None,
        )


def coupled_rd_prepare_ad(
    source: torch.Tensor,
    receiver: torch.Tensor,
    plane_point: torch.Tensor,
    plane_normal: torch.Tensor,
    edge_pos: torch.Tensor,
    edge_dir: torch.Tensor,
    edge_t_min: torch.Tensor,
    edge_t_max: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable fixed-winner coupled stationary geometry re-solve."""

    edge_point, reflection_point, active = _CoupledRdPrepareAdFunction.apply(
        source,
        receiver,
        plane_point,
        plane_normal,
        edge_pos,
        edge_dir,
        edge_t_min,
        edge_t_max,
    )
    return {
        "edge_point": edge_point,
        "reflection_point": reflection_point,
        "active": active,
    }


def path_los_export(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if tx_power.shape[0] != tx_positions.shape[0]:
        raise ValueError("tx_power must have one value per transmitter")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    exported = _required_native_op("path_los_export")(
        tx_positions, tx_power, rx_positions, float(frequency_hz)
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.path_los_export must return a dict")
    return exported


_PATH_BLOCK_SCHEMA: tuple[tuple[str, torch.dtype], ...] = (
    ("valid", torch.bool),
    ("tx_id", torch.int32),
    ("rx_id", torch.int32),
    ("depth", torch.int32),
    ("component_id", torch.int32),
    ("primitive_id", torch.int32),
    ("edge_id", torch.int32),
    ("path_length_m", torch.float32),
    ("delay_s", torch.float32),
    ("path_gain", torch.float32),
)


_DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA: tuple[tuple[str, torch.dtype], ...] = (
    ("path_field", torch.complex64),
    ("interaction_position", torch.float32),
    ("interaction_normal", torch.float32),
    ("material_id", torch.int32),
    ("primitive_sequence", torch.int32),
    ("material_sequence", torch.int32),
    ("interaction_positions", torch.float32),
    ("interaction_normals", torch.float32),
)


def _validate_path_block(name: str, block: dict[str, torch.Tensor]) -> None:
    if not isinstance(block, dict):
        raise TypeError(f"{name} must be a dict")
    expected_shape: tuple[int, ...] | None = None
    for key, dtype in _PATH_BLOCK_SCHEMA:
        tensor = block.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}.{key} must be a torch.Tensor")
        validate_cuda_tensor(f"{name}.{key}", tensor, dtype=dtype, ndim=1)
        if expected_shape is None:
            expected_shape = tuple(tensor.shape)
        elif tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name}.{key} must share the path count")


def _validate_deterministic_topology_block(
    name: str, block: dict[str, torch.Tensor], sequence_width: int
) -> None:
    _validate_path_block(name, block)
    _validate_topology_extra_fields(
        name,
        block,
        int(sequence_width),
        {key: True for key, _dtype in _DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA},
    )


def _validate_topology_extra_fields(
    name: str,
    block: dict[str, torch.Tensor],
    sequence_width: int,
    expected_presence: dict[str, bool],
) -> None:
    path_count = int(block["valid"].shape[0])
    for key, dtype in _DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA:
        present = key in block
        if present != expected_presence[key]:
            state = "include" if expected_presence[key] else "omit"
            raise TypeError(f"{name}.{key} must {state} the concat schema")
        if not present:
            continue
        tensor = block[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}.{key} must be a torch.Tensor")
        if key in {"interaction_position", "interaction_normal"}:
            validate_cuda_tensor(
                f"{name}.{key}", tensor, dtype=dtype, ndim=2, trailing_shape=(3,)
            )
            expected = (path_count, 3)
        elif key in {"primitive_sequence", "material_sequence"}:
            validate_cuda_tensor(f"{name}.{key}", tensor, dtype=dtype, ndim=2)
            expected = (path_count, sequence_width)
        elif key in {"interaction_positions", "interaction_normals"}:
            validate_cuda_tensor(f"{name}.{key}", tensor, dtype=dtype, ndim=3)
            expected = (path_count, sequence_width, 3)
        else:
            validate_cuda_tensor(f"{name}.{key}", tensor, dtype=dtype, ndim=1)
            expected = (path_count,)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name}.{key} must have shape {expected}")


def path_concat_vec3(
    blocks: tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> torch.Tensor:
    if not blocks:
        raise ValueError("blocks must not be empty")
    for index, block in enumerate(blocks):
        validate_cuda_tensor(
            f"blocks[{index}]", block, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
    out = _required_native_op("path_concat_vec3")(tuple(blocks))
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    return out


def deterministic_concat_topology_blocks(
    blocks: tuple[dict[str, torch.Tensor], ...] | list[dict[str, torch.Tensor]],
    *,
    sequence_width: int,
) -> dict[str, torch.Tensor]:
    if not blocks:
        raise ValueError("blocks must not be empty")
    if sequence_width < 0:
        raise ValueError("sequence_width must be non-negative")
    concat_presence = {
        key: key in blocks[0] for key, _dtype in _DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA
    }
    for index, block in enumerate(blocks):
        _validate_path_block(f"blocks[{index}]", block)
        _validate_topology_extra_fields(
            f"blocks[{index}]", block, int(sequence_width), concat_presence
        )
    exported = _required_native_op("deterministic_concat_topology_blocks")(
        tuple(blocks), int(sequence_width)
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.deterministic_concat_topology_blocks must return a dict"
        )
    _validate_path_block("deterministic_concat_topology_blocks", exported)
    _validate_topology_extra_fields(
        "deterministic_concat_topology_blocks",
        exported,
        int(sequence_width),
        concat_presence,
    )
    expected_count = sum(int(block["valid"].shape[0]) for block in blocks)
    if exported["valid"].shape != (expected_count,):
        raise ValueError(
            "_channel_native.deterministic_concat_topology_blocks returned bad path count"
        )
    return exported


def deterministic_gather_topology_block(
    block: dict[str, torch.Tensor],
    order: torch.Tensor,
    *,
    max_count: int,
    sequence_width: int,
) -> dict[str, torch.Tensor]:
    if sequence_width < 0:
        raise ValueError("sequence_width must be non-negative")
    if max_count < -1:
        raise ValueError("max_count must be -1 or non-negative")
    _validate_path_block("block", block)
    field_presence = {
        key: key in block for key, _dtype in _DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA
    }
    _validate_topology_extra_fields("block", block, int(sequence_width), field_presence)
    validate_cuda_tensor("order", order, dtype=torch.long, ndim=1)
    if order.get_device() != block["valid"].get_device():
        raise ValueError("order must share block device")

    exported = _required_native_op("deterministic_gather_topology_block")(
        block,
        order,
        int(max_count),
        int(sequence_width),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.deterministic_gather_topology_block must return a dict"
        )
    _validate_path_block("deterministic_gather_topology_block", exported)
    _validate_topology_extra_fields(
        "deterministic_gather_topology_block",
        exported,
        int(sequence_width),
        field_presence,
    )
    expected_count = (
        int(order.shape[0])
        if max_count < 0
        else min(int(order.shape[0]), int(max_count))
    )
    if exported["valid"].shape != (expected_count,):
        raise ValueError(
            "_channel_native.deterministic_gather_topology_block returned bad path count"
        )
    return exported


def path_los_visibility_inputs(
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_id", tx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("rx_id", rx_id, dtype=torch.int32, ndim=1)
    if rx_id.shape != tx_id.shape:
        raise ValueError("rx_id must match tx_id")
    exported = _required_native_op("path_los_visibility_inputs")(
        tx_positions, rx_positions, tx_id, rx_id
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.path_los_visibility_inputs must return a dict")
    validate_cuda_tensor(
        "start", exported["start"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "end", exported["end"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    if exported["active"].shape != tx_id.shape:
        raise ValueError(
            "_channel_native.path_los_visibility_inputs returned bad active shape"
        )
    return exported


def path_filter_los(
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    path_length_m: torch.Tensor,
    delay_s: torch.Tensor,
    path_gain: torch.Tensor,
    visible: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("tx_id", tx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("rx_id", rx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("path_length_m", path_length_m, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("delay_s", delay_s, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    for name, tensor in {
        "rx_id": rx_id,
        "path_length_m": path_length_m,
        "delay_s": delay_s,
        "path_gain": path_gain,
        "visible": visible,
    }.items():
        if tensor.shape != tx_id.shape:
            raise ValueError(f"{name} must match tx_id")
    block = _required_native_op("path_filter_los")(
        tx_id, rx_id, path_length_m, delay_s, path_gain, visible
    )
    _validate_path_block("path_filter_los", block)
    return block


def deterministic_los_topology_block(
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    path_length_m: torch.Tensor,
    delay_s: torch.Tensor,
    path_gain: torch.Tensor,
    visible: torch.Tensor | None,
    *,
    frequency_hz: float,
    sequence_width: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("tx_id", tx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("rx_id", rx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("path_length_m", path_length_m, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("delay_s", delay_s, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=1)
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    if sequence_width < 0:
        raise ValueError("sequence_width must be non-negative")
    for name, tensor in {
        "rx_id": rx_id,
        "path_length_m": path_length_m,
        "delay_s": delay_s,
        "path_gain": path_gain,
    }.items():
        if tensor.shape != tx_id.shape:
            raise ValueError(f"{name} must match tx_id")
    if visible is None:
        block = _required_native_op("deterministic_los_topology_block_all_visible")(
            tx_id,
            rx_id,
            path_length_m,
            delay_s,
            path_gain,
            float(frequency_hz),
            int(sequence_width),
        )
    else:
        validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
        if visible.shape != tx_id.shape:
            raise ValueError("visible must match tx_id")
        block = _required_native_op("deterministic_los_topology_block")(
            tx_id,
            rx_id,
            path_length_m,
            delay_s,
            path_gain,
            visible,
            float(frequency_hz),
            int(sequence_width),
        )
    if not isinstance(block, dict):
        raise TypeError(
            "_channel_native.deterministic_los_topology_block must return a dict"
        )
    _validate_deterministic_topology_block(
        "deterministic_los_topology_block", block, int(sequence_width)
    )
    return block


def deterministic_topology_default_fields(
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=1)
    exported = _required_native_op("deterministic_topology_default_fields")(reference)
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.deterministic_topology_default_fields must return a dict"
        )
    validate_cuda_tensor(
        "interaction_position",
        exported["interaction_position"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "interaction_normal",
        exported["interaction_normal"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "material_id", exported["material_id"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "path_field", exported["path_field"], dtype=torch.complex64, ndim=1
    )
    count = int(reference.shape[0])
    if (
        exported["interaction_position"].shape != (count, 3)
        or exported["interaction_normal"].shape != (count, 3)
        or exported["material_id"].shape != (count,)
        or exported["path_field"].shape != (count,)
    ):
        raise ValueError(
            "_channel_native.deterministic_topology_default_fields returned bad shape"
        )
    return exported


def deterministic_pad_topology_sequences(
    *,
    depth: torch.Tensor,
    primitive_id: torch.Tensor,
    material_id: torch.Tensor,
    interaction_position: torch.Tensor,
    interaction_normal: torch.Tensor,
    primitive_sequence: torch.Tensor,
    material_sequence: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    width: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("depth", depth, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("primitive_id", primitive_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("material_id", material_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "interaction_position",
        interaction_position,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "interaction_normal",
        interaction_normal,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "primitive_sequence", primitive_sequence, dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "material_sequence", material_sequence, dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "interaction_positions", interaction_positions, dtype=torch.float32, ndim=3
    )
    validate_cuda_tensor(
        "interaction_normals", interaction_normals, dtype=torch.float32, ndim=3
    )
    if width < 0:
        raise ValueError("width must be non-negative")
    count = int(depth.shape[0])
    for name, tensor in {
        "primitive_id": primitive_id,
        "material_id": material_id,
    }.items():
        if tensor.shape != (count,):
            raise ValueError(f"{name} must match depth")
    for name, tensor in {
        "interaction_position": interaction_position,
        "interaction_normal": interaction_normal,
    }.items():
        if tensor.shape != (count, 3):
            raise ValueError(f"{name} must have shape (N, 3)")
    for name, tensor in {
        "primitive_sequence": primitive_sequence,
        "material_sequence": material_sequence,
    }.items():
        if tensor.shape[0] != count:
            raise ValueError(f"{name} must share the path count")
    for name, tensor in {
        "interaction_positions": interaction_positions,
        "interaction_normals": interaction_normals,
    }.items():
        if tensor.shape[0] != count or tensor.shape[2] != 3:
            raise ValueError(f"{name} must have shape (N, D, 3)")
    exported = _required_native_op("deterministic_pad_topology_sequences")(
        depth,
        primitive_id,
        material_id,
        interaction_position,
        interaction_normal,
        primitive_sequence,
        material_sequence,
        interaction_positions,
        interaction_normals,
        int(width),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.deterministic_pad_topology_sequences must return a dict"
        )
    validate_cuda_tensor(
        "primitive_sequence", exported["primitive_sequence"], dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "material_sequence", exported["material_sequence"], dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "interaction_positions",
        exported["interaction_positions"],
        dtype=torch.float32,
        ndim=3,
    )
    validate_cuda_tensor(
        "interaction_normals",
        exported["interaction_normals"],
        dtype=torch.float32,
        ndim=3,
    )
    expected_i32 = (count, int(width))
    expected_vec = (count, int(width), 3)
    if (
        exported["primitive_sequence"].shape != expected_i32
        or exported["material_sequence"].shape != expected_i32
    ):
        raise ValueError(
            "_channel_native.deterministic_pad_topology_sequences returned bad sequence shape"
        )
    if (
        exported["interaction_positions"].shape != expected_vec
        or exported["interaction_normals"].shape != expected_vec
    ):
        raise ValueError(
            "_channel_native.deterministic_pad_topology_sequences returned bad interaction shape"
        )
    return exported


def deterministic_topology_base_fields(
    *,
    rx_id: torch.Tensor,
    path_length_m: torch.Tensor,
    delay_s: torch.Tensor,
    path_gain: torch.Tensor,
    tx_index: int,
    component_id: int,
    depth_source: torch.Tensor,
    depth_value: int,
    primitive_source: torch.Tensor,
    primitive_value: int,
    edge_source: torch.Tensor,
    edge_value: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("rx_id", rx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("path_length_m", path_length_m, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("delay_s", delay_s, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("depth_source", depth_source, dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "primitive_source", primitive_source, dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor("edge_source", edge_source, dtype=torch.int32, ndim=1)
    count = int(rx_id.shape[0])
    for name, tensor in {
        "path_length_m": path_length_m,
        "delay_s": delay_s,
        "path_gain": path_gain,
    }.items():
        if tensor.shape != (count,):
            raise ValueError(f"{name} must match rx_id")
    for name, tensor in {
        "depth_source": depth_source,
        "primitive_source": primitive_source,
        "edge_source": edge_source,
    }.items():
        if tensor.shape not in {(0,), (count,)}:
            raise ValueError(f"{name} must be empty or match rx_id")
    block = _required_native_op("deterministic_topology_base_fields")(
        rx_id,
        path_length_m,
        delay_s,
        path_gain,
        int(tx_index),
        int(component_id),
        depth_source,
        int(depth_value),
        primitive_source,
        int(primitive_value),
        edge_source,
        int(edge_value),
    )
    if not isinstance(block, dict):
        raise TypeError(
            "_channel_native.deterministic_topology_base_fields must return a dict"
        )
    _validate_path_block("deterministic_topology_base_fields", block)
    return block


def deterministic_repeat_range(
    reference: torch.Tensor, *, start: int, end: int, repeats: int
) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=1)
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    out = _required_native_op("deterministic_repeat_range")(
        reference, int(start), int(end), int(repeats)
    )
    validate_cuda_tensor("out", out, dtype=torch.int32, ndim=1)
    if out.shape != ((int(end) - int(start)) * int(repeats),):
        raise ValueError(
            "_channel_native.deterministic_repeat_range returned bad shape"
        )
    return out


def deterministic_face_anchor_points(
    vertices: torch.Tensor, faces: torch.Tensor
) -> torch.Tensor:
    validate_cuda_tensor(
        "vertices", vertices, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("faces", faces, dtype=torch.int32, ndim=2, trailing_shape=(3,))
    if vertices.get_device() != faces.get_device():
        raise ValueError("vertices and faces must share a CUDA device")
    out = _required_native_op("deterministic_face_anchor_points")(vertices, faces)
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.deterministic_face_anchor_points must return a tensor"
        )
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if out.shape != (faces.shape[0], 3):
        raise ValueError(
            "_channel_native.deterministic_face_anchor_points returned bad shape"
        )
    return out


def deterministic_reflection_epc_input_batch(
    *,
    tx: torch.Tensor,
    rx_positions: torch.Tensor,
    sequences: torch.Tensor,
    tri_a: torch.Tensor,
    normals: torch.Tensor,
    rx_start: int,
    rx_end: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("tx", tx, dtype=torch.float32, ndim=1)
    if tx.shape != (3,):
        raise ValueError("tx must have shape (3,)")
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "tri_a", tri_a, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "normals", normals, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if normals.shape != tri_a.shape:
        raise ValueError("normals must match tri_a")
    if sequences.dtype not in {torch.int32, torch.int64}:
        raise TypeError("sequences must have dtype torch.int32 or torch.int64")
    if not sequences.is_cuda:
        raise ValueError("sequences must be a CUDA tensor")
    if sequences.ndim != 2:
        raise ValueError("sequences must have shape (S, depth)")
    if not sequences.is_contiguous():
        raise ValueError("sequences must be contiguous")
    if rx_start < 0:
        raise ValueError("rx_start must be non-negative")
    if rx_end < rx_start:
        raise ValueError("rx_end must be greater than or equal to rx_start")
    if rx_end > int(rx_positions.shape[0]):
        raise ValueError("rx_end is out of range")
    exported = _required_native_op("deterministic_reflection_epc_input_batch")(
        tx,
        rx_positions,
        sequences,
        tri_a,
        normals,
        int(rx_start),
        int(rx_end),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.deterministic_reflection_epc_input_batch must return a dict"
        )
    expected_fields = {
        "tx_batch",
        "rx_batch",
        "rx_indices",
        "sequence_batch",
        "direct_plane_points",
        "direct_plane_normals",
    }
    if set(exported) != expected_fields:
        raise ValueError(
            "_channel_native.deterministic_reflection_epc_input_batch returned unexpected fields"
        )
    pair_count = (int(rx_end) - int(rx_start)) * int(sequences.shape[0])
    depth = int(sequences.shape[1])
    validate_cuda_tensor(
        "tx_batch",
        exported["tx_batch"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "rx_batch",
        exported["rx_batch"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "rx_indices", exported["rx_indices"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "sequence_batch", exported["sequence_batch"], dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "direct_plane_points",
        exported["direct_plane_points"],
        dtype=torch.float32,
        ndim=3,
    )
    validate_cuda_tensor(
        "direct_plane_normals",
        exported["direct_plane_normals"],
        dtype=torch.float32,
        ndim=3,
    )
    if exported["tx_batch"].shape != (pair_count, 3):
        raise ValueError(
            "_channel_native.deterministic_reflection_epc_input_batch returned bad tx_batch shape"
        )
    if exported["rx_batch"].shape != (pair_count, 3):
        raise ValueError(
            "_channel_native.deterministic_reflection_epc_input_batch returned bad rx_batch shape"
        )
    if exported["rx_indices"].shape != (pair_count,):
        raise ValueError(
            "_channel_native.deterministic_reflection_epc_input_batch returned bad rx_indices shape"
        )
    if exported["sequence_batch"].shape != (pair_count, depth):
        raise ValueError(
            "_channel_native.deterministic_reflection_epc_input_batch returned bad sequence_batch shape"
        )
    if exported["direct_plane_points"].shape != (pair_count, depth, 3):
        raise ValueError(
            "_channel_native.deterministic_reflection_epc_input_batch returned bad direct_plane_points shape"
        )
    if exported["direct_plane_normals"].shape != (pair_count, depth, 3):
        raise ValueError(
            "_channel_native.deterministic_reflection_epc_input_batch returned bad direct_plane_normals shape"
        )
    return exported


def deterministic_face_sequence_chunk(
    reference: torch.Tensor,
    *,
    face_count: int,
    depth: int,
    start: int,
    end: int,
    adjacent_distinct: bool = False,
) -> torch.Tensor:
    if not isinstance(reference, torch.Tensor):
        raise TypeError("reference must be a torch.Tensor")
    if not reference.is_cuda:
        raise ValueError("reference must be a CUDA tensor")
    if face_count <= 0:
        raise ValueError("face_count must be positive")
    if depth <= 0:
        raise ValueError("depth must be positive")
    if start < 0:
        raise ValueError("start must be non-negative")
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    out = _required_native_op("deterministic_face_sequence_chunk")(
        reference,
        int(face_count),
        int(depth),
        int(start),
        int(end),
        bool(adjacent_distinct),
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.deterministic_face_sequence_chunk must return a tensor"
        )
    validate_cuda_tensor("out", out, dtype=torch.int32, ndim=2)
    if out.shape != (int(end) - int(start), int(depth)):
        raise ValueError(
            "_channel_native.deterministic_face_sequence_chunk returned bad shape"
        )
    return out


def deterministic_mapped_face_sequence_chunk(
    face_ids: torch.Tensor,
    *,
    depth: int,
    start: int,
    end: int,
    adjacent_distinct: bool = False,
) -> torch.Tensor:
    if not isinstance(face_ids, torch.Tensor):
        raise TypeError("face_ids must be a torch.Tensor")
    if face_ids.dtype not in {torch.int32, torch.int64}:
        raise TypeError("face_ids must have dtype torch.int32 or torch.int64")
    if not face_ids.is_cuda:
        raise ValueError("face_ids must be a CUDA tensor")
    if face_ids.ndim != 1:
        raise ValueError("face_ids must be one-dimensional")
    if not face_ids.is_contiguous():
        raise ValueError("face_ids must be contiguous")
    if depth <= 0:
        raise ValueError("depth must be positive")
    if start < 0:
        raise ValueError("start must be non-negative")
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    out = _required_native_op("deterministic_mapped_face_sequence_chunk")(
        face_ids,
        int(depth),
        int(start),
        int(end),
        bool(adjacent_distinct),
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.deterministic_mapped_face_sequence_chunk must return a tensor"
        )
    validate_cuda_tensor("out", out, dtype=torch.int32, ndim=2)
    if out.shape != (int(end) - int(start), int(depth)):
        raise ValueError(
            "_channel_native.deterministic_mapped_face_sequence_chunk returned bad shape"
        )
    return out


def deterministic_reflection_order1_compact(
    *,
    visible: torch.Tensor,
    epc_faces: torch.Tensor,
    epc_hits: torch.Tensor,
    epc_normals: torch.Tensor,
    sequence_batch: torch.Tensor,
    rx_indices: torch.Tensor,
    tx: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    tx_index: int,
    face_eps_r: torch.Tensor,
    face_sigma_e: torch.Tensor,
    face_mu_r: torch.Tensor,
    face_gain: torch.Tensor,
    face_material_id: torch.Tensor,
    grouped_export: bool,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    if epc_faces.dtype not in {torch.int32, torch.int64}:
        raise TypeError("epc_faces must have dtype torch.int32 or torch.int64")
    if not epc_faces.is_cuda or epc_faces.ndim != 2 or not epc_faces.is_contiguous():
        raise ValueError(
            "epc_faces must be a contiguous CUDA tensor with shape (N, depth)"
        )
    validate_cuda_tensor("epc_hits", epc_hits, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("epc_normals", epc_normals, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("sequence_batch", sequence_batch, dtype=torch.int32, ndim=2)
    validate_cuda_tensor("rx_indices", rx_indices, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("tx", tx, dtype=torch.float32, ndim=1)
    if tx.shape != (3,):
        raise ValueError("tx must have shape (3,)")
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_eps_r", face_eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_sigma_e", face_sigma_e, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_mu_r", face_mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_gain", face_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "face_material_id", face_material_id, dtype=torch.int32, ndim=1
    )
    count = int(visible.shape[0])
    if epc_faces.shape[0] != count or epc_faces.shape[1] < 1:
        raise ValueError("epc_faces must match visible and include the first bounce")
    if epc_hits.shape[0] != count or epc_hits.shape[1] < 1 or epc_hits.shape[2] != 3:
        raise ValueError("epc_hits must have shape (N, depth, 3)")
    if epc_normals.shape != epc_hits.shape:
        raise ValueError("epc_normals must match epc_hits")
    if sequence_batch.shape[0] != count or sequence_batch.shape[1] < 1:
        raise ValueError(
            "sequence_batch must match visible and include the first bounce"
        )
    if rx_indices.shape != (count,):
        raise ValueError("rx_indices must match visible")
    if not 0 <= int(tx_index) < int(tx_power.shape[0]):
        raise ValueError("tx_index is out of range")
    if (
        face_sigma_e.shape != face_eps_r.shape
        or face_mu_r.shape != face_eps_r.shape
        or face_gain.shape != face_eps_r.shape
        or face_material_id.shape != face_eps_r.shape
    ):
        raise ValueError("face material tensors must share shape")
    exported = _required_native_op("deterministic_reflection_order1_compact")(
        visible,
        epc_faces,
        epc_hits,
        epc_normals,
        sequence_batch,
        rx_indices,
        tx,
        rx_positions,
        tx_power,
        int(tx_index),
        face_eps_r,
        face_sigma_e,
        face_mu_r,
        face_gain,
        face_material_id,
        bool(grouped_export),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.deterministic_reflection_order1_compact must return a dict"
        )
    expected_fields = {
        "selected_faces",
        "selected_points",
        "selected_normals",
        "selected_rx_id",
        "tx_keep",
        "rx_keep",
        "tx_power",
        "eps_r",
        "sigma_e",
        "mu_r",
        "gain",
        "material_id",
    }
    if set(exported) != expected_fields:
        raise ValueError(
            "_channel_native.deterministic_reflection_order1_compact returned unexpected fields"
        )
    selected_count = int(exported["selected_faces"].shape[0])
    validate_cuda_tensor(
        "selected_faces", exported["selected_faces"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "selected_points",
        exported["selected_points"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "selected_normals",
        exported["selected_normals"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "selected_rx_id", exported["selected_rx_id"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "tx_keep", exported["tx_keep"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "rx_keep", exported["rx_keep"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", exported["tx_power"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("eps_r", exported["eps_r"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("sigma_e", exported["sigma_e"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("mu_r", exported["mu_r"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("gain", exported["gain"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "material_id", exported["material_id"], dtype=torch.int32, ndim=1
    )
    for key in expected_fields - {"selected_faces"}:
        if exported[key].shape[0] != selected_count:
            raise ValueError(
                f"_channel_native.deterministic_reflection_order1_compact returned bad {key} shape"
            )
    return exported


def deterministic_reflection_sequence_compact(
    *,
    visible: torch.Tensor,
    epc_sequences: torch.Tensor,
    epc_hits: torch.Tensor,
    epc_normals: torch.Tensor,
    rx_indices: torch.Tensor,
    tx: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    tx_index: int,
    face_eps_r: torch.Tensor,
    face_sigma_e: torch.Tensor,
    face_mu_r: torch.Tensor,
    face_gain: torch.Tensor,
    face_material_id: torch.Tensor,
    max_count: int = -1,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    if epc_sequences.dtype not in {torch.int32, torch.int64}:
        raise TypeError("epc_sequences must have dtype torch.int32 or torch.int64")
    if (
        not epc_sequences.is_cuda
        or epc_sequences.ndim != 2
        or not epc_sequences.is_contiguous()
    ):
        raise ValueError(
            "epc_sequences must be a contiguous CUDA tensor with shape (N, depth)"
        )
    validate_cuda_tensor("epc_hits", epc_hits, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("epc_normals", epc_normals, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("rx_indices", rx_indices, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("tx", tx, dtype=torch.float32, ndim=1)
    if tx.shape != (3,):
        raise ValueError("tx must have shape (3,)")
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_eps_r", face_eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_sigma_e", face_sigma_e, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_mu_r", face_mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_gain", face_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "face_material_id", face_material_id, dtype=torch.int32, ndim=1
    )
    if max_count < -1:
        raise ValueError("max_count must be -1 or non-negative")
    count = int(visible.shape[0])
    depth = int(epc_sequences.shape[1])
    if depth <= 0 or epc_sequences.shape[0] != count:
        raise ValueError("epc_sequences must match visible and have positive depth")
    if epc_hits.shape != (count, depth, 3):
        raise ValueError("epc_hits must have shape (N, depth, 3)")
    if epc_normals.shape != epc_hits.shape:
        raise ValueError("epc_normals must match epc_hits")
    if rx_indices.shape != (count,):
        raise ValueError("rx_indices must match visible")
    if not 0 <= int(tx_index) < int(tx_power.shape[0]):
        raise ValueError("tx_index is out of range")
    if (
        face_sigma_e.shape != face_eps_r.shape
        or face_mu_r.shape != face_eps_r.shape
        or face_gain.shape != face_eps_r.shape
        or face_material_id.shape != face_eps_r.shape
    ):
        raise ValueError("face material tensors must share shape")
    exported = _required_native_op("deterministic_reflection_sequence_compact")(
        visible,
        epc_sequences,
        epc_hits,
        epc_normals,
        rx_indices,
        tx,
        rx_positions,
        tx_power,
        int(tx_index),
        face_eps_r,
        face_sigma_e,
        face_mu_r,
        face_gain,
        face_material_id,
        int(max_count),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.deterministic_reflection_sequence_compact must return a dict"
        )
    expected_fields = {
        "selected_sequences",
        "selected_hits",
        "selected_normals",
        "selected_rx_id",
        "selected_tx",
        "selected_rx",
        "tx_power",
        "eps_r",
        "sigma_e",
        "mu_r",
        "gain",
        "first_face",
        "material_id",
        "material_sequence",
        "first_hit",
        "first_normal",
    }
    if set(exported) != expected_fields:
        raise ValueError(
            "_channel_native.deterministic_reflection_sequence_compact returned unexpected fields"
        )
    selected_count = int(exported["selected_sequences"].shape[0])
    validate_cuda_tensor(
        "selected_sequences", exported["selected_sequences"], dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "selected_hits", exported["selected_hits"], dtype=torch.float32, ndim=3
    )
    validate_cuda_tensor(
        "selected_normals", exported["selected_normals"], dtype=torch.float32, ndim=3
    )
    validate_cuda_tensor(
        "selected_rx_id", exported["selected_rx_id"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "selected_tx",
        exported["selected_tx"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "selected_rx",
        exported["selected_rx"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor("tx_power", exported["tx_power"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("eps_r", exported["eps_r"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("sigma_e", exported["sigma_e"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("mu_r", exported["mu_r"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("gain", exported["gain"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor(
        "first_face", exported["first_face"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "material_id", exported["material_id"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "material_sequence", exported["material_sequence"], dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "first_hit",
        exported["first_hit"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "first_normal",
        exported["first_normal"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    expected_shapes = {
        "selected_sequences": (selected_count, depth),
        "selected_hits": (selected_count, depth, 3),
        "selected_normals": (selected_count, depth, 3),
        "selected_rx_id": (selected_count,),
        "selected_tx": (selected_count, 3),
        "selected_rx": (selected_count, 3),
        "tx_power": (selected_count,),
        "eps_r": (selected_count, depth),
        "sigma_e": (selected_count, depth),
        "mu_r": (selected_count, depth),
        "gain": (selected_count, depth),
        "first_face": (selected_count,),
        "material_id": (selected_count,),
        "material_sequence": (selected_count, depth),
        "first_hit": (selected_count, 3),
        "first_normal": (selected_count, 3),
    }
    for key, shape in expected_shapes.items():
        if tuple(exported[key].shape) != shape:
            raise ValueError(
                f"_channel_native.deterministic_reflection_sequence_compact returned bad {key} shape"
            )
    return exported


def deterministic_diffraction_order1_compact(
    *,
    valid: torch.Tensor,
    rx_id: torch.Tensor,
    depth: torch.Tensor,
    edge_id: torch.Tensor,
    delay_s: torch.Tensor,
    x_re: torch.Tensor,
    x_im: torch.Tensor,
    y_re: torch.Tensor,
    y_im: torch.Tensor,
    z_re: torch.Tensor,
    z_im: torch.Tensor,
    interaction_position: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("valid", valid, dtype=torch.bool, ndim=1)
    validate_cuda_tensor("rx_id", rx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("depth", depth, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("edge_id", edge_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("delay_s", delay_s, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("x_re", x_re, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("x_im", x_im, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("y_re", y_re, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("y_im", y_im, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("z_re", z_re, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("z_im", z_im, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "interaction_position",
        interaction_position,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    count = int(valid.shape[0])
    for name, tensor in {
        "rx_id": rx_id,
        "depth": depth,
        "edge_id": edge_id,
        "delay_s": delay_s,
        "x_re": x_re,
        "x_im": x_im,
        "y_re": y_re,
        "y_im": y_im,
        "z_re": z_re,
        "z_im": z_im,
    }.items():
        if tensor.shape != (count,):
            raise ValueError(f"{name} must match valid")
    if interaction_position.shape != (count, 3):
        raise ValueError("interaction_position must match valid")

    exported = _required_native_op("deterministic_diffraction_order1_compact")(
        valid,
        rx_id,
        depth,
        edge_id,
        delay_s,
        x_re,
        x_im,
        y_re,
        y_im,
        z_re,
        z_im,
        interaction_position,
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.deterministic_diffraction_order1_compact must return a dict"
        )
    expected_fields = {
        "rx_id",
        "depth",
        "edge_id",
        "delay_s",
        "x_re",
        "x_im",
        "y_re",
        "y_im",
        "z_re",
        "z_im",
        "interaction_position",
    }
    if set(exported) != expected_fields:
        raise ValueError(
            "_channel_native.deterministic_diffraction_order1_compact returned unexpected fields"
        )
    selected_count = int(exported["rx_id"].shape[0])
    validate_cuda_tensor("rx_id", exported["rx_id"], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("depth", exported["depth"], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("edge_id", exported["edge_id"], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("delay_s", exported["delay_s"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("x_re", exported["x_re"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("x_im", exported["x_im"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("y_re", exported["y_re"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("y_im", exported["y_im"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("z_re", exported["z_re"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("z_im", exported["z_im"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "interaction_position",
        exported["interaction_position"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    for key in expected_fields - {"interaction_position"}:
        if exported[key].shape != (selected_count,):
            raise ValueError(
                f"_channel_native.deterministic_diffraction_order1_compact returned bad {key} shape"
            )
    if exported["interaction_position"].shape != (selected_count, 3):
        raise ValueError(
            "_channel_native.deterministic_diffraction_order1_compact returned bad interaction_position shape"
        )
    return exported


def deterministic_sort_order(
    valid: torch.Tensor,
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    depth: torch.Tensor,
    component_id: torch.Tensor,
    primitive_id: torch.Tensor,
    edge_id: torch.Tensor,
    primitive_sequence: torch.Tensor,
) -> torch.Tensor:
    validate_cuda_tensor("valid", valid, dtype=torch.bool, ndim=1)
    for name, tensor in (
        ("tx_id", tx_id),
        ("rx_id", rx_id),
        ("depth", depth),
        ("component_id", component_id),
        ("primitive_id", primitive_id),
        ("edge_id", edge_id),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.int32, ndim=1)
        if tensor.shape != valid.shape:
            raise ValueError(f"{name} must match valid")
    validate_cuda_tensor(
        "primitive_sequence", primitive_sequence, dtype=torch.int32, ndim=2
    )
    if primitive_sequence.shape[0] != valid.shape[0]:
        raise ValueError("primitive_sequence must match valid rows")
    out = _required_native_op("deterministic_sort_order")(
        valid,
        tx_id,
        rx_id,
        depth,
        component_id,
        primitive_id,
        edge_id,
        primitive_sequence,
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel_native.deterministic_sort_order must return a tensor")
    validate_cuda_tensor("out", out, dtype=torch.long, ndim=1)
    if out.shape != valid.shape:
        raise ValueError("_channel_native.deterministic_sort_order returned bad shape")
    return out


def _validate_path_reflection_candidates(
    name: str, candidates: dict[str, torch.Tensor]
) -> None:
    _validate_path_block(name, candidates)
    path_count = candidates["valid"].shape
    for key in ("seg0_start", "seg0_end", "seg1_start", "seg1_end"):
        tensor = candidates.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}.{key} must be a torch.Tensor")
        validate_cuda_tensor(
            f"{name}.{key}", tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
        if tensor.shape[0] != path_count[0]:
            raise ValueError(f"{name}.{key} must share the path count")
    active = candidates.get("active")
    if not isinstance(active, torch.Tensor):
        raise TypeError(f"{name}.active must be a torch.Tensor")
    validate_cuda_tensor(f"{name}.active", active, dtype=torch.bool, ndim=1)
    if tuple(active.shape) != path_count:
        raise ValueError(f"{name}.active must share the path count")


def path_reflection_candidates(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_normals: torch.Tensor,
    face_gain: torch.Tensor,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "vertices", vertices, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("faces", faces, dtype=torch.int32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor(
        "face_normals", face_normals, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("face_gain", face_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if face_normals.shape[0] != faces.shape[0] or face_gain.shape[0] != faces.shape[0]:
        raise ValueError("face_normals and face_gain must match faces")
    if tx_power.shape[0] != tx_positions.shape[0]:
        raise ValueError("tx_power must match tx_positions")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    candidates = _required_native_op("path_reflection_candidates")(
        vertices,
        faces,
        face_normals,
        face_gain,
        tx_positions,
        tx_power,
        rx_positions,
        float(frequency_hz),
    )
    if not isinstance(candidates, dict):
        raise TypeError("_channel_native.path_reflection_candidates must return a dict")
    _validate_path_reflection_candidates("path_reflection_candidates", candidates)
    return candidates


def path_filter_block(
    block: dict[str, torch.Tensor],
    visible0: torch.Tensor,
    visible1: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _validate_path_block("block", block)
    validate_cuda_tensor("visible0", visible0, dtype=torch.bool, ndim=1)
    validate_cuda_tensor("visible1", visible1, dtype=torch.bool, ndim=1)
    if visible0.shape != block["valid"].shape or visible1.shape != block["valid"].shape:
        raise ValueError("visible masks must share the path count")
    out = _required_native_op("path_filter_block")(block, visible0, visible1)
    if not isinstance(out, dict):
        raise TypeError("_channel_native.path_filter_block must return a dict")
    _validate_path_block("path_filter_block", out)
    return out


def path_diffraction_block(
    raydn_output: tuple[torch.Tensor, ...],
    *,
    tx_index: int,
) -> dict[str, torch.Tensor]:
    if not isinstance(raydn_output, tuple) or len(raydn_output) != 18:
        raise TypeError(
            "raydn_output must be the 18-tensor RayDN diffraction path tuple"
        )
    for index in (1, 3, 4, 5):
        validate_cuda_tensor(
            f"raydn_output[{index}]",
            raydn_output[index],
            dtype=torch.int32 if index != 1 else torch.bool,
            ndim=1,
        )
    for index in (8, 9, 10, 11, 12, 13, 14):
        validate_cuda_tensor(
            f"raydn_output[{index}]", raydn_output[index], dtype=torch.float32, ndim=1
        )
    capacity = raydn_output[1].shape
    for index in (3, 4, 5, 8, 9, 10, 11, 12, 13, 14):
        if raydn_output[index].shape != capacity:
            raise ValueError("RayDN diffraction path tensors must share capacity")
    if tx_index < 0:
        raise ValueError("tx_index must be non-negative")
    out = _required_native_op("path_diffraction_block")(raydn_output, int(tx_index))
    if not isinstance(out, dict):
        raise TypeError("_channel_native.path_diffraction_block must return a dict")
    _validate_path_block("path_diffraction_block", out)
    return out


def path_diffraction_paths_order1(
    scene_handle: object,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    edge_geometry: tuple[torch.Tensor, ...],
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_mu_r: torch.Tensor,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    *,
    wavelength: float,
) -> dict[str, torch.Tensor]:
    if len(edge_geometry) != 11:
        raise TypeError(
            "edge_geometry must contain the 11-tensor diffraction edge geometry tuple"
        )
    (
        selected,
        edge_pos,
        edge_dir,
        _lengths,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
    ) = edge_geometry
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("selected", selected, dtype=torch.bool, ndim=1)
    validate_cuda_tensor(
        "edge_pos", edge_pos, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "edge_dir", edge_dir, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("line_min", line_min, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("line_max", line_max, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("n0", n0, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("n1", n1, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("face0", face0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face1", face1, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("exterior_angle", exterior_angle, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_eta_r", material_eta_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_sigma", material_sigma, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_mu_r", material_mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_gain", material_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("material_valid", material_valid, dtype=torch.bool, ndim=1)
    if tx_power.shape[0] != tx_positions.shape[0]:
        raise ValueError("tx_power must match tx_positions")
    if wavelength <= 0.0:
        raise ValueError("wavelength must be positive")
    out = _required_native_op("path_diffraction_paths_order1")(
        _raydn_scene_handle_id(scene_handle),
        tx_positions,
        tx_power,
        rx_positions,
        selected,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        material_eta_r,
        material_sigma,
        material_mu_r,
        material_gain,
        material_valid,
        float(wavelength),
        _raydn_module_handle(),
    )
    if not isinstance(out, dict):
        raise TypeError(
            "_channel_native.path_diffraction_paths_order1 must return a dict"
        )
    _validate_path_block("path_diffraction_paths_order1", out)
    return out


def path_merge_blocks(
    blocks: tuple[dict[str, torch.Tensor], ...] | list[dict[str, torch.Tensor]],
    *,
    tx_count: int,
    max_depth: int,
) -> dict[str, torch.Tensor]:
    if not blocks:
        raise ValueError("blocks must not be empty")
    for index, block in enumerate(blocks):
        _validate_path_block(f"blocks[{index}]", block)
    if tx_count < 0:
        raise ValueError("tx_count must be non-negative")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    out = _required_native_op("path_merge_blocks")(
        tuple(blocks), int(tx_count), int(max_depth)
    )
    if not isinstance(out, dict):
        raise TypeError("_channel_native.path_merge_blocks must return a dict")
    _validate_path_block("path_merge_blocks", out)
    return out


def path_finalize_blocks(
    los: dict[str, torch.Tensor],
    reflection: dict[str, torch.Tensor],
    diffraction: dict[str, torch.Tensor],
    *,
    max_paths: int | None,
    tx_count: int,
    max_depth: int,
) -> dict[str, torch.Tensor]:
    _validate_path_block("los", los)
    _validate_path_block("reflection", reflection)
    _validate_path_block("diffraction", diffraction)
    if tx_count < 0:
        raise ValueError("tx_count must be non-negative")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    max_paths_value = -1 if max_paths is None else int(max_paths)
    if max_paths_value < -1:
        raise ValueError("max_paths must be positive")
    finalized = _required_native_op("path_finalize_blocks")(
        los,
        reflection,
        diffraction,
        max_paths_value,
        int(tx_count),
        int(max_depth),
    )
    _validate_path_block("path_finalize_blocks", finalized)
    return finalized


def mc_los_path_gain_backward(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    frequency_hz: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if not isinstance(grad_output, torch.Tensor):
        raise TypeError("grad_output must be a torch.Tensor")
    if grad_output.dtype != torch.float32:
        raise TypeError("grad_output must have dtype torch.float32")
    if not grad_output.is_cuda:
        raise ValueError("grad_output must be a CUDA tensor")
    if grad_output.ndim != 2:
        raise ValueError("grad_output must have 2 dimensions")
    if grad_output.shape != (tx_positions.shape[0], rx_positions.shape[0]):
        raise ValueError("grad_output must match the LoS path-gain matrix shape")
    if tx_power.shape[0] != tx_positions.shape[0]:
        raise ValueError("tx_power must have one value per transmitter")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    native = native_extension()
    if native is None or not hasattr(native, "mc_los_path_gain_backward"):
        raise RuntimeError(
            "_channel_native.mc_los_path_gain_backward CUDA kernel is required"
        )
    gradients = native.mc_los_path_gain_backward(
        tx_positions,
        tx_power,
        rx_positions,
        grad_output,
        float(frequency_hz),
    )
    if not isinstance(gradients, tuple) or len(gradients) != 4:
        raise TypeError(
            "_channel_native.mc_los_path_gain_backward must return 4 tensors"
        )
    validate_cuda_tensor(
        "grad_tx", gradients[0], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("grad_power", gradients[1], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "grad_rx", gradients[2], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "grad_frequency", gradients[3], dtype=torch.float32, ndim=1
    )
    if gradients[0].shape != tx_positions.shape:
        raise ValueError(
            "_channel_native.mc_los_path_gain_backward returned bad grad_tx shape"
        )
    if gradients[1].shape != tx_power.shape:
        raise ValueError(
            "_channel_native.mc_los_path_gain_backward returned bad grad_power shape"
        )
    if gradients[2].shape != rx_positions.shape:
        raise ValueError(
            "_channel_native.mc_los_path_gain_backward returned bad grad_rx shape"
        )
    if gradients[3].shape != (1,):
        raise ValueError(
            "_channel_native.mc_los_path_gain_backward returned bad grad_frequency shape"
        )
    return gradients


def mc_los_path_gain_jvp(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_tangent: torch.Tensor,
    power_tangent: torch.Tensor,
    rx_tangent: torch.Tensor,
    has_tx_tangent: bool,
    has_power_tangent: bool,
    has_rx_tangent: bool,
    *,
    frequency_hz: float,
    frequency_tangent: float = 0.0,
) -> torch.Tensor:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if tx_power.shape[0] != tx_positions.shape[0]:
        raise ValueError("tx_power must have one value per transmitter")
    if has_tx_tangent:
        validate_cuda_tensor(
            "tx_tangent", tx_tangent, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
        if tx_tangent.shape != tx_positions.shape:
            raise ValueError("tx_tangent must match tx_positions")
    if has_power_tangent:
        validate_cuda_tensor(
            "power_tangent", power_tangent, dtype=torch.float32, ndim=1
        )
        if power_tangent.shape != tx_power.shape:
            raise ValueError("power_tangent must match tx_power")
    if has_rx_tangent:
        validate_cuda_tensor(
            "rx_tangent", rx_tangent, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
        if rx_tangent.shape != rx_positions.shape:
            raise ValueError("rx_tangent must match rx_positions")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    native = native_extension()
    if native is None or not hasattr(native, "mc_los_path_gain_jvp"):
        raise RuntimeError(
            "_channel_native.mc_los_path_gain_jvp CUDA kernel is required"
        )
    out = native.mc_los_path_gain_jvp(
        tx_positions,
        tx_power,
        rx_positions,
        tx_tangent,
        power_tangent,
        rx_tangent,
        bool(has_tx_tangent),
        bool(has_power_tangent),
        bool(has_rx_tangent),
        float(frequency_hz),
        float(frequency_tangent),
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel_native.mc_los_path_gain_jvp must return a tensor")
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2)
    if out.shape != (tx_positions.shape[0], rx_positions.shape[0]):
        raise ValueError(
            "_channel_native.mc_los_path_gain_jvp returned an unexpected shape"
        )
    return out


def mc_finalize_component_maps(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("reflection", reflection, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("diffraction", diffraction, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("transmission", transmission, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("scattering", scattering, dtype=torch.float32, ndim=3)
    if reflection.shape != los.shape:
        raise ValueError("reflection must match los shape")
    if diffraction.shape != los.shape:
        raise ValueError("diffraction must match los shape")
    if transmission.shape != los.shape:
        raise ValueError("transmission must match los shape")
    if scattering.shape != los.shape:
        raise ValueError("scattering must match los shape")

    native = native_extension()
    if native is None or not hasattr(native, "mc_finalize_component_maps"):
        raise RuntimeError(
            "_channel_native.mc_finalize_component_maps CUDA kernel is required"
        )
    exported = native.mc_finalize_component_maps(
        los, reflection, diffraction, transmission, scattering
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.mc_finalize_component_maps must return a dict")
    return exported


_MC_FINALIZE_FIELDS = (
    "path_gain",
    "los_power",
    "reflection_power",
    "diffraction_power",
    "transmission_power",
    "scattering_power",
)


class _McFinalizeComponentMapsAdFunction(torch.autograd.Function):
    """Differentiable component-map finalization (plan 07 AD-3).

    The finalize kernel is a purely linear elementwise sum plus per-component
    power reductions, so the map cotangent is the path_gain cotangent viewed
    back to map layout plus the broadcast power cotangent, and the
    pushforward is the finalize kernel itself applied to the tangents.
    """

    @staticmethod
    def forward(los, reflection, diffraction, transmission, scattering):
        out = mc_finalize_component_maps(
            los, reflection, diffraction, transmission, scattering
        )
        return tuple(out[name] for name in _MC_FINALIZE_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        ctx.map_shape = tuple(inputs[0].shape)
        primal = torch.autograd.forward_ad.unpack_dual(inputs[0]).primal
        ctx.save_for_forward(primal)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_path_gain, *grad_powers):
        tx_count, dim0, dim1 = ctx.map_shape
        base = (
            grad_path_gain.unflatten(1, (dim0, dim1))
            if grad_path_gain is not None
            else None
        )
        grads = []
        for index in range(5):
            if not ctx.needs_input_grad[index]:
                grads.append(None)
                continue
            grad_power = grad_powers[index]
            if grad_power is None:
                grads.append(base)
            elif base is None:
                grads.append(grad_power.expand(ctx.map_shape))
            else:
                grads.append(base + grad_power)
        return tuple(grads)

    @staticmethod
    def jvp(ctx, t_los, t_reflection, t_diffraction, t_transmission, t_scattering):
        tangents = [
            _ad_native_tangent_or_none(value) for value in
            (t_los, t_reflection, t_diffraction, t_transmission, t_scattering)
        ]
        if all(value is None for value in tangents):
            return (None,) * len(_MC_FINALIZE_FIELDS)
        (reference,) = ctx.saved_tensors
        zero = None
        filled = []
        for value in tangents:
            if value is None:
                if zero is None:
                    zero = torch.zeros_like(_ad_native_tensor(reference))
                value = zero
            filled.append(value)
        with torch_compat.disable_functorch():
            out = mc_finalize_component_maps(*filled)
        return tuple(out[name] for name in _MC_FINALIZE_FIELDS)


def mc_finalize_component_maps_ad(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`mc_finalize_component_maps`."""

    values = _McFinalizeComponentMapsAdFunction.apply(
        los, reflection, diffraction, transmission, scattering
    )
    return dict(zip(_MC_FINALIZE_FIELDS, values, strict=True))


def mc_los_component_maps_adjoint(
    grad_maps: torch.Tensor,
    visible: torch.Tensor | None,
) -> torch.Tensor:
    """Adjoint of the (visibility-masked) LoS component-map layout."""

    if not isinstance(grad_maps, torch.Tensor):
        raise TypeError("grad_maps must be a torch.Tensor")
    if grad_maps.dtype != torch.float32 or not grad_maps.is_cuda:
        raise TypeError("grad_maps must be a float32 CUDA tensor")
    if grad_maps.ndim != 3:
        raise ValueError("grad_maps must have 3 dimensions")
    if visible is None:
        visible = grad_maps.new_empty((0, 0), dtype=torch.bool)
    out = _required_native_op("mc_los_component_maps_adjoint")(grad_maps, visible)
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.mc_los_component_maps_adjoint must return a tensor"
        )
    return out


class _McLosGridMapsAdFunction(torch.autograd.Function):
    """Differentiable LoS/transmission component-map layout (plan 07 AD-3).

    The forward is the primal layout kernel plus the per-tx visibility mask
    application, a permutation times a frozen 0/1 mask of the (tx, cells)
    matrix; its adjoint is one masked gather kernel and its pushforward is
    the forward itself on the tangent matrix.
    """

    @staticmethod
    def forward(matrix, visible, rows, cols):
        maps = mc_los_component_maps_from_matrix(matrix, rows=rows, cols=cols)
        if visible is not None:
            for tx_index in range(int(matrix.shape[0])):
                mc_apply_los_visibility(
                    maps, matrix, visible[tx_index], tx_index=tx_index
                )
        return maps

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        matrix, visible, rows, cols = inputs
        ctx.rows = int(rows)
        ctx.cols = int(cols)
        ctx.has_visible = visible is not None
        if visible is not None:
            ctx.save_for_backward(visible)
            ctx.save_for_forward(visible)
        else:
            ctx.save_for_backward()
            ctx.save_for_forward()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_maps):
        if grad_maps is None or not ctx.needs_input_grad[0]:
            return None, None, None, None
        visible = ctx.saved_tensors[0] if ctx.has_visible else None
        return (
            mc_los_component_maps_adjoint(grad_maps, visible),
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, t_matrix, _t_visible, _t_rows, _t_cols):
        tangent = _ad_native_tangent_or_none(t_matrix)
        if tangent is None:
            return None
        visible = ctx.saved_tensors[0] if ctx.has_visible else None
        with torch_compat.disable_functorch():
            maps = mc_los_component_maps_from_matrix(
                tangent, rows=ctx.rows, cols=ctx.cols
            )
            if visible is not None:
                visible = _ad_native_tensor(visible)
                for tx_index in range(int(tangent.shape[0])):
                    mc_apply_los_visibility(
                        maps, tangent, visible[tx_index], tx_index=tx_index
                    )
        return maps


def mc_los_grid_maps_ad(
    matrix: torch.Tensor,
    visible: torch.Tensor | None,
    *,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Differentiable grid component maps from a (tx, cells) matrix."""

    return _McLosGridMapsAdFunction.apply(matrix, visible, int(rows), int(cols))


def mc_zero_matrix(reference: torch.Tensor, *, rows: int, cols: int) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    if rows < 0 or cols < 0:
        raise ValueError("rows and cols must be non-negative")
    out = _required_native_op("mc_zero_matrix")(reference, int(rows), int(cols))
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel_native.mc_zero_matrix must return a tensor")
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2)
    if out.shape != (int(rows), int(cols)):
        raise ValueError("_channel_native.mc_zero_matrix returned an unexpected shape")
    return out


def mc_point_component_power(
    path_gain: torch.Tensor, *, include_los: bool
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=2)
    exported = _required_native_op("mc_point_component_power")(
        path_gain, bool(include_los)
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.mc_point_component_power must return a dict")
    for name in ("los", "reflection", "diffraction"):
        validate_cuda_tensor(name, exported[name], dtype=torch.float32, ndim=0)
    return exported


def mc_component_map_buffer(
    reference: torch.Tensor,
    *,
    tx_count: int,
    dim0: int,
    dim1: int,
) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    if tx_count < 0 or dim0 < 0 or dim1 < 0:
        raise ValueError("tx_count, dim0, and dim1 must be non-negative")
    native = native_extension()
    if native is None or not hasattr(native, "mc_component_map_buffer"):
        raise RuntimeError(
            "_channel_native.mc_component_map_buffer CUDA kernel is required"
        )
    maps = native.mc_component_map_buffer(
        reference, int(tx_count), int(dim0), int(dim1)
    )
    if not isinstance(maps, torch.Tensor):
        raise TypeError("_channel_native.mc_component_map_buffer must return a tensor")
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    if maps.shape != (tx_count, dim0, dim1):
        raise ValueError(
            "_channel_native.mc_component_map_buffer returned an unexpected shape"
        )
    return maps


def mc_store_component_map(
    maps: torch.Tensor,
    source: torch.Tensor,
    *,
    tx_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("source", source, dtype=torch.float32, ndim=2)
    if source.shape != maps.shape[1:]:
        raise ValueError("source shape must match one maps slot")
    native = native_extension()
    if native is None or not hasattr(native, "mc_store_component_map"):
        raise RuntimeError(
            "_channel_native.mc_store_component_map CUDA kernel is required"
        )
    out = native.mc_store_component_map(maps, source, int(tx_index))
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel_native.mc_store_component_map must return a tensor")
    validate_cuda_tensor("maps", out, dtype=torch.float32, ndim=3)
    return out


def mc_store_scaled_component_map(
    maps: torch.Tensor,
    source: torch.Tensor,
    scale_values: torch.Tensor,
    *,
    tx_index: int,
    scale_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("source", source, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("scale_values", scale_values, dtype=torch.float32, ndim=1)
    if source.shape != maps.shape[1:]:
        raise ValueError("source shape must match one maps slot")
    native = native_extension()
    if native is None or not hasattr(native, "mc_store_scaled_component_map"):
        raise RuntimeError(
            "_channel_native.mc_store_scaled_component_map CUDA kernel is required"
        )
    out = native.mc_store_scaled_component_map(
        maps,
        source,
        scale_values,
        int(tx_index),
        int(scale_index),
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel_native.mc_store_scaled_component_map must return a tensor"
        )
    validate_cuda_tensor("maps", out, dtype=torch.float32, ndim=3)
    return out


def mc_sample_directions(count: int, reference: torch.Tensor) -> torch.Tensor:
    if count < 0:
        raise ValueError("count must be non-negative")
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)

    native = native_extension()
    if native is None or not hasattr(native, "mc_sample_directions"):
        raise RuntimeError(
            "_channel_native.mc_sample_directions CUDA kernel is required"
        )
    directions = native.mc_sample_directions(int(count), reference)
    if not isinstance(directions, torch.Tensor):
        raise TypeError("_channel_native.mc_sample_directions must return a tensor")
    validate_cuda_tensor(
        "directions", directions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    return directions


def mc_transmitter_tensors(
    flat_positions: tuple[float, ...],
    powers: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    if len(flat_positions) % 3 != 0:
        raise ValueError("flat_positions must contain xyz triples")
    if len(flat_positions) // 3 != len(powers):
        raise ValueError("powers must match flat_positions")
    native = native_extension()
    if native is None or not hasattr(native, "mc_transmitter_tensors"):
        raise RuntimeError(
            "_channel_native.mc_transmitter_tensors CUDA helper is required"
        )
    exported = native.mc_transmitter_tensors(flat_positions, powers)
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.mc_transmitter_tensors must return a dict")
    validate_cuda_tensor(
        "positions",
        exported["positions"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor("power", exported["power"], dtype=torch.float32, ndim=1)
    return exported


def mc_pack_vec3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("x", x, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("y", y, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("z", z, dtype=torch.float32, ndim=1)
    if y.shape != x.shape or z.shape != x.shape:
        raise ValueError("x, y, and z must have the same shape")
    native = native_extension()
    if native is None or not hasattr(native, "mc_pack_vec3"):
        raise RuntimeError("_channel_native.mc_pack_vec3 CUDA kernel is required")
    packed = native.mc_pack_vec3(x, y, z)
    if not isinstance(packed, torch.Tensor):
        raise TypeError("_channel_native.mc_pack_vec3 must return a tensor")
    validate_cuda_tensor(
        "packed", packed, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if packed.shape[0] != x.shape[0]:
        raise ValueError("_channel_native.mc_pack_vec3 returned an unexpected shape")
    return packed


def mc_los_component_maps(los: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=3)
    native = native_extension()
    if native is None or not hasattr(native, "mc_los_component_maps"):
        raise RuntimeError(
            "_channel_native.mc_los_component_maps CUDA kernel is required"
        )
    maps = native.mc_los_component_maps(los)
    if not isinstance(maps, torch.Tensor):
        raise TypeError("_channel_native.mc_los_component_maps must return a tensor")
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    return maps


def mc_los_component_maps_from_matrix(
    los: torch.Tensor, *, rows: int, cols: int
) -> torch.Tensor:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=2)
    if rows < 0 or cols < 0:
        raise ValueError("rows and cols must be non-negative")
    if los.shape[1] != rows * cols:
        raise ValueError("los columns must match rows * cols")
    maps = _required_native_op("mc_los_component_maps_from_matrix")(
        los, int(rows), int(cols)
    )
    if not isinstance(maps, torch.Tensor):
        raise TypeError(
            "_channel_native.mc_los_component_maps_from_matrix must return a tensor"
        )
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    if maps.shape != (los.shape[0], cols, rows):
        raise ValueError(
            "_channel_native.mc_los_component_maps_from_matrix returned an unexpected shape"
        )
    return maps


def mc_apply_los_visibility(
    maps: torch.Tensor,
    los: torch.Tensor,
    visible: torch.Tensor,
    *,
    tx_index: int,
) -> torch.Tensor:
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    if maps.shape[0] != los.shape[0] or los.shape[1] != maps.shape[1] * maps.shape[2]:
        raise ValueError("los must match maps flattened grid layout")
    native = native_extension()
    if native is None or not hasattr(native, "mc_apply_los_visibility"):
        raise RuntimeError(
            "_channel_native.mc_apply_los_visibility CUDA kernel is required"
        )
    out = native.mc_apply_los_visibility(maps, los, visible, int(tx_index))
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel_native.mc_apply_los_visibility must return a tensor")
    return out


def mc_los_visibility_inputs(
    tx_positions: torch.Tensor,
    *,
    tx_index: int,
    rx_count: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if rx_count < 0:
        raise ValueError("rx_count must be non-negative")
    native = native_extension()
    if native is None or not hasattr(native, "mc_los_visibility_inputs"):
        raise RuntimeError(
            "_channel_native.mc_los_visibility_inputs CUDA kernel is required"
        )
    exported = native.mc_los_visibility_inputs(
        tx_positions, int(tx_index), int(rx_count)
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.mc_los_visibility_inputs must return a dict")
    validate_cuda_tensor(
        "start", exported["start"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    return exported


def mc_receiver_grid_points(
    reference: torch.Tensor,
    *,
    origin: tuple[float, float, float],
    x_axis: tuple[float, float, float],
    y_axis: tuple[float, float, float],
    shape: tuple[int, int],
    spacing: tuple[float, float],
) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    rows, cols = shape
    if rows < 0 or cols < 0:
        raise ValueError("shape entries must be non-negative")
    if spacing[0] <= 0.0 or spacing[1] <= 0.0:
        raise ValueError("spacing entries must be positive")
    native = native_extension()
    if native is None or not hasattr(native, "mc_receiver_grid_points"):
        raise RuntimeError(
            "_channel_native.mc_receiver_grid_points CUDA kernel is required"
        )
    points = native.mc_receiver_grid_points(
        reference,
        int(rows),
        int(cols),
        float(origin[0]),
        float(origin[1]),
        float(origin[2]),
        float(x_axis[0]),
        float(x_axis[1]),
        float(x_axis[2]),
        float(y_axis[0]),
        float(y_axis[1]),
        float(y_axis[2]),
        float(spacing[0]),
        float(spacing[1]),
    )
    if not isinstance(points, torch.Tensor):
        raise TypeError("_channel_native.mc_receiver_grid_points must return a tensor")
    validate_cuda_tensor(
        "points", points, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if points.shape[0] != rows * cols:
        raise ValueError(
            "_channel_native.mc_receiver_grid_points returned an unexpected shape"
        )
    return points


def mc_reflection_launch_inputs(
    tx_positions: torch.Tensor,
    *,
    tx_index: int,
    sample_count: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    native = native_extension()
    if native is None or not hasattr(native, "mc_reflection_launch_inputs"):
        raise RuntimeError(
            "_channel_native.mc_reflection_launch_inputs CUDA kernel is required"
        )
    exported = native.mc_reflection_launch_inputs(
        tx_positions, int(tx_index), int(sample_count)
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.mc_reflection_launch_inputs must return a dict"
        )
    validate_cuda_tensor(
        "ray_o", exported["ray_o"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("ray_tmax", exported["ray_tmax"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    validate_cuda_tensor(
        "tx_pol", exported["tx_pol"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    return exported


def mc_sionna_reflection_accumulate(
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    trace_valid: torch.Tensor,
    trace_t: torch.Tensor,
    trace_prim: torch.Tensor,
    face_normals: torch.Tensor,
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_thickness: torch.Tensor,
    *,
    contribution_depth: int,
    grid_axis: int,
    grid_position: float,
    grid_coord0_min: float,
    grid_coord0_max: float,
    grid_coord1_min: float,
    grid_coord1_max: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    solid_angle_per_ray: float,
    grid_cell_area: float,
) -> torch.Tensor:
    native = native_extension()
    if native is None or not hasattr(native, "mc_sionna_reflection_accumulate"):
        raise RuntimeError(
            "_channel_native.mc_sionna_reflection_accumulate CUDA kernel is required"
        )
    return native.mc_sionna_reflection_accumulate(
        ray_o,
        ray_d,
        trace_valid,
        trace_t,
        trace_prim,
        face_normals,
        material_eta_r,
        material_sigma,
        material_gain,
        material_valid,
        material_thickness,
        int(contribution_depth),
        int(grid_axis),
        float(grid_position),
        float(grid_coord0_min),
        float(grid_coord0_max),
        float(grid_coord1_min),
        float(grid_coord1_max),
        int(grid_resolution0),
        int(grid_resolution1),
        float(wavelength),
        float(solid_angle_per_ray),
        float(grid_cell_area),
    )


def mc_diffraction_state_wi(
    state_edge_pos: torch.Tensor, state_src: torch.Tensor
) -> torch.Tensor:
    validate_cuda_tensor(
        "state_edge_pos",
        state_edge_pos,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "state_src", state_src, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if state_src.shape != state_edge_pos.shape:
        raise ValueError("state_src must match state_edge_pos shape")
    native = native_extension()
    if native is None or not hasattr(native, "mc_diffraction_state_wi"):
        raise RuntimeError(
            "_channel_native.mc_diffraction_state_wi CUDA kernel is required"
        )
    state_wi = native.mc_diffraction_state_wi(state_edge_pos, state_src)
    if not isinstance(state_wi, torch.Tensor):
        raise TypeError("_channel_native.mc_diffraction_state_wi must return a tensor")
    validate_cuda_tensor(
        "state_wi", state_wi, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    return state_wi


def mc_sionna_diffraction_tape_accumulate(*args: object) -> torch.Tensor:
    native = native_extension()
    if native is None or not hasattr(native, "mc_sionna_diffraction_tape_accumulate"):
        raise RuntimeError(
            "_channel_native.mc_sionna_diffraction_tape_accumulate CUDA kernel is required"
        )
    output = native.mc_sionna_diffraction_tape_accumulate(*args)
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "_channel_native.mc_sionna_diffraction_tape_accumulate must return a tensor"
        )
    return output


# ---------------------------------------------------------------------------
# MC basic incoherent power-map AD (plan 07 AD-3). The solver emits a REAL
# power map assembled from per-component maps; the differentiable inputs are
# the compiled material store leaves (per-face eta_r / sigma_e / gain /
# thickness, per-layer CSR parameters), the carrier frequency and the LoS
# endpoint positions. The RayD trace and sampling tapes, the sampled ray
# directions, the deposit binning and the visibility masks are all frozen
# winners (plan 07 section 4): gradients cover the continuous part only. The
# reflection deposit weight is analytically independent of the ray origin
# (the weight is |Gamma|^2 * solid_angle * (lambda/4pi)^2 / (A_cell * |cos|),
# with the 1/d^2 spreading carried by the frozen ray density), so the
# transmitter-position gradient of the reflection map is exactly zero and is
# returned without a kernel launch.
# ---------------------------------------------------------------------------

_LIGHT_SPEED_M_PER_S_AD = 299_792_458.0


class _McLosPathGainAdFunction(torch.autograd.Function):
    """Differentiable LoS power-gain matrix (plan 07 AD-3).

    Differentiable inputs: tx_positions, rx_positions and the carrier
    frequency. tx_power stays fixed under the plan 07 contract; requesting
    its gradient fails loudly. The forward is the primal path_los_export
    kernel; only the path_gain_matrix output is exposed.
    """

    @staticmethod
    def forward(tx_positions, tx_power, rx_positions, frequency, frequency_value):
        exported = path_los_export(
            tx_positions,
            tx_power,
            rx_positions,
            frequency_hz=frequency_value,
        )
        return exported["path_gain_matrix"]

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        tx_positions, tx_power, rx_positions, frequency, frequency_value = inputs
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (tx_positions, tx_power, rx_positions)
        )
        ctx.frequency_value = frequency_value
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        _ad_reject_fixed_inputs(
            "mc_los_path_gain_ad",
            ctx.needs_input_grad,
            ((1, "tx_power"),),
        )
        need_tx = bool(ctx.needs_input_grad[0])
        need_rx = bool(ctx.needs_input_grad[2])
        need_frequency = bool(ctx.needs_input_grad[3])
        if grad_output is None or not (need_tx or need_rx or need_frequency):
            return None, None, None, None, None
        tx_positions, tx_power, rx_positions = ctx.saved_tensors
        grad_tx, _grad_power, grad_rx, grad_frequency = mc_los_path_gain_backward(
            tx_positions,
            tx_power,
            rx_positions,
            grad_output,
            frequency_hz=ctx.frequency_value,
        )
        return (
            grad_tx if need_tx else None,
            None,
            grad_rx if need_rx else None,
            _ad_frequency_grad(grad_frequency, ctx.frequency_meta)
            if need_frequency
            else None,
            None,
        )

    @staticmethod
    def jvp(ctx, t_tx, t_power, t_rx, t_frequency, _t_frequency_value):
        _ad_reject_fixed_tangents(
            "mc_los_path_gain_ad", ((t_power, "tx_power"),)
        )
        saved = ctx.saved_tensors
        tangent_tx = _ad_geometry_tangent(
            "mc_los_path_gain_ad tangent_tx", t_tx, saved[0]
        )
        tangent_rx = _ad_geometry_tangent(
            "mc_los_path_gain_ad tangent_rx", t_rx, saved[2]
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        if tangent_tx is None and tangent_rx is None and tangent_frequency == 0.0:
            return None
        tx_positions, tx_power, rx_positions = saved
        with torch_compat.disable_functorch():
            return mc_los_path_gain_jvp(
                _ad_native_tensor(tx_positions),
                _ad_native_tensor(tx_power),
                _ad_native_tensor(rx_positions),
                tangent_tx if tangent_tx is not None else _ad_native_tensor(tx_positions),
                _ad_native_tensor(tx_power),
                tangent_rx if tangent_rx is not None else _ad_native_tensor(rx_positions),
                tangent_tx is not None,
                False,
                tangent_rx is not None,
                frequency_hz=ctx.frequency_value,
                frequency_tangent=tangent_frequency,
            )


def mc_los_path_gain_ad(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> torch.Tensor:
    """Differentiable LoS path-gain matrix (endpoints and frequency).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam, audit M3); when not
    supplied it is read here, exactly once per apply.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    return _McLosPathGainAdFunction.apply(
        tx_positions, tx_power, rx_positions, frequency, float(frequency_value)
    )


def mc_reflection_ad_max_depth() -> int:
    """Depth cap of the native reflection radiomap AD companions.

    The backward/jvp kernels stage per-bounce state in fixed-size register
    arrays, so ``contribution_depth`` must not exceed this cap. Exposed so the
    solver can reject an over-deep AD configuration before any launch.
    """

    return int(_required_native_op("mc_reflection_ad_max_depth")())


def mc_sionna_reflection_accumulate_backward(
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    trace_valid: torch.Tensor,
    trace_t: torch.Tensor,
    trace_prim: torch.Tensor,
    face_normals: torch.Tensor,
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_thickness: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    need_materials: bool,
    need_frequency: bool,
    contribution_depth: int,
    grid_axis: int,
    grid_position: float,
    grid_coord0_min: float,
    grid_coord0_max: float,
    grid_coord1_min: float,
    grid_coord1_max: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    solid_angle_per_ray: float,
    grid_cell_area: float,
    wavelength_dfreq: float,
) -> tuple[torch.Tensor, ...]:
    gradients = _required_native_op("mc_sionna_reflection_accumulate_backward")(
        ray_o,
        ray_d,
        trace_valid,
        trace_t,
        trace_prim,
        face_normals,
        material_eta_r,
        material_sigma,
        material_gain,
        material_valid,
        material_thickness,
        grad_output,
        bool(need_materials),
        bool(need_frequency),
        int(contribution_depth),
        int(grid_axis),
        float(grid_position),
        float(grid_coord0_min),
        float(grid_coord0_max),
        float(grid_coord1_min),
        float(grid_coord1_max),
        int(grid_resolution0),
        int(grid_resolution1),
        float(wavelength),
        float(solid_angle_per_ray),
        float(grid_cell_area),
        float(wavelength_dfreq),
    )
    if not isinstance(gradients, tuple) or len(gradients) != 5:
        raise TypeError(
            "_channel_native.mc_sionna_reflection_accumulate_backward must "
            "return 5 tensors"
        )
    return gradients


def mc_sionna_reflection_accumulate_jvp(
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    trace_valid: torch.Tensor,
    trace_t: torch.Tensor,
    trace_prim: torch.Tensor,
    face_normals: torch.Tensor,
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_thickness: torch.Tensor,
    tangent_eta_r: torch.Tensor | None,
    tangent_sigma: torch.Tensor | None,
    tangent_gain: torch.Tensor | None,
    tangent_thickness: torch.Tensor | None,
    *,
    contribution_depth: int,
    grid_axis: int,
    grid_position: float,
    grid_coord0_min: float,
    grid_coord0_max: float,
    grid_coord1_min: float,
    grid_coord1_max: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    solid_angle_per_ray: float,
    grid_cell_area: float,
    wavelength_tangent: float,
) -> torch.Tensor:
    output = _required_native_op("mc_sionna_reflection_accumulate_jvp")(
        ray_o,
        ray_d,
        trace_valid,
        trace_t,
        trace_prim,
        face_normals,
        material_eta_r,
        material_sigma,
        material_gain,
        material_valid,
        material_thickness,
        tangent_eta_r if tangent_eta_r is not None else material_eta_r,
        tangent_sigma if tangent_sigma is not None else material_sigma,
        tangent_gain if tangent_gain is not None else material_gain,
        tangent_thickness if tangent_thickness is not None else material_thickness,
        tangent_eta_r is not None,
        tangent_sigma is not None,
        tangent_gain is not None,
        tangent_thickness is not None,
        int(contribution_depth),
        int(grid_axis),
        float(grid_position),
        float(grid_coord0_min),
        float(grid_coord0_max),
        float(grid_coord1_min),
        float(grid_coord1_max),
        int(grid_resolution0),
        int(grid_resolution1),
        float(wavelength),
        float(solid_angle_per_ray),
        float(grid_cell_area),
        float(wavelength_tangent),
    )
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "_channel_native.mc_sionna_reflection_accumulate_jvp must return a tensor"
        )
    return output


class _McReflectionMapAdFunction(torch.autograd.Function):
    """Differentiable Sionna reflection radiomap for one transmitter.

    Differentiable inputs: per-face eta_r / sigma_e / gain / thickness, the
    carrier frequency and the transmitter anchor. The trace tape, sampled
    directions, face normals and validity masks are frozen winners and fail
    loudly when a gradient is requested. The deposit weight is independent of
    the ray origin, so the transmitter anchor receives an exact zero gradient
    without a kernel launch (its binning influence is discrete and frozen).
    """

    @staticmethod
    def forward(
        tx_anchor,
        eta_r,
        sigma_e,
        gain,
        thickness,
        frequency,
        ray_o,
        ray_d,
        trace_valid,
        trace_t,
        trace_prim,
        face_normals,
        material_valid,
        params,
    ):
        return mc_sionna_reflection_accumulate(
            ray_o,
            ray_d,
            trace_valid,
            trace_t,
            trace_prim,
            face_normals,
            eta_r,
            sigma_e,
            gain,
            material_valid,
            thickness,
            contribution_depth=params["contribution_depth"],
            grid_axis=params["grid_axis"],
            grid_position=params["grid_position"],
            grid_coord0_min=params["grid_coord0_min"],
            grid_coord0_max=params["grid_coord0_max"],
            grid_coord1_min=params["grid_coord1_min"],
            grid_coord1_max=params["grid_coord1_max"],
            grid_resolution0=params["grid_resolution0"],
            grid_resolution1=params["grid_resolution1"],
            wavelength=params["wavelength"],
            solid_angle_per_ray=params["solid_angle_per_ray"],
            grid_cell_area=params["grid_cell_area"],
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:5]
        )
        frequency = inputs[5]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.params = inputs[13]
        ctx.anchor_shape = tuple(inputs[0].shape)
        saved = (*primals[1:], *inputs[6:13])
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        none_grads = (None,) * 14
        _ad_reject_fixed_inputs(
            "mc_sionna_reflection_accumulate_ad",
            ctx.needs_input_grad,
            (
                (6, "ray_o"),
                (7, "ray_d"),
                (8, "trace_valid"),
                (9, "trace_t"),
                (10, "trace_prim"),
                (11, "face_normals"),
                (12, "material_valid"),
            ),
        )
        if grad_output is None:
            return none_grads
        (
            eta_r,
            sigma_e,
            gain,
            thickness,
            ray_o,
            ray_d,
            trace_valid,
            trace_t,
            trace_prim,
            face_normals,
            material_valid,
        ) = ctx.saved_tensors
        need_anchor = bool(ctx.needs_input_grad[0])
        need_eta = bool(ctx.needs_input_grad[1])
        need_sigma = bool(ctx.needs_input_grad[2])
        need_gain = bool(ctx.needs_input_grad[3])
        need_thickness = bool(ctx.needs_input_grad[4])
        need_materials = need_eta or need_sigma or need_gain or need_thickness
        need_frequency = bool(ctx.needs_input_grad[5])
        grad_anchor = None
        if need_anchor:
            # The deposit weight carries no ray-origin dependence; the frozen
            # binning is the only door and it is discrete. Exact zero.
            grad_anchor = mc_zero_matrix(
                ray_o, rows=ctx.anchor_shape[0], cols=ctx.anchor_shape[1]
            )
        if not (need_materials or need_frequency):
            return (grad_anchor,) + (None,) * 13
        params = ctx.params
        wavelength = float(params["wavelength"])
        gradients = mc_sionna_reflection_accumulate_backward(
            ray_o,
            ray_d,
            trace_valid,
            trace_t,
            trace_prim,
            face_normals,
            eta_r,
            sigma_e,
            gain,
            material_valid,
            thickness,
            grad_output,
            need_materials=need_materials,
            need_frequency=need_frequency,
            contribution_depth=params["contribution_depth"],
            grid_axis=params["grid_axis"],
            grid_position=params["grid_position"],
            grid_coord0_min=params["grid_coord0_min"],
            grid_coord0_max=params["grid_coord0_max"],
            grid_coord1_min=params["grid_coord1_min"],
            grid_coord1_max=params["grid_coord1_max"],
            grid_resolution0=params["grid_resolution0"],
            grid_resolution1=params["grid_resolution1"],
            wavelength=wavelength,
            solid_angle_per_ray=params["solid_angle_per_ray"],
            grid_cell_area=params["grid_cell_area"],
            wavelength_dfreq=-wavelength * wavelength / _LIGHT_SPEED_M_PER_S_AD,
        )
        grad_frequency = (
            _ad_frequency_grad(gradients[4], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            grad_anchor,
            gradients[0] if need_eta else None,
            gradients[1] if need_sigma else None,
            gradients[2] if need_gain else None,
            gradients[3] if need_thickness else None,
            grad_frequency,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        t_anchor,
        t_eta,
        t_sigma,
        t_gain,
        t_thickness,
        t_frequency,
        t_ray_o,
        t_ray_d,
        t_trace_valid,
        t_trace_t,
        t_trace_prim,
        t_face_normals,
        t_material_valid,
        _t_params,
    ):
        _ad_reject_fixed_tangents(
            "mc_sionna_reflection_accumulate_ad",
            (
                (t_ray_o, "ray_o"),
                (t_ray_d, "ray_d"),
                (t_face_normals, "face_normals"),
            ),
        )
        saved = ctx.saved_tensors
        face_shape = tuple(saved[0].shape)
        tangent_eta = _ad_checked_tangent(
            "mc_sionna_reflection_accumulate_ad tangent_eta_r",
            _ad_native_tangent_or_none(t_eta),
            face_shape,
        )
        tangent_sigma = _ad_checked_tangent(
            "mc_sionna_reflection_accumulate_ad tangent_sigma_e",
            _ad_native_tangent_or_none(t_sigma),
            face_shape,
        )
        tangent_gain = _ad_checked_tangent(
            "mc_sionna_reflection_accumulate_ad tangent_gain",
            _ad_native_tangent_or_none(t_gain),
            face_shape,
        )
        tangent_thickness = _ad_checked_tangent(
            "mc_sionna_reflection_accumulate_ad tangent_thickness",
            _ad_native_tangent_or_none(t_thickness),
            face_shape,
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        params = ctx.params
        if (
            tangent_eta is None
            and tangent_sigma is None
            and tangent_gain is None
            and tangent_thickness is None
            and tangent_frequency == 0.0
        ):
            if _ad_native_tangent_or_none(
                t_anchor if isinstance(t_anchor, torch.Tensor) else None
            ) is None:
                return None
            # Transmitter-anchor-only tangent: the deposit weight carries no
            # ray-origin dependence (frozen binning), so the map tangent is
            # exactly zero. A concrete zero tensor is required: this torch
            # build rejects a None jvp when an input tangent is live.
            return mc_zero_matrix(
                _ad_native_tensor(saved[4]),
                rows=params["grid_resolution1"],
                cols=params["grid_resolution0"],
            )
        (
            eta_r,
            sigma_e,
            gain,
            thickness,
            ray_o,
            ray_d,
            trace_valid,
            trace_t,
            trace_prim,
            face_normals,
            material_valid,
        ) = saved
        wavelength = float(params["wavelength"])
        with torch_compat.disable_functorch():
            return mc_sionna_reflection_accumulate_jvp(
                _ad_native_tensor(ray_o),
                _ad_native_tensor(ray_d),
                _ad_native_tensor(trace_valid),
                _ad_native_tensor(trace_t),
                _ad_native_tensor(trace_prim),
                _ad_native_tensor(face_normals),
                _ad_native_tensor(eta_r),
                _ad_native_tensor(sigma_e),
                _ad_native_tensor(gain),
                _ad_native_tensor(material_valid),
                _ad_native_tensor(thickness),
                tangent_eta,
                tangent_sigma,
                tangent_gain,
                tangent_thickness,
                contribution_depth=params["contribution_depth"],
                grid_axis=params["grid_axis"],
                grid_position=params["grid_position"],
                grid_coord0_min=params["grid_coord0_min"],
                grid_coord0_max=params["grid_coord0_max"],
                grid_coord1_min=params["grid_coord1_min"],
                grid_coord1_max=params["grid_coord1_max"],
                grid_resolution0=params["grid_resolution0"],
                grid_resolution1=params["grid_resolution1"],
                wavelength=wavelength,
                solid_angle_per_ray=params["solid_angle_per_ray"],
                grid_cell_area=params["grid_cell_area"],
                wavelength_tangent=(
                    -wavelength * wavelength / _LIGHT_SPEED_M_PER_S_AD
                )
                * tangent_frequency,
            )


def mc_sionna_reflection_accumulate_ad(
    tx_anchor: torch.Tensor,
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_gain: torch.Tensor,
    material_thickness: torch.Tensor,
    frequency: torch.Tensor | float,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    trace_valid: torch.Tensor,
    trace_t: torch.Tensor,
    trace_prim: torch.Tensor,
    face_normals: torch.Tensor,
    material_valid: torch.Tensor,
    *,
    contribution_depth: int,
    grid_axis: int,
    grid_position: float,
    grid_coord0_min: float,
    grid_coord0_max: float,
    grid_coord1_min: float,
    grid_coord1_max: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    solid_angle_per_ray: float,
    grid_cell_area: float,
) -> torch.Tensor:
    """Differentiable :func:`mc_sionna_reflection_accumulate` (one tx)."""

    params = {
        "contribution_depth": int(contribution_depth),
        "grid_axis": int(grid_axis),
        "grid_position": float(grid_position),
        "grid_coord0_min": float(grid_coord0_min),
        "grid_coord0_max": float(grid_coord0_max),
        "grid_coord1_min": float(grid_coord1_min),
        "grid_coord1_max": float(grid_coord1_max),
        "grid_resolution0": int(grid_resolution0),
        "grid_resolution1": int(grid_resolution1),
        "wavelength": float(wavelength),
        "solid_angle_per_ray": float(solid_angle_per_ray),
        "grid_cell_area": float(grid_cell_area),
    }
    return _McReflectionMapAdFunction.apply(
        tx_anchor,
        material_eta_r,
        material_sigma,
        material_gain,
        material_thickness,
        frequency,
        ray_o,
        ray_d,
        trace_valid,
        trace_t,
        trace_prim,
        face_normals,
        material_valid,
        params,
    )


def mc_sionna_diffraction_tape_accumulate_backward(
    tape_tensors: tuple[torch.Tensor, ...],
    state_tensors: tuple[torch.Tensor, ...],
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_mu_r: torch.Tensor,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_thickness: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    need_materials: bool,
    need_source: bool,
    need_frequency: bool,
    grid_axis: int,
    grid_position: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    grid_cell_area: float,
    seed: int,
    total_edge_length: float,
    wavelength_dfreq: float,
) -> tuple[torch.Tensor, ...]:
    gradients = _required_native_op("mc_sionna_diffraction_tape_accumulate_backward")(
        *tape_tensors,
        *state_tensors,
        material_eta_r,
        material_sigma,
        material_mu_r,
        material_gain,
        material_valid,
        material_thickness,
        grad_output,
        bool(need_materials),
        bool(need_source),
        bool(need_frequency),
        int(grid_axis),
        float(grid_position),
        int(grid_resolution0),
        int(grid_resolution1),
        float(wavelength),
        float(grid_cell_area),
        int(seed),
        float(total_edge_length),
        float(wavelength_dfreq),
    )
    if not isinstance(gradients, tuple) or len(gradients) != 6:
        raise TypeError(
            "_channel_native.mc_sionna_diffraction_tape_accumulate_backward "
            "must return 6 tensors"
        )
    return gradients


def mc_sionna_diffraction_tape_accumulate_jvp(
    tape_tensors: tuple[torch.Tensor, ...],
    state_tensors: tuple[torch.Tensor, ...],
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_mu_r: torch.Tensor,
    material_gain: torch.Tensor,
    material_valid: torch.Tensor,
    material_thickness: torch.Tensor,
    tangent_eta_r: torch.Tensor | None,
    tangent_sigma: torch.Tensor | None,
    tangent_gain: torch.Tensor | None,
    tangent_thickness: torch.Tensor | None,
    tangent_source: torch.Tensor | None,
    *,
    grid_axis: int,
    grid_position: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    grid_cell_area: float,
    seed: int,
    total_edge_length: float,
    wavelength_tangent: float,
) -> torch.Tensor:
    output = _required_native_op("mc_sionna_diffraction_tape_accumulate_jvp")(
        *tape_tensors,
        *state_tensors,
        material_eta_r,
        material_sigma,
        material_mu_r,
        material_gain,
        material_valid,
        material_thickness,
        tangent_eta_r if tangent_eta_r is not None else material_eta_r,
        tangent_sigma if tangent_sigma is not None else material_sigma,
        tangent_gain if tangent_gain is not None else material_gain,
        tangent_thickness if tangent_thickness is not None else material_thickness,
        tangent_source
        if tangent_source is not None
        else material_eta_r.new_empty((3,)),
        tangent_eta_r is not None,
        tangent_sigma is not None,
        tangent_gain is not None,
        tangent_thickness is not None,
        tangent_source is not None,
        int(grid_axis),
        float(grid_position),
        int(grid_resolution0),
        int(grid_resolution1),
        float(wavelength),
        float(grid_cell_area),
        int(seed),
        float(total_edge_length),
        float(wavelength_tangent),
    )
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "_channel_native.mc_sionna_diffraction_tape_accumulate_jvp must "
            "return a tensor"
        )
    return output


class _McDiffractionMapAdFunction(torch.autograd.Function):
    """Differentiable Sionna diffraction radiomap for one transmitter.

    Differentiable inputs: per-face eta_r / sigma_e / gain / thickness, the
    carrier frequency and the transmitter anchor (the state sources are the
    anchor broadcast per winner edge state, so the anchor gradient is the
    per-state source gradient summed natively). The RayD sampling tape
    (active / state / cell / u), the per-lane Keller-cone azimuth, the edge
    state tables and the deposit binning are frozen winners and fail loudly
    when a gradient is requested; the continuous source dependence (incident
    spherical wave, incidence angles, cone orientation and the recomputed
    plane-crossing Jacobian) flows through the templated dual row.
    """

    @staticmethod
    def forward(
        tx_anchor,
        eta_r,
        sigma_e,
        gain,
        thickness,
        frequency,
        tape_active,
        tape_state,
        tape_cell,
        tape_u,
        state_edge_pos,
        state_edge_dir,
        state_t_min,
        state_t_max,
        state_n0,
        state_n1,
        state_prim0,
        state_prim1,
        state_exterior_angle,
        state_src,
        state_src_power,
        material_mu_r,
        material_valid,
        params,
    ):
        return mc_sionna_diffraction_tape_accumulate(
            tape_active,
            tape_state,
            tape_cell,
            tape_u,
            state_edge_pos,
            state_edge_dir,
            state_t_min,
            state_t_max,
            state_n0,
            state_n1,
            state_prim0,
            state_prim1,
            state_exterior_angle,
            state_src,
            state_src_power,
            eta_r,
            sigma_e,
            material_mu_r,
            gain,
            material_valid,
            thickness,
            params["grid_axis"],
            params["grid_position"],
            params["grid_coord0_min"],
            params["grid_coord0_max"],
            params["grid_coord1_min"],
            params["grid_coord1_max"],
            params["grid_resolution0"],
            params["grid_resolution1"],
            params["wavelength"],
            params["grid_cell_area"],
            params["seed"],
            params["total_edge_length"],
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:5]
        )
        frequency = inputs[5]
        ctx.frequency_meta = (
            (frequency.dtype, frequency.device)
            if isinstance(frequency, torch.Tensor)
            else None
        )
        ctx.params = inputs[23]
        ctx.anchor_shape = tuple(inputs[0].shape)
        saved = (*primals[1:], *inputs[6:23])
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        none_grads = (None,) * 24
        _ad_reject_fixed_inputs(
            "mc_sionna_diffraction_tape_accumulate_ad",
            ctx.needs_input_grad,
            (
                (6, "tape_active"),
                (7, "tape_state"),
                (8, "tape_cell"),
                (9, "tape_u"),
                (10, "state_edge_pos"),
                (11, "state_edge_dir"),
                (12, "state_t_min"),
                (13, "state_t_max"),
                (14, "state_n0"),
                (15, "state_n1"),
                (18, "state_exterior_angle"),
                (19, "state_src"),
                (20, "state_src_power"),
                (21, "material_mu_r"),
                (22, "material_valid"),
            ),
        )
        if grad_output is None:
            return none_grads
        saved = ctx.saved_tensors
        (eta_r, sigma_e, gain, thickness) = saved[:4]
        tape_tensors = saved[4:8]
        state_tensors = saved[8:19]
        material_mu_r = saved[19]
        material_valid = saved[20]
        need_anchor = bool(ctx.needs_input_grad[0])
        need_eta = bool(ctx.needs_input_grad[1])
        need_sigma = bool(ctx.needs_input_grad[2])
        need_gain = bool(ctx.needs_input_grad[3])
        need_thickness = bool(ctx.needs_input_grad[4])
        need_materials = need_eta or need_sigma or need_gain or need_thickness
        need_frequency = bool(ctx.needs_input_grad[5])
        if not (need_anchor or need_materials or need_frequency):
            return none_grads
        params = ctx.params
        wavelength = float(params["wavelength"])
        gradients = mc_sionna_diffraction_tape_accumulate_backward(
            tuple(tape_tensors),
            tuple(state_tensors),
            eta_r,
            sigma_e,
            material_mu_r,
            gain,
            material_valid,
            thickness,
            grad_output,
            need_materials=need_materials,
            need_source=need_anchor,
            need_frequency=need_frequency,
            grid_axis=params["grid_axis"],
            grid_position=params["grid_position"],
            grid_resolution0=params["grid_resolution0"],
            grid_resolution1=params["grid_resolution1"],
            wavelength=wavelength,
            grid_cell_area=params["grid_cell_area"],
            seed=params["seed"],
            total_edge_length=params["total_edge_length"],
            wavelength_dfreq=-wavelength * wavelength / _LIGHT_SPEED_M_PER_S_AD,
        )
        grad_frequency = (
            _ad_frequency_grad(gradients[5], ctx.frequency_meta)
            if need_frequency
            else None
        )
        return (
            gradients[4] if need_anchor else None,
            gradients[0] if need_eta else None,
            gradients[1] if need_sigma else None,
            gradients[2] if need_gain else None,
            gradients[3] if need_thickness else None,
            grad_frequency,
        ) + (None,) * 18

    @staticmethod
    def jvp(ctx, t_anchor, t_eta, t_sigma, t_gain, t_thickness, t_frequency, *t_rest):
        _ad_reject_fixed_tangents(
            "mc_sionna_diffraction_tape_accumulate_ad",
            (
                (t_rest[4], "state_edge_pos"),
                (t_rest[5], "state_edge_dir"),
                (t_rest[8], "state_n0"),
                (t_rest[9], "state_n1"),
                (t_rest[13], "state_src"),
                (t_rest[14], "state_src_power"),
            ),
        )
        saved = ctx.saved_tensors
        face_shape = tuple(saved[0].shape)
        tangent_eta = _ad_checked_tangent(
            "mc_sionna_diffraction_tape_accumulate_ad tangent_eta_r",
            _ad_native_tangent_or_none(t_eta),
            face_shape,
        )
        tangent_sigma = _ad_checked_tangent(
            "mc_sionna_diffraction_tape_accumulate_ad tangent_sigma_e",
            _ad_native_tangent_or_none(t_sigma),
            face_shape,
        )
        tangent_gain = _ad_checked_tangent(
            "mc_sionna_diffraction_tape_accumulate_ad tangent_gain",
            _ad_native_tangent_or_none(t_gain),
            face_shape,
        )
        tangent_thickness = _ad_checked_tangent(
            "mc_sionna_diffraction_tape_accumulate_ad tangent_thickness",
            _ad_native_tangent_or_none(t_thickness),
            face_shape,
        )
        tangent_anchor = _ad_native_tangent_or_none(
            t_anchor if isinstance(t_anchor, torch.Tensor) else None
        )
        tangent_frequency = _ad_frequency_tangent(t_frequency)
        params = ctx.params
        if (
            tangent_eta is None
            and tangent_sigma is None
            and tangent_gain is None
            and tangent_thickness is None
            and tangent_anchor is None
            and tangent_frequency == 0.0
        ):
            return None
        (eta_r, sigma_e, gain, thickness) = saved[:4]
        tape_tensors = tuple(_ad_native_tensor(value) for value in saved[4:8])
        state_tensors = tuple(_ad_native_tensor(value) for value in saved[8:19])
        wavelength = float(params["wavelength"])
        with torch_compat.disable_functorch():
            return mc_sionna_diffraction_tape_accumulate_jvp(
                tape_tensors,
                state_tensors,
                _ad_native_tensor(eta_r),
                _ad_native_tensor(sigma_e),
                _ad_native_tensor(saved[19]),
                _ad_native_tensor(gain),
                _ad_native_tensor(saved[20]),
                _ad_native_tensor(thickness),
                tangent_eta,
                tangent_sigma,
                tangent_gain,
                tangent_thickness,
                tangent_anchor,
                grid_axis=params["grid_axis"],
                grid_position=params["grid_position"],
                grid_resolution0=params["grid_resolution0"],
                grid_resolution1=params["grid_resolution1"],
                wavelength=wavelength,
                grid_cell_area=params["grid_cell_area"],
                seed=params["seed"],
                total_edge_length=params["total_edge_length"],
                wavelength_tangent=(
                    -wavelength * wavelength / _LIGHT_SPEED_M_PER_S_AD
                )
                * tangent_frequency,
            )


def mc_sionna_diffraction_tape_accumulate_ad(
    tx_anchor: torch.Tensor,
    material_eta_r: torch.Tensor,
    material_sigma: torch.Tensor,
    material_gain: torch.Tensor,
    material_thickness: torch.Tensor,
    frequency: torch.Tensor | float,
    tape_tensors: tuple[torch.Tensor, ...],
    state_tensors: tuple[torch.Tensor, ...],
    material_mu_r: torch.Tensor,
    material_valid: torch.Tensor,
    *,
    grid_axis: int,
    grid_position: float,
    grid_coord0_min: float,
    grid_coord0_max: float,
    grid_coord1_min: float,
    grid_coord1_max: float,
    grid_resolution0: int,
    grid_resolution1: int,
    wavelength: float,
    grid_cell_area: float,
    seed: int,
    total_edge_length: float,
) -> torch.Tensor:
    """Differentiable :func:`mc_sionna_diffraction_tape_accumulate` (one tx)."""

    if len(tape_tensors) != 4:
        raise ValueError("tape_tensors must hold (active, state, cell, u)")
    if len(state_tensors) != 11:
        raise ValueError("state_tensors must hold the 11 packed state tables")
    params = {
        "grid_axis": int(grid_axis),
        "grid_position": float(grid_position),
        "grid_coord0_min": float(grid_coord0_min),
        "grid_coord0_max": float(grid_coord0_max),
        "grid_coord1_min": float(grid_coord1_min),
        "grid_coord1_max": float(grid_coord1_max),
        "grid_resolution0": int(grid_resolution0),
        "grid_resolution1": int(grid_resolution1),
        "wavelength": float(wavelength),
        "grid_cell_area": float(grid_cell_area),
        "seed": int(seed),
        "total_edge_length": float(total_edge_length),
    }
    return _McDiffractionMapAdFunction.apply(
        tx_anchor,
        material_eta_r,
        material_sigma,
        material_gain,
        material_thickness,
        frequency,
        *tape_tensors,
        *state_tensors,
        material_mu_r,
        material_valid,
        params,
    )


def mc_selected_edge_indices(selected: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("selected", selected, dtype=torch.bool, ndim=1)
    native = native_extension()
    if native is None or not hasattr(native, "mc_selected_edge_indices"):
        raise RuntimeError(
            "_channel_native.mc_selected_edge_indices CUDA kernel is required"
        )
    indices = native.mc_selected_edge_indices(selected)
    if not isinstance(indices, torch.Tensor):
        raise TypeError("_channel_native.mc_selected_edge_indices must return a tensor")
    validate_cuda_tensor("indices", indices, dtype=torch.int32, ndim=1)
    return indices


def mc_diffraction_state_pack(
    edge_indices: torch.Tensor,
    edge_pos: torch.Tensor,
    edge_dir: torch.Tensor,
    line_min: torch.Tensor,
    line_max: torch.Tensor,
    n0: torch.Tensor,
    n1: torch.Tensor,
    face0: torch.Tensor,
    face1: torch.Tensor,
    exterior_angle: torch.Tensor,
    tx: torch.Tensor,
    tx_power: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    validate_cuda_tensor("edge_indices", edge_indices, dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "edge_pos", edge_pos, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "edge_dir", edge_dir, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("line_min", line_min, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("line_max", line_max, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("n0", n0, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("n1", n1, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("face0", face0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face1", face1, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("exterior_angle", exterior_angle, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("tx", tx, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=0)
    if tx.shape[0] != 3:
        raise ValueError("tx must have shape (3,)")
    native = native_extension()
    if native is None or not hasattr(native, "mc_diffraction_state_pack"):
        raise RuntimeError(
            "_channel_native.mc_diffraction_state_pack CUDA kernel is required"
        )
    states = native.mc_diffraction_state_pack(
        edge_indices,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
    )
    if not isinstance(states, tuple) or len(states) != 12:
        raise TypeError(
            "_channel_native.mc_diffraction_state_pack must return 12 tensors"
        )
    validate_cuda_tensor("state_edge_index", states[0], dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "state_edge_pos", states[1], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "state_edge_dir", states[2], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_line_min", states[3], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("state_line_max", states[4], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "state_n0", states[5], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "state_n1", states[6], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_face0", states[7], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_face1", states[8], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_exterior_angle", states[9], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "state_src", states[10], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_src_power", states[11], dtype=torch.float32, ndim=1)
    return states


def mc_face_material_tensors(
    material_eps_r: torch.Tensor,
    material_sigma_e: torch.Tensor,
    material_mu_r: torch.Tensor,
    face_material_id: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("material_eps_r", material_eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "material_sigma_e", material_sigma_e, dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor("material_mu_r", material_mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "face_material_id", face_material_id, dtype=torch.int32, ndim=1
    )
    if material_sigma_e.shape != material_eps_r.shape:
        raise ValueError("material_sigma_e must match material_eps_r shape")
    if material_mu_r.shape != material_eps_r.shape:
        raise ValueError("material_mu_r must match material_eps_r shape")

    native = native_extension()
    if native is None or not hasattr(native, "mc_face_material_tensors"):
        raise RuntimeError(
            "_channel_native.mc_face_material_tensors CUDA kernel is required"
        )
    exported = native.mc_face_material_tensors(
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        face_material_id,
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.mc_face_material_tensors must return a dict")
    return exported


def deterministic_los_field(
    path_gain: torch.Tensor,
    path_length_m: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("path_length_m", path_length_m, dtype=torch.float32, ndim=1)
    if path_length_m.shape != path_gain.shape:
        raise ValueError("path_length_m must match path_gain")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    native = native_extension()
    if native is None or not hasattr(native, "deterministic_los_field"):
        raise RuntimeError(
            "_channel_native.deterministic_los_field CUDA kernel is required"
        )
    exported = native.deterministic_los_field(
        path_gain, path_length_m, float(frequency_hz)
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.deterministic_los_field must return a dict")
    validate_cuda_tensor(
        "path_gain", exported["path_gain"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "field_real", exported["field_real"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "field_imag", exported["field_imag"], dtype=torch.float32, ndim=1
    )
    if exported["path_gain"].shape != path_gain.shape:
        raise ValueError("_channel_native.deterministic_los_field returned bad shape")
    return exported


def deterministic_diffraction_vector_field(
    x_re: torch.Tensor,
    x_im: torch.Tensor,
    y_re: torch.Tensor,
    y_im: torch.Tensor,
    z_re: torch.Tensor,
    z_im: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("x_re", x_re, dtype=torch.float32, ndim=1)
    for name, tensor in {
        "x_im": x_im,
        "y_re": y_re,
        "y_im": y_im,
        "z_re": z_re,
        "z_im": z_im,
    }.items():
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=1)
        if tensor.shape != x_re.shape:
            raise ValueError(f"{name} must match x_re")

    native = native_extension()
    if native is None or not hasattr(native, "deterministic_diffraction_vector_field"):
        raise RuntimeError(
            "_channel_native.deterministic_diffraction_vector_field CUDA kernel is required"
        )
    exported = native.deterministic_diffraction_vector_field(
        x_re, x_im, y_re, y_im, z_re, z_im
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.deterministic_diffraction_vector_field must return a dict"
        )
    validate_cuda_tensor(
        "path_gain", exported["path_gain"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "field_real", exported["field_real"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "field_imag", exported["field_imag"], dtype=torch.float32, ndim=1
    )
    if exported["path_gain"].shape != x_re.shape:
        raise ValueError(
            "_channel_native.deterministic_diffraction_vector_field returned bad shape"
        )
    return exported


def deterministic_reflection_field(
    tx_position: torch.Tensor,
    rx_position: torch.Tensor,
    hit_position: torch.Tensor,
    normal: torch.Tensor,
    tx_power: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_position", tx_position, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "rx_position", rx_position, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "hit_position", hit_position, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "normal", normal, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("eps_r", eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("sigma_e", sigma_e, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("mu_r", mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("gain", gain, dtype=torch.float32, ndim=1)
    count = tx_position.shape[0]
    if (
        rx_position.shape != tx_position.shape
        or hit_position.shape != tx_position.shape
        or normal.shape != tx_position.shape
    ):
        raise ValueError("reflection field vec3 tensors must have matching shape")
    for name, tensor in {
        "tx_power": tx_power,
        "eps_r": eps_r,
        "sigma_e": sigma_e,
        "mu_r": mu_r,
        "gain": gain,
    }.items():
        if tensor.shape[0] != count:
            raise ValueError(f"{name} must match path count")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    native = native_extension()
    if native is None or not hasattr(native, "deterministic_reflection_field"):
        raise RuntimeError(
            "_channel_native.deterministic_reflection_field CUDA kernel is required"
        )
    exported = native.deterministic_reflection_field(
        tx_position,
        rx_position,
        hit_position,
        normal,
        tx_power,
        eps_r,
        sigma_e,
        mu_r,
        gain,
        float(frequency_hz),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.deterministic_reflection_field must return a dict"
        )
    validate_cuda_tensor(
        "path_gain", exported["path_gain"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "field_real", exported["field_real"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "field_imag", exported["field_imag"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "path_length_m", exported["path_length_m"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor("delay_s", exported["delay_s"], dtype=torch.float32, ndim=1)
    if exported["path_length_m"].shape != (count,) or exported["delay_s"].shape != (
        count,
    ):
        raise ValueError(
            "_channel_native.deterministic_reflection_field returned bad length shape"
        )
    return exported


def deterministic_reflection_sequence_field(
    tx_position: torch.Tensor,
    rx_position: torch.Tensor,
    hit_positions: torch.Tensor,
    normals: torch.Tensor,
    tx_power: torch.Tensor,
    eps_r: torch.Tensor,
    sigma_e: torch.Tensor,
    mu_r: torch.Tensor,
    gain: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_position", tx_position, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "rx_position", rx_position, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("hit_positions", hit_positions, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("normals", normals, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("eps_r", eps_r, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("sigma_e", sigma_e, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("mu_r", mu_r, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("gain", gain, dtype=torch.float32, ndim=2)
    count = tx_position.shape[0]
    if rx_position.shape != tx_position.shape:
        raise ValueError("rx_position must match tx_position")
    if hit_positions.shape[0] != count or hit_positions.shape[2] != 3:
        raise ValueError("hit_positions must have shape (path_count, depth, 3)")
    if normals.shape != hit_positions.shape:
        raise ValueError("normals must match hit_positions")
    depth_shape = hit_positions.shape[:2]
    for name, tensor in {
        "eps_r": eps_r,
        "sigma_e": sigma_e,
        "mu_r": mu_r,
        "gain": gain,
    }.items():
        if tensor.shape != depth_shape:
            raise ValueError(f"{name} must have shape (path_count, depth)")
    if tx_power.shape[0] != count:
        raise ValueError("tx_power must match path count")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    native = native_extension()
    if native is None or not hasattr(native, "deterministic_reflection_sequence_field"):
        raise RuntimeError(
            "_channel_native.deterministic_reflection_sequence_field CUDA kernel is required"
        )
    exported = native.deterministic_reflection_sequence_field(
        tx_position,
        rx_position,
        hit_positions,
        normals,
        tx_power,
        eps_r,
        sigma_e,
        mu_r,
        gain,
        float(frequency_hz),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.deterministic_reflection_sequence_field must return a dict"
        )
    validate_cuda_tensor(
        "path_gain", exported["path_gain"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "field_real", exported["field_real"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "field_imag", exported["field_imag"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "path_length_m", exported["path_length_m"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor("delay_s", exported["delay_s"], dtype=torch.float32, ndim=1)
    if exported["path_length_m"].shape != (count,) or exported["delay_s"].shape != (
        count,
    ):
        raise ValueError(
            "_channel_native.deterministic_reflection_sequence_field returned bad length shape"
        )
    return exported


def deterministic_delay_to_path_length(delay_s: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("delay_s", delay_s, dtype=torch.float32, ndim=1)
    native = native_extension()
    if native is None or not hasattr(native, "deterministic_delay_to_path_length"):
        raise RuntimeError(
            "_channel_native.deterministic_delay_to_path_length CUDA kernel is required"
        )
    path_length = native.deterministic_delay_to_path_length(delay_s)
    validate_cuda_tensor("path_length_m", path_length, dtype=torch.float32, ndim=1)
    if path_length.shape != delay_s.shape:
        raise ValueError(
            "_channel_native.deterministic_delay_to_path_length returned bad shape"
        )
    return path_length


def deterministic_pack_complex(
    field_real: torch.Tensor, field_imag: torch.Tensor
) -> torch.Tensor:
    validate_cuda_tensor("field_real", field_real, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("field_imag", field_imag, dtype=torch.float32, ndim=1)
    if field_imag.shape != field_real.shape:
        raise ValueError("field_imag must match field_real")
    native = native_extension()
    if native is None or not hasattr(native, "deterministic_pack_complex"):
        raise RuntimeError(
            "_channel_native.deterministic_pack_complex CUDA kernel is required"
        )
    field = native.deterministic_pack_complex(field_real, field_imag)
    validate_cuda_tensor("field", field, dtype=torch.complex64, ndim=1)
    if field.shape != field_real.shape:
        raise ValueError(
            "_channel_native.deterministic_pack_complex returned bad shape"
        )
    return field


def deterministic_phase_from_field(
    field_real: torch.Tensor, field_imag: torch.Tensor
) -> torch.Tensor:
    validate_cuda_tensor("field_real", field_real, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("field_imag", field_imag, dtype=torch.float32, ndim=1)
    if field_imag.shape != field_real.shape:
        raise ValueError("field_imag must match field_real")
    native = native_extension()
    if native is None or not hasattr(native, "deterministic_phase_from_field"):
        raise RuntimeError(
            "_channel_native.deterministic_phase_from_field CUDA kernel is required"
        )
    phase = native.deterministic_phase_from_field(field_real, field_imag)
    validate_cuda_tensor("phase_rad", phase, dtype=torch.float32, ndim=1)
    if phase.shape != field_real.shape:
        raise ValueError(
            "_channel_native.deterministic_phase_from_field returned bad shape"
        )
    return phase


def deterministic_zero_field_phase(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=1)
    native = native_extension()
    if native is None or not hasattr(native, "deterministic_zero_field_phase"):
        raise RuntimeError(
            "_channel_native.deterministic_zero_field_phase CUDA kernel is required"
        )
    exported = native.deterministic_zero_field_phase(reference)
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.deterministic_zero_field_phase must return a dict"
        )
    validate_cuda_tensor(
        "path_field", exported["path_field"], dtype=torch.complex64, ndim=1
    )
    validate_cuda_tensor(
        "phase_rad", exported["phase_rad"], dtype=torch.float32, ndim=1
    )
    if (
        exported["path_field"].shape != reference.shape
        or exported["phase_rad"].shape != reference.shape
    ):
        raise ValueError(
            "_channel_native.deterministic_zero_field_phase returned bad shape"
        )
    return exported


def deterministic_phase_from_length(
    path_length_m: torch.Tensor, *, frequency_hz: float
) -> torch.Tensor:
    validate_cuda_tensor("path_length_m", path_length_m, dtype=torch.float32, ndim=1)
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    native = native_extension()
    if native is None or not hasattr(native, "deterministic_phase_from_length"):
        raise RuntimeError(
            "_channel_native.deterministic_phase_from_length CUDA kernel is required"
        )
    phase = native.deterministic_phase_from_length(path_length_m, float(frequency_hz))
    validate_cuda_tensor("phase_rad", phase, dtype=torch.float32, ndim=1)
    if phase.shape != path_length_m.shape:
        raise ValueError(
            "_channel_native.deterministic_phase_from_length returned bad shape"
        )
    return phase


def deterministic_field_from_power_phase(
    path_gain: torch.Tensor, phase_rad: torch.Tensor
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("phase_rad", phase_rad, dtype=torch.float32, ndim=1)
    if phase_rad.shape != path_gain.shape:
        raise ValueError("phase_rad must match path_gain")
    native = native_extension()
    if native is None or not hasattr(native, "deterministic_field_from_power_phase"):
        raise RuntimeError(
            "_channel_native.deterministic_field_from_power_phase CUDA kernel is required"
        )
    exported = native.deterministic_field_from_power_phase(path_gain, phase_rad)
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.deterministic_field_from_power_phase must return a dict"
        )
    validate_cuda_tensor(
        "field_real", exported["field_real"], dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor(
        "field_imag", exported["field_imag"], dtype=torch.float32, ndim=1
    )
    if (
        exported["field_real"].shape != path_gain.shape
        or exported["field_imag"].shape != path_gain.shape
    ):
        raise ValueError(
            "_channel_native.deterministic_field_from_power_phase returned bad shape"
        )
    return exported


def deterministic_accumulate_flat(
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    component_id: torch.Tensor,
    path_gain: torch.Tensor,
    field_real: torch.Tensor,
    field_imag: torch.Tensor,
    *,
    num_tx: int,
    num_rx: int,
    coherent: bool,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("tx_id", tx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("rx_id", rx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("component_id", component_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("field_real", field_real, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("field_imag", field_imag, dtype=torch.float32, ndim=1)
    for name, tensor in {
        "rx_id": rx_id,
        "component_id": component_id,
        "path_gain": path_gain,
        "field_real": field_real,
        "field_imag": field_imag,
    }.items():
        if tensor.shape != tx_id.shape:
            raise ValueError(f"{name} must match tx_id shape")
    if num_tx < 0 or num_rx < 0:
        raise ValueError("num_tx and num_rx must be non-negative")

    native = native_extension()
    if native is None or not hasattr(native, "deterministic_accumulate_flat"):
        raise RuntimeError(
            "_channel_native.deterministic_accumulate_flat CUDA kernel is required"
        )
    exported = native.deterministic_accumulate_flat(
        tx_id,
        rx_id,
        component_id,
        path_gain,
        field_real,
        field_imag,
        int(num_tx),
        int(num_rx),
        bool(coherent),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.deterministic_accumulate_flat must return a dict"
        )
    validate_cuda_tensor(
        "power_total", exported["power_total"], dtype=torch.float32, ndim=2
    )
    validate_cuda_tensor(
        "field_total_real", exported["field_total_real"], dtype=torch.float32, ndim=2
    )
    validate_cuda_tensor(
        "field_total_imag", exported["field_total_imag"], dtype=torch.float32, ndim=2
    )
    validate_cuda_tensor(
        "component_power", exported["component_power"], dtype=torch.float32, ndim=3
    )
    validate_cuda_tensor(
        "component_field_real",
        exported["component_field_real"],
        dtype=torch.float32,
        ndim=3,
    )
    validate_cuda_tensor(
        "component_field_imag",
        exported["component_field_imag"],
        dtype=torch.float32,
        ndim=3,
    )
    expected_component_shape = (5, int(num_tx), int(num_rx))
    if tuple(exported["power_total"].shape) != (int(num_tx), int(num_rx)):
        raise ValueError(
            "_channel_native.deterministic_accumulate_flat returned bad power_total shape"
        )
    if tuple(exported["component_power"].shape) != expected_component_shape:
        raise ValueError(
            "_channel_native.deterministic_accumulate_flat returned bad component shape"
        )
    return exported


_DETERMINISTIC_ACCUM_FIELDS = (
    "power_total",
    "field_total_real",
    "field_total_imag",
    "component_power",
    "component_field_real",
    "component_field_imag",
)


def deterministic_accumulate_flat_backward(
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    component_id: torch.Tensor,
    component_field_real: torch.Tensor,
    component_field_imag: torch.Tensor,
    field_total_real: torch.Tensor,
    field_total_imag: torch.Tensor,
    power_total: torch.Tensor,
    *,
    grad_power_total: torch.Tensor | None = None,
    grad_field_total_real: torch.Tensor | None = None,
    grad_field_total_imag: torch.Tensor | None = None,
    grad_component_power: torch.Tensor | None = None,
    grad_component_field_real: torch.Tensor | None = None,
    grad_component_field_imag: torch.Tensor | None = None,
    num_tx: int,
    num_rx: int,
    coherent: bool,
) -> dict[str, torch.Tensor]:
    out = _required_native_op("deterministic_accumulate_flat_backward")(
        tx_id,
        rx_id,
        component_id,
        component_field_real,
        component_field_imag,
        field_total_real,
        field_total_imag,
        power_total,
        grad_power_total,
        grad_field_total_real,
        grad_field_total_imag,
        grad_component_power,
        grad_component_field_real,
        grad_component_field_imag,
        int(num_tx),
        int(num_rx),
        bool(coherent),
    )
    expected = {"grad_path_gain", "grad_field_real", "grad_field_imag"}
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError(
            "_channel_native.deterministic_accumulate_flat_backward returned"
            " invalid fields"
        )
    return out


def deterministic_accumulate_flat_jvp(
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    component_id: torch.Tensor,
    component_field_real: torch.Tensor,
    component_field_imag: torch.Tensor,
    power_total: torch.Tensor,
    *,
    tangent_path_gain: torch.Tensor | None = None,
    tangent_field_real: torch.Tensor | None = None,
    tangent_field_imag: torch.Tensor | None = None,
    num_tx: int,
    num_rx: int,
    coherent: bool,
) -> dict[str, torch.Tensor]:
    out = _required_native_op("deterministic_accumulate_flat_jvp")(
        tx_id,
        rx_id,
        component_id,
        component_field_real,
        component_field_imag,
        power_total,
        tangent_path_gain,
        tangent_field_real,
        tangent_field_imag,
        int(num_tx),
        int(num_rx),
        bool(coherent),
    )
    if not isinstance(out, dict) or set(out) != set(_DETERMINISTIC_ACCUM_FIELDS):
        raise TypeError(
            "_channel_native.deterministic_accumulate_flat_jvp returned"
            " invalid fields"
        )
    return out


class _DeterministicAccumulateFlatAdFunction(torch.autograd.Function):
    """Differentiable deterministic flat-path accumulation (plan 07).

    The forward is the primal native accumulator: each kept path's complex
    field and real power scatter into a frozen (component_slot, tx, rx)
    cell over the five slots los / reflection / diffraction / transmission /
    scattering, then coherent cells square the summed field (|sum E|^2 over
    the four field slots) while incoherent cells sum per-path powers and
    expose a sqrt-power pseudo-field. Scattering is a power-domain slot in
    both modes: its gains add linearly to the totals and its cell field is
    a diagnostic that reaches no total. The scatter is linear in the
    per-path field/power, so the adjoint is one masked-gather kernel
    through the same frozen gates with the |.|^2 / sqrt cell nonlinearities
    linearized at the saved forward outputs, and the pushforward is the
    same scatter applied to the tangents. The slot/tx/rx ids are discrete
    winners and stay fixed. A float64 input batch routes through the
    float64 companion forward so torch.autograd.gradcheck can run in
    strict double precision.
    """

    @staticmethod
    def forward(
        tx_id,
        rx_id,
        component_id,
        path_gain,
        field_real,
        field_imag,
        num_tx,
        num_rx,
        coherent,
    ):
        op_name = (
            "deterministic_accumulate_flat_fwd64"
            if path_gain.dtype == torch.float64
            else "deterministic_accumulate_flat"
        )
        out = _required_native_op(op_name)(
            tx_id,
            rx_id,
            component_id,
            path_gain,
            field_real,
            field_imag,
            int(num_tx),
            int(num_rx),
            bool(coherent),
        )
        return tuple(out[name] for name in _DETERMINISTIC_ACCUM_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        tx_id, rx_id, component_id = (
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:3]
        )
        ctx.num_tx = int(inputs[6])
        ctx.num_rx = int(inputs[7])
        ctx.coherent = bool(inputs[8])
        (
            power_total,
            field_total_real,
            field_total_imag,
            _component_power,
            component_field_real,
            component_field_imag,
        ) = output
        saved = (
            tx_id,
            rx_id,
            component_id,
            component_field_real,
            component_field_imag,
            field_total_real,
            field_total_imag,
            power_total,
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx,
        grad_power_total,
        grad_field_total_real,
        grad_field_total_imag,
        grad_component_power,
        grad_component_field_real,
        grad_component_field_imag,
    ):
        none_grads = (None,) * 9
        need_gain = bool(ctx.needs_input_grad[3])
        need_field = bool(ctx.needs_input_grad[4]) or bool(
            ctx.needs_input_grad[5]
        )
        grads = (
            grad_power_total,
            grad_field_total_real,
            grad_field_total_imag,
            grad_component_power,
            grad_component_field_real,
            grad_component_field_imag,
        )
        if not (need_gain or need_field) or all(
            value is None for value in grads
        ):
            return none_grads
        (
            tx_id,
            rx_id,
            component_id,
            component_field_real,
            component_field_imag,
            field_total_real,
            field_total_imag,
            power_total,
        ) = ctx.saved_tensors
        out = deterministic_accumulate_flat_backward(
            tx_id,
            rx_id,
            component_id,
            component_field_real,
            component_field_imag,
            field_total_real,
            field_total_imag,
            power_total,
            grad_power_total=grad_power_total,
            grad_field_total_real=grad_field_total_real,
            grad_field_total_imag=grad_field_total_imag,
            grad_component_power=grad_component_power,
            grad_component_field_real=grad_component_field_real,
            grad_component_field_imag=grad_component_field_imag,
            num_tx=ctx.num_tx,
            num_rx=ctx.num_rx,
            coherent=ctx.coherent,
        )
        return (
            None,
            None,
            None,
            out["grad_path_gain"] if ctx.needs_input_grad[3] else None,
            out["grad_field_real"] if ctx.needs_input_grad[4] else None,
            out["grad_field_imag"] if ctx.needs_input_grad[5] else None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(
        ctx,
        _t_tx_id,
        _t_rx_id,
        _t_component_id,
        t_path_gain,
        t_field_real,
        t_field_imag,
        _t_num_tx,
        _t_num_rx,
        _t_coherent,
    ):
        tangent_gain = _ad_native_tangent_or_none(t_path_gain)
        tangent_real = _ad_native_tangent_or_none(t_field_real)
        tangent_imag = _ad_native_tangent_or_none(t_field_imag)
        if tangent_gain is None and tangent_real is None and tangent_imag is None:
            return (None,) * len(_DETERMINISTIC_ACCUM_FIELDS)
        (
            tx_id,
            rx_id,
            component_id,
            component_field_real,
            component_field_imag,
            _field_total_real,
            _field_total_imag,
            power_total,
        ) = (_ad_native_tensor(value) for value in ctx.saved_tensors)
        with torch_compat.disable_functorch():
            out = deterministic_accumulate_flat_jvp(
                tx_id,
                rx_id,
                component_id,
                component_field_real,
                component_field_imag,
                power_total,
                tangent_path_gain=tangent_gain,
                tangent_field_real=tangent_real,
                tangent_field_imag=tangent_imag,
                num_tx=ctx.num_tx,
                num_rx=ctx.num_rx,
                coherent=ctx.coherent,
            )
        return tuple(out[name] for name in _DETERMINISTIC_ACCUM_FIELDS)


def deterministic_accumulate_flat_ad(
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    component_id: torch.Tensor,
    path_gain: torch.Tensor,
    field_real: torch.Tensor,
    field_imag: torch.Tensor,
    *,
    num_tx: int,
    num_rx: int,
    coherent: bool,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`deterministic_accumulate_flat` (plan 07)."""

    values = _DeterministicAccumulateFlatAdFunction.apply(
        tx_id,
        rx_id,
        component_id,
        path_gain,
        field_real,
        field_imag,
        int(num_tx),
        int(num_rx),
        bool(coherent),
    )
    return dict(zip(_DETERMINISTIC_ACCUM_FIELDS, values, strict=True))
