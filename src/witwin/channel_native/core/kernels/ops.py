from __future__ import annotations

import torch

from .extension import native_extension
from .metadata import make_metadata, validate_metadata  # noqa: F401


def validate_cuda_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    ndim: int,
    trailing_shape: tuple[int, ...] = (),
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if trailing_shape and tuple(tensor.shape[-len(trailing_shape) :]) != trailing_shape:
        raise ValueError(f"{name} must end with shape {trailing_shape}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def _required_native_op(name: str):
    native = native_extension()
    if native is None or not hasattr(native, name):
        raise RuntimeError(f"_channel_native.{name} CUDA kernel is required")
    return getattr(native, name)


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


def noop_metadata(
    *, accumulation_strategy: str = "none"
) -> dict[str, bool | float | int | str]:
    return make_metadata(
        primitive="noop_metadata",
        accumulation_strategy=accumulation_strategy,
        scheduling_strategy="none",
        ad_status="none",
    )


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
    for name in ("light_seed", "sensor_seed", "connection_seed", "diffraction_seed"):
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
    if strategy_count <= 0:
        raise ValueError("strategy_count must be positive")
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
    if set(exported) != {"path_gain", "los", "reflection", "diffraction"}:
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
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("reflection", reflection, dtype=torch.float32, ndim=2)
    validate_cuda_tensor("diffraction", diffraction, dtype=torch.float32, ndim=2)
    if reflection.shape != los.shape or diffraction.shape != los.shape:
        raise ValueError("point component matrices must have matching shapes")
    exported = _required_native_op("bdpt_finalize_point_components")(
        los, reflection, diffraction
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
    for name in ("los_power", "reflection_power", "diffraction_power"):
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
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("reflection", reflection, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("diffraction", diffraction, dtype=torch.float32, ndim=3)
    if reflection.shape != los.shape or diffraction.shape != los.shape:
        raise ValueError("component maps must share shape")
    exported = _required_native_op("bdpt_finalize_component_maps")(
        los, reflection, diffraction
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
    for name in ("los_power", "reflection_power", "diffraction_power"):
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


def bdpt_diffraction_edge_geometry(
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
    geometry = _required_native_op("bdpt_diffraction_edge_geometry")(
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
            "_channel_native.bdpt_diffraction_edge_geometry must return 11 tensors"
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


def bdpt_surface_group_edge_candidates(
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
    candidates = _required_native_op("bdpt_surface_group_edge_candidates")(
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
            "_channel_native.bdpt_surface_group_edge_candidates must return 2 tensors"
        )
    validate_cuda_tensor(
        "triangle_edge_count", candidates[0], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "triangle_edge_indices", candidates[1], dtype=torch.int32, ndim=2
    )
    return candidates


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
        _raydn_module_handle(),
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


def raydn_diffraction_paths_order1_forward(*args: object) -> tuple[torch.Tensor, ...]:
    if not args:
        raise TypeError(
            "raydn_diffraction_paths_order1_forward requires a RayDN scene handle"
        )
    native_args = (_raydn_scene_handle_id(args[0]), *args[1:])
    out = _required_native_op("raydn_diffraction_paths_order1_forward")(
        *native_args,
        _raydn_module_handle(),
    )
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "_channel_native.raydn_diffraction_paths_order1_forward must return a tensor sequence"
        )
    return tuple(out)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    if not isinstance(gradients, tuple) or len(gradients) != 3:
        raise TypeError(
            "_channel_native.mc_los_path_gain_backward must return 3 tensors"
        )
    validate_cuda_tensor(
        "grad_tx", gradients[0], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("grad_power", gradients[1], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "grad_rx", gradients[2], dtype=torch.float32, ndim=2, trailing_shape=(3,)
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
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("reflection", reflection, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("diffraction", diffraction, dtype=torch.float32, ndim=3)
    if reflection.shape != los.shape:
        raise ValueError("reflection must match los shape")
    if diffraction.shape != los.shape:
        raise ValueError("diffraction must match los shape")

    native = native_extension()
    if native is None or not hasattr(native, "mc_finalize_component_maps"):
        raise RuntimeError(
            "_channel_native.mc_finalize_component_maps CUDA kernel is required"
        )
    exported = native.mc_finalize_component_maps(los, reflection, diffraction)
    if not isinstance(exported, dict):
        raise TypeError("_channel_native.mc_finalize_component_maps must return a dict")
    return exported


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
    expected_component_shape = (3, int(num_tx), int(num_rx))
    if tuple(exported["power_total"].shape) != (int(num_tx), int(num_rx)):
        raise ValueError(
            "_channel_native.deterministic_accumulate_flat returned bad power_total shape"
        )
    if tuple(exported["component_power"].shape) != expected_component_shape:
        raise ValueError(
            "_channel_native.deterministic_accumulate_flat returned bad component shape"
        )
    return exported
