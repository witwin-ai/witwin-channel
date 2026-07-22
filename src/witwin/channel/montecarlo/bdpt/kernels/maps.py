from __future__ import annotations

import torch

from witwin.channel.runtime.native_buffers import bdpt_zero_matrix  # noqa: F401
from witwin.channel.runtime.symbols import (
    native_extension,
    required_symbol as _required_native_op,
)
from witwin.channel.runtime.tensor_contracts import validate_cuda_tensor


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
            "_channel.bdpt_store_point_component_column must return a tensor"
        )
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2)
    if out.shape != target.shape:
        raise ValueError(
            "_channel.bdpt_store_point_component_column returned an unexpected shape"
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
            "_channel.bdpt_finalize_point_components must return a dict"
        )
    validate_cuda_tensor(
        "path_gain", exported["path_gain"], dtype=torch.float32, ndim=2
    )
    if exported["path_gain"].shape != los.shape:
        raise ValueError(
            "_channel.bdpt_finalize_point_components returned bad path_gain shape"
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
            "_channel.bdpt_point_component_power CUDA kernel is required"
        )
    exported = native.bdpt_point_component_power(path_gain, bool(include_los))
    if not isinstance(exported, dict):
        raise TypeError("_channel.bdpt_point_component_power must return a dict")
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
        raise TypeError("_channel.bdpt_transmitter_tensors must return a dict")
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
            "_channel.bdpt_receiver_grid_points must return a tensor"
        )
    validate_cuda_tensor(
        "points", points, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if points.shape[0] != rows * cols:
        raise ValueError(
            "_channel.bdpt_receiver_grid_points returned an unexpected shape"
        )
    return points


def bdpt_los_export(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_polarizations: torch.Tensor,
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
    validate_cuda_tensor(
        "tx_polarizations",
        tx_polarizations,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    if tx_power.shape[0] != tx_positions.shape[0]:
        raise ValueError("tx_power must have one value per transmitter")
    if tx_polarizations.shape != tx_positions.shape:
        raise ValueError("tx_polarizations must match tx_positions shape")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    exported = _required_native_op("bdpt_los_export")(
        tx_positions,
        tx_power,
        rx_positions,
        float(frequency_hz),
        tx_polarizations,
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel.bdpt_los_export must return a dict")
    validate_cuda_tensor(
        "path_gain_matrix", exported["path_gain_matrix"], dtype=torch.float32, ndim=2
    )
    return exported


def bdpt_los_component_maps(los: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("los", los, dtype=torch.float32, ndim=3)
    maps = _required_native_op("bdpt_los_component_maps")(los)
    if not isinstance(maps, torch.Tensor):
        raise TypeError("_channel.bdpt_los_component_maps must return a tensor")
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    if maps.shape != los.shape:
        raise ValueError(
            "_channel.bdpt_los_component_maps returned an unexpected shape"
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
            "_channel.bdpt_los_component_maps_from_matrix must return a tensor"
        )
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    if maps.shape != (los.shape[0], int(cols), int(rows)):
        raise ValueError(
            "_channel.bdpt_los_component_maps_from_matrix returned an unexpected shape"
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
        raise TypeError("_channel.bdpt_los_visibility_inputs must return a dict")
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
            "_channel.bdpt_apply_los_visibility must return a tensor"
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
            "_channel.bdpt_component_map_buffer must return a tensor"
        )
    validate_cuda_tensor("maps", maps, dtype=torch.float32, ndim=3)
    if maps.shape != (tx_count, dim0, dim1):
        raise ValueError(
            "_channel.bdpt_component_map_buffer returned an unexpected shape"
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
        raise TypeError("_channel.bdpt_store_component_map must return a tensor")
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
            "_channel.bdpt_store_scaled_component_map must return a tensor"
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
            "_channel.bdpt_finalize_component_maps must return a dict"
        )
    validate_cuda_tensor(
        "path_gain", exported["path_gain"], dtype=torch.float32, ndim=3
    )
    if exported["path_gain"].shape != los.shape:
        raise ValueError(
            "_channel.bdpt_finalize_component_maps returned bad path_gain shape"
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


# ---------------------------------------------------------------------------
# ADR-022 finalize AD companion facades (plan 10a section 6.5 / 6.6). The
# finalize map is linear (elementwise sum into path_gain + per-component 0-dim
# power reductions), so the backward is the transpose scaling and the jvp is the
# forward map on the tangents; both are deterministic with no atomics. Pure
# dispatch, no Torch physics.
# ---------------------------------------------------------------------------

_BDPT_FINALIZE_COMPONENT_GRADS = (
    "grad_los",
    "grad_reflection",
    "grad_diffraction",
    "grad_transmission",
    "grad_scattering",
)
_BDPT_FINALIZE_TANGENTS = (
    "tangent_path_gain",
    "tangent_los_power",
    "tangent_reflection_power",
    "tangent_diffraction_power",
    "tangent_transmission_power",
    "tangent_scattering_power",
)


def _bdpt_finalize_backward(
    op_name: str,
    ndim: int,
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
    *,
    grad_path_gain: torch.Tensor | None,
    grad_los_power: torch.Tensor | None,
    grad_reflection_power: torch.Tensor | None,
    grad_diffraction_power: torch.Tensor | None,
    grad_transmission_power: torch.Tensor | None,
    grad_scattering_power: torch.Tensor | None,
    need_grad_components: bool,
) -> dict[str, torch.Tensor | None]:
    for name, tensor in (
        ("los", los),
        ("reflection", reflection),
        ("diffraction", diffraction),
        ("transmission", transmission),
        ("scattering", scattering),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=ndim)
        if tensor.shape != los.shape:
            raise ValueError("component tensors must share shape")
    if grad_path_gain is not None:
        validate_cuda_tensor(
            "grad_path_gain",
            grad_path_gain,
            dtype=torch.float32,
            ndim=ndim,
            require_contiguous=False,
        )
    for name, tensor in (
        ("grad_los_power", grad_los_power),
        ("grad_reflection_power", grad_reflection_power),
        ("grad_diffraction_power", grad_diffraction_power),
        ("grad_transmission_power", grad_transmission_power),
        ("grad_scattering_power", grad_scattering_power),
    ):
        if tensor is not None:
            validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=0)
    exported = _required_native_op(op_name)(
        los,
        reflection,
        diffraction,
        transmission,
        scattering,
        grad_path_gain,
        grad_los_power,
        grad_reflection_power,
        grad_diffraction_power,
        grad_transmission_power,
        grad_scattering_power,
        bool(need_grad_components),
    )
    if not isinstance(exported, dict) or set(exported) != set(
        _BDPT_FINALIZE_COMPONENT_GRADS
    ):
        raise TypeError(f"_channel.{op_name} returned unexpected fields")
    return exported


def _bdpt_finalize_jvp(
    op_name: str,
    ndim: int,
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
    *,
    tangent_los: torch.Tensor | None,
    tangent_reflection: torch.Tensor | None,
    tangent_diffraction: torch.Tensor | None,
    tangent_transmission: torch.Tensor | None,
    tangent_scattering: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    for name, tensor in (
        ("los", los),
        ("reflection", reflection),
        ("diffraction", diffraction),
        ("transmission", transmission),
        ("scattering", scattering),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=ndim)
        if tensor.shape != los.shape:
            raise ValueError("component tensors must share shape")
    exported = _required_native_op(op_name)(
        los,
        reflection,
        diffraction,
        transmission,
        scattering,
        tangent_los,
        tangent_reflection,
        tangent_diffraction,
        tangent_transmission,
        tangent_scattering,
    )
    if not isinstance(exported, dict) or set(exported) != set(_BDPT_FINALIZE_TANGENTS):
        raise TypeError(f"_channel.{op_name} returned unexpected fields")
    return exported


def bdpt_finalize_point_components_backward(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
    *,
    grad_path_gain: torch.Tensor | None = None,
    grad_los_power: torch.Tensor | None = None,
    grad_reflection_power: torch.Tensor | None = None,
    grad_diffraction_power: torch.Tensor | None = None,
    grad_transmission_power: torch.Tensor | None = None,
    grad_scattering_power: torch.Tensor | None = None,
    need_grad_components: bool = False,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`bdpt_finalize_point_components` (spec 6.5)."""

    return _bdpt_finalize_backward(
        "bdpt_finalize_point_components_backward",
        2,
        los,
        reflection,
        diffraction,
        transmission,
        scattering,
        grad_path_gain=grad_path_gain,
        grad_los_power=grad_los_power,
        grad_reflection_power=grad_reflection_power,
        grad_diffraction_power=grad_diffraction_power,
        grad_transmission_power=grad_transmission_power,
        grad_scattering_power=grad_scattering_power,
        need_grad_components=need_grad_components,
    )


def bdpt_finalize_point_components_jvp(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
    *,
    tangent_los: torch.Tensor | None = None,
    tangent_reflection: torch.Tensor | None = None,
    tangent_diffraction: torch.Tensor | None = None,
    tangent_transmission: torch.Tensor | None = None,
    tangent_scattering: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`bdpt_finalize_point_components` (spec 6.5)."""

    return _bdpt_finalize_jvp(
        "bdpt_finalize_point_components_jvp",
        2,
        los,
        reflection,
        diffraction,
        transmission,
        scattering,
        tangent_los=tangent_los,
        tangent_reflection=tangent_reflection,
        tangent_diffraction=tangent_diffraction,
        tangent_transmission=tangent_transmission,
        tangent_scattering=tangent_scattering,
    )


def bdpt_finalize_component_maps_backward(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
    *,
    grad_path_gain: torch.Tensor | None = None,
    grad_los_power: torch.Tensor | None = None,
    grad_reflection_power: torch.Tensor | None = None,
    grad_diffraction_power: torch.Tensor | None = None,
    grad_transmission_power: torch.Tensor | None = None,
    grad_scattering_power: torch.Tensor | None = None,
    need_grad_components: bool = False,
) -> dict[str, torch.Tensor | None]:
    """VJP of :func:`bdpt_finalize_component_maps` (spec 6.6, 3-D maps)."""

    return _bdpt_finalize_backward(
        "bdpt_finalize_component_maps_backward",
        3,
        los,
        reflection,
        diffraction,
        transmission,
        scattering,
        grad_path_gain=grad_path_gain,
        grad_los_power=grad_los_power,
        grad_reflection_power=grad_reflection_power,
        grad_diffraction_power=grad_diffraction_power,
        grad_transmission_power=grad_transmission_power,
        grad_scattering_power=grad_scattering_power,
        need_grad_components=need_grad_components,
    )


def bdpt_finalize_component_maps_jvp(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
    *,
    tangent_los: torch.Tensor | None = None,
    tangent_reflection: torch.Tensor | None = None,
    tangent_diffraction: torch.Tensor | None = None,
    tangent_transmission: torch.Tensor | None = None,
    tangent_scattering: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """JVP of :func:`bdpt_finalize_component_maps` (spec 6.6, 3-D maps)."""

    return _bdpt_finalize_jvp(
        "bdpt_finalize_component_maps_jvp",
        3,
        los,
        reflection,
        diffraction,
        transmission,
        scattering,
        tangent_los=tangent_los,
        tangent_reflection=tangent_reflection,
        tangent_diffraction=tangent_diffraction,
        tangent_transmission=tangent_transmission,
        tangent_scattering=tangent_scattering,
    )
