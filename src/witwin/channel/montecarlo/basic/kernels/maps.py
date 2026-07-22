from __future__ import annotations

import torch

from witwin.channel.propagation.topology import path_los_export
from witwin.channel.runtime import torch_compat
from witwin.channel.runtime.autograd_contracts import (
    _ad_checked_tangent,
    _ad_frequency_grad,
    _ad_frequency_tangent,
    _ad_frequency_value,
    _ad_geometry_tangent,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    _ad_reject_fixed_inputs,
    _ad_reject_fixed_tangents,
)
from witwin.channel.runtime.symbols import (
    native_extension,
    required_symbol as _required_native_op,
)
from witwin.channel.runtime.tensor_contracts import validate_cuda_tensor


_LIGHT_SPEED_M_PER_S_AD = 299_792_458.0

_MC_FINALIZE_FIELDS = (
    "path_gain",
    "los_power",
    "reflection_power",
    "diffraction_power",
    "transmission_power",
    "scattering_power",
)


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


class _McFinalizeComponentMapsAdFunction(torch.autograd.Function):
    """Native linear map finalization and power-reduction AD (plan 07 AD-3)."""

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
            _ad_native_tangent_or_none(value)
            for value in (
                t_los,
                t_reflection,
                t_diffraction,
                t_transmission,
                t_scattering,
            )
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
    """Native AD for the visibility-masked matrix-to-map layout (plan 07 AD-3)."""

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


def mc_los_path_gain_backward(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    grad_output: torch.Tensor,
    tx_polarizations: torch.Tensor,
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
    validate_cuda_tensor(
        "tx_polarizations",
        tx_polarizations,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
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
        tx_polarizations,
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
    validate_cuda_tensor("grad_frequency", gradients[3], dtype=torch.float32, ndim=1)
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
    tx_polarizations: torch.Tensor,
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
    validate_cuda_tensor(
        "tx_polarizations",
        tx_polarizations,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
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
        tx_polarizations,
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel_native.mc_los_path_gain_jvp must return a tensor")
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2)
    if out.shape != (tx_positions.shape[0], rx_positions.shape[0]):
        raise ValueError(
            "_channel_native.mc_los_path_gain_jvp returned an unexpected shape"
        )
    return out


class _McLosPathGainAdFunction(torch.autograd.Function):
    """Differentiable LoS power-gain matrix (plan 07 AD-3).

    Differentiable inputs: tx_positions, rx_positions and the carrier
    frequency. tx_power stays fixed under the plan 07 contract; requesting
    its gradient fails loudly. The forward is the primal path_los_export
    kernel; only the path_gain_matrix output is exposed.
    """

    @staticmethod
    def forward(
        tx_positions, tx_power, rx_positions, frequency, frequency_value, tx_pol
    ):
        exported = path_los_export(
            tx_positions,
            tx_power,
            rx_positions,
            tx_pol,
            frequency_hz=frequency_value,
        )
        return exported["path_gain_matrix"]

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        tx_positions, tx_power, rx_positions, frequency, frequency_value, tx_pol = (
            inputs
        )
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (tx_positions, tx_power, rx_positions, tx_pol)
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
            return None, None, None, None, None, None
        tx_positions, tx_power, rx_positions, tx_pol = ctx.saved_tensors
        grad_tx, _grad_power, grad_rx, grad_frequency = mc_los_path_gain_backward(
            tx_positions,
            tx_power,
            rx_positions,
            grad_output,
            tx_pol,
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
            None,
        )

    @staticmethod
    def jvp(ctx, t_tx, t_power, t_rx, t_frequency, _t_frequency_value, _t_tx_pol):
        _ad_reject_fixed_tangents("mc_los_path_gain_ad", ((t_power, "tx_power"),))
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
        tx_positions, tx_power, rx_positions, tx_pol = saved
        with torch_compat.disable_functorch():
            return mc_los_path_gain_jvp(
                _ad_native_tensor(tx_positions),
                _ad_native_tensor(tx_power),
                _ad_native_tensor(rx_positions),
                tangent_tx
                if tangent_tx is not None
                else _ad_native_tensor(tx_positions),
                _ad_native_tensor(tx_power),
                tangent_rx
                if tangent_rx is not None
                else _ad_native_tensor(rx_positions),
                tangent_tx is not None,
                False,
                tangent_rx is not None,
                _ad_native_tensor(tx_pol),
                frequency_hz=ctx.frequency_value,
                frequency_tangent=tangent_frequency,
            )


def mc_los_path_gain_ad(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_polarizations: torch.Tensor,
    *,
    frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> torch.Tensor:
    """Differentiable LoS path-gain matrix (endpoints and frequency).

    ``frequency_value`` optionally carries the precomputed host scalar of
    ``frequency`` (one read per solve at the seam, audit M3); when not
    supplied it is read here, exactly once per apply. ``tx_polarizations`` is
    the fixed per-transmitter polarization (frozen winner) driving the dipole
    sin^2 pattern; its endpoint dependence is differentiated natively.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    return _McLosPathGainAdFunction.apply(
        tx_positions,
        tx_power,
        rx_positions,
        frequency,
        float(frequency_value),
        tx_polarizations,
    )


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
    tx_pol: torch.Tensor,
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
        tx_pol,
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
    tx_pol: torch.Tensor,
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
        tx_pol,
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
    tx_pol: torch.Tensor,
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
        tx_pol,
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
            tx_pol=params["tx_pol"],
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
            torch.autograd.forward_ad.unpack_dual(value).primal for value in inputs[:5]
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
            tx_pol=params["tx_pol"],
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
            if (
                _ad_native_tangent_or_none(
                    t_anchor if isinstance(t_anchor, torch.Tensor) else None
                )
                is None
            ):
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
                tx_pol=params["tx_pol"],
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
                wavelength_tangent=(-wavelength * wavelength / _LIGHT_SPEED_M_PER_S_AD)
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
    tx_pol: torch.Tensor,
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
        "tx_pol": tx_pol,
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
    tx_pol: torch.Tensor,
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
        tx_pol,
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
    tx_pol: torch.Tensor,
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
        tx_pol,
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
            params["tx_pol"],
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal for value in inputs[:5]
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
            tx_pol=params["tx_pol"],
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
                tx_pol=params["tx_pol"],
                grid_axis=params["grid_axis"],
                grid_position=params["grid_position"],
                grid_resolution0=params["grid_resolution0"],
                grid_resolution1=params["grid_resolution1"],
                wavelength=wavelength,
                grid_cell_area=params["grid_cell_area"],
                seed=params["seed"],
                total_edge_length=params["total_edge_length"],
                wavelength_tangent=(-wavelength * wavelength / _LIGHT_SPEED_M_PER_S_AD)
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
    tx_pol: torch.Tensor,
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
        "tx_pol": tx_pol,
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
