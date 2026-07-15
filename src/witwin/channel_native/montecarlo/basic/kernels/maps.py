from __future__ import annotations

import torch

from witwin.channel_native.propagation.topology import path_los_export
from witwin.channel_native.runtime import torch_compat
from witwin.channel_native.runtime.autograd_contracts import (
    _ad_frequency_grad,
    _ad_frequency_tangent,
    _ad_frequency_value,
    _ad_geometry_tangent,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    _ad_reject_fixed_inputs,
    _ad_reject_fixed_tangents,
)
from witwin.channel_native.runtime.symbols import (
    native_extension,
    required_symbol as _required_native_op,
)
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor


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
