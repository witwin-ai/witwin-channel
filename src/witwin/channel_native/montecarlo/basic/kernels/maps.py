from __future__ import annotations

import torch

from witwin.channel_native.propagation.topology import path_los_export
from witwin.channel_native.runtime import torch_compat
from witwin.channel_native.runtime.autograd_contracts import (
    _ad_frequency_grad,
    _ad_frequency_tangent,
    _ad_frequency_value,
    _ad_geometry_tangent,
    _ad_native_tensor,
    _ad_reject_fixed_inputs,
    _ad_reject_fixed_tangents,
)
from witwin.channel_native.runtime.symbols import native_extension
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor


_LIGHT_SPEED_M_PER_S_AD = 299_792_458.0


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
