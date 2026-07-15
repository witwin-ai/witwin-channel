from __future__ import annotations

import torch

from witwin.channel_native.runtime import torch_compat
from witwin.channel_native.runtime.autograd_contracts import (
    _ad_native_tangent_or_none,
    _ad_native_tensor,
)
from witwin.channel_native.runtime.symbols import (
    native_extension,
    required_symbol as _required_native_op,
)
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor


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
