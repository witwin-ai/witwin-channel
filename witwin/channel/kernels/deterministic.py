# Copyright Xingyu Chen.
# Native deterministic accumulation kernel facades.

"""Native deterministic accumulation kernel facades."""

from __future__ import annotations

from typing import Any, TypedDict, cast

import torch

from witwin.channel.kernels.spec import (
    AdInputs,
    NativeOps,
    flag,
    integer,
    save,
    tensor,
    opaque,
)
from witwin.channel.runtime import (
    _ad_first_order_only,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    disable_functorch,
    required_symbol as _required_native_op,
    validate_cuda_tensor,
)


# ---------------------------------------------------------------------------
# accumulation
# ---------------------------------------------------------------------------
class AccumulateFlatOutputs(TypedDict):
    """Per-pair totals and per-component maps of the flat deterministic accumulator."""

    power_total: torch.Tensor
    field_total_real: torch.Tensor
    field_total_imag: torch.Tensor
    component_power: torch.Tensor
    component_field_real: torch.Tensor
    component_field_imag: torch.Tensor


_native = NativeOps(lambda symbol: _required_native_op(symbol))

_ACCUM_IDS = (
    tensor("valid", torch.bool, 1),
    tensor("tx_id", torch.int32, 1),
    tensor("rx_id", torch.int32, 1),
    tensor("component_id", torch.int32, 1),
)
_ACCUM_CONFIG = (
    integer("num_tx"), integer("num_rx"), flag("coherent"), integer("scattering_combine_domain")
)
_ACCUM_PRIMALS = (
    _ACCUM_IDS,
    tensor("path_gain", torch.float32, 1, grad="diff"),
    tensor("field_real", torch.float32, 1, grad="diff"),
    tensor("field_imag", torch.float32, 1, grad="diff"),
)
_ACCUMULATE_FLAT = _native.op("deterministic_accumulate_flat", _ACCUM_PRIMALS, _ACCUM_CONFIG)
_ACCUMULATE_FLAT_FWD64 = _native.op(
    "deterministic_accumulate_flat_fwd64", _ACCUM_PRIMALS, _ACCUM_CONFIG
)
_ACCUMULATE_FLAT_BACKWARD = _native.op(
    "deterministic_accumulate_flat_backward", _ACCUM_IDS, tensor("component_field_real"),
    tensor("component_field_imag"), tensor("field_total_real"), tensor("field_total_imag"),
    tensor("power_total"), opaque("grad_power_total"), opaque("grad_field_total_real"),
    opaque("grad_field_total_imag"), opaque("grad_component_power"),
    opaque("grad_component_field_real"), opaque("grad_component_field_imag"), _ACCUM_CONFIG,
)
_ACCUMULATE_FLAT_JVP = _native.op(
    "deterministic_accumulate_flat_jvp", _ACCUM_IDS, tensor("component_field_real"),
    tensor("component_field_imag"), tensor("power_total"), opaque("tangent_path_gain"),
    opaque("tangent_field_real"), opaque("tangent_field_imag"), _ACCUM_CONFIG,
)
# Output name, rank of the six accumulator outputs.
_ACCUM_OUTPUT_RANKS = (
    ("power_total", 2),
    ("field_total_real", 2),
    ("field_total_imag", 2),
    ("component_power", 3),
    ("component_field_real", 3),
    ("component_field_imag", 3),
)


def deterministic_accumulate_flat(
    valid: torch.Tensor, tx_id: torch.Tensor, rx_id: torch.Tensor, component_id: torch.Tensor,
    path_gain: torch.Tensor, field_real: torch.Tensor, field_imag: torch.Tensor, *, num_tx: int,
    num_rx: int, coherent: bool, scattering_combine_domain: int = 0,
) -> AccumulateFlatOutputs:
    if int(scattering_combine_domain) not in (0, 1):
        raise ValueError("scattering_combine_domain must be 0 (power) or 1 (coherent)")
    tensors = (valid, tx_id, rx_id, component_id, path_gain, field_real, field_imag)
    _ACCUMULATE_FLAT.check(tensors, *_ACCUMULATE_FLAT.names[:7])
    for name, tensor_value in zip(_ACCUMULATE_FLAT.names, tensors):
        if name != "tx_id" and tensor_value.shape != tx_id.shape:
            raise ValueError(f"{name} must match tx_id shape")
    if num_tx < 0 or num_rx < 0:
        raise ValueError("num_tx and num_rx must be non-negative")

    exported: AccumulateFlatOutputs = _ACCUMULATE_FLAT(
        *tensors, num_tx, num_rx, coherent, scattering_combine_domain
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_accumulate_flat must return a dict"
        )
    outputs = cast(dict[str, torch.Tensor], exported)
    for name, ndim in _ACCUM_OUTPUT_RANKS:
        validate_cuda_tensor(name, outputs[name], dtype=torch.float32, ndim=ndim)
    # Six materialized slots (coupled reflection and diffraction): los / reflection / diffraction /
    # transmission / scattering / coupled.
    expected_component_shape = (6, int(num_tx), int(num_rx))
    if tuple(exported["power_total"].shape) != (int(num_tx), int(num_rx)):
        raise ValueError(
            "_channel.deterministic_accumulate_flat returned bad power_total shape"
        )
    if tuple(exported["component_power"].shape) != expected_component_shape:
        raise ValueError(
            "_channel.deterministic_accumulate_flat returned bad component shape"
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
    valid: torch.Tensor, tx_id: torch.Tensor, rx_id: torch.Tensor, component_id: torch.Tensor,
    component_field_real: torch.Tensor, component_field_imag: torch.Tensor,
    field_total_real: torch.Tensor, field_total_imag: torch.Tensor, power_total: torch.Tensor, *,
    grad_power_total: torch.Tensor | None = None, grad_field_total_real: torch.Tensor | None = None,
    grad_field_total_imag: torch.Tensor | None = None,
    grad_component_power: torch.Tensor | None = None,
    grad_component_field_real: torch.Tensor | None = None,
    grad_component_field_imag: torch.Tensor | None = None, num_tx: int, num_rx: int, coherent: bool,
    scattering_combine_domain: int = 0,
) -> dict[str, torch.Tensor]:
    out = _ACCUMULATE_FLAT_BACKWARD(
        valid, tx_id, rx_id, component_id, component_field_real, component_field_imag,
        field_total_real, field_total_imag, power_total, grad_power_total, grad_field_total_real,
        grad_field_total_imag, grad_component_power, grad_component_field_real,
        grad_component_field_imag, num_tx, num_rx, coherent, scattering_combine_domain,
    )
    expected = {"grad_path_gain", "grad_field_real", "grad_field_imag"}
    if not isinstance(out, dict) or set(out) != expected:
        raise TypeError(
            "_channel.deterministic_accumulate_flat_backward returned"
            " invalid fields"
        )
    return out


def deterministic_accumulate_flat_jvp(
    valid: torch.Tensor, tx_id: torch.Tensor, rx_id: torch.Tensor, component_id: torch.Tensor,
    component_field_real: torch.Tensor, component_field_imag: torch.Tensor,
    power_total: torch.Tensor, *, tangent_path_gain: torch.Tensor | None = None,
    tangent_field_real: torch.Tensor | None = None, tangent_field_imag: torch.Tensor | None = None,
    num_tx: int, num_rx: int, coherent: bool, scattering_combine_domain: int = 0,
) -> dict[str, torch.Tensor]:
    out = _ACCUMULATE_FLAT_JVP(
        valid, tx_id, rx_id, component_id, component_field_real, component_field_imag,
        power_total, tangent_path_gain, tangent_field_real, tangent_field_imag, num_tx, num_rx,
        coherent, scattering_combine_domain,
    )
    if not isinstance(out, dict) or set(out) != set(_DETERMINISTIC_ACCUM_FIELDS):
        raise TypeError(
            "_channel.deterministic_accumulate_flat_jvp returned invalid fields"
        )
    return out


_ACCUM_AD = AdInputs("deterministic_accumulate_flat_ad", _ACCUM_PRIMALS, _ACCUM_CONFIG)


class _DeterministicAccumulateFlatAdFunction(torch.autograd.Function):
    """Differentiable deterministic flat-path accumulation (the AD contract).

    The forward is the primal native accumulator: each kept path's complex
    field and real power scatter into a frozen (component_slot, tx, rx)
    cell over the six slots los / reflection / diffraction / transmission /
    scattering / coupled, then coherent cells square the summed field
    (|sum E|^2 over the five coherent field slots) while incoherent cells sum
    per-path powers and expose a sqrt-power pseudo-field. Scattering is a
    power-domain slot in
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
        valid: Any, tx_id: Any, rx_id: Any, component_id: Any, path_gain: Any, field_real: Any,
        field_imag: Any, num_tx: Any, num_rx: Any, coherent: Any, scattering_combine_domain: Any,
    ) -> Any:
        op = _ACCUMULATE_FLAT_FWD64 if path_gain.dtype == torch.float64 else _ACCUMULATE_FLAT
        out = op(
            valid, tx_id, rx_id, component_id, path_gain, field_real, field_imag, num_tx, num_rx,
            coherent, scattering_combine_domain,
        )
        return tuple(out[name] for name in _DETERMINISTIC_ACCUM_FIELDS)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        valid, tx_id, rx_id, component_id = _ACCUM_AD.primals(
            inputs, "valid", "tx_id", "rx_id", "component_id"
        )
        named = _ACCUM_AD.unpack(inputs)
        ctx.num_tx = int(named["num_tx"])
        ctx.num_rx = int(named["num_rx"])
        ctx.coherent = bool(named["coherent"])
        ctx.scattering_combine_domain = int(named["scattering_combine_domain"])
        (
            power_total,
            field_total_real,
            field_total_imag,
            _component_power,
            component_field_real,
            component_field_imag,
        ) = output
        save(
            ctx, valid, tx_id, rx_id, component_id, component_field_real, component_field_imag,
            field_total_real, field_total_imag, power_total,
        )

    @staticmethod
    @_ad_first_order_only
    def backward(
        ctx: Any, grad_power_total: Any, grad_field_total_real: Any, grad_field_total_imag: Any,
        grad_component_power: Any, grad_component_field_real: Any, grad_component_field_imag: Any,
    ) -> Any:
        need_gain = _ACCUM_AD.need(ctx, "path_gain")
        need_field = _ACCUM_AD.needs(ctx, "field_real", "field_imag")
        grads = (
            grad_power_total,
            grad_field_total_real,
            grad_field_total_imag,
            grad_component_power,
            grad_component_field_real,
            grad_component_field_imag,
        )
        if not (need_gain or need_field) or all(value is None for value in grads):
            return _ACCUM_AD.none
        out = deterministic_accumulate_flat_backward(
            *ctx.saved_tensors,
            grad_power_total=grad_power_total,
            grad_field_total_real=grad_field_total_real,
            grad_field_total_imag=grad_field_total_imag,
            grad_component_power=grad_component_power,
            grad_component_field_real=grad_component_field_real,
            grad_component_field_imag=grad_component_field_imag,
            num_tx=ctx.num_tx,
            num_rx=ctx.num_rx,
            coherent=ctx.coherent,
            scattering_combine_domain=ctx.scattering_combine_domain,
        )
        return _ACCUM_AD.grads(
            ctx, path_gain=out["grad_path_gain"], field_real=out["grad_field_real"],
            field_imag=out["grad_field_imag"],
        )

    @staticmethod
    def jvp(ctx: Any, *tangents: Any) -> Any:
        tangent_gain, tangent_real, tangent_imag = (
            _ad_native_tangent_or_none(_ACCUM_AD.tangent(tangents, name))
            for name in ("path_gain", "field_real", "field_imag")
        )
        if tangent_gain is None and tangent_real is None and tangent_imag is None:
            return (None,) * len(_DETERMINISTIC_ACCUM_FIELDS)
        (
            valid,
            tx_id,
            rx_id,
            component_id,
            component_field_real,
            component_field_imag,
            _field_total_real,
            _field_total_imag,
            power_total,
        ) = (_ad_native_tensor(value) for value in ctx.saved_tensors)
        with disable_functorch():
            out = deterministic_accumulate_flat_jvp(
                valid,
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
                scattering_combine_domain=ctx.scattering_combine_domain,
            )
        return tuple(out[name] for name in _DETERMINISTIC_ACCUM_FIELDS)


def deterministic_accumulate_flat_ad(
    valid: torch.Tensor, tx_id: torch.Tensor, rx_id: torch.Tensor, component_id: torch.Tensor,
    path_gain: torch.Tensor, field_real: torch.Tensor, field_imag: torch.Tensor, *, num_tx: int,
    num_rx: int, coherent: bool, scattering_combine_domain: int = 0,
) -> AccumulateFlatOutputs:
    """Differentiable :func:`deterministic_accumulate_flat` (the AD contract)."""

    values = _DeterministicAccumulateFlatAdFunction.apply(
        valid, tx_id, rx_id, component_id, path_gain, field_real, field_imag, int(num_tx),
        int(num_rx), bool(coherent), int(scattering_combine_domain),
    )
    return cast(
        AccumulateFlatOutputs, dict(zip(_DETERMINISTIC_ACCUM_FIELDS, values, strict=True))
    )
