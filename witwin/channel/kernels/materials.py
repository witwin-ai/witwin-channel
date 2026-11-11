# Copyright Xingyu Chen.
# Native material kernel facades.

"""Native material kernel facades."""

from __future__ import annotations

from typing import Any, cast

import torch

from witwin.channel.kernels.spec import (
    AdInputs,
    NativeOps,
    flag,
    frequency_meta,
    real,
    save,
    tensor,
    value,
)
from witwin.channel.runtime import (
    _ad_first_order_only,
    _ad_frequency_grad,
    _ad_frequency_tangent,
    _ad_frequency_value,
    _ad_native_tensor,
    disable_functorch,
    required_symbol as _required_native_op,
    validate_cuda_tensor,
)

__all__ = [
    "_EmLayerStackAdFunction",
    "_validate_layer_csr",
    "em_layer_stack_ad",
    "em_layer_stack_backward",
    "em_layer_stack_eval",
    "em_layer_stack_jvp",
    "mc_face_material_tensors",
    "validate_layer_csr",
]


_native = NativeOps(lambda symbol: _required_native_op(symbol))

# The CSR layer store shared by every layer-stack operation, in ABI order.
_LAYER_CSR = (
    tensor("layer_offset", torch.int32, 1),
    tensor("layer_count", torch.int32, 1),
    tensor("layer_thickness_m", torch.float32, 1, grad="diff"),
    tensor("layer_eps_r", torch.float32, 1, grad="diff"),
    tensor("layer_sigma_e", torch.float32, 1, grad="diff"),
    tensor("layer_mu_r", torch.float32, 1, grad="fixed"),
)
_STACK_PRIMALS = (
    tensor("cos_theta", torch.float32, 1, grad="diff"),
    tensor("material_id", torch.int32, 1),
    _LAYER_CSR,
)
_EM_LAYER_STACK_EVAL = _native.op("em_layer_stack_eval", _STACK_PRIMALS, real("frequency_hz"))
_EM_LAYER_STACK_BACKWARD = _native.op(
    "em_layer_stack_backward", _STACK_PRIMALS, real("frequency_hz"),
    value("grad_outputs", convert=list), flag("need_cos_theta"), flag("need_layers"),
    flag("need_frequency"),
)
_EM_LAYER_STACK_JVP = _native.op(
    "em_layer_stack_jvp", _STACK_PRIMALS, real("frequency_hz"), value("tangent_cos_theta"),
    value("tangent_layer_thickness"), value("tangent_layer_eps_r"),
    value("tangent_layer_sigma_e"), real("tangent_frequency"),
)
_MC_FACE_MATERIAL_TENSORS = _native.op(
    "mc_face_material_tensors",
    tensor("material_eps_r", torch.float32, 1),
    tensor("material_sigma_e", torch.float32, 1),
    tensor("material_mu_r", torch.float32, 1),
    tensor("face_material_id", torch.int32, 1),
)


# ---------------------------------------------------------------------------
# contracts
# ---------------------------------------------------------------------------
def _validate_layer_csr(
    layer_offset: torch.Tensor, layer_count: torch.Tensor, layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor, layer_sigma_e: torch.Tensor, layer_mu_r: torch.Tensor, device: int,
) -> None:
    csr = (layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r)
    _LAYER_CSR[0].check(layer_offset)
    _LAYER_CSR[1].check(layer_count)
    if layer_count.shape != layer_offset.shape:
        raise ValueError("layer_count must match layer_offset length")
    for arg, tensor_value in zip(_LAYER_CSR[2:], csr[2:], strict=True):
        arg.check(tensor_value)
        if tensor_value.shape != layer_thickness_m.shape:
            raise ValueError(f"{arg.name} must match layer_thickness_m length")
    for arg, tensor_value in zip(_LAYER_CSR, csr, strict=True):
        if tensor_value.get_device() != device:
            raise ValueError(f"{arg.name} must share the op device")


validate_layer_csr = _validate_layer_csr


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
    cos_theta: torch.Tensor, material_id: torch.Tensor, layer_offset: torch.Tensor,
    layer_count: torch.Tensor, layer_thickness_m: torch.Tensor, layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor, layer_mu_r: torch.Tensor, *, frequency_hz: float,
) -> dict[str, torch.Tensor]:
    _EM_LAYER_STACK_EVAL.check((cos_theta, material_id), "cos_theta", "material_id")
    count = int(cos_theta.shape[0])
    if material_id.shape != (count,):
        raise ValueError("material_id must match cos_theta length")
    if material_id.get_device() != cos_theta.get_device():
        raise ValueError("material_id must share cos_theta device")
    _validate_layer_csr(
        layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r,
        cos_theta.get_device(),
    )
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    out = _EM_LAYER_STACK_EVAL(
        cos_theta, material_id, layer_offset, layer_count, layer_thickness_m, layer_eps_r,
        layer_sigma_e, layer_mu_r, frequency_hz,
    )
    if not isinstance(out, dict):
        raise TypeError("_channel.em_layer_stack_eval must return a dict")
    if set(out) != set(_EM_LAYER_STACK_FIELDS):
        raise ValueError("em_layer_stack_eval returned unexpected fields")
    for name in _EM_LAYER_STACK_FIELDS:
        validate_cuda_tensor(name, out[name], dtype=torch.float32, ndim=1)
        if out[name].shape != (count,):
            raise ValueError(f"em_layer_stack_eval returned bad {name} shape")
    return out

# The eight primal tensors the layer-stack AD saves, in ``em_layer_stack_backward`` order.
_LayerStackSaved = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]


def em_layer_stack_backward(
    cos_theta: torch.Tensor, material_id: torch.Tensor, layer_offset: torch.Tensor,
    layer_count: torch.Tensor, layer_thickness_m: torch.Tensor, layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor, layer_mu_r: torch.Tensor,
    grad_outputs: tuple[torch.Tensor | None, ...], *, frequency_hz: float, need_cos_theta: bool,
    need_layers: bool, need_frequency: bool,
) -> dict[str, torch.Tensor]:
    if len(grad_outputs) != len(_EM_LAYER_STACK_FIELDS):
        raise ValueError(
            "grad_outputs must carry one cotangent slot per stack output"
        )
    # Autograd may hand cotangents with arbitrary strides (e.g. slices of the
    # realization reflectance rows); the native ABI requires contiguous
    # cotangents. contiguous is a no-op when the stride is already dense.
    grad_outputs = tuple(
        None if value is None else value.contiguous() for value in grad_outputs
    )
    out = _EM_LAYER_STACK_BACKWARD(
        cos_theta, material_id, layer_offset, layer_count, layer_thickness_m, layer_eps_r,
        layer_sigma_e, layer_mu_r, frequency_hz, grad_outputs, need_cos_theta, need_layers,
        need_frequency,
    )
    if not isinstance(out, dict):
        raise TypeError("_channel.em_layer_stack_backward must return a dict")
    return out


def em_layer_stack_jvp(
    cos_theta: torch.Tensor, material_id: torch.Tensor, layer_offset: torch.Tensor,
    layer_count: torch.Tensor, layer_thickness_m: torch.Tensor, layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor, layer_mu_r: torch.Tensor, *, frequency_hz: float,
    tangent_cos_theta: torch.Tensor | None, tangent_layer_thickness: torch.Tensor | None,
    tangent_layer_eps_r: torch.Tensor | None, tangent_layer_sigma_e: torch.Tensor | None,
    tangent_frequency: float,
) -> dict[str, torch.Tensor]:
    out = _EM_LAYER_STACK_JVP(
        cos_theta, material_id, layer_offset, layer_count, layer_thickness_m, layer_eps_r,
        layer_sigma_e, layer_mu_r, frequency_hz, tangent_cos_theta, tangent_layer_thickness,
        tangent_layer_eps_r, tangent_layer_sigma_e, tangent_frequency,
    )
    if not isinstance(out, dict):
        raise TypeError("_channel.em_layer_stack_jvp must return a dict")
    return out


def mc_face_material_tensors(
    material_eps_r: torch.Tensor, material_sigma_e: torch.Tensor, material_mu_r: torch.Tensor,
    face_material_id: torch.Tensor,
) -> dict[str, torch.Tensor]:
    values = (material_eps_r, material_sigma_e, material_mu_r, face_material_id)
    _MC_FACE_MATERIAL_TENSORS.check(values)
    if material_sigma_e.shape != material_eps_r.shape:
        raise ValueError("material_sigma_e must match material_eps_r shape")
    if material_mu_r.shape != material_eps_r.shape:
        raise ValueError("material_mu_r must match material_eps_r shape")
    exported = _MC_FACE_MATERIAL_TENSORS(*values)
    if not isinstance(exported, dict):
        raise TypeError("_channel.mc_face_material_tensors must return a dict")
    return exported


# ---------------------------------------------------------------------------
# autograd
# ---------------------------------------------------------------------------
_STACK_AD = AdInputs(
    "em_layer_stack_ad", _STACK_PRIMALS, value("frequency", grad="diff"),
    value("frequency_value"),
)
_STACK_PRIMAL_NAMES = _STACK_AD.names[:8]


class _EmLayerStackAdFunction(torch.autograd.Function):
    """Differentiable layer-stack r/t coefficients and power budgets.

    Differentiable inputs: cos_theta (per row), the CSR layer thickness /
    eps_r / sigma_e and the carrier frequency. layer_mu_r and the CSR
    topology stay fixed under the AD contract; requesting the mu_r
    gradient fails loudly. Layer gradients accumulate atomically because the
    CSR store is shared by every row.
    """

    @staticmethod
    def forward(
        cos_theta: Any, material_id: Any, layer_offset: Any, layer_count: Any,
        layer_thickness_m: Any, layer_eps_r: Any, layer_sigma_e: Any, layer_mu_r: Any,
        frequency: Any, frequency_value: Any,
    ) -> Any:
        out = em_layer_stack_eval(
            cos_theta, material_id, layer_offset, layer_count, layer_thickness_m, layer_eps_r,
            layer_sigma_e, layer_mu_r, frequency_hz=frequency_value,
        )
        return tuple(out[name] for name in _EM_LAYER_STACK_FIELDS)

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> Any:
        ctx.set_materialize_grads(False)
        primals = _STACK_AD.primals(inputs, *_STACK_PRIMAL_NAMES)
        ctx.frequency_value = _STACK_AD.get(inputs, "frequency_value")
        ctx.frequency_meta = frequency_meta(_STACK_AD.get(inputs, "frequency"))
        save(ctx, *primals)

    @staticmethod
    @_ad_first_order_only
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        _STACK_AD.reject_fixed_grads(ctx)
        need_cos = _STACK_AD.need(ctx, "cos_theta")
        need_layers = _STACK_AD.needs(ctx, "layer_thickness_m", "layer_eps_r", "layer_sigma_e")
        need_frequency = _STACK_AD.need(ctx, "frequency")
        if not (need_cos or need_layers or need_frequency) or all(
            value is None for value in grad_outputs
        ):
            return _STACK_AD.none
        saved = cast(_LayerStackSaved, ctx.saved_tensors)
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
        return _STACK_AD.grads(
            ctx,
            cos_theta=out["grad_cos_theta"] if need_cos else None,
            layer_thickness_m=out["grad_layer_thickness_m"] if need_layers else None,
            layer_eps_r=out["grad_layer_eps_r"] if need_layers else None,
            layer_sigma_e=out["grad_layer_sigma_e"] if need_layers else None,
            frequency=grad_frequency,
        )

    @staticmethod
    def jvp(ctx: Any, *tangents: Any) -> Any:
        _STACK_AD.reject_fixed_tangents(tangents)
        saved = ctx.saved_tensors
        tangent_cos = _STACK_AD.checked_tangent(tangents, "cos_theta", tuple(saved[0].shape))
        layer_shape = tuple(saved[4].shape)
        tangent_thickness, tangent_eps, tangent_sigma = (
            _STACK_AD.checked_tangent(tangents, name, layer_shape)
            for name in ("layer_thickness_m", "layer_eps_r", "layer_sigma_e")
        )
        tangent_frequency = _ad_frequency_tangent(_STACK_AD.tangent(tangents, "frequency"))
        if (
            tangent_cos is None
            and tangent_thickness is None
            and tangent_eps is None
            and tangent_sigma is None
            and tangent_frequency == 0.0
        ):
            return (None,) * len(_EM_LAYER_STACK_FIELDS)
        with disable_functorch():
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
    cos_theta: torch.Tensor, material_id: torch.Tensor, layer_offset: torch.Tensor,
    layer_count: torch.Tensor, layer_thickness_m: torch.Tensor, layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor, layer_mu_r: torch.Tensor, *, frequency: torch.Tensor | float,
    frequency_value: float | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable :func:`em_layer_stack_eval` (solver derivatives).

    ``frequency_value`` is the precomputed host scalar of ``frequency``; a
    seam that applies several Functions per solve reads the 0-d tensor once
    and threads the float here so no Function re-reads it. When
    not supplied it is read here, exactly once per apply.
    """

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _EmLayerStackAdFunction.apply(
        cos_theta, material_id, layer_offset, layer_count, layer_thickness_m, layer_eps_r,
        layer_sigma_e, layer_mu_r, frequency, float(frequency_value),
    )
    return dict(zip(_EM_LAYER_STACK_FIELDS, values, strict=True))
