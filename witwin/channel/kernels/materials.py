# Copyright Xingyu Chen.
# Native material kernel facades.

"""Native material kernel facades."""

from __future__ import annotations

import torch

from witwin.channel.runtime import (
    _ad_checked_tangent,
    _ad_first_order_only,
    _ad_frequency_grad,
    _ad_frequency_tangent,
    _ad_frequency_value,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    _ad_reject_fixed_inputs,
    _ad_reject_fixed_tangents,
    disable_functorch,
    required_symbol as _required_native_op,
    validate_cuda_tensor,
)

__all__ = [
    "_EmLayerStackAdFunction",
    "_validate_layer_csr",
    "bdpt_face_material_tensors",
    "bdpt_face_material_tensors_from_host",
    "em_layer_stack_ad",
    "em_layer_stack_backward",
    "em_layer_stack_eval",
    "em_layer_stack_jvp",
    "mc_face_material_tensors",
    "validate_layer_csr",
]


# ---------------------------------------------------------------------------
# contracts
# ---------------------------------------------------------------------------
def _validate_layer_csr(
    layer_offset: torch.Tensor, layer_count: torch.Tensor, layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor, layer_sigma_e: torch.Tensor, layer_mu_r: torch.Tensor, device: int,
) -> None:
    validate_cuda_tensor("layer_offset", layer_offset, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("layer_count", layer_count, dtype=torch.int32, ndim=1)
    if layer_count.shape != layer_offset.shape:
        raise ValueError("layer_count must match layer_offset length")
    for name, tensor in (
        ("layer_thickness_m", layer_thickness_m),
        ("layer_eps_r", layer_eps_r),
        ("layer_sigma_e", layer_sigma_e),
        ("layer_mu_r", layer_mu_r),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=1)
        if tensor.shape != layer_thickness_m.shape:
            raise ValueError(f"{name} must match layer_thickness_m length")
    for name, tensor in (
        ("layer_offset", layer_offset),
        ("layer_count", layer_count),
        ("layer_thickness_m", layer_thickness_m),
        ("layer_eps_r", layer_eps_r),
        ("layer_sigma_e", layer_sigma_e),
        ("layer_mu_r", layer_mu_r),
    ):
        if tensor.get_device() != device:
            raise ValueError(f"{name} must share the op device")


validate_layer_csr = _validate_layer_csr


# ---------------------------------------------------------------------------
# functional
# ---------------------------------------------------------------------------
def bdpt_face_material_tensors(
    material_eps_r: torch.Tensor, material_sigma_e: torch.Tensor, material_mu_r: torch.Tensor,
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
        raise TypeError("_channel.bdpt_face_material_tensors must return a dict")
    validate_cuda_tensor("eps_r", exported["eps_r"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("sigma_e", exported["sigma_e"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("mu_r", exported["mu_r"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("gain", exported["gain"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("valid", exported["valid"], dtype=torch.bool, ndim=1)
    return exported


def bdpt_face_material_tensors_from_host(
    material_eps_r: tuple[float, ...], material_sigma_e: tuple[float, ...],
    material_mu_r: tuple[float, ...], face_material_id: tuple[int, ...],
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
            "_channel.bdpt_face_material_tensors_from_host must return a dict"
        )
    validate_cuda_tensor("eps_r", exported["eps_r"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("sigma_e", exported["sigma_e"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("mu_r", exported["mu_r"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("gain", exported["gain"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("valid", exported["valid"], dtype=torch.bool, ndim=1)
    return exported


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
        raise TypeError("_channel.em_layer_stack_eval must return a dict")
    if set(out) != set(_EM_LAYER_STACK_FIELDS):
        raise ValueError("em_layer_stack_eval returned unexpected fields")
    for name in _EM_LAYER_STACK_FIELDS:
        validate_cuda_tensor(name, out[name], dtype=torch.float32, ndim=1)
        if out[name].shape != (count,):
            raise ValueError(f"em_layer_stack_eval returned bad {name} shape")
    return out


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
        raise TypeError("_channel.em_layer_stack_jvp must return a dict")
    return out


def mc_face_material_tensors(
    material_eps_r: torch.Tensor, material_sigma_e: torch.Tensor, material_mu_r: torch.Tensor,
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

    exported = _required_native_op("mc_face_material_tensors")(
        material_eps_r,
        material_sigma_e,
        material_mu_r,
        face_material_id,
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel.mc_face_material_tensors must return a dict")
    return exported


# ---------------------------------------------------------------------------
# autograd
# ---------------------------------------------------------------------------
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
        cos_theta, material_id, layer_offset, layer_count, layer_thickness_m, layer_eps_r,
        layer_sigma_e, layer_mu_r, frequency, frequency_value,
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
    @_ad_first_order_only
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
        ctx, t_cos_theta, _t_material_id, _t_layer_offset, _t_layer_count, t_layer_thickness,
        t_layer_eps_r, t_layer_sigma_e, t_layer_mu_r, t_frequency, _t_frequency_value,
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
    """Differentiable:func:`em_layer_stack_eval` (solver derivatives).

 ``frequency_value`` is the precomputed host scalar of ``frequency``; a
 seam that applies several Functions per solve reads the 0-d tensor once
 and threads the float here so no Function re-reads it. When
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