"""Native MC straight-penetration wall-product contracts (ADR-027)."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from witwin.channel_native.runtime import torch_compat
from witwin.channel_native.runtime.autograd_contracts import (
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
from witwin.channel_native.runtime.capacity import (
    CapacityFailureBit,
    CapacityFailureState,
    require_capacity_failure_state,
)
from witwin.channel_native.runtime.symbols import required_symbol
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor


_CONTRACT_FAILURE_BIT = int(CapacityFailureBit.PAIR_CONTRACT_ERROR)


@dataclass(frozen=True, slots=True)
class McTransmissionWallProduct:
    """Fixed-capacity resident outputs of the MC wall-product estimator."""

    scaled_power: torch.Tensor
    transmittance: torch.Tensor
    wall_count: torch.Tensor
    penetrated: torch.Tensor


def _validate_inputs(
    valid: torch.Tensor,
    num_hits: torch.Tensor,
    reached_target: torch.Tensor,
    direction: torch.Tensor,
    normal: torch.Tensor,
    global_primitive_id: torch.Tensor,
    face_material_id: torch.Tensor,
    geometry_mode_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    pair_polarization: torch.Tensor,
    base_power: torch.Tensor,
    failure_state: CapacityFailureState,
    *,
    frequency_hz: float,
) -> tuple[int, int]:
    validate_cuda_tensor("valid", valid, dtype=torch.bool, ndim=2)
    rows, hit_capacity = valid.shape
    validate_cuda_tensor("num_hits", num_hits, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("reached_target", reached_target, dtype=torch.bool, ndim=1)
    validate_cuda_tensor(
        "direction", direction, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "normal", normal, dtype=torch.float32, ndim=3, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "global_primitive_id", global_primitive_id, dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "face_material_id", face_material_id, dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "geometry_mode_id", geometry_mode_id, dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor("layer_offset", layer_offset, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("layer_count", layer_count, dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "layer_thickness_m", layer_thickness_m, dtype=torch.float32, ndim=1
    )
    validate_cuda_tensor("layer_eps_r", layer_eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("layer_sigma_e", layer_sigma_e, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("layer_mu_r", layer_mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "pair_polarization",
        pair_polarization,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor("base_power", base_power, dtype=torch.float32, ndim=1)
    if num_hits.shape != (rows,):
        raise ValueError("num_hits must match valid rows")
    if reached_target.shape != (rows,):
        raise ValueError("reached_target must match valid rows")
    if direction.shape != (rows, 3):
        raise ValueError("direction must have shape (N, 3)")
    if normal.shape != (rows, hit_capacity, 3):
        raise ValueError("normal must have shape (N, D, 3)")
    if global_primitive_id.shape != valid.shape:
        raise ValueError("global_primitive_id must match valid")
    if pair_polarization.shape != (rows, 3):
        raise ValueError("pair_polarization must have shape (N, 3)")
    if base_power.shape != (rows,):
        raise ValueError("base_power must match valid rows")
    material_count = layer_offset.shape[0]
    if layer_count.shape != (material_count,):
        raise ValueError("layer_count must match layer_offset")
    if geometry_mode_id.shape != (material_count,):
        raise ValueError("geometry_mode_id must match material rows")
    layer_length = layer_thickness_m.shape[0]
    if any(
        tensor.shape != (layer_length,)
        for tensor in (layer_eps_r, layer_sigma_e, layer_mu_r)
    ):
        raise ValueError("layer property tensors must have one shared length")
    device = valid.device
    for name, tensor in (
        ("num_hits", num_hits),
        ("reached_target", reached_target),
        ("direction", direction),
        ("normal", normal),
        ("global_primitive_id", global_primitive_id),
        ("face_material_id", face_material_id),
        ("geometry_mode_id", geometry_mode_id),
        ("layer_offset", layer_offset),
        ("layer_count", layer_count),
        ("layer_thickness_m", layer_thickness_m),
        ("layer_eps_r", layer_eps_r),
        ("layer_sigma_e", layer_sigma_e),
        ("layer_mu_r", layer_mu_r),
        ("pair_polarization", pair_polarization),
        ("base_power", base_power),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must share the valid device")
    require_capacity_failure_state(failure_state, device=device)
    _validate_frequency_hz(frequency_hz)
    return int(rows), int(hit_capacity)


def _arguments(
    valid: torch.Tensor,
    num_hits: torch.Tensor,
    reached_target: torch.Tensor,
    direction: torch.Tensor,
    normal: torch.Tensor,
    global_primitive_id: torch.Tensor,
    face_material_id: torch.Tensor,
    geometry_mode_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    pair_polarization: torch.Tensor,
    base_power: torch.Tensor,
    frequency_hz: float,
    failure_state: CapacityFailureState,
) -> tuple[object, ...]:
    return (
        valid,
        num_hits,
        reached_target,
        direction,
        normal,
        global_primitive_id,
        face_material_id,
        geometry_mode_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        pair_polarization,
        base_power,
        float(frequency_hz),
        failure_state.bits,
        _CONTRACT_FAILURE_BIT,
    )


def _result(exported: object, *, rows: int) -> McTransmissionWallProduct:
    if not isinstance(exported, dict):
        raise TypeError("native MC transmission wall product must return a dict")
    scaled_power = exported["scaled_power"]
    transmittance = exported["transmittance"]
    wall_count = exported["wall_count"]
    penetrated = exported["penetrated"]
    validate_cuda_tensor("scaled_power", scaled_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("transmittance", transmittance, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("wall_count", wall_count, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("penetrated", penetrated, dtype=torch.bool, ndim=1)
    if any(
        tensor.shape != (rows,)
        for tensor in (scaled_power, transmittance, wall_count, penetrated)
    ):
        raise ValueError("native MC transmission wall product returned wrong rows")
    return McTransmissionWallProduct(
        scaled_power=scaled_power,
        transmittance=transmittance,
        wall_count=wall_count,
        penetrated=penetrated,
    )


def mc_transmission_wall_product(
    valid: torch.Tensor,
    num_hits: torch.Tensor,
    reached_target: torch.Tensor,
    direction: torch.Tensor,
    normal: torch.Tensor,
    global_primitive_id: torch.Tensor,
    face_material_id: torch.Tensor,
    geometry_mode_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    pair_polarization: torch.Tensor,
    base_power: torch.Tensor,
    failure_state: CapacityFailureState,
    *,
    frequency_hz: float,
) -> McTransmissionWallProduct:
    """Evaluate the live ADR-027 fixed-capacity MC estimator."""

    rows, _ = _validate_inputs(
        valid,
        num_hits,
        reached_target,
        direction,
        normal,
        global_primitive_id,
        face_material_id,
        geometry_mode_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        pair_polarization,
        base_power,
        failure_state,
        frequency_hz=float(frequency_hz),
    )
    exported = required_symbol("mc_transmission_wall_product")(
        *_arguments(
            valid,
            num_hits,
            reached_target,
            direction,
            normal,
            global_primitive_id,
            face_material_id,
            geometry_mode_id,
            layer_offset,
            layer_count,
            layer_thickness_m,
            layer_eps_r,
            layer_sigma_e,
            layer_mu_r,
            pair_polarization,
            base_power,
            float(frequency_hz),
            failure_state,
        )
    )
    return _result(exported, rows=rows)


def mc_transmission_wall_product_backward(
    *inputs: torch.Tensor,
    frequency_hz: float,
    failure_state: CapacityFailureState,
    grad_scaled_power: torch.Tensor | None,
    grad_transmittance: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 16:
        raise ValueError("MC transmission wall-product backward requires 16 inputs")
    rows, _ = _validate_inputs(*inputs, failure_state, frequency_hz=float(frequency_hz))
    for name, gradient in (
        ("grad_scaled_power", grad_scaled_power),
        ("grad_transmittance", grad_transmittance),
    ):
        if gradient is not None:
            validate_cuda_tensor(
                name,
                gradient,
                dtype=torch.float32,
                ndim=1,
                require_contiguous=False,
            )
            if gradient.shape != (rows,):
                raise ValueError(f"{name} must match the output rows")
    exported = required_symbol("mc_transmission_wall_product_backward")(
        *_arguments(*inputs, float(frequency_hz), failure_state),
        grad_scaled_power,
        grad_transmittance,
    )
    if not isinstance(exported, tuple) or len(exported) != 7:
        raise TypeError(
            "native MC transmission wall-product backward must return 7 tensors"
        )
    expected = (
        (torch.float32, inputs[3].shape),
        (torch.float32, inputs[4].shape),
        (torch.float32, inputs[10].shape),
        (torch.float32, inputs[11].shape),
        (torch.float32, inputs[12].shape),
        (torch.float32, inputs[15].shape),
        (torch.float32, (1,)),
    )
    for index, (tensor, (dtype, shape)) in enumerate(
        zip(exported, expected, strict=True)
    ):
        validate_cuda_tensor(f"gradient[{index}]", tensor, dtype=dtype, ndim=len(shape))
        if tensor.shape != shape:
            raise ValueError(f"gradient[{index}] has the wrong shape")
    return exported


def mc_transmission_wall_product_jvp(
    *inputs: torch.Tensor,
    frequency_hz: float,
    failure_state: CapacityFailureState,
    tangent_direction: torch.Tensor | None,
    tangent_normal: torch.Tensor | None,
    tangent_layer_thickness_m: torch.Tensor | None,
    tangent_layer_eps_r: torch.Tensor | None,
    tangent_layer_sigma_e: torch.Tensor | None,
    tangent_base_power: torch.Tensor | None,
    tangent_frequency: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 16:
        raise ValueError("MC transmission wall-product JVP requires 16 inputs")
    rows, _ = _validate_inputs(*inputs, failure_state, frequency_hz=float(frequency_hz))
    exported = required_symbol("mc_transmission_wall_product_jvp")(
        *_arguments(*inputs, float(frequency_hz), failure_state),
        tangent_direction,
        tangent_normal,
        tangent_layer_thickness_m,
        tangent_layer_eps_r,
        tangent_layer_sigma_e,
        tangent_base_power,
        float(tangent_frequency),
    )
    if not isinstance(exported, dict):
        raise TypeError("native MC transmission wall-product JVP must return a dict")
    scaled_power = exported["scaled_power"]
    transmittance = exported["transmittance"]
    for name, tensor in (
        ("scaled_power", scaled_power),
        ("transmittance", transmittance),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=1)
        if tensor.shape != (rows,):
            raise ValueError(
                f"native MC transmission wall-product JVP {name} has wrong rows"
            )
    return scaled_power, transmittance


class _McTransmissionWallProductAd(torch.autograd.Function):
    @staticmethod
    def forward(*inputs):
        frequency_value = inputs[17]
        failure_state = CapacityFailureState(bits=inputs[18])
        result = mc_transmission_wall_product(
            *inputs[:16], failure_state, frequency_hz=frequency_value
        )
        return (
            result.scaled_power,
            result.transmittance,
            result.wall_count,
            result.penetrated,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(output[2], output[3])
        ctx.frequency_value = float(inputs[17])
        ctx.frequency_meta = (
            (inputs[16].dtype, inputs[16].device)
            if isinstance(inputs[16], torch.Tensor)
            else None
        )
        primals = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (*inputs[:16], inputs[18])
        )
        ctx.save_for_backward(*primals)
        ctx.save_for_forward(*primals)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_scaled, grad_transmittance, _grad_count, _grad_penetrated):
        _ad_reject_fixed_inputs(
            "mc_transmission_wall_product_ad",
            ctx.needs_input_grad,
            tuple(
                (index, name)
                for index, name in (
                    (0, "valid"),
                    (1, "num_hits"),
                    (2, "reached_target"),
                    (5, "global_primitive_id"),
                    (6, "face_material_id"),
                    (7, "geometry_mode_id"),
                    (8, "layer_offset"),
                    (9, "layer_count"),
                    (13, "layer_mu_r"),
                    (14, "pair_polarization"),
                    (18, "capacity_failure_state"),
                )
            ),
        )
        live = (3, 4, 10, 11, 12, 15, 16)
        if (grad_scaled is None and grad_transmittance is None) or not any(
            ctx.needs_input_grad[index] for index in live
        ):
            return (None,) * 19
        saved = ctx.saved_tensors
        inputs = saved[:16]
        gradients = mc_transmission_wall_product_backward(
            *inputs,
            frequency_hz=ctx.frequency_value,
            failure_state=CapacityFailureState(bits=saved[16]),
            grad_scaled_power=grad_scaled,
            grad_transmittance=grad_transmittance,
        )
        returned: list[torch.Tensor | None] = [None] * 19
        for input_index, gradient_index in zip(live[:-1], range(6), strict=True):
            if ctx.needs_input_grad[input_index]:
                returned[input_index] = gradients[gradient_index]
        if ctx.needs_input_grad[16]:
            returned[16] = _ad_frequency_grad(gradients[6], ctx.frequency_meta)
        return tuple(returned)

    @staticmethod
    def jvp(ctx, *tangents):
        _ad_reject_fixed_tangents(
            "mc_transmission_wall_product_ad",
            tuple(
                (tangents[index], name)
                for index, name in (
                    (0, "valid"),
                    (1, "num_hits"),
                    (2, "reached_target"),
                    (5, "global_primitive_id"),
                    (6, "face_material_id"),
                    (7, "geometry_mode_id"),
                    (8, "layer_offset"),
                    (9, "layer_count"),
                    (13, "layer_mu_r"),
                    (14, "pair_polarization"),
                    (18, "capacity_failure_state"),
                )
            ),
        )
        saved = ctx.saved_tensors
        inputs = tuple(_ad_native_tensor(value) for value in saved[:16])
        tangent_direction = _ad_geometry_tangent(
            "tangent_direction", tangents[3], inputs[3]
        )
        tangent_normal = _ad_geometry_tangent("tangent_normal", tangents[4], inputs[4])
        continuous_tangents = tuple(
            _ad_checked_tangent(
                name, _ad_native_tangent_or_none(tangent), tuple(primal.shape)
            )
            for name, tangent, primal in (
                ("tangent_layer_thickness_m", tangents[10], inputs[10]),
                ("tangent_layer_eps_r", tangents[11], inputs[11]),
                ("tangent_layer_sigma_e", tangents[12], inputs[12]),
                ("tangent_base_power", tangents[15], inputs[15]),
            )
        )
        tangent_frequency = _ad_frequency_tangent(tangents[16])
        if (
            tangent_direction is None
            and tangent_normal is None
            and all(value is None for value in continuous_tangents)
            and tangent_frequency == 0.0
        ):
            return None, None, None, None
        with torch_compat.disable_functorch():
            scaled, transmittance = mc_transmission_wall_product_jvp(
                *inputs,
                frequency_hz=ctx.frequency_value,
                failure_state=CapacityFailureState(bits=_ad_native_tensor(saved[16])),
                tangent_direction=tangent_direction,
                tangent_normal=tangent_normal,
                tangent_layer_thickness_m=continuous_tangents[0],
                tangent_layer_eps_r=continuous_tangents[1],
                tangent_layer_sigma_e=continuous_tangents[2],
                tangent_base_power=continuous_tangents[3],
                tangent_frequency=tangent_frequency,
            )
        return scaled, transmittance, None, None


def mc_transmission_wall_product_ad(
    valid: torch.Tensor,
    num_hits: torch.Tensor,
    reached_target: torch.Tensor,
    direction: torch.Tensor,
    normal: torch.Tensor,
    global_primitive_id: torch.Tensor,
    face_material_id: torch.Tensor,
    geometry_mode_id: torch.Tensor,
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    pair_polarization: torch.Tensor,
    base_power: torch.Tensor,
    frequency: torch.Tensor | float,
    failure_state: CapacityFailureState,
    *,
    frequency_value: float | None = None,
) -> McTransmissionWallProduct:
    """Differentiable fixed-topology wall product with native VJP/JVP."""

    if frequency_value is None:
        frequency_value = _ad_frequency_value(frequency)
    values = _McTransmissionWallProductAd.apply(
        valid,
        num_hits,
        reached_target,
        direction,
        normal,
        global_primitive_id,
        face_material_id,
        geometry_mode_id,
        layer_offset,
        layer_count,
        layer_thickness_m,
        layer_eps_r,
        layer_sigma_e,
        layer_mu_r,
        pair_polarization,
        base_power,
        frequency,
        float(frequency_value),
        failure_state.bits,
    )
    return McTransmissionWallProduct(*values)


def _validate_frequency_hz(frequency_hz: float) -> None:
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be finite and positive")


__all__ = [
    "McTransmissionWallProduct",
    "mc_transmission_wall_product",
    "mc_transmission_wall_product_ad",
    "mc_transmission_wall_product_backward",
    "mc_transmission_wall_product_jvp",
]
