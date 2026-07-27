from __future__ import annotations

import torch

from witwin.channel.runtime import torch_compat
from witwin.channel.runtime.autograd_contracts import (
    _ad_first_order_only,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
)
from witwin.channel.runtime.capacity import (
    CapacityFailureState,
    require_capacity_failure_state,
)
from witwin.channel.runtime.symbols import required_symbol as _required_native_op
from witwin.channel.runtime.tensor_contracts import validate_cuda_tensor


_MC_COMPONENT_MAP_FIELDS = (
    "los",
    "reflection",
    "diffraction",
    "transmission",
    "scattering",
)


def _validate_capacity_component_maps(
    maps: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if len(maps) != len(_MC_COMPONENT_MAP_FIELDS):
        raise ValueError("MC capacity sanitizer requires five component maps")
    reference = maps[0]
    for name, value in zip(_MC_COMPONENT_MAP_FIELDS, maps, strict=True):
        validate_cuda_tensor(name, value, dtype=torch.float32, ndim=3)
        if value.shape != reference.shape:
            raise ValueError(f"{name} must match los shape")
        if value.device != reference.device:
            raise ValueError(f"{name} must share the los device")
    return reference


def _capacity_component_maps_result(
    exported: object,
    *,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if not isinstance(exported, dict) or set(exported) != set(_MC_COMPONENT_MAP_FIELDS):
        raise TypeError("native MC capacity map sanitizer returned bad fields")
    values = tuple(exported[name] for name in _MC_COMPONENT_MAP_FIELDS)
    for name, value in zip(_MC_COMPONENT_MAP_FIELDS, values, strict=True):
        validate_cuda_tensor(name, value, dtype=torch.float32, ndim=3)
        if value.shape != reference.shape or value.device != reference.device:
            raise ValueError(f"native MC capacity sanitizer returned bad {name}")
    return values


def _mc_capacity_failure_component_maps_sanitize_native(
    failure_state_bits: torch.Tensor,
    *maps: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    reference = _validate_capacity_component_maps(maps)
    exported = _required_native_op("mc_capacity_failure_component_maps_sanitize")(
        failure_state_bits, *maps
    )
    return _capacity_component_maps_result(exported, reference=reference)


def _mc_capacity_failure_component_maps_sanitize_backward_native(
    failure_state_bits: torch.Tensor,
    reference: torch.Tensor,
    *gradients: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    exported = _required_native_op(
        "mc_capacity_failure_component_maps_sanitize_backward"
    )(failure_state_bits, reference, *gradients)
    return _capacity_component_maps_result(exported, reference=reference)


def _mc_capacity_failure_component_maps_sanitize_jvp_native(
    failure_state_bits: torch.Tensor,
    reference: torch.Tensor,
    *tangents: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    exported = _required_native_op("mc_capacity_failure_component_maps_sanitize_jvp")(
        failure_state_bits, reference, *tangents
    )
    return _capacity_component_maps_result(exported, reference=reference)


class _McCapacityFailureComponentMapsSanitizeFunction(torch.autograd.Function):
    """Native identity/zero transaction boundary for all MC Basic maps."""

    @staticmethod
    def forward(failure_state_bits, *maps):
        return _mc_capacity_failure_component_maps_sanitize_native(
            failure_state_bits, *maps
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        failure_state_bits = torch.autograd.forward_ad.unpack_dual(inputs[0]).primal
        reference = torch.autograd.forward_ad.unpack_dual(output[0]).primal
        ctx.save_for_backward(failure_state_bits, reference)
        ctx.save_for_forward(failure_state_bits, reference)

    @staticmethod
    @_ad_first_order_only
    @torch.autograd.function.once_differentiable
    def backward(ctx, *gradients):
        if not any(ctx.needs_input_grad[1:]):
            return (None,) * 6
        failure_state_bits, reference = ctx.saved_tensors
        values = _mc_capacity_failure_component_maps_sanitize_backward_native(
            failure_state_bits, reference, *gradients
        )
        return (
            None,
            *(
                value if ctx.needs_input_grad[index] else None
                for index, value in enumerate(values, start=1)
            ),
        )

    @staticmethod
    def jvp(ctx, _failure_state_tangent, *tangents):
        tangents = tuple(_ad_native_tangent_or_none(value) for value in tangents)
        if all(value is None for value in tangents):
            return (None,) * len(_MC_COMPONENT_MAP_FIELDS)
        failure_state_bits, reference = (
            _ad_native_tensor(value) for value in ctx.saved_tensors
        )
        with torch_compat.disable_functorch():
            return _mc_capacity_failure_component_maps_sanitize_jvp_native(
                failure_state_bits, reference, *tangents
            )


def mc_capacity_failure_component_maps_sanitize(
    los: torch.Tensor,
    reflection: torch.Tensor,
    diffraction: torch.Tensor,
    transmission: torch.Tensor,
    scattering: torch.Tensor,
    *,
    failure_state: CapacityFailureState,
) -> dict[str, torch.Tensor]:
    """Make every MC Basic component map inert after transaction failure."""

    maps = (los, reflection, diffraction, transmission, scattering)
    reference = _validate_capacity_component_maps(maps)
    require_capacity_failure_state(failure_state, device=reference.device)
    values = _McCapacityFailureComponentMapsSanitizeFunction.apply(
        failure_state.bits, *maps
    )
    return dict(zip(_MC_COMPONENT_MAP_FIELDS, values, strict=True))
