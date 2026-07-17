"""Shared autograd validation and transform contracts for native facades."""

from __future__ import annotations

import torch

from witwin.channel_native.runtime import torch_compat
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor


def _ad_still_wrapped(value: torch.Tensor) -> bool:
    return torch_compat.is_transform_wrapped_tensor(value)


def _ad_raise_composed_transforms() -> None:
    # Plan 07 section 7 contract: fail loudly instead of feeding the native
    # kernels an unwrapped tensor that has silently lost its transform
    # tracking (which would produce exact-zero tangents/gradients).
    raise NotImplementedError(
        "raydn_*_ad entry points support a single forward-mode transform"
        " level; composed functorch transforms (e.g. torch.func.grad over"
        " forward-mode jvp) are not supported by the native geometry kernels"
        " (first-order only)"
    )


def _ad_native_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    if torch_compat.transform_level(value) >= 0:
        # The tensor is functorch-wrapped. Unwrapping is only sound for a
        # single Jvp transform (torch.func.jvp); under nested transforms or
        # a Grad transform (e.g. torch.func.grad over forward-mode jvp, the
        # standard HVP recipe) unwrapping would silently sever the outer
        # transform and return exact zeros.
        stack = torch_compat.interpreter_stack()
        if len(stack) > 1 or any(
            not torch_compat.is_jvp_transform(entry) for entry in stack
        ):
            _ad_raise_composed_transforms()
    value = torch.autograd.forward_ad.unpack_dual(value).primal
    if _ad_still_wrapped(value):
        value = torch_compat.unwrap_transform_tensor(value)
    if _ad_still_wrapped(value):
        _ad_raise_composed_transforms()
    return value


def _ad_native_tangent_or_none(value: torch.Tensor | None) -> torch.Tensor | None:
    value = _ad_native_tensor(value)
    if value is None:
        return None
    try:
        # Efficient zero tangents (ZeroTensor) have no storage; treat them as
        # absent so the kernels take their tangent-free fast path.
        value.data_ptr()
    except RuntimeError:
        return None
    return value


def _ad_checked_tangent(
    name: str,
    tangent: torch.Tensor | None,
    primal_shape: tuple[int, ...],
) -> torch.Tensor | None:
    """Validate an unwrapped jvp tangent against its primal contract.

    Strided tangents are passed through unchanged: the native kernels consume
    explicit strides, so no Python-side layout copy or staging is needed.
    """

    if tangent is None:
        return None
    if tuple(tangent.shape) != tuple(primal_shape):
        raise ValueError(
            f"{name} must match its primal shape {tuple(primal_shape)};"
            f" got {tuple(tangent.shape)}"
        )
    if tangent.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if not tangent.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    return tangent


def _ad_check_rows(name: str, tensor: torch.Tensor, rows: int) -> None:
    if tensor.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows to match the ray batch")


def _ad_check_active(active: torch.Tensor | None, rows: int) -> None:
    if active is None:
        return
    validate_cuda_tensor("active", active, dtype=torch.bool, ndim=1)
    if active.shape[0] not in (0, rows):
        raise ValueError("active must be empty or match the ray batch size")


def _ad_check_optional_grad(
    name: str,
    grad: torch.Tensor | None,
    allowed_shapes: tuple[tuple[int, ...], ...],
) -> None:
    # Cotangents from autograd may be strided views; the native kernels
    # consume explicit strides, so contiguity is deliberately not required.
    if grad is None:
        return
    if not isinstance(grad, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if grad.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if not grad.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tuple(grad.shape) not in allowed_shapes:
        raise ValueError(
            f"{name} must have shape in {allowed_shapes}; got {tuple(grad.shape)}"
        )


def _ad_check_tangent_vec3(
    name: str,
    tangent: torch.Tensor | None,
    rows: int | None,
) -> None:
    """Validate a facade-level jvp tangent.

    ``rows=None`` checks only the ``(V, 3)`` layout; the native entry point
    enforces that a vertex tangent matches the scene's global vertex table.
    Strided tangents are allowed: the native kernels consume explicit strides.
    """

    if tangent is None:
        return
    if not isinstance(tangent, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tangent.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if not tangent.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tangent.ndim != 2 or tangent.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if rows is not None and tangent.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows to match the ray batch")


def _ad_active_ctx(active: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    if active is not None:
        return active
    return torch.empty((0,), device=like.device, dtype=torch.bool)


def _ad_frequency_value(frequency: torch.Tensor | float) -> float:
    """Read the scalar carrier frequency once per solve.

    A 0-d CUDA tensor frequency costs one device-to-host synchronization per
    read (documented plan 07 AD-1 decision: one sync per solve, never per
    path); the native entry points keep a double scalar. The solve seams
    call this once and thread the float to every ``field_*_ad`` facade as
    ``frequency_value`` so no Function pays a second read (audit M3); a
    facade called without it reads here exactly once.
    """

    if isinstance(frequency, torch.Tensor):
        if frequency.ndim != 0:
            raise ValueError("frequency must be a Python float or a 0-d tensor")
        return float(_ad_native_tensor(frequency).detach())
    return float(frequency)


def _ad_frequency_tangent(tangent: torch.Tensor | None) -> float:
    tangent = _ad_native_tangent_or_none(tangent)
    if tangent is None:
        return 0.0
    if tangent.ndim != 0:
        raise ValueError("frequency tangent must be a 0-d tensor")
    return float(tangent.detach())


def _ad_frequency_grad(
    grad_frequency: torch.Tensor, meta: tuple[torch.dtype, torch.device]
) -> torch.Tensor:
    dtype, device = meta
    return grad_frequency.to(dtype=dtype, device=device)[0]


def _ad_reject_fixed_inputs(
    op_name: str,
    needs_input_grad: tuple[bool, ...],
    fixed: tuple[tuple[int, str], ...],
) -> None:
    for index, name in fixed:
        if needs_input_grad[index]:
            raise NotImplementedError(
                f"{op_name} does not differentiate {name}: tx_power, the "
                "polarizations, mu_r, material ids and valid masks stay fixed "
                "under the plan 07 fixed-topology contract"
            )


def _ad_reject_fixed_tangents(
    op_name: str,
    tangents: tuple[tuple[object, str], ...],
) -> None:
    for tangent, name in tangents:
        if isinstance(tangent, torch.Tensor) and (
            _ad_native_tangent_or_none(tangent) is not None
        ):
            raise NotImplementedError(
                f"{op_name} does not differentiate {name}: tx_power, the "
                "polarizations, mu_r, material ids and valid masks stay fixed "
                "under the plan 07 fixed-topology contract"
            )


def _ad_geometry_live(*values: object) -> bool:
    """True when any geometry input participates in AD (grad or tangent).

    Drives the AD-2 need_grad_geometry plumbing and the conditional
    differentiability of path_length_m / delay_s: a materials-only graph
    keeps them detached exactly as in AD-1, so it never pays for geometry
    adjoints it did not request.
    """

    for value in values:
        if not isinstance(value, torch.Tensor):
            continue
        if value.requires_grad:
            return True
        if torch.autograd.forward_ad.unpack_dual(value).tangent is not None:
            return True
    return False


_participates_in_ad = _ad_geometry_live


def _frequency_participates_in_ad(frequency: float | torch.Tensor) -> bool:
    return _participates_in_ad(frequency)


def _ad_geometry_tangent(
    name: str, tangent: object, primal: torch.Tensor
) -> torch.Tensor | None:
    """Unwrap and validate a geometry tangent against its primal tensor."""

    value = _ad_native_tangent_or_none(
        tangent if isinstance(tangent, torch.Tensor) else None
    )
    if value is None:
        return None
    if tuple(value.shape) != tuple(primal.shape):
        raise ValueError(
            f"{name} must match its primal shape {tuple(primal.shape)};"
            f" got {tuple(value.shape)}"
        )
    if value.dtype != primal.dtype:
        raise TypeError(f"{name} must match the primal dtype {primal.dtype}")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    return value


__all__ = [
    "_ad_active_ctx",
    "_ad_check_active",
    "_ad_check_optional_grad",
    "_ad_check_rows",
    "_ad_check_tangent_vec3",
    "_ad_checked_tangent",
    "_ad_frequency_grad",
    "_ad_frequency_tangent",
    "_ad_frequency_value",
    "_ad_geometry_live",
    "_ad_geometry_tangent",
    "_ad_native_tangent_or_none",
    "_ad_native_tensor",
    "_ad_raise_composed_transforms",
    "_ad_reject_fixed_inputs",
    "_ad_reject_fixed_tangents",
    "_ad_still_wrapped",
    "_frequency_participates_in_ad",
]
