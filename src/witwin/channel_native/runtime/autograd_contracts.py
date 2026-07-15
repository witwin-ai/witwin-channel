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


__all__ = [
    "_ad_active_ctx",
    "_ad_check_active",
    "_ad_check_optional_grad",
    "_ad_check_rows",
    "_ad_check_tangent_vec3",
    "_ad_checked_tangent",
    "_ad_native_tangent_or_none",
    "_ad_native_tensor",
    "_ad_raise_composed_transforms",
    "_ad_still_wrapped",
]
