from __future__ import annotations

import torch


def normalize_vec3(values: torch.Tensor, *, eps: float = 1.0e-12) -> torch.Tensor:
    return values / torch.linalg.vector_norm(
        values, dim=-1, keepdim=True
    ).clamp_min(eps)


def require_tensor(
    name: str,
    value: object,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
    device: torch.device | None = None,
    cuda: bool = False,
    contiguous: bool = False,
    dtype_error: type[Exception] = ValueError,
) -> torch.Tensor:
    """Validate one declared tensor field of a typed row or capacity contract.

    This is the single owner of that check. The row contracts, the capacity
    contracts, and the consumer contracts each carried their own copy; the only
    behaviour that ever differed between them is which exception a dtype
    mismatch raises, which ``dtype_error`` keeps caller-declared. The checks run
    in the order the copies used, so every rejected input still fails on the
    same clause with the same message.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if cuda and not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if value.dtype != dtype:
        raise dtype_error(f"{name} must use {dtype}, got {value.dtype}")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {value.ndim}")
    if contiguous and not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    return value
