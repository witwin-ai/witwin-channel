"""Shared central finite-difference engine for the AD test suite.

Two-point central differences, ``(f(x + h) - f(x - h)) / (2 h)``, reusable by
later AD phases. Callables receive a perturbed copy of ``x`` (same device and
dtype) and must return a detached tensor; accumulation happens in float64 on
the CPU to keep the FD noise floor below the comparison tolerances.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def _as_cpu_double(value: torch.Tensor) -> torch.Tensor:
    return value.detach().double().cpu()


def central_difference_gradient(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    step: float,
) -> torch.Tensor:
    """Full gradient of a scalar-valued ``f`` at ``x`` (one FD pair per coordinate)."""

    flat = x.detach().clone().reshape(-1)
    grad = torch.zeros(flat.shape[0], dtype=torch.float64)
    for index in range(flat.shape[0]):
        plus = flat.clone()
        minus = flat.clone()
        plus[index] += step
        minus[index] -= step
        f_plus = _as_cpu_double(f(plus.reshape(x.shape)))
        f_minus = _as_cpu_double(f(minus.reshape(x.shape)))
        if f_plus.numel() != 1 or f_minus.numel() != 1:
            raise ValueError("central_difference_gradient requires a scalar-valued f")
        grad[index] = (f_plus.item() - f_minus.item()) / (2.0 * step)
    return grad.reshape(x.shape)


def central_difference_directional(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    direction: torch.Tensor,
    step: float,
) -> torch.Tensor:
    """Directional derivative of tensor-valued ``f`` at ``x`` along ``direction``."""

    base = x.detach()
    offset = step * direction.detach().to(device=base.device, dtype=base.dtype)
    f_plus = _as_cpu_double(f(base + offset))
    f_minus = _as_cpu_double(f(base - offset))
    return (f_plus - f_minus) / (2.0 * step)


def relative_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    abs_floor: float,
) -> float:
    """Relative L2 error with an absolute floor on the normalization scale."""

    actual_flat = _as_cpu_double(actual).reshape(-1)
    expected_flat = _as_cpu_double(expected).reshape(-1)
    scale = max(
        float(torch.linalg.norm(expected_flat)),
        float(torch.linalg.norm(actual_flat)),
        abs_floor,
    )
    return float(torch.linalg.norm(actual_flat - expected_flat)) / scale
