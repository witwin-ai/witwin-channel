# Copyright Xingyu Chen.
# Torch oracle of the resident Kirchhoff table lookup in the native float32 operation order.

"""Torch oracle of the resident Kirchhoff table lookup in the native float32 operation order."""

from __future__ import annotations

import torch

_TWO_PI = 6.2831853071795864769


def _positive_phi(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    phi = torch.atan2(y, x)
    return torch.where(phi < 0.0, phi + torch.tensor(_TWO_PI, dtype=phi.dtype), phi)


def _linear_axis(coord: torch.Tensor, n: int, period: float, periodic: bool):
    if n == 1:
        zero = torch.zeros_like(coord, dtype=torch.int64)
        return zero, zero, torch.zeros_like(coord)
    t = coord * torch.tensor(n / period, dtype=coord.dtype) - 0.5
    if periodic:
        base = torch.floor(t)
        weight = t - base
        i0 = torch.remainder(base.to(torch.int64), n)
        return i0, torch.remainder(i0 + 1, n), weight
    t = t.clamp(0.0, float(n - 1))
    i0 = torch.floor(t).to(torch.int64).clamp(max=n - 2)
    return i0, i0 + 1, t - i0.to(t.dtype)


def eval_table(
    table, valid: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(f_te, f_tm)`` of a ``KirchhoffTable`` at local-frame ``wi``/``wo`` pairs ([N, 3]).

    Pairs below the horizon, or with ``valid`` False, return 0. Isotropic tables
    (one incident azimuth) are looked up at the relative azimuth ``phi_o - phi_i``.
    """

    wi = wi.to(torch.float32)
    wo = wo.to(torch.float32)
    f_te = table.f_te.to(torch.float32)
    f_tm = table.f_tm.to(torch.float32)
    nti, npi, nto, npo = (int(size) for size in f_te.shape)
    phi_i = _positive_phi(wi[:, 1], wi[:, 0])
    phi_o = _positive_phi(wo[:, 1], wo[:, 0])
    if npi == 1:
        phi_o = phi_o - phi_i
        phi_o = torch.where(phi_o < 0.0, phi_o + torch.tensor(_TWO_PI, dtype=phi_o.dtype), phi_o)
    axes = (
        _linear_axis(wi[:, 2], nti, 1.0, False),
        _linear_axis(phi_i, npi, _TWO_PI, True),
        _linear_axis(wo[:, 2], nto, 1.0, False),
        _linear_axis(phi_o, npo, _TWO_PI, True),
    )
    te = torch.zeros_like(wi[:, 0])
    tm = torch.zeros_like(wi[:, 0])
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                for d in (0, 1):
                    weight = None
                    index = []
                    for axis, corner in zip(axes, (a, b, c, d)):
                        low, high, w = axis
                        index.append(high if corner else low)
                        factor = w if corner else 1.0 - w
                        weight = factor if weight is None else weight * factor
                    te = te + weight * f_te[index[0], index[1], index[2], index[3]]
                    tm = tm + weight * f_tm[index[0], index[1], index[2], index[3]]
    lit = valid & (wi[:, 2] > 0.0) & (wo[:, 2] > 0.0)
    zero = torch.zeros_like(te)
    return torch.where(lit, te, zero), torch.where(lit, tm, zero)
