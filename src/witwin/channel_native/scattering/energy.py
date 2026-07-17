"""Per-event energy budgets for rough layered surfaces (contract section 6).

Splits the smooth-stack budgets into the pieces the solvers use for event
probabilities: coherent specular reflection ``R_coh_q = R_bar_q * C_r^2``
(with ``C_r = exp(-2*(k0*cos_theta_i*sigma_h)^2)``), diffuse reflection
``R_diff_q = max(0, R_bar_q - R_coh_q)``, smooth-stack transmission
``T_bar_q`` (no diffuse transmission in v1) and absorption ``A_q``. The
Production values are computed once on the Kirchhoff table's incidence grid
(32 cell centers uniform in cos) and linearly interpolated at the runtime
angles, so budgets and tables share one discretization.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from witwin.channel_native.materials.evaluation import layer_stack_rt
from witwin.channel_native.physics.conventions import C0

from .tables import N_COS_THETA_I, _cos_centers

__all__ = ["event_budget"]

_PASSIVITY_TOL = 1e-4


def event_budget(
    cos_theta_i: torch.Tensor,
    layers: Sequence[tuple],
    roughness,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    """Event energy budgets at batched incidence cosines (torch, on device).

    ``layers`` is the production layer list ``[(thickness_m, eps_r, sigma_e,
    mu_r), ...]``; ``roughness`` is a ``Roughness`` or ``None`` (smooth:
    ``C_r = 1`` and ``R_diff = 0`` exactly). Returns per-pol tensors
    (``*_te``/``*_tm``) plus unpolarized means, all shaped like
    ``cos_theta_i`` on its device. Raises when the production stack violates
    passivity ``R + T + A <= 1 + 1e-4`` at any grid angle.
    """

    sigma_h = 0.0 if roughness is None else float(roughness.rms_height_m)
    k0 = 2.0 * math.pi * float(frequency_hz) / C0
    grid = _cos_centers(N_COS_THETA_I)
    rt = layer_stack_rt(layers, grid, float(frequency_hz))

    c_r2 = np.exp(-2.0 * (k0 * grid * sigma_h) ** 2) ** 2
    columns: dict[str, np.ndarray] = {}
    for pol, r_bar, t_bar, a_bar in (
        ("te", rt.R_te, rt.T_te, rt.A_te),
        ("tm", rt.R_tm, rt.T_tm, rt.A_tm),
    ):
        r_coh = r_bar * c_r2
        r_diff = np.maximum(0.0, r_bar - r_coh)
        total = r_coh + r_diff + t_bar + a_bar
        if float(total.max()) > 1.0 + _PASSIVITY_TOL:
            raise ValueError(
                f"passivity violated for {pol.upper()}: R+T+A = "
                f"{float(total.max()):.6f} > 1 + {_PASSIVITY_TOL:g}"
            )
        columns[f"R_coh_{pol}"] = r_coh
        columns[f"R_diff_{pol}"] = r_diff
        columns[f"T_{pol}"] = t_bar
        columns[f"A_{pol}"] = a_bar
    for base in ("R_coh", "R_diff", "T", "A"):
        columns[base] = 0.5 * (columns[f"{base}_te"] + columns[f"{base}_tm"])

    # Linear interpolation of each grid column at the runtime cosines.
    cos_q = cos_theta_i.to(dtype=torch.float32)
    device = cos_q.device
    t = (cos_q * N_COS_THETA_I - 0.5).clamp(0.0, float(N_COS_THETA_I - 1))
    i0 = torch.floor(t).clamp(max=float(N_COS_THETA_I - 2)).long()
    w = t - i0.to(t.dtype)
    out: dict[str, torch.Tensor] = {}
    for name, values in columns.items():
        col = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)).to(device)
        out[name] = col[i0] * (1.0 - w) + col[i0 + 1] * w
    return out
