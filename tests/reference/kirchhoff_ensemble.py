"""Reference Torch Kirchhoff ensemble row physics (ADR-010 op 1).

The previous production per-row physics of
``propagation/enumerated/scattering.py::_ensemble_rows`` after the RayD
visibility filter (frame projections, table lookup, radiometric gain). The
Kirchhoff table lookup itself was already native (``eval_bsdf`` ->
``scattering_table_eval``) and is reused unchanged. Test-only: MUST NOT be
imported from production packages.
"""

from __future__ import annotations

import torch

from witwin.channel_native.core.tensor_math import normalize_vec3
from witwin.channel_native.scattering import eval_bsdf


def _sp_basis(
    n: torch.Tensor, d: torch.Tensor, backup_axis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    s = torch.cross(n, d, dim=-1)
    degenerate = torch.linalg.vector_norm(s, dim=-1, keepdim=True) < 1.0e-6
    s = torch.where(degenerate, backup_axis, normalize_vec3(s))
    p = torch.cross(s, d, dim=-1)
    return s, p


def kirchhoff_ensemble_rows(
    points: torch.Tensor,
    n_o: torch.Tensor,
    t1r: torch.Tensor,
    t2r: torch.Tensor,
    wi_local: torch.Tensor,
    cos_i: torch.Tensor,
    r1: torch.Tensor,
    a_te2: torch.Tensor,
    a_tm2: torch.Tensor,
    weights: torch.Tensor,
    material_id: torch.Tensor,
    backup_axis: torch.Tensor,
    rx_positions: torch.Tensor,
    rx_pol: torch.Tensor,
    rc: torch.Tensor,
    sc: torch.Tensor,
    tables: dict[int, object],
    coef: float,
    threshold: float,
) -> dict[str, torch.Tensor]:
    """Per visibility-surviving row physics matching the removed Torch source."""

    to_rx = rx_positions[rc] - points[sc]
    r2_row = torch.linalg.vector_norm(to_rx, dim=-1).clamp_min(1.0e-6)
    wo_row = to_rx / r2_row[:, None]
    cos_o_row = (wo_row * n_o[sc]).sum(-1)
    wo_local = torch.stack(
        (
            (wo_row * t1r[sc]).sum(-1),
            (wo_row * t2r[sc]).sum(-1),
            cos_o_row,
        ),
        dim=-1,
    )
    f_te = torch.zeros_like(cos_o_row)
    f_tm = torch.zeros_like(cos_o_row)
    for material_index, table in tables.items():
        mask = material_id[sc] == material_index
        if not bool(mask.any()):
            continue
        te, tm = eval_bsdf(
            table, wi_local[sc][mask].contiguous(), wo_local[mask].contiguous()
        )
        f_te[mask] = te
        f_tm[mask] = tm

    s_o, p_o = _sp_basis(n_o[sc], wo_row, backup_axis[sc])
    pol_r = rx_pol[rc]
    pol_r_perp = pol_r - (pol_r * wo_row).sum(-1, keepdim=True) * wo_row
    g_te2 = (pol_r_perp * s_o).sum(-1).square()
    g_tm2 = (pol_r_perp * p_o).sum(-1).square()
    f_eff = f_te * a_te2[sc] * g_te2 + f_tm * a_tm2[sc] * g_tm2

    gain = (
        coef
        * f_eff
        * cos_i[sc]
        * cos_o_row
        * weights[sc]
        / (r1[sc].square() * r2_row.square())
    )
    keep = gain > threshold
    amplitude = gain.clamp_min(0.0).sqrt()
    length = r1[sc] + r2_row
    return {
        "gain": gain,
        "amplitude": amplitude,
        "length": length,
        "direction": wo_row,
        "keep": keep,
    }
