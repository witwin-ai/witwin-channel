# Copyright Xingyu Chen.
# Tests chain oracle selfchecks.

"""Self-consistency gates for the ADR-021 chain scattering oracles.

These run WITHOUT the witwin native extension (pure float64 Torch on CPU): they
pin the reference oracles in ``chain_ensemble`` / ``chain_realization`` against
independent references and physical invariants, so a later native lockstep
starts from a trusted float64 ground truth. Gates (ADR-021 acceptance protocol):

a. ``d1 = d2 = 0`` ensemble oracle reduces to the single-bounce op-1 oracle
   ``kirchhoff_ensemble.kirchhoff_ensemble_gain_reference``.
b. Specular-limit collapse: a smooth (h = 0) flat plate at the vertex makes the
   realization aperture integral collapse to the closed-form triangle area
   (``q_par = 0``), the image-source specular limit of the coherent oracle.
c. Reciprocity: swapping TX/RX reproduces the same ensemble ``path_gain``.
d. Energy: the hemispherical integral of the oracle BSDF stays within the
   ``(1 - C_r^2)|R|^2`` diffuse budget on a canonical material.
"""

from __future__ import annotations

import math

import torch

from tests.reference import kirchhoff_ensemble as ke
from tests.reference import kirchhoff_table_build as ktb
from tests.reference.chain_ensemble import _empty_chain, chain_ensemble_gain_reference
from tests.reference.chain_realization import chain_realization_eval, duffy_gl_nodes

_FREQ = 3.0e9
_C0 = 299792458.0
_LAYERS = torch.tensor([[0.1, 4.0, 0.02, 1.0]], dtype=torch.float64)


def _orthonormal_frame(n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ref = torch.zeros_like(n)
    pick = n.abs().argmin(dim=-1)
    ref.scatter_(-1, pick.unsqueeze(-1), 1.0)
    t1 = torch.nn.functional.normalize(ref - (ref * n).sum(-1, keepdim=True) * n, dim=-1)
    t2 = torch.cross(n, t1, dim=-1)
    return t1, t2


def _build_table(
    nti: int = 6, npi: int = 1, nto: int = 6, npo: int = 8, *, sigma_h: float = 0.01,
    corr: float = 0.15, sinkhorn_iters: int = 128,
) -> dict[str, torch.Tensor]:
    """Canonical float64 Kirchhoff table (isotropic, reciprocal, energy-balanced)."""

    k0 = 2.0 * math.pi * _FREQ / _C0
    cos_i = ktb.cos_centers(nti)
    phi_i = ktb.phi_centers(npi)
    cos_o = ktb.cos_centers(nto)
    phi_o = ktb.phi_centers(npo)
    return ktb.torch_build_table(
        torch.tensor(sigma_h, dtype=torch.float64),
        torch.tensor(corr, dtype=torch.float64),
        torch.tensor(corr, dtype=torch.float64),
        _LAYERS, torch.tensor(_FREQ, dtype=torch.float64),
        cos_i, phi_i, cos_o, phi_o,
        n_terms=ktb.build_n_terms(k0, sigma_h), isotropic=True,
        sinkhorn_iters=sinkhorn_iters,
    )


# ---------------------------------------------------------------------------
# Gate (a): d1 = d2 = 0 reduces to the op-1 single-bounce oracle.
# ---------------------------------------------------------------------------


def test_degenerate_chain_matches_single_bounce_oracle():
    torch.manual_seed(11)
    n = 32
    dtype = torch.float64
    table = _build_table()
    f_te, f_tm = table["f_te"], table["f_tm"]

    n_o = torch.nn.functional.normalize(
        torch.randn(n, 3, dtype=dtype) * 0.15 + torch.tensor([0.0, 0.0, 1.0]), dim=-1
    )
    t1r, t2r = _orthonormal_frame(n_o)
    backup = t1r.clone()
    vertex = torch.randn(n, 3, dtype=dtype) * 0.3
    vertex[:, 2] = 0.0
    source = vertex + torch.randn(n, 3, dtype=dtype) * 0.4 + torch.tensor([0.0, 0.0, 2.5])
    target = vertex + torch.randn(n, 3, dtype=dtype) * 0.4 + torch.tensor([0.0, 0.0, 3.0])
    tx_pol = torch.nn.functional.normalize(torch.randn(n, 3, dtype=dtype), dim=-1)
    rx_pol = torch.nn.functional.normalize(torch.randn(n, 3, dtype=dtype), dim=-1)
    weights = torch.rand(n, dtype=dtype) + 0.05
    coef = torch.tensor(1.7e-4, dtype=dtype)
    threshold = -1.0  # keep everything so the keep masks agree

    l1 = torch.linalg.vector_norm(vertex - source, dim=-1)
    l2 = torch.linalg.vector_norm(target - vertex, dim=-1)

    chain = chain_ensemble_gain_reference(
        source, tx_pol, _empty_chain(n, source.device, dtype), vertex, n_o, t1r, t2r,
        backup, f_te, f_tm, _empty_chain(n, source.device, dtype), target, rx_pol,
        l1, l2, weights, coef, torch.tensor(_FREQ, dtype=dtype), threshold,
    )

    # Reconstruct the op-1 gathered inputs from the same geometry.
    d_i = torch.nn.functional.normalize(vertex - source, dim=-1)
    wi_hat = -d_i
    cos_i = (wi_hat * n_o).sum(-1)
    wi_local = torch.stack(
        ((wi_hat * t1r).sum(-1), (wi_hat * t2r).sum(-1), cos_i), dim=-1
    )
    s_i, p_i = ke._sp_basis(n_o, d_i, backup)
    pol_t_perp = tx_pol - (tx_pol * d_i).sum(-1, keepdim=True) * d_i
    a_te2 = (pol_t_perp * s_i).sum(-1).square()
    a_tm2 = (pol_t_perp * p_i).sum(-1).square()
    to_rx = target - vertex
    r2 = torch.linalg.vector_norm(to_rx, dim=-1)
    wo = to_rx / r2[:, None]
    cos_o = (wo * n_o).sum(-1)
    idx = torch.arange(n)
    ref = ke.kirchhoff_ensemble_gain_reference(
        wo, r2, cos_o, n_o, t1r, t2r, wi_local, cos_i, l1, a_te2, a_tm2, weights,
        backup, rx_pol, idx, idx, f_te, f_tm, coef,
    )

    for key in ("gain", "amplitude", "length"):
        torch.testing.assert_close(chain[key], ref[key], rtol=1.0e-9, atol=1.0e-18)
    # Non-trivial: the shared table lookup produced real above-horizon gain.
    assert float(chain["gain"].abs().max()) > 0.0


# ---------------------------------------------------------------------------
# Gate (b): specular-limit collapse of the realization aperture integral.
# ---------------------------------------------------------------------------


def _flat_plate(grid: int, extent: float, dtype):
    """A flat z = 0 plate split into ``grid*grid`` triangles with matching UVs."""

    xs = torch.linspace(-extent, extent, grid + 1, dtype=dtype)
    tris, uvs = [], []
    for i in range(grid):
        for j in range(grid):
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = xs[j], xs[j + 1]
            tris.append(torch.tensor(
                [[x0, y0, 0.0], [x1, y0, 0.0], [x0, y1, 0.0]], dtype=dtype))
            u0 = (x0 + extent) / (2 * extent)
            u1 = (x1 + extent) / (2 * extent)
            v0 = (y0 + extent) / (2 * extent)
            v1 = (y1 + extent) / (2 * extent)
            uvs.append(torch.tensor([[u0, v0], [u1, v0], [u0, v1]], dtype=dtype))
    return torch.stack(tris), torch.stack(uvs)


def test_specular_flat_plate_integral_collapse():
    dtype = torch.float64
    grid = 6
    patch_tris, patch_uvs = _flat_plate(grid, 0.4, dtype)
    p = patch_tris.shape[0]
    rows = torch.arange(p)
    centroids = patch_tris.mean(dim=1)
    heights = torch.zeros(48, 48, dtype=dtype)  # smooth plate: h = 0

    n_rows = torch.zeros(p, 3, dtype=dtype)
    n_rows[:, 2] = 1.0
    k0 = torch.tensor(2.0 * math.pi * _FREQ / _C0, dtype=dtype)
    # Exact specular reflection across +z: d_o is the mirror of the arriving
    # d_i, so q_par = k0*(d_o - d_i) has zero tangential component.
    theta = math.radians(28.0)
    d_i = torch.tensor([math.sin(theta), 0.0, -math.cos(theta)], dtype=dtype).expand(p, 3).contiguous()
    d_o = torch.tensor([math.sin(theta), 0.0, math.cos(theta)], dtype=dtype).expand(p, 3).contiguous()

    # Two-bounce chain: one specular reflection in C1 before the scatter (the
    # image-source leg), scatter-terminal C2. Exercises transport_chain; the
    # aperture integral depends only on d_i/d_o/heights, so it must still
    # collapse to the triangle area.
    c1 = {
        "positions": torch.tensor([0.6, 0.0, 0.9], dtype=dtype).expand(p, 1, 3).contiguous(),
        "normals": torch.tensor([0.0, 0.0, 1.0], dtype=dtype).expand(p, 1, 3).contiguous(),
        "eps_r": torch.full((p, 1), 4.0, dtype=dtype),
        "sigma_e": torch.full((p, 1), 0.02, dtype=dtype),
        "mu_r": torch.ones(p, 1, dtype=dtype),
        "gain": torch.ones(p, 1, dtype=dtype),
        "thickness": torch.full((p, 1), 0.1, dtype=dtype),
        "sigma_b": torch.zeros(p, 1, dtype=dtype),
        "rough": torch.zeros(p, 1, dtype=torch.bool),
    }
    c2 = _empty_chain(p, heights.device, dtype)
    vertex = centroids
    source = torch.tensor([-1.2, 0.0, 1.5], dtype=dtype).expand(p, 3).contiguous()
    target = torch.tensor([1.2, 0.0, 1.5], dtype=dtype).expand(p, 3).contiguous()
    r_te = torch.complex(torch.full((p,), -0.6, dtype=dtype), torch.zeros(p, dtype=dtype))
    r_tm = torch.complex(torch.full((p,), -0.4, dtype=dtype), torch.zeros(p, dtype=dtype))
    l1 = torch.full((p,), 2.0, dtype=dtype)
    l2 = torch.full((p,), 1.8, dtype=dtype)
    quad_a, quad_b, quad_w = duffy_gl_nodes(heights.device, dtype)

    out = chain_realization_eval(
        heights, patch_tris, patch_uvs, rows, source,
        torch.tensor([0.0, 0.0, 1.0], dtype=dtype).expand(p, 3).contiguous(), c1,
        vertex, n_rows, r_te, r_tm, d_i, d_o, c2, target,
        torch.tensor([0.0, 0.0, 1.0], dtype=dtype).expand(p, 3).contiguous(),
        l1, l2, centroids, quad_a, quad_b, quad_w, k0, torch.tensor(_FREQ, dtype=dtype),
    )

    e1 = patch_tris[:, 1] - patch_tris[:, 0]
    e2 = patch_tris[:, 2] - patch_tris[:, 0]
    area = 0.5 * torch.linalg.vector_norm(torch.cross(e1, e2, dim=-1), dim=-1)
    # q_par = 0, h = 0 -> integral = |e1 x e2| * sum(w) = 2*area*0.5 = area.
    torch.testing.assert_close(out["integral"].real, area, rtol=1.0e-9, atol=1.0e-15)
    torch.testing.assert_close(
        out["integral"].imag, torch.zeros_like(area), rtol=0.0, atol=1.0e-12
    )
    assert torch.isfinite(out["total"].real) and torch.isfinite(out["total"].imag)

    # A generic (non-specular) direction must NOT collapse to the area: the
    # collapse is specific to the specular / stationary-phase limit.
    d_o_gen = torch.nn.functional.normalize(
        torch.tensor([0.5, 0.4, 0.9], dtype=dtype), dim=-1
    ).expand(p, 3).contiguous()
    generic = chain_realization_eval(
        heights, patch_tris, patch_uvs, rows, source,
        torch.tensor([0.0, 0.0, 1.0], dtype=dtype).expand(p, 3).contiguous(), c1,
        vertex, n_rows, r_te, r_tm, d_i, d_o_gen, c2, target,
        torch.tensor([0.0, 0.0, 1.0], dtype=dtype).expand(p, 3).contiguous(),
        l1, l2, centroids, quad_a, quad_b, quad_w, k0, torch.tensor(_FREQ, dtype=dtype),
    )
    assert float((generic["integral"].real - area).abs().max()) > 1.0e-6


# ---------------------------------------------------------------------------
# Gate (c): TX/RX reciprocity of the ensemble path_gain (single scatter).
# ---------------------------------------------------------------------------


def test_ensemble_reciprocity_tx_rx_swap():
    dtype = torch.float64
    table = _build_table()
    f_te, f_tm = table["f_te"], table["f_tm"]
    n = 24

    # Coplanar (y = 0) geometry with n = +z: the relative azimuth is 0 or pi,
    # both fixed points of azimuth negation, so the isotropic-table lookup is
    # exactly reciprocal under the wi<->wo swap.
    n_o = torch.tensor([0.0, 0.0, 1.0], dtype=dtype).expand(n, 3).contiguous()
    t1r = torch.tensor([1.0, 0.0, 0.0], dtype=dtype).expand(n, 3).contiguous()
    t2r = torch.tensor([0.0, 1.0, 0.0], dtype=dtype).expand(n, 3).contiguous()
    backup = t1r.clone()
    torch.manual_seed(5)
    vertex = torch.zeros(n, 3, dtype=dtype)
    vertex[:, 0] = torch.randn(n, dtype=dtype) * 0.2
    sx = torch.rand(n, dtype=dtype) * 1.5 + 0.5
    tx_x = torch.rand(n, dtype=dtype) * 1.5 + 0.5
    source = torch.stack([-sx, torch.zeros(n, dtype=dtype), torch.rand(n, dtype=dtype) + 1.5], -1)
    target = torch.stack([tx_x, torch.zeros(n, dtype=dtype), torch.rand(n, dtype=dtype) + 1.5], -1)
    tx_pol = torch.nn.functional.normalize(torch.randn(n, 3, dtype=dtype), dim=-1)
    rx_pol = torch.nn.functional.normalize(torch.randn(n, 3, dtype=dtype), dim=-1)
    weights = torch.rand(n, dtype=dtype) + 0.05
    coef = torch.tensor(2.3e-4, dtype=dtype)
    l1 = torch.linalg.vector_norm(vertex - source, dim=-1)
    l2 = torch.linalg.vector_norm(target - vertex, dim=-1)
    freq = torch.tensor(_FREQ, dtype=dtype)

    def empty():
        return _empty_chain(n, source.device, dtype)

    fwd = chain_ensemble_gain_reference(
        source, tx_pol, empty(), vertex, n_o, t1r, t2r, backup, f_te, f_tm, empty(),
        target, rx_pol, l1, l2, weights, coef, freq, -1.0,
    )
    rev = chain_ensemble_gain_reference(
        target, rx_pol, empty(), vertex, n_o, t1r, t2r, backup, f_te, f_tm, empty(),
        source, tx_pol, l2, l1, weights, coef, freq, -1.0,
    )
    torch.testing.assert_close(fwd["gain"], rev["gain"], rtol=1.0e-12, atol=1.0e-18)
    assert float(fwd["gain"].abs().max()) > 0.0


# ---------------------------------------------------------------------------
# Gate (d): hemispherical BSDF energy stays within the (1-C_r^2)|R|^2 budget.
# ---------------------------------------------------------------------------


def test_bsdf_hemispherical_energy_within_budget():
    nti, npi, nto, npo = 6, 1, 6, 8
    table = _build_table(nti, npi, nto, npo, sinkhorn_iters=256)
    cos_o = ktb.cos_centers(nto)
    # Sinkhorn measure: sum over the outgoing hemisphere of f * cos_o * dOmega,
    # dOmega = (1/nto)*(2pi/npo) (matches _sinkhorn_balance).
    d_omega = (1.0 / nto) * (2.0 * math.pi / npo)
    weight = (cos_o * d_omega).reshape(1, 1, nto, 1)
    for name, budget in (("f_te", "r_diff_te"), ("f_tm", "r_diff_tm")):
        energy = (table[name] * weight).sum(dim=(2, 3))  # [nti, npi]
        r_diff = table[budget]  # [nti, npi]
        active = r_diff > 0.0
        # Sinkhorn drives the integral to r_diff exactly; converged tightly.
        torch.testing.assert_close(
            energy[active], r_diff[active], rtol=1.0e-4, atol=1.0e-9
        )
        # No channel exceeds its (1-C_r^2)|R|^2 diffuse budget.
        assert bool((energy <= r_diff + 1.0e-6).all())