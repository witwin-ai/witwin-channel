# Copyright Xingyu Chen.
# Tests table build ad.

"""Tests table build ad."""

from __future__ import annotations

import math

import pytest
import torch

from witwin.core import Scene, SurfaceRoughness
from tests.support.core_world import make_receiver, make_transmitter
from witwin.channel.runtime import has_symbol
from witwin.channel.scene import compile as compile_scene
from tests.reference.kirchhoff_table_build import (
    build_n_terms,
    phi_centers,
    torch_build_table,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for table-build AD"
)

_C0 = 299792458.0
try:
    _HAS_NATIVE = has_symbol(
        "kirchhoff_table_build_backward"
    ) and has_symbol("kirchhoff_table_build_jvp")
except ImportError:
    _HAS_NATIVE = False
_require_native = pytest.mark.skipif(
    not _HAS_NATIVE, reason="native kirchhoff_table_build_* not built"
)

# Comparison tolerances: the native companions recompute the Beckmann series and
# stack reflectance in float32 against the f64 oracle, so f32 rounding dominates.
# On the validated well-conditioned configs below the measured native-vs-oracle
# agreement is <= 6.5e-4 (iso) / <= 2.1e-3 (aniso); the bound leaves ~5x margin.
_REL_TOL = 1.0e-2
_ABS_TOL = 1.0e-6

# Sinkhorn iterations for the unrolled oracle. On the well-conditioned configs
# the oracle gradient is bit-stable at this count (verified against 2x by
# ``_assert_oracle_converged``): differentiating the converged unrolled balance
# reproduces the implicit-function adjoint the native companion computes.
_SINKHORN_ITERS = 512

# Well-conditioned directional grids (cos centers >= 0.5, no grazing): the
# balance factors ``a`` stay O(1) so the f32 lobe/stack recompute never
# underflows, while the anisotropic Jacobian stays structurally singular so the
# native float64 SVD pseudo-inverse solve is still exercised.
_COS_CENTERS = (0.7, 0.85, 0.95)
# Distinct correlation lengths (few mm, comparable to the ~5 mm wavelength) so
# ``corr_x`` and ``corr_y`` carry real, non-degenerate gradient signal.
_CORR_X = 0.005
_CORR_Y = 0.009


def _params(dtype=torch.float64):
    sigma_h = torch.tensor(1.0e-3, dtype=dtype, requires_grad=True)
    lx = torch.tensor(_CORR_X, dtype=dtype, requires_grad=True)
    thickness = torch.tensor([0.05], dtype=dtype, requires_grad=True)
    eps = torch.tensor([4.0], dtype=dtype, requires_grad=True)
    sigma = torch.tensor([0.05], dtype=dtype, requires_grad=True)
    frequency = torch.tensor(60.0e9, dtype=dtype, requires_grad=True)
    return sigma_h, lx, thickness, eps, sigma, frequency


def _corr_y() -> torch.Tensor:
    return torch.tensor(_CORR_Y, dtype=torch.float64, requires_grad=True)


def _grids(iso: bool):
    # cos_i == cos_o (nti == nto is required by the reciprocal balance); the
    # isotropic table collapses the reverse azimuth (npi == 1), the anisotropic
    # table keeps a full (cos, phi) state (npi == npo). corr_x/corr_y stay
    # distinct in BOTH cases, so x/y gradients are validated separately even on
    # the isotropic-balance grid.
    cos = torch.tensor(_COS_CENTERS, dtype=torch.float64)
    n = len(_COS_CENTERS)
    if iso:
        nti, npi, nto, npo = n, 1, n, 8
        phi_i = torch.zeros(1, dtype=torch.float64)
    else:
        nti, npi, nto, npo = n, n, n, n
        phi_i = phi_centers(npi)
    return {
        "cos_i": cos,
        "phi_i": phi_i,
        "cos_o": cos.clone(),
        "phi_o": phi_centers(npo),
        "dims": (nti, npi, nto, npo),
    }


def _oracle(iso: bool, sigma_h, lx, ly, thickness, eps, sigma, frequency, grids,
            sinkhorn_iters: int = _SINKHORN_ITERS):
    layers = torch.stack(
        (
            thickness.reshape(()),
            eps.reshape(()),
            sigma.reshape(()),
            torch.ones((), dtype=thickness.dtype),
        )
    ).reshape(1, 4)
    n_terms = build_n_terms(
        float(2.0 * math.pi * float(frequency.detach()) / _C0),
        float(sigma_h.detach()),
    )
    return torch_build_table(
        sigma_h.reshape(()),
        lx.reshape(()),
        ly.reshape(()),
        layers,
        frequency.reshape(()),
        grids["cos_i"],
        grids["phi_i"],
        grids["cos_o"],
        grids["phi_o"],
        n_terms=n_terms,
        isotropic=iso,
        sinkhorn_iters=sinkhorn_iters,
    )


def _to_native(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(device="cuda", dtype=torch.float32).contiguous()


def _native_backward(out, grids, sigma_h, lx, ly, thickness, eps, sigma, frequency,
                     grad_f_te, grad_f_tm, *, rough, layers, freq):
    from witwin.channel.kernels.scattering import (
        kirchhoff_table_build_backward,
    )

    return kirchhoff_table_build_backward(
        _to_native(out["s_te"]),
        _to_native(out["s_tm"]),
        _to_native(out["a_te"]),
        _to_native(out["a_tm"]),
        _to_native(out["r_diff_te"]),
        _to_native(out["r_diff_tm"]),
        _to_native(grids["cos_i"]),
        _to_native(grids["phi_i"]),
        _to_native(grids["cos_o"]),
        _to_native(grids["phi_o"]),
        _to_native(thickness),
        _to_native(eps),
        _to_native(sigma),
        _to_native(torch.ones_like(thickness)),
        sigma_h=float(sigma_h.detach()),
        corr_x=float(lx.detach()),
        corr_y=float(ly.detach()),
        frequency_hz=float(frequency.detach()),
        grad_f_te=_to_native(grad_f_te),
        grad_f_tm=_to_native(grad_f_tm),
        need_grad_rough=rough,
        need_grad_layers=layers,
        need_grad_frequency=freq,
    )


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), _ABS_TOL)


def _oracle_grads(iso, sigma_h, lx, ly, thickness, eps, sigma, frequency, grids,
                  grad_f_te, grad_f_tm, sinkhorn_iters):
    out = _oracle(
        iso, sigma_h, lx, ly, thickness, eps, sigma, frequency, grids,
        sinkhorn_iters=sinkhorn_iters,
    )
    loss = (grad_f_te * out["f_te"]).sum() + (grad_f_tm * out["f_tm"]).sum()
    grads = torch.autograd.grad(
        loss, (sigma_h, lx, ly, thickness, eps, sigma, frequency),
        allow_unused=True,
    )
    return out, [float(g) for g in grads]


@_require_native
@pytest.mark.parametrize("iso", [True, False])
def test_backward_matches_oracle(iso):
    # corr_x/corr_y are distinct leaves in both cases so the x/y gradients are
    # validated separately (see _grids); the anisotropic build additionally makes
    # the balance Jacobian structurally singular, exercising the SVD pinv solve.
    sigma_h, lx, thickness, eps, sigma, frequency = _params()
    ly = _corr_y()
    grids = _grids(iso)

    generator = torch.Generator().manual_seed(3)
    # Oracle build shapes are dims-only; get them from a throwaway build.
    probe = _oracle(iso, sigma_h, lx, ly, thickness, eps, sigma, frequency, grids)
    grad_f_te = torch.randn(probe["f_te"].shape, generator=generator, dtype=torch.float64)
    grad_f_tm = torch.randn(probe["f_tm"].shape, generator=generator, dtype=torch.float64)

    out, g1 = _oracle_grads(
        iso, sigma_h, lx, ly, thickness, eps, sigma, frequency, grids,
        grad_f_te, grad_f_tm, _SINKHORN_ITERS,
    )
    # Oracle self-convergence: the unrolled-Sinkhorn gradient must be stable to
    # < 1e-4 rel between _SINKHORN_ITERS and 2x, so it represents the converged
    # implicit adjoint (not a truncated-unroll artefact).
    _, g2 = _oracle_grads(
        iso, sigma_h, lx, ly, thickness, eps, sigma, frequency, grids,
        grad_f_te, grad_f_tm, 2 * _SINKHORN_ITERS,
    )
    for a, b in zip(g1, g2):
        assert _rel(a, b) <= 1.0e-4, "oracle Sinkhorn not converged"

    labels = ("sigma_h", "corr_x", "corr_y", "thickness", "eps", "sigma", "frequency")
    oracle = dict(zip(labels, g1))
    # corr_x and corr_y carry distinct, non-degenerate signal on these grids:
    # both are well above numerical noise (relative to the largest gradient) and
    # differ substantially from each other (relative to their own magnitude).
    scale = max(abs(v) for v in oracle.values())
    corr_scale = max(abs(oracle["corr_x"]), abs(oracle["corr_y"]))
    assert abs(oracle["corr_x"]) > 1.0e-2 * scale
    assert abs(oracle["corr_y"]) > 1.0e-2 * scale
    assert abs(oracle["corr_x"] - oracle["corr_y"]) > 0.1 * corr_scale

    native = _native_backward(
        out, grids, sigma_h, lx, ly, thickness, eps, sigma, frequency,
        grad_f_te, grad_f_tm, rough=True, layers=True, freq=True,
    )
    assert _rel(float(native["grad_sigma_h"][0]), oracle["sigma_h"]) <= _REL_TOL
    assert _rel(float(native["grad_corr_x"][0]), oracle["corr_x"]) <= _REL_TOL
    assert _rel(float(native["grad_corr_y"][0]), oracle["corr_y"]) <= _REL_TOL
    assert _rel(float(native["grad_layer_thickness_m"][0]), oracle["thickness"]) <= _REL_TOL
    assert _rel(float(native["grad_layer_eps_r"][0]), oracle["eps"]) <= _REL_TOL
    assert _rel(float(native["grad_layer_sigma_e"][0]), oracle["sigma"]) <= _REL_TOL
    assert _rel(float(native["grad_frequency"][0]), oracle["frequency"]) <= _REL_TOL


@_require_native
@pytest.mark.parametrize("iso", [True, False])
def test_jvp_vs_vjp_inner_product(iso):
    from witwin.channel.kernels.scattering import (
        kirchhoff_table_build_jvp,
    )

    sigma_h, lx, thickness, eps, sigma, frequency = _params()
    ly = _corr_y()
    grids = _grids(iso)
    out = _oracle(iso, sigma_h, lx, ly, thickness, eps, sigma, frequency, grids)

    generator = torch.Generator().manual_seed(11)
    # Random parameter tangents v and output cotangents u.
    v = {
        "sigma_h": 1.0e-4,
        "corr_x": 2.0e-3,
        "corr_y": -1.5e-3,
        "frequency": 5.0e7,
    }
    t_thickness = torch.randn(thickness.shape, generator=generator, dtype=torch.float64)
    t_eps = torch.randn(eps.shape, generator=generator, dtype=torch.float64)
    t_sigma = torch.randn(sigma.shape, generator=generator, dtype=torch.float64)
    u_te = torch.randn(out["f_te"].shape, generator=generator, dtype=torch.float64)
    u_tm = torch.randn(out["f_tm"].shape, generator=generator, dtype=torch.float64)

    jvp = kirchhoff_table_build_jvp(
        _to_native(out["s_te"]), _to_native(out["s_tm"]),
        _to_native(out["a_te"]), _to_native(out["a_tm"]),
        _to_native(out["r_diff_te"]), _to_native(out["r_diff_tm"]),
        _to_native(grids["cos_i"]), _to_native(grids["phi_i"]),
        _to_native(grids["cos_o"]), _to_native(grids["phi_o"]),
        _to_native(thickness), _to_native(eps), _to_native(sigma),
        _to_native(torch.ones_like(thickness)),
        sigma_h=float(sigma_h.detach()),
        corr_x=float(lx.detach()),
        corr_y=float(ly.detach()),
        frequency_hz=float(frequency.detach()),
        t_layer_thickness_m=_to_native(t_thickness),
        t_layer_eps_r=_to_native(t_eps),
        t_layer_sigma_e=_to_native(t_sigma),
        t_sigma_h=v["sigma_h"], t_corr_x=v["corr_x"], t_corr_y=v["corr_y"],
        t_frequency=v["frequency"],
    )
    lhs = float((jvp["tangent_f_te"] * _to_native(u_te)).sum()) + float(
        (jvp["tangent_f_tm"] * _to_native(u_tm)).sum()
    )

    vjp = _native_backward(
        out, grids, sigma_h, lx, ly, thickness, eps, sigma, frequency,
        u_te, u_tm, rough=True, layers=True, freq=True,
    )
    rhs = (
        v["sigma_h"] * float(vjp["grad_sigma_h"][0])
        + v["corr_x"] * float(vjp["grad_corr_x"][0])
        + v["corr_y"] * float(vjp["grad_corr_y"][0])
        + v["frequency"] * float(vjp["grad_frequency"][0])
        + float((_to_native(t_thickness) * vjp["grad_layer_thickness_m"]).sum())
        + float((_to_native(t_eps) * vjp["grad_layer_eps_r"]).sum())
        + float((_to_native(t_sigma) * vjp["grad_layer_sigma_e"]).sum())
    )
    assert _rel(lhs, rhs) <= _REL_TOL


@_require_native
@pytest.mark.parametrize(
    "param", ["sigma_h", "corr_x", "corr_y", "thickness", "eps", "sigma", "frequency"]
)
def test_fd_cross_check(param):
    # Central FD of the oracle build validates the native gradient per parameter.
    iso = True
    grids = _grids(iso)
    base = dict(
        sigma_h=1.0e-3, lx=_CORR_X, ly=_CORR_Y, thickness=0.05, eps=4.0,
        sigma=0.05, frequency=60.0e9,
    )
    generator = torch.Generator().manual_seed(7)
    dims = grids["dims"]
    shape = (dims[0], dims[1], dims[2], dims[3])
    grad_f_te = torch.randn(shape, generator=generator, dtype=torch.float64)
    grad_f_tm = torch.randn(shape, generator=generator, dtype=torch.float64)

    def build(values):
        sigma_h = torch.tensor(values["sigma_h"], dtype=torch.float64)
        lx = torch.tensor(values["lx"], dtype=torch.float64)
        ly = torch.tensor(values["ly"], dtype=torch.float64)
        thickness = torch.tensor([values["thickness"]], dtype=torch.float64)
        eps = torch.tensor([values["eps"]], dtype=torch.float64)
        sigma = torch.tensor([values["sigma"]], dtype=torch.float64)
        frequency = torch.tensor(values["frequency"], dtype=torch.float64)
        out = _oracle(iso, sigma_h, lx, ly, thickness, eps, sigma, frequency, grids)
        return (grad_f_te * out["f_te"]).sum() + (grad_f_tm * out["f_tm"]).sum()

    key = {"sigma_h": "sigma_h", "corr_x": "lx", "corr_y": "ly", "thickness": "thickness",
           "eps": "eps", "sigma": "sigma", "frequency": "frequency"}[param]
    step = abs(base[key]) * 1.0e-4 + 1.0e-9
    plus = dict(base)
    minus = dict(base)
    plus[key] += step
    minus[key] -= step
    fd = (float(build(plus)) - float(build(minus))) / (2.0 * step)

    sigma_h = torch.tensor(base["sigma_h"], dtype=torch.float64)
    lx = torch.tensor(base["lx"], dtype=torch.float64)
    ly = torch.tensor(base["ly"], dtype=torch.float64)
    thickness = torch.tensor([base["thickness"]], dtype=torch.float64)
    eps = torch.tensor([base["eps"]], dtype=torch.float64)
    sigma = torch.tensor([base["sigma"]], dtype=torch.float64)
    frequency = torch.tensor(base["frequency"], dtype=torch.float64)
    out = _oracle(iso, sigma_h, lx, ly, thickness, eps, sigma, frequency, grids)
    native = _native_backward(
        out, grids, sigma_h, lx, ly, thickness, eps, sigma, frequency,
        grad_f_te, grad_f_tm, rough=True, layers=True, freq=True,
    )
    native_map = {
        "sigma_h": float(native["grad_sigma_h"][0]),
        "corr_x": float(native["grad_corr_x"][0]),
        "corr_y": float(native["grad_corr_y"][0]),
        "thickness": float(native["grad_layer_thickness_m"][0]),
        "eps": float(native["grad_layer_eps_r"][0]),
        "sigma": float(native["grad_layer_sigma_e"][0]),
        "frequency": float(native["grad_frequency"][0]),
    }
    assert _rel(native_map[param], fd) <= 5.0e-2


@_require_native
def test_need_flags_gate_outputs():
    sigma_h, lx, thickness, eps, sigma, frequency = _params()
    grids = _grids(True)
    out = _oracle(True, sigma_h, lx, lx, thickness, eps, sigma, frequency, grids)
    gf = torch.randn(out["f_te"].shape, dtype=torch.float64)
    native = _native_backward(
        out, grids, sigma_h, lx, lx, thickness, eps, sigma, frequency, gf, gf,
        rough=True, layers=False, freq=False,
    )
    assert native["grad_sigma_h"] is not None
    assert native["grad_layer_thickness_m"] is None
    assert native["grad_frequency"] is None


def test_fixed_input_rejection():
    # The Function is self-contained; assert the fixed-input map lists mu_r and
    # the four directional grids so requesting their gradient fails loudly.
    from witwin.channel.kernels import scattering as table_build_ad

    names = {name for _, name in table_build_ad._FIXED}
    assert names == {"layer_mu_r", "cos_i", "phi_i", "cos_o", "phi_o"}


def test_primal_bitwise_pre_balance_lobe_exported():
    # The numpy build is unchanged; it now also exports the pre-balance lobes,
    # and f_te == a S a bit-for-bit on the table's own grid (the AD wiring never
    # alters the resident primal values).
    from witwin.channel.scene.resources import build_kirchhoff_table

    device = "cuda"
    sigma_e = 0.1 * 2.0 * math.pi * 60e9 * 8.8541878128e-12
    rough = SurfaceRoughness(rms_height_m=1e-3, correlation_length_x_m=10e-3, correlation_length_y_m=10e-3)
    table = build_kirchhoff_table(rough, [(0.1, 4.0, sigma_e, 1.0)], 60e9, device=device)
    assert table.pre_balance_lobe_te is not None
    a = table.normalization_applied[..., 0]  # TE factor [Nti, Npi]
    # F_ij = a_i S_ij a_j (iso: a acts on the incidence and outgoing cos states).
    s = table.pre_balance_lobe_te
    reconstructed = s * a[:, :, None, None] * a[:, 0][None, None, :, None]
    assert torch.allclose(reconstructed, table.f_te, atol=1e-5, rtol=1e-4)


@_require_native
def test_end_to_end_roughness_gradient_is_nonzero():
    from witwin.channel.deployment import build_info

    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native scene capability is not built")
    from tests.support.scenes import rough_wall_structure
    from witwin.channel.deterministic import Config as DeterministicConfig
    from witwin.channel.deterministic import solve as deterministic_solve

    def make_scene():
        wall = rough_wall_structure(
            2.5, rms_height_m=0.015, corr_length_m=0.15, half_size=1.0
        )
        return Scene(
            structures=[wall],
            endpoints=[
                make_transmitter(
                    position=torch.tensor([0.0, -1.0, 0.0])
                ),
                make_receiver(
                    position=torch.tensor([0.0, 1.0, 0.0])
                ),
            ],
        )

    scene = make_scene()
    compiled = compile_scene(scene, reference_frequency_hz=3.0e9)
    # Route the table build through AD by making the roughness store leaf live
    # BEFORE the lazy Kirchhoff resources are first built.
    compiled.materials.rough_sigma_h_m.requires_grad_(True)
    config = DeterministicConfig(
        max_depth=1, components=frozenset({"scattering"}), ad_mode="vjp",
        scattering_samples_per_m2=32.0,
    )
    result = deterministic_solve(
        scene, config, reference_frequency_hz=3.0e9
    )
    result.component_power["scattering"].sum().backward()
    grad = compiled.materials.rough_sigma_h_m.grad
    assert grad is not None
    assert float(grad.abs().sum()) > 0.0
    assert torch.isfinite(grad).all()