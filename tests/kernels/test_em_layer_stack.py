# Copyright Xingyu Chen.
# Analytic and oracle-parity tests for the shared em/ layer-stack core.

"""Analytic and oracle-parity tests for the shared em/ layer-stack core."""

import cmath
import math

import pytest
import torch

from witwin.channel.kernels import materials as ops

EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
C0 = 299792458.0

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA torch is required"
)


def _csr(materials: list[list[tuple[float, float, float, float]]]) -> dict[str, torch.Tensor]:
    offsets: list[int] = []
    counts: list[int] = []
    thickness: list[float] = []
    eps_r: list[float] = []
    sigma_e: list[float] = []
    mu_r: list[float] = []
    for layers in materials:
        offsets.append(len(thickness))
        counts.append(len(layers))
        for layer in layers:
            thickness.append(layer[0])
            eps_r.append(layer[1])
            sigma_e.append(layer[2])
            mu_r.append(layer[3])
    return {
        "layer_offset": torch.tensor(offsets, device="cuda", dtype=torch.int32),
        "layer_count": torch.tensor(counts, device="cuda", dtype=torch.int32),
        "layer_thickness_m": torch.tensor(thickness, device="cuda", dtype=torch.float32),
        "layer_eps_r": torch.tensor(eps_r, device="cuda", dtype=torch.float32),
        "layer_sigma_e": torch.tensor(sigma_e, device="cuda", dtype=torch.float32),
        "layer_mu_r": torch.tensor(mu_r, device="cuda", dtype=torch.float32),
    }


def _eval(
    cos_thetas: list[float],
    material_ids: list[int],
    materials: list[list[tuple[float, float, float, float]]],
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    return ops.em_layer_stack_eval(
        torch.tensor(cos_thetas, device="cuda", dtype=torch.float32),
        torch.tensor(material_ids, device="cuda", dtype=torch.int32),
        **_csr(materials),
        frequency_hz=frequency_hz,
    )


def _passive_sqrt(z: complex) -> complex:
    magnitude = abs(z)
    return complex(
        math.sqrt(max(0.5 * (magnitude + z.real), 0.0)),
        -math.sqrt(max(0.5 * (magnitude - z.real), 0.0)),
    )


def _transfer_matrix_rt(
    layers: list[tuple[float, float, float, float]],
    cos_theta_i: float,
    frequency_hz: float,
    pol: str,
) -> tuple[complex, complex]:
    omega = 2.0 * math.pi * frequency_hz
    k0 = omega / C0
    k_par = k0 * math.sqrt(max(0.0, 1.0 - cos_theta_i * cos_theta_i))

    def kz_and_admittance(eps_r: float, sigma: float, mu_r: float) -> tuple[complex, complex]:
        eps = EPS0 * eps_r - 1j * sigma / omega
        mu = MU0 * mu_r
        k = omega * _passive_sqrt(eps * mu)
        k_z = _passive_sqrt(k * k - k_par * k_par)
        admittance = k_z / (omega * mu) if pol == "te" else omega * eps / k_z
        return k_z, admittance

    _, y_entry = kz_and_admittance(1.0, 0.0, 1.0)
    m11, m12, m21, m22 = 1.0 + 0j, 0j, 0j, 1.0 + 0j
    for thickness, eps_r, sigma, mu_r in layers:
        k_z, admittance = kz_and_admittance(eps_r, sigma, mu_r)
        delta = k_z * thickness
        a11 = cmath.cos(delta)
        a12 = 1j * cmath.sin(delta) / admittance
        a21 = 1j * admittance * cmath.sin(delta)
        a22 = cmath.cos(delta)
        m11, m12, m21, m22 = (
            m11 * a11 + m12 * a21,
            m11 * a12 + m12 * a22,
            m21 * a11 + m22 * a21,
            m21 * a12 + m22 * a22,
        )
    y_exit = y_entry
    b = m11 + m12 * y_exit
    c = m21 + m22 * y_exit
    r = (y_entry * b - c) / (y_entry * b + c)
    t = 2.0 * y_entry / (y_entry * b + c)
    return r, t


def _complex(out: dict[str, torch.Tensor], name: str, index: int = 0) -> complex:
    return complex(
        out[f"{name}_real"][index].item(), out[f"{name}_imag"][index].item()
    )


def test_normal_incidence_lossless_matches_impedance_formula():
    frequency = 3.0e9
    layer = (0.03, 4.0, 0.0, 1.0)
    out = _eval([1.0], [0], [[layer]], frequency)

    # Independent impedance-formula reference: normal-incidence slab with
    # index n = 2 in vacuum, interface amplitude (1 - n)/(1 + n).
    n = 2.0
    k0 = 2.0 * math.pi * frequency / C0
    r12 = (1.0 - n) / (1.0 + n)
    p = cmath.exp(-1j * n * k0 * layer[0])
    r_expected = r12 * (1.0 - p * p) / (1.0 - r12 * r12 * p * p)

    for pol in ("te", "tm"):
        r_oracle, t_oracle = _transfer_matrix_rt([layer], 1.0, frequency, pol)
        r_native = _complex(out, f"r_{pol}")
        t_native = _complex(out, f"t_{pol}")
        assert abs(r_native - r_oracle) <= 2.0e-5 * max(abs(r_oracle), 1.0)
        assert abs(t_native - t_oracle) <= 2.0e-5 * max(abs(t_oracle), 1.0)
    # At normal incidence the TE and TM amplitudes coincide (up to the shared
    # tangential-E convention) and match the impedance formula.
    assert abs(_complex(out, "r_te") - r_expected) <= 5.0e-5
    assert abs(_complex(out, "r_tm") - r_expected) <= 5.0e-5
    assert abs(out["cap_R_te"][0].item() + out["cap_T_te"][0].item() - 1.0) <= 1.0e-5


def test_brewster_angle_tm_reflection_vanishes():
    # Lossless non-magnetic slab: at theta_B = atan(n) the TM interface
    # coefficient vanishes at both faces, so the full stack r_TM ~ 0 while
    # r_TE stays finite.
    frequency = 6.0e9
    n = 2.0
    cos_brewster = 1.0 / math.sqrt(1.0 + n * n)
    out = _eval([cos_brewster], [0], [[(0.02, n * n, 0.0, 1.0)]], frequency)
    assert out["cap_R_tm"][0].item() < 1.0e-8
    assert out["cap_R_te"][0].item() > 0.05
    assert abs(out["cap_R_tm"][0].item() + out["cap_T_tm"][0].item() - 1.0) <= 1.0e-5


def test_pec_limit_reflects_everything_without_nan():
    frequency = 3.0e9
    out = _eval(
        [1.0, 0.5],
        [0, 0],
        [[(0.1, 1.0, 1.0e9, 1.0)]],
        frequency,
    )
    for name in ("cap_R_te", "cap_R_tm", "cap_T_te", "cap_T_tm"):
        assert torch.isfinite(out[name]).all()
    assert (out["cap_R_te"] > 0.999).all()
    assert (out["cap_R_tm"] > 0.999).all()
    assert (out["cap_T_te"] < 1.0e-20).all()
    assert (out["cap_T_tm"] < 1.0e-20).all()


def test_vacuum_single_layer_is_pure_interior_phase():
    frequency = 3.0e9
    thickness = 0.5
    k0 = 2.0 * math.pi * frequency / C0
    for cos_theta in (1.0, math.cos(math.radians(45.0))):
        out = _eval([cos_theta], [0], [[(thickness, 1.0, 0.0, 1.0)]], frequency)
        expected = cmath.exp(-1j * k0 * cos_theta * thickness)
        for pol in ("te", "tm"):
            r = _complex(out, f"r_{pol}")
            t = _complex(out, f"t_{pol}")
            assert abs(r) < 1.0e-6
            assert abs(t.real - expected.real) < 2.0e-5
            assert abs(t.imag - expected.imag) < 2.0e-5


def test_lossless_stack_conserves_energy():
    frequency = 5.0e9
    materials = [
        [(0.04, 4.0, 0.0, 1.0)],
        [(0.02, 2.5, 0.0, 1.0), (0.03, 6.0, 0.0, 2.0)],
    ]
    cos_thetas = [1.0, 0.9, 0.6, 0.3]
    out = _eval(
        cos_thetas + cos_thetas,
        [0] * len(cos_thetas) + [1] * len(cos_thetas),
        materials,
        frequency,
    )
    for pol in ("te", "tm"):
        budget = out[f"cap_R_{pol}"] + out[f"cap_T_{pol}"]
        torch.testing.assert_close(
            budget, torch.ones_like(budget), rtol=0.0, atol=1.0e-5
        )


def test_extreme_conductivity_and_thickness_stay_finite():
    frequency = 2.0e9
    out = _eval(
        [1.0, 0.7, 0.1],
        [0, 0, 0],
        [[(10.0, 3.0, 1.0e9, 1.0)]],
        frequency,
    )
    for name in (
        "r_te_real", "r_te_imag", "r_tm_real", "r_tm_imag",
        "t_te_real", "t_te_imag", "t_tm_real", "t_tm_imag",
        "cap_R_te", "cap_R_tm", "cap_T_te", "cap_T_tm",
    ):
        assert torch.isfinite(out[name]).all(), name


def test_two_half_layers_match_one_full_layer():
    frequency = 4.0e9
    full = (0.1, 4.5, 0.02, 1.0)
    half = (0.05, 4.5, 0.02, 1.0)
    out = _eval(
        [0.8, 0.8],
        [0, 1],
        [[full], [half, half]],
        frequency,
    )
    for pol in ("te", "tm"):
        r_full = _complex(out, f"r_{pol}", 0)
        r_split = _complex(out, f"r_{pol}", 1)
        t_full = _complex(out, f"t_{pol}", 0)
        t_split = _complex(out, f"t_{pol}", 1)
        assert abs(r_full - r_split) <= 1.0e-5 * max(abs(r_full), 1.0)
        assert abs(t_full - t_split) <= 1.0e-5 * max(abs(t_full), 1.0)


def test_single_layer_matches_complex128_oracle_at_oblique_lossy_conditions():
    # Static parity anchor for the em/ core: the TE amplitude also matches the
    # frozen field_reflection slab formula (same admittance algebra); the TM
    # amplitude is checked against the transfer-matrix oracle because the
    # legacy reflection kernel uses the opposite TM basis sign convention.
    frequency = 3.5e9
    layer = (0.12, 4.0, 0.025, 1.0)
    cos_thetas = [1.0, 0.85, 0.55, 0.25]
    out = _eval(cos_thetas, [0] * len(cos_thetas), [[layer]], frequency)
    for index, cos_theta in enumerate(cos_thetas):
        for pol in ("te", "tm"):
            r_oracle, t_oracle = _transfer_matrix_rt([layer], cos_theta, frequency, pol)
            r_native = _complex(out, f"r_{pol}", index)
            t_native = _complex(out, f"t_{pol}", index)
            assert abs(r_native - r_oracle) <= 2.0e-5 * max(abs(r_oracle), 1.0)
            assert abs(t_native - t_oracle) <= 2.0e-5 * max(abs(t_oracle), 1.0)

    # Frozen-kernel parity (TE, normal incidence): the legacy slab_fresnel
    # formula r = r_int*(1 - p^2)/(1 - r_int^2*p^2) with
    # r_int = (mu - sqrt(mu*eta))/(mu + sqrt(mu*eta)).
    omega = 2.0 * math.pi * frequency
    eta = layer[1] - 1j * layer[2] / (omega * EPS0)
    root = _passive_sqrt(eta * 1.0)
    r_int = (1.0 - root) / (1.0 + root)
    q = 2.0 * math.pi * layer[0] / (C0 / frequency) * root
    p2 = cmath.exp(-2j * q)
    legacy = r_int * (1.0 - p2) / (1.0 - r_int * r_int * p2)
    assert abs(_complex(out, "r_te", 0) - legacy) <= 5.0e-5


def test_zero_layer_material_is_transparent():
    out = _eval([0.7], [0], [[]], 3.0e9)
    assert abs(_complex(out, "r_te")) == 0.0
    assert _complex(out, "t_te") == 1.0 + 0j
    assert out["cap_T_te"][0].item() == 1.0
    assert out["cap_R_tm"][0].item() == 0.0


def test_direct_native_ad_companions_preserve_schema_and_optional_contract():
    frequency = 3.0e9
    cos_theta = torch.tensor([0.7], device="cuda", dtype=torch.float32)
    material_id = torch.tensor([0], device="cuda", dtype=torch.int32)
    layers = _csr([[(0.1, 4.0, 0.02, 1.0)]])
    primal = ops.em_layer_stack_eval(
        cos_theta,
        material_id,
        **layers,
        frequency_hz=frequency,
    )
    grad_outputs = tuple(
        torch.ones_like(primal[name]) if index % 2 == 0 else None
        for index, name in enumerate(ops._EM_LAYER_STACK_FIELDS)
    )

    backward = ops.em_layer_stack_backward(
        cos_theta,
        material_id,
        **layers,
        grad_outputs=grad_outputs,
        frequency_hz=frequency,
        need_cos_theta=True,
        need_layers=True,
        need_frequency=True,
    )
    assert set(backward) == {
        "grad_cos_theta",
        "grad_layer_thickness_m",
        "grad_layer_eps_r",
        "grad_layer_sigma_e",
        "grad_frequency",
    }
    assert backward["grad_cos_theta"].shape == cos_theta.shape
    assert backward["grad_layer_thickness_m"].shape == layers["layer_thickness_m"].shape
    assert backward["grad_layer_eps_r"].shape == layers["layer_eps_r"].shape
    assert backward["grad_layer_sigma_e"].shape == layers["layer_sigma_e"].shape
    assert backward["grad_frequency"].shape == (1,)

    jvp = ops.em_layer_stack_jvp(
        cos_theta,
        material_id,
        **layers,
        frequency_hz=frequency,
        tangent_cos_theta=torch.full_like(cos_theta, 0.1),
        tangent_layer_thickness=torch.full_like(layers["layer_thickness_m"], 0.01),
        tangent_layer_eps_r=torch.full_like(layers["layer_eps_r"], 0.2),
        tangent_layer_sigma_e=None,
        tangent_frequency=1.0e6,
    )
    assert set(jvp) == set(ops._EM_LAYER_STACK_FIELDS)
    assert all(value.shape == cos_theta.shape for value in jvp.values())

    with pytest.raises(ValueError, match="one cotangent slot"):
        ops.em_layer_stack_backward(
            cos_theta,
            material_id,
            **layers,
            grad_outputs=(None,),
            frequency_hz=frequency,
            need_cos_theta=False,
            need_layers=False,
            need_frequency=False,
        )


def test_layer_stack_facade_rejects_row_shape_and_nonpositive_frequency():
    cos_theta = torch.tensor([0.7], device="cuda", dtype=torch.float32)
    layers = _csr([[(0.1, 4.0, 0.02, 1.0)]])
    with pytest.raises(ValueError, match="must match cos_theta length"):
        ops.em_layer_stack_eval(
            cos_theta,
            torch.tensor([0, 0], device="cuda", dtype=torch.int32),
            **layers,
            frequency_hz=3.0e9,
        )
    with pytest.raises(ValueError, match="frequency_hz must be positive"):
        ops.em_layer_stack_eval(
            cos_theta,
            torch.tensor([0], device="cuda", dtype=torch.int32),
            **layers,
            frequency_hz=0.0,
        )