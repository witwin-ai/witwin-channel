"""Kirchhoff ensemble table: energy, limits, reciprocity, applicability.

Reference material follows the contract test spec (eps 4 - 0.1j,
sigma_h = 1 mm). The correlation length / frequency pairs are chosen inside
the applicability domain ``k0*l >= 6``: at 6 GHz that requires
l >= 4.8 cm (so l = 10 cm is used there); the l = 10 mm surface is tested
at 60 GHz where k0*l = 12.6. The 6 GHz / l = 10 mm combination is the
out-of-domain case and must raise.
"""

import math

import pytest
import torch

from witwin.channel.core.materials import Roughness
from witwin.channel.scattering import build_kirchhoff_table, eval_bsdf

_EPS0 = 8.8541878128e-12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _lossy_layers(frequency_hz: float, thickness_m: float) -> list[tuple]:
    """Single slab with eps_r = 4 - 0.1j at ``frequency_hz`` (opaque-ish)."""

    sigma_e = 0.1 * 2.0 * math.pi * frequency_hz * _EPS0
    return [(thickness_m, 4.0, sigma_e, 1.0)]


@pytest.fixture(scope="module")
def table_60ghz():
    rough = Roughness(rms_height_m=1e-3, corr_length_x_m=10e-3, corr_length_y_m=10e-3)
    return build_kirchhoff_table(rough, _lossy_layers(60e9, 0.1), 60e9, device=DEVICE)


def _hemisphere_sum(table, f, ti: int, pi: int) -> float:
    weight = table.cos_theta_o[:, None] * table.bin_solid_angle
    return float((f[ti, pi] * weight).sum())


@pytest.mark.parametrize("ti", [2, 6, 12, 20, 27, 31])
def test_energy_matches_r_diff(table_60ghz, ti):
    """Hemisphere-summed f*|cos| equals R_diff per pol within 1e-3 relative."""

    table = table_60ghz
    for f, r_diff in (
        (table.f_te, table.r_diff_te),
        (table.f_tm, table.r_diff_tm),
    ):
        total = _hemisphere_sum(table, f, ti, 0)
        budget = float(r_diff[ti, 0])
        assert budget > 0.0
        assert abs(total - budget) < 1e-3 * budget


def test_energy_6ghz_contract_material():
    """The contract's 6 GHz eps 4-0.1j material builds and conserves energy."""

    rough = Roughness(rms_height_m=1e-3, corr_length_x_m=0.1, corr_length_y_m=0.1)
    table = build_kirchhoff_table(rough, _lossy_layers(6e9, 0.5), 6e9, device=DEVICE)
    for ti in (8, 16, 24, 31):
        for f, r_diff in ((table.f_te, table.r_diff_te), (table.f_tm, table.r_diff_tm)):
            total = _hemisphere_sum(table, f, ti, 0)
            budget = float(r_diff[ti, 0])
            assert budget > 0.0
            assert abs(total - budget) < 1e-3 * budget


def test_smooth_limit_near_zero_lobe():
    """sigma_h -> 1e-6 gives R_diff ~ 0 and a near-zero diffuse lobe."""

    rough = Roughness(rms_height_m=1e-6, corr_length_x_m=0.1, corr_length_y_m=0.1)
    table = build_kirchhoff_table(rough, _lossy_layers(60e9, 0.1), 60e9, device=DEVICE)
    assert float(table.r_diff_unpol.max()) < 1e-6
    assert float(table.f_te.max()) < 1e-3
    assert float(table.f_tm.max()) < 1e-3
    # Energy identity still holds bin by bin (normalization is exact).
    total = _hemisphere_sum(table, table.f_te, 16, 0)
    assert abs(total - float(table.r_diff_te[16, 0])) < 1e-6


def test_reciprocity(table_60ghz):
    """The final energy-normalized production table is reciprocal."""

    table = table_60ghz
    assert table.reciprocity_error < 1e-3
    # For the isotropic table with implicit
    # incidence azimuth 0, the swapped pair maps to the flipped phi_o index
    # (delta_phi -> -delta_phi).
    f = table.f_te
    swapped = torch.flip(f.permute(2, 1, 0, 3), dims=(3,))
    peak = float(f.max())
    assert float((f - swapped).abs().max()) < 1e-6 * peak


def test_final_table_reciprocity_at_bin_centers(table_60ghz):
    """Native CUDA evaluation preserves the final table's swap identity."""

    table = table_60ghz
    ti, to, po = 20, 12, 17
    cos_i = float(table.cos_theta_i[ti])
    cos_o = float(table.cos_theta_o[to])
    phi = float(table.phi_o[po])
    sin_i = math.sqrt(1.0 - cos_i**2)
    sin_o = math.sqrt(1.0 - cos_o**2)
    wi = torch.tensor([[sin_i, 0.0, cos_i]], device=table.device)
    wo = torch.tensor(
        [[sin_o * math.cos(phi), sin_o * math.sin(phi), cos_o]], device=table.device
    )
    valid = torch.ones(1, dtype=torch.bool, device=wi.device)
    f_te, f_tm = eval_bsdf(table, valid, wi, wo)
    assert f_te.item() == pytest.approx(float(table.f_te[ti, 0, to, po]), rel=1e-5)
    assert f_tm.item() == pytest.approx(float(table.f_tm[ti, 0, to, po]), rel=1e-5)
    f_te_rev, _ = eval_bsdf(table, valid, wo, wi)
    assert f_te.item() == pytest.approx(f_te_rev.item(), rel=1e-5, abs=1e-8)


def test_anisotropic_table_has_phi_i_axis(table_60ghz):
    rough = Roughness(rms_height_m=1e-3, corr_length_x_m=10e-3, corr_length_y_m=20e-3)
    table = build_kirchhoff_table(rough, _lossy_layers(60e9, 0.1), 60e9, device=DEVICE)
    assert table.anisotropic
    assert table.phi_i.numel() == 64
    assert table.f_te.shape == (32, 64, 32, 64)
    # Isotropic surface collapses the phi_i axis to one entry.
    assert not table_60ghz.anisotropic
    assert table_60ghz.phi_i.numel() == 1
    assert table_60ghz.f_te.shape == (32, 1, 32, 64)


def test_symmetric_balance_factors_are_finite(table_60ghz):
    """The two-sided balance factors are finite and nonnegative."""

    table = table_60ghz
    scales = table.normalization_applied
    assert bool(torch.isfinite(scales).all())
    assert float(scales.min()) >= 0.0
    assert float(scales.max()) > 0.0


def test_out_of_domain_roughness_raises():
    """Applicability guards: k0*l >= 6 and slope <= 0.5, contract wording."""

    # 6 GHz with l = 10 mm: k0*l = 1.26 < 6 (tangent-plane violated).
    rough = Roughness(rms_height_m=1e-3, corr_length_x_m=10e-3, corr_length_y_m=10e-3)
    with pytest.raises(ValueError, match="kirchhoff_domain_exceeded"):
        build_kirchhoff_table(rough, _lossy_layers(6e9, 0.1), 6e9, device=DEVICE)
    # 60 GHz with sigma_h = 4 mm, l = 10 mm: slope 0.57 > 0.5.
    steep = Roughness(rms_height_m=4e-3, corr_length_x_m=10e-3, corr_length_y_m=10e-3)
    with pytest.raises(ValueError, match="kirchhoff_domain_exceeded"):
        build_kirchhoff_table(steep, _lossy_layers(60e9, 0.1), 60e9, device=DEVICE)


def test_domain_metadata(table_60ghz):
    table = table_60ghz
    assert table.tangent_plane_ok and table.slope_ok
    assert table.k0_l_min == pytest.approx(
        2.0 * math.pi * 60e9 / 299792458.0 * 10e-3, rel=1e-6
    )
    assert table.rms_slope_max == pytest.approx(math.sqrt(2.0) * 0.1, rel=1e-6)
    assert table.frequency_hz == 60e9
