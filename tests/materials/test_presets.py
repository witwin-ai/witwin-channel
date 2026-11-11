# Copyright Xingyu Chen.
# Tests ITU-R P.2040 material presets and TR 38.901/dipole antenna pattern factories.

from __future__ import annotations

import math

import pytest
import torch

from witwin.core import AntennaPattern, MaterialLayer, PhysicalMaterial, PowerLawDispersion
from witwin.channel.presets import (
    ITU_MATERIALS,
    dipole_pattern,
    itu_material,
    itu_material_names,
    itu_properties,
    tr38901_pattern,
)

REQUIRED = {
    "concrete", "brick", "plasterboard", "wood", "glass", "ceiling_board", "chipboard",
    "floorboard", "metal", "very_dry_ground", "medium_dry_ground", "wet_ground",
}


def test_catalogue_covers_the_itu_table_without_overlapping_rows() -> None:
    assert REQUIRED <= set(itu_material_names())
    for bands in ITU_MATERIALS.values():
        ordered = sorted(bands, key=lambda band: band.min_frequency_hz)
        for low, high in zip(ordered, ordered[1:]):
            assert low.max_frequency_hz < high.min_frequency_hz


@pytest.mark.parametrize(
    ("name", "frequency_hz", "eps_r", "sigma"),
    [
        ("concrete", 1e9, 5.24, 0.0462),
        ("concrete", 28e9, 5.24, 0.0462 * 28**0.7822),
        ("medium_dry_ground", 4e9, 15.0 * 4**-0.1, 0.035 * 4**1.63),
        ("glass", 300e9, 5.79, 0.0004 * 300**1.658),
    ],
)
def test_properties_follow_the_power_law(name, frequency_hz, eps_r, sigma) -> None:
    got_eps, got_sigma = itu_properties(name, frequency_hz)

    assert got_eps == pytest.approx(eps_r, rel=1e-12)
    assert got_sigma == pytest.approx(sigma, rel=1e-12)


def test_out_of_range_and_unknown_materials_fail_clearly() -> None:
    with pytest.raises(ValueError, match=r"'brick' is defined for 1-40 GHz; 60 GHz"):
        itu_material("brick", 60e9)
    with pytest.raises(ValueError, match=r"5-50 GHz is outside"):
        itu_material("brick", (5e9, 50e9))
    with pytest.raises(ValueError, match="unknown ITU-R P.2040 material 'adamantium'"):
        itu_material("adamantium", 3e9)


@pytest.mark.parametrize("frequency_hz", [2.4e9, 28e9])
def test_dispersive_material_evaluates_the_itu_law_through_core(frequency_hz) -> None:
    material = itu_material("concrete", (1e9, 40e9), thickness_m=0.3)

    assert isinstance(material, PhysicalMaterial)
    assert isinstance(material.dispersion, PowerLawDispersion)
    assert material.capabilities().dispersive
    assert material.thickness_m == 0.3
    eps_r, sigma = itu_properties("concrete", frequency_hz)
    sample = material.evaluate_at_frequency(frequency_hz)
    assert sample.eps_r.real == pytest.approx(eps_r, rel=1e-12)
    assert sample.sigma_e == pytest.approx(sigma, rel=1e-12)
    # Channel compiles a single-layer material through MaterialLayer with the same dispersion.
    layer = MaterialLayer(
        thickness_m=0.3, eps_r=material.eps_r, sigma_e=material.sigma_e,
        dispersion=material.dispersion,
    )
    assert layer.evaluate_at_frequency(frequency_hz).eps_r == sample.eps_r


def test_dispersive_material_refuses_evaluation_outside_its_range() -> None:
    material = itu_material("wet_ground", 3e9)

    with pytest.raises(ValueError, match=r"'wet_ground' is defined for 1-10 GHz; 28 GHz"):
        material.evaluate_at_frequency(28e9)


def test_non_dispersive_material_is_frozen_at_the_requested_frequency() -> None:
    material = itu_material("glass", 5e9, dispersive=False, scattering_coefficient=0.2)
    eps_r, sigma = itu_properties("glass", 5e9)

    assert material.dispersion is None
    assert (material.eps_r, material.sigma_e) == (pytest.approx(eps_r), pytest.approx(sigma))
    assert material.scattering_coefficient == 0.2
    assert material.name == "itu_glass"
    with pytest.raises(ValueError, match="not a band"):
        itu_material("glass", (5e9, 6e9), dispersive=False)


def _sphere(n_theta: int = 721, n_phi: int = 1440) -> tuple[torch.Tensor, torch.Tensor]:
    theta = (torch.arange(n_theta, dtype=torch.float64) + 0.5) * math.pi / n_theta
    phi = torch.arange(n_phi, dtype=torch.float64) * 2 * math.pi / n_phi
    t, p = torch.meshgrid(theta, phi, indexing="ij")
    direction = torch.stack((t.sin() * p.cos(), t.sin() * p.sin(), t.cos()), dim=-1)
    weight = t.sin() * (math.pi / n_theta) * (2 * math.pi / n_phi)
    return direction.to(torch.float32), weight


@pytest.mark.parametrize("kind", ["short", "half_wave"])
def test_dipoles_are_normalized_to_unit_mean_power(kind) -> None:
    pattern = dipole_pattern(kind)
    direction, weight = _sphere()

    power = pattern.custom(direction).abs().double().square()

    assert isinstance(pattern, AntennaPattern) and pattern.kind == "custom"
    assert float((power * weight).sum()) / (4 * math.pi) == pytest.approx(1.0, rel=2e-3)
    broadside = pattern.custom(torch.tensor([[1.0, 0.0, 0.0]]))
    axial = pattern.custom(torch.tensor([[0.0, 0.0, 1.0]]))
    assert float(broadside.abs().square()) == pytest.approx(1.5 if kind == "short" else 1.6409)
    assert float(axial.abs()) == pytest.approx(0.0, abs=1e-6)


def test_tr38901_element_peak_beamwidth_and_floor() -> None:
    pattern = tr38901_pattern()
    directions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [math.cos(math.radians(32.5)), math.sin(math.radians(32.5)), 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    gain_db = 20 * torch.log10(pattern.custom(directions).abs())

    torch.testing.assert_close(gain_db, torch.tensor([8.0, 5.0, -22.0]), atol=1e-4, rtol=0)
    with pytest.raises(ValueError, match="beamwidth_deg"):
        tr38901_pattern(beamwidth_deg=0.0)
