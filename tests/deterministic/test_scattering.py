"""Deterministic Kirchhoff rough-surface scattering (plan 05 wave 3).

Normalization cross-check used throughout: in the narrow-lobe limit the
patch-quadrature plane sum collapses to the image-source correspondence
``P_scatter -> R_diff * P_t * (lambda/(4*pi*(r1+r2)))^2``, so scattering plus
the C_r-attenuated specular approaches the smooth-wall reflection power.
The production table redistributes a peaked lobe on its fixed 32x64 grid
(exact per-bin energy, interpolated eval), which costs ~10% of the
image-correspondence value for near-smooth surfaces; the energy inequality
(passivity) holds unconditionally.
"""

import math

import numpy as np
import pytest
import torch

from tests.support.scenes import rough_wall_structure
from witwin.core import Scene
from tests.support.core_world import make_receiver, make_receiver_grid, make_transmitter
from witwin.channel.deployment import build_info
from witwin.core import PhaseScreen, SurfaceRoughness
from witwin.channel.deterministic import Config, solve
from tests.reference.em_oracle import C0, layer_stack_rt
from witwin.channel.scene.resources import (
    N_COS_THETA_O,
    N_PHI_O,
    _cos_centers,
    build_kirchhoff_table,
    generate_gaussian_realization,
    realization_seed,
)

_FREQUENCY_HZ = 3.0e9
_K0 = 2.0 * math.pi * _FREQUENCY_HZ / C0
_HALF_SIZE = 2.0
_LAYERS = [(0.1, 4.0, 0.01, 1.0)]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA torch is required"
)


def _require_rayd() -> None:
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native scene capability is not built")


def _scene(
    rms_height_m: float,
    corr_length_m: float = 0.15,
    *,
    half_size: float = _HALF_SIZE,
    phase_screen: object | None = None,
    swap_endpoints: bool = False,
    receivers: list | None = None,
) -> Scene:
    wall = rough_wall_structure(
        2.5,
        rms_height_m=rms_height_m,
        corr_length_m=corr_length_m,
        half_size=half_size,
        phase_screen=phase_screen,
        with_uv=phase_screen is not None,
    )
    tx_pos = torch.tensor([0.0, -1.0, 0.0])
    rx_pos = torch.tensor([0.0, 1.0, 0.0])
    if swap_endpoints:
        tx_pos, rx_pos = rx_pos, tx_pos
    return Scene(
        structures=[wall],
        endpoints=[
            make_transmitter(position=tx_pos),
            *(receivers or [make_receiver(position=rx_pos)]),
        ],
    )


def _config(**overrides) -> Config:
    settings = {
        "max_depth": 1,
        "components": {"reflection", "scattering"},
        "scattering_samples_per_m2": 64.0,
        "scattering_max_paths_per_pair": 65536,
    }
    settings.update(overrides)
    return Config(**settings)


def _specular_cr2(rms_height_m: float) -> float:
    cos_theta = 2.5 / math.sqrt(2.5**2 + 1.0)
    return math.exp(-2.0 * (_K0 * cos_theta * rms_height_m) ** 2) ** 2


def _smooth_reflection_power() -> float:
    result = solve(
        _scene(0.0),
        Config(max_depth=1, components={"reflection"}),
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    return float(result.component_power["reflection"])


def test_rough_wall_energy_budget_and_smooth_limit():
    _require_rayd()
    smooth = _smooth_reflection_power()

    rough = solve(_scene(0.015), _config(), reference_frequency_hz=_FREQUENCY_HZ)
    reflection = float(rough.component_power["reflection"])
    scattering = float(rough.component_power["scattering"])
    assert scattering > 0.0
    # Passivity: rough reflection + scattering never exceeds the smooth wall.
    assert (reflection + scattering) / smooth <= 1.0 + 1.0e-3
    assert rough.metadata["components"]["scattering"] == "enabled"
    assert rough.metadata["scattering"]["scattering_paths_incoherent"] is True
    assert rough.metadata["scattering"]["path_count"] > 0

    # Smooth limit: a vanishing rms height gives ~zero scattering and the
    # reflection returns to the smooth-wall value.
    tiny = solve(_scene(5.0e-5, 0.3), _config(), reference_frequency_hz=_FREQUENCY_HZ)
    assert float(tiny.component_power["scattering"]) / smooth < 1.0e-4
    assert float(tiny.component_power["reflection"]) == pytest.approx(
        smooth, rel=1.0e-3
    )


def test_narrow_lobe_normalization_cross_check():
    """Image-source correspondence: scattering ~= R_diff * smooth reflection.

    For a near-smooth surface the diffuse lobe is a narrow cone around the
    specular direction and the patch sum must recover the R_diff share of the
    smooth-wall image power. The fixed 32x64 table grid redistributes the
    peaked lobe (exact bin energy, interpolated eval), which costs ~10%; the
    band [0.75, 1.0] fails on any 4*pi / cos-factor / r^2 normalization slip.
    """

    _require_rayd()
    rough = solve(
        _scene(0.004, 0.3, half_size=1.0),
        _config(scattering_samples_per_m2=256.0),
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    smooth_small = solve(
        _scene(0.0, 0.3, half_size=1.0),
        Config(max_depth=1, components={"reflection"}),
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    expected = (1.0 - _specular_cr2(0.004)) * float(
        smooth_small.component_power["reflection"]
    )
    ratio = float(rough.component_power["scattering"]) / expected
    assert 0.75 <= ratio <= 1.0 + 1.0e-3


def test_specular_attenuation_matches_coherent_factor():
    _require_rayd()
    smooth = _smooth_reflection_power()
    for rms in (0.008, 0.015):
        rough = solve(_scene(rms), _config(), reference_frequency_hz=_FREQUENCY_HZ)
        observed = float(rough.component_power["reflection"]) / smooth
        assert observed == pytest.approx(_specular_cr2(rms), rel=1.0e-3)


def test_scattering_convergence_under_sample_doubling():
    _require_rayd()
    base = solve(
        _scene(0.015),
        _config(scattering_samples_per_m2=64.0),
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    fine = solve(
        _scene(0.015),
        _config(scattering_samples_per_m2=128.0),
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    p_base = float(base.component_power["scattering"])
    p_fine = float(fine.component_power["scattering"])
    assert abs(p_fine - p_base) / p_base < 0.02


def test_scattering_reciprocity():
    """Swapping tx/rx conserves the scattering power within quadrature
    tolerance. The analytic kernel is exactly reciprocal; the residual comes
    from the table's per-incidence-bin exact-energy normalization (the scale
    factor is a function of the incidence bin, so the frozen table is
    reciprocal only up to the bin-to-bin scale variation)."""

    _require_rayd()
    forward = solve(_scene(0.015), _config(), reference_frequency_hz=_FREQUENCY_HZ)
    reverse = solve(
        _scene(0.015, swap_endpoints=True),
        _config(),
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    p_fwd = float(forward.component_power["scattering"])
    p_rev = float(reverse.component_power["scattering"])
    assert p_fwd == pytest.approx(p_rev, rel=3.0e-2)


def test_grid_receiver_scattering_map():
    _require_rayd()
    grid = make_receiver_grid(
        origin=torch.tensor([0.0, 0.5, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(4, 4),
        spacing=(0.25, 0.25),
    )
    result = solve(
        _scene(0.015, receivers=[grid]),
        _config(scattering_samples_per_m2=16.0),
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    scattering = result.component_power["scattering"]
    assert scattering.shape == (1, 4, 4)
    assert bool((scattering > 0.0).all())


def test_table_passivity_per_incidence_angle():
    """R_coh + integrated diffuse lobe <= R_bar + 1e-3 per angle (TE and TM)."""

    rough = SurfaceRoughness(
        rms_height_m=0.015, correlation_length_x_m=0.15, correlation_length_y_m=0.15
    )
    table = build_kirchhoff_table(rough, _LAYERS, _FREQUENCY_HZ, device="cpu")
    cos_i = _cos_centers(32)
    rt = layer_stack_rt(_LAYERS, cos_i, _FREQUENCY_HZ)
    c_r2 = np.exp(-2.0 * (_K0 * cos_i * 0.015) ** 2) ** 2
    d_omega = (1.0 / N_COS_THETA_O) * (2.0 * math.pi / N_PHI_O)
    cos_o = torch.from_numpy(_cos_centers(N_COS_THETA_O)).to(torch.float32)
    weight = cos_o[None, None, :, None] * d_omega
    for channel, r_bar in (("f_te", rt.R_te), ("f_tm", rt.R_tm)):
        lobe_integral = (
            (getattr(table, channel) * weight).sum(dim=(2, 3)).squeeze(1).numpy()
        )
        r_coh = r_bar * c_r2
        assert np.all(r_coh + lobe_integral <= r_bar + 1.0e-3)


def test_incoherent_power_accumulation_in_coherent_mode():
    """Scattering folds as power: total = |coherent field|^2 + scattering."""

    _require_rayd()
    result = solve(_scene(0.015), _config(), reference_frequency_hz=_FREQUENCY_HZ)
    coherent_power = result.field.abs().square()
    expected = coherent_power + result.component_power["scattering"]
    torch.testing.assert_close(result.path_gain, expected, rtol=1.0e-5, atol=1.0e-12)


def _screen(realization_id: int, *, flat: bool = False) -> PhaseScreen:
    if flat:
        return PhaseScreen(
            height=torch.zeros(64, 64),
            height_scale_m=1.0e-9,
            realization_id=realization_id,
            mode="realization_coherent",
        )
    rough = SurfaceRoughness(
        rms_height_m=0.008, correlation_length_x_m=0.15, correlation_length_y_m=0.15
    )
    height = generate_gaussian_realization(
        rough,
        extent_m=2.0,
        resolution=256,
        seed=realization_seed(0, 1, realization_id),
        device="cpu",
    )
    return PhaseScreen(
        height=height,
        height_scale_m=1.0,
        realization_id=realization_id,
        mode="realization_coherent",
    )


def test_realization_coherent_flat_screen_matches_smooth_reflection():
    """h = 0 phase screen reproduces the smooth specular power (stationary
    phase of the patch quadrature -> image source), and it REPLACES the delta
    specular for that surface (reflection component reports zero)."""

    _require_rayd()
    smooth = solve(
        _scene(0.0, half_size=1.0),
        Config(max_depth=1, components={"reflection"}),
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    flat = solve(
        _scene(0.0, half_size=1.0, phase_screen=_screen(0, flat=True)),
        _config(),
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    p_smooth = float(smooth.component_power["reflection"])
    p_flat = float(flat.component_power["scattering"])
    assert p_flat == pytest.approx(p_smooth, rel=0.2)  # Fresnel edge ringing
    assert float(flat.component_power["reflection"]) == 0.0
    assert flat.metadata["scattering"]["realization_structure_count"] == 1


def test_realization_coherent_reproducible_and_realization_dependent():
    _require_rayd()
    cfg = _config()
    first = solve(
        _scene(0.008, half_size=1.0, phase_screen=_screen(1)),
        cfg,
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    second = solve(
        _scene(0.008, half_size=1.0, phase_screen=_screen(1)),
        cfg,
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    other = solve(
        _scene(0.008, half_size=1.0, phase_screen=_screen(2)),
        cfg,
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    p_first = first.component_power["scattering"]
    # Bit-exact reproducibility for a fixed (scene_seed, surface, realization).
    assert torch.equal(p_first, second.component_power["scattering"])
    p1 = float(p_first)
    p2 = float(other.component_power["scattering"])
    assert p2 != p1
    assert 0.05 <= p2 / p1 <= 20.0  # same order of magnitude (speckle)


def test_realization_vs_ensemble_exclusivity_raises():
    _require_rayd()
    screen = PhaseScreen(
        height=torch.zeros(16, 16),
        height_scale_m=1.0e-9,
        mode="ensemble_bsdf",
    )
    scene = _scene(0.015, phase_screen=screen)
    with pytest.raises(RuntimeError, match="never be summed"):
        solve(scene, _config(), reference_frequency_hz=_FREQUENCY_HZ)


def test_phase_screen_geometry_limit_guard():
    _require_rayd()
    steep = PhaseScreen(
        height=torch.rand(64, 64),  # O(1) meter heights over a 4 m wall
        height_scale_m=1.0,
        mode="realization_coherent",
    )
    scene = _scene(0.008, phase_screen=steep)
    with pytest.raises(RuntimeError, match="phase_screen_geometry_limit_exceeded"):
        solve(scene, _config(), reference_frequency_hz=_FREQUENCY_HZ)


def test_out_of_domain_roughness_raises_kirchhoff_domain_exceeded():
    _require_rayd()
    # k0 * corr_length = 3.1 < 6: outside the tangent-plane domain.
    scene = _scene(0.008, corr_length_m=0.05)
    with pytest.raises((RuntimeError, ValueError), match="kirchhoff_domain_exceeded"):
        solve(scene, _config(), reference_frequency_hz=_FREQUENCY_HZ)
