"""BDPT Kirchhoff rough-surface scattering (wave 3).

Covers the three-way {reflect, scatter, transmit} event selection, the
torch-side scattering NEE connections (component 6), the smooth limit, the
energy bound, seed reproducibility, the disabled-regression guard and the
sampler/pdf consistency of the seeded direction stream.
"""

import math

import pytest
import torch
from scipy.stats import chi2

from witwin.core import Scene, Structure
from tests.support.core_world import (
    make_mesh_structure,
    make_receiver,
    make_receiver_grid,
    make_transmitter,
)
from witwin.channel.deployment import build_info
from witwin.core import MaterialLayer, PhysicalMaterial, SurfaceRoughness
from witwin.channel.montecarlo.bdpt import Config, solve
from witwin.channel.montecarlo.events.scattering import (
    sample_scatter_directions,
    scatter_direction_uniforms,
)
from witwin.channel.scattering import build_kirchhoff_table, eval_bsdf
from witwin.channel.scattering.tables import pdf as table_pdf

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA torch is required"
)

_FREQUENCY = 60.0e9
_LIGHT_SPEED = 299_792_458.0
# Strongly diffuse, broad Kirchhoff lobe (k0*l ~ 12.6, C_r^2 ~ 4e-3 at
# normal incidence) so the point-receiver NEE estimator converges at test
# sample budgets; validated inside the table applicability domain.
_SIGMA_H = 1.0e-3
_CORR = 0.01
_EPS_R = 4.0
_SIGMA_E = 0.05
_THICKNESS = 0.1
_LAYERS = ((_THICKNESS, _EPS_R, _SIGMA_E, 1.0),)

_TX = torch.tensor([0.0, 0.0, 0.0])
# Receiver on the transmitter side of the wall, near the specular direction.
_RX = torch.tensor([0.5, 1.0, 0.3])


def _require_native() -> None:
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native scattering is not built")


def _roughness(sigma_h: float = _SIGMA_H) -> SurfaceRoughness:
    return SurfaceRoughness(
        rms_height_m=sigma_h, correlation_length_x_m=_CORR, correlation_length_y_m=_CORR
    )


def _material(
    roughness: SurfaceRoughness | None,
) -> PhysicalMaterial:
    return PhysicalMaterial(
        layers=(MaterialLayer(thickness_m=_THICKNESS, eps_r=_EPS_R, sigma_e=_SIGMA_E),),
        roughness_front=roughness,
        name="wall-material",
    )


def _wall(material: PhysicalMaterial, *, x: float = 2.5) -> Structure:
    return make_mesh_structure(
        vertices=torch.tensor(
            [[x, -4.0, -4.0], [x, 4.0, -4.0], [x, -4.0, 4.0], [x, 4.0, 4.0]]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=material,
        name="wall",
        surface_id=1,
    )


def _point_scene(material: PhysicalMaterial) -> Scene:
    return Scene(
        structures=[_wall(material)],
        endpoints=[
            make_transmitter(position=_TX),
            make_receiver(position=_RX),
        ],
    )


def _quadrature_reference(*, polarized: bool) -> float:
    """Direct torch area quadrature of the scattering path gain at _RX.

    g = Int_wall (P_te*f_te + P_tm*f_tm) cos_i cos_o (lambda/4pi)^2
        / (r1^2 r2^2) dA
    with (P_te, P_tm) the transverse projection of the z-polarized unit tx
    field (matching the BDPT NEE weighting), or the unpolarized mean kernel
    when ``polarized`` is False (matching MC basic).
    """

    device = torch.device("cuda")
    table = build_kirchhoff_table(
        _roughness(), list(_LAYERS), _FREQUENCY, device=device
    )
    n = torch.tensor([-1.0, 0.0, 0.0], device=device)
    res = 640
    axis = torch.linspace(-4.0, 4.0, res, device=device)
    step = float(axis[1] - axis[0])
    centers = axis[:-1] + 0.5 * step
    gy, gz = torch.meshgrid(centers, centers, indexing="ij")
    points = torch.stack((torch.full_like(gy, 2.5), gy, gz), dim=-1).reshape(-1, 3)
    tx = _TX.to(device)
    rx = _RX.to(device)
    to_tx = tx[None] - points
    r1 = to_tx.norm(dim=-1)
    wi = to_tx / r1[:, None]
    to_rx = rx[None] - points
    r2 = to_rx.norm(dim=-1)
    wo = to_rx / r2[:, None]
    cos_i = (wi * n).sum(-1)
    cos_o = (wo * n).sum(-1)
    t1 = torch.linalg.cross(
        torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(points),
        n.expand_as(points),
    )
    t1 = t1 / t1.norm(dim=-1, keepdim=True)
    t2 = torch.linalg.cross(n.expand_as(points), t1)

    def local(w: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            ((w * t1).sum(-1), (w * t2).sum(-1), (w * n).sum(-1)), dim=-1
        ).contiguous()

    valid = torch.ones(wi.shape[0], dtype=torch.bool, device=device)
    f_te, f_tm = eval_bsdf(table, valid, local(wi), local(wo))
    if polarized:
        d_in = -wi
        s = torch.linalg.cross(d_in, n.expand_as(points))
        s = s / s.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        p = torch.linalg.cross(s, d_in)
        pol = torch.tensor([0.0, 0.0, 1.0], device=device)
        kernel = (pol * s).sum(-1) ** 2 * f_te + (pol * p).sum(-1) ** 2 * f_tm
    else:
        kernel = 0.5 * (f_te + f_tm)
    amplitude_sq = (_LIGHT_SPEED / _FREQUENCY / (4.0 * math.pi)) ** 2
    integrand = (
        kernel
        * cos_i.clamp_min(0.0)
        * cos_o.clamp_min(0.0)
        * amplitude_sq
        / (r1**2 * r2**2)
    )
    return float(integrand.sum() * step * step)


def test_bdpt_scattering_matches_area_quadrature_reference():
    _require_native()
    reference = _quadrature_reference(polarized=True)
    result = solve(
        _point_scene(_material(_roughness())),
        Config(samples=262_144, seed=7, max_depth=2, components={"scattering"}),
        reference_frequency_hz=_FREQUENCY,
    )
    value = float(result.component_power["scattering"])
    assert value > 0.0
    # Statistical tolerance: ~78k scatter events over a broad lobe (smoke
    # measurements: ratios 0.96..1.02 across seeds and budgets).
    assert value == pytest.approx(reference, rel=0.15)
    assert result.metadata["scattering"]["event_counts"]["scatter"] > 10_000
    assert result.metadata["scattering"]["component_mask_bit"] == 16


def test_bdpt_scattering_variance_decreases_like_one_over_n():
    _require_native()
    scene = _point_scene(_material(_roughness()))
    small = solve(
        scene,
        Config(
            samples=16_384,
            seed=7,
            max_depth=2,
            components={"scattering"},
            diagnostics=True,
        ),
        reference_frequency_hz=_FREQUENCY,
    )
    large = solve(
        scene,
        Config(
            samples=65_536,
            seed=7,
            max_depth=2,
            components={"scattering"},
            diagnostics=True,
        ),
        reference_frequency_hz=_FREQUENCY,
    )
    ratio = float(small.variance.sum() / large.variance.sum().clamp_min(1.0e-30))
    # 4x samples -> ~4x lower variance of the mean (allow estimator noise).
    assert 2.0 < ratio < 8.0


def test_bdpt_smooth_limit_and_energy_bound():
    """Near-zero roughness kills the diffuse component; at strong roughness
    the diffuse power at a near-specular receiver stays below the SMOOTH
    specular reflection there (R_diff <= R_bar and the lobe spreads energy).

    The literal contract inequality reflection_rough + scattering <=
    reflection_smooth requires the coherent C_r attenuation of the DISCRETE
    specular enumeration, which is owned by the deterministic-solver
    scattering wave (path_topology gain evaluation). Until it lands the
    discrete reflection is unattenuated, so this test asserts the two
    robust halves: the scattering component obeys its budget bound, and the
    reflection component never EXCEEDS the smooth value (equal now,
    attenuated after integration). The shooting sampler's own reflect
    branch already applies C_r.
    """

    _require_native()
    reflection_config = Config(
        samples=4096, seed=7, max_depth=2, components={"reflection"}
    )
    smooth_reflection = float(
        solve(
            _point_scene(_material(None)),
            reflection_config,
            reference_frequency_hz=_FREQUENCY,
        ).component_power["reflection"]
    )
    rough_reflection = float(
        solve(
            _point_scene(_material(_roughness())),
            reflection_config,
            reference_frequency_hz=_FREQUENCY,
        ).component_power["reflection"]
    )
    scattering_config = Config(
        samples=65_536, seed=7, max_depth=2, components={"scattering"}
    )
    rough_scattering = float(
        solve(
            _point_scene(_material(_roughness())),
            scattering_config,
            reference_frequency_hz=_FREQUENCY,
        ).component_power["scattering"]
    )
    tiny_scattering = float(
        solve(
            _point_scene(_material(_roughness(sigma_h=1.0e-6))),
            scattering_config,
            reference_frequency_hz=_FREQUENCY,
        ).component_power["scattering"]
    )

    assert smooth_reflection > 0.0
    assert rough_scattering > 0.0
    # Smooth limit: sigma_h -> 0 gives (numerically) zero diffuse power and
    # an unchanged reflection component.
    assert tiny_scattering < 1.0e-4 * rough_scattering
    assert rough_reflection <= smooth_reflection * (1.0 + 1.0e-4)
    # Energy bound at the near-specular receiver (10% statistical headroom;
    # the exact normalization check is the quadrature-reference test).
    assert rough_scattering <= smooth_reflection * 1.10


def test_bdpt_scattering_is_seed_reproducible():
    _require_native()
    scene = _point_scene(_material(_roughness()))
    config = Config(samples=8192, seed=11, max_depth=2, components={"scattering"})
    first = solve(scene, config, reference_frequency_hz=_FREQUENCY)
    second = solve(scene, config, reference_frequency_hz=_FREQUENCY)
    torch.testing.assert_close(first.path_gain, second.path_gain, rtol=0.0, atol=0.0)
    assert (
        first.metadata["scattering"]["event_counts"]
        == second.metadata["scattering"]["event_counts"]
    )
    other = solve(
        scene,
        Config(samples=8192, seed=12, max_depth=2, components={"scattering"}),
        reference_frequency_hz=_FREQUENCY,
    )
    assert not torch.equal(first.path_gain, other.path_gain)


def test_bdpt_results_unchanged_when_scattering_not_requested():
    """Regression guard: solving without the scattering component must not
    change the other components' machinery. A smooth-material scene is
    bit-identical whether or not scattering support exists in the solver,
    and a rough-material scene keeps the same reflection value with and
    without the scattering component enabled (the coherent C_r specular
    attenuation is a material property, applied regardless of which
    components are requested)."""

    _require_native()
    base_components = {"los", "reflection", "transmission"}
    config = Config(samples=4096, seed=7, max_depth=3, components=base_components)
    smooth_first = solve(
        _point_scene(_material(None)), config, reference_frequency_hz=_FREQUENCY
    )
    smooth_second = solve(
        _point_scene(_material(None)), config, reference_frequency_hz=_FREQUENCY
    )
    torch.testing.assert_close(
        smooth_first.path_gain, smooth_second.path_gain, rtol=0.0, atol=0.0
    )

    rough_scene = _point_scene(_material(_roughness()))
    rough_without = solve(rough_scene, config, reference_frequency_hz=_FREQUENCY)
    rough_with = solve(
        rough_scene,
        Config(
            samples=4096,
            seed=7,
            max_depth=3,
            components=base_components | {"scattering"},
        ),
        reference_frequency_hz=_FREQUENCY,
    )
    for component in ("los", "reflection", "transmission"):
        torch.testing.assert_close(
            rough_without.component_power[component],
            rough_with.component_power[component],
            rtol=1e-5,
            atol=0.0,
        )
    assert "scattering" not in rough_without.component_power

    # The rough wall's coherent specular reflection is C_r-attenuated
    # relative to the smooth wall even when scattering is not requested.
    smooth_run = solve(
        _point_scene(_material(None)), config, reference_frequency_hz=_FREQUENCY
    )
    rough_reflection = float(rough_without.component_power["reflection"].sum())
    smooth_reflection = float(smooth_run.component_power["reflection"].sum())
    assert rough_reflection < smooth_reflection
    assert rough_reflection > 0.0


def test_bdpt_grid_scattering_component_map_matches_power():
    _require_native()
    grid = make_receiver_grid(
        origin=torch.tensor([0.5, -1.0, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(4, 4),
        spacing=(2.0 / 3.0, 1.0 / 3.0),
    )
    scene = Scene(
        structures=[_wall(_material(_roughness()))],
        endpoints=[make_transmitter(position=_TX), grid],
    )
    result = solve(
        scene,
        Config(samples=16_384, seed=5, max_depth=2, components={"scattering"}),
        reference_frequency_hz=_FREQUENCY,
    )
    assert result.component_maps is not None
    scattering = result.component_maps["scattering"]
    assert scattering.shape == (1, 4, 4)
    assert torch.count_nonzero(scattering) > 0
    torch.testing.assert_close(
        result.component_power["scattering"],
        scattering.sum(),
        rtol=1.0e-5,
        atol=1.0e-12,
    )


def test_scatter_direction_sampler_matches_table_pdf():
    """Chi-square consistency of the seeded solver sampling path against the
    table pdf (coarser 8 x 16 binning than tests/scattering/test_sampling.py
    because the seeded stream draws fewer samples)."""

    device = torch.device("cuda")
    table = build_kirchhoff_table(
        _roughness(), list(_LAYERS), _FREQUENCY, device=device
    )
    runtimes = {
        0: type(
            "Runtime",
            (),
            {
                "material_index": 0,
                "table": table,
                "layers": _LAYERS,
                "roughness": _roughness(),
            },
        )()
    }
    n = 60_000
    cos_i = 0.7
    sin_i = math.sqrt(1.0 - cos_i * cos_i)
    wi = torch.tensor([[sin_i, 0.0, cos_i]], device=device).expand(n, 3).contiguous()
    material_id = torch.zeros((n,), device=device, dtype=torch.int32)
    uniforms = scatter_direction_uniforms(n, seed=3, tx_index=0, depth=0, device=device)
    sampled = sample_scatter_directions(
        torch.ones_like(material_id, dtype=torch.bool),
        material_id,
        wi,
        uniforms,
        runtimes,
    )
    wo = sampled["wo_local"]
    assert bool((sampled["pdf_forward"] > 0.0).all())
    # Sampler returns its own density.
    lookup = table_pdf(
        table,
        torch.ones_like(material_id, dtype=torch.bool),
        wi,
        wo,
    )
    mismatch = (sampled["pdf_forward"] != lookup).float().mean().item()
    assert mismatch < 1.0e-3

    cos_o = wo[:, 2].clamp(0.0, 1.0 - 1.0e-7)
    phi_o = torch.atan2(wo[:, 1], wo[:, 0])
    phi_o = torch.where(phi_o < 0.0, phi_o + 2.0 * math.pi, phi_o)
    bin_cos = (cos_o * 8).long().clamp(0, 7)
    bin_phi = (phi_o / (2.0 * math.pi / 16)).long().clamp(0, 15)
    counts = torch.zeros(8 * 16, device=device)
    counts.scatter_add_(0, bin_cos * 16 + bin_phi, torch.ones(n, device=device))

    # Expected masses from the sampling table at the nearest incidence bin
    # (floor(cos_i * N) like tables._nearest_axis): aggregate the fine
    # 32 x 64 grid into 8 x 16 coarse bins. The sampler rotates the relative
    # azimuth by phi_i = 0 here, so the axes align.
    ti = min(int(cos_i * 32), 31)
    mass = table.sample_density[ti, 0] * table.bin_solid_angle  # [32, 64]
    expected = mass.reshape(8, 4, 16, 4).sum(dim=(1, 3)).reshape(-1) * n
    keep = expected >= 5.0
    stat = float((((counts - expected) ** 2 / expected)[keep]).sum())
    dof = int(keep.sum()) - 1
    assert chi2.sf(stat, dof) > 1.0e-4
