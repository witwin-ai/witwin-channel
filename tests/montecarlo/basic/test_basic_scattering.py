"""MC basic Kirchhoff scattering radiomap (wave 3).

The map area-samples rough faces and deposits the unpolarized Kirchhoff
diffuse path gain per cell; acceptance follows the wave-2 style: a direct
torch area-quadrature reference validates the normalization, the smooth
limit and an energy bound guard the physics, and rough materials must be
completely inert when scattering is not requested.
"""

import math

import pytest
import torch

from tests.support.core_world import (
    make_mesh_structure,
    make_receiver,
    make_receiver_grid,
    make_transmitter,
)
from witwin.core import (
    MaterialLayer,
    PhysicalMaterial,
    ReceiverGrid,
    Scene,
    Structure,
    SurfaceRoughness,
)
from witwin.channel.deployment import build_info
from witwin.channel.montecarlo.basic import Config, solve as solve_basic
from witwin.channel.scene.resources import build_kirchhoff_table, eval_bsdf

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA torch is required"
)

_FREQUENCY = 60.0e9
_LIGHT_SPEED = 299_792_458.0
_SIGMA_H = 1.0e-3
_CORR = 0.01
_EPS_R = 4.0
_SIGMA_E = 0.05
_THICKNESS = 0.1
_LAYERS = ((_THICKNESS, _EPS_R, _SIGMA_E, 1.0),)

_TX = torch.tensor([0.0, 0.0, 0.0])
_RX = torch.tensor([0.5, 1.0, 0.3])


def _solve(scene: Scene, config: Config):
    return solve_basic(scene, config, reference_frequency_hz=_FREQUENCY)


def _require_native() -> None:
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native scattering is not built")


def _roughness(sigma_h: float = _SIGMA_H) -> SurfaceRoughness:
    return SurfaceRoughness(
        rms_height_m=sigma_h,
        correlation_length_x_m=_CORR,
        correlation_length_y_m=_CORR,
    )


def _material(roughness: SurfaceRoughness | None) -> PhysicalMaterial:
    return PhysicalMaterial(
        layers=(
            MaterialLayer(
                thickness_m=_THICKNESS,
                eps_r=_EPS_R,
                sigma_e=_SIGMA_E,
            ),
        ),
        roughness_front=roughness,
        name="wall-material",
    )


def _wall(
    material: PhysicalMaterial, *, x: float = 2.5, surface_id: int = 1
) -> Structure:
    return make_mesh_structure(
        vertices=torch.tensor(
            [[x, -4.0, -4.0], [x, 4.0, -4.0], [x, -4.0, 4.0], [x, 4.0, 4.0]]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=material,
        name=f"wall-{surface_id}",
        surface_id=surface_id,
    )


def _single_cell_grid() -> ReceiverGrid:
    return make_receiver_grid(
        origin=_RX,
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(1, 1),
        spacing=(0.5, 0.5),
    )


def _grid_scene(structures, grid: ReceiverGrid | None = None) -> Scene:
    return Scene(
        structures=structures,
        endpoints=[
            make_transmitter(_TX),
            grid if grid is not None else _single_cell_grid(),
        ],
    )


def _quadrature_reference_unpolarized() -> float:
    """Direct area quadrature of the unpolarized scattering gain at _RX."""

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
    to_tx = _TX.to(device)[None] - points
    r1 = to_tx.norm(dim=-1)
    wi = to_tx / r1[:, None]
    to_rx = _RX.to(device)[None] - points
    r2 = to_rx.norm(dim=-1)
    wo = to_rx / r2[:, None]
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
    amplitude_sq = (_LIGHT_SPEED / _FREQUENCY / (4.0 * math.pi)) ** 2
    integrand = (
        0.5
        * (f_te + f_tm)
        * (wi * n).sum(-1).clamp_min(0.0)
        * (wo * n).sum(-1).clamp_min(0.0)
        * amplitude_sq
        / (r1**2 * r2**2)
    )
    return float(integrand.sum() * step * step)


def test_basic_scattering_map_matches_area_quadrature_reference():
    _require_native()
    reference = _quadrature_reference_unpolarized()
    result = _solve(
        _grid_scene([_wall(_material(_roughness()))]),
        Config(samples=65_536, seed=5, components={"scattering"}),
    )
    value = float(result.component_maps["scattering"][0, 0, 0])
    assert value > 0.0
    # Area-sampled estimator over a broad diffuse lobe (smoke measurements:
    # within 0.3-5% of the reference at this budget).
    assert value == pytest.approx(reference, rel=0.15)
    torch.testing.assert_close(
        result.component_power["scattering"],
        result.component_maps["scattering"].sum(),
        rtol=1.0e-5,
        atol=1.0e-12,
    )
    assert result.metadata["components"]["scattering"] == "enabled"
    assert result.metadata["scattering"]["rough_face_count"] == 2
    assert result.metadata["scattering"]["sample_count"] == 65_536


def test_basic_smooth_limit_and_energy_bound():
    """sigma_h -> 0 kills the diffuse map; at strong roughness the diffuse
    map total stays below the SMOOTH reflection map total on the same grid
    (R_diff <= R_bar, near-specular geometry). The reflection map itself is
    unchanged by roughness in v1: the coherent C_r attenuation of the
    specular estimators belongs to the deterministic-solver scattering wave,
    so asserting the literal reflection+scattering sum bound would encode
    the known-missing attenuation as correct; the quadrature-reference test
    above is the exact normalization check.
    """

    _require_native()
    scattering_config = Config(samples=32_768, seed=5, components={"scattering"})
    rough_scattering = float(
        _solve(
            _grid_scene([_wall(_material(_roughness()))]), scattering_config
        ).component_power["scattering"]
    )
    tiny_scattering = float(
        _solve(
            _grid_scene([_wall(_material(_roughness(sigma_h=1.0e-6)))]),
            scattering_config,
        ).component_power["scattering"]
    )
    reflection_config = Config(samples=16_384, seed=5, components={"reflection"})
    smooth_reflection = float(
        _solve(
            _grid_scene([_wall(_material(None))]), reflection_config
        ).component_power["reflection"]
    )
    rough_reflection = float(
        _solve(
            _grid_scene([_wall(_material(_roughness()))]), reflection_config
        ).component_power["reflection"]
    )

    assert rough_scattering > 0.0
    assert smooth_reflection > 0.0
    # Smooth limit: numerically zero diffuse power, reflection unchanged.
    assert tiny_scattering < 1.0e-4 * rough_scattering
    assert rough_reflection == pytest.approx(smooth_reflection, rel=1.0e-6)
    # Energy bound with statistical headroom.
    assert rough_scattering <= smooth_reflection * 1.25


def test_basic_scattering_is_seed_reproducible():
    _require_native()
    scene = _grid_scene([_wall(_material(_roughness()))])
    config = Config(samples=8192, seed=11, components={"scattering"})
    first = _solve(scene, config)
    second = _solve(scene, config)
    torch.testing.assert_close(
        first.component_maps["scattering"],
        second.component_maps["scattering"],
        rtol=0.0,
        atol=0.0,
    )
    other = _solve(scene, Config(samples=8192, seed=12, components={"scattering"}))
    assert not torch.equal(
        first.component_maps["scattering"], other.component_maps["scattering"]
    )


def test_basic_results_unchanged_when_scattering_not_requested():
    """Regression guard: without the scattering component, roughness is
    never read - a rough-material scene reproduces the smooth variant of
    the same scene. Reruns of the SAME scene are bit-identical; across the
    two Scene instances the reflection map is compared at float tolerance
    because its atomic accumulation order varies with the BVH build."""

    _require_native()
    grid = make_receiver_grid(
        origin=torch.tensor([0.5, -1.0, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(4, 4),
        spacing=(2.0 / 3.0, 1.0 / 3.0),
    )
    config = Config(samples=4096, seed=7, max_depth=2, components={"los", "reflection"})
    rough_scene = _grid_scene([_wall(_material(_roughness()))], grid)
    rough_run = _solve(rough_scene, config)
    rough_rerun = _solve(rough_scene, config)
    torch.testing.assert_close(
        rough_run.path_gain, rough_rerun.path_gain, rtol=0.0, atol=0.0
    )
    smooth_run = _solve(_grid_scene([_wall(_material(None))], grid), config)
    torch.testing.assert_close(
        rough_run.path_gain, smooth_run.path_gain, rtol=1.0e-6, atol=1.0e-15
    )
    for component in ("los", "reflection"):
        torch.testing.assert_close(
            rough_run.component_maps[component],
            smooth_run.component_maps[component],
            rtol=1.0e-6,
            atol=1.0e-15,
        )
    assert "scattering" not in rough_run.component_power


def test_basic_scattering_requires_unobstructed_incident_segment():
    """v1 keeps the incident side simple: a blocking wall between the
    transmitter and the rough face truthfully zeroes the map (no
    through-wall incident paths)."""

    _require_native()
    blocker = _wall(
        PhysicalMaterial.perfect_conductor(),
        x=1.5,
        surface_id=2,
    )
    result = _solve(
        _grid_scene([_wall(_material(_roughness())), blocker]),
        Config(samples=8192, seed=5, components={"scattering"}),
    )
    assert float(result.component_maps["scattering"].abs().max()) == 0.0
    assert result.metadata["scattering"]["tx_visible_samples"] == 0
    assert result.metadata["contribution_capacity"] == 1


def test_basic_point_receivers_report_zero_scattering_power():
    _require_native()
    scene = Scene(
        structures=[_wall(_material(_roughness()))],
        endpoints=[make_transmitter(_TX), make_receiver(_RX)],
    )
    result = _solve(
        scene, Config(samples=1024, seed=5, components={"los", "scattering"})
    )
    # MC basic carries scattering on grid maps only; point receivers report
    # a truthful zero (mirrors the transmission point-receiver contract).
    assert float(result.component_power["scattering"]) == 0.0
