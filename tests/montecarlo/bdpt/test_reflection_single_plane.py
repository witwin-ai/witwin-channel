# Copyright Xingyu Chen.
# Tests reflection single plane.

import pytest
import torch

from tests.support.scenes import same_side_wall_reflection_scene
from witwin.core import ReceiverGrid, Scene
from tests.support.core_world import make_receiver_grid, make_transmitter
from witwin.channel.deployment import build_info
from witwin.channel.montecarlo.bdpt import Config, solve
from witwin.channel.montecarlo import bdpt as bdpt_solver
from witwin.channel.path import Config as PathConfig
from witwin.channel.path import solve as solve_paths


def _scene_with_tx_power(power_w: float) -> Scene:
    base = same_side_wall_reflection_scene()
    return Scene(
        structures=base.structures,
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, -1.0, 0.5]), power_w=power_w),
            *(endpoint for endpoint in base.endpoints if endpoint.role == "rx"),
        ],
    )


def _grid() -> ReceiverGrid:
    return make_receiver_grid(
        origin=torch.tensor([0.0, -1.0, 0.0]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(4, 4),
        spacing=(2.0 / 3.0, 1.0 / 3.0),
    )


def _with_grid(scene: Scene, grid: ReceiverGrid | None = None) -> Scene:
    return scene.with_endpoints(
        (
            *tuple(endpoint for endpoint in scene.endpoints if endpoint.role == "tx"),
            _grid() if grid is None else grid,
        )
    )


def test_bdpt_single_plane_reflection_returns_nonzero_native_component_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    result = solve(
        _with_grid(same_side_wall_reflection_scene()),
        Config(samples=2048, seed=5, components={"reflection"}),
        reference_frequency_hz=3.0e9,
    )

    assert result.component_maps is not None
    reflection = result.component_maps["reflection"]
    assert reflection.shape == (1, 4, 4)
    assert torch.isfinite(reflection).all()
    assert torch.count_nonzero(reflection > 0.0).item() > 0
    assert result.component_power["reflection"].item() > 0.0
    torch.testing.assert_close(
        result.component_power["reflection"], reflection.sum(), rtol=1e-5, atol=1e-8
    )
    assert result.metadata["components"]["reflection"] == "enabled"


def test_bdpt_point_receiver_reflection_returns_native_component_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT point reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    result = solve(
        same_side_wall_reflection_scene(),
        Config(
            samples=2048,
            seed=5,
            components={"reflection"},
            receiver_strategy="point_sphere",
        ),
        reference_frequency_hz=3.0e9,
    )

    assert result.path_gain.shape == (1, 1)
    assert result.component_maps is None
    assert torch.isfinite(result.path_gain).all()
    assert result.component_power["reflection"].item() > 0.0
    torch.testing.assert_close(
        result.path_gain.sum(),
        result.component_power["reflection"],
        rtol=1e-5,
        atol=1e-8,
    )
    assert result.metadata["components"]["reflection"] == "enabled"


def test_bdpt_point_reflection_solver_does_not_use_image_source_path_export():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    assert not hasattr(bdpt_solver, "reflection_paths_order1")

    solve(
        same_side_wall_reflection_scene(),
        Config(
            samples=64,
            seed=5,
            components={"reflection"},
            receiver_strategy="point_sphere",
        ),
        reference_frequency_hz=3.0e9,
    )


def test_bdpt_grid_reflection_solver_does_not_use_image_source_path_export():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    assert not hasattr(bdpt_solver, "reflection_paths_order1")

    result = solve(
        _with_grid(same_side_wall_reflection_scene()),
        Config(samples=512, seed=5, components={"reflection"}),
        reference_frequency_hz=3.0e9,
    )

    assert result.component_power["reflection"].item() > 0.0


@pytest.mark.parametrize(
    ("samples", "relative_tolerance"),
    [
        (4096, 0.20),
        (16384, 0.10),
    ],
)
def test_bdpt_single_plane_reflection_converges_to_maintained_reference(
    samples, relative_tolerance
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT reflection convergence")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    base = same_side_wall_reflection_scene()
    grid = _grid()
    scene = _with_grid(base, grid)
    result = solve(
        scene,
        Config(samples=samples, seed=5, components={"reflection"}),
        reference_frequency_hz=3.0e9,
    )

    observed = result.component_power["reflection"].detach().cpu()
    paths = solve_paths(
        scene, PathConfig(components={"reflection"}), reference_frequency_hz=3.0e9
    )
    reference = paths.a[..., 0].abs().square()[paths.valid].sum().cpu()
    torch.testing.assert_close(observed, reference, rtol=1.0e-5, atol=1.0e-12)


def test_bdpt_point_delta_reflection_uses_unfolded_distance_and_fresnel_bound():
    """Guards audit MC-1: contributions must attenuate over the unfolded path
    (tx -> surface -> rx), bounded by the |R| <= 1 free-space gain."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    scene = same_side_wall_reflection_scene()
    samples = 2048
    result = solve(
        scene,
        Config(
            samples=samples,
            seed=5,
            components={"reflection"},
            receiver_strategy="point_sphere",
            export_paths=True,
        ),
        reference_frequency_hz=3.0e9,
    )

    assert result.path_samples is not None
    valid = result.path_samples.valid
    assert bool(valid.any())
    lengths = result.path_samples.path_length_m[valid]
    contributions = result.path_samples.contribution[valid]
    tx = next(
        endpoint.position for endpoint in scene.endpoints if endpoint.role == "tx"
    )
    rx = next(
        endpoint.position for endpoint in scene.endpoints if endpoint.role == "rx"
    )
    direct = float((tx - rx).norm())
    # Wall is at x=2.5 with tx/rx at x=0: every reflected connection travels
    # at least 2x the wall distance plus cannot be shorter than the LoS.
    assert float(lengths.min()) > direct
    assert float(lengths.min()) > 2.0 * 2.5
    wavelength = 299_792_458.0 / 3.0e9
    free_space_bound = (wavelength / (4.0 * torch.pi * lengths)) ** 2
    assert bool((contributions <= free_space_bound * 1.0001).all())
    assert bool((contributions > 0.0).all())


def test_bdpt_grid_reflection_map_does_not_change_when_export_paths_enabled():
    """Guards audit MC-1d: toggling export_paths must not switch the estimator."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    scene = _with_grid(same_side_wall_reflection_scene())
    plain = solve(
        scene,
        Config(samples=1024, seed=5, components={"reflection"}),
        reference_frequency_hz=3.0e9,
    )
    exported = solve(
        scene,
        Config(samples=1024, seed=5, components={"reflection"}, export_paths=True),
        reference_frequency_hz=3.0e9,
    )

    torch.testing.assert_close(
        exported.component_maps["reflection"],
        plain.component_maps["reflection"],
    )


def test_bdpt_grid_reflection_map_scales_linearly_with_tx_power():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    config = Config(samples=2048, seed=5, components={"reflection"})
    unit = solve(
        _with_grid(_scene_with_tx_power(1.0)),
        config,
        reference_frequency_hz=3.0e9,
    )
    scaled = solve(
        _with_grid(_scene_with_tx_power(2.0)),
        config,
        reference_frequency_hz=3.0e9,
    )

    torch.testing.assert_close(
        scaled.component_maps["reflection"],
        2.0 * unit.component_maps["reflection"],
        rtol=1.0e-5,
        atol=1.0e-12,
    )


def test_bdpt_point_and_single_cell_grid_reflection_are_identical():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    point_scene = same_side_wall_reflection_scene()
    point = solve(
        point_scene,
        Config(
            samples=2048,
            seed=29,
            components={"reflection"},
            receiver_strategy="point_sphere",
        ),
        reference_frequency_hz=3.0e9,
    )
    receiver_position = next(
        endpoint.position for endpoint in point_scene.endpoints if endpoint.role == "rx"
    )
    grid_scene = _with_grid(
        point_scene,
        make_receiver_grid(
            origin=receiver_position,
            x_axis=torch.tensor([0.0, 1.0, 0.0]),
            y_axis=torch.tensor([0.0, 0.0, 1.0]),
            shape=(1, 1),
            spacing=(1.0, 1.0),
        ),
    )
    grid = solve(
        grid_scene,
        Config(samples=2048, seed=29, components={"reflection"}),
        reference_frequency_hz=3.0e9,
    )

    torch.testing.assert_close(
        grid.path_gain.reshape(1, 1), point.path_gain, rtol=1.0e-6, atol=1.0e-12
    )