# Copyright Xingyu Chen.
# MC basic point receivers, multiple and mixed receivers, and seeded launch rays.

"""MC basic point receivers, multiple and mixed receivers, and seeded launch rays."""

import math

import pytest
import torch

from tests.support.core_world import make_receiver, make_receiver_grid
from tests.support.scenes import same_side_wall_reflection_scene
from witwin.core import Scene
from witwin.channel import deterministic
from witwin.channel.montecarlo.basic import Config, solve

_FREQUENCY_HZ = 3.0e9

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _solve(scene: Scene, config: Config):
    return solve(scene, config, reference_frequency_hz=_FREQUENCY_HZ)


def _wall_scene(*receivers) -> Scene:
    base = same_side_wall_reflection_scene()
    tx = next(endpoint for endpoint in base.endpoints if endpoint.role == "tx")
    return Scene(structures=base.structures, endpoints=[tx, *receivers])


def _grid(x: float, shape=(4, 4)):
    return make_receiver_grid(
        origin=torch.tensor([x, -1.0, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=shape,
        spacing=(0.5, 0.5),
    )


def _deterministic_reflection(scene: Scene) -> torch.Tensor:
    result = deterministic.solve(
        scene,
        deterministic.Config(max_depth=1, components={"reflection"}),
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    return result.path_gain.reshape(-1)


def test_point_reflection_agrees_with_deterministic_within_seed_spread() -> None:
    scene = same_side_wall_reflection_scene()
    expected = float(_deterministic_reflection(scene)[0])
    assert expected > 0.0
    estimates = []
    for seed in range(6):
        result = _solve(
            scene,
            Config(
                samples=1 << 22, seed=seed, components={"reflection"}, receiver_radius_m=0.2
            ),
        )
        assert result.path_gain.shape == (1, 1)
        estimates.append(float(result.path_gain[0, 0]))
    values = torch.tensor(estimates, dtype=torch.float64)
    mean = float(values.mean())
    sem = float(values.std(unbiased=True)) / math.sqrt(len(estimates))
    # The seeds give independent estimates (C-3) ...
    assert sem > 0.0
    # ... whose mean agrees with the image-method gain within 4 standard errors plus the
    # second-order sphere-averaging bias (radius / distance)^2 ~ 1e-2.
    assert abs(mean - expected) <= 4.0 * sem + 2.0e-2 * expected, (mean, sem, expected)


def test_same_seed_reproduces_and_seed_changes_the_reflection_estimate() -> None:
    scene = _wall_scene(_grid(1.0))
    config = Config(samples=1 << 16, seed=3, components={"reflection"})
    first = _solve(scene, config).path_gain
    again = _solve(scene, config).path_gain
    other = _solve(scene, Config(samples=1 << 16, seed=4, components={"reflection"})).path_gain
    torch.testing.assert_close(first, again, rtol=1e-6, atol=0.0)
    assert not torch.equal(first, other)


def test_two_grids_concatenate_the_single_grid_estimates() -> None:
    config = Config(samples=1 << 16, seed=5, components={"los", "reflection", "transmission"})
    a, b = _grid(1.0), _grid(1.5, shape=(2, 3))
    both = _solve(_wall_scene(a, b), config)
    assert both.path_gain.shape == (1, 16 + 6)
    assert both.component_maps is not None
    assert both.component_maps["reflection"].shape == (1, 22)
    for grid, columns in ((a, slice(0, 16)), (b, slice(16, 22))):
        single = _solve(_wall_scene(grid), config)
        rows, cols = grid.shape
        for name in ("los", "reflection", "transmission"):
            expected = single.component_maps[name].transpose(1, 2).reshape(1, rows * cols)
            torch.testing.assert_close(
                both.component_maps[name][:, columns], expected, rtol=1e-5, atol=1e-12
            )


def test_mixed_grid_and_points_cover_every_receiver() -> None:
    points = [make_receiver(torch.tensor([0.0, 1.0, 0.5])), make_receiver(torch.tensor([0.5, 1.2, 0.4]))]
    scene = _wall_scene(points[0], _grid(1.0), points[1])
    result = _solve(scene, Config(samples=1 << 18, seed=1, components={"los", "reflection"}))
    assert result.path_gain.shape == (1, 18)
    assert result.component_maps["reflection"].shape == (1, 18)
    point_columns = result.component_maps["reflection"][0, [0, 17]]
    assert bool((point_columns > 0.0).all())
    total = sum(result.component_maps[name] for name in ("los", "reflection"))
    torch.testing.assert_close(result.path_gain, total, rtol=1e-6, atol=0.0)


def test_point_receivers_refuse_diffraction_before_native_work() -> None:
    with pytest.raises(ValueError, match="samples diffraction only on a ReceiverGrid"):
        _solve(same_side_wall_reflection_scene(), Config(samples=256, components={"diffraction"}))


def test_receiver_radius_must_be_positive() -> None:
    with pytest.raises(ValueError, match="receiver_radius_m"):
        Config(receiver_radius_m=0.0)


def test_scattering_samples_default_to_samples() -> None:
    assert Config(samples=512).scattering_sample_count == 512
    assert Config(samples=512, scattering_samples=64).scattering_sample_count == 64
