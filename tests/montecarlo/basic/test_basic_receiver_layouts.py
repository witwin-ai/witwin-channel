# Copyright Xingyu Chen.
# MC basic point receivers, multiple and mixed receivers, and seeded launch rays.

"""MC basic point receivers, multiple and mixed receivers, and seeded launch rays."""

import math

import pytest
import torch

from tests.support.core_world import make_receiver, make_receiver_grid
from tests.support.scenes import same_side_wall_reflection_scene, wedge_diffraction_scene
from witwin.core import Scene
from witwin.channel import radiomap
from witwin.channel.radiomap import Stochastic, solve

_FREQUENCY_HZ = 3.0e9

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _solve(scene: Scene, config: Stochastic):
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
    result = radiomap.solve(
        scene,
        radiomap.Coherent(max_depth=1, components={"reflection"}),
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
            Stochastic(
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
    config = Stochastic(samples=1 << 16, seed=3, components={"reflection"})
    first = _solve(scene, config).path_gain
    again = _solve(scene, config).path_gain
    other = _solve(scene, Stochastic(samples=1 << 16, seed=4, components={"reflection"})).path_gain
    torch.testing.assert_close(first, again, rtol=1e-6, atol=0.0)
    assert not torch.equal(first, other)


def test_two_grids_concatenate_the_single_grid_estimates() -> None:
    config = Stochastic(samples=1 << 16, seed=5, components={"los", "reflection", "transmission"})
    a, b = _grid(1.0), _grid(1.5, shape=(2, 3))
    both = _solve(_wall_scene(a, b), config)
    assert both.path_gain.shape == (1, 16 + 6)
    assert both.component_maps is not None
    assert both.component_maps["reflection"].shape == (1, 22)
    for grid, columns in ((a, slice(0, 16)), (b, slice(16, 22))):
        single = _solve(_wall_scene(grid), config)
        rows, cols = grid.shape
        assert single.path_gain.shape == (1, rows, cols)
        torch.testing.assert_close(
            both.path_gain[:, columns], single.path_gain.reshape(1, rows * cols), rtol=1e-5,
            atol=1e-12,
        )
        for name in ("los", "reflection", "transmission"):
            expected = single.component_maps[name].reshape(1, rows * cols)
            torch.testing.assert_close(
                both.component_maps[name][:, columns], expected, rtol=1e-5, atol=1e-12
            )


def test_mixed_grid_and_points_cover_every_receiver() -> None:
    points = [make_receiver(torch.tensor([0.0, 1.0, 0.5])), make_receiver(torch.tensor([0.5, 1.2, 0.4]))]
    scene = _wall_scene(points[0], _grid(1.0), points[1])
    result = _solve(scene, Stochastic(samples=1 << 18, seed=1, components={"los", "reflection"}))
    assert result.path_gain.shape == (1, 18)
    assert result.component_maps["reflection"].shape == (1, 18)
    point_columns = result.component_maps["reflection"][0, [0, 17]]
    assert bool((point_columns > 0.0).all())
    total = sum(result.component_maps[name] for name in ("los", "reflection"))
    torch.testing.assert_close(result.path_gain, total, rtol=1e-6, atol=0.0)


def _seed_mean_and_sem(values: list[float]) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(tensor.mean()), float(tensor.std(unbiased=True)) / math.sqrt(len(values))


def test_point_diffraction_agrees_with_the_grid_cell_estimator() -> None:
    # The capture sphere and a small grid cell around the same point estimate the same
    # Keller-cone flux density. (Both differ from witwin.channel.deterministic on this finite
    # wedge by the pre-existing MC diffraction convention, so the grid cell is the oracle.)
    scene = wedge_diffraction_scene()
    tx = next(endpoint for endpoint in scene.endpoints if endpoint.role == "tx")
    centre, h = torch.tensor([3.0, 1.0, 0.5]), 0.1
    x_axis, y_axis = torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])
    grid = make_receiver_grid(
        origin=centre - (x_axis + y_axis) * h, x_axis=x_axis, y_axis=y_axis, shape=(3, 3),
        spacing=(h, h),
    )
    grid_scene = Scene(structures=scene.structures, endpoints=[tx, grid])
    point_values, cell_values = [], []
    for seed in range(6):
        point = _solve(
            scene,
            Stochastic(samples=1 << 20, seed=seed, components={"diffraction"}, receiver_radius_m=0.05),
        )
        assert point.component_maps["diffraction"].shape == (1, 1)
        point_values.append(float(point.path_gain[0, 0]))
        cell = _solve(grid_scene, Stochastic(samples=1 << 20, seed=seed, components={"diffraction"}))
        cell_values.append(float(cell.component_maps["diffraction"][0, 1, 1]))
    point_mean, point_sem = _seed_mean_and_sem(point_values)
    cell_mean, cell_sem = _seed_mean_and_sem(cell_values)
    assert point_mean > 0.0 and point_sem > 0.0
    # Sphere (radius 0.05 m) and cell (0.1 m square) average a field that varies over ~1.4 m.
    tolerance = 4.0 * math.hypot(point_sem, cell_sem) + 0.1 * cell_mean
    assert abs(point_mean - cell_mean) <= tolerance, (point_mean, cell_mean, tolerance)


def test_receiver_radius_must_be_positive() -> None:
    with pytest.raises(ValueError, match="receiver_radius_m"):
        Stochastic(receiver_radius_m=0.0)


def test_scattering_samples_default_to_samples() -> None:
    assert Stochastic(samples=512).scattering_sample_count == 512
    assert Stochastic(samples=512, scattering_samples=64).scattering_sample_count == 64


def test_primal_face_material_tensors_are_reused_until_the_store_changes() -> None:
    from witwin.channel import radiomap
    from witwin.channel.scene import compile as compile_scene
    from witwin.channel.scene.endpoints import bind_solver_scene

    compiled = compile_scene(same_side_wall_reflection_scene(), reference_frequency_hz=_FREQUENCY_HZ)
    scene = bind_solver_scene(compiled)
    device = torch.device("cuda")
    first = radiomap._stochastic._face_material_tensors(scene, device=device, ad=False)
    again = radiomap._stochastic._face_material_tensors(scene, device=device, ad=False)
    assert all(a is b for a, b in zip(first, again))
    with torch.no_grad():
        compiled.materials.eps_r.add_(1.0)
    changed = radiomap._stochastic._face_material_tensors(scene, device=device, ad=False)
    assert changed[0] is not first[0]
    torch.testing.assert_close(changed[0], first[0] + 1.0)


def test_point_diffraction_ad_forward_is_the_primal_and_carries_a_source_gradient() -> None:
    base = torch.tensor([0.0, -1.0, 0.5])
    config = dict(samples=1 << 16, seed=2, components={"diffraction"}, receiver_radius_m=0.1)
    primal = _solve(wedge_diffraction_scene(tx=base.clone()), Stochastic(**config)).path_gain
    leaf = base.clone().cuda().requires_grad_(True)
    result = _solve(wedge_diffraction_scene(tx=leaf), Stochastic(**config, ad_mode="vjp"))
    torch.testing.assert_close(result.path_gain.detach(), primal, rtol=1e-6, atol=0.0)
    result.path_gain.sum().backward()
    assert leaf.grad is not None
    assert bool(torch.isfinite(leaf.grad).all()) and float(leaf.grad.abs().max()) > 0.0
