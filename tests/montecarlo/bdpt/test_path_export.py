import pytest
import torch

from tests.support.scenes import (
    empty_space_los_scene,
    same_side_wall_reflection_scene,
    single_wall_reflection_scene,
    wedge_diffraction_scene,
)
from witwin.channel_native import ReceiverGrid
from witwin.channel_native.montecarlo.bdpt import BDPTPathSamples, Config, solve


def _reflection_grid() -> ReceiverGrid:
    return ReceiverGrid(
        origin=torch.tensor([5.0, -1.0, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(3, 2),
        spacing=(1.0, 0.5),
    )


def _same_side_reflection_grid() -> ReceiverGrid:
    return ReceiverGrid(
        origin=torch.tensor([0.0, -1.0, 0.0]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(4, 4),
        spacing=(2.0 / 3.0, 1.0 / 3.0),
    )


def test_bdpt_path_export_is_capped_and_schema_stable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT path export")

    result = solve(
        empty_space_los_scene(),
        Config(samples=32, components={"los"}, export_paths=True, max_exported_paths=5),
    )

    assert isinstance(result.path_samples, BDPTPathSamples)
    assert result.path_samples.contribution.shape[0] == 5
    assert result.path_samples.contribution.shape[0] == result.path_samples.valid.shape[0]
    assert result.path_samples.valid.dtype == torch.bool
    assert result.path_samples.topology.shape[0] == result.path_samples.contribution.shape[0]
    assert result.path_samples.mis_weight.shape == result.path_samples.contribution.shape
    assert result.path_samples.contribution.sum() < result.path_gain.sum()
    torch.testing.assert_close(
        result.path_samples.component_id,
        torch.zeros_like(result.path_samples.component_id),
    )
    torch.testing.assert_close(
        result.path_samples.light_depth,
        torch.zeros_like(result.path_samples.light_depth),
    )
    torch.testing.assert_close(
        result.path_samples.sensor_depth,
        torch.zeros_like(result.path_samples.sensor_depth),
    )
    torch.testing.assert_close(result.path_samples.grid_linear_id, result.path_samples.rx_id)
    assert torch.all(result.path_samples.pdf[result.path_samples.valid] > 0.0)
    assert not torch.allclose(result.path_samples.pdf, result.path_samples.valid.to(torch.float32))
    torch.testing.assert_close(
        result.path_samples.mis_weight,
        result.path_samples.valid.to(torch.float32),
    )
    expected_length = torch.tensor(
        [5.0, 10.0, 5.0, 10.0, 5.0],
        device=result.path_samples.path_length_m.device,
        dtype=torch.float32,
    )
    torch.testing.assert_close(result.path_samples.path_length_m, expected_length, rtol=1e-6, atol=1e-6)


def test_bdpt_path_export_includes_scattering_component_samples():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT path export")

    result = solve(
        same_side_wall_reflection_scene().add(_same_side_reflection_grid()),
        Config(samples=64, seed=3, components={"los", "reflection"}, export_paths=True),
    )

    assert isinstance(result.path_samples, BDPTPathSamples)
    assert result.path_samples.contribution.shape[0] > 0
    assert result.path_samples.component_id.shape == result.path_samples.contribution.shape
    assert result.path_samples.light_depth.shape == result.path_samples.contribution.shape
    assert result.path_samples.sensor_depth.shape == result.path_samples.contribution.shape
    assert torch.any(result.path_samples.component_id == 1)
    assert torch.any(result.path_samples.light_depth == 1)
    assert torch.all(result.path_samples.sensor_depth == 0)
    assert torch.all(result.path_samples.grid_linear_id >= 0)
    valid_float = result.path_samples.valid.to(torch.float32)
    assert torch.all(result.path_samples.pdf[result.path_samples.valid] > 0.0)
    assert not torch.allclose(result.path_samples.pdf, valid_float)
    torch.testing.assert_close(result.path_samples.mis_weight, valid_float)


def test_bdpt_reflection_path_export_uses_seeded_native_subpath_samples():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT reflection path export")

    scene = same_side_wall_reflection_scene().add(_same_side_reflection_grid())
    first = solve(scene, Config(samples=16, seed=11, components={"reflection"}, export_paths=True))
    second = solve(scene, Config(samples=16, seed=11, components={"reflection"}, export_paths=True))
    changed = solve(scene, Config(samples=16, seed=12, components={"reflection"}, export_paths=True))

    assert isinstance(first.path_samples, BDPTPathSamples)
    assert isinstance(second.path_samples, BDPTPathSamples)
    assert isinstance(changed.path_samples, BDPTPathSamples)
    assert first.path_samples.contribution.shape[0] > 0
    assert first.path_samples.contribution.shape[0] == second.path_samples.contribution.shape[0]
    torch.testing.assert_close(first.path_samples.contribution, second.path_samples.contribution)
    torch.testing.assert_close(first.path_samples.rx_id, second.path_samples.rx_id)
    assert changed.path_samples.contribution.shape[0] > 0
    if changed.path_samples.contribution.shape == first.path_samples.contribution.shape:
        assert not torch.equal(first.path_samples.contribution, changed.path_samples.contribution)
    torch.testing.assert_close(first.path_samples.component_id, torch.ones_like(first.path_samples.component_id))
    torch.testing.assert_close(first.path_samples.light_depth, torch.ones_like(first.path_samples.light_depth))


def test_bdpt_diffraction_path_export_is_seeded_by_native_direct_keller_tape():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT diffraction path export")

    scene = wedge_diffraction_scene().add(_reflection_grid())
    first = solve(scene, Config(samples=16, seed=17, components={"diffraction"}, export_paths=True))
    second = solve(scene, Config(samples=16, seed=17, components={"diffraction"}, export_paths=True))
    changed = solve(scene, Config(samples=16, seed=18, components={"diffraction"}, export_paths=True))

    assert isinstance(first.path_samples, BDPTPathSamples)
    assert isinstance(second.path_samples, BDPTPathSamples)
    assert isinstance(changed.path_samples, BDPTPathSamples)
    assert first.path_samples.contribution.shape[0] > 0
    torch.testing.assert_close(first.path_samples.contribution, second.path_samples.contribution)
    torch.testing.assert_close(first.path_samples.rx_id, second.path_samples.rx_id)
    assert changed.path_samples.contribution.shape[0] > 0
    if changed.path_samples.contribution.shape == first.path_samples.contribution.shape:
        assert not torch.equal(first.path_samples.contribution, changed.path_samples.contribution)
    torch.testing.assert_close(
        first.path_samples.component_id,
        torch.full_like(first.path_samples.component_id, 2),
    )
    torch.testing.assert_close(first.path_samples.light_depth, torch.ones_like(first.path_samples.light_depth))
    torch.testing.assert_close(first.path_samples.sensor_depth, torch.zeros_like(first.path_samples.sensor_depth))
    torch.testing.assert_close(first.path_samples.grid_linear_id, first.path_samples.rx_id)
    assert torch.all(first.path_samples.pdf[first.path_samples.valid] > 0.0)
    assert torch.all(first.path_samples.path_length_m[first.path_samples.valid] > 0.0)


def test_bdpt_path_export_omits_blocked_reflection_candidates():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT path export")

    result = solve(
        single_wall_reflection_scene().add(_reflection_grid()),
        Config(samples=64, seed=3, components={"reflection"}, export_paths=True),
    )

    assert isinstance(result.path_samples, BDPTPathSamples)
    assert result.path_samples.contribution.shape[0] == 0
    torch.testing.assert_close(result.path_gain, torch.zeros_like(result.path_gain))
