import pytest
import torch

from tests.support.scenes import empty_space_los_scene, single_wall_reflection_scene
from witwin.channel import ReceiverGrid
from witwin.channel.montecarlo.bdpt import Config, solve


def _reflection_grid() -> ReceiverGrid:
    return ReceiverGrid(
        origin=torch.tensor([0.0, -1.0, 0.0]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(2, 2),
        spacing=(1.0, 0.5),
    )


def test_bdpt_variance_is_reported_when_diagnostics_are_enabled():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT variance")

    result = solve(empty_space_los_scene(), Config(samples=32, components={"los"}, diagnostics=True))

    assert result.variance is not None
    assert result.variance.shape == result.path_gain.shape
    assert torch.isfinite(result.variance).all()
    torch.testing.assert_close(result.variance, torch.zeros_like(result.variance), rtol=0.0, atol=1.0e-20)
    assert torch.max(result.path_gain.square() / 32).item() > 1.0e-20
    assert result.metadata["variance"] is True


def test_bdpt_variance_is_reported_for_scattering_components():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT variance")

    result = solve(
        single_wall_reflection_scene().add(_reflection_grid()),
        Config(samples=32, components={"reflection"}, diagnostics=True),
    )

    assert result.variance is not None
    assert result.variance.shape == result.path_gain.shape
    assert torch.isfinite(result.variance).all()
    positive = result.path_gain > 0.0
    if torch.any(positive):
        assert torch.any(result.variance[positive] > 0.0)
    else:
        torch.testing.assert_close(result.variance, torch.zeros_like(result.variance))
    assert result.metadata["variance"] is True


def test_bdpt_grid_variance_uses_public_non_square_layout():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT variance")

    result = solve(
        single_wall_reflection_scene().add(
            ReceiverGrid(
                origin=torch.tensor([0.0, -1.0, 0.0]),
                x_axis=torch.tensor([0.0, 1.0, 0.0]),
                y_axis=torch.tensor([0.0, 0.0, 1.0]),
                shape=(2, 3),
                spacing=(1.0, 0.5),
            )
        ),
        Config(samples=32, components={"reflection"}, diagnostics=True),
    )

    assert result.variance is not None
    assert result.path_gain.shape == (1, 3, 2)
    assert result.variance.shape == (1, 3, 2)
