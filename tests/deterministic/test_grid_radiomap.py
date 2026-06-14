import pytest
import torch

from tests.deterministic.test_component_layout import _grid_scene
from witwin.channel_native.deterministic import Config, solve


def test_grid_radiomap_outputs_total_and_component_maps():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic radiomap")

    result = solve(_grid_scene(), Config(max_depth=0, components={"los"}, diagnostics=True))

    assert result.path_gain.shape == (1, 3, 2)
    assert set(result.component_power) == {"los", "reflection", "diffraction"}
    torch.testing.assert_close(result.path_gain, result.component_power["los"], rtol=1.0e-5, atol=1.0e-8)
    assert result.diagnostics is not None
    assert result.diagnostics["path_gain_shape"] == (1, 3, 2)
