import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel_native.deterministic import Config, solve
from witwin.channel_native.path import Config as PathConfig
from witwin.channel_native.path import solve as solve_paths


def test_export_paths_preserves_path_table_columns_and_field_values():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic path export")

    scene = empty_space_los_scene()
    result = solve(scene, Config(max_depth=0, components={"los"}, export_paths=True))
    reference = solve_paths(scene, PathConfig(max_depth=0, components={"los"}))

    assert result.paths is not None
    assert result.paths.valid.shape == reference.valid.shape
    torch.testing.assert_close(result.paths.tx_id, reference.tx_id)
    torch.testing.assert_close(result.paths.rx_id, reference.rx_id)
    torch.testing.assert_close(result.paths.depth, reference.depth)
    torch.testing.assert_close(result.paths.component_id, reference.component_id)
    torch.testing.assert_close(result.paths.primitive_id, reference.primitive_id)
    torch.testing.assert_close(result.paths.edge_id, reference.edge_id)
    torch.testing.assert_close(result.paths.path_length_m, reference.path_length_m)
    torch.testing.assert_close(result.paths.delay_s, reference.delay_s)
    torch.testing.assert_close(result.paths.path_gain, reference.path_gain)
    assert result.paths.interaction_position.shape == (reference.valid.numel(), 3)
    assert result.paths.interaction_normal.shape == (reference.valid.numel(), 3)
    assert result.paths.material_id.shape == reference.valid.shape
    assert result.paths.primitive_sequence.shape == (reference.valid.numel(), 0)
    assert result.paths.material_sequence.shape == (reference.valid.numel(), 0)
    assert result.paths.interaction_positions.shape == (reference.valid.numel(), 0, 3)
    assert result.paths.interaction_normals.shape == (reference.valid.numel(), 0, 3)
    exported_field = torch.complex(result.paths.field_real, result.paths.field_imag)
    torch.testing.assert_close(exported_field.abs().square(), reference.path_gain, rtol=1.0e-5, atol=1.0e-8)
    assert result.paths.phase_rad.dtype == torch.float32
    assert result.paths.interaction_count.tolist() == [0] * reference.valid.numel()
