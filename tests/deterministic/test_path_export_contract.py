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
    valid = reference.valid
    path_count = int(valid.sum())

    assert result.paths is not None
    assert result.paths.valid.shape == (path_count,)
    torch.testing.assert_close(result.paths.tx_id, reference.tx_id[valid])
    torch.testing.assert_close(result.paths.rx_id, reference.rx_id[valid])
    assert torch.all(result.paths.depth == 0)
    assert torch.all(result.paths.component_id == 0)
    assert torch.all(result.paths.primitive_id == -1)
    assert torch.all(result.paths.edge_id == -1)
    torch.testing.assert_close(result.paths.path_length_m, reference.path_length_m[valid])
    torch.testing.assert_close(result.paths.delay_s, reference.tau[valid])
    reference_coefficient = reference.a[..., 0][valid]
    torch.testing.assert_close(result.paths.coefficient, reference_coefficient)
    tx_power = torch.tensor(
        [transmitter.power_w for transmitter in scene.transmitters],
        device=reference.a.device,
        dtype=torch.float32,
    )
    reference_gain = (
        reference_coefficient.abs().square()
        * tx_power[result.paths.tx_id.to(torch.int64)]
    )
    torch.testing.assert_close(result.paths.path_gain, reference_gain)
    assert result.paths.interaction_position.shape == (path_count, 3)
    assert result.paths.interaction_normal.shape == (path_count, 3)
    assert result.paths.material_id.shape == (path_count,)
    assert result.paths.primitive_sequence.shape == (path_count, 0)
    assert result.paths.material_sequence.shape == (path_count, 0)
    assert result.paths.interaction_positions.shape == (path_count, 0, 3)
    assert result.paths.interaction_normals.shape == (path_count, 0, 3)
    exported_field = torch.complex(result.paths.field_real, result.paths.field_imag)
    torch.testing.assert_close(
        exported_field.abs().square(), reference_gain, rtol=1.0e-5, atol=1.0e-8
    )
    assert result.paths.phase_rad.dtype == torch.float32
    assert result.paths.interaction_count.tolist() == [0] * path_count
