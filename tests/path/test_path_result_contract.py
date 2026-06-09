import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel_native.path import Config, solve


def test_path_result_exports_topology_and_field_tensors():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path solver")

    result = solve(empty_space_los_scene(), Config(components={"los"}))
    path_count = result.valid.numel()

    for tensor in (
        result.valid,
        result.tx_id,
        result.rx_id,
        result.depth,
        result.component_id,
        result.primitive_id,
        result.edge_id,
        result.path_length_m,
        result.delay_s,
        result.path_gain,
    ):
        assert tensor.is_cuda
        assert tensor.shape == (path_count,)
        assert tensor.is_contiguous()

    assert result.valid.dtype == torch.bool
    assert result.tx_id.dtype == torch.int32
    assert result.primitive_id.tolist() == [-1] * path_count
    assert result.edge_id.tolist() == [-1] * path_count
    assert result.path_gain.dtype == torch.float32
