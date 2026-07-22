import pytest
import torch

from tests.support.scenes import empty_space_los_scene
from witwin.channel.path import Config, solve


def test_path_result_exports_topology_and_field_tensors():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for path solver")

    result = solve(empty_space_los_scene(), Config(components={"los"}))
    for tensor in (
        result.valid,
        result.tx_id,
        result.rx_id,
        result.primitive_id,
        result.path_length_m,
        result.tau,
        result.a,
        result.field_xyz,
        result.interaction_type,
        result.material_id,
        result.position,
        result.normal,
    ):
        assert tensor.is_cuda
        assert tensor.is_contiguous()

    assert result.valid.dtype == torch.bool
    assert result.tx_id.dtype == torch.int32
    assert result.a.dtype == torch.complex64
    assert result.field_xyz.dtype == torch.complex64
    assert result.tx_id.shape == result.valid.shape
    assert result.rx_id.shape == result.valid.shape
    assert result.path_length_m.shape == result.valid.shape
