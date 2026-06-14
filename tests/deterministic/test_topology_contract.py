import pytest
import torch

from tests.support.scenes import empty_space_los_scene, same_side_wall_reflection_scene
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.deterministic import Config
from witwin.channel_native.deterministic.topology import export_topology


def test_topology_export_normalizes_los_columns():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic topology export")

    topology = export_topology(empty_space_los_scene(), Config(max_depth=0, components={"los"}))
    path_count = topology.valid.numel()

    assert path_count == 4
    assert topology.valid.dtype == torch.bool
    assert topology.tx_id.dtype == torch.int32
    assert topology.rx_id.dtype == torch.int32
    assert topology.depth.tolist() == [0] * path_count
    assert topology.component_id.tolist() == [0] * path_count
    assert topology.primitive_id.tolist() == [-1] * path_count
    assert topology.edge_id.tolist() == [-1] * path_count
    assert topology.interaction_position.shape == (path_count, 3)
    assert topology.interaction_normal.shape == (path_count, 3)
    assert topology.material_id.tolist() == [-1] * path_count


def test_topology_applies_max_paths_after_stable_sort():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic topology export")

    full = export_topology(empty_space_los_scene(), Config(max_depth=0, components={"los"}))
    limited = export_topology(empty_space_los_scene(), Config(max_depth=0, components={"los"}, max_paths=2))

    assert limited.valid.numel() == 2
    torch.testing.assert_close(limited.tx_id, full.tx_id[:2])
    torch.testing.assert_close(limited.rx_id, full.rx_id[:2])
    torch.testing.assert_close(limited.path_length_m, full.path_length_m[:2])


def test_reflection_topology_exports_interaction_metadata():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic topology export")
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native reflection is not built")

    topology = export_topology(
        same_side_wall_reflection_scene(),
        Config(components={"reflection"}, coherent=False),
    )

    assert topology.valid.numel() >= 1
    assert torch.all(topology.component_id == 1)
    assert torch.all(topology.primitive_id >= 0)
    assert torch.all(topology.material_id >= 0)
    assert torch.any(topology.interaction_position.abs() > 0.0)
    assert torch.allclose(
        torch.linalg.vector_norm(topology.interaction_normal, dim=1),
        torch.ones_like(topology.path_length_m),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
