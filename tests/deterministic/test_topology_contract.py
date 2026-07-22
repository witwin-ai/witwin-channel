import pytest
import torch

from tests.support.scenes import empty_space_los_scene, same_side_wall_reflection_scene
from witwin.channel.core.kernels.extension import build_info
from witwin.channel.deterministic import Config
from witwin.channel.propagation.enumerated.engine import evaluate_enumerated_paths


def test_topology_export_normalizes_los_columns():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic topology export")

    paths, _ = evaluate_enumerated_paths(
        empty_space_los_scene(), Config(max_depth=0, components={"los"})
    )
    topology = paths.topology
    geometry = paths.geometry
    path_count = topology.valid.numel()

    assert path_count == 4
    assert topology.valid.dtype == torch.bool
    assert topology.tx_id.dtype == torch.int32
    assert topology.rx_id.dtype == torch.int32
    assert topology.depth.tolist() == [0] * path_count
    assert topology.component_id.tolist() == [0] * path_count
    assert topology.primitive_id.tolist() == [-1] * path_count
    assert topology.edge_id.tolist() == [-1] * path_count
    assert geometry.interaction_position.shape == (path_count, 3)
    assert geometry.interaction_normal.shape == (path_count, 3)
    assert topology.material_id.tolist() == [-1] * path_count


def test_topology_applies_global_max_paths_after_stable_sort():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic topology export")

    full, _ = evaluate_enumerated_paths(
        empty_space_los_scene(), Config(max_depth=0, components={"los"})
    )
    limited, _ = evaluate_enumerated_paths(
        empty_space_los_scene(), Config(max_depth=0, components={"los"}, max_paths=2)
    )

    assert limited.topology.valid.numel() == 2
    torch.testing.assert_close(limited.topology.tx_id, full.topology.tx_id[:2])
    torch.testing.assert_close(limited.topology.rx_id, full.topology.rx_id[:2])
    torch.testing.assert_close(
        limited.geometry.path_length_m, full.geometry.path_length_m[:2]
    )


def test_topology_supports_explicit_per_pair_max_paths_scope():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic topology export")

    limited, _ = evaluate_enumerated_paths(
        empty_space_los_scene(),
        Config(
            max_depth=0,
            components={"los"},
            max_paths=1,
            max_paths_scope="per_pair",
        ),
    )

    assert limited.topology.valid.numel() == 4


def test_reflection_topology_exports_interaction_metadata():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic topology export")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    paths, _ = evaluate_enumerated_paths(
        same_side_wall_reflection_scene(),
        Config(components={"reflection"}, coherent=False),
    )
    topology = paths.topology
    geometry = paths.geometry

    assert topology.valid.numel() >= 1
    assert torch.all(topology.component_id == 1)
    assert torch.all(topology.primitive_id >= 0)
    assert torch.all(topology.material_id >= 0)
    assert torch.any(geometry.interaction_position.abs() > 0.0)
    assert torch.allclose(
        torch.linalg.vector_norm(geometry.interaction_normal, dim=1),
        torch.ones_like(geometry.path_length_m),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
