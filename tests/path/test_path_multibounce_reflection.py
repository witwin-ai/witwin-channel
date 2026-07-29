# Copyright Xingyu Chen.
# Tests path multibounce reflection.

import pytest
import torch

from tests.deterministic.test_reflection_multibounce import (
    parallel_wall_corridor_scene,
    two_wall_multibounce_scene,
)
from tests.support.core_world import (
    make_mesh_structure,
    make_receiver,
    make_transmitter,
)
from witwin.core import PhysicalMaterial, Scene, Structure
from witwin.channel.propagation.topology import evaluated_paths_from_block
from witwin.channel.deployment import build_info
from witwin.channel.deterministic import Config as DeterministicConfig
from witwin.channel.deterministic import solve as solve_deterministic
from witwin.channel.path import Config, solve


def _require_native() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native multi-bounce topology")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")


def _cube(center: tuple[float, float, float], *, surface_id: int) -> Structure:
    cx, cy, cz = center
    vertices = torch.tensor(
        [
            [cx - 0.5, cy - 0.5, cz - 0.5],
            [cx + 0.5, cy - 0.5, cz - 0.5],
            [cx + 0.5, cy + 0.5, cz - 0.5],
            [cx - 0.5, cy + 0.5, cz - 0.5],
            [cx - 0.5, cy - 0.5, cz + 0.5],
            [cx + 0.5, cy - 0.5, cz + 0.5],
            [cx + 0.5, cy + 0.5, cz + 0.5],
            [cx - 0.5, cy + 0.5, cz + 0.5],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=torch.int64,
    )
    return make_mesh_structure(
        vertices=vertices,
        faces=faces,
        material=PhysicalMaterial(eps_r=4.0, sigma_e=0.01),
        name=f"cube-{surface_id}",
        surface_id=surface_id,
    )


def _three_cube_scene() -> Scene:
    return Scene(
        structures=[
            _cube((0.0, 2.0, 1.0), surface_id=31),
            _cube((0.0, -2.0, 1.0), surface_id=32),
            _cube((2.0, 2.0, 1.0), surface_id=33),
        ],
        endpoints=[
            make_transmitter(torch.tensor([-3.0, 0.0, 1.0])),
            make_receiver(torch.tensor([3.0, 0.0, 1.0])),
        ],
    )


def test_path_and_deterministic_share_two_wall_canonical_sequences():
    _require_native()
    scene = two_wall_multibounce_scene()
    config = Config(components={"reflection"}, max_depth=2)

    paths = solve(scene, config, reference_frequency_hz=3.0e9)
    deterministic = solve_deterministic(
        scene,
        DeterministicConfig(components={"reflection"}, max_depth=2, export_paths=True),
        reference_frequency_hz=3.0e9,
    )

    assert deterministic.paths is not None
    torch.testing.assert_close(
        paths.path_length_m[paths.valid],
        deterministic.paths.path_length_m,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        paths.tau[paths.valid], deterministic.paths.delay_s, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        paths.primitive_id[paths.valid], deterministic.paths.primitive_sequence
    )
    torch.testing.assert_close(
        paths.interaction_type[paths.valid],
        torch.tensor(
            [[1, 0], [1, 0], [1, 1]], device=paths.a.device, dtype=torch.int32
        ),
    )


def test_path_and_deterministic_match_three_cube_depth_three_topology():
    _require_native()
    scene = _three_cube_scene()
    path = solve(
        scene,
        Config(components={"reflection"}, max_depth=3),
        reference_frequency_hz=3.0e9,
    )
    deterministic = solve_deterministic(
        scene,
        DeterministicConfig(components={"reflection"}, max_depth=3, export_paths=True),
        reference_frequency_hz=3.0e9,
    )

    assert deterministic.paths is not None
    valid = path.valid
    assert int(valid.sum().item()) > 0
    torch.testing.assert_close(
        path.primitive_id[valid], deterministic.paths.primitive_sequence
    )
    torch.testing.assert_close(
        path.path_length_m[path.valid],
        deterministic.paths.path_length_m,
        rtol=0.0,
        atol=1.0e-5,
    )
    torch.testing.assert_close(
        path.tau[path.valid], deterministic.paths.delay_s, rtol=0.0, atol=1.0e-12
    )
    canonical = torch.cat(
        (path.interaction_type[valid], path.primitive_id[valid]), dim=1
    )
    assert torch.unique(canonical, dim=0).shape[0] == canonical.shape[0]


@pytest.mark.parametrize("max_depth", range(1, 6))
def test_path_reflection_depth_one_through_five_is_effective(max_depth):
    _require_native()
    result = solve(
        parallel_wall_corridor_scene(),
        Config(components={"reflection"}, max_depth=max_depth),
        reference_frequency_hz=3.0e9,
    )

    depth = (result.interaction_type[result.valid] != 0).sum(dim=-1)
    assert set(depth.tolist()) == set(range(1, max_depth + 1))
    assert result.metadata["effective_max_depth"] == max_depth
    assert result.metadata["component_max_depth"]["reflection"] == max_depth


def test_shared_topology_deduplicates_canonical_sequences_and_caps_each_pair():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for canonical topology selection")
    device = torch.device("cuda")
    count = 6
    pair = torch.tensor([0, 0, 0, 1, 1, 1], device=device, dtype=torch.int32)
    sequence = torch.tensor(
        [[3], [3], [4], [3], [3], [4]], device=device, dtype=torch.int32
    )
    block = {
        "valid": torch.ones((count,), device=device, dtype=torch.bool),
        "tx_id": torch.zeros((count,), device=device, dtype=torch.int32),
        "rx_id": pair,
        "depth": torch.ones((count,), device=device, dtype=torch.int32),
        "component_id": torch.ones((count,), device=device, dtype=torch.int32),
        "primitive_id": sequence[:, 0].contiguous(),
        "edge_id": torch.full((count,), -1, device=device, dtype=torch.int32),
        "path_length_m": torch.arange(count, device=device, dtype=torch.float32),
        "delay_s": torch.arange(count, device=device, dtype=torch.float32),
        "path_gain": torch.ones((count,), device=device, dtype=torch.float32),
        "path_field": torch.ones((count,), device=device, dtype=torch.complex64),
        "interaction_position": torch.zeros((count, 3), device=device),
        "interaction_normal": torch.zeros((count, 3), device=device),
        "material_id": torch.zeros((count,), device=device, dtype=torch.int32),
        "primitive_sequence": sequence,
        "material_sequence": torch.zeros((count, 1), device=device, dtype=torch.int32),
        "interaction_positions": torch.zeros((count, 1, 3), device=device),
        "interaction_normals": torch.zeros((count, 1, 3), device=device),
    }

    selected, _ = evaluated_paths_from_block(
        block,
        max_paths=1,
        max_paths_scope="per_pair",
        tx_count=1,
        rx_count=2,
        max_depth=1,
        launch_count=0,
    )

    assert selected.topology.rx_id.tolist() == [0, 1]
    assert selected.topology.primitive_sequence[:, 0].tolist() == [3, 3]
    assert selected.geometry.path_length_m.tolist() == [0.0, 3.0]