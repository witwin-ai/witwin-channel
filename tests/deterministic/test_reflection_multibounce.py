import pytest
import torch

from tests.support.core_world import (
    make_mesh_structure,
    make_receiver,
    make_transmitter,
)
from witwin.channel.core.kernels.extension import build_info
from witwin.channel.deterministic import Config, solve
from witwin.channel.propagation.fields.kernels import (
    deterministic as deterministic_fields,
)
from witwin.channel.propagation.enumerated import reflection as topology
from witwin.channel.propagation.geometry import reevaluate as topology_geometry
from witwin.channel.propagation.topology.export import evaluated_paths_from_block
from witwin.channel.path import Config as PathConfig
from witwin.channel.path import solve as solve_paths
from witwin.core import PhysicalMaterial, Scene

_REFERENCE_FREQUENCY_HZ = 3.0e9


def test_multibounce_sort_order_uses_full_primitive_sequence():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native deterministic topology sorting")

    device = torch.device("cuda")
    primitive_sequence = torch.tensor(
        [[4, 2], [4, 1], [3, 9]], device=device, dtype=torch.int32
    )
    count = int(primitive_sequence.shape[0])
    block = {
        "valid": torch.ones((count,), device=device, dtype=torch.bool),
        "tx_id": torch.zeros((count,), device=device, dtype=torch.int32),
        "rx_id": torch.zeros((count,), device=device, dtype=torch.int32),
        "depth": torch.full((count,), 2, device=device, dtype=torch.int32),
        "component_id": torch.ones((count,), device=device, dtype=torch.int32),
        "primitive_id": torch.full((count,), 4, device=device, dtype=torch.int32),
        "edge_id": torch.full((count,), -1, device=device, dtype=torch.int32),
        "path_length_m": torch.tensor(
            [42.0, 41.0, 39.0], device=device, dtype=torch.float32
        ),
        "delay_s": torch.zeros((count,), device=device, dtype=torch.float32),
        "path_gain": torch.ones((count,), device=device, dtype=torch.float32),
        "path_field": torch.ones((count,), device=device, dtype=torch.complex64),
        "interaction_position": torch.zeros(
            (count, 3), device=device, dtype=torch.float32
        ),
        "interaction_normal": torch.zeros(
            (count, 3), device=device, dtype=torch.float32
        ),
        "material_id": torch.zeros((count,), device=device, dtype=torch.int32),
        "primitive_sequence": primitive_sequence,
        "material_sequence": torch.zeros((count, 2), device=device, dtype=torch.int32),
        "interaction_positions": torch.zeros(
            (count, 2, 3), device=device, dtype=torch.float32
        ),
        "interaction_normals": torch.zeros(
            (count, 2, 3), device=device, dtype=torch.float32
        ),
    }

    sorted_paths, _ = evaluated_paths_from_block(
        block,
        max_paths=None,
        max_paths_scope="global",
        tx_count=1,
        rx_count=1,
        max_depth=2,
        launch_count=0,
    )

    torch.testing.assert_close(
        sorted_paths.topology.primitive_sequence,
        torch.tensor([[3, 9], [4, 1], [4, 2]], device=device, dtype=torch.int32),
    )
    torch.testing.assert_close(
        sorted_paths.geometry.path_length_m,
        torch.tensor([39.0, 41.0, 42.0], device=device, dtype=torch.float32),
    )


def test_multibounce_grouping_splits_non_coplanar_faces_with_same_surface_id():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native deterministic face grouping")

    device = torch.device("cuda")
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [[0, 1, 2], [1, 3, 2], [4, 5, 6]], device=device, dtype=torch.long
    )
    tri_a = points[faces[:, 0]]
    normals = torch.nn.functional.normalize(
        torch.cross(points[faces[:, 1]] - tri_a, points[faces[:, 2]] - tri_a, dim=1),
        dim=1,
    )
    same_surface = torch.full((faces.shape[0],), 7, device=device, dtype=torch.long)

    groups = topology_geometry._coplanar_face_groups(tri_a, normals, same_surface)

    assert int(groups["group_count"]) == 2
    torch.testing.assert_close(
        groups["face_group_id"],
        torch.tensor([0, 0, 1], device=device, dtype=torch.int32),
    )
    torch.testing.assert_close(
        groups["representative_faces"],
        torch.tensor([0, 2], device=device, dtype=torch.long),
    )


def two_wall_multibounce_scene() -> Scene:
    wall_x = make_mesh_structure(
        vertices=torch.tensor(
            [
                [2.0, -1.0, 0.0],
                [2.0, 3.0, 0.0],
                [2.0, -1.0, 2.0],
                [2.0, 3.0, 2.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=PhysicalMaterial(eps_r=3.0, sigma_e=0.005),
        name="wall-x",
        surface_id=10,
    )
    wall_y = make_mesh_structure(
        vertices=torch.tensor(
            [
                [0.0, 2.0, 0.0],
                [3.0, 2.0, 0.0],
                [0.0, 2.0, 2.0],
                [3.0, 2.0, 2.0],
            ]
        ),
        faces=torch.tensor([[0, 2, 1], [1, 2, 3]]),
        material=PhysicalMaterial(eps_r=4.0, sigma_e=0.01),
        name="wall-y",
        surface_id=11,
    )
    return Scene(
        structures=[wall_x, wall_y],
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, 0.0, 1.0])),
            make_receiver(position=torch.tensor([0.0, 1.0, 1.0])),
        ],
    )


def test_two_bounce_reflection_exports_depth_two_fields():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic multi-bounce reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    scene = two_wall_multibounce_scene()
    result = solve(
        scene,
        Config(
            components={"reflection"}, max_depth=2, coherent=True, export_paths=True
        ),
        reference_frequency_hz=3.0e9,
    )
    reference = solve_paths(
        scene,
        PathConfig(components={"reflection"}, max_depth=2),
        reference_frequency_hz=3.0e9,
    )

    assert result.paths is not None
    # One specular path per coplanar wall: the historical expectation carried
    # a twin path per wall triangle (D-1 double counting, +6 dB coherent).
    torch.testing.assert_close(
        result.paths.depth,
        torch.tensor([1, 1, 2], device=result.paths.depth.device, dtype=torch.int32),
    )
    torch.testing.assert_close(
        result.paths.primitive_sequence,
        torch.tensor(
            [[0, -1], [2, -1], [1, 2]],
            device=result.paths.primitive_sequence.device,
            dtype=torch.int32,
        ),
    )
    expected_length_all = torch.tensor(
        [4.123105526, 3.0, 5.0],
        device=result.paths.path_length_m.device,
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        result.paths.path_length_m, expected_length_all, rtol=1.0e-5, atol=1.0e-6
    )
    expected_field = reference.a[..., 0][reference.valid]
    torch.testing.assert_close(
        result.paths.coefficient, expected_field, rtol=5.0e-4, atol=1.0e-7
    )
    torch.testing.assert_close(
        result.paths.path_gain,
        expected_field.abs().square(),
        rtol=5.0e-4,
        atol=1.0e-10,
    )
    depth_two = result.paths.depth == 2
    assert bool(depth_two.any())
    assert result.paths.primitive_sequence.shape[1] >= 2
    assert result.paths.material_sequence.shape == result.paths.primitive_sequence.shape
    assert (
        result.paths.interaction_positions.shape[:2]
        == result.paths.primitive_sequence.shape
    )
    assert (
        result.paths.interaction_normals.shape
        == result.paths.interaction_positions.shape
    )
    assert torch.all(result.paths.primitive_sequence[depth_two, :2] >= 0)
    assert torch.all(result.paths.material_sequence[depth_two, :2] >= 0)
    assert torch.all(result.paths.interaction_positions[depth_two, :2].isfinite())
    assert torch.all(result.paths.interaction_normals[depth_two, :2].norm(dim=-1) > 0.0)
    path_field = torch.complex(
        result.paths.field_real[depth_two], result.paths.field_imag[depth_two]
    )
    torch.testing.assert_close(
        result.paths.path_gain[depth_two],
        path_field.abs().square(),
        rtol=2.0e-4,
        atol=1.0e-10,
    )
    torch.testing.assert_close(
        result.path_gain, result.field.abs().square(), rtol=2.0e-4, atol=1.0e-10
    )


def parallel_wall_corridor_scene() -> Scene:
    wall_left = make_mesh_structure(
        vertices=torch.tensor(
            [
                [-2.0, -3.0, 0.0],
                [-2.0, 3.0, 0.0],
                [-2.0, -3.0, 2.0],
                [-2.0, 3.0, 2.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=PhysicalMaterial(eps_r=3.0, sigma_e=0.005),
        name="corridor-left",
        surface_id=20,
    )
    wall_right = make_mesh_structure(
        vertices=torch.tensor(
            [
                [2.0, -3.0, 0.0],
                [2.0, 3.0, 0.0],
                [2.0, -3.0, 2.0],
                [2.0, 3.0, 2.0],
            ]
        ),
        faces=torch.tensor([[0, 2, 1], [1, 2, 3]]),
        material=PhysicalMaterial(eps_r=4.0, sigma_e=0.01),
        name="corridor-right",
        surface_id=21,
    )
    return Scene(
        structures=[wall_left, wall_right],
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, 0.0, 1.0])),
            make_receiver(position=torch.tensor([0.5, 0.5, 1.0])),
        ],
    )


def test_three_bounce_reflection_mixes_depth_two_and_three_blocks():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic multi-bounce reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    result = solve(
        parallel_wall_corridor_scene(),
        Config(
            components={"reflection"}, max_depth=3, coherent=True, export_paths=True
        ),
        reference_frequency_hz=3.0e9,
    )

    assert result.paths is not None
    depths = result.paths.depth
    assert bool((depths == 2).any())
    assert bool((depths == 3).any())
    assert result.paths.primitive_sequence.shape[1] == 3
    assert result.paths.material_sequence.shape == result.paths.primitive_sequence.shape
    assert (
        result.paths.interaction_positions.shape[:2]
        == result.paths.primitive_sequence.shape
    )
    depth_two = depths == 2
    depth_three = depths == 3
    # Padded tail entries stay -1; active entries carry real primitive ids.
    assert torch.all(result.paths.primitive_sequence[depth_two, :2] >= 0)
    assert torch.all(result.paths.primitive_sequence[depth_two, 2] < 0)
    assert torch.all(result.paths.primitive_sequence[depth_three] >= 0)
    assert torch.isfinite(result.paths.path_gain).all()
    assert torch.all(result.paths.path_gain > 0.0)
    # Corridor double/triple bounces alternate between the two walls, so the
    # unfolded path lengths must strictly grow with depth.
    assert float(result.paths.path_length_m[depth_three].min()) > float(
        result.paths.path_length_m[depth_two].max()
    )


def test_two_bounce_reflection_uses_native_sequence_field(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic multi-bounce reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    original = deterministic_fields.deterministic_reflection_sequence_field
    calls = 0

    def count_native_sequence_field(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(
        deterministic_fields,
        "deterministic_reflection_sequence_field",
        count_native_sequence_field,
    )
    result = solve(
        two_wall_multibounce_scene(),
        Config(
            components={"reflection"}, max_depth=2, coherent=True, export_paths=True
        ),
        reference_frequency_hz=3.0e9,
    )

    assert result.paths is not None
    assert bool((result.paths.depth == 2).any())
    assert calls > 0


def test_two_bounce_reflection_does_not_use_python_product():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic multi-bounce reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    assert not hasattr(topology, "product")
    result = solve(
        two_wall_multibounce_scene(),
        Config(
            components={"reflection"}, max_depth=2, coherent=True, export_paths=True
        ),
        reference_frequency_hz=3.0e9,
    )

    assert result.paths is not None
    assert bool((result.paths.depth == 2).any())


def test_two_bounce_reflection_uses_rayd_epc_path_export(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic multi-bounce reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    from witwin.channel.propagation.geometry.kernels import (
        bridge as geometry_bridge,
    )

    original = geometry_bridge.rayd_reflection_epc_paths_forward
    calls = {"count": 0}

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(geometry_bridge, "rayd_reflection_epc_paths_forward", counted)
    result = solve(
        two_wall_multibounce_scene(),
        Config(
            components={"reflection"}, max_depth=2, coherent=True, export_paths=True
        ),
        reference_frequency_hz=3.0e9,
    )

    assert result.paths is not None
    assert bool((result.paths.depth == 2).any())
    assert calls["count"] >= 1


def test_two_bounce_reflection_respects_max_paths_before_candidate_guardrail():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic multi-bounce reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    face_count = 400
    base = torch.arange(face_count, dtype=torch.float32)
    vertices = torch.stack(
        (
            torch.stack((base, torch.zeros_like(base), torch.zeros_like(base)), dim=1),
            torch.stack(
                (base + 0.25, torch.zeros_like(base), torch.zeros_like(base)), dim=1
            ),
            torch.stack(
                (base, torch.full_like(base, 0.25), torch.zeros_like(base)), dim=1
            ),
        ),
        dim=1,
    ).reshape(-1, 3)
    faces = torch.arange(face_count * 3, dtype=torch.int64).reshape(face_count, 3)
    scene = Scene(
        structures=[
            make_mesh_structure(
                vertices=vertices,
                faces=faces,
                material=PhysicalMaterial(eps_r=3.0, sigma_e=0.005),
                name="many-faces",
                surface_id=12,
            )
        ],
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, 0.0, 1.0])),
            make_receiver(position=torch.tensor([1.0, 1.0, 1.0])),
        ],
    )

    result = solve(
        scene,
        Config(
            components={"reflection"},
            max_depth=2,
            coherent=True,
            export_paths=True,
            max_paths=1,
            diagnostics=True,
        ),
        reference_frequency_hz=3.0e9,
    )

    assert result.paths is not None
    assert result.paths.valid.numel() <= 1
    assert result.diagnostics is not None
    assert (
        result.diagnostics["path_planning"]["candidate_count"] < face_count * face_count
    )


def test_two_bounce_reflection_plans_by_surface_groups_before_face_guardrail():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for deterministic multi-bounce reflection")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")

    face_count = 400
    base = torch.arange(face_count, dtype=torch.float32)
    vertices = torch.stack(
        (
            torch.stack((base, torch.zeros_like(base), torch.zeros_like(base)), dim=1),
            torch.stack(
                (base + 0.25, torch.zeros_like(base), torch.zeros_like(base)), dim=1
            ),
            torch.stack(
                (base, torch.full_like(base, 0.25), torch.zeros_like(base)), dim=1
            ),
        ),
        dim=1,
    ).reshape(-1, 3)
    faces = torch.arange(face_count * 3, dtype=torch.int64).reshape(face_count, 3)
    scene = Scene(
        structures=[
            make_mesh_structure(
                vertices=vertices,
                faces=faces,
                material=PhysicalMaterial(eps_r=3.0, sigma_e=0.005),
                name="many-faces",
                surface_id=12,
            )
        ],
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, 0.0, 1.0])),
            make_receiver(position=torch.tensor([1.0, 1.0, 1.0])),
        ],
    )

    result = solve(
        scene,
        Config(
            components={"reflection"}, max_depth=2, coherent=True, export_paths=True
        ),
        reference_frequency_hz=3.0e9,
    )

    assert result.paths is not None
    assert result.paths.valid.numel() < face_count * face_count
