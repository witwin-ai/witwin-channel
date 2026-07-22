import pytest
import torch

from witwin.channel import Scene, Structure
from witwin.channel.propagation.geometry.kernels import bridge as ops
from witwin.channel.core.materials import PerfectConductor
from witwin.channel.runtime import symbols


def _native_single_triangle_scene():
    symbols.native_extension()
    scene = Scene(
        structures=[Structure(
            vertices=torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            dtype=torch.float32,
            ),
            faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
            material=PerfectConductor(),
            surface_id=2,
        )],
        transmitters=[],
        receivers=[],
        frequency=3.0e9,
    )
    return scene.rayd_scene()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")
def test_reflection_epc_paths_forward_exports_direct_plane_path_table_fields():
    scene = _native_single_triangle_scene()
    source = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
    receiver = torch.tensor([[-0.5, 0.0, 1.0]], device="cuda", dtype=torch.float32)
    expected_prim_ids = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    direct_plane_points = torch.tensor([[[0.0, 0.0, 0.0]]], device="cuda", dtype=torch.float32)
    direct_plane_normals = torch.tensor([[[0.0, 0.0, 1.0]]], device="cuda", dtype=torch.float32)
    surface_group_id = torch.tensor([0], device="cuda", dtype=torch.int32)
    surface_group_size = torch.tensor([1], device="cuda", dtype=torch.int32)
    surface_group_members = torch.tensor([0], device="cuda", dtype=torch.int32)

    out = ops.rayd_reflection_epc_paths_forward(
        scene,
        source,
        receiver,
        None,
        expected_prim_ids,
        direct_plane_points,
        direct_plane_normals,
        surface_group_id,
        surface_group_size,
        surface_group_members,
        1,
        1,
    )

    valid, path_length, resolved_prim_ids, surface_group_ids, hit_positions, normals = out
    assert bool(valid[0].item())
    torch.testing.assert_close(path_length, torch.tensor([2.0615528], device="cuda"), rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(resolved_prim_ids, expected_prim_ids)
    torch.testing.assert_close(surface_group_ids, torch.tensor([[0]], device="cuda", dtype=torch.int32))
    torch.testing.assert_close(hit_positions[0, 0], torch.tensor([-0.25, 0.0, 0.0], device="cuda"))
    torch.testing.assert_close(normals[0, 0], torch.tensor([0.0, 0.0, 1.0], device="cuda"))
