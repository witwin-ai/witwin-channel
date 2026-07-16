import pytest
import torch

from witwin.channel_native import Scene, Structure
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.core.materials import PerfectConductor
from witwin.channel_native.runtime import symbols


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
    return scene.raydn_scene()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")
def test_reflection_accumulation_accepts_material_payload_and_solid_angle():
    scene = _native_single_triangle_scene()
    ray_o = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)
    ray_d = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
    ray_tmax = torch.tensor([2.0], device="cuda", dtype=torch.float32)
    active = torch.ones((1,), device="cuda", dtype=torch.bool)
    tx_pol = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    material_eta_r = torch.ones((1,), device="cuda", dtype=torch.float32)
    material_sigma = torch.zeros((1,), device="cuda", dtype=torch.float32)
    material_mu_r = torch.ones((1,), device="cuda", dtype=torch.float32)
    material_gain = torch.ones((1,), device="cuda", dtype=torch.float32)
    material_valid = torch.ones((1,), device="cuda", dtype=torch.bool)

    out = ops.raydn_reflection_accumulation_forward(
        scene,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        ray_o,
        tx_pol,
        material_eta_r,
        material_sigma,
        material_mu_r,
        material_gain,
        material_valid,
        1,
        2,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        4,
        4,
        1.0,
        0.25,
        False,
        False,
        0,
        1,
        0,
        64,
        4,
        0,
        False,
    )

    assert out[0].shape == (4, 4)
    assert out[-1].dtype == torch.int32


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

    out = ops.raydn_reflection_epc_paths_forward(
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
