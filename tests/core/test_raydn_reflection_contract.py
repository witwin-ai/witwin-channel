import pytest
import torch

from witwin.channel_native.core.kernels import raydn_backend


def _native_single_triangle_scene():
    raydn_backend.require_native_extension()
    device = torch.device("cuda")
    vertices = [
        torch.tensor(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
            device=device,
            dtype=torch.float32,
        )
    ]
    faces = [torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)]
    empty_uv = [torch.empty((0, 2), device=device, dtype=torch.float32)]
    empty_face_uv = [torch.empty((0, 3), device=device, dtype=torch.int32)]
    empty_transform = [torch.empty((0, 4), device=device, dtype=torch.float32)]
    with torch._C._DisableFuncTorch():
        return torch.classes.raydn.Scene(
            vertices,
            faces,
            empty_uv,
            empty_face_uv,
            empty_transform,
            empty_transform,
            [2],
        )


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

    out = torch.ops.raydn.reflection_accumulation_forward(
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
    )

    assert out[0].shape == (4, 4)
    assert out[-1].dtype == torch.int32
