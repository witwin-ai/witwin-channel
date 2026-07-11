import pytest
import torch

from witwin.channel_native import Scene, Structure
from witwin.channel_native.core.kernels import ops, raydn_backend
from witwin.channel_native.core.materials import PerfectConductor


def _native_single_triangle_scene():
    raydn_backend.require_native_extension()
    device = torch.device("cuda")
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
def test_reflection_accumulation_exports_wedge_events_when_requested():
    scene = _native_single_triangle_scene()
    ray_o = torch.tensor([[0.0, 0.0, -1.0]], device="cuda", dtype=torch.float32)
    ray_d = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
    ray_tmax = torch.tensor([2.0], device="cuda", dtype=torch.float32)
    active = torch.ones((1,), device="cuda", dtype=torch.bool)
    tx_pol = torch.tensor([[1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    material_eta_r = torch.full((1,), 4.0, device="cuda", dtype=torch.float32)
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
        True,
        False,
        4,
        1,
        0,
        64,
        4,
        0,
        False,
    )

    torch.cuda.synchronize()
    assert len(out) >= 18
    wedge_count = int(out[8].item())
    assert wedge_count == 1
    assert out[9].shape == (4,)
    assert out[10].shape == (4, 3)
    assert out[12].shape == (4,)
    assert out[13].shape == (4, 3)
    assert out[14].shape == (4, 3)
    assert out[15].shape == (4,)
    assert out[16].shape == (4, 3)
    assert out[17].shape == (4,)
    assert int(out[9][0].item()) == 0
    assert int(out[12][0].item()) == 0
    assert int(out[17][0].item()) == 0
    torch.testing.assert_close(out[10][0], torch.tensor([0.0, 0.0, 0.0], device="cuda"))
    torch.testing.assert_close(out[13][0], ray_d[0])
    torch.testing.assert_close(out[14][0], ray_o[0])
