import pytest
import torch

from witwin.channel_native import Scene, Structure
from witwin.channel_native.propagation.geometry.kernels import bridge as ops
from witwin.channel_native.core.materials import PerfectConductor
from witwin.channel_native.runtime import symbols


def _native_unoccluded_scene():
    symbols.native_extension()
    scene = Scene(
        structures=[Structure(
            vertices=torch.tensor(
            [[20.0, 0.0, 0.0], [20.0, 1.0, 0.0], [20.0, 0.0, 1.0]],
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
def test_diffraction_sample_tape_accepts_sample_state_slots_and_weights():
    scene = _native_unoccluded_scene()
    state_edge_index = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    state_edge_pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.25, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )
    state_edge_dir = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        device="cuda",
        dtype=torch.float32,
    )
    state_edge_t_min = torch.tensor([-0.25, -0.25], device="cuda", dtype=torch.float32)
    state_edge_t_max = torch.tensor([0.25, 0.25], device="cuda", dtype=torch.float32)
    state_n0 = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32)
    state_n1 = torch.tensor([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]], device="cuda", dtype=torch.float32)
    state_prim0 = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
    state_prim1 = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
    state_exterior_angle = torch.tensor([torch.pi, torch.pi], device="cuda", dtype=torch.float32)
    state_src = torch.tensor([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], device="cuda", dtype=torch.float32)
    state_src_power = torch.ones((2,), device="cuda", dtype=torch.float32)
    active = torch.tensor([False, True], device="cuda", dtype=torch.bool)
    material_eta_r = torch.ones((1,), device="cuda", dtype=torch.float32)
    material_sigma = torch.zeros((1,), device="cuda", dtype=torch.float32)
    material_mu_r = torch.ones((1,), device="cuda", dtype=torch.float32)
    material_gain = torch.ones((1,), device="cuda", dtype=torch.float32)
    material_valid = torch.ones((1,), device="cuda", dtype=torch.bool)
    sample_state_index = torch.tensor([1], device="cuda", dtype=torch.int32)
    sample_edge_weight = torch.tensor([1.0], device="cuda", dtype=torch.float32)

    out = ops.rayd_diffraction_sample_tape_forward(
        scene,
        active,
        state_edge_index,
        state_edge_pos,
        state_edge_dir,
        state_edge_t_min,
        state_edge_t_max,
        state_n0,
        state_n1,
        state_prim0,
        state_prim1,
        state_exterior_angle,
        state_src,
        state_src_power,
        None,
        None,
        material_eta_r,
        material_sigma,
        material_mu_r,
        material_gain,
        material_valid,
        2,
        0,
        2.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        2,
        2,
        1.0,
        1.0,
        1,
        0,
        0,
        17,
        1,
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        1,
        sample_state_index,
        sample_edge_weight,
    )

    assert len(out) == 19
    torch.cuda.synchronize()
    assert float(out[0].sum().item()) > 0.0
    assert int(out[7].item()) == 1
    assert int(out[14].sum().item()) == 1
    assert int(out[15][0].item()) == 1
