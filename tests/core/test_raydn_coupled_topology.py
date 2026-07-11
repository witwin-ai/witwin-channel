import math

import pytest
import torch

from tests.support.scenes import coupled_wall_wedge_scene
from witwin.channel_native import Scene, Structure
from witwin.channel_native.core.kernels import ops, raydn_backend
from witwin.channel_native.core.materials import PerfectConductor


def _wall_and_wedge_scene():
    return coupled_wall_wedge_scene().raydn_scene()


def _coupled_inputs(*, edge_id: int, reverse_endpoints: bool = False):
    device = torch.device("cuda")
    source = torch.tensor([[0.0, -2.0, 1.0]], device=device)
    # The receiver has the same radial distance from the edge axis as the
    # reflected image source, so the analytic stationary point is y=0.
    receiver = torch.tensor([[0.0, 2.0, 5.0]], device=device)
    if reverse_endpoints:
        source, receiver = receiver, source
    return (
        source,
        receiver,
        torch.tensor([0], device=device, dtype=torch.int32),
        torch.tensor([[0.0, 0.0, 0.0]], device=device),
        torch.tensor([[0.0, 0.0, 1.0]], device=device),
        torch.tensor([edge_id], device=device, dtype=torch.int32),
        torch.tensor([[2.0, 0.0, 2.0]], device=device),
        torch.tensor([[0.0, 1.0, 0.0]], device=device),
        torch.tensor([-1.0], device=device),
        torch.tensor([1.0], device=device),
        torch.tensor([0, 0, 1, 1, 2, 2], device=device, dtype=torch.int32),
        torch.tensor([2, 2, 2], device=device, dtype=torch.int32),
        torch.tensor([0, 1, 2, 3, 4, 5], device=device, dtype=torch.int32),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")
def test_coupled_reflection_diffraction_geometry_matches_image_solution():
    raydn_backend.require_native_extension()
    scene = _wall_and_wedge_scene()
    records = scene.edge_records()
    axis = ((records.edge_v0 == 4) & (records.edge_v1 == 5)) | (
        (records.edge_v0 == 5) & (records.edge_v1 == 4)
    )
    edge_id = int(torch.nonzero(axis, as_tuple=False)[0, 0].item())
    result = ops.raydn_coupled_rd_geometry_forward(
        scene, *_coupled_inputs(edge_id=edge_id), False
    )

    assert bool(result["valid"][0].item())
    assert (
        "field" not in result
        and "path_field" not in result
        and "path_gain" not in result
    )
    torch.testing.assert_close(
        result["interaction_type_sequence"],
        torch.tensor([[1, 2]], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(
        result["primitive_sequence"],
        torch.tensor([[0, -1]], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(
        result["edge_sequence"],
        torch.tensor([[-1, edge_id]], device="cuda", dtype=torch.int32),
    )
    expected_reflection = torch.tensor([2.0 / 3.0, -4.0 / 3.0, 0.0], device="cuda")
    expected_edge = torch.tensor([2.0, 0.0, 2.0], device="cuda")
    torch.testing.assert_close(
        result["interaction_positions"][0, 0],
        expected_reflection,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    torch.testing.assert_close(result["interaction_positions"][0, 1], expected_edge)
    expected_length = 2.0 * math.sqrt(17.0)
    torch.testing.assert_close(
        result["path_length_m"],
        torch.tensor([expected_length], device="cuda"),
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    torch.testing.assert_close(
        result["delay_s"],
        result["path_length_m"] / 299792458.0,
        rtol=1.0e-6,
        atol=0.0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")
def test_coupled_diffraction_reflection_is_reciprocal_geometry():
    raydn_backend.require_native_extension()
    scene = _wall_and_wedge_scene()
    records = scene.edge_records()
    axis = ((records.edge_v0 == 4) & (records.edge_v1 == 5)) | (
        (records.edge_v0 == 5) & (records.edge_v1 == 4)
    )
    edge_id = int(torch.nonzero(axis, as_tuple=False)[0, 0].item())
    rd = ops.raydn_coupled_rd_geometry_forward(
        scene, *_coupled_inputs(edge_id=edge_id), False
    )
    dr = ops.raydn_coupled_rd_geometry_forward(
        scene, *_coupled_inputs(edge_id=edge_id, reverse_endpoints=True), True
    )

    assert bool(rd["valid"][0].item()) and bool(dr["valid"][0].item())
    torch.testing.assert_close(
        dr["interaction_type_sequence"],
        torch.tensor([[2, 1]], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(dr["path_length_m"], rd["path_length_m"])
    torch.testing.assert_close(
        dr["interaction_positions"], rd["interaction_positions"].flip(1)
    )
    torch.testing.assert_close(
        dr["primitive_sequence"],
        torch.tensor([[-1, 0]], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(
        dr["edge_sequence"],
        torch.tensor([[edge_id, -1]], device="cuda", dtype=torch.int32),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")
def test_coupled_geometry_rejects_stationary_point_outside_edge_bounds():
    raydn_backend.require_native_extension()
    scene = _wall_and_wedge_scene()
    records = scene.edge_records()
    axis = ((records.edge_v0 == 4) & (records.edge_v1 == 5)) | (
        (records.edge_v0 == 5) & (records.edge_v1 == 4)
    )
    edge_id = int(torch.nonzero(axis, as_tuple=False)[0, 0].item())
    inputs = list(_coupled_inputs(edge_id=edge_id))
    inputs[8] = torch.tensor([0.25], device="cuda")

    result = ops.raydn_coupled_rd_geometry_forward(scene, *inputs, False)

    assert not bool(result["valid"][0].item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")
def test_coupled_geometry_rejects_blocked_secondary_segment():
    raydn_backend.require_native_extension()
    base = coupled_wall_wedge_scene()
    blocker = Structure(
        vertices=torch.tensor(
            [[0.0, 1.0, 2.5], [2.0, 1.0, 2.5], [0.0, 1.0, 4.5], [2.0, 1.0, 4.5]]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=PerfectConductor(),
        name="secondary-segment-blocker",
        surface_id=99,
    )
    scene = Scene(
        structures=[*base.structures, blocker],
        transmitters=base.transmitters,
        receivers=base.receivers,
        frequency=base.frequency,
    ).raydn_scene()
    records = scene.edge_records()
    axis = ((records.edge_v0 == 4) & (records.edge_v1 == 5)) | (
        (records.edge_v0 == 5) & (records.edge_v1 == 4)
    )
    edge_id = int(torch.nonzero(axis, as_tuple=False)[0, 0].item())

    result = ops.raydn_coupled_rd_geometry_forward(
        scene, *_coupled_inputs(edge_id=edge_id), False
    )

    assert not bool(result["valid"][0].item())
