# Copyright Xingyu Chen.
# Tests rayd coupled topology.

import math

import pytest
import torch

from tests.support.scenes import coupled_wall_wedge_scene
from tests.support.core_world import make_mesh_structure
from witwin.core import Scene
from witwin.channel.scene import compile as compile_scene
from witwin.channel.kernels import geometry as ops
from witwin.core import PhysicalMaterial
from witwin.channel import runtime


def _wall_and_wedge_scene():
    return compile_scene(coupled_wall_wedge_scene(), reference_frequency_hz=3.0e9).rayd


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
    runtime.native_extension()
    scene = _wall_and_wedge_scene()
    records = scene.edge_records()
    axis = ((records.edge_v0 == 4) & (records.edge_v1 == 5)) | (
        (records.edge_v0 == 5) & (records.edge_v1 == 4)
    )
    edge_id = int(torch.nonzero(axis, as_tuple=False)[0, 0].item())
    result = ops.coupled_rd_geometry_forward(
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
    runtime.native_extension()
    scene = _wall_and_wedge_scene()
    records = scene.edge_records()
    axis = ((records.edge_v0 == 4) & (records.edge_v1 == 5)) | (
        (records.edge_v0 == 5) & (records.edge_v1 == 4)
    )
    edge_id = int(torch.nonzero(axis, as_tuple=False)[0, 0].item())
    rd = ops.coupled_rd_geometry_forward(
        scene, *_coupled_inputs(edge_id=edge_id), False
    )
    dr = ops.coupled_rd_geometry_forward(
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
    runtime.native_extension()
    scene = _wall_and_wedge_scene()
    records = scene.edge_records()
    axis = ((records.edge_v0 == 4) & (records.edge_v1 == 5)) | (
        (records.edge_v0 == 5) & (records.edge_v1 == 4)
    )
    edge_id = int(torch.nonzero(axis, as_tuple=False)[0, 0].item())
    inputs = list(_coupled_inputs(edge_id=edge_id))
    inputs[8] = torch.tensor([0.25], device="cuda")

    result = ops.coupled_rd_geometry_forward(scene, *inputs, False)

    assert not bool(result["valid"][0].item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")
def test_coupled_geometry_rejects_blocked_secondary_segment():
    runtime.native_extension()
    base = coupled_wall_wedge_scene()
    blocker = make_mesh_structure(
        vertices=torch.tensor(
            [[0.0, 1.0, 2.5], [2.0, 1.0, 2.5], [0.0, 1.0, 4.5], [2.0, 1.0, 4.5]]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=PhysicalMaterial.perfect_conductor(),
        name="secondary-segment-blocker",
        surface_id=99,
    )
    scene = Scene(
        structures=[*base.structures, blocker],
        endpoints=base.endpoints,
    )
    scene = compile_scene(scene, reference_frequency_hz=3.0e9).rayd
    records = scene.edge_records()
    axis = ((records.edge_v0 == 4) & (records.edge_v1 == 5)) | (
        (records.edge_v0 == 5) & (records.edge_v1 == 4)
    )
    edge_id = int(torch.nonzero(axis, as_tuple=False)[0, 0].item())

    result = ops.coupled_rd_geometry_forward(
        scene, *_coupled_inputs(edge_id=edge_id), False
    )

    assert not bool(result["valid"][0].item())



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA torch is required")
def test_coupled_dd_certifies_stationary_root_and_utd_leg_domain():
    """D-D rows publish only a converged, ordinary three-leg UTD path.

    Rows 1-3 reproduce the three-cube edge-pair classes 33->30, 35->30,
    and 42->30 at the lower side of the audited RSB.  Row 4 is the exact
    single-cube grid witness that exposed a false sequential-map residual
    assertion despite a one-ULP signed Fermat bracket.  Row 0 is an independent
    parallel-edge construction: every leg is longer than ``UTD_MIN_DISTANCE``
    and both points are interior, but the old 64-map FP32 residual remains
    3.9e-5 m.  The convex bracketed solve must publish its converged pair rather
    than silently dropping that physical path.  The 33->30 and 35->30 limits
    land on a shared endpoint with middle legs below 1e-4 m and are outside the
    sequential UTD domain.  The 42->30 limit is an ordinary, slowly converging
    path and remains valid.  Expected points come from independent FP64
    closed-form (parallel row), 1024-map (42->30), and joint FP64 Fermat
    (single-cube witness) oracles.
    """

    runtime.native_extension()
    device = torch.device("cuda")
    far_geometry = make_mesh_structure(
        vertices=torch.tensor(
            [[100.0, 100.0, 100.0], [101.0, 100.0, 100.0], [100.0, 101.0, 100.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        material=PhysicalMaterial.perfect_conductor(),
        name="far-from-coupled-dd-rays",
        surface_id=999,
    )
    scene = compile_scene(
        Scene(structures=[far_geometry]), reference_frequency_hz=3.0e9
    ).rayd

    source = torch.tensor(
        [
            [-1.0, -0.7, 0.1],
            [0.0, -0.5, 0.4],
            [0.0, -0.5, 0.4],
            [0.0, -0.5, 0.4],
            [-0.2, -0.5, 0.42],
        ],
        device=device,
    )
    receiver = torch.tensor(
        [
            [1.0, 0.7, 0.1],
            [-0.003125, 0.7687255859375, 0.1],
            [-0.003125, 0.7687255859375, 0.1],
            [-0.003125, 0.7687255859375, 0.1],
            [-0.203125, -0.5343749523162842, 0.1],
        ],
        device=device,
    )
    edge1_position = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [0.1, -0.05, 0.25],
            [0.3, 0.15, 0.25],
            [0.05, 0.25, 0.05],
            [0.0, -0.1, 0.25],
        ],
        device=device,
    )
    edge1_direction = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        device=device,
    )
    edge2_position = torch.tensor(
        [
            [0.01, -1.0, 0.0],
            [0.1, 0.15, 0.05],
            [0.1, 0.15, 0.05],
            [0.1, 0.15, 0.05],
            [0.1, 0.1, 0.15],
        ],
        device=device,
    )
    edge2_direction = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
    )
    edge1_t_min = torch.tensor([0.0, 0.0, 0.0, 0.0, -0.1], device=device)
    edge1_t_max = torch.tensor([2.0, 0.2, 0.2, 0.2, 0.1], device=device)
    edge2_t_min = torch.tensor([0.0, 0.0, 0.0, 0.0, -0.1], device=device)
    edge2_t_max = torch.tensor([2.0, 0.2, 0.2, 0.2, 0.1], device=device)
    edge1_id = torch.tensor([100, 33, 35, 42, 13], device=device, dtype=torch.int32)
    edge2_id = torch.tensor([101, 30, 30, 30, 9], device=device, dtype=torch.int32)

    result = ops.coupled_dd_geometry_forward(
        scene,
        source,
        receiver,
        edge1_id,
        edge1_position,
        edge1_direction,
        edge1_t_min,
        edge1_t_max,
        edge2_id,
        edge2_position,
        edge2_direction,
        edge2_t_min,
        edge2_t_max,
    )

    expected_active = torch.tensor(
        [True, False, False, True, True], device=device, dtype=torch.bool
    )
    torch.testing.assert_close(result["candidate_active"], expected_active)
    torch.testing.assert_close(result["valid"], expected_active)
    assert torch.isnan(result["interaction_positions"][1:3]).all()
    expected_parallel_q = torch.tensor(
        [
            [0.0, -0.0000174564630282285, 0.0],
            [0.01, 0.00694763017014247, 0.0],
        ],
        device=device,
    )
    torch.testing.assert_close(
        result["interaction_positions"][0],
        expected_parallel_q,
        rtol=0.0,
        atol=2.0e-5,
    )
    expected_slow_q = torch.tensor(
        [
            [0.05000000074505806, 0.25, 0.24873210088185085],
            [0.10000000149011612, 0.15000000596046448, 0.22623235887015736],
        ],
        device=device,
    )
    torch.testing.assert_close(
        result["interaction_positions"][3], expected_slow_q, rtol=0.0, atol=1.0e-6
    )
    expected_single_cube_q = torch.tensor(
        [
            [0.00442336198782044, -0.10000000149011612, 0.25],
            [0.10000000149011612, 0.10000000149011612, 0.21404440328522903],
        ],
        device=device,
    )
    torch.testing.assert_close(
        result["interaction_positions"][4],
        expected_single_cube_q,
        rtol=0.0,
        atol=1.0e-6,
    )
    assert torch.isfinite(result["path_length_m"][0])
    assert torch.isfinite(result["path_length_m"][3])
    assert torch.isfinite(result["path_length_m"][4])

    valid_rows = torch.tensor([0, 3, 4], device=device)
    stationary_points = result["interaction_positions"][valid_rows]
    q1 = stationary_points[:, 0]
    q2 = stationary_points[:, 1]
    incoming = q1 - source[valid_rows]
    middle = q2 - q1
    to_receiver = receiver[valid_rows] - q2
    incoming_hat = incoming / torch.linalg.vector_norm(incoming, dim=1, keepdim=True)
    middle_hat = middle / torch.linalg.vector_norm(middle, dim=1, keepdim=True)
    receiver_hat = to_receiver / torch.linalg.vector_norm(
        to_receiver, dim=1, keepdim=True
    )
    edge1_gradient = torch.sum(
        edge1_direction[valid_rows] * (incoming_hat - middle_hat), dim=1
    )
    edge2_gradient = torch.sum(
        edge2_direction[valid_rows] * (middle_hat - receiver_hat), dim=1
    )
    assert torch.max(torch.abs(edge1_gradient)).item() <= 8.0e-6
    assert torch.max(torch.abs(edge2_gradient)).item() <= 8.0e-6

    # Reverse the physical path, swap the two edges, and reverse both edge-axis
    # parameterizations.  The stationary points and ordinary-domain
    # classification are physical quantities, so they must not depend on path
    # direction or on which endpoint was chosen as an edge origin.
    reverse = ops.coupled_dd_geometry_forward(
        scene,
        receiver,
        source,
        edge2_id,
        edge2_position
        + edge2_direction * (edge2_t_min + edge2_t_max)[:, None],
        -edge2_direction,
        edge2_t_min,
        edge2_t_max,
        edge1_id,
        edge1_position
        + edge1_direction * (edge1_t_min + edge1_t_max)[:, None],
        -edge1_direction,
        edge1_t_min,
        edge1_t_max,
    )
    torch.testing.assert_close(reverse["candidate_active"], expected_active)
    torch.testing.assert_close(reverse["valid"], expected_active)
    path_scale = float(result["path_length_m"][valid_rows].max().item())
    world_scale = float(result["interaction_positions"][valid_rows].abs().max().item())
    reversal_atol = 32.0 * torch.finfo(torch.float32).eps * max(path_scale, world_scale)
    torch.testing.assert_close(
        reverse["interaction_positions"][valid_rows],
        result["interaction_positions"][valid_rows].flip(1),
        rtol=0.0,
        atol=reversal_atol,
    )
    torch.testing.assert_close(
        reverse["path_length_m"][valid_rows],
        result["path_length_m"][valid_rows],
        rtol=0.0,
        atol=reversal_atol,
    )
