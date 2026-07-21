from __future__ import annotations

import pytest
import torch

from witwin.channel_native import Scene, Structure
from witwin.channel_native.core.materials import Dielectric
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.propagation.geometry.kernels import bridge
from witwin.channel_native.propagation.models.penetration import (
    SegmentPenetrationPolicy,
    SegmentPenetrationResult,
)
from witwin.channel_native.propagation.topology.kernels.transmission import (
    enumerated_transmission_topology_pack,
)
from witwin.channel_native.runtime.capacity import create_capacity_failure_state


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

_WALL_VERTICES = (
    (-8.0, -8.0, 0.0),
    (8.0, -8.0, 0.0),
    (0.0, 8.0, 0.0),
    (-8.0, -8.0, 1.0),
    (8.0, -8.0, 1.0),
    (0.0, 8.0, 1.0),
    (-8.0, -8.0, 2.0),
    (8.0, -8.0, 2.0),
    (0.0, 8.0, 2.0),
)
_WALL_FACES = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
_RESULT_FIELDS = (
    "valid",
    "num_hits",
    "reached_target",
    "overflow",
    "distance",
    "direction",
    "t",
    "position",
    "normal",
    "geometric_normal",
    "global_primitive_id",
)


def _scene() -> tuple[object, torch.Tensor]:
    try:
        if build_info()["rayd_integration"] != "source-linked":
            pytest.skip("RayD source-linked extension is not built")
    except ModuleNotFoundError:
        pytest.skip("native extension is not built")
    scene = Scene(
        structures=[
            Structure(
                vertices=torch.tensor(_WALL_VERTICES, dtype=torch.float32),
                faces=torch.tensor(_WALL_FACES, dtype=torch.int32),
                material=Dielectric(eps_r=2.0),
            )
        ],
        transmitters=[],
        receivers=[],
        frequency=3.5e9,
    )
    rayd = scene.rayd_scene()
    return rayd, rayd.mesh_tensors[0][0]


def _request(
    rayd: object,
    origins: torch.Tensor,
    targets: torch.Tensor,
    *,
    hit_capacity: int = 2,
    policy: SegmentPenetrationPolicy = SegmentPenetrationPolicy.EnumeratedFullDistance,
):
    failure_state = create_capacity_failure_state(origins)
    kwargs = {
        "input_active_any": True,
        "hit_capacity": hit_capacity,
        "policy": policy,
        "scene_diagonal": 23.0,
        "failure_state": failure_state,
    }
    return failure_state, kwargs


def _assert_result_inert(result: SegmentPenetrationResult) -> None:
    assert not result.valid.any().item()
    assert torch.count_nonzero(result.num_hits).item() == 0
    assert not result.reached_target.any().item()
    assert torch.count_nonzero(result.distance).item() == 0
    assert torch.count_nonzero(result.direction).item() == 0
    assert torch.equal(result.t, torch.full_like(result.t, -1.0))
    assert torch.count_nonzero(result.position).item() == 0
    assert torch.count_nonzero(result.normal).item() == 0
    assert torch.count_nonzero(result.geometric_normal).item() == 0
    assert torch.equal(
        result.global_primitive_id,
        torch.full_like(result.global_primitive_id, -1),
    )


def test_forward_tape_and_topology_pack_share_capacity_transaction() -> None:
    rayd, _vertices = _scene()
    origins = torch.tensor(
        [[0.0, 0.0, -1.0], [7.0, 7.0, -1.0]],
        dtype=torch.float32,
        device="cuda",
    )
    targets = torch.tensor(
        [[0.0, 0.0, 1.5], [7.0, 7.0, 1.5]],
        dtype=torch.float32,
        device="cuda",
    )
    failure_state, kwargs = _request(rayd, origins, targets)

    plain = bridge.rayd_segment_penetration_forward(
        rayd, origins, targets, None, **kwargs
    )
    taped = bridge.rayd_segment_penetration_forward_tape(
        rayd, origins, targets, None, **kwargs
    )

    for name in _RESULT_FIELDS:
        assert torch.equal(getattr(plain, name), getattr(taped.result, name))
        assert getattr(plain, name).is_contiguous()
    assert failure_state.bits.tolist() == [0]
    assert plain.num_hits.tolist() == [2, 0]
    assert plain.reached_target.tolist() == [True, True]
    assert plain.global_primitive_id[0].tolist() == [0, 1]

    packed = enumerated_transmission_topology_pack(
        taped.result,
        torch.zeros(3, dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        tx_count=1,
        rx_count=2,
    )
    assert packed.failure_state is failure_state
    assert packed.valid.tolist() == [True, False]
    assert packed.tx_id.tolist() == [0, -1]
    assert packed.rx_id.tolist() == [0, -1]
    assert packed.depth.tolist() == [2, 0]
    assert packed.primitive_sequence[0].tolist() == [0, 1]
    assert packed.execution.device_candidate_count.tolist() == [1]
    assert packed.execution.device_guardrail_count.tolist() == [0]


def test_pair_major_batch_covers_clear_one_hit_exact_capacity_and_zero_length() -> None:
    rayd, _vertices = _scene()
    origins = torch.tensor(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [7.0, 7.0, -1.0],
            [3.0, 3.0, 0.5],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    targets = torch.tensor(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.5],
            [7.0, 7.0, 1.5],
            [3.0, 3.0, 0.5],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        failure_state, kwargs = _request(rayd, origins, targets)
        result = bridge.rayd_segment_penetration_forward(
            rayd, origins, targets, None, **kwargs
        )
        packed = enumerated_transmission_topology_pack(
            result,
            torch.zeros(3, dtype=torch.int32, device="cuda"),
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            tx_count=2,
            rx_count=2,
        )
    stream.synchronize()

    assert failure_state.bits.tolist() == [0]
    assert result.num_hits.tolist() == [1, 2, 0, 0]
    assert result.reached_target.tolist() == [True, True, True, True]
    assert result.valid.tolist() == [
        [True, False],
        [True, True],
        [False, False],
        [False, False],
    ]
    assert packed.failure_state is failure_state
    assert packed.valid.tolist() == [True, True, False, False]
    assert packed.tx_id.tolist() == [0, 0, -1, -1]
    assert packed.rx_id.tolist() == [0, 1, -1, -1]
    assert packed.depth.tolist() == [1, 2, 0, 0]
    assert packed.execution.device_candidate_count.tolist() == [2]
    assert packed.execution.device_guardrail_count.tolist() == [0]


def test_invalid_material_makes_capacity_row_inert_without_partial_result() -> None:
    rayd, _vertices = _scene()
    origins = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device="cuda")
    targets = torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float32, device="cuda")
    failure_state, kwargs = _request(rayd, origins, targets, hit_capacity=1)
    result = bridge.rayd_segment_penetration_forward(
        rayd, origins, targets, None, **kwargs
    )
    packed = enumerated_transmission_topology_pack(
        result,
        torch.zeros(3, dtype=torch.int32, device="cuda"),
        torch.ones(1, dtype=torch.int32, device="cuda"),
        tx_count=1,
        rx_count=1,
    )

    assert failure_state.bits.tolist() == [0]
    assert result.num_hits.tolist() == [1]
    assert packed.valid.tolist() == [False]
    assert packed.tx_id.tolist() == [-1]
    assert packed.rx_id.tolist() == [-1]
    assert packed.primitive_sequence.tolist() == [[-1]]
    assert packed.execution.device_candidate_count.tolist() == [1]
    assert packed.execution.device_guardrail_count.tolist() == [1]


def test_forward_jvp_and_backward_obey_adjoint_identity() -> None:
    rayd, vertices = _scene()
    origins = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device="cuda")
    targets = torch.tensor([[0.0, 0.0, 1.5]], dtype=torch.float32, device="cuda")
    failure_state, kwargs = _request(rayd, origins, targets)
    tape = bridge.rayd_segment_penetration_forward_tape(
        rayd, origins, targets, None, **kwargs
    )
    tangent_vertices = torch.full_like(vertices, 0.001)
    tangent_origins = torch.tensor(
        [[0.01, -0.02, 0.03]], dtype=torch.float32, device="cuda"
    )
    tangent_targets = torch.tensor(
        [[-0.03, 0.01, 0.02]], dtype=torch.float32, device="cuda"
    )
    tangents = bridge.rayd_segment_penetration_jvp(
        rayd,
        origins,
        targets,
        None,
        **kwargs,
        tape=tape,
        tangent_vertices=tangent_vertices,
        tangent_origins=tangent_origins,
        tangent_targets=tangent_targets,
    )
    cotangents = {
        "grad_distance": torch.full_like(tape.result.distance, 0.3),
        "grad_direction": torch.full_like(tape.result.direction, -0.2),
        "grad_t": torch.full_like(tape.result.t, 0.4),
        "grad_position": torch.full_like(tape.result.position, 0.1),
        "grad_normal": torch.full_like(tape.result.normal, -0.15),
        "grad_geometric_normal": torch.full_like(tape.result.geometric_normal, 0.05),
    }
    gradients = bridge.rayd_segment_penetration_backward(
        rayd,
        origins,
        targets,
        None,
        **kwargs,
        tape=tape,
        **cotangents,
        need_grad_vertices=True,
        need_grad_origins=True,
        need_grad_targets=True,
    )
    assert gradients.grad_vertices is not None
    assert gradients.grad_origins is not None
    assert gradients.grad_targets is not None

    lhs = sum(
        (getattr(tangents, tangent_name) * cotangents[gradient_name]).sum()
        for tangent_name, gradient_name in (
            ("tangent_distance", "grad_distance"),
            ("tangent_direction", "grad_direction"),
            ("tangent_t", "grad_t"),
            ("tangent_position", "grad_position"),
            ("tangent_normal", "grad_normal"),
            ("tangent_geometric_normal", "grad_geometric_normal"),
        )
    )
    rhs = (
        (tangent_vertices * gradients.grad_vertices).sum()
        + (tangent_origins * gradients.grad_origins).sum()
        + (tangent_targets * gradients.grad_targets).sum()
    )
    torch.testing.assert_close(lhs, rhs, rtol=2.0e-4, atol=2.0e-5)


def test_overflow_sanitizes_penetration_and_downstream_topology() -> None:
    rayd, _vertices = _scene()
    origins = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device="cuda")
    targets = torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32, device="cuda")
    failure_state, kwargs = _request(rayd, origins, targets, hit_capacity=2)
    tape = bridge.rayd_segment_penetration_forward_tape(
        rayd, origins, targets, None, **kwargs
    )

    assert failure_state.bits.tolist() == [1 << 7]
    assert tape.result.overflow.tolist() == [True]
    _assert_result_inert(tape.result)

    packed = enumerated_transmission_topology_pack(
        tape.result,
        torch.zeros(3, dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        tx_count=1,
        rx_count=1,
    )
    assert packed.failure_state is failure_state
    assert not packed.valid.any().item()
    assert packed.execution.device_candidate_count.tolist() == [0]
    assert packed.execution.device_guardrail_count.tolist() == [0]


def test_monte_carlo_policy_batches_clear_one_exact_capacity_and_zero_length() -> None:
    rayd, _vertices = _scene()
    origins = torch.tensor(
        [
            [7.0, 7.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [3.0, 3.0, 0.5],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    targets = torch.tensor(
        [
            [7.0, 7.0, 1.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.5],
            [3.0, 3.0, 0.5],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        failure_state, kwargs = _request(
            rayd,
            origins,
            targets,
            policy=SegmentPenetrationPolicy.MonteCarloTargetInset,
        )
        result = bridge.rayd_segment_penetration_forward(
            rayd, origins, targets, None, **kwargs
        )
    stream.synchronize()

    assert failure_state.bits.tolist() == [0]
    assert result.num_hits.tolist() == [0, 1, 2, 0]
    assert result.reached_target.tolist() == [True, True, True, True]
    assert result.valid.tolist() == [
        [False, False],
        [True, False],
        [True, True],
        [False, False],
    ]
    assert result.global_primitive_id[1].tolist() == [0, -1]
    assert result.global_primitive_id[2].tolist() == [0, 1]


def test_monte_carlo_d_plus_one_overflow_poison_is_batch_wide() -> None:
    rayd, _vertices = _scene()
    origins = torch.tensor(
        [[0.0, 0.0, -1.0], [7.0, 7.0, -1.0]],
        dtype=torch.float32,
        device="cuda",
    )
    targets = torch.tensor(
        [[0.0, 0.0, 3.0], [7.0, 7.0, 3.0]],
        dtype=torch.float32,
        device="cuda",
    )
    failure_state, kwargs = _request(
        rayd,
        origins,
        targets,
        policy=SegmentPenetrationPolicy.MonteCarloTargetInset,
    )
    result = bridge.rayd_segment_penetration_forward(
        rayd, origins, targets, None, **kwargs
    )

    assert failure_state.bits.tolist() == [1 << 7]
    assert result.overflow.tolist() == [True, False]
    _assert_result_inert(result)
