# Copyright Xingyu Chen.
# Tests transmission topology capacity.

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from witwin.channel.propagation.penetration import (
    SegmentPenetrationResult,
)
from witwin.channel.kernels import topology as transmission_kernels
from witwin.channel.kernels.topology import (
    enumerated_transmission_topology_pack,
)
from witwin.channel.runtime import create_capacity_failure_state


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _penetration() -> SegmentPenetrationResult:
    valid = torch.tensor(
        [[False, False], [True, False], [True, True]],
        device="cuda",
        dtype=torch.bool,
    )
    failure_state = create_capacity_failure_state(valid)
    position = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
            [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        ],
        device="cuda",
    )
    normal = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.25, 0.5, 0.75], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        device="cuda",
    )
    return SegmentPenetrationResult(
        hit_capacity=2,
        failure_state=failure_state,
        valid=valid,
        num_hits=torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32),
        reached_target=torch.ones(3, device="cuda", dtype=torch.bool),
        overflow=torch.zeros(3, device="cuda", dtype=torch.bool),
        distance=torch.tensor([1.0, 2.0, 3.0], device="cuda"),
        direction=torch.tensor([[1.0, 0.0, 0.0]] * 3, device="cuda"),
        t=torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 2.0]], device="cuda"),
        position=position,
        normal=normal,
        geometric_normal=torch.full_like(position, -9.0),
        global_primitive_id=torch.tensor(
            [[-1, -1], [0, -1], [0, 1]], device="cuda", dtype=torch.int32
        ),
    )


def _empty_penetration(rows: int, hit_capacity: int) -> SegmentPenetrationResult:
    valid = torch.zeros((rows, hit_capacity), device="cuda", dtype=torch.bool)
    return SegmentPenetrationResult(
        hit_capacity=hit_capacity,
        failure_state=create_capacity_failure_state(valid),
        valid=valid,
        num_hits=torch.zeros(rows, device="cuda", dtype=torch.int32),
        reached_target=torch.ones(rows, device="cuda", dtype=torch.bool),
        overflow=torch.zeros(rows, device="cuda", dtype=torch.bool),
        distance=torch.ones(rows, device="cuda"),
        direction=torch.zeros((rows, 3), device="cuda"),
        t=torch.empty((rows, hit_capacity), device="cuda"),
        position=torch.empty((rows, hit_capacity, 3), device="cuda"),
        normal=torch.empty((rows, hit_capacity, 3), device="cuda"),
        geometric_normal=torch.empty((rows, hit_capacity, 3), device="cuda"),
        global_primitive_id=torch.empty(
            (rows, hit_capacity), device="cuda", dtype=torch.int32
        ),
    )


def test_pack_preserves_pair_capacity_normal_bits_and_device_counts() -> None:
    penetration = _penetration()

    packed = enumerated_transmission_topology_pack(
        penetration,
        torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        tx_count=1,
        rx_count=3,
    )

    assert packed.candidate_capacity == 3
    assert packed.sequence_width == 2
    assert packed.failure_state is penetration.failure_state
    assert packed.valid.tolist() == [False, True, False]
    assert packed.execution.device_candidate_count.tolist() == [2]
    assert packed.execution.device_guardrail_count.tolist() == [1]
    assert packed.interaction_position[1].tolist() == [1.0, 2.0, 3.0]
    assert packed.interaction_normal[1].tolist() == [0.25, 0.5, 0.75]
    assert packed.path_gain[1].view(torch.int32).item() == 0x7FC00000
    assert packed.path_field[1:2].view(torch.float32).view(torch.int32).tolist() == [
        0x7FC00000,
        0x7FC00000,
    ]
    block = packed.as_block()
    assert block["valid"] is packed.valid
    assert block["interaction_normal"] is packed.interaction_normal


@pytest.mark.parametrize(
    ("tx_count", "rx_count", "hit_capacity"), [(0, 3, 0), (2, 1, 0)]
)
def test_zero_pair_or_hit_capacity_is_inert(
    tx_count: int, rx_count: int, hit_capacity: int,
) -> None:
    rows = tx_count * rx_count
    penetration = _empty_penetration(rows, hit_capacity)

    packed = enumerated_transmission_topology_pack(
        penetration,
        torch.empty(0, device="cuda", dtype=torch.int32),
        torch.empty(0, device="cuda", dtype=torch.int32),
        tx_count=tx_count,
        rx_count=rx_count,
    )

    assert packed.valid.shape == (rows,)
    assert packed.primitive_sequence.shape == (rows, hit_capacity)
    assert not packed.valid.any().item()
    assert packed.execution.device_candidate_count.tolist() == [0]


def test_multi_endpoint_rows_are_tx_major_on_a_nondefault_stream() -> None:
    rows = 4
    penetration = _empty_penetration(rows, 1)
    penetration.valid.fill_(True)
    penetration.num_hits.fill_(1)
    penetration.global_primitive_id.fill_(0)
    penetration.position.zero_()
    penetration.normal.zero_()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        packed = enumerated_transmission_topology_pack(
            penetration,
            torch.tensor([0], device="cuda", dtype=torch.int32),
            torch.tensor([0], device="cuda", dtype=torch.int32),
            tx_count=2,
            rx_count=2,
        )
    stream.synchronize()

    assert packed.valid.tolist() == [True, True, True, True]
    assert packed.tx_id.tolist() == [0, 0, 1, 1]
    assert packed.rx_id.tolist() == [0, 1, 0, 1]


def test_contract_failure_sanitizes_entire_topology_and_counts() -> None:
    penetration = _penetration()
    penetration.global_primitive_id[2, 1] = 2_000_000_000

    packed = enumerated_transmission_topology_pack(
        penetration,
        torch.tensor([0, 0], device="cuda", dtype=torch.int32),
        torch.tensor([0], device="cuda", dtype=torch.int32),
        tx_count=1,
        rx_count=3,
    )

    assert penetration.failure_state.bits.tolist() != [0]
    assert not packed.valid.any().item()
    assert packed.execution.device_candidate_count.tolist() == [0]
    assert packed.execution.device_guardrail_count.tolist() == [0]
    assert packed.primitive_sequence.tolist() == [[-1, -1]] * 3
    assert torch.count_nonzero(packed.path_field).item() == 0


@pytest.mark.parametrize("corruption", ["count", "prefix", "reached", "overflow"])
def test_discrete_corruption_fails_before_payload_gather(corruption: str) -> None:
    penetration = _penetration()
    if corruption == "count":
        penetration.num_hits[1] = 3
    elif corruption == "prefix":
        penetration.valid[2, 0] = False
    elif corruption == "reached":
        penetration.reached_target[1] = False
    else:
        penetration.overflow[1] = True
    penetration.position.fill_(float("nan"))
    penetration.normal.fill_(float("nan"))

    packed = enumerated_transmission_topology_pack(
        penetration,
        torch.tensor([0, 0], device="cuda", dtype=torch.int32),
        torch.tensor([0], device="cuda", dtype=torch.int32),
        tx_count=1,
        rx_count=3,
    )

    assert penetration.failure_state.bits.tolist() != [0]
    assert not packed.valid.any().item()
    assert torch.count_nonzero(packed.interaction_positions).item() == 0


def test_preexisting_failure_sanitizes_without_reading_poison_payload() -> None:
    penetration = _penetration()
    penetration.failure_state.bits.fill_(1)
    penetration.global_primitive_id.fill_(2_000_000_000)
    penetration.position.fill_(float("nan"))

    packed = enumerated_transmission_topology_pack(
        penetration,
        torch.tensor([0], device="cuda", dtype=torch.int32),
        torch.tensor([0], device="cuda", dtype=torch.int32),
        tx_count=1,
        rx_count=3,
    )

    assert not packed.valid.any().item()
    assert packed.execution.device_candidate_count.tolist() == [0]


def test_missing_native_symbol_has_no_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    penetration = _penetration()

    def missing(name: str) -> object:
        raise RuntimeError(f"missing {name}")

    monkeypatch.setattr(transmission_kernels, "_required_native_op", missing)
    with pytest.raises(RuntimeError, match="enumerated_transmission_topology_pack"):
        enumerated_transmission_topology_pack(
            penetration,
            torch.tensor([0, 0], device="cuda", dtype=torch.int32),
            torch.tensor([0], device="cuda", dtype=torch.int32),
            tx_count=1,
            rx_count=3,
        )


def test_pack_backward_routes_only_valid_continuous_geometry() -> None:
    base = _penetration()
    distance = base.distance.detach().clone().requires_grad_(True)
    position = base.position.detach().clone().requires_grad_(True)
    normal = base.normal.detach().clone().requires_grad_(True)
    penetration = replace(
        base,
        distance=distance,
        position=position,
        normal=normal,
    )
    packed = enumerated_transmission_topology_pack(
        penetration,
        torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        tx_count=1,
        rx_count=3,
    )

    objective = (
        packed.path_length_m.sum()
        + packed.interaction_position.sum()
        + packed.interaction_normal.sum()
        + packed.interaction_positions.sum()
        + packed.interaction_normals.sum()
    )
    grad_distance, grad_position, grad_normal = torch.autograd.grad(
        objective, (distance, position, normal)
    )

    assert grad_distance.tolist() == [0.0, 1.0, 0.0]
    assert grad_position.tolist() == [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]
    assert grad_normal.tolist() == grad_position.tolist()


def test_pack_jvp_routes_only_valid_continuous_geometry() -> None:
    base = _penetration()
    tangent_distance = torch.tensor([3.0, 5.0, 7.0], device="cuda")
    tangent_position = torch.arange(
        base.position.numel(), device="cuda", dtype=torch.float32
    ).reshape_as(base.position)
    tangent_normal = tangent_position + 100.0

    with torch.autograd.forward_ad.dual_level():
        penetration = replace(
            base,
            distance=torch.autograd.forward_ad.make_dual(
                base.distance, tangent_distance
            ),
            position=torch.autograd.forward_ad.make_dual(
                base.position, tangent_position
            ),
            normal=torch.autograd.forward_ad.make_dual(base.normal, tangent_normal),
        )
        packed = enumerated_transmission_topology_pack(
            penetration,
            torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            tx_count=1,
            rx_count=3,
        )
        path_length_tangent = torch.autograd.forward_ad.unpack_dual(
            packed.path_length_m
        ).tangent
        position_tangent = torch.autograd.forward_ad.unpack_dual(
            packed.interaction_positions
        ).tangent
        normal_tangent = torch.autograd.forward_ad.unpack_dual(
            packed.interaction_normals
        ).tangent

        assert path_length_tangent is not None
        assert position_tangent is not None
        assert normal_tangent is not None
        assert path_length_tangent.tolist() == [0.0, 5.0, 0.0]
        assert position_tangent[1, 0].tolist() == tangent_position[1, 0].tolist()
        assert normal_tangent[1, 0].tolist() == tangent_normal[1, 0].tolist()
        assert torch.count_nonzero(position_tangent[[0, 2]]).item() == 0
        assert torch.count_nonzero(normal_tangent[[0, 2]]).item() == 0