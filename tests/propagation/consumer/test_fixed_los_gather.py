from __future__ import annotations

import pytest
import torch

from witwin.channel.propagation.consumer import EndpointBatch, PropagationTopology
from witwin.channel.propagation.consumer import fixed_los_gather


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _topology(
    source_index: torch.Tensor,
    sink_index: torch.Tensor,
    *,
    source_ids: tuple[int, ...] = (101, 101, 303, 303),
    sink_ids: tuple[int, ...] = (707, 707, 707, 909),
    depth: torch.Tensor | None = None,
    component_id: torch.Tensor | None = None,
) -> PropagationTopology:
    device = source_index.device
    rows = int(source_index.shape[0])
    empty_sequence = torch.empty((rows, 0), device=device, dtype=torch.int32)
    return PropagationTopology(
        source_index=source_index,
        sink_index=sink_index,
        source_id=torch.tensor(source_ids, device=device, dtype=torch.int64),
        sink_id=torch.tensor(sink_ids, device=device, dtype=torch.int64),
        depth=(
            torch.zeros((rows,), device=device, dtype=torch.int32)
            if depth is None
            else depth
        ),
        component_id=(
            torch.zeros((rows,), device=device, dtype=torch.int32)
            if component_id is None
            else component_id
        ),
        primitive_id=torch.full((rows,), -1, device=device, dtype=torch.int32),
        edge_id=torch.full((rows,), -1, device=device, dtype=torch.int32),
        material_id=torch.full((rows,), -1, device=device, dtype=torch.int32),
        primitive_sequence=empty_sequence,
        material_sequence=empty_sequence,
        interaction_type=empty_sequence,
    )


def _endpoints(
    source_positions: torch.Tensor | None = None,
    sink_positions: torch.Tensor | None = None,
    source_powers: torch.Tensor | None = None,
    source_polarizations: torch.Tensor | None = None,
    sink_polarizations: torch.Tensor | None = None,
) -> tuple[EndpointBatch, EndpointBatch]:
    device = torch.device("cuda")
    if source_positions is None:
        source_positions = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            device=device,
            dtype=torch.float32,
        )
    if sink_positions is None:
        sink_positions = torch.tensor(
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            device=device,
            dtype=torch.float32,
        )
    if source_powers is None:
        source_powers = torch.tensor(
            [13.0, 14.0], device=device, dtype=torch.float32
        )
    if source_polarizations is None:
        source_polarizations = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            device=device,
            dtype=torch.float32,
        )
    if sink_polarizations is None:
        sink_polarizations = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        )
    return (
        EndpointBatch(
            stable_ids=torch.tensor(
                [101, 303], device=device, dtype=torch.int64
            ),
            positions_m=source_positions,
            polarizations=source_polarizations,
            powers_w=source_powers,
        ),
        EndpointBatch(
            stable_ids=torch.tensor(
                [707, 909], device=device, dtype=torch.int64
            ),
            positions_m=sink_positions,
            polarizations=sink_polarizations,
        ),
    )


def _canonical_topology() -> PropagationTopology:
    return _topology(
        torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.int32),
        torch.tensor([0, 0, 0, 1], device="cuda", dtype=torch.int32),
    )


def test_fixed_los_gather_returns_exact_k_rows_and_pair_offsets() -> None:
    topology = _canonical_topology()
    sources, sinks = _endpoints()

    rows = fixed_los_gather(topology, sources, sinks)

    assert rows.row_count == 4
    assert rows.source.shape == (4, 3)
    assert rows.target.shape == (4, 3)
    assert torch.equal(
        rows.source,
        torch.tensor(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
            device="cuda",
        ),
    )
    assert torch.equal(
        rows.pair_index,
        torch.tensor([0, 0, 1, 3], device="cuda", dtype=torch.int64),
    )
    assert torch.equal(
        rows.pair_offsets,
        torch.tensor([0, 2, 3, 3, 4], device="cuda", dtype=torch.int64),
    )
    assert rows.validation_d2h_copies == 1
    assert rows.validation_d2h_bytes == 4
    assert rows.validation_synchronizations == 1


def test_fixed_los_gather_vjp_scatter_adds_duplicate_endpoint_rows() -> None:
    topology = _canonical_topology()
    source_positions = torch.arange(
        6, device="cuda", dtype=torch.float32
    ).reshape(2, 3).requires_grad_()
    sink_positions = torch.arange(
        6, 12, device="cuda", dtype=torch.float32
    ).reshape(2, 3).requires_grad_()
    source_powers = torch.ones(2, device="cuda", requires_grad=True)
    source_polarizations = torch.eye(
        3, device="cuda", dtype=torch.float32
    )[:2].clone().requires_grad_()
    sink_polarizations = torch.eye(
        3, device="cuda", dtype=torch.float32
    )[1:].clone().requires_grad_()
    sources, sinks = _endpoints(
        source_positions,
        sink_positions,
        source_powers,
        source_polarizations,
        sink_polarizations,
    )

    rows = fixed_los_gather(topology, sources, sinks)
    loss = (
        rows.source.sum()
        + 2.0 * rows.target.sum()
        + 3.0 * rows.tx_power.sum()
        + 4.0 * rows.tx_polarization.sum()
        + 5.0 * rows.rx_polarization.sum()
    )
    loss.backward()

    assert torch.equal(source_positions.grad, torch.full_like(source_positions, 2.0))
    assert torch.equal(
        sink_positions.grad,
        torch.tensor(
            [[6.0, 6.0, 6.0], [2.0, 2.0, 2.0]], device="cuda"
        ),
    )
    assert torch.equal(source_powers.grad, torch.tensor([6.0, 6.0], device="cuda"))
    assert torch.equal(
        source_polarizations.grad,
        torch.full_like(source_polarizations, 8.0),
    )
    assert torch.equal(
        sink_polarizations.grad,
        torch.tensor(
            [[15.0, 15.0, 15.0], [5.0, 5.0, 5.0]], device="cuda"
        ),
    )


def test_fixed_los_gather_jvp_uses_native_endpoint_gather() -> None:
    topology = _canonical_topology()
    sources, sinks = _endpoints()
    primals = (
        sources.positions_m,
        sinks.positions_m,
        sources.powers_w,
        sources.polarizations,
        sinks.polarizations,
    )
    assert primals[2] is not None
    tangents = (
        torch.arange(6, device="cuda", dtype=torch.float32).reshape(2, 3),
        torch.arange(6, 12, device="cuda", dtype=torch.float32).reshape(2, 3),
        torch.tensor([2.0, 3.0], device="cuda"),
        torch.full((2, 3), 4.0, device="cuda"),
        torch.tensor(
            [[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], device="cuda"
        ),
    )

    def evaluate(source_pos, sink_pos, powers, source_pol, sink_pol):
        source_batch, sink_batch = _endpoints(
            source_pos,
            sink_pos,
            powers,
            source_pol,
            sink_pol,
        )
        rows = fixed_los_gather(topology, source_batch, sink_batch)
        return (
            rows.source,
            rows.target,
            rows.tx_power,
            rows.tx_polarization,
            rows.rx_polarization,
        )

    _, tangent_rows = torch.func.jvp(evaluate, primals, tangents)

    assert torch.equal(
        tangent_rows[0],
        torch.tensor(
            [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [3.0, 4.0, 5.0]],
            device="cuda",
        ),
    )
    assert torch.equal(
        tangent_rows[1],
        torch.tensor(
            [
                [6.0, 7.0, 8.0],
                [6.0, 7.0, 8.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
            ],
            device="cuda",
        ),
    )
    assert torch.equal(
        tangent_rows[2], torch.tensor([2.0, 2.0, 3.0, 3.0], device="cuda")
    )
    assert torch.equal(tangent_rows[3], torch.full((4, 3), 4.0, device="cuda"))
    assert torch.equal(
        tangent_rows[4],
        torch.tensor(
            [
                [5.0, 6.0, 7.0],
                [5.0, 6.0, 7.0],
                [5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0],
            ],
            device="cuda",
        ),
    )


@pytest.mark.parametrize(
    ("source_index", "sink_index", "source_ids", "sink_ids", "depth", "component"),
    [
        ([0, 2], [0, 0], (101, 303), (707, 707), [0, 0], [0, 0]),
        ([1, 0], [0, 0], (303, 101), (707, 707), [0, 0], [0, 0]),
        ([0, 1], [0, 0], (101, 303), (707, 707), [0, 1], [0, 0]),
        ([0, 1], [0, 0], (101, 303), (707, 707), [0, 0], [0, 1]),
        ([0, 1], [0, 0], (999, 303), (707, 707), [0, 0], [0, 0]),
    ],
)
def test_fixed_los_gather_rejects_invalid_frozen_topology_before_result(
    source_index,
    sink_index,
    source_ids,
    sink_ids,
    depth,
    component,
) -> None:
    topology = _topology(
        torch.tensor(source_index, device="cuda", dtype=torch.int32),
        torch.tensor(sink_index, device="cuda", dtype=torch.int32),
        source_ids=source_ids,
        sink_ids=sink_ids,
        depth=torch.tensor(depth, device="cuda", dtype=torch.int32),
        component_id=torch.tensor(component, device="cuda", dtype=torch.int32),
    )
    sources, sinks = _endpoints()

    with pytest.raises(RuntimeError, match="fixed LoS topology validation failed"):
        fixed_los_gather(topology, sources, sinks)


def test_fixed_los_gather_accepts_zero_rows_without_validation_sync() -> None:
    device = torch.device("cuda")
    topology = _topology(
        torch.empty((0,), device=device, dtype=torch.int32),
        torch.empty((0,), device=device, dtype=torch.int32),
        source_ids=(),
        sink_ids=(),
    )
    sources, sinks = _endpoints()

    rows = fixed_los_gather(topology, sources, sinks)

    assert rows.source.shape == (0, 3)
    assert rows.target.shape == (0, 3)
    assert rows.tx_power.shape == (0,)
    assert rows.pair_index.shape == (0,)
    assert torch.equal(rows.pair_offsets, torch.zeros(5, device="cuda", dtype=torch.int64))
    assert rows.validation_d2h_copies == 0
    assert rows.validation_d2h_bytes == 0
    assert rows.validation_synchronizations == 0
