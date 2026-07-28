"""Block-diagonal slot batching for fixed-topology reevaluation (ADR-041).

Before this contract a caller who needed the same frozen rows at ``T`` world
instants had two options, and both were wrong. A Python loop over instants pays
``T`` validation copies and ``T`` synchronizations for a capability whose whole
point is to pay one. Stacking the instants into the endpoint batches under the
old pairing law makes ``pair_count`` the full ``(T*S) x (T*K)`` outer product,
which is quadratic in the number of instants and pairs endpoints that never
coexist.

These tests pin the third option: one launch per bucket, one validation copy,
one synchronization, and a pair count linear in the slot count - and pin that
it is bit-for-bit the answer the loop would have given.
"""

from __future__ import annotations

import pytest
import torch

from witwin.channel.constants import C0
from witwin.channel.propagation.consumer import (
    FixedTopologyRequest,
    PropagationConvention,
    capabilities,
    prepare_fixed_topology,
    replicate_over_slots,
)

from tests.propagation.consumer._multi_endpoint_world import (
    FREQUENCY_HZ,
    FROZEN_PAIR_OFFSETS,
    FROZEN_ROW_COUNT,
    LOS_ROW,
    SINK_POSITIONS,
    SOURCE_POSITIONS,
    compiled_world,
    cuda_positions,
    discover,
    frozen_topology,
    replay,
    sink_batch,
    slot_offsets,
    source_batch,
    stack_over_slots,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

SLOT_SOURCES = len(SOURCE_POSITIONS)
SLOT_SINKS = len(SINK_POSITIONS)
PAIRS_PER_SLOT = SLOT_SOURCES * SLOT_SINKS


def _stacked(slot_count: int, *, step: float = 0.01):
    offsets = slot_offsets(slot_count, step=step)
    return (
        source_batch(stack_over_slots(cuda_positions(SOURCE_POSITIONS), offsets)),
        sink_batch(stack_over_slots(cuda_positions(SINK_POSITIONS), offsets)),
        offsets,
    )


def _replicated(prepared, slot_count: int):
    return replicate_over_slots(
        prepared, slot_count, source_count=SLOT_SOURCES, sink_count=SLOT_SINKS
    )


def _single_slot(compiled, prepared, offset: torch.Tensor):
    return replay(
        compiled,
        prepared,
        source_batch(cuda_positions(SOURCE_POSITIONS) + offset),
        sink_batch(cuda_positions(SINK_POSITIONS) + offset),
    )


def test_slot_batching_is_bitwise_identical_to_a_per_slot_loop() -> None:
    compiled = compiled_world()
    prepared = frozen_topology(compiled)
    assert prepared.row_count == FROZEN_ROW_COUNT
    slot_count = 8
    sources, sinks, offsets = _stacked(slot_count)

    batched = replay(
        compiled, _replicated(prepared, slot_count), sources, sinks,
        slot_count=slot_count,
    )

    # Every row is an independent evaluation of its own endpoints, so exact
    # equality is the correct assertion. A tolerance here would hide a real
    # reduction-order or gather defect.
    for slot in range(slot_count):
        one = _single_slot(compiled, prepared, offsets[slot])
        rows = slice(slot * FROZEN_ROW_COUNT, (slot + 1) * FROZEN_ROW_COUNT)
        assert torch.equal(
            batched.paths.geometry.delay_s[rows], one.paths.geometry.delay_s
        )
        assert torch.equal(
            batched.paths.geometry.path_length_m[rows],
            one.paths.geometry.path_length_m,
        )
        assert torch.equal(
            batched.paths.transport.coefficient[rows].real,
            one.paths.transport.coefficient.real,
        )
        assert torch.equal(
            batched.paths.transport.coefficient[rows].imag,
            one.paths.transport.coefficient.imag,
        )
        assert torch.equal(batched.row_valid[rows], one.row_valid)


def test_slot_pairing_is_block_diagonal() -> None:
    compiled = compiled_world()
    prepared = frozen_topology(compiled)
    single = replay(
        compiled,
        prepared,
        source_batch(cuda_positions(SOURCE_POSITIONS)),
        sink_batch(cuda_positions(SINK_POSITIONS)),
    )
    assert single.paths.pair_count == PAIRS_PER_SLOT
    assert single.paths.pair_offsets.tolist() == FROZEN_PAIR_OFFSETS

    for slot_count in (1, 8, 64, 256):
        sources, sinks, _offsets = _stacked(slot_count)
        result = replay(
            compiled, _replicated(prepared, slot_count), sources, sinks,
            slot_count=slot_count,
        )
        # Linear, not quadratic: the old law would publish
        # (slot_count*2) * (slot_count*2) pairs here.
        assert result.paths.pair_count == slot_count * PAIRS_PER_SLOT
        assert result.paths.pair_count == slot_count * single.paths.pair_count
        offsets = result.paths.pair_offsets
        assert offsets.numel() == slot_count * PAIRS_PER_SLOT + 1
        # The fixture's empty pair segments - the second source publishes no
        # row - reappear once per slot, in the same positions.
        per_slot = torch.tensor(FROZEN_PAIR_OFFSETS, device=offsets.device)
        for slot in range(min(slot_count, 8)):
            start = slot * PAIRS_PER_SLOT
            segment = offsets[start : start + PAIRS_PER_SLOT + 1]
            assert torch.equal(
                segment - segment[0], per_slot
            ), f"slot {slot} segmentation differs"


def test_the_budget_is_flat_in_slot_count() -> None:
    compiled = compiled_world()
    prepared = frozen_topology(compiled)

    for slot_count in (1, 16, 64, 256, 1024):
        sources, sinks, _offsets = _stacked(slot_count, step=0.001)
        result = replay(
            compiled, _replicated(prepared, slot_count), sources, sinks,
            slot_count=slot_count,
        )
        diagnostics = result.diagnostics
        assert diagnostics.validation_d2h_copies == 1
        assert diagnostics.validation_d2h_bytes == 4
        assert diagnostics.validation_sync_count == 1
        assert diagnostics.compact_count_d2h_copies == 0
        assert diagnostics.compact_count_d2h_bytes == 0
        assert diagnostics.compact_sync_count == 0
        assert diagnostics.discovery_launch_count == 0
        assert result.paths.path_count == slot_count * FROZEN_ROW_COUNT


def test_bucket_count_is_unchanged_by_replication() -> None:
    compiled = compiled_world()
    prepared = frozen_topology(compiled)
    replicated = _replicated(prepared, 64)

    # The bucket count IS the native launch count of a replay, so a batched
    # frame must not add one.
    assert len(replicated.buckets) == len(prepared.buckets)
    assert [bucket.component for bucket in replicated.buckets] == [
        bucket.component for bucket in prepared.buckets
    ]
    assert [bucket.depth for bucket in replicated.buckets] == [
        bucket.depth for bucket in prepared.buckets
    ]
    for original, tiled in zip(prepared.buckets, replicated.buckets, strict=True):
        assert tiled.row_count == 64 * original.row_count
        rows = tiled.rows
        assert torch.equal(rows, rows.sort().values), "bucket rows stay ascending"
        assert torch.equal(rows[: original.row_count], original.rows)
    assert replicated.provenance is prepared.provenance
    assert replicated.row_count == 64 * prepared.row_count


def test_replicating_a_single_slot_returns_the_same_handle() -> None:
    compiled = compiled_world()
    prepared = frozen_topology(compiled)

    assert _replicated(prepared, 1) is prepared


def test_row_validity_is_per_slot_and_inert_at_input() -> None:
    compiled = compiled_world()
    prepared = frozen_topology(compiled)
    slot_count = 6
    blocked_slot = 3
    offsets = slot_offsets(slot_count, step=0.01)
    sinks = cuda_positions(SINK_POSITIONS).unsqueeze(0).repeat(slot_count, 1, 1)
    sinks = sinks + offsets.unsqueeze(1)
    # Slot 3 only: put the first sink behind the wall at x = 4, so its
    # line-of-sight row stops existing while every other slot is untouched.
    sinks[blocked_slot, 0] = cuda_positions([[6.0, 0.4, 0.0]])[0]
    sink_positions = sinks.reshape(-1, 3).contiguous()
    sources = source_batch(
        stack_over_slots(cuda_positions(SOURCE_POSITIONS), offsets)
    )

    result = replay(
        compiled,
        _replicated(prepared, slot_count),
        sources,
        sink_batch(sink_positions),
        slot_count=slot_count,
    )
    valid = result.row_valid.reshape(slot_count, FROZEN_ROW_COUNT)
    assert not bool(valid[blocked_slot, LOS_ROW])

    dead = (~result.row_valid).nonzero(as_tuple=False).reshape(-1)
    assert dead.numel() > 0
    zeros = torch.zeros_like(dead, dtype=torch.float32)
    # Inert at the input, not post-masked: the kernel that owns the value
    # produced these zeros.
    assert torch.equal(result.paths.geometry.delay_s[dead], zeros)
    assert torch.equal(result.paths.geometry.path_length_m[dead], zeros)
    assert torch.equal(
        result.paths.transport.coefficient[dead],
        torch.zeros_like(dead, dtype=torch.complex64),
    )

    for slot in range(slot_count):
        if slot == blocked_slot:
            continue
        one = _single_slot(compiled, prepared, offsets[slot])
        rows = slice(slot * FROZEN_ROW_COUNT, (slot + 1) * FROZEN_ROW_COUNT)
        assert torch.equal(
            result.paths.geometry.delay_s[rows], one.paths.geometry.delay_s
        )
        assert torch.equal(
            result.paths.transport.coefficient[rows],
            one.paths.transport.coefficient,
        )
        assert torch.equal(result.row_valid[rows], one.row_valid)


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector)


def test_forward_duals_survive_slot_replication() -> None:
    """ADR-038: a dead tangent looks exactly like a correct lateral zero.

    The radial slot exists so the test can tell those two apart at all; the
    lateral slot exists so a tangent that is merely noisy cannot pass.
    """

    forward_ad = torch.autograd.forward_ad
    compiled = compiled_world()
    prepared = frozen_topology(compiled)
    slot_count = 2
    speed = 12.0
    source = cuda_positions(SOURCE_POSITIONS)
    sink = cuda_positions(SINK_POSITIONS)
    line_of_sight = _unit(sink[0] - source[0])

    positions = sink.repeat(slot_count, 1)
    velocities = torch.zeros_like(positions)
    velocities[0] = line_of_sight * speed          # slot 0: receding radially
    velocities[SLOT_SINKS] = cuda_positions([[0.0, 0.0, speed]])[0]  # slot 1: lateral

    with forward_ad.dual_level():
        dual = forward_ad.make_dual(positions, velocities)
        # Slot replication is a gather on the dual tensor itself, never a
        # rebuild from Python values, which is what keeps the tangent alive.
        dual = dual.index_select(
            0, torch.arange(positions.shape[0], device=positions.device)
        )
        result = replay(
            compiled,
            _replicated(prepared, slot_count),
            source_batch(source.repeat(slot_count, 1)),
            sink_batch(dual),
            slot_count=slot_count,
            ad_mode="jvp",
        )
        tangent = forward_ad.unpack_dual(result.paths.geometry.delay_s).tangent

    assert tangent is not None, "the slot-replicated delay carries no tangent"
    radial = float(tangent[LOS_ROW])
    lateral = float(tangent[FROZEN_ROW_COUNT + LOS_ROW])

    analytic_rate = speed / C0
    assert abs(radial - analytic_rate) / analytic_rate < 2.0e-3
    doppler_hz = -FREQUENCY_HZ * radial
    analytic_doppler = -FREQUENCY_HZ * analytic_rate
    assert abs(doppler_hz - analytic_doppler) / abs(analytic_doppler) < 2.0e-3
    assert doppler_hz < 0.0, "a receding sink must Doppler-shift downward"
    # Exactly zero, not merely small: the velocity has no component along the
    # line of sight, so the correct tangent is the zero float.
    assert lateral == 0.0


def test_slot_count_validation_fails_before_native_work() -> None:
    compiled = compiled_world()
    prepared = frozen_topology(compiled)
    slot_count = 4
    sources, sinks, _offsets = _stacked(slot_count)
    replicated = _replicated(prepared, slot_count)

    def request(**overrides):
        arguments = dict(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=FREQUENCY_HZ,
            topology=replicated,
            response="scalar_transport",
            ad_mode="none",
            slot_count=slot_count,
        )
        arguments.update(overrides)
        return FixedTopologyRequest(**arguments)

    for bad in (0, -1):
        with pytest.raises(ValueError, match="slot_count"):
            request(slot_count=bad)

    with pytest.raises(ValueError, match="sources"):
        request(slot_count=3)

    odd_sinks = sink_batch(
        stack_over_slots(cuda_positions(SINK_POSITIONS), slot_offsets(5))
    )
    with pytest.raises(ValueError, match="sinks"):
        request(sinks=odd_sinks)

    with pytest.raises(ValueError, match="topology rows"):
        request(topology=_replicated(prepared, 3))

    # Every refusal above happens in the request constructor, so no scene, no
    # kernel, and no allocation is ever reached.
    assert compiled.reference_frequency_hz == FREQUENCY_HZ


def test_the_raw_route_refuses_slot_batching_by_name() -> None:
    """The raw LoS pairing law lives inside a native gather this stage owns no
    change to, so slot batching is refused there rather than approximated."""

    compiled = compiled_world()
    sources = source_batch(cuda_positions(SOURCE_POSITIONS))
    sinks = sink_batch(cuda_positions(SINK_POSITIONS))
    raw = discover(compiled, sources, sinks).paths.topology

    with pytest.raises(NotImplementedError, match="prepare_fixed_topology"):
        FixedTopologyRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=FREQUENCY_HZ,
            topology=raw,
            response="scalar_transport",
            ad_mode="none",
            slot_count=2,
        )


def test_a_row_that_pairs_across_slots_is_rejected() -> None:
    """The block-diagonal law is enforced, not assumed."""

    from witwin.channel.propagation.consumer import PropagationTopology

    compiled = compiled_world()
    prepared = frozen_topology(compiled)
    slot_count = 2
    sources, sinks, _offsets = _stacked(slot_count)
    replicated = _replicated(prepared, slot_count)
    topology = replicated.topology
    # Move the last slot's first row onto slot 0's sink while leaving its
    # source in slot 1. Its stable id still matches, so only the block law can
    # catch it.
    broken_sink_index = topology.sink_index.clone()
    broken_sink_index[FROZEN_ROW_COUNT] = 0
    broken = PropagationTopology(
        source_index=topology.source_index,
        sink_index=broken_sink_index,
        source_id=topology.source_id,
        sink_id=topology.sink_id,
        depth=topology.depth,
        component_id=topology.component_id,
        primitive_id=topology.primitive_id,
        edge_id=topology.edge_id,
        material_id=topology.material_id,
        primitive_sequence=topology.primitive_sequence,
        material_sequence=topology.material_sequence,
        interaction_type=topology.interaction_type,
        provenance=topology.provenance,
    )
    handle = prepare_fixed_topology(broken)

    with pytest.raises(ValueError, match="frozen topology validation failed"):
        replay(compiled, handle, sources, sinks, slot_count=slot_count)


def test_the_contract_declares_the_slot_layout() -> None:
    convention = PropagationConvention()
    record = capabilities()

    assert record.supports_slot_batching is True
    assert record.max_slot_count is None
    assert "block_diagonal_slots" in convention.slot_pair_layout
    assert "pair_count=slot_count" in convention.slot_pair_layout
    # The single-slot law is stated separately and is not redefined.
    assert convention.pair_layout.startswith("sink_major_source_minor")
