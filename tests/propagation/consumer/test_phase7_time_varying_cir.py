"""The time-varying channel impulse response consumer (ADR-041).

A pair's ``delay_s`` and transport already were its impulse response; what was
missing was a time axis that costs one launch rather than one per instant.
These tests pin that the axis changes nothing about the answer - every instant
is bit-for-bit the single-slot reevaluation of that instant - and that it
changes everything about the cost.
"""

from __future__ import annotations

import pytest
import torch

from witwin.channel.constants import C0
from witwin.channel.propagation.consumer import (
    TimeVaryingRequest,
    evaluate_time_varying,
)

from tests.propagation.consumer._multi_endpoint_world import (
    FREQUENCY_HZ,
    FROZEN_PAIR_OFFSETS,
    FROZEN_ROW_COUNT,
    LOS_ROW,
    SINK_POSITIONS,
    SOURCE_POSITIONS,
    SOURCE_POWER_W,
    compiled_world,
    cuda_positions,
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


def _times(slot_count: int, *, step: float = 1.0e-3) -> torch.Tensor:
    return torch.arange(slot_count, dtype=torch.float64) * step


def _cir(compiled, prepared, sources, sinks, times, **overrides):
    arguments = dict(
        sources=sources,
        sinks=sinks,
        reference_frequency_hz=FREQUENCY_HZ,
        topology=prepared,
        times_s=times,
        response="scalar_transport",
        ad_mode="none",
    )
    arguments.update(overrides)
    return evaluate_time_varying(compiled, TimeVaryingRequest(**arguments))


def test_time_varying_cir_matches_a_per_time_reevaluate() -> None:
    compiled = compiled_world()
    prepared = frozen_topology(compiled)
    slot_count = 16
    offsets = slot_offsets(slot_count)
    times = _times(slot_count)

    result = _cir(
        compiled,
        prepared,
        source_batch(stack_over_slots(cuda_positions(SOURCE_POSITIONS), offsets)),
        sink_batch(stack_over_slots(cuda_positions(SINK_POSITIONS), offsets)),
        times,
    )

    assert result.slot_count == slot_count
    assert result.row_count == FROZEN_ROW_COUNT
    assert tuple(result.delay_s.shape) == (slot_count, FROZEN_ROW_COUNT)
    assert tuple(result.transport.coefficient.shape) == (
        slot_count,
        FROZEN_ROW_COUNT,
    )
    assert tuple(result.row_valid.shape) == (slot_count, FROZEN_ROW_COUNT)
    assert result.transport.response == "scalar_transport"
    assert result.transport.field is None and result.transport.matrix is None
    assert result.pair_count == PAIRS_PER_SLOT
    assert result.pair_offsets.tolist() == FROZEN_PAIR_OFFSETS
    # float64 in, the same float64 out - the time labels are never narrowed.
    assert result.times_s.dtype == torch.float64
    assert result.times_s.tolist() == times.tolist()

    for slot in range(slot_count):
        one = replay(
            compiled,
            prepared,
            source_batch(cuda_positions(SOURCE_POSITIONS) + offsets[slot]),
            sink_batch(cuda_positions(SINK_POSITIONS) + offsets[slot]),
        )
        assert torch.equal(result.delay_s[slot], one.paths.geometry.delay_s)
        assert torch.equal(
            result.path_length_m[slot], one.paths.geometry.path_length_m
        )
        assert torch.equal(
            result.transport.coefficient[slot], one.paths.transport.coefficient
        )
        assert torch.equal(result.row_valid[slot], one.row_valid)


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector)


def test_time_varying_cir_is_a_valid_impulse_response() -> None:
    """A straight-line radial trajectory has a closed-form delay and rate."""

    forward_ad = torch.autograd.forward_ad
    compiled = compiled_world()
    prepared = frozen_topology(compiled)
    slot_count = 16
    speed = 12.0
    step_s = 1.0e-3
    source = cuda_positions(SOURCE_POSITIONS)
    sink = cuda_positions(SINK_POSITIONS)
    direction = _unit(sink[0] - source[0])

    times = _times(slot_count, step=step_s)
    displacement = (
        torch.tensor(times.tolist(), dtype=torch.float32, device="cuda").reshape(-1, 1)
        * speed
        * direction.reshape(1, 3)
    )
    positions = (sink.unsqueeze(0) + displacement.unsqueeze(1)).reshape(-1, 3)
    positions = positions.contiguous()
    velocities = torch.zeros_like(positions)
    velocities[LOS_ROW::SLOT_SINKS] = direction * speed

    with forward_ad.dual_level():
        result = _cir(
            compiled,
            prepared,
            source_batch(source.repeat(slot_count, 1)),
            sink_batch(forward_ad.make_dual(positions, velocities)),
            times,
            ad_mode="jvp",
        )
        delay = forward_ad.unpack_dual(result.delay_s)
        tangent = delay.tangent
        primal = delay.primal.detach().to(dtype=torch.float64)

    assert tangent is not None
    # |p(t) - p_tx| / c, computed in float64 from the float32 positions the
    # kernel actually consumed.
    tracked = positions.reshape(slot_count, SLOT_SINKS, 3)[:, 0].to(
        dtype=torch.float64
    )
    reference = torch.linalg.vector_norm(
        tracked - source[0].to(dtype=torch.float64), dim=1
    ) / C0
    measured = primal[:, LOS_ROW]
    assert torch.allclose(measured, reference, rtol=1.0e-6, atol=0.0)

    # The forward tangent is the analytic rate, and consecutive samples of the
    # published delay reproduce it. The finite difference is the ORACLE here;
    # the tangent is the production number (ADR-038).
    analytic_rate = speed / C0
    rate = tangent[:, LOS_ROW].to(dtype=torch.float64)
    assert torch.allclose(
        rate, torch.full_like(rate, analytic_rate), rtol=2.0e-3, atol=0.0
    )
    finite_difference = (measured[1:] - measured[:-1]) / step_s
    assert torch.allclose(
        finite_difference, rate[:-1], rtol=1.0e-3, atol=0.0
    )


def test_time_varying_cir_does_not_reapply_transmit_power() -> None:
    """ADR-039: the scalar transport carries sqrt(powers_w) exactly once.

    The brief asked for a doubled power and a magnitude ratio of exactly 2.0,
    which is the signature of applying the amplitude TWICE. Under ADR-039 the
    amplitude is ``sqrt(powers_w)``, so the request that produces an exact
    factor of two is a QUADRUPLED power; that is what is asserted, and the two
    wrong answers the brief names are still the two this discriminates against:
    1.0 means the declared power never reached the coefficient, 4.0 means it
    reached it twice.
    """

    compiled = compiled_world()
    prepared = frozen_topology(compiled)
    slot_count = 8
    offsets = slot_offsets(slot_count)
    sinks = sink_batch(stack_over_slots(cuda_positions(SINK_POSITIONS), offsets))
    stacked_sources = stack_over_slots(cuda_positions(SOURCE_POSITIONS), offsets)
    times = _times(slot_count)

    base = _cir(
        compiled, prepared, source_batch(stacked_sources), sinks, times
    )
    louder = _cir(
        compiled,
        prepared,
        source_batch(stacked_sources, power_w=4.0 * SOURCE_POWER_W),
        sinks,
        times,
    )

    assert torch.equal(
        louder.transport.coefficient, 2.0 * base.transport.coefficient
    )
    assert torch.equal(louder.delay_s, base.delay_s)

    # The Jones operator is a polarization-basis map, not a transported field,
    # so it is excitation-free and must not move at all.
    def jones(power_w: float):
        return _cir(
            compiled,
            prepared,
            source_batch(stacked_sources, power_w=power_w, with_basis=True),
            sink_batch(
                stack_over_slots(cuda_positions(SINK_POSITIONS), offsets),
                with_basis=True,
            ),
            times,
            response="polarimetric_transport",
        )

    quiet = jones(SOURCE_POWER_W)
    loud = jones(4.0 * SOURCE_POWER_W)
    assert quiet.transport.response == "polarimetric_transport"
    assert tuple(quiet.transport.matrix.shape) == (
        slot_count,
        FROZEN_ROW_COUNT,
        2,
        2,
    )
    assert torch.equal(loud.transport.matrix, quiet.transport.matrix)
    assert quiet.transport.coefficient is None


def test_the_cir_carries_no_second_compaction() -> None:
    compiled = compiled_world()
    prepared = frozen_topology(compiled)

    for slot_count in (1, 32, 128):
        offsets = slot_offsets(slot_count, step=0.001)
        result = _cir(
            compiled,
            prepared,
            source_batch(
                stack_over_slots(cuda_positions(SOURCE_POSITIONS), offsets)
            ),
            sink_batch(stack_over_slots(cuda_positions(SINK_POSITIONS), offsets)),
            _times(slot_count),
        )
        diagnostics = result.diagnostics
        assert diagnostics.compact_count_d2h_copies == 0
        assert diagnostics.compact_count_d2h_bytes == 0
        assert diagnostics.compact_sync_count == 0
        assert diagnostics.discovery_launch_count == 0
        assert diagnostics.validation_d2h_copies == 1
        assert diagnostics.validation_sync_count == 1
        assert result.row_count == prepared.row_count == FROZEN_ROW_COUNT


def test_peak_memory_for_a_long_frame() -> None:
    compiled = compiled_world()
    prepared = frozen_topology(compiled)
    slot_count = 1024
    offsets = slot_offsets(slot_count, step=1.0e-4)
    sources = source_batch(
        stack_over_slots(cuda_positions(SOURCE_POSITIONS), offsets)
    )
    sinks = sink_batch(stack_over_slots(cuda_positions(SINK_POSITIONS), offsets))
    times = _times(slot_count)

    _cir(compiled, prepared, sources, sinks, times)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.max_memory_allocated()
    result = _cir(compiled, prepared, sources, sinks, times)
    torch.cuda.synchronize()
    peak_mb = (torch.cuda.max_memory_allocated() - before) / 2**20

    assert result.slot_count == slot_count
    assert peak_mb < 64.0, f"a 1024-instant frame peaked at {peak_mb:.1f} MB"


def test_the_time_varying_request_rejects_a_malformed_time_axis() -> None:
    compiled = compiled_world()
    prepared = frozen_topology(compiled)
    offsets = slot_offsets(4)
    sources = source_batch(
        stack_over_slots(cuda_positions(SOURCE_POSITIONS), offsets)
    )
    sinks = sink_batch(stack_over_slots(cuda_positions(SINK_POSITIONS), offsets))

    with pytest.raises(TypeError, match="float64"):
        _cir(compiled, prepared, sources, sinks, torch.arange(4, dtype=torch.float32))
    with pytest.raises(ValueError, match="non-empty"):
        _cir(compiled, prepared, sources, sinks, torch.zeros((0,), dtype=torch.float64))
    with pytest.raises(ValueError, match="sources"):
        _cir(compiled, prepared, sources, sinks, _times(3))
    with pytest.raises(TypeError, match="PreparedFixedTopology"):
        _cir(compiled, prepared.topology, sources, sinks, _times(4))
