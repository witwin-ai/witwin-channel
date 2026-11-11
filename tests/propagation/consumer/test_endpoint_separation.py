# Copyright Xingyu Chen.
# Tests the consumer's coincident source/sink check and its one-read cost.

from __future__ import annotations

import math
import warnings

import pytest
import torch

from witwin.channel.propagation.consumer import (
    EndpointBatch,
    PropagationRequest,
    check_endpoint_separation,
    evaluate,
)
from witwin.channel.scene import compile
from witwin.core import Scene

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

FREQUENCY_HZ = 3.5e9
FAR_FIELD_M = 299_792_458.0 / FREQUENCY_HZ / (4.0 * math.pi)
SOURCE = (0.0, 0.0, 1.5)


def _batch(positions, *, source: bool, first_id: int) -> EndpointBatch:
    count = len(positions)
    basis = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device="cuda")
    return EndpointBatch(
        stable_ids=torch.arange(first_id, first_id + count, device="cuda"),
        positions_m=torch.tensor(positions, device="cuda", dtype=torch.float32),
        polarizations=torch.tensor([[0.0, 0.0, 1.0]] * count, device="cuda"),
        polarization_basis=basis.expand(count, 2, 3).contiguous(),
        powers_w=torch.ones((count,), device="cuda") if source else None,
    )


def _pair(sink_positions):
    return (
        _batch([(5.0, 5.0, 5.0), SOURCE], source=True, first_id=100),
        _batch(sink_positions, source=False, first_id=700),
    )


@pytest.mark.parametrize("offset", [0.0, 1.0e-4], ids=["on_source", "near_source"])
def test_coincident_sink_is_refused_by_index_and_stable_id(offset: float) -> None:
    sources, sinks = _pair([(0.0, 1.0, 1.5), (0.0, offset, 1.5)])
    with pytest.raises(
        ValueError,
        match=r"source 1 \(stable id 101\) .* coincides with sink 1 \(stable id 701\)",
    ):
        check_endpoint_separation(sources, sinks, reference_frequency_hz=FREQUENCY_HZ)


def test_separated_sinks_pass_with_one_synchronization() -> None:
    sources, sinks = _pair(
        [(0.0, 0.25, 1.5), (0.0, 1.01 * FAR_FIELD_M, 1.5), (0.25, 0.25, 1.5)]
    )
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            check_endpoint_separation(sources, sinks, reference_frequency_hz=FREQUENCY_HZ)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
    syncs = [item for item in caught if "called a synchronizing" in str(item.message)]
    assert len(syncs) == 1


def test_sink_just_inside_the_far_field_radius_is_refused() -> None:
    sources, sinks = _pair([(0.0, 0.99 * FAR_FIELD_M, 1.5)])
    with pytest.raises(ValueError, match="coincidence tolerance"):
        check_endpoint_separation(sources, sinks, reference_frequency_hz=FREQUENCY_HZ)


def test_evaluate_does_not_read_endpoints_and_publishes_a_zero_coincident_los_row() -> None:
    """The documented gap the check closes: evaluate itself cannot see the pair."""

    compiled = compile(Scene(), reference_frequency_hz=FREQUENCY_HZ)
    sources, sinks = _pair([(0.0, 1.0, 1.5), SOURCE])
    result = evaluate(
        compiled,
        PropagationRequest(
            sources=sources, sinks=sinks, reference_frequency_hz=FREQUENCY_HZ,
            components=frozenset({"los"}), max_depth=0, response="scalar_transport",
            topology_mode="discover", ad_mode="none",
        ),
    )
    coefficient = result.paths.transport.coefficient
    pair = result.paths.pair_index
    coincident = pair == 1 * 2 + 1
    assert int(coincident.sum()) == 1
    assert float(coefficient[coincident].abs().max()) == 0.0
    assert float(coefficient[~coincident].abs().min()) > 0.0
