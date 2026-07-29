# Copyright Xingyu Chen.
# Tests diffraction.

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from witwin.channel.interactions import diffraction


@pytest.mark.parametrize(
    ("metadata", "expected"),
    (
        ({}, False),
        ({"mitsuba": {"merge_shapes": False}}, False),
        ({"mitsuba": {"merge_shapes": True}}, True),
        ({"mitsuba": object()}, False),
    ),
)
def test_prepare_diffraction_order1_plan_preserves_imported_edge_policy(metadata, expected):
    plan = diffraction.prepare_diffraction_order1_plan(
        metadata=metadata,
        tx_count=2,
        rx_count=5,
    )

    assert [field.name for field in fields(plan)] == [
        "preserve_imported_edges",
        "tx_count",
        "rx_count",
    ]
    assert plan.preserve_imported_edges is expected
    assert plan.tx_count == 2
    assert plan.rx_count == 5
    with pytest.raises(FrozenInstanceError):
        plan.rx_count = 6


def test_tx_requests_are_lazy_and_do_not_prefetch_tensor_rows():
    tx_positions = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    plan = diffraction.prepare_diffraction_order1_plan(
        metadata={},
        tx_count=2,
        rx_count=1,
    )
    requests = diffraction.iter_diffraction_tx_requests(
        plan,
        tx_positions=tx_positions,
    )

    first = next(requests)
    tx_positions[1, 0] = 99.0
    second = next(requests)

    assert first.tx_index == 0
    assert first.tx.data_ptr() == tx_positions[0].data_ptr()
    assert first.tx.stride() == tx_positions[0].stride()
    assert second.tx_index == 1
    assert second.tx.data_ptr() == tx_positions[1].data_ptr()
    assert float(second.tx[0]) == 99.0
    with pytest.raises(StopIteration):
        next(requests)


def test_rx_chunk_requests_preserve_state_capacity_and_order(monkeypatch):
    monkeypatch.setattr(diffraction, "_MULTIBOUNCE_PAIR_CHUNK_SIZE", 6)
    plan = diffraction.DiffractionOrder1Plan(
        preserve_imported_edges=False,
        tx_count=1,
        rx_count=5,
    )
    requests = list(
        diffraction.iter_diffraction_rx_chunk_requests(
            plan,
            state_count=2,
        )
    )

    assert [
        (request.rx_start, request.rx_end, request.capacity)
        for request in requests
    ] == [(0, 3, 6), (3, 5, 4)]