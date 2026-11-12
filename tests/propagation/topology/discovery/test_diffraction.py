# Copyright Xingyu Chen.
# Tests diffraction.

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from witwin.channel.interactions import diffraction


def test_prepare_diffraction_order1_plan_carries_only_endpoint_counts():
    plan = diffraction.prepare_diffraction_order1_plan(
        tx_count=2,
        rx_count=5,
    )

    assert [field.name for field in fields(plan)] == [
        "tx_count",
        "rx_count",
    ]
    assert plan.tx_count == 2
    assert plan.rx_count == 5
    with pytest.raises(FrozenInstanceError):
        plan.rx_count = 6


def test_rx_chunk_requests_preserve_state_capacity_and_order(monkeypatch):
    monkeypatch.setattr(diffraction, "_MULTIBOUNCE_PAIR_CHUNK_SIZE", 6)
    plan = diffraction.DiffractionOrder1Plan(
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