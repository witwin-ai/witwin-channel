from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from witwin.channel_native.propagation.topology.discovery import transmission


def test_pair_plan_preserves_tx_major_rx_minor_order():
    plan = transmission.prepare_transmission_pair_plan(
        tx_count=2,
        rx_count=3,
        max_depth=2,
        device=torch.device("cpu"),
    )

    assert [field.name for field in fields(plan)] == [
        "tx_index",
        "rx_index",
        "pair_count",
        "max_depth",
    ]
    assert plan.tx_index.tolist() == [0, 0, 0, 1, 1, 1]
    assert plan.rx_index.tolist() == [0, 1, 2, 0, 1, 2]
    assert plan.tx_index.dtype == torch.int64
    assert plan.rx_index.dtype == torch.int64
    assert plan.pair_count == 6
    assert plan.max_depth == 2
    with pytest.raises(FrozenInstanceError):
        plan.max_depth = 3


def test_active_rows_are_lazy_and_observe_mutated_march_state():
    plan = transmission.prepare_transmission_pair_plan(
        tx_count=1,
        rx_count=3,
        max_depth=2,
        device=torch.device("cpu"),
    )
    done = torch.zeros(3, dtype=torch.bool)
    invalid = torch.zeros(3, dtype=torch.bool)
    requests = transmission.iter_transmission_active_rows(
        plan,
        done=done,
        invalid=invalid,
    )

    first = next(requests)
    done[1] = True
    invalid[2] = True
    second = next(requests)
    done[0] = True

    assert first.step == 0
    assert first.rows.tolist() == [0, 1, 2]
    assert second.step == 1
    assert second.rows.tolist() == [0]
    with pytest.raises(StopIteration):
        next(requests)


def test_active_rows_allow_exactly_max_depth_plus_one_probes():
    plan = transmission.prepare_transmission_pair_plan(
        tx_count=1,
        rx_count=1,
        max_depth=2,
        device=torch.device("cpu"),
    )
    requests = list(
        transmission.iter_transmission_active_rows(
            plan,
            done=torch.zeros(1, dtype=torch.bool),
            invalid=torch.zeros(1, dtype=torch.bool),
        )
    )

    assert [request.step for request in requests] == [0, 1, 2]


def test_winner_rows_keep_los_exclusive_and_count_guardrails():
    winners = transmission.select_transmission_winner_rows(
        done=torch.tensor([True, True, False, True]),
        invalid=torch.tensor([False, False, True, False]),
        depth_count=torch.tensor([1, 0, 1, 1]),
        bad_material=torch.tensor([False, False, False, True]),
    )

    assert [field.name for field in fields(winners)] == [
        "chosen",
        "candidate_count",
        "guardrail_count",
    ]
    assert winners.chosen.tolist() == [0]
    assert winners.candidate_count == 3
    assert winners.guardrail_count == 2
    with pytest.raises(FrozenInstanceError):
        winners.candidate_count = 4
