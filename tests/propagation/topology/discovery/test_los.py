# Copyright Xingyu Chen.
# Tests los.

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from witwin.channel.interactions.los import (
    LosCandidatePlan,
    prepare_los_candidates,
)


def test_prepare_los_candidates_is_discrete_typed_identity_plan():
    tx_id = torch.tensor([0, 1, 1], dtype=torch.int32)
    rx_id = torch.tensor([2, 0, 1], dtype=torch.int32)

    plan = prepare_los_candidates(
        tx_id=tx_id,
        rx_id=rx_id,
        sequence_width=2,
    )

    assert isinstance(plan, LosCandidatePlan)
    assert [field.name for field in fields(plan)] == [
        "tx_id",
        "rx_id",
        "sequence_width",
        "candidate_count",
    ]
    assert plan.tx_id is tx_id
    assert plan.rx_id is rx_id
    assert plan.sequence_width == 2
    assert plan.candidate_count == 3
    with pytest.raises(FrozenInstanceError):
        plan.sequence_width = 3