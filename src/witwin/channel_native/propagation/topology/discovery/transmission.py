"""Discrete straight-segment transmission pair discovery."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class TransmissionPairPlan:
    tx_index: torch.Tensor
    rx_index: torch.Tensor
    pair_count: int
    max_depth: int


@dataclass(frozen=True, slots=True)
class TransmissionActiveRows:
    step: int
    rows: torch.Tensor


@dataclass(frozen=True, slots=True)
class TransmissionWinnerRows:
    chosen: torch.Tensor
    candidate_count: int
    guardrail_count: int


def prepare_transmission_pair_plan(
    *,
    tx_count: int,
    rx_count: int,
    max_depth: int,
    device: torch.device,
) -> TransmissionPairPlan:
    tx_index = torch.arange(
        tx_count, device=device, dtype=torch.int64
    ).repeat_interleave(rx_count)
    rx_index = torch.arange(rx_count, device=device, dtype=torch.int64).repeat(
        tx_count
    )
    return TransmissionPairPlan(
        tx_index=tx_index,
        rx_index=rx_index,
        pair_count=int(tx_count) * int(rx_count),
        max_depth=int(max_depth),
    )


def iter_transmission_active_rows(
    plan: TransmissionPairPlan,
    *,
    done: torch.Tensor,
    invalid: torch.Tensor,
) -> Iterator[TransmissionActiveRows]:
    # The final probe distinguishes a clear segment behind the last accepted
    # wall from a chain that exceeds the depth cap.
    for step in range(plan.max_depth + 1):
        rows = torch.nonzero(~done & ~invalid, as_tuple=False).reshape(-1)
        if int(rows.shape[0]) == 0:
            break
        yield TransmissionActiveRows(step=step, rows=rows)


def select_transmission_winner_rows(
    *,
    done: torch.Tensor,
    invalid: torch.Tensor,
    depth_count: torch.Tensor,
    bad_material: torch.Tensor,
) -> TransmissionWinnerRows:
    penetrated = done & ~invalid & (depth_count >= 1)
    selected = penetrated & ~bad_material
    candidate_count = int((invalid | penetrated).sum())
    guardrail_count = int((invalid | (penetrated & bad_material)).sum())
    return TransmissionWinnerRows(
        chosen=torch.nonzero(selected, as_tuple=False).reshape(-1),
        candidate_count=candidate_count,
        guardrail_count=guardrail_count,
    )
