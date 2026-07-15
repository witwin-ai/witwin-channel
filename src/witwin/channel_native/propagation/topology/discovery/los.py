"""Discrete line-of-sight candidate planning."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class LosCandidatePlan:
    tx_id: torch.Tensor
    rx_id: torch.Tensor
    sequence_width: int
    candidate_count: int


def prepare_los_candidates(
    *,
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    sequence_width: int,
) -> LosCandidatePlan:
    return LosCandidatePlan(
        tx_id=tx_id,
        rx_id=rx_id,
        sequence_width=int(sequence_width),
        candidate_count=int(tx_id.numel()),
    )
