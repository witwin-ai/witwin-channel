"""Lazy coupled reflection-diffraction candidate discovery planning."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch


_COUPLED_CANDIDATE_CHUNK_SIZE = 65_536
_MAX_COUPLED_CANDIDATES = 1_000_000


@dataclass(frozen=True, slots=True)
class CoupledCandidatePlan:
    tx_count: int
    rx_count: int
    representative_faces: torch.Tensor
    selected_edges: torch.Tensor
    edge_count: int
    candidates_per_pair: int
    base_candidate_count: int
    theoretical_candidate_count: int
    chunk_size: int


@dataclass(frozen=True, slots=True)
class CoupledCandidateRequest:
    chunk_start: int
    chunk_end: int
    candidate_count: int
    linear: torch.Tensor
    tx_slot: torch.Tensor
    rx_slot: torch.Tensor
    face_id: torch.Tensor
    edge_id: torch.Tensor
    reverse: bool
    component_id: int


def prepare_coupled_candidate_plan(
    *,
    tx_count: int,
    rx_count: int,
    representative_faces: torch.Tensor,
    selected_edges: torch.Tensor,
    candidate_limit: int,
    chunk_size: int = _COUPLED_CANDIDATE_CHUNK_SIZE,
) -> CoupledCandidatePlan:
    tx_count = int(tx_count)
    rx_count = int(rx_count)
    group_count = int(representative_faces.shape[0])
    edge_count = int(selected_edges.shape[0])
    candidates_per_pair = group_count * edge_count
    base_candidate_count = tx_count * rx_count * candidates_per_pair
    theoretical_candidate_count = base_candidate_count * 2
    effective_candidate_limit = min(candidate_limit, _MAX_COUPLED_CANDIDATES)
    if theoretical_candidate_count > effective_candidate_limit:
        raise RuntimeError(
            "coupled reflection-diffraction topology requires "
            f"{theoretical_candidate_count} candidates, exceeding "
            f"coupled_candidate_limit={effective_candidate_limit}"
        )
    return CoupledCandidatePlan(
        tx_count=tx_count,
        rx_count=rx_count,
        representative_faces=representative_faces,
        selected_edges=selected_edges,
        edge_count=edge_count,
        candidates_per_pair=candidates_per_pair,
        base_candidate_count=base_candidate_count,
        theoretical_candidate_count=theoretical_candidate_count,
        chunk_size=int(chunk_size),
    )


def iter_coupled_candidate_requests(
    plan: CoupledCandidatePlan,
    *,
    device: torch.device,
) -> Iterator[CoupledCandidateRequest]:
    for start in range(0, plan.base_candidate_count, plan.chunk_size):
        end = min(start + plan.chunk_size, plan.base_candidate_count)
        linear = torch.arange(start, end, device=device, dtype=torch.int64)
        pair_slot = torch.div(
            linear,
            plan.candidates_per_pair,
            rounding_mode="floor",
        )
        local_slot = torch.remainder(linear, plan.candidates_per_pair)
        tx_slot = torch.div(pair_slot, plan.rx_count, rounding_mode="floor")
        rx_slot = torch.remainder(pair_slot, plan.rx_count)
        face_slot = torch.div(local_slot, plan.edge_count, rounding_mode="floor")
        edge_slot = torch.remainder(local_slot, plan.edge_count)
        face_id = plan.representative_faces[face_slot]
        edge_id = plan.selected_edges[edge_slot]
        candidate_count = int(linear.shape[0])
        for reverse, component_id in ((False, 3), (True, 4)):
            yield CoupledCandidateRequest(
                chunk_start=start,
                chunk_end=end,
                candidate_count=candidate_count,
                linear=linear,
                tx_slot=tx_slot,
                rx_slot=rx_slot,
                face_id=face_id,
                edge_id=edge_id,
                reverse=reverse,
                component_id=component_id,
            )
