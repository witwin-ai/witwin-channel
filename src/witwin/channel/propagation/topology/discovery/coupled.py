"""Lazy coupled reflection-diffraction candidate discovery planning."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch


_COUPLED_CANDIDATE_CHUNK_SIZE = 65_536
# cid-7 (D->D) uses a larger chunk so the ~1260-ordered-pair stream collapses to
# a single native launch per receiver block instead of ~8 (ADR-013 G-H runtime
# gate). The R->D / D->R stream stays at 65_536 so its cid-3/4 row identity is
# byte-identical to the frozen P1 baseline; only the cid-7 order (not yet frozen)
# depends on this constant, and it is preserved because the linear candidate
# order is chunk-size independent. Peak transient: one 1M-candidate DD chunk
# materializes ~30 float32 fields/candidate (~100 MB), well within budget; the
# streaming block budget (<=1M candidates/block) caps a block's DD stream below
# this size, so a block never splits its DD stream.
_COUPLED_DD_CANDIDATE_CHUNK_SIZE = 1_048_576
_MAX_COUPLED_CANDIDATES = 1_000_000


@dataclass(frozen=True, slots=True)
class CoupledCandidatePlan:
    tx_count: int
    rx_count: int
    representative_faces: torch.Tensor
    selected_edges: torch.Tensor
    edge_count: int
    candidates_per_pair: int
    dd_candidates_per_pair: int
    base_candidate_count: int
    dd_base_candidate_count: int
    theoretical_candidate_count: int
    chunk_size: int
    dd_chunk_size: int


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


@dataclass(frozen=True, slots=True)
class CoupledDdCandidateRequest:
    chunk_start: int
    chunk_end: int
    candidate_count: int
    linear: torch.Tensor
    tx_slot: torch.Tensor
    rx_slot: torch.Tensor
    edge1_id: torch.Tensor
    edge2_id: torch.Tensor
    component_id: int


def prepare_coupled_candidate_plan(
    *,
    tx_count: int,
    rx_count: int,
    representative_faces: torch.Tensor,
    selected_edges: torch.Tensor,
    candidate_limit: int,
    chunk_size: int = _COUPLED_CANDIDATE_CHUNK_SIZE,
    dd_chunk_size: int = _COUPLED_DD_CANDIDATE_CHUNK_SIZE,
) -> CoupledCandidatePlan:
    tx_count = int(tx_count)
    rx_count = int(rx_count)
    group_count = int(representative_faces.shape[0])
    edge_count = int(selected_edges.shape[0])
    candidates_per_pair = group_count * edge_count
    # cid 7 (double diffraction) enumerates ordered edge pairs e1 != e2 by
    # index; collinear geometric duplicates are the native kernel's job. One
    # direction only: the ordered pair already covers both traversals.
    dd_candidates_per_pair = edge_count * (edge_count - 1)
    base_candidate_count = tx_count * rx_count * candidates_per_pair
    dd_base_candidate_count = tx_count * rx_count * dd_candidates_per_pair
    # The budget must count the whole coupled union that a block evaluates:
    # both R->D / D->R directions (x2) plus the one-direction D->D stream
    # (ADR-013 D1: per-receiver = tx*(2*groups*edges + edges*(edges-1))).
    theoretical_candidate_count = base_candidate_count * 2 + dd_base_candidate_count
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
        dd_candidates_per_pair=dd_candidates_per_pair,
        base_candidate_count=base_candidate_count,
        dd_base_candidate_count=dd_base_candidate_count,
        theoretical_candidate_count=theoretical_candidate_count,
        chunk_size=int(chunk_size),
        dd_chunk_size=int(dd_chunk_size),
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


def iter_coupled_dd_candidate_requests(
    plan: CoupledCandidatePlan,
    *,
    device: torch.device,
) -> Iterator[CoupledDdCandidateRequest]:
    """Stream cid-7 ordered edge-pair candidates (e1 != e2 by index).

    The candidate space is (tx, rx, e1, e2) with e1 != e2, giving
    ``edge_count*(edge_count-1)`` ordered pairs per (tx, rx) pair and one
    direction only. Geometric collinearity (edges on the same physical line) is
    the native kernel's responsibility, not this index-level enumeration.
    """

    stride = plan.edge_count - 1
    for start in range(0, plan.dd_base_candidate_count, plan.dd_chunk_size):
        end = min(start + plan.dd_chunk_size, plan.dd_base_candidate_count)
        linear = torch.arange(start, end, device=device, dtype=torch.int64)
        pair_slot = torch.div(
            linear,
            plan.dd_candidates_per_pair,
            rounding_mode="floor",
        )
        local_slot = torch.remainder(linear, plan.dd_candidates_per_pair)
        tx_slot = torch.div(pair_slot, plan.rx_count, rounding_mode="floor")
        rx_slot = torch.remainder(pair_slot, plan.rx_count)
        first_slot = torch.div(local_slot, stride, rounding_mode="floor")
        remainder_slot = torch.remainder(local_slot, stride)
        # Skip the diagonal (e1 == e2): the second index steps over its own
        # position so the pair is always ordered and distinct by index.
        second_slot = torch.where(
            remainder_slot < first_slot, remainder_slot, remainder_slot + 1
        )
        edge1_id = plan.selected_edges[first_slot]
        edge2_id = plan.selected_edges[second_slot]
        candidate_count = int(linear.shape[0])
        yield CoupledDdCandidateRequest(
            chunk_start=start,
            chunk_end=end,
            candidate_count=candidate_count,
            linear=linear,
            tx_slot=tx_slot,
            rx_slot=rx_slot,
            edge1_id=edge1_id,
            edge2_id=edge2_id,
            component_id=7,
        )
