"""Reflection discovery limits shared by enumerated topology owners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

import torch

from witwin.channel_native.propagation.topology.kernels import (
    construction as topology_construction,
)

_ORDER1_EXHAUSTIVE_GROUP_LIMIT = 4096
_MAX_MULTIBOUNCE_FACE_SEQUENCES = 100_000
_MULTIBOUNCE_SEQUENCE_CHUNK_SIZE = 65_536
_MULTIBOUNCE_PAIR_CHUNK_SIZE = 4_194_304
_MULTIBOUNCE_DISCOVERY_RAYS = 262_144


def _face_sequence_count(
    face_count: int, depth: int, *, adjacent_distinct: bool
) -> int:
    if adjacent_distinct and depth > 1:
        if face_count <= 1:
            return 0
        return int(face_count) * int(face_count - 1) ** int(depth - 1)
    return int(face_count) ** int(depth)


def _face_sequence_chunks(
    face_count: int,
    depth: int,
    *,
    chunk_size: int,
    reference: torch.Tensor,
    face_ids: torch.Tensor | None = None,
    adjacent_distinct: bool = False,
) -> object:
    total = _face_sequence_count(face_count, depth, adjacent_distinct=adjacent_distinct)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        if face_ids is None:
            sequences = topology_construction.deterministic_face_sequence_chunk(
                reference,
                face_count=face_count,
                depth=depth,
                start=start,
                end=end,
                adjacent_distinct=adjacent_distinct,
            )
        else:
            sequences = topology_construction.deterministic_mapped_face_sequence_chunk(
                face_ids,
                depth=depth,
                start=start,
                end=end,
                adjacent_distinct=adjacent_distinct,
            )
        if int(sequences.shape[0]) > 0:
            yield sequences


class TraceReflectionGroupChains(Protocol):
    def __call__(
        self,
        tx: torch.Tensor,
        *,
        face_group_id: torch.Tensor,
        max_depth: int,
    ) -> torch.Tensor: ...


class RecordReflectionCandidateCount(Protocol):
    def __call__(self, candidate_count: int) -> None: ...


@dataclass(frozen=True, slots=True)
class ReflectionOrder1Plan:
    exhaustive: bool
    group_count: int
    representative_faces: torch.Tensor
    base_sequences: torch.Tensor | None
    face_group_id: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class ReflectionOrder1EpcRequest:
    tx_index: int
    tx: torch.Tensor
    epc_inputs: dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class ReflectionMultibouncePlan:
    exhaustive: bool
    group_count: int
    representative_faces: torch.Tensor
    face_group_id: torch.Tensor | None
    min_depth: int
    max_depth: int


@dataclass(frozen=True, slots=True)
class ReflectionMultibounceEpcRequest:
    depth: int
    tx_index: int
    tx: torch.Tensor
    epc_inputs: dict[str, torch.Tensor]


def prepare_reflection_order1_plan(
    *,
    group_count: int,
    representative_faces: torch.Tensor,
    face_group_id: torch.Tensor,
) -> ReflectionOrder1Plan:
    exhaustive = group_count <= _ORDER1_EXHAUSTIVE_GROUP_LIMIT
    base_sequences = (
        topology_construction.deterministic_mapped_face_sequence_chunk(
            representative_faces,
            depth=1,
            start=0,
            end=group_count,
        )
        if exhaustive
        else None
    )
    selected_face_group_id = (
        None if exhaustive else face_group_id.to(dtype=torch.long).contiguous()
    )
    return ReflectionOrder1Plan(
        exhaustive=exhaustive,
        group_count=group_count,
        representative_faces=representative_faces,
        base_sequences=base_sequences,
        face_group_id=selected_face_group_id,
    )


def iter_reflection_order1_epc_requests(
    plan: ReflectionOrder1Plan,
    *,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    tri_a: torch.Tensor,
    normals: torch.Tensor,
    trace_group_chains: TraceReflectionGroupChains,
) -> Iterator[ReflectionOrder1EpcRequest]:
    rx_count = int(rx_positions.shape[0])
    if plan.group_count <= 0 or rx_count <= 0:
        return

    for tx_index, tx in enumerate(tx_positions):
        if plan.exhaustive:
            sequences = plan.base_sequences
        else:
            chains = trace_group_chains(
                tx, face_group_id=plan.face_group_id, max_depth=1
            )
            first_groups = torch.unique(chains[chains[:, 0] >= 0][:, 0])
            if int(first_groups.numel()) == 0:
                continue
            sequences = (
                plan.representative_faces[first_groups].reshape(-1, 1).contiguous()
            )
        sequence_count = int(sequences.shape[0])
        if sequence_count <= 0:
            continue
        rx_chunk_size = max(1, _MULTIBOUNCE_PAIR_CHUNK_SIZE // sequence_count)
        for rx_start in range(0, rx_count, rx_chunk_size):
            rx_end = min(rx_start + rx_chunk_size, rx_count)
            epc_inputs = topology_construction.deterministic_reflection_epc_input_batch(
                tx=tx,
                rx_positions=rx_positions.contiguous(),
                sequences=sequences.contiguous(),
                tri_a=tri_a.contiguous(),
                normals=normals.contiguous(),
                rx_start=rx_start,
                rx_end=rx_end,
            )
            yield ReflectionOrder1EpcRequest(
                tx_index=tx_index,
                tx=tx,
                epc_inputs=epc_inputs,
            )


def prepare_reflection_multibounce_plan(
    *,
    group_count: int,
    representative_faces: torch.Tensor,
    face_group_id: torch.Tensor,
    min_depth: int,
    max_depth: int,
) -> ReflectionMultibouncePlan:
    exhaustive = all(
        _face_sequence_count(group_count, depth, adjacent_distinct=True)
        <= _MAX_MULTIBOUNCE_FACE_SEQUENCES
        for depth in range(min_depth, max_depth + 1)
    )
    selected_face_group_id = (
        None if exhaustive else face_group_id.to(dtype=torch.long).contiguous()
    )
    return ReflectionMultibouncePlan(
        exhaustive=exhaustive,
        group_count=group_count,
        representative_faces=representative_faces,
        face_group_id=selected_face_group_id,
        min_depth=min_depth,
        max_depth=max_depth,
    )


def iter_reflection_multibounce_epc_requests(
    plan: ReflectionMultibouncePlan,
    *,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    sequence_reference: torch.Tensor,
    tri_a: torch.Tensor,
    normals: torch.Tensor,
    trace_group_chains: TraceReflectionGroupChains,
    record_candidate_count: RecordReflectionCandidateCount,
) -> Iterator[ReflectionMultibounceEpcRequest]:
    rx_count = int(rx_positions.shape[0])
    if plan.exhaustive:
        for depth in range(plan.min_depth, plan.max_depth + 1):
            candidate_count = _face_sequence_count(
                plan.group_count, depth, adjacent_distinct=True
            )
            record_candidate_count(candidate_count)
            chunk_size = min(_MULTIBOUNCE_SEQUENCE_CHUNK_SIZE, max(candidate_count, 1))
            for sequences in _face_sequence_chunks(
                plan.group_count,
                depth,
                chunk_size=chunk_size,
                reference=sequence_reference,
                face_ids=plan.representative_faces,
                adjacent_distinct=True,
            ):
                sequence_count = int(sequences.shape[0])
                if sequence_count <= 0:
                    continue
                rx_chunk_size = max(1, _MULTIBOUNCE_PAIR_CHUNK_SIZE // sequence_count)
                for rx_start in range(0, rx_count, rx_chunk_size):
                    rx_end = min(rx_start + rx_chunk_size, rx_count)
                    for tx_index, tx in enumerate(tx_positions):
                        epc_inputs = topology_construction.deterministic_reflection_epc_input_batch(
                            tx=tx,
                            rx_positions=rx_positions.contiguous(),
                            sequences=sequences.contiguous(),
                            tri_a=tri_a.contiguous(),
                            normals=normals.contiguous(),
                            rx_start=rx_start,
                            rx_end=rx_end,
                        )
                        yield ReflectionMultibounceEpcRequest(
                            depth=depth,
                            tx_index=tx_index,
                            tx=tx,
                            epc_inputs=epc_inputs,
                        )
    else:
        for tx_index, tx in enumerate(tx_positions):
            group_chains = trace_group_chains(
                tx,
                face_group_id=plan.face_group_id,
                max_depth=plan.max_depth,
            )
            for depth in range(plan.min_depth, plan.max_depth + 1):
                reached = group_chains[:, depth - 1] >= 0
                if not bool(reached.any()):
                    continue
                unique_chains = torch.unique(group_chains[reached][:, :depth], dim=0)
                record_candidate_count(int(unique_chains.shape[0]))
                sequences_all = plan.representative_faces[unique_chains].contiguous()
                for start in range(
                    0,
                    int(sequences_all.shape[0]),
                    _MULTIBOUNCE_SEQUENCE_CHUNK_SIZE,
                ):
                    sequences = sequences_all[
                        start : start + _MULTIBOUNCE_SEQUENCE_CHUNK_SIZE
                    ].contiguous()
                    rx_chunk_size = max(
                        1, _MULTIBOUNCE_PAIR_CHUNK_SIZE // int(sequences.shape[0])
                    )
                    for rx_start in range(0, rx_count, rx_chunk_size):
                        rx_end = min(rx_start + rx_chunk_size, rx_count)
                        epc_inputs = topology_construction.deterministic_reflection_epc_input_batch(
                            tx=tx,
                            rx_positions=rx_positions.contiguous(),
                            sequences=sequences.contiguous(),
                            tri_a=tri_a.contiguous(),
                            normals=normals.contiguous(),
                            rx_start=rx_start,
                            rx_end=rx_end,
                        )
                        yield ReflectionMultibounceEpcRequest(
                            depth=depth,
                            tx_index=tx_index,
                            tx=tx,
                            epc_inputs=epc_inputs,
                        )
