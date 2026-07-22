"""Lazy first-order diffraction discovery planning."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import torch

from witwin.channel.propagation.topology.discovery.reflection import (
    _MULTIBOUNCE_PAIR_CHUNK_SIZE,
)


@dataclass(frozen=True, slots=True)
class DiffractionOrder1Plan:
    preserve_imported_edges: bool
    tx_count: int
    rx_count: int


@dataclass(frozen=True, slots=True)
class DiffractionTxRequest:
    tx_index: int
    tx: torch.Tensor


@dataclass(frozen=True, slots=True)
class DiffractionRxChunkRequest:
    rx_start: int
    rx_end: int
    capacity: int


def prepare_diffraction_order1_plan(
    *,
    metadata: Mapping[str, object],
    tx_count: int,
    rx_count: int,
) -> DiffractionOrder1Plan:
    mitsuba_metadata = metadata.get("mitsuba", {})
    # Channel's merge_shapes import keeps the selected boundary-edge table
    # intact. The synthetic-scene path instead merges coincident structure
    # boundaries into one physical wedge (the single-wedge test contract).
    preserve_imported_edges = isinstance(mitsuba_metadata, dict) and bool(
        mitsuba_metadata.get("merge_shapes", False)
    )
    return DiffractionOrder1Plan(
        preserve_imported_edges=preserve_imported_edges,
        tx_count=int(tx_count),
        rx_count=int(rx_count),
    )


def iter_diffraction_tx_requests(
    plan: DiffractionOrder1Plan,
    *,
    tx_positions: torch.Tensor,
) -> Iterator[DiffractionTxRequest]:
    for tx_index in range(plan.tx_count):
        yield DiffractionTxRequest(tx_index=tx_index, tx=tx_positions[tx_index])


def iter_diffraction_rx_chunk_requests(
    plan: DiffractionOrder1Plan,
    *,
    state_count: int,
) -> Iterator[DiffractionRxChunkRequest]:
    rx_chunk_size = max(1, _MULTIBOUNCE_PAIR_CHUNK_SIZE // state_count)
    for rx_start in range(0, plan.rx_count, rx_chunk_size):
        rx_end = min(rx_start + rx_chunk_size, plan.rx_count)
        yield DiffractionRxChunkRequest(
            rx_start=rx_start,
            rx_end=rx_end,
            capacity=(rx_end - rx_start) * state_count,
        )
