"""Structural endpoint-row selection for a prepared frozen topology.

The zero-interaction LoS route keeps its fused native gather in
:mod:`witwin.channel.propagation.consumer._fixed_los`. A frozen topology that
carries interactions cannot use that gather: its contract is LoS-only and it
rejects a reflection row inside the validation kernel. This module is the
boundary work that replaces it for the prepared route.

Everything here is non-numerical: integer contract validation reduced to one
device bitmask, ``index_select`` row selection of caller-owned endpoint
tensors, and the CSR pair segmentation built from integer row identity. No
geometry, no field, and no material value is computed, transformed, or
re-derived; every physical quantity is produced later by a native kernel that
owns it.

The validation budget matches the native LoS gather exactly: one four-byte
device-to-host copy and one synchronization for the whole batch, before any
native work runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import EndpointBatch, PropagationTopology


# Same bit vocabulary the native LoS validator publishes, so a caller reading
# the two error messages does not have to learn two encodings. Depth and
# component (bits 4 and 8) are host-validated once by ``prepare_fixed_topology``
# and therefore never fire here.
_INDEX_BOUNDS = 1
_PAIR_ORDER = 2
_STABLE_ID = 16

VALIDATION_D2H_COPIES = 1
VALIDATION_D2H_BYTES = 4
VALIDATION_SYNCHRONIZATIONS = 1


@dataclass(frozen=True, slots=True)
class PreparedRows:
    """Frozen rows bound to the current endpoint batches."""

    source: torch.Tensor
    target: torch.Tensor
    tx_power: torch.Tensor
    tx_polarization: torch.Tensor
    rx_polarization: torch.Tensor
    source_row_index: torch.Tensor
    sink_row_index: torch.Tensor
    pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    validation_d2h_copies: int
    validation_d2h_bytes: int
    validation_synchronizations: int

    @property
    def row_count(self) -> int:
        return int(self.pair_index.shape[0])


def _bit(flag: torch.Tensor, bit: int) -> torch.Tensor:
    return flag.to(dtype=torch.int32) * bit


def _order_violation(
    pair_index: torch.Tensor, in_bounds: torch.Tensor
) -> torch.Tensor:
    """True when an in-bounds row breaks non-decreasing pair-major order."""

    if pair_index.shape[0] < 2:
        return torch.zeros((), dtype=torch.bool, device=pair_index.device)
    descending = pair_index[1:] < pair_index[:-1]
    return (descending & in_bounds[1:] & in_bounds[:-1]).any()


def _contract_error(
    topology: PropagationTopology,
    sources: EndpointBatch,
    sinks: EndpointBatch,
    source_row_index: torch.Tensor,
    sink_row_index: torch.Tensor,
    pair_index: torch.Tensor,
) -> torch.Tensor:
    """One device-resident int32 bitmask covering every frozen row."""

    in_bounds = (
        (source_row_index >= 0)
        & (source_row_index < sources.count)
        & (sink_row_index >= 0)
        & (sink_row_index < sinks.count)
    )
    clamped_source = source_row_index.clamp(0, sources.count - 1)
    clamped_sink = sink_row_index.clamp(0, sinks.count - 1)
    identity_broken = (
        (topology.source_id != sources.stable_ids[clamped_source])
        | (topology.sink_id != sinks.stable_ids[clamped_sink])
    ) & in_bounds
    return (
        _bit((~in_bounds).any(), _INDEX_BOUNDS)
        | _bit(_order_violation(pair_index, in_bounds), _PAIR_ORDER)
        | _bit(identity_broken.any(), _STABLE_ID)
    )


def _pair_segmentation(
    pair_index: torch.Tensor, pair_count: int
) -> torch.Tensor:
    counts = torch.zeros(
        (pair_count + 1,), dtype=torch.int64, device=pair_index.device
    )
    counts.index_add_(
        0,
        pair_index + 1,
        torch.ones_like(pair_index),
    )
    return counts.cumsum(0)


def prepared_row_gather(
    topology: PropagationTopology,
    sources: EndpointBatch,
    sinks: EndpointBatch,
) -> PreparedRows:
    """Validate frozen rows and bind them to the current endpoint batches."""

    if topology.device != sources.device or topology.device != sinks.device:
        raise ValueError("topology and endpoint tensors must share one CUDA device")
    assert sources.powers_w is not None
    rows = topology.row_count
    source_count = sources.count
    sink_count = sinks.count
    if rows > 0 and (source_count == 0 or sink_count == 0):
        raise ValueError("non-empty frozen topology requires endpoint rows")
    source_row_index = topology.source_index.to(dtype=torch.int64)
    sink_row_index = topology.sink_index.to(dtype=torch.int64)
    pair_index = sink_row_index * source_count + source_row_index
    if rows > 0:
        error = int(
            _contract_error(
                topology,
                sources,
                sinks,
                source_row_index,
                sink_row_index,
                pair_index,
            ).item()
        )
        if error != 0:
            raise ValueError(
                "frozen topology validation failed against the current "
                f"endpoint batches (error bitmask {error})"
            )
    return PreparedRows(
        source=sources.positions_m.index_select(0, source_row_index).contiguous(),
        target=sinks.positions_m.index_select(0, sink_row_index).contiguous(),
        tx_power=sources.powers_w.index_select(0, source_row_index).contiguous(),
        tx_polarization=(
            sources.polarizations.index_select(0, source_row_index).contiguous()
        ),
        rx_polarization=(
            sinks.polarizations.index_select(0, sink_row_index).contiguous()
        ),
        source_row_index=source_row_index,
        sink_row_index=sink_row_index,
        pair_index=pair_index,
        pair_offsets=_pair_segmentation(pair_index, source_count * sink_count),
        validation_d2h_copies=VALIDATION_D2H_COPIES if rows else 0,
        validation_d2h_bytes=VALIDATION_D2H_BYTES if rows else 0,
        validation_synchronizations=VALIDATION_SYNCHRONIZATIONS if rows else 0,
    )


def select_rows(values: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Row selection that preserves frozen order and contiguity."""

    return values.index_select(0, rows).contiguous()


__all__ = [
    "PreparedRows",
    "VALIDATION_D2H_BYTES",
    "VALIDATION_D2H_COPIES",
    "VALIDATION_SYNCHRONIZATIONS",
    "prepared_row_gather",
    "select_rows",
]
