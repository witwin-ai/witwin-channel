"""Time-varying channel impulse response over one slot-batched replay.

A ``(source, sink)`` pair's ``delay_s`` and transport already ARE its impulse
response; nothing physical is missing from the consumer contract. What was
missing was a time axis: a caller who wanted the response at ``T`` instants had
to run ``T`` reevaluations, which is ``T`` validation copies and ``T``
synchronizations for a capability whose whole point is to have exactly one.

This module is that axis and nothing else. It replicates the frozen rows over
``T`` block-diagonal slots, runs ONE reevaluation, and publishes ``[T, K]``
views over the storage that replay produced. It owns no physics, adds no
compaction, allocates no result, and introduces no native symbol. It also does
not compile scenes: one :class:`~witwin.channel.scene.compiled.CompiledScene`
covers one structure-geometry epoch, and a world whose structures move is
``T`` epochs, which is a motion-event cadence rather than an inner loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .contracts import (
    Complex3Transport,
    EndpointBatch,
    FixedTopologyEvaluation,
    FixedTopologyRequest,
    PreparedFixedTopology,
    PropagationAdMode,
    PropagationCapabilities,
    PropagationConvention,
    PropagationDiagnostics,
    PropagationResponse,
    PropagationTransport,
    PropagationWorldMotion,
    ScalarTransport,
    _require_slot_divisible,
)
from ._prepared import replicate_over_slots
from .service import reevaluate

if TYPE_CHECKING:
    from witwin.channel.scene.compiled import CompiledScene


def _slot_view(values: torch.Tensor, slot_count: int) -> torch.Tensor:
    """Split a slot-major row axis into ``[slot_count, K, ...]``.

    ``view`` rather than ``reshape`` on purpose: the row axis is outermost and
    the replay publishes contiguous storage, so a layout that needed a copy
    here would be a silent regression rather than something to paper over.
    """

    return values.view(slot_count, -1, *values.shape[1:])


@dataclass(frozen=True, slots=True)
class TimeVaryingRequest:
    """A frozen topology replayed at a whole block of world instants.

    ``sources`` and ``sinks`` are the per-instant endpoint batches stacked
    slot-major: rows ``[t*S, (t+1)*S)`` are the sources at ``times_s[t]``.
    Their ``stable_ids`` repeat per slot, because the slots are the same
    physical endpoints observed at different instants, and the frozen rows name
    those endpoints by identity.

    ``topology`` is the PER-SLOT prepared topology, not a replicated one:
    :func:`witwin.channel.propagation.consumer.evaluate_time_varying` performs
    the replication so a caller cannot half-apply it.

    ``times_s`` is float64 and carried verbatim into the result. It labels the
    slots; it is never differenced, integrated, or otherwise used to compute a
    published number. A delay RATE comes from the ADR-038 forward dual on the
    endpoint positions, not from a finite difference across these samples.

    Because they are labels, they are the caller's assertion and not a checked
    fact, and the sharp edge is worth naming: nothing reconciles ``times_s``
    against the world the ``CompiledScene`` was built from. Endpoint motion
    legitimately runs many instants against one compiled scene, so
    ``times_s != compiled.time_s`` cannot be refused - which means a caller who
    labels slots ``t = 1, 2`` while the structures still stand at their
    ``t = 0`` pose gets a result that claims to be a channel it is not, with
    every row valid. One ``CompiledScene`` is one structure-geometry epoch;
    keeping the labels inside it is the caller's obligation, and
    ``CompiledScene.time_s`` is published so that obligation is checkable.
    """

    sources: EndpointBatch
    sinks: EndpointBatch
    reference_frequency_hz: float | torch.Tensor
    topology: PreparedFixedTopology
    times_s: torch.Tensor
    response: PropagationResponse
    ad_mode: PropagationAdMode
    world_motion: PropagationWorldMotion = "frozen_world"

    def __post_init__(self) -> None:
        if not isinstance(self.topology, PreparedFixedTopology):
            raise TypeError(
                "topology must be a PreparedFixedTopology; call "
                "prepare_fixed_topology once per frozen topology and reuse it"
            )
        times = self.times_s
        if not isinstance(times, torch.Tensor):
            raise TypeError("times_s must be a torch.Tensor")
        if times.dtype != torch.float64:
            raise TypeError(f"times_s must use torch.float64, got {times.dtype}")
        if times.ndim != 1 or int(times.shape[0]) == 0:
            raise ValueError("times_s must be a non-empty 1-D tensor")

    @property
    def slot_count(self) -> int:
        return int(self.times_s.shape[0])


@dataclass(frozen=True, slots=True, eq=False)
class TimeVaryingTransport:
    """Slot-shaped views over the transport one replay published.

    Exactly the tensors of the requested response are present and the rest are
    ``None``, so a reader cannot pick up a field the response never produced.
    """

    response: PropagationResponse
    coefficient: torch.Tensor | None = None
    field: torch.Tensor | None = None
    direction: torch.Tensor | None = None
    matrix: torch.Tensor | None = None
    source_basis: torch.Tensor | None = None
    sink_basis: torch.Tensor | None = None

    @classmethod
    def from_transport(
        cls, transport: PropagationTransport, slot_count: int
    ) -> TimeVaryingTransport:
        if isinstance(transport, ScalarTransport):
            return cls(
                response="scalar_transport",
                coefficient=_slot_view(transport.coefficient, slot_count),
            )
        if isinstance(transport, Complex3Transport):
            return cls(
                response="complex3_transport",
                field=_slot_view(transport.field, slot_count),
                direction=_slot_view(transport.direction, slot_count),
            )
        return cls(
            response="polarimetric_transport",
            matrix=_slot_view(transport.matrix, slot_count),
            source_basis=_slot_view(transport.source_basis, slot_count),
            sink_basis=_slot_view(transport.sink_basis, slot_count),
        )


@dataclass(frozen=True, slots=True, eq=False)
class TimeVaryingEvaluation:
    """One frozen topology evaluated at ``slot_count`` world instants.

    Every published tensor is slot-major: index ``[t]`` is the complete frozen
    row set at ``times_s[t]``, in frozen row order, so ``delay_s[t]`` and the
    transport at ``[t]`` are the impulse response of that instant and
    ``pair_offsets`` segments it by ``(source, sink)`` pair.

    ``pair_offsets`` and ``pair_count`` are PER SLOT. They are frozen: the same
    rows are replayed at every instant, so every slot carries the same
    segmentation, including the empty segments of pairs that publish no row.

    ``row_valid`` keeps its ADR-037 meaning per slot and remains the sole
    authority: a row that stops existing at instant ``t`` publishes exact zeros
    at ``[t]`` and stays alive at the other instants. Replay is still
    subtractive (ADR-040) - a path that comes into existence part way through
    the block is not discovered here and is silently absent from every slot.
    """

    slot_count: int
    row_count: int
    times_s: torch.Tensor
    delay_s: torch.Tensor
    path_length_m: torch.Tensor
    transport: TimeVaryingTransport
    pair_count: int
    pair_offsets: torch.Tensor
    convention: PropagationConvention
    capabilities: PropagationCapabilities
    diagnostics: PropagationDiagnostics
    row_valid: torch.Tensor | None = None

    @classmethod
    def from_evaluation(
        cls,
        evaluation: FixedTopologyEvaluation,
        times_s: torch.Tensor,
        slot_count: int,
    ) -> TimeVaryingEvaluation:
        paths = evaluation.paths
        pair_count = paths.pair_count // slot_count
        return cls(
            slot_count=slot_count,
            row_count=paths.path_count // slot_count,
            times_s=times_s,
            delay_s=_slot_view(paths.geometry.delay_s, slot_count),
            path_length_m=_slot_view(paths.geometry.path_length_m, slot_count),
            transport=TimeVaryingTransport.from_transport(
                paths.transport, slot_count
            ),
            pair_count=pair_count,
            # Every slot repeats the same segmentation, so the leading slot's
            # prefix of the CSR vector IS the per-slot segmentation. Narrowing
            # is a view; rebuilding it would be a second, redundant owner.
            pair_offsets=paths.pair_offsets[: pair_count + 1],
            convention=evaluation.convention,
            capabilities=evaluation.capabilities,
            diagnostics=evaluation.diagnostics,
            row_valid=(
                None
                if evaluation.row_valid is None
                else _slot_view(evaluation.row_valid, slot_count)
            ),
        )


def evaluate_time_varying(
    compiled_scene: CompiledScene, request: TimeVaryingRequest
) -> TimeVaryingEvaluation:
    """Replay one frozen topology across a whole block of world instants.

    One launch per ``(component, depth)`` bucket, one validation copy, and one
    synchronization for the entire block, whatever ``len(times_s)`` is. That is
    the point: a Python loop over instants keeps every individual call inside
    the ADR-032 budget while multiplying the budget of the frame by the number
    of instants.

    The instants must share one compiled scene, which is what makes them one
    frame, pulse train, or symbol block. Structure motion changes the compiled
    scene, so it is a new call with a new scene, not another slot.
    """

    if not isinstance(request, TimeVaryingRequest):
        raise TypeError("request must be a TimeVaryingRequest")
    slot_count = request.slot_count
    _require_slot_divisible("sources", request.sources.count, slot_count)
    _require_slot_divisible("sinks", request.sinks.count, slot_count)
    replicated = replicate_over_slots(
        request.topology,
        slot_count,
        source_count=request.sources.count // slot_count,
        sink_count=request.sinks.count // slot_count,
    )
    evaluation = reevaluate(
        compiled_scene,
        FixedTopologyRequest(
            sources=request.sources,
            sinks=request.sinks,
            reference_frequency_hz=request.reference_frequency_hz,
            topology=replicated,
            response=request.response,
            ad_mode=request.ad_mode,
            world_motion=request.world_motion,
            slot_count=slot_count,
        ),
    )
    return TimeVaryingEvaluation.from_evaluation(
        evaluation, request.times_s, slot_count
    )


__all__ = [
    "TimeVaryingEvaluation",
    "TimeVaryingRequest",
    "TimeVaryingTransport",
    "evaluate_time_varying",
]
