"""Preparing a discovered topology for repeated fixed-topology replay.

``prepare_fixed_topology`` is the one place the consumer looks at frozen rows on
the host: it validates their depth/interaction padding and partitions them into
the ascending ``(component, depth)`` buckets a replay launches one at a time.
``replicate_over_slots`` then tiles that partition over ADR-041 block-diagonal
slots by index arithmetic alone.

Both are vocabulary-level boundary work with no physics in them, and both are
published through the package facade rather than through
:mod:`witwin.channel.propagation.consumer.contracts`, which stays the place a
reader looks up a TYPE.
"""

from __future__ import annotations

import torch

from .contracts import (
    _CAPABILITIES,
    _FIXED_TOPOLOGY_COMPONENT_IDS,
    _require_slot_count,
    FixedTopologyBucket,
    PreparedFixedTopology,
    PropagationTopology,
)


def _fixed_topology_component_name(component_id: int) -> str:
    for name, value in _FIXED_TOPOLOGY_COMPONENT_IDS:
        if value == component_id:
            return name
    raise NotImplementedError(
        f"fixed-topology reevaluation does not support component id "
        f"{component_id}; supported components are "
        f"{sorted(_CAPABILITIES.fixed_topology_components)}"
    )


def _require_bucket_depth(component: str, depth: int) -> None:
    if component == "los" and depth != 0:
        raise ValueError("a los row must have depth 0")
    if component == "reflection" and depth < 1:
        raise ValueError("a reflection row must have depth >= 1")


def _malformed_rows(topology: PropagationTopology, width: int) -> torch.Tensor:
    """Rows whose depth or interaction padding contradicts the contract."""

    depth = topology.depth.to(dtype=torch.int64)
    slots = torch.arange(
        width, device=topology.device, dtype=torch.int64
    ).reshape(1, -1)
    active = slots < depth.reshape(-1, 1)
    sequence = topology.primitive_sequence.to(dtype=torch.int64)
    return (
        (depth < 0)
        | (depth > width)
        | ((sequence < 0) & active).any(dim=1)
        | ((sequence != -1) & ~active).any(dim=1)
    )


def prepare_fixed_topology(
    topology: PropagationTopology,
) -> PreparedFixedTopology:
    """Partition a frozen topology by component and interaction depth.

    This is the one place the consumer looks at a frozen topology on the host.
    It validates the depth/interaction padding of every row, rejects any
    component outside ``capabilities().fixed_topology_components``, and returns
    the ascending ``(component, depth)`` buckets that
    :func:`witwin.channel.propagation.consumer.reevaluate` replays.

    Call it once per frozen topology and reuse the handle. It synchronizes; a
    per-frame call would reintroduce exactly the host observation the fixed
    topology capability exists to avoid.
    """

    if not isinstance(topology, PropagationTopology):
        raise TypeError("topology must be a PropagationTopology")
    if topology.row_count == 0:
        return PreparedFixedTopology(
            topology=topology,
            buckets=(),
            prepare_d2h_copies=0,
            prepare_d2h_bytes=0,
            prepare_synchronizations=0,
        )
    width = int(topology.primitive_sequence.shape[1])
    if bool(_malformed_rows(topology, width).any().item()):
        raise ValueError(
            "frozen topology rows disagree with their interaction sequence "
            "padding; depth must be in [0, sequence width] and unused slots "
            "must hold -1"
        )
    key = topology.component_id.to(dtype=torch.int64) * (width + 1) + (
        topology.depth.to(dtype=torch.int64)
    )
    distinct = torch.unique(key).tolist()
    buckets = []
    for value in distinct:
        component_id, depth = divmod(int(value), width + 1)
        component = _fixed_topology_component_name(component_id)
        _require_bucket_depth(component, depth)
        buckets.append(
            FixedTopologyBucket(
                component=component,
                depth=depth,
                rows=torch.nonzero(key == value, as_tuple=False).reshape(-1),
            )
        )
    return PreparedFixedTopology(
        topology=topology,
        buckets=tuple(buckets),
        prepare_d2h_copies=2 + len(buckets),
        prepare_d2h_bytes=1 + 8 * (len(distinct) + len(buckets)),
        prepare_synchronizations=2 + len(buckets),
    )


def replicate_over_slots(
    prepared: PreparedFixedTopology,
    slot_count: int,
    *,
    source_count: int,
    sink_count: int,
) -> PreparedFixedTopology:
    """Tile a frozen topology over ``slot_count`` block-diagonal slots.

    ``source_count`` and ``sink_count`` are the PER-SLOT endpoint counts of the
    stacked batches the replicated topology will be replayed against. They are
    required rather than inferred: an endpoint that publishes no frozen row
    never appears in ``source_index``, so the largest index in a topology is
    not the endpoint count and inferring one would silently mislabel every
    later slot.

    This is pure index arithmetic and bucket re-partitioning: row ``t*K + r``
    names the same frozen row ``r`` shifted into slot ``t``, so the frozen row
    order is preserved inside every slot and the ``(component, depth)`` bucket
    COUNT is unchanged - only the bucket row counts grow. No compaction, no
    physics, no native symbol, no host observation. ``slot_count == 1`` returns
    the handle unchanged, so a single-slot replay is bit-identical to one that
    never asked for slots.

    ``provenance`` is forwarded verbatim, so a replicated topology is checked
    for staleness exactly like the topology it came from (ADR-040).
    """

    if not isinstance(prepared, PreparedFixedTopology):
        raise TypeError("prepared must be a PreparedFixedTopology")
    slot_count = _require_slot_count(slot_count)
    for name, value in (("source_count", source_count), ("sink_count", sink_count)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive int")
    if slot_count == 1:
        return prepared
    topology = prepared.topology
    rows = topology.row_count
    device = topology.device
    slot = torch.arange(slot_count, device=device, dtype=torch.int32)
    replicated = PropagationTopology(
        source_index=(
            topology.source_index.repeat(slot_count)
            + slot.mul(source_count).repeat_interleave(rows)
        ),
        sink_index=(
            topology.sink_index.repeat(slot_count)
            + slot.mul(sink_count).repeat_interleave(rows)
        ),
        source_id=topology.source_id.repeat(slot_count),
        sink_id=topology.sink_id.repeat(slot_count),
        depth=topology.depth.repeat(slot_count),
        component_id=topology.component_id.repeat(slot_count),
        primitive_id=topology.primitive_id.repeat(slot_count),
        edge_id=topology.edge_id.repeat(slot_count),
        material_id=topology.material_id.repeat(slot_count),
        primitive_sequence=topology.primitive_sequence.repeat(slot_count, 1),
        material_sequence=topology.material_sequence.repeat(slot_count, 1),
        interaction_type=topology.interaction_type.repeat(slot_count, 1),
        provenance=topology.provenance,
    )
    row_offset = torch.arange(
        slot_count, device=device, dtype=torch.int64
    ).mul(rows).reshape(-1, 1)
    return PreparedFixedTopology(
        topology=replicated,
        buckets=tuple(
            FixedTopologyBucket(
                component=bucket.component,
                depth=bucket.depth,
                rows=(bucket.rows.reshape(1, -1) + row_offset).reshape(-1),
            )
            for bucket in prepared.buckets
        ),
        # Replication observes nothing, so the handle keeps the cost of the one
        # preparation it was derived from rather than claiming a new one.
        prepare_d2h_copies=prepared.prepare_d2h_copies,
        prepare_d2h_bytes=prepared.prepare_d2h_bytes,
        prepare_synchronizations=prepared.prepare_synchronizations,
    )


__all__ = ["prepare_fixed_topology", "replicate_over_slots"]
