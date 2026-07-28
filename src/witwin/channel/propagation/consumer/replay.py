"""Everything the consumer needs to evaluate rows it already owns.

This module is the single owner of the consumer's replay machinery: the native
ABI facades it dispatches, the structural row gathers that bind frozen rows to
the current endpoint batches, the source-amplitude and Jones composition
helpers, and the prepared-topology bucket replay itself. They were six modules
that only ever called each other, so a reader following one replay had to walk
six files to see one code path.

Nothing here is a contract: :mod:`witwin.channel.propagation.consumer.contracts`
stays the single place a reader looks up a published type. Nothing here decides
admission either; that is
:mod:`witwin.channel.propagation.consumer.policy`. This module is dispatch,
structural selection, and packing over native owners.

Four things drive the shape of the code.

**The compact finalizer.** Exact valid rows and their pair segmentation come
from one native owner, through one autograd Function with registered
forward/backward/JVP companions.

**The frozen line-of-sight gather.** A zero-interaction frozen topology is bound
to its endpoint batches by a fused native gather with its own AD family. Its
contract is line-of-sight only, so a frozen topology that carries interactions
uses the structural gather beside it instead: integer contract validation
reduced to one device bitmask, ``index_select`` row selection of caller-owned
endpoint tensors, and the CSR pair segmentation built from integer row
identity. No geometry, no field, and no material value is computed, transformed,
or re-derived; every physical quantity is produced later by a native kernel that
owns it. The validation budget matches the native LoS gather exactly: one
four-byte device-to-host copy and one synchronization for the whole batch,
before any native work runs.

**The composed Jones operator.** The native field transport is linear in the
transmit polarization and linear in the receive polarization:

* ``project_to_wedge_plane(v, e) = v - e*(v.e)`` is linear in ``v``;
* a Fresnel bounce scales the s and p components by coefficients that depend on
  the incidence frame and the material, never on the field itself;
* the trailing free-space factor is a complex scalar;
* ``project_receiver(E, d, p) = E . project_to_wedge_plane(p, d)``.

So the map from a source transverse component to a sink transverse component is
bilinear, and the four entries of the operator are recovered exactly by
exciting the SAME native transport twice, once per source basis vector, and
projecting each response onto both sink basis vectors. Nothing here computes
physics: the composition chooses excitations, dispatches the native owners, and
stacks their published results. Both transverse bases are produced by the native
``consumer_los_jones`` endpoint-basis owner rather than by a Torch normalize or
cross product. A reflection row has two different directions - the launch
direction toward its first interaction and the arrival direction from its last
interaction - and the basis for each is obtained by handing that leg's two
endpoints to the native owner, which recomputes the direction with the same
``safe_normalize`` the field kernel uses. The bases are structurally
primal-only: the composition feeds them to the native companions as
``tx_polarization`` and ``rx_polarization``, both of which reject gradients by
contract.

**The prepared-topology replay.** A frozen reflection row is a face sequence,
not a fixed point in space. At new endpoint positions its stationary point has
to be resolved again, because the specular point moves and can leave its facet
or become occluded. The replay runs exactly the owners the discovery path used -
the RayD fixed-winner EPC re-solve for the geometry, and the native reflection
field transport for the field - so a reevaluated row is the value discovery
would have produced at those endpoints.

The native reflection transport takes ONE uniform interaction depth per launch,
so a mixed-depth frozen batch is replayed one ``(component, depth)`` bucket at a
time. Those buckets come from ``prepare_fixed_topology`` and are host-known, so
the per-call path never observes a device count to decide how to launch.

A frozen path can legitimately stop existing. Failing the whole batch would
force a caller back to full discovery the first time one path dies, which
defeats the capability, so validity is published per row. An invalid row is NOT
a failure: it is the correct, complete answer that this frozen path does not
exist at these endpoints. Capacity, ABI, contract, and device failures remain
all-or-nothing and still raise before a result exists.

An invalid row is made inert at the input, not patched at the output: its
transmit polarization is replaced by the zero vector, and the native transport
carries that exactly through projection, every Fresnel bounce, and the trailing
free-space scalar, so all four field outputs come out as exact zeros from the
kernel that owns them. Only the scalar path geometry, which has no such inert
excitation, is selected against the mask afterwards.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from witwin.channel.propagation.rows import (
    EvaluatedPaths,
    PathFields,
    PathGeometry,
    PathTopology,
)
from witwin.channel.propagation.topology.kernels import (
    evaluated_paths_compact_finalize_backward,
    evaluated_paths_compact_finalize_jvp,
)
from witwin.channel.runtime import (
    _ad_first_order_only,
    _ad_geometry_live,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    disable_functorch,
    required_symbol as _required_native_op,
)

from .contracts import (
    EndpointBatch,
    FixedTopologyBucket,
    PreparedFixedTopology,
    PropagationTopology,
)

if TYPE_CHECKING:
    from witwin.channel.scene.compiler import CompiledScene


# --- Native compact finalization ------------------------------------------

_TOPOLOGY_FIELDS = (
    "valid",
    "tx_id",
    "rx_id",
    "depth",
    "component_id",
    "primitive_id",
    "edge_id",
    "material_id",
    "primitive_sequence",
    "material_sequence",
    "interaction_type",
)
_CONTINUOUS_FIELDS = (
    "path_length_m",
    "delay_s",
    "field_direction",
    "interaction_position",
    "interaction_normal",
    "interaction_positions",
    "interaction_normals",
    "path_gain",
    "path_field",
    "field_xyz",
    "coefficient",
)
_STRUCTURAL_FIELDS = (
    "selected_row_index",
    "pair_index",
    "pair_offsets",
    "source_id",
    "sink_id",
)
_COMPACT_OUTPUT_FIELDS = (
    *_STRUCTURAL_FIELDS,
    *_TOPOLOGY_FIELDS,
    *_CONTINUOUS_FIELDS,
)
_DISCRETE_OUTPUT_COUNT = len(_STRUCTURAL_FIELDS) + len(_TOPOLOGY_FIELDS)
COMPACT_COUNT_D2H_COPIES = 1
COMPACT_COUNT_D2H_BYTES = 8
COMPACT_COUNT_SYNCHRONIZATIONS = 1


@dataclass(frozen=True, slots=True)
class CompactEvaluatedPaths:
    path_count: int
    pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    source_id: torch.Tensor
    sink_id: torch.Tensor
    evaluated: EvaluatedPaths
    count_d2h_copies: int
    count_d2h_bytes: int
    count_synchronizations: int
    native_launch_count: int


@dataclass(frozen=True, slots=True)
class LoSJonesRows:
    matrix: torch.Tensor
    source_basis: torch.Tensor
    sink_basis: torch.Tensor
    native_launch_count: int


def _candidate_tensors(paths: EvaluatedPaths) -> tuple[torch.Tensor, ...]:
    return (
        *(getattr(paths.topology, name) for name in _TOPOLOGY_FIELDS),
        *(getattr(paths.geometry, name) for name in _CONTINUOUS_FIELDS[:7]),
        *(getattr(paths.fields, name) for name in _CONTINUOUS_FIELDS[7:]),
    )


def _validate_inputs(
    paths: EvaluatedPaths,
    source_stable_ids: torch.Tensor,
    sink_stable_ids: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if not isinstance(paths, EvaluatedPaths):
        raise TypeError("paths must be EvaluatedPaths")
    tensors = _candidate_tensors(paths)
    device = tensors[0].device
    if device.type != "cuda":
        raise ValueError("compact evaluated-path finalization requires CUDA")
    for name, tensor in zip(
        (*_TOPOLOGY_FIELDS, *_CONTINUOUS_FIELDS), tensors, strict=True
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must share evaluated path device")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    for name, lookup in (
        ("source_stable_ids", source_stable_ids),
        ("sink_stable_ids", sink_stable_ids),
    ):
        if not isinstance(lookup, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if (
            lookup.device != device
            or lookup.dtype != torch.int64
            or lookup.ndim != 1
            or not lookup.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be a contiguous CUDA int64 vector on {device}"
            )
    return tensors


def evaluated_paths_compact_finalize(
    *inputs: torch.Tensor, rows_are_compact: bool
) -> dict[str, object]:
    return _required_native_op("evaluated_paths_compact_finalize")(
        *inputs, rows_are_compact
    )


class _CompactEvaluatedPathsFunction(torch.autograd.Function):
    @staticmethod
    def forward(*inputs):
        raw = evaluated_paths_compact_finalize(
            *inputs, rows_are_compact=False
        )
        expected = {"path_count", *_COMPACT_OUTPUT_FIELDS}
        if not isinstance(raw, dict) or set(raw) != expected:
            raise TypeError("native compact finalizer returned bad fields")
        if type(raw["path_count"]) is not int:
            raise TypeError("native compact finalizer returned a non-int path_count")
        outputs = tuple(raw[name] for name in _COMPACT_OUTPUT_FIELDS)
        if raw["path_count"] != outputs[1].shape[0]:
            raise RuntimeError("native compact finalizer returned inconsistent K")
        return outputs

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        ctx.candidate_count = int(inputs[0].shape[0])
        ctx.sequence_width = int(inputs[8].shape[1])
        saved = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (output[5], output[0])
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        ctx.mark_non_differentiable(*output[:_DISCRETE_OUTPUT_COUNT])

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, *grad_outputs):
        none_grads = (None,) * 24
        continuous_grads = grad_outputs[_DISCRETE_OUTPUT_COUNT:]
        if all(value is None for value in continuous_grads):
            return none_grads
        if not any(ctx.needs_input_grad[11:22]):
            return none_grads
        valid, selected_row_index = ctx.saved_tensors
        raw = evaluated_paths_compact_finalize_backward(
            valid,
            selected_row_index,
            *continuous_grads,
            candidate_count=ctx.candidate_count,
            sequence_width=ctx.sequence_width,
        )
        if not isinstance(raw, dict) or set(raw) != set(_CONTINUOUS_FIELDS):
            raise TypeError("native compact finalizer backward returned bad fields")
        return (
            *(None for _ in range(11)),
            *(
                raw[name] if ctx.needs_input_grad[index] else None
                for index, name in enumerate(_CONTINUOUS_FIELDS, start=11)
            ),
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        continuous_tangents = tuple(
            _ad_native_tangent_or_none(value) for value in tangents[11:22]
        )
        if all(value is None for value in continuous_tangents):
            return (None,) * len(_COMPACT_OUTPUT_FIELDS)
        valid, selected_row_index = (
            _ad_native_tensor(value) for value in ctx.saved_tensors
        )
        with disable_functorch():
            raw = evaluated_paths_compact_finalize_jvp(
                valid,
                selected_row_index,
                *continuous_tangents,
                candidate_count=ctx.candidate_count,
                sequence_width=ctx.sequence_width,
            )
        if not isinstance(raw, dict) or set(raw) != set(_CONTINUOUS_FIELDS):
            raise TypeError("native compact finalizer JVP returned bad fields")
        return (
            *(None for _ in range(_DISCRETE_OUTPUT_COUNT)),
            *(raw[name] for name in _CONTINUOUS_FIELDS),
        )


def compact_evaluated_paths(
    paths: EvaluatedPaths,
    *,
    source_stable_ids: torch.Tensor,
    sink_stable_ids: torch.Tensor,
    rows_are_compact: bool = False,
) -> CompactEvaluatedPaths:
    """Publish exact valid rows and pair segmentation from the sole native owner."""

    tensors = _validate_inputs(paths, source_stable_ids, sink_stable_ids)
    if rows_are_compact:
        native = evaluated_paths_compact_finalize(
            *tensors,
            source_stable_ids,
            sink_stable_ids,
            rows_are_compact=True,
        )
        expected = {"path_count", *_COMPACT_OUTPUT_FIELDS}
        if not isinstance(native, dict) or set(native) != expected:
            raise TypeError("native exact-row finalizer returned bad fields")
        if native["path_count"] != paths.row_count:
            raise RuntimeError("native exact-row finalizer changed K")
        for name, tensor in zip(
            (*_TOPOLOGY_FIELDS, *_CONTINUOUS_FIELDS), tensors, strict=True
        ):
            alias = native[name]
            if (
                alias.data_ptr() != tensor.data_ptr()
                or alias.shape != tensor.shape
                or alias.stride() != tensor.stride()
            ):
                raise RuntimeError(
                    f"native exact-row finalizer copied payload field {name}"
                )
        raw = native
        evaluated = paths
    else:
        outputs = _CompactEvaluatedPathsFunction.apply(
            *tensors, source_stable_ids, sink_stable_ids
        )
        raw = dict(zip(_COMPACT_OUTPUT_FIELDS, outputs, strict=True))
        topology = PathTopology(
            **{name: raw[name] for name in _TOPOLOGY_FIELDS}
        )
        geometry = PathGeometry(
            row_identity=topology.row_identity,
            **{name: raw[name] for name in _CONTINUOUS_FIELDS[:7]},
        )
        fields = PathFields(
            row_identity=topology.row_identity,
            **{name: raw[name] for name in _CONTINUOUS_FIELDS[7:]},
        )
        evaluated = EvaluatedPaths(
            topology=topology,
            geometry=geometry,
            fields=fields,
        )
    return CompactEvaluatedPaths(
        path_count=int(raw["pair_index"].shape[0]),
        pair_index=raw["pair_index"],
        pair_offsets=raw["pair_offsets"],
        source_id=raw["source_id"],
        sink_id=raw["sink_id"],
        evaluated=evaluated,
        count_d2h_copies=(
            COMPACT_COUNT_D2H_COPIES
            if paths.row_count > 0 and not rows_are_compact
            else 0
        ),
        count_d2h_bytes=(
            COMPACT_COUNT_D2H_BYTES
            if paths.row_count > 0 and not rows_are_compact
            else 0
        ),
        count_synchronizations=(
            COMPACT_COUNT_SYNCHRONIZATIONS
            if paths.row_count > 0 and not rows_are_compact
            else 0
        ),
        native_launch_count=1,
    )


def consumer_los_jones(
    *,
    pair_index: torch.Tensor,
    source_positions: torch.Tensor,
    sink_positions: torch.Tensor,
    source_reference_basis: torch.Tensor,
    sink_reference_basis: torch.Tensor,
    frequency_hz: float,
) -> LoSJonesRows:
    """Evaluate primal-only LoS transport between row-specific transverse bases."""

    tensors = (
        pair_index,
        source_positions,
        sink_positions,
        source_reference_basis,
        sink_reference_basis,
    )
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("LoS Jones inputs must be torch tensors")
    if any(value.requires_grad for value in tensors):
        raise RuntimeError("LoS Jones transport does not support AD")
    raw = _required_native_op("consumer_los_jones")(
        pair_index,
        source_positions,
        sink_positions,
        source_reference_basis,
        sink_reference_basis,
        float(frequency_hz),
    )
    if not isinstance(raw, dict) or set(raw) != {
        "matrix",
        "source_basis",
        "sink_basis",
    }:
        raise TypeError("native LoS Jones operator returned bad fields")
    return LoSJonesRows(
        matrix=raw["matrix"],
        source_basis=raw["source_basis"],
        sink_basis=raw["sink_basis"],
        native_launch_count=1 if pair_index.shape[0] > 0 else 0,
    )


# --- Frozen line-of-sight gather ------------------------------------------

_ROW_FIELDS = (
    "source",
    "target",
    "tx_power",
    "tx_polarization",
    "rx_polarization",
)
_LOS_GATHER_OUTPUT_FIELDS = (*_ROW_FIELDS, "pair_index", "pair_offsets")
_ENDPOINT_GRAD_FIELDS = (
    "source_positions",
    "sink_positions",
    "source_powers",
    "source_polarizations",
    "sink_polarizations",
)


@dataclass(frozen=True, slots=True)
class FixedLoSRows:
    """Exact frozen LoS rows ready for the RayD-owned free-space field family."""

    source: torch.Tensor
    target: torch.Tensor
    tx_power: torch.Tensor
    tx_polarization: torch.Tensor
    rx_polarization: torch.Tensor
    pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    validation_d2h_copies: int
    validation_d2h_bytes: int
    validation_synchronizations: int

    @property
    def row_count(self) -> int:
        return int(self.pair_index.shape[0])


class _FixedLoSGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(*inputs):
        raw = _required_native_op("consumer_fixed_los_gather")(*inputs)
        if not isinstance(raw, dict) or set(raw) != set(
            _LOS_GATHER_OUTPUT_FIELDS
        ):
            raise TypeError("native fixed LoS gather returned bad fields")
        return tuple(raw[name] for name in _LOS_GATHER_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        source_index, sink_index = (
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in inputs[:2]
        )
        ctx.source_count = int(inputs[6].shape[0])
        ctx.sink_count = int(inputs[7].shape[0])
        ctx.save_for_backward(source_index, sink_index)
        ctx.save_for_forward(source_index, sink_index)
        ctx.mark_non_differentiable(*output[5:])

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, *grad_outputs):
        endpoint_grads = grad_outputs[:5]
        if all(value is None for value in endpoint_grads):
            return (None,) * 13
        source_index, sink_index = ctx.saved_tensors
        raw = _required_native_op("consumer_fixed_los_gather_backward")(
            source_index,
            sink_index,
            *endpoint_grads,
            ctx.source_count,
            ctx.sink_count,
        )
        if not isinstance(raw, dict) or set(raw) != set(_ENDPOINT_GRAD_FIELDS):
            raise TypeError("native fixed LoS gather backward returned bad fields")
        return (
            *(None for _ in range(6)),
            *(
                raw[name] if ctx.needs_input_grad[index] else None
                for index, name in enumerate(_ENDPOINT_GRAD_FIELDS, start=6)
            ),
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        endpoint_tangents = tuple(
            _ad_native_tangent_or_none(value) for value in tangents[6:11]
        )
        if all(value is None for value in endpoint_tangents):
            return (None,) * len(_LOS_GATHER_OUTPUT_FIELDS)
        source_index, sink_index = (
            _ad_native_tensor(value) for value in ctx.saved_tensors
        )
        with disable_functorch():
            raw = _required_native_op("consumer_fixed_los_gather_jvp")(
                source_index,
                sink_index,
                *endpoint_tangents,
                ctx.source_count,
                ctx.sink_count,
            )
        if not isinstance(raw, dict) or set(raw) != set(_ROW_FIELDS):
            raise TypeError("native fixed LoS gather jvp returned bad fields")
        return (*(raw[name] for name in _ROW_FIELDS), None, None)


def fixed_los_gather(
    topology: PropagationTopology,
    sources: EndpointBatch,
    sinks: EndpointBatch,
) -> FixedLoSRows:
    """Validate and gather frozen LoS rows without Python/Torch indexing."""

    if not isinstance(topology, PropagationTopology):
        raise TypeError("topology must be a PropagationTopology")
    if not isinstance(sources, EndpointBatch) or not isinstance(sinks, EndpointBatch):
        raise TypeError("sources and sinks must be EndpointBatch instances")
    if sources.powers_w is None:
        raise ValueError("sources.powers_w is required")
    if sinks.powers_w is not None:
        raise ValueError("sinks.powers_w must be absent")
    if topology.device != sources.device or topology.device != sinks.device:
        raise ValueError("topology and endpoint tensors must share one CUDA device")

    values = _FixedLoSGatherFunction.apply(
        topology.source_index,
        topology.sink_index,
        topology.source_id,
        topology.sink_id,
        topology.depth,
        topology.component_id,
        sources.positions_m,
        sinks.positions_m,
        sources.powers_w,
        sources.polarizations,
        sinks.polarizations,
        sources.stable_ids,
        sinks.stable_ids,
    )
    rows = int(topology.source_index.shape[0])
    return FixedLoSRows(
        **dict(zip(_LOS_GATHER_OUTPUT_FIELDS, values, strict=True)),
        validation_d2h_copies=1 if rows else 0,
        validation_d2h_bytes=4 if rows else 0,
        validation_synchronizations=1 if rows else 0,
    )


def fixed_los_geometry_live(rows: FixedLoSRows) -> bool:
    """ADR-038 liveness for the raw frozen line-of-sight route.

    A zero-interaction row is a function of its two gathered endpoints alone, so
    this is the complete liveness question for that route. It is answered here,
    once, from the gathered rows, above any frequency-column loop that replays
    them.
    """

    return _ad_geometry_live(rows.source, rows.target)


def require_fixed_los_geometry_live(rows: FixedLoSRows, decided: bool) -> None:
    """Fail loudly if one column disagrees with the hoisted decision.

    The field facade keeps deciding liveness for itself - that is its ADR-038
    contract - and this makes "every column decides the same thing" a checked
    invariant instead of an assumption, before the operator runs.
    """

    if fixed_los_geometry_live(rows) != decided:
        raise RuntimeError(
            "fixed LoS geometry liveness disagrees with the decision taken "
            f"above the frequency-column loop (decided {decided}); every "
            "column must answer the same ADR-038 question"
        )


# --- Structural row selection for a prepared frozen topology ---------------

# Same bit vocabulary the native LoS validator publishes, so a caller reading
# the two error messages does not have to learn two encodings. Depth and
# component (bits 4 and 8) are host-validated once by ``prepare_fixed_topology``
# and therefore never fire here.
_INDEX_BOUNDS = 1
_PAIR_ORDER = 2
_STABLE_ID = 16
# A slot-batched row whose sink does not live in the slot its source does. The
# native validator has no such bit because slot batching is host-side
# structural packing; the number continues the same vocabulary so a caller
# reading a bitmask never has to learn two encodings.
_SLOT_BLOCK = 32

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
    slot_broken: torch.Tensor | None,
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
    error = (
        _bit((~in_bounds).any(), _INDEX_BOUNDS)
        | _bit(_order_violation(pair_index, in_bounds), _PAIR_ORDER)
        | _bit(identity_broken.any(), _STABLE_ID)
    )
    if slot_broken is None:
        return error
    return error | _bit((slot_broken & in_bounds).any(), _SLOT_BLOCK)


def _slot_pairing(
    source_row_index: torch.Tensor,
    sink_row_index: torch.Tensor,
    slot_sources: int,
    slot_sinks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-diagonal pair index, plus the rows that break the block law.

    A row belongs to the slot its SOURCE lives in. Its sink must live in the
    same slot, which is exactly the statement that the sink's within-slot index
    lands in ``[0, slot_sinks)``; a row that pairs across slots would otherwise
    silently land in another slot's pair segment.
    """

    slot = source_row_index.div(slot_sources, rounding_mode="floor")
    source_slot_index = source_row_index - slot * slot_sources
    sink_slot_index = sink_row_index - slot * slot_sinks
    pair_index = (
        slot * (slot_sources * slot_sinks)
        + sink_slot_index * slot_sources
        + source_slot_index
    )
    broken = (sink_slot_index < 0) | (sink_slot_index >= slot_sinks)
    return pair_index, broken


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
    *,
    slot_count: int = 1,
) -> PreparedRows:
    """Validate frozen rows and bind them to the current endpoint batches.

    ``slot_count`` selects the pairing law. One slot is the outer product the
    consumer has always published; more than one is the block-diagonal layout
    of :attr:`PropagationConvention.slot_pair_layout`, under which ``pair_count``
    grows linearly rather than quadratically in the slot count. The validation
    budget is unchanged: one four-byte copy and one synchronization for the
    whole batch, whatever the slot count.
    """

    if topology.device != sources.device or topology.device != sinks.device:
        raise ValueError("topology and endpoint tensors must share one CUDA device")
    assert sources.powers_w is not None
    rows = topology.row_count
    source_count = sources.count
    sink_count = sinks.count
    if rows > 0 and (source_count == 0 or sink_count == 0):
        raise ValueError("non-empty frozen topology requires endpoint rows")
    slot_sources = source_count // slot_count
    slot_sinks = sink_count // slot_count
    source_row_index = topology.source_index.to(dtype=torch.int64)
    sink_row_index = topology.sink_index.to(dtype=torch.int64)
    if slot_count == 1:
        pair_index = sink_row_index * source_count + source_row_index
        slot_broken = None
    else:
        # A zero per-slot count only ever accompanies an empty frozen batch,
        # rejected above when it is not; clamp the divisor so an empty batch
        # does not divide by zero on its way to an empty result.
        pair_index, slot_broken = _slot_pairing(
            source_row_index,
            sink_row_index,
            max(slot_sources, 1),
            max(slot_sinks, 1),
        )
    pair_count = slot_count * slot_sources * slot_sinks
    if rows > 0:
        error = int(
            _contract_error(
                topology,
                sources,
                sinks,
                source_row_index,
                sink_row_index,
                pair_index,
                slot_broken,
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
        pair_offsets=_pair_segmentation(pair_index, pair_count),
        validation_d2h_copies=VALIDATION_D2H_COPIES if rows else 0,
        validation_d2h_bytes=VALIDATION_D2H_BYTES if rows else 0,
        validation_synchronizations=VALIDATION_SYNCHRONIZATIONS if rows else 0,
    )


def select_rows(values: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Row selection that preserves frozen order and contiguity."""

    return values.index_select(0, rows).contiguous()


# --- Source amplitude ------------------------------------------------------


def excited_field(
    field_vector: torch.Tensor, tx_power: torch.Tensor, *, ad_mode: str
) -> torch.Tensor:
    """Return the source-excited complex3 field for a unit-excitation one.

    The field transport kernels carry ``sqrt(tx_power)`` into ``path_field`` and
    ``path_gain`` but leave their complex3 vector at unit excitation, so there
    is no excited vector on that launch. ADR-039 adds the native owner of
    exactly that quantity; this only chooses between its primal and
    differentiable entry points. No amplitude is computed here.
    """

    from witwin.channel.propagation.fields.kernels import (
        source_amplitude as field_amplitude,
    )

    if ad_mode == "none":
        return field_amplitude.field_source_amplitude_scale(
            field_vector, tx_power
        )["path_field_vector"]
    return field_amplitude.field_source_amplitude_scale_ad(field_vector, tx_power)


# --- Composed source-basis to sink-basis Jones operator --------------------

_FieldOp = Callable[[torch.Tensor], dict[str, torch.Tensor]]


def _primal(value: torch.Tensor) -> torch.Tensor:
    """Detached primal view for an input a native op consumes as a constant."""

    primal = torch.autograd.forward_ad.unpack_dual(value).primal
    return primal.detach().contiguous()


def transverse_basis(
    reference_basis: torch.Tensor,
    leg_origin: torch.Tensor,
    leg_target: torch.Tensor,
    *,
    frequency_hz: float,
) -> torch.Tensor:
    """Row-aligned orthonormal basis transverse to ``leg_target - leg_origin``.

    ``consumer_los_jones`` indexes its endpoint tables through ``pair_index``,
    so handing it per-row tables and the diagonal pair index makes it evaluate
    exactly one leg per row. The same reference basis is supplied for both
    endpoints, which makes its two published bases identical, and the source
    one is returned. This reuses the shipped native endpoint-basis owner
    instead of restating its projection and orthonormalization in Torch.
    """

    rows = int(leg_origin.shape[0])
    pair_index = torch.arange(
        rows, device=leg_origin.device, dtype=torch.int64
    ) * (rows + 1)
    reference = _primal(reference_basis)
    return consumer_los_jones(
        pair_index=pair_index,
        source_positions=_primal(leg_origin),
        sink_positions=_primal(leg_target),
        source_reference_basis=reference,
        sink_reference_basis=reference,
        frequency_hz=frequency_hz,
    ).source_basis


def _project(
    field_vector: torch.Tensor,
    direction: torch.Tensor,
    sink_vector: torch.Tensor,
) -> torch.Tensor:
    from witwin.channel.propagation.fields.kernels import (
        autograd_projection as field_projection,
    )

    return field_projection.field_project_complex3_ad(
        field_vector, direction, sink_vector
    )["coefficient"]


def compose_jones(
    excite: _FieldOp,
    *,
    source_basis: torch.Tensor,
    sink_reference_basis: torch.Tensor,
    arrival_origin: torch.Tensor,
    arrival_target: torch.Tensor,
    frequency_hz: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Build the ``(N, 2, 2)`` operator from two excitations of ``excite``.

    ``matrix[k, i, j]`` is the response of sink basis vector ``i`` to source
    basis vector ``j``, which is the index convention the native LoS Jones
    owner publishes. Returns the operator, the sink basis, and the first
    column's full field result so the caller can publish its geometry without
    re-running the transport.
    """

    columns = tuple(
        excite(source_basis[:, index].contiguous()) for index in (0, 1)
    )
    sink_basis = transverse_basis(
        sink_reference_basis,
        arrival_origin,
        arrival_target,
        frequency_hz=frequency_hz,
    )
    direction = columns[0]["direction"]
    matrix = torch.stack(
        [
            torch.stack(
                [
                    _project(
                        column["field_vector"],
                        direction,
                        sink_basis[:, index].contiguous(),
                    )
                    for column in columns
                ],
                dim=-1,
            )
            for index in (0, 1)
        ],
        dim=-2,
    )
    return matrix, sink_basis, columns[0]


# --- Prepared-topology bucket replay ---------------------------------------

_MATERIAL_FIELDS = ("eps_r", "sigma_e", "mu_r", "gain", "thickness")


@dataclass(frozen=True, slots=True)
class GeometryLiveness:
    """ADR-038 liveness, decided once above every loop that reuses it.

    ADR-038 requires the conditional differentiability of ``path_length_m`` and
    ``delay_s`` to be decided where forward duals are still visible, because
    ``Function.apply`` unpacks them before ``setup_context`` runs. A wideband
    request evaluates the same frozen rows at several frequencies, so it drives
    the same field operators repeatedly over identical geometry inputs.
    Deciding liveness inside that loop, or letting the first column decide for
    the rest, is exactly the shape of the defect ADR-038 removed.

    So the decision is made ONCE here, above the column loop, from the inputs
    every column shares, and :meth:`require` re-asserts it against the actual
    operator inputs of every bucket of every column. The field facades keep
    deciding for themselves - that is their ADR-038 contract and their frozen
    surface - and this record makes it a checked invariant that they all decide
    the same thing, rather than an assumption. A disagreement is a loud failure
    before the operator runs, never a column that silently answers a different
    question than its siblings.

    Two flags rather than one because the two bucket kinds consume different
    geometry: a zero-depth line-of-sight row is a function of the endpoints
    alone, while a reflection row additionally resolves its stationary point
    against the scene vertices, so a differentiable mesh makes the reflection
    geometry live even when the endpoints are primal.
    """

    los: bool
    reflection: bool
    # ADR-043: whether the published arrival direction carries a derivative.
    # This is a whole-result decision taken from the request's component set,
    # never a per-row one: a batch that carries a component whose direction
    # seam Channel does not own publishes a fully detached field_direction for
    # every row rather than a live derivative for some rows and a silent zero
    # for the rest.
    direction_components: bool = True

    @classmethod
    def of(
        cls,
        source: torch.Tensor,
        target: torch.Tensor,
        vertices: object,
        *,
        direction_components: bool = True,
    ) -> GeometryLiveness:
        endpoints = _ad_geometry_live(source, target)
        return cls(
            los=endpoints,
            reflection=endpoints or _ad_geometry_live(vertices),
            direction_components=direction_components,
        )

    def at_depth(self, depth: int) -> bool:
        return self.reflection if depth else self.los

    def direction_at_depth(self, depth: int) -> bool:
        """Direction liveness for one bucket, under both decisions.

        A direction derivative is a geometry derivative, so it can only be live
        where the geometry is. The component half is host-known and identical
        for every bucket and every column, which is what makes the whole-result
        rule hold without a second device observation.
        """

        return self.at_depth(depth) and self.direction_components

    def require(self, depth: int, *values: object) -> None:
        """Fail loudly if one column's inputs disagree with the decision."""

        decided = self.at_depth(depth)
        if _ad_geometry_live(*values) != decided:
            raise RuntimeError(
                "fixed-topology geometry liveness disagrees with the decision "
                f"taken above the frequency-column loop (depth {depth}, "
                f"decided {decided}); every column must answer the same ADR-038 "
                "question"
            )


@dataclass(frozen=True, slots=True)
class BucketInputs:
    """Everything one bucket hands to the native transport."""

    depth: int
    source: torch.Tensor
    target: torch.Tensor
    tx_power: torch.Tensor
    tx_polarization: torch.Tensor
    rx_polarization: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor
    material: tuple[torch.Tensor, ...]
    valid: torch.Tensor

    @property
    def row_count(self) -> int:
        return int(self.source.shape[0])


@dataclass(frozen=True, slots=True)
class FixedRowOutputs:
    """Frozen-order ``K`` row outputs of one reevaluation.

    ``path_field`` and ``path_field_vector`` are the source-excited transport,
    matching what discovery publishes. ``path_field_vector`` is only produced
    for the complex3 response, which is the only reader of it.
    """

    path_field: torch.Tensor
    path_field_vector: torch.Tensor | None
    path_length_m: torch.Tensor
    delay_s: torch.Tensor
    direction: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor
    matrix: torch.Tensor | None
    source_basis: torch.Tensor | None
    sink_basis: torch.Tensor | None
    row_valid: torch.Tensor | None


def require_smooth_reflection_scene(compiled: CompiledScene) -> None:
    """Reject a scene whose reflection rows need rough-surface attenuation.

    The coherent rough-reflection factor and the realization phase-screen delta
    replacement are owned by the discovery-side field loop and are gated on
    host material state there. Reproducing that gate here would duplicate
    another owner's policy, and silently disagreeing with ``evaluate`` on a
    rough scene is worse than refusing, so this route requires a smooth scene.
    """

    if bool((compiled.materials.scatter_model_id == 1).any()):
        raise NotImplementedError(
            "fixed-topology reflection reevaluation requires a smooth scene; "
            "rough-surface coherent attenuation is owned by the discovery "
            "field loop and is not reproduced here"
        )
    screens = getattr(compiled.assignments, "structure_phase_screens", {})
    if any(
        getattr(screen, "mode", None) == "realization_coherent"
        for screen in screens.values()
    ):
        raise NotImplementedError(
            "fixed-topology reflection reevaluation does not support "
            "realization_coherent phase screens"
        )


def _scene_tables(compiled: CompiledScene) -> dict[str, object]:
    # CompiledScene owns the lazy scene-static cache (Plan-13 pattern); the
    # replay just consumes it. The primal per-frame case stages host-to-device
    # once per compiled scene instead of once per call.
    return compiled.fixed_reevaluation_tables()


def _reflection_inputs(
    compiled: CompiledScene,
    bucket: FixedTopologyBucket,
    tables: dict[str, object],
    prepared: PreparedFixedTopology,
    rows: PreparedRows,
) -> BucketInputs:
    from witwin.channel.propagation.geometry.reevaluate import (
        reflection_epc_paths,
    )

    depth = bucket.depth
    sequence = select_rows(
        prepared.topology.primitive_sequence, bucket.rows
    )[:, :depth].contiguous()
    face_id = sequence.to(dtype=torch.int64)
    source = select_rows(rows.source, bucket.rows)
    target = select_rows(rows.target, bucket.rows)
    epc = reflection_epc_paths(
        compiled, tables["vertices"], source, target, face_id, depth
    )
    valid = epc["valid"]
    material = tables["material"]
    return BucketInputs(
        depth=depth,
        source=source,
        target=target,
        tx_power=select_rows(rows.tx_power, bucket.rows),
        tx_polarization=_inert_where_invalid(
            select_rows(rows.tx_polarization, bucket.rows), valid
        ),
        rx_polarization=select_rows(rows.rx_polarization, bucket.rows),
        interaction_positions=epc["hit_positions"],
        interaction_normals=epc["normals"],
        material=tuple(material[name][face_id].contiguous() for name in _MATERIAL_FIELDS),
        valid=valid,
    )


def _los_inputs(
    compiled: CompiledScene, bucket: FixedTopologyBucket, rows: PreparedRows
) -> BucketInputs:
    source = select_rows(rows.source, bucket.rows)
    target = select_rows(rows.target, bucket.rows)
    empty = source.new_empty((int(source.shape[0]), 0, 3))
    # Re-test visibility with the same native gate discovery applies to LoS
    # candidates, so a sink that moved behind a wall publishes row_valid=False
    # and exact zeros instead of a full-strength free-space answer. A
    # structure-less scene cannot occlude anything and skips the launch.
    if compiled.structures:
        from witwin.channel.propagation.geometry.visibility import (
            VisibilityQuery,
            run_visibility_query,
        )

        valid = run_visibility_query(
            VisibilityQuery(
                rayd=compiled.rayd,
                start=source.contiguous(),
                end=target.contiguous(),
                active=None,
            )
        ).visible
    else:
        valid = torch.ones(
            (int(source.shape[0]),), dtype=torch.bool, device=source.device
        )
    return BucketInputs(
        depth=0,
        source=source,
        target=target,
        tx_power=select_rows(rows.tx_power, bucket.rows),
        tx_polarization=_inert_where_invalid(
            select_rows(rows.tx_polarization, bucket.rows), valid
        ),
        rx_polarization=select_rows(rows.rx_polarization, bucket.rows),
        interaction_positions=empty,
        interaction_normals=empty,
        material=(),
        valid=valid,
    )


def _inert_where_invalid(
    values: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Select a row's value or the inert constant; never a numerical blend."""

    shape = (-1, *((1,) * (values.ndim - 1)))
    return torch.where(valid.reshape(shape), values, values.new_zeros(()))


def _field_op(
    inputs: BucketInputs,
    *,
    ad_mode: str,
    frequency: float | torch.Tensor,
    frequency_value: float,
    geometry_live: GeometryLiveness | None,
    ledger: object | None = None,
) -> Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
    from witwin.channel.propagation.fields.kernels import (
        autograd as field_autograd,
    )
    from witwin.channel.propagation.fields.kernels import (
        functional as field_functional,
    )

    differentiable = ad_mode != "none"

    def run(
        tx_polarization: torch.Tensor, rx_polarization: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if inputs.depth == 0:
            leading = (inputs.source, inputs.target)
            trailing: tuple[torch.Tensor, ...] = ()
        else:
            leading = (
                inputs.source,
                inputs.target,
                inputs.interaction_positions,
                inputs.interaction_normals,
            )
            trailing = inputs.material
        if geometry_live is not None:
            # The liveness these exact inputs imply must equal the one decided
            # above the frequency-column loop, for every bucket of every column.
            geometry_live.require(inputs.depth, *leading)
        arguments = (
            *leading,
            inputs.tx_power,
            tx_polarization,
            rx_polarization,
            *trailing,
        )
        if differentiable:
            operator = (
                field_autograd.field_free_space_ad
                if inputs.depth == 0
                else field_autograd.field_reflection_sequence_ad
            )
            if ledger is not None:
                ledger.add(*arguments)
            return operator(
                *arguments,
                frequency=frequency,
                frequency_value=frequency_value,
                direction_live=(
                    geometry_live is not None
                    and geometry_live.direction_at_depth(inputs.depth)
                ),
            )
        operator = (
            field_functional.field_free_space
            if inputs.depth == 0
            else field_functional.field_reflection_sequence
        )
        return operator(*arguments, frequency_hz=frequency_value)

    return run


def _leg_endpoints(inputs: BucketInputs) -> tuple[torch.Tensor, torch.Tensor]:
    """Where the first leg ends and where the last leg starts.

    A reflection row launches toward its first interaction and arrives from
    its last one, so its transverse bases live in two different planes. Both
    are read off the interaction table the native transport itself consumes;
    neither direction is recomputed here.
    """

    if inputs.depth == 0:
        return inputs.target, inputs.source
    return (
        inputs.interaction_positions[:, 0].contiguous(),
        inputs.interaction_positions[:, inputs.depth - 1].contiguous(),
    )


def _jones_values(
    inputs: BucketInputs,
    run: Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]],
    *,
    source_reference: torch.Tensor,
    sink_reference: torch.Tensor,
    frequency_value: float,
) -> dict[str, torch.Tensor]:
    launch_target, arrival_origin = _leg_endpoints(inputs)
    source_basis = _inert_where_invalid(
        transverse_basis(
            source_reference,
            inputs.source,
            launch_target,
            frequency_hz=frequency_value,
        ),
        inputs.valid,
    )
    matrix, sink_basis, column = compose_jones(
        lambda polarization: run(polarization, polarization),
        source_basis=source_basis,
        sink_reference_basis=sink_reference,
        arrival_origin=arrival_origin,
        arrival_target=inputs.target,
        frequency_hz=frequency_value,
    )
    return {
        **column,
        "matrix": matrix,
        "source_basis": source_basis,
        "sink_basis": _inert_where_invalid(sink_basis, inputs.valid),
    }


def _bucket_values(
    inputs: BucketInputs,
    run: Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]],
    *,
    response: str,
    ad_mode: str,
    source_reference: torch.Tensor | None,
    sink_reference: torch.Tensor | None,
    frequency_value: float,
    ledger: object | None = None,
) -> dict[str, torch.Tensor]:
    if response != "polarimetric_transport":
        values = run(inputs.tx_polarization, inputs.rx_polarization)
    else:
        assert source_reference is not None and sink_reference is not None
        values = _jones_values(
            inputs,
            run,
            source_reference=source_reference,
            sink_reference=sink_reference,
            frequency_value=frequency_value,
        )
    if response != "complex3_transport":
        return values
    if ledger is not None and ad_mode != "none":
        ledger.add(values["field_vector"], inputs.tx_power)
    return {
        **values,
        "path_field_vector": excited_field(
            values["field_vector"], inputs.tx_power, ad_mode=ad_mode
        ),
    }


def _pad_interactions(values: torch.Tensor, width: int) -> torch.Tensor:
    depth = int(values.shape[1])
    if depth == width:
        return values
    padding = values.new_zeros((int(values.shape[0]), width - depth, 3))
    return torch.cat((values, padding), dim=1)


def _publish_bucket(
    outputs: dict[str, torch.Tensor],
    values: dict[str, torch.Tensor],
    inputs: BucketInputs,
    bucket: FixedTopologyBucket,
    width: int,
) -> None:
    rows = bucket.rows
    valid = inputs.valid
    outputs["path_field"].index_copy_(0, rows, values["path_field"])
    if outputs["path_field_vector"] is not None:
        outputs["path_field_vector"].index_copy_(
            0, rows, values["path_field_vector"]
        )
    outputs["path_length_m"].index_copy_(
        0, rows, _inert_where_invalid(values["path_length_m"], valid)
    )
    outputs["delay_s"].index_copy_(
        0, rows, _inert_where_invalid(values["delay_s"], valid)
    )
    outputs["direction"].index_copy_(
        0, rows, _inert_where_invalid(values["direction"], valid)
    )
    outputs["interaction_positions"].index_copy_(
        0, rows, _pad_interactions(inputs.interaction_positions, width)
    )
    outputs["interaction_normals"].index_copy_(
        0, rows, _pad_interactions(inputs.interaction_normals, width)
    )
    outputs["row_valid"].index_copy_(0, rows, valid)
    for name in ("matrix", "source_basis", "sink_basis"):
        if outputs[name] is not None:
            outputs[name].index_copy_(0, rows, values[name])


def _allocate(
    rows: PreparedRows, width: int, response: str
) -> dict[str, torch.Tensor | None]:
    count = rows.row_count
    device = rows.source.device
    polarimetric = response == "polarimetric_transport"
    return {
        "path_field": torch.zeros(
            (count,), dtype=torch.complex64, device=device
        ),
        "path_field_vector": (
            torch.zeros((count, 3), dtype=torch.complex64, device=device)
            if response == "complex3_transport"
            else None
        ),
        "path_length_m": torch.zeros((count,), device=device),
        "delay_s": torch.zeros((count,), device=device),
        "direction": torch.zeros((count, 3), device=device),
        "interaction_positions": torch.zeros((count, width, 3), device=device),
        "interaction_normals": torch.zeros((count, width, 3), device=device),
        "row_valid": torch.ones((count,), dtype=torch.bool, device=device),
        "matrix": (
            torch.zeros((count, 2, 2), dtype=torch.complex64, device=device)
            if polarimetric
            else None
        ),
        "source_basis": (
            torch.zeros((count, 2, 3), device=device) if polarimetric else None
        ),
        "sink_basis": (
            torch.zeros((count, 2, 3), device=device) if polarimetric else None
        ),
    }


def evaluate_prepared(
    compiled: CompiledScene,
    prepared: PreparedFixedTopology,
    rows: PreparedRows,
    *,
    response: str,
    ad_mode: str,
    frequency: float | torch.Tensor,
    frequency_value: float,
    source_reference_basis: torch.Tensor | None,
    sink_reference_basis: torch.Tensor | None,
    publish_row_validity: bool,
    geometry_live: GeometryLiveness | None = None,
    ledger: object | None = None,
) -> FixedRowOutputs:
    """Replay every host-known bucket of a prepared frozen topology.

    ``geometry_live`` carries an ADR-038 liveness decision the caller took
    before this call. A wideband caller runs this replay once per frequency
    column over identical geometry inputs and passes the SAME record to every
    column, which turns "every column answers the same liveness question" into a
    checked invariant. A single-frequency caller leaves it ``None``.
    """

    width = int(prepared.topology.primitive_sequence.shape[1])
    outputs = _allocate(rows, width, response)
    tables = (
        _scene_tables(compiled)
        if any(bucket.depth > 0 for bucket in prepared.buckets)
        else {}
    )
    for bucket in prepared.buckets:
        inputs = (
            _los_inputs(compiled, bucket, rows)
            if bucket.depth == 0
            else _reflection_inputs(compiled, bucket, tables, prepared, rows)
        )
        values = _bucket_values(
            inputs,
            _field_op(
                inputs,
                ad_mode=ad_mode,
                frequency=frequency,
                frequency_value=frequency_value,
                geometry_live=geometry_live,
                ledger=ledger,
            ),
            response=response,
            ad_mode=ad_mode,
            source_reference=(
                None
                if source_reference_basis is None
                else select_rows(source_reference_basis, bucket.rows)
            ),
            sink_reference=(
                None
                if sink_reference_basis is None
                else select_rows(sink_reference_basis, bucket.rows)
            ),
            frequency_value=frequency_value,
            ledger=ledger,
        )
        _publish_bucket(outputs, values, inputs, bucket, width)
    return FixedRowOutputs(
        path_field=outputs["path_field"],
        path_field_vector=outputs["path_field_vector"],
        path_length_m=outputs["path_length_m"],
        delay_s=outputs["delay_s"],
        direction=outputs["direction"],
        interaction_positions=outputs["interaction_positions"],
        interaction_normals=outputs["interaction_normals"],
        matrix=outputs["matrix"],
        source_basis=outputs["source_basis"],
        sink_basis=outputs["sink_basis"],
        row_valid=outputs["row_valid"] if publish_row_validity else None,
    )


def scene_vertex_table(compiled: CompiledScene) -> object:
    """The scene-static vertex tensor the reflection stationary point uses.

    Exposed so a caller can decide ADR-038 geometry liveness before the first
    bucket runs without reaching past this module into the compiled scene's
    lazy table cache. Only a batch that actually carries a reflection bucket
    should ask: the tables are built lazily, and a line-of-sight replay must not
    start paying for them.
    """

    return _scene_tables(compiled)["vertices"]


__all__ = [
    "COMPACT_COUNT_D2H_BYTES",
    "COMPACT_COUNT_D2H_COPIES",
    "COMPACT_COUNT_SYNCHRONIZATIONS",
    "VALIDATION_D2H_BYTES",
    "VALIDATION_D2H_COPIES",
    "VALIDATION_SYNCHRONIZATIONS",
    "BucketInputs",
    "CompactEvaluatedPaths",
    "FixedLoSRows",
    "FixedRowOutputs",
    "GeometryLiveness",
    "LoSJonesRows",
    "PreparedRows",
    "compact_evaluated_paths",
    "compose_jones",
    "consumer_los_jones",
    "evaluate_prepared",
    "excited_field",
    "fixed_los_gather",
    "fixed_los_geometry_live",
    "prepared_row_gather",
    "require_fixed_los_geometry_live",
    "require_smooth_reflection_scene",
    "scene_vertex_table",
    "select_rows",
    "transverse_basis",
]
