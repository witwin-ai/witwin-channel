"""ADR-032 canonical exact-row owner and pair segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.channel.propagation.topology.kernels.blocks import (
    _DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA,
    _PATH_BLOCK_SCHEMA,
    _validate_deterministic_topology_block,
)
from witwin.channel.propagation.topology.kernels.compact_autograd import (
    evaluated_paths_compact_finalize_backward,
    evaluated_paths_compact_finalize_jvp,
)
from witwin.channel.runtime import torch_compat
from witwin.channel.runtime.autograd_contracts import (
    _ad_first_order_only,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
)
from witwin.channel.runtime.symbols import required_symbol as _required_native_op
from witwin.channel.runtime.tensor_contracts import validate_cuda_tensor


_BLOCK_FIELDS = tuple(
    name
    for name, _dtype in (
        *_PATH_BLOCK_SCHEMA,
        *_DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA,
    )
)
_EXTRA_CONTINUOUS_FIELDS = ("field_xyz", "coefficient")
_FORWARD_BLOCK_FIELDS = (*_BLOCK_FIELDS, *_EXTRA_CONTINUOUS_FIELDS)
_DISCRETE_INPUT_FIELDS = (
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
)
_CONTINUOUS_INPUT_FIELDS = (
    "path_length_m",
    "delay_s",
    "interaction_position",
    "interaction_normal",
    "interaction_positions",
    "interaction_normals",
    "path_gain",
    "path_field",
    "field_xyz",
    "coefficient",
)
_STRUCTURAL_OUTPUT_FIELDS = (
    "selected_row_index",
    "pair_index",
    "pair_offsets",
    "source_id",
    "sink_id",
)
_FUNCTION_OUTPUT_FIELDS = (
    *_STRUCTURAL_OUTPUT_FIELDS,
    *_DISCRETE_INPUT_FIELDS,
    *_CONTINUOUS_INPUT_FIELDS,
)
_DISCRETE_OUTPUT_COUNT = len(_STRUCTURAL_OUTPUT_FIELDS) + len(
    _DISCRETE_INPUT_FIELDS
)


class _EnumeratedCanonicalCompactFunction(torch.autograd.Function):
    @staticmethod
    def forward(*inputs):
        block = {
            name: value
            for name, value in zip(
                (*_DISCRETE_INPUT_FIELDS, *_CONTINUOUS_INPUT_FIELDS),
                inputs[:20],
                strict=True,
            )
        }
        (
            source_stable_ids,
            sink_stable_ids,
            pair_count,
            num_tx,
            num_rx,
            max_paths,
            scope,
            sequence_width,
        ) = inputs[20:]
        raw = _required_native_op("enumerated_canonical_compact")(
            block,
            int(pair_count),
            int(num_tx),
            int(num_rx),
            int(max_paths),
            int(scope),
            int(sequence_width),
            source_stable_ids,
            sink_stable_ids,
        )
        expected = {
            *_FORWARD_BLOCK_FIELDS,
            *_STRUCTURAL_OUTPUT_FIELDS,
            "path_count",
            "count_d2h_copies",
            "count_d2h_bytes",
            "count_synchronizations",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise TypeError("native canonical compact owner returned bad fields")
        return tuple(raw[name] for name in _FUNCTION_OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        ctx.candidate_count = int(inputs[0].shape[0])
        ctx.sequence_width = int(inputs[27])
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
        input_grads: list[torch.Tensor | None] = [None] * 28
        continuous_grads = grad_outputs[_DISCRETE_OUTPUT_COUNT:]
        if all(value is None for value in continuous_grads):
            return tuple(input_grads)
        if not any(ctx.needs_input_grad[10:20]):
            return tuple(input_grads)
        valid, selected_row_index = ctx.saved_tensors
        native_grads = (
            continuous_grads[0],
            continuous_grads[1],
            None,
            *continuous_grads[2:],
        )
        raw = evaluated_paths_compact_finalize_backward(
            valid,
            selected_row_index,
            *native_grads,
            candidate_count=ctx.candidate_count,
            sequence_width=ctx.sequence_width,
        )
        expected = {
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
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise TypeError("native canonical compact backward returned bad fields")
        for index, name in enumerate(_CONTINUOUS_INPUT_FIELDS, start=10):
            if ctx.needs_input_grad[index]:
                input_grads[index] = raw[name]
        return tuple(input_grads)

    @staticmethod
    def jvp(ctx, *tangents):
        continuous_tangents = tuple(
            _ad_native_tangent_or_none(value) for value in tangents[10:20]
        )
        if all(value is None for value in continuous_tangents):
            return (None,) * len(_FUNCTION_OUTPUT_FIELDS)
        valid, selected_row_index = (
            _ad_native_tensor(value) for value in ctx.saved_tensors
        )
        native_tangents = (
            continuous_tangents[0],
            continuous_tangents[1],
            None,
            *continuous_tangents[2:],
        )
        with torch_compat.disable_functorch():
            raw = evaluated_paths_compact_finalize_jvp(
                valid,
                selected_row_index,
                *native_tangents,
                candidate_count=ctx.candidate_count,
                sequence_width=ctx.sequence_width,
            )
        expected = {
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
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise TypeError("native canonical compact JVP returned bad fields")
        return (
            *(None for _ in range(_DISCRETE_OUTPUT_COUNT)),
            *(raw[name] for name in _CONTINUOUS_INPUT_FIELDS),
        )


@dataclass(frozen=True, slots=True)
class CanonicalCompactRows:
    block: dict[str, torch.Tensor]
    selected_row_index: torch.Tensor
    pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    source_id: torch.Tensor | None
    sink_id: torch.Tensor | None
    path_count: int
    count_d2h_copies: int
    count_d2h_bytes: int
    count_synchronizations: int


@dataclass(frozen=True, slots=True)
class ExactPairMetadata:
    pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    source_id: torch.Tensor | None
    sink_id: torch.Tensor | None
    path_count: int
    count_d2h_copies: int
    count_d2h_bytes: int
    count_synchronizations: int


def _validate_counts(
    *,
    pair_count: int,
    num_tx: int,
    num_rx: int,
) -> None:
    if min(pair_count, num_tx, num_rx) < 0:
        raise ValueError("endpoint counts must be non-negative")
    if pair_count != num_tx * num_rx:
        raise ValueError("pair_count must equal num_tx * num_rx")


def _validate_stable_ids(
    source_stable_ids: torch.Tensor | None,
    sink_stable_ids: torch.Tensor | None,
    *,
    reference: torch.Tensor,
    num_tx: int,
    num_rx: int,
) -> None:
    if (source_stable_ids is None) != (sink_stable_ids is None):
        raise ValueError("source and sink stable IDs must be provided together")
    if source_stable_ids is None:
        return
    assert sink_stable_ids is not None
    for name, value, count in (
        ("source_stable_ids", source_stable_ids, num_tx),
        ("sink_stable_ids", sink_stable_ids, num_rx),
    ):
        validate_cuda_tensor(name, value, dtype=torch.int64, ndim=1)
        if value.device != reference.device or value.shape != (count,):
            raise ValueError(f"{name} must have shape ({count},) on the path device")


def _metadata_from_raw(
    raw: dict[str, object],
    *,
    pair_count: int,
    stable_ids_requested: bool,
) -> ExactPairMetadata:
    expected = {
        "pair_index",
        "pair_offsets",
        "source_id",
        "sink_id",
        "path_count",
        "count_d2h_copies",
        "count_d2h_bytes",
        "count_synchronizations",
    }
    if set(raw) != expected:
        raise TypeError("native pair metadata owner returned bad fields")
    path_count = raw["path_count"]
    if type(path_count) is not int:
        raise TypeError("native pair metadata owner returned a non-int path_count")
    pair_index = raw["pair_index"]
    pair_offsets = raw["pair_offsets"]
    source_id = raw["source_id"]
    sink_id = raw["sink_id"]
    if not all(
        isinstance(value, torch.Tensor)
        for value in (pair_index, pair_offsets, source_id, sink_id)
    ):
        raise TypeError("native pair metadata owner returned non-tensor fields")
    assert isinstance(pair_index, torch.Tensor)
    assert isinstance(pair_offsets, torch.Tensor)
    assert isinstance(source_id, torch.Tensor)
    assert isinstance(sink_id, torch.Tensor)
    for name, value, shape in (
        ("pair_index", pair_index, (path_count,)),
        ("pair_offsets", pair_offsets, (pair_count + 1,)),
    ):
        validate_cuda_tensor(name, value, dtype=torch.int64, ndim=1)
        if value.shape != shape:
            raise RuntimeError(f"native {name} has shape {tuple(value.shape)}, expected {shape}")
    expected_ids = path_count if stable_ids_requested else 0
    if source_id.shape != (expected_ids,) or sink_id.shape != (expected_ids,):
        raise RuntimeError("native stable ID metadata has an invalid row count")
    return ExactPairMetadata(
        pair_index=pair_index,
        pair_offsets=pair_offsets,
        source_id=source_id if stable_ids_requested else None,
        sink_id=sink_id if stable_ids_requested else None,
        path_count=path_count,
        count_d2h_copies=int(raw["count_d2h_copies"]),
        count_d2h_bytes=int(raw["count_d2h_bytes"]),
        count_synchronizations=int(raw["count_synchronizations"]),
    )


def enumerated_canonical_compact(
    block: dict[str, torch.Tensor],
    *,
    pair_count: int,
    num_tx: int,
    num_rx: int,
    max_paths: int | None,
    max_paths_scope: str,
    sequence_width: int,
    source_stable_ids: torch.Tensor | None = None,
    sink_stable_ids: torch.Tensor | None = None,
) -> CanonicalCompactRows:
    """Select, deduplicate, limit, and gather exact rows in one native owner."""

    _validate_counts(pair_count=pair_count, num_tx=num_tx, num_rx=num_rx)
    normalized = dict(block)
    _validate_deterministic_topology_block("block", normalized, sequence_width)
    row_count = int(normalized["valid"].shape[0])
    if "field_xyz" not in normalized:
        normalized["field_xyz"] = torch.zeros(
            (row_count, 3),
            device=normalized["valid"].device,
            dtype=torch.complex64,
        )
    if "coefficient" not in normalized:
        normalized["coefficient"] = normalized["path_field"]
    _validate_stable_ids(
        source_stable_ids,
        sink_stable_ids,
        reference=normalized["valid"],
        num_tx=num_tx,
        num_rx=num_rx,
    )
    if max_paths is not None and max_paths <= 0:
        raise ValueError("max_paths must be positive")
    scope = {"global": 0, "per_pair": 1}.get(max_paths_scope)
    if scope is None:
        raise ValueError("max_paths_scope must be 'global' or 'per_pair'")
    outputs = _EnumeratedCanonicalCompactFunction.apply(
        *(normalized[name] for name in _DISCRETE_INPUT_FIELDS),
        *(normalized[name] for name in _CONTINUOUS_INPUT_FIELDS),
        source_stable_ids,
        sink_stable_ids,
        int(pair_count),
        int(num_tx),
        int(num_rx),
        -1 if max_paths is None else int(max_paths),
        int(scope),
        int(sequence_width),
    )
    raw = dict(zip(_FUNCTION_OUTPUT_FIELDS, outputs, strict=True))
    gathered = {name: raw[name] for name in _FORWARD_BLOCK_FIELDS}
    _validate_deterministic_topology_block(
        "enumerated_canonical_compact", gathered, sequence_width
    )
    path_count = int(raw["pair_index"].shape[0])
    has_candidates = row_count > 0
    metadata = _metadata_from_raw(
        {
            "pair_index": raw["pair_index"],
            "pair_offsets": raw["pair_offsets"],
            "source_id": raw["source_id"],
            "sink_id": raw["sink_id"],
            "path_count": path_count,
            "count_d2h_copies": 1 if has_candidates else 0,
            "count_d2h_bytes": 8 if has_candidates else 0,
            "count_synchronizations": 1 if has_candidates else 0,
        },
        pair_count=pair_count,
        stable_ids_requested=source_stable_ids is not None,
    )
    selected_row_index = raw["selected_row_index"]
    validate_cuda_tensor(
        "selected_row_index", selected_row_index, dtype=torch.int64, ndim=1
    )
    if selected_row_index.shape != (metadata.path_count,):
        raise RuntimeError("selected_row_index does not match exact K")
    if gathered["valid"].shape != (metadata.path_count,):
        raise RuntimeError("canonical compact payload does not match exact K")
    return CanonicalCompactRows(
        block=gathered,
        selected_row_index=selected_row_index,
        pair_index=metadata.pair_index,
        pair_offsets=metadata.pair_offsets,
        source_id=metadata.source_id,
        sink_id=metadata.sink_id,
        path_count=metadata.path_count,
        count_d2h_copies=metadata.count_d2h_copies,
        count_d2h_bytes=metadata.count_d2h_bytes,
        count_synchronizations=metadata.count_synchronizations,
    )


def enumerated_exact_pair_metadata(
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    *,
    pair_count: int,
    num_tx: int,
    num_rx: int,
    source_stable_ids: torch.Tensor | None = None,
    sink_stable_ids: torch.Tensor | None = None,
) -> ExactPairMetadata:
    """Attach pair metadata to trusted exact rows without another count read."""

    _validate_counts(pair_count=pair_count, num_tx=num_tx, num_rx=num_rx)
    for name, value in (("tx_id", tx_id), ("rx_id", rx_id)):
        validate_cuda_tensor(name, value, dtype=torch.int32, ndim=1)
    if tx_id.shape != rx_id.shape or tx_id.device != rx_id.device:
        raise ValueError("tx_id and rx_id must share shape and device")
    _validate_stable_ids(
        source_stable_ids,
        sink_stable_ids,
        reference=tx_id,
        num_tx=num_tx,
        num_rx=num_rx,
    )
    raw = _required_native_op("enumerated_exact_pair_metadata")(
        tx_id,
        rx_id,
        int(pair_count),
        int(num_tx),
        int(num_rx),
        source_stable_ids,
        sink_stable_ids,
    )
    if not isinstance(raw, dict):
        raise TypeError("native exact-row metadata owner returned a non-dict")
    return _metadata_from_raw(
        raw,
        pair_count=pair_count,
        stable_ids_requested=source_stable_ids is not None,
    )


__all__ = [
    "CanonicalCompactRows",
    "ExactPairMetadata",
    "enumerated_canonical_compact",
    "enumerated_exact_pair_metadata",
]
