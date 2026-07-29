# Copyright Xingyu Chen.
# Native discrete path-topology kernel facades.

"""Native discrete path-topology kernel facades.

Thin facades over the ``_channel`` topology ABI: the path/topology block
schemas and their validators, the reflection candidate export, the native
compaction owners, the topology construction primitives, the small packing
primitives, the Monte Carlo direction sampler, the shared compact-autograd
companions, the ADR-032 canonical exact-row owner, and the ADR-027
component-5 transmission topology packer.

blocks
------
The ``_PATH_BLOCK_SCHEMA`` / ``_DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA`` field
contracts, the validators every other section reuses, and the LOS export,
filter, merge and finalize block operations.

candidates
----------
The reflection candidate export, which publishes a path block plus the two
visibility segments the caller traces.

compaction
----------
The native order-1 reflection, reflection-sequence and order-1 diffraction
compaction owners plus the deterministic row sort order.

construction
------------
The topology block constructors: LOS blocks, default and base fields,
sequence padding, EPC input batches, and the face-sequence chunk generators.

primitives
----------
Small native packing and counting primitives shared across the topology
stage.

sampling
--------
The Monte Carlo direction sampler that ``propagation.topology`` publishes as
its single sampling owner.

compact autograd
----------------
Shared native companions for exact-row compact autograd.

canonical compact
-----------------
ADR-032 canonical exact-row owner and pair segmentation.

transmission
------------
Native component-5 topology packing for RayD segment penetration. This facade
owns both the dispatch and the fixed-capacity row table it publishes:
``TransmissionTopologyCapacity`` is the named typed contract this one native
operation converts its result into, and it has no other producer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from witwin.channel.propagation.penetration import (
    SegmentPenetrationResult,
)
from witwin.channel.runtime import (
    CapacityExecutionCounts,
    CapacityFailureState,
    _ad_first_order_only,
    _ad_native_tangent_or_none,
    _ad_native_tensor,
    disable_functorch,
    require_capacity_failure_state,
    require_host_count,
    require_tensor,
    required_symbol as _required_native_op,
    validate_cuda_tensor,
)

__all__ = [
    "_DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA",
    "_EnumeratedTransmissionTopologyPackFunction",
    "_PATH_BLOCK_SCHEMA",
    "_validate_deterministic_topology_block",
    "_validate_path_block",
    "_validate_path_reflection_candidates",
    "_validate_topology_extra_fields",
    "COMPACT_CONTINUOUS_FIELDS",
    "CanonicalCompactRows",
    "ExactPairMetadata",
    "TransmissionTopologyCapacity",
    "core_pack_int2",
    "deterministic_component_counts",
    "deterministic_concat_topology_blocks",
    "deterministic_diffraction_order1_compact",
    "deterministic_diffraction_state_pack",
    "deterministic_diffraction_state_pack_selected",
    "deterministic_face_anchor_points",
    "deterministic_face_sequence_chunk",
    "deterministic_gather_topology_block",
    "deterministic_los_topology_block",
    "deterministic_mapped_face_sequence_chunk",
    "deterministic_pad_topology_sequences",
    "deterministic_reflection_epc_input_batch",
    "deterministic_reflection_order1_compact",
    "deterministic_reflection_sequence_compact",
    "deterministic_repeat_range",
    "deterministic_selected_edge_count",
    "deterministic_sort_order",
    "deterministic_topology_base_fields",
    "deterministic_topology_default_fields",
    "enumerated_canonical_compact",
    "enumerated_exact_pair_metadata",
    "enumerated_transmission_topology_pack",
    "evaluated_paths_compact_finalize_backward",
    "evaluated_paths_compact_finalize_jvp",
    "mc_sample_directions",
    "mc_selected_edge_indices",
    "path_concat_vec3",
    "path_diffraction_block",
    "path_filter_block",
    "path_filter_los",
    "path_finalize_blocks",
    "path_los_export",
    "path_los_visibility_inputs",
    "path_merge_blocks",
    "path_reflection_candidates",
]


# -------------------------------------------------------------------------
# blocks
# -------------------------------------------------------------------------
def path_los_export(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_polarizations: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "tx_polarizations",
        tx_polarizations,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    if tx_power.shape[0] != tx_positions.shape[0]:
        raise ValueError("tx_power must have one value per transmitter")
    if tx_polarizations.shape != tx_positions.shape:
        raise ValueError("tx_polarizations must match tx_positions shape")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    exported = _required_native_op("path_los_export")(
        tx_positions, tx_power, rx_positions, float(frequency_hz), tx_polarizations
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel.path_los_export must return a dict")
    return exported


_PATH_BLOCK_SCHEMA: tuple[tuple[str, torch.dtype], ...] = (
    ("valid", torch.bool),
    ("tx_id", torch.int32),
    ("rx_id", torch.int32),
    ("depth", torch.int32),
    ("component_id", torch.int32),
    ("primitive_id", torch.int32),
    ("edge_id", torch.int32),
    ("path_length_m", torch.float32),
    ("delay_s", torch.float32),
    ("path_gain", torch.float32),
)


_DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA: tuple[tuple[str, torch.dtype], ...] = (
    ("path_field", torch.complex64),
    ("interaction_position", torch.float32),
    ("interaction_normal", torch.float32),
    ("material_id", torch.int32),
    ("primitive_sequence", torch.int32),
    ("material_sequence", torch.int32),
    ("interaction_positions", torch.float32),
    ("interaction_normals", torch.float32),
)


def _validate_path_block(name: str, block: dict[str, torch.Tensor]) -> None:
    if not isinstance(block, dict):
        raise TypeError(f"{name} must be a dict")
    expected_shape: tuple[int, ...] | None = None
    for key, dtype in _PATH_BLOCK_SCHEMA:
        tensor = block.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}.{key} must be a torch.Tensor")
        validate_cuda_tensor(f"{name}.{key}", tensor, dtype=dtype, ndim=1)
        if expected_shape is None:
            expected_shape = tuple(tensor.shape)
        elif tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name}.{key} must share the path count")


def _validate_deterministic_topology_block(
    name: str, block: dict[str, torch.Tensor], sequence_width: int
) -> None:
    _validate_path_block(name, block)
    _validate_topology_extra_fields(
        name,
        block,
        int(sequence_width),
        {key: True for key, _dtype in _DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA},
    )


def _validate_topology_extra_fields(
    name: str,
    block: dict[str, torch.Tensor],
    sequence_width: int,
    expected_presence: dict[str, bool],
) -> None:
    path_count = int(block["valid"].shape[0])
    for key, dtype in _DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA:
        present = key in block
        if present != expected_presence[key]:
            state = "include" if expected_presence[key] else "omit"
            raise TypeError(f"{name}.{key} must {state} the concat schema")
        if not present:
            continue
        tensor = block[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}.{key} must be a torch.Tensor")
        if key in {"interaction_position", "interaction_normal"}:
            validate_cuda_tensor(
                f"{name}.{key}", tensor, dtype=dtype, ndim=2, trailing_shape=(3,)
            )
            expected = (path_count, 3)
        elif key in {"primitive_sequence", "material_sequence"}:
            validate_cuda_tensor(f"{name}.{key}", tensor, dtype=dtype, ndim=2)
            expected = (path_count, sequence_width)
        elif key in {"interaction_positions", "interaction_normals"}:
            validate_cuda_tensor(f"{name}.{key}", tensor, dtype=dtype, ndim=3)
            expected = (path_count, sequence_width, 3)
        else:
            validate_cuda_tensor(f"{name}.{key}", tensor, dtype=dtype, ndim=1)
            expected = (path_count,)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name}.{key} must have shape {expected}")


def deterministic_concat_topology_blocks(
    blocks: tuple[dict[str, torch.Tensor], ...] | list[dict[str, torch.Tensor]],
    *,
    sequence_width: int,
) -> dict[str, torch.Tensor]:
    if not blocks:
        raise ValueError("blocks must not be empty")
    if sequence_width < 0:
        raise ValueError("sequence_width must be non-negative")
    concat_presence = {
        key: key in blocks[0] for key, _dtype in _DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA
    }
    for index, block in enumerate(blocks):
        _validate_path_block(f"blocks[{index}]", block)
        _validate_topology_extra_fields(
            f"blocks[{index}]", block, int(sequence_width), concat_presence
        )
    exported = _required_native_op("deterministic_concat_topology_blocks")(
        tuple(blocks), int(sequence_width)
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_concat_topology_blocks must return a dict"
        )
    _validate_path_block("deterministic_concat_topology_blocks", exported)
    _validate_topology_extra_fields(
        "deterministic_concat_topology_blocks",
        exported,
        int(sequence_width),
        concat_presence,
    )
    expected_count = sum(int(block["valid"].shape[0]) for block in blocks)
    if exported["valid"].shape != (expected_count,):
        raise ValueError(
            "_channel.deterministic_concat_topology_blocks returned bad path count"
        )
    return exported


def deterministic_gather_topology_block(
    block: dict[str, torch.Tensor],
    order: torch.Tensor,
    *,
    max_count: int,
    sequence_width: int,
) -> dict[str, torch.Tensor]:
    if sequence_width < 0:
        raise ValueError("sequence_width must be non-negative")
    if max_count < -1:
        raise ValueError("max_count must be -1 or non-negative")
    _validate_path_block("block", block)
    field_presence = {
        key: key in block for key, _dtype in _DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA
    }
    _validate_topology_extra_fields("block", block, int(sequence_width), field_presence)
    validate_cuda_tensor("order", order, dtype=torch.long, ndim=1)
    if order.get_device() != block["valid"].get_device():
        raise ValueError("order must share block device")

    exported = _required_native_op("deterministic_gather_topology_block")(
        block,
        order,
        int(max_count),
        int(sequence_width),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_gather_topology_block must return a dict"
        )
    _validate_path_block("deterministic_gather_topology_block", exported)
    _validate_topology_extra_fields(
        "deterministic_gather_topology_block",
        exported,
        int(sequence_width),
        field_presence,
    )
    expected_count = (
        int(order.shape[0])
        if max_count < 0
        else min(int(order.shape[0]), int(max_count))
    )
    if exported["valid"].shape != (expected_count,):
        raise ValueError(
            "_channel.deterministic_gather_topology_block returned bad path count"
        )
    return exported


def path_los_visibility_inputs(
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_id", tx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("rx_id", rx_id, dtype=torch.int32, ndim=1)
    if rx_id.shape != tx_id.shape:
        raise ValueError("rx_id must match tx_id")
    exported = _required_native_op("path_los_visibility_inputs")(
        tx_positions, rx_positions, tx_id, rx_id
    )
    if not isinstance(exported, dict):
        raise TypeError("_channel.path_los_visibility_inputs must return a dict")
    validate_cuda_tensor(
        "start", exported["start"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "end", exported["end"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    if exported["active"].shape != tx_id.shape:
        raise ValueError(
            "_channel.path_los_visibility_inputs returned bad active shape"
        )
    return exported


def path_filter_los(
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    path_length_m: torch.Tensor,
    delay_s: torch.Tensor,
    path_gain: torch.Tensor,
    visible: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("tx_id", tx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("rx_id", rx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("path_length_m", path_length_m, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("delay_s", delay_s, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    for name, tensor in {
        "rx_id": rx_id,
        "path_length_m": path_length_m,
        "delay_s": delay_s,
        "path_gain": path_gain,
        "visible": visible,
    }.items():
        if tensor.shape != tx_id.shape:
            raise ValueError(f"{name} must match tx_id")
    block = _required_native_op("path_filter_los")(
        tx_id, rx_id, path_length_m, delay_s, path_gain, visible
    )
    _validate_path_block("path_filter_los", block)
    return block


def _validate_path_reflection_candidates(
    name: str, candidates: dict[str, torch.Tensor]
) -> None:
    _validate_path_block(name, candidates)
    path_count = candidates["valid"].shape
    for key in ("seg0_start", "seg0_end", "seg1_start", "seg1_end"):
        tensor = candidates.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}.{key} must be a torch.Tensor")
        validate_cuda_tensor(
            f"{name}.{key}", tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
        if tensor.shape[0] != path_count[0]:
            raise ValueError(f"{name}.{key} must share the path count")
    active = candidates.get("active")
    if not isinstance(active, torch.Tensor):
        raise TypeError(f"{name}.active must be a torch.Tensor")
    validate_cuda_tensor(f"{name}.active", active, dtype=torch.bool, ndim=1)
    if tuple(active.shape) != path_count:
        raise ValueError(f"{name}.active must share the path count")


def path_filter_block(
    block: dict[str, torch.Tensor],
    visible0: torch.Tensor,
    visible1: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _validate_path_block("block", block)
    validate_cuda_tensor("visible0", visible0, dtype=torch.bool, ndim=1)
    validate_cuda_tensor("visible1", visible1, dtype=torch.bool, ndim=1)
    if visible0.shape != block["valid"].shape or visible1.shape != block["valid"].shape:
        raise ValueError("visible masks must share the path count")
    out = _required_native_op("path_filter_block")(block, visible0, visible1)
    if not isinstance(out, dict):
        raise TypeError("_channel.path_filter_block must return a dict")
    _validate_path_block("path_filter_block", out)
    return out


def path_diffraction_block(
    rayd_output: tuple[torch.Tensor, ...],
    *,
    tx_index: int,
) -> dict[str, torch.Tensor]:
    if not isinstance(rayd_output, tuple) or len(rayd_output) != 18:
        raise TypeError(
            "rayd_output must be the 18-tensor RayD diffraction path tuple"
        )
    for index in (1, 3, 4, 5):
        validate_cuda_tensor(
            f"rayd_output[{index}]",
            rayd_output[index],
            dtype=torch.int32 if index != 1 else torch.bool,
            ndim=1,
        )
    for index in (8, 9, 10, 11, 12, 13, 14):
        validate_cuda_tensor(
            f"rayd_output[{index}]", rayd_output[index], dtype=torch.float32, ndim=1
        )
    capacity = rayd_output[1].shape
    for index in (3, 4, 5, 8, 9, 10, 11, 12, 13, 14):
        if rayd_output[index].shape != capacity:
            raise ValueError("RayD diffraction path tensors must share capacity")
    if tx_index < 0:
        raise ValueError("tx_index must be non-negative")
    out = _required_native_op("path_diffraction_block")(rayd_output, int(tx_index))
    if not isinstance(out, dict):
        raise TypeError("_channel.path_diffraction_block must return a dict")
    _validate_path_block("path_diffraction_block", out)
    return out


def path_merge_blocks(
    blocks: tuple[dict[str, torch.Tensor], ...] | list[dict[str, torch.Tensor]],
    *,
    tx_count: int,
    max_depth: int,
) -> dict[str, torch.Tensor]:
    if not blocks:
        raise ValueError("blocks must not be empty")
    for index, block in enumerate(blocks):
        _validate_path_block(f"blocks[{index}]", block)
    if tx_count < 0:
        raise ValueError("tx_count must be non-negative")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    out = _required_native_op("path_merge_blocks")(
        tuple(blocks), int(tx_count), int(max_depth)
    )
    if not isinstance(out, dict):
        raise TypeError("_channel.path_merge_blocks must return a dict")
    _validate_path_block("path_merge_blocks", out)
    return out


def path_finalize_blocks(
    los: dict[str, torch.Tensor],
    reflection: dict[str, torch.Tensor],
    diffraction: dict[str, torch.Tensor],
    *,
    max_paths: int | None,
    tx_count: int,
    max_depth: int,
) -> dict[str, torch.Tensor]:
    _validate_path_block("los", los)
    _validate_path_block("reflection", reflection)
    _validate_path_block("diffraction", diffraction)
    if tx_count < 0:
        raise ValueError("tx_count must be non-negative")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    max_paths_value = -1 if max_paths is None else int(max_paths)
    if max_paths_value < -1:
        raise ValueError("max_paths must be positive")
    finalized = _required_native_op("path_finalize_blocks")(
        los,
        reflection,
        diffraction,
        max_paths_value,
        int(tx_count),
        int(max_depth),
    )
    _validate_path_block("path_finalize_blocks", finalized)
    return finalized


# -------------------------------------------------------------------------
# candidates
# -------------------------------------------------------------------------
def path_reflection_candidates(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_normals: torch.Tensor,
    face_gain: torch.Tensor,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "vertices", vertices, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("faces", faces, dtype=torch.int32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor(
        "face_normals", face_normals, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("face_gain", face_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if face_normals.shape[0] != faces.shape[0] or face_gain.shape[0] != faces.shape[0]:
        raise ValueError("face_normals and face_gain must match faces")
    if tx_power.shape[0] != tx_positions.shape[0]:
        raise ValueError("tx_power must match tx_positions")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    candidates = _required_native_op("path_reflection_candidates")(
        vertices,
        faces,
        face_normals,
        face_gain,
        tx_positions,
        tx_power,
        rx_positions,
        float(frequency_hz),
    )
    if not isinstance(candidates, dict):
        raise TypeError("_channel.path_reflection_candidates must return a dict")
    _validate_path_reflection_candidates("path_reflection_candidates", candidates)
    return candidates


# -------------------------------------------------------------------------
# compaction
# -------------------------------------------------------------------------
def deterministic_reflection_order1_compact(
    *,
    visible: torch.Tensor,
    epc_faces: torch.Tensor,
    epc_hits: torch.Tensor,
    epc_normals: torch.Tensor,
    sequence_batch: torch.Tensor,
    rx_indices: torch.Tensor,
    tx: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    tx_index: int,
    face_eps_r: torch.Tensor,
    face_sigma_e: torch.Tensor,
    face_mu_r: torch.Tensor,
    face_gain: torch.Tensor,
    face_material_id: torch.Tensor,
    grouped_export: bool,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    if epc_faces.dtype not in {torch.int32, torch.int64}:
        raise TypeError("epc_faces must have dtype torch.int32 or torch.int64")
    if not epc_faces.is_cuda or epc_faces.ndim != 2 or not epc_faces.is_contiguous():
        raise ValueError(
            "epc_faces must be a contiguous CUDA tensor with shape (N, depth)"
        )
    validate_cuda_tensor("epc_hits", epc_hits, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("epc_normals", epc_normals, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("sequence_batch", sequence_batch, dtype=torch.int32, ndim=2)
    validate_cuda_tensor("rx_indices", rx_indices, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("tx", tx, dtype=torch.float32, ndim=1)
    if tx.shape != (3,):
        raise ValueError("tx must have shape (3,)")
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_eps_r", face_eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_sigma_e", face_sigma_e, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_mu_r", face_mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_gain", face_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "face_material_id", face_material_id, dtype=torch.int32, ndim=1
    )
    count = int(visible.shape[0])
    if epc_faces.shape[0] != count or epc_faces.shape[1] < 1:
        raise ValueError("epc_faces must match visible and include the first bounce")
    if epc_hits.shape[0] != count or epc_hits.shape[1] < 1 or epc_hits.shape[2] != 3:
        raise ValueError("epc_hits must have shape (N, depth, 3)")
    if epc_normals.shape != epc_hits.shape:
        raise ValueError("epc_normals must match epc_hits")
    if sequence_batch.shape[0] != count or sequence_batch.shape[1] < 1:
        raise ValueError(
            "sequence_batch must match visible and include the first bounce"
        )
    if rx_indices.shape != (count,):
        raise ValueError("rx_indices must match visible")
    if not 0 <= int(tx_index) < int(tx_power.shape[0]):
        raise ValueError("tx_index is out of range")
    if (
        face_sigma_e.shape != face_eps_r.shape
        or face_mu_r.shape != face_eps_r.shape
        or face_gain.shape != face_eps_r.shape
        or face_material_id.shape != face_eps_r.shape
    ):
        raise ValueError("face material tensors must share shape")
    exported = _required_native_op("deterministic_reflection_order1_compact")(
        visible,
        epc_faces,
        epc_hits,
        epc_normals,
        sequence_batch,
        rx_indices,
        tx,
        rx_positions,
        tx_power,
        int(tx_index),
        face_eps_r,
        face_sigma_e,
        face_mu_r,
        face_gain,
        face_material_id,
        bool(grouped_export),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_reflection_order1_compact must return a dict"
        )
    expected_fields = {
        "selected_faces",
        "selected_points",
        "selected_normals",
        "selected_rx_id",
        "tx_keep",
        "rx_keep",
        "tx_power",
        "eps_r",
        "sigma_e",
        "mu_r",
        "gain",
        "material_id",
    }
    if set(exported) != expected_fields:
        raise ValueError(
            "_channel.deterministic_reflection_order1_compact returned unexpected fields"
        )
    selected_count = int(exported["selected_faces"].shape[0])
    validate_cuda_tensor(
        "selected_faces", exported["selected_faces"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "selected_points",
        exported["selected_points"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "selected_normals",
        exported["selected_normals"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "selected_rx_id", exported["selected_rx_id"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "tx_keep", exported["tx_keep"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "rx_keep", exported["rx_keep"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", exported["tx_power"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("eps_r", exported["eps_r"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("sigma_e", exported["sigma_e"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("mu_r", exported["mu_r"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("gain", exported["gain"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "material_id", exported["material_id"], dtype=torch.int32, ndim=1
    )
    for key in expected_fields - {"selected_faces"}:
        if exported[key].shape[0] != selected_count:
            raise ValueError(
                f"_channel.deterministic_reflection_order1_compact returned bad {key} shape"
            )
    return exported


def deterministic_reflection_sequence_compact(
    *,
    visible: torch.Tensor,
    epc_sequences: torch.Tensor,
    epc_hits: torch.Tensor,
    epc_normals: torch.Tensor,
    rx_indices: torch.Tensor,
    tx: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    tx_index: int,
    face_eps_r: torch.Tensor,
    face_sigma_e: torch.Tensor,
    face_mu_r: torch.Tensor,
    face_gain: torch.Tensor,
    face_material_id: torch.Tensor,
    max_count: int = -1,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    if epc_sequences.dtype not in {torch.int32, torch.int64}:
        raise TypeError("epc_sequences must have dtype torch.int32 or torch.int64")
    if (
        not epc_sequences.is_cuda
        or epc_sequences.ndim != 2
        or not epc_sequences.is_contiguous()
    ):
        raise ValueError(
            "epc_sequences must be a contiguous CUDA tensor with shape (N, depth)"
        )
    validate_cuda_tensor("epc_hits", epc_hits, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("epc_normals", epc_normals, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("rx_indices", rx_indices, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("tx", tx, dtype=torch.float32, ndim=1)
    if tx.shape != (3,):
        raise ValueError("tx must have shape (3,)")
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_eps_r", face_eps_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_sigma_e", face_sigma_e, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_mu_r", face_mu_r, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("face_gain", face_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "face_material_id", face_material_id, dtype=torch.int32, ndim=1
    )
    if max_count < -1:
        raise ValueError("max_count must be -1 or non-negative")
    count = int(visible.shape[0])
    depth = int(epc_sequences.shape[1])
    if depth <= 0 or epc_sequences.shape[0] != count:
        raise ValueError("epc_sequences must match visible and have positive depth")
    if epc_hits.shape != (count, depth, 3):
        raise ValueError("epc_hits must have shape (N, depth, 3)")
    if epc_normals.shape != epc_hits.shape:
        raise ValueError("epc_normals must match epc_hits")
    if rx_indices.shape != (count,):
        raise ValueError("rx_indices must match visible")
    if not 0 <= int(tx_index) < int(tx_power.shape[0]):
        raise ValueError("tx_index is out of range")
    if (
        face_sigma_e.shape != face_eps_r.shape
        or face_mu_r.shape != face_eps_r.shape
        or face_gain.shape != face_eps_r.shape
        or face_material_id.shape != face_eps_r.shape
    ):
        raise ValueError("face material tensors must share shape")
    exported = _required_native_op("deterministic_reflection_sequence_compact")(
        visible,
        epc_sequences,
        epc_hits,
        epc_normals,
        rx_indices,
        tx,
        rx_positions,
        tx_power,
        int(tx_index),
        face_eps_r,
        face_sigma_e,
        face_mu_r,
        face_gain,
        face_material_id,
        int(max_count),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_reflection_sequence_compact must return a dict"
        )
    expected_fields = {
        "selected_sequences",
        "selected_hits",
        "selected_normals",
        "selected_rx_id",
        "selected_tx",
        "selected_rx",
        "tx_power",
        "eps_r",
        "sigma_e",
        "mu_r",
        "gain",
        "first_face",
        "material_id",
        "material_sequence",
        "first_hit",
        "first_normal",
    }
    if set(exported) != expected_fields:
        raise ValueError(
            "_channel.deterministic_reflection_sequence_compact returned unexpected fields"
        )
    selected_count = int(exported["selected_sequences"].shape[0])
    validate_cuda_tensor(
        "selected_sequences", exported["selected_sequences"], dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "selected_hits", exported["selected_hits"], dtype=torch.float32, ndim=3
    )
    validate_cuda_tensor(
        "selected_normals", exported["selected_normals"], dtype=torch.float32, ndim=3
    )
    validate_cuda_tensor(
        "selected_rx_id", exported["selected_rx_id"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "selected_tx",
        exported["selected_tx"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "selected_rx",
        exported["selected_rx"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor("tx_power", exported["tx_power"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("eps_r", exported["eps_r"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("sigma_e", exported["sigma_e"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("mu_r", exported["mu_r"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor("gain", exported["gain"], dtype=torch.float32, ndim=2)
    validate_cuda_tensor(
        "first_face", exported["first_face"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "material_id", exported["material_id"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "material_sequence", exported["material_sequence"], dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "first_hit",
        exported["first_hit"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "first_normal",
        exported["first_normal"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    expected_shapes = {
        "selected_sequences": (selected_count, depth),
        "selected_hits": (selected_count, depth, 3),
        "selected_normals": (selected_count, depth, 3),
        "selected_rx_id": (selected_count,),
        "selected_tx": (selected_count, 3),
        "selected_rx": (selected_count, 3),
        "tx_power": (selected_count,),
        "eps_r": (selected_count, depth),
        "sigma_e": (selected_count, depth),
        "mu_r": (selected_count, depth),
        "gain": (selected_count, depth),
        "first_face": (selected_count,),
        "material_id": (selected_count,),
        "material_sequence": (selected_count, depth),
        "first_hit": (selected_count, 3),
        "first_normal": (selected_count, 3),
    }
    for key, shape in expected_shapes.items():
        if tuple(exported[key].shape) != shape:
            raise ValueError(
                f"_channel.deterministic_reflection_sequence_compact returned bad {key} shape"
            )
    return exported


def deterministic_diffraction_order1_compact(
    *,
    valid: torch.Tensor,
    rx_id: torch.Tensor,
    depth: torch.Tensor,
    edge_id: torch.Tensor,
    delay_s: torch.Tensor,
    x_re: torch.Tensor,
    x_im: torch.Tensor,
    y_re: torch.Tensor,
    y_im: torch.Tensor,
    z_re: torch.Tensor,
    z_im: torch.Tensor,
    interaction_position: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("valid", valid, dtype=torch.bool, ndim=1)
    validate_cuda_tensor("rx_id", rx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("depth", depth, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("edge_id", edge_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("delay_s", delay_s, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("x_re", x_re, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("x_im", x_im, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("y_re", y_re, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("y_im", y_im, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("z_re", z_re, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("z_im", z_im, dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "interaction_position",
        interaction_position,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    count = int(valid.shape[0])
    for name, tensor in {
        "rx_id": rx_id,
        "depth": depth,
        "edge_id": edge_id,
        "delay_s": delay_s,
        "x_re": x_re,
        "x_im": x_im,
        "y_re": y_re,
        "y_im": y_im,
        "z_re": z_re,
        "z_im": z_im,
    }.items():
        if tensor.shape != (count,):
            raise ValueError(f"{name} must match valid")
    if interaction_position.shape != (count, 3):
        raise ValueError("interaction_position must match valid")

    exported = _required_native_op("deterministic_diffraction_order1_compact")(
        valid,
        rx_id,
        depth,
        edge_id,
        delay_s,
        x_re,
        x_im,
        y_re,
        y_im,
        z_re,
        z_im,
        interaction_position,
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_diffraction_order1_compact must return a dict"
        )
    expected_fields = {
        "rx_id",
        "depth",
        "edge_id",
        "delay_s",
        "x_re",
        "x_im",
        "y_re",
        "y_im",
        "z_re",
        "z_im",
        "interaction_position",
    }
    if set(exported) != expected_fields:
        raise ValueError(
            "_channel.deterministic_diffraction_order1_compact returned unexpected fields"
        )
    selected_count = int(exported["rx_id"].shape[0])
    validate_cuda_tensor("rx_id", exported["rx_id"], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("depth", exported["depth"], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("edge_id", exported["edge_id"], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("delay_s", exported["delay_s"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("x_re", exported["x_re"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("x_im", exported["x_im"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("y_re", exported["y_re"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("y_im", exported["y_im"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("z_re", exported["z_re"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("z_im", exported["z_im"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "interaction_position",
        exported["interaction_position"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    for key in expected_fields - {"interaction_position"}:
        if exported[key].shape != (selected_count,):
            raise ValueError(
                f"_channel.deterministic_diffraction_order1_compact returned bad {key} shape"
            )
    if exported["interaction_position"].shape != (selected_count, 3):
        raise ValueError(
            "_channel.deterministic_diffraction_order1_compact returned bad interaction_position shape"
        )
    return exported


def deterministic_sort_order(
    valid: torch.Tensor,
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    depth: torch.Tensor,
    component_id: torch.Tensor,
    primitive_id: torch.Tensor,
    edge_id: torch.Tensor,
    primitive_sequence: torch.Tensor,
) -> torch.Tensor:
    validate_cuda_tensor("valid", valid, dtype=torch.bool, ndim=1)
    for name, tensor in (
        ("tx_id", tx_id),
        ("rx_id", rx_id),
        ("depth", depth),
        ("component_id", component_id),
        ("primitive_id", primitive_id),
        ("edge_id", edge_id),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.int32, ndim=1)
        if tensor.shape != valid.shape:
            raise ValueError(f"{name} must match valid")
    validate_cuda_tensor(
        "primitive_sequence", primitive_sequence, dtype=torch.int32, ndim=2
    )
    if primitive_sequence.shape[0] != valid.shape[0]:
        raise ValueError("primitive_sequence must match valid rows")
    out = _required_native_op("deterministic_sort_order")(
        valid,
        tx_id,
        rx_id,
        depth,
        component_id,
        primitive_id,
        edge_id,
        primitive_sequence,
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel.deterministic_sort_order must return a tensor")
    validate_cuda_tensor("out", out, dtype=torch.long, ndim=1)
    if out.shape != valid.shape:
        raise ValueError("_channel.deterministic_sort_order returned bad shape")
    return out


# -------------------------------------------------------------------------
# construction
# -------------------------------------------------------------------------
def deterministic_los_topology_block(
    tx_id: torch.Tensor,
    rx_id: torch.Tensor,
    path_length_m: torch.Tensor,
    delay_s: torch.Tensor,
    path_gain: torch.Tensor,
    visible: torch.Tensor | None,
    *,
    frequency_hz: float,
    sequence_width: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("tx_id", tx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("rx_id", rx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("path_length_m", path_length_m, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("delay_s", delay_s, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=1)
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    if sequence_width < 0:
        raise ValueError("sequence_width must be non-negative")
    for name, tensor in {
        "rx_id": rx_id,
        "path_length_m": path_length_m,
        "delay_s": delay_s,
        "path_gain": path_gain,
    }.items():
        if tensor.shape != tx_id.shape:
            raise ValueError(f"{name} must match tx_id")
    if visible is None:
        block = _required_native_op("deterministic_los_topology_block_all_visible")(
            tx_id,
            rx_id,
            path_length_m,
            delay_s,
            path_gain,
            float(frequency_hz),
            int(sequence_width),
        )
    else:
        validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
        if visible.shape != tx_id.shape:
            raise ValueError("visible must match tx_id")
        block = _required_native_op("deterministic_los_topology_block")(
            tx_id,
            rx_id,
            path_length_m,
            delay_s,
            path_gain,
            visible,
            float(frequency_hz),
            int(sequence_width),
        )
    if not isinstance(block, dict):
        raise TypeError(
            "_channel.deterministic_los_topology_block must return a dict"
        )
    _validate_deterministic_topology_block(
        "deterministic_los_topology_block", block, int(sequence_width)
    )
    return block


def deterministic_topology_default_fields(
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=1)
    exported = _required_native_op("deterministic_topology_default_fields")(reference)
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_topology_default_fields must return a dict"
        )
    validate_cuda_tensor(
        "interaction_position",
        exported["interaction_position"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "interaction_normal",
        exported["interaction_normal"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "material_id", exported["material_id"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "path_field", exported["path_field"], dtype=torch.complex64, ndim=1
    )
    count = int(reference.shape[0])
    if (
        exported["interaction_position"].shape != (count, 3)
        or exported["interaction_normal"].shape != (count, 3)
        or exported["material_id"].shape != (count,)
        or exported["path_field"].shape != (count,)
    ):
        raise ValueError(
            "_channel.deterministic_topology_default_fields returned bad shape"
        )
    return exported


def deterministic_pad_topology_sequences(
    *,
    depth: torch.Tensor,
    primitive_id: torch.Tensor,
    material_id: torch.Tensor,
    interaction_position: torch.Tensor,
    interaction_normal: torch.Tensor,
    primitive_sequence: torch.Tensor,
    material_sequence: torch.Tensor,
    interaction_positions: torch.Tensor,
    interaction_normals: torch.Tensor,
    width: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("depth", depth, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("primitive_id", primitive_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("material_id", material_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "interaction_position",
        interaction_position,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "interaction_normal",
        interaction_normal,
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "primitive_sequence", primitive_sequence, dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "material_sequence", material_sequence, dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "interaction_positions", interaction_positions, dtype=torch.float32, ndim=3
    )
    validate_cuda_tensor(
        "interaction_normals", interaction_normals, dtype=torch.float32, ndim=3
    )
    if width < 0:
        raise ValueError("width must be non-negative")
    count = int(depth.shape[0])
    for name, tensor in {
        "primitive_id": primitive_id,
        "material_id": material_id,
    }.items():
        if tensor.shape != (count,):
            raise ValueError(f"{name} must match depth")
    for name, tensor in {
        "interaction_position": interaction_position,
        "interaction_normal": interaction_normal,
    }.items():
        if tensor.shape != (count, 3):
            raise ValueError(f"{name} must have shape (N, 3)")
    for name, tensor in {
        "primitive_sequence": primitive_sequence,
        "material_sequence": material_sequence,
    }.items():
        if tensor.shape[0] != count:
            raise ValueError(f"{name} must share the path count")
    for name, tensor in {
        "interaction_positions": interaction_positions,
        "interaction_normals": interaction_normals,
    }.items():
        if tensor.shape[0] != count or tensor.shape[2] != 3:
            raise ValueError(f"{name} must have shape (N, D, 3)")
    exported = _required_native_op("deterministic_pad_topology_sequences")(
        depth,
        primitive_id,
        material_id,
        interaction_position,
        interaction_normal,
        primitive_sequence,
        material_sequence,
        interaction_positions,
        interaction_normals,
        int(width),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_pad_topology_sequences must return a dict"
        )
    validate_cuda_tensor(
        "primitive_sequence", exported["primitive_sequence"], dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "material_sequence", exported["material_sequence"], dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "interaction_positions",
        exported["interaction_positions"],
        dtype=torch.float32,
        ndim=3,
    )
    validate_cuda_tensor(
        "interaction_normals",
        exported["interaction_normals"],
        dtype=torch.float32,
        ndim=3,
    )
    expected_i32 = (count, int(width))
    expected_vec = (count, int(width), 3)
    if (
        exported["primitive_sequence"].shape != expected_i32
        or exported["material_sequence"].shape != expected_i32
    ):
        raise ValueError(
            "_channel.deterministic_pad_topology_sequences returned bad sequence shape"
        )
    if (
        exported["interaction_positions"].shape != expected_vec
        or exported["interaction_normals"].shape != expected_vec
    ):
        raise ValueError(
            "_channel.deterministic_pad_topology_sequences returned bad interaction shape"
        )
    return exported


def deterministic_topology_base_fields(
    *,
    rx_id: torch.Tensor,
    path_length_m: torch.Tensor,
    delay_s: torch.Tensor,
    path_gain: torch.Tensor,
    tx_index: int,
    component_id: int,
    depth_source: torch.Tensor,
    depth_value: int,
    primitive_source: torch.Tensor,
    primitive_value: int,
    edge_source: torch.Tensor,
    edge_value: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("rx_id", rx_id, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("path_length_m", path_length_m, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("delay_s", delay_s, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("path_gain", path_gain, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("depth_source", depth_source, dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "primitive_source", primitive_source, dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor("edge_source", edge_source, dtype=torch.int32, ndim=1)
    count = int(rx_id.shape[0])
    for name, tensor in {
        "path_length_m": path_length_m,
        "delay_s": delay_s,
        "path_gain": path_gain,
    }.items():
        if tensor.shape != (count,):
            raise ValueError(f"{name} must match rx_id")
    for name, tensor in {
        "depth_source": depth_source,
        "primitive_source": primitive_source,
        "edge_source": edge_source,
    }.items():
        if tensor.shape not in {(0,), (count,)}:
            raise ValueError(f"{name} must be empty or match rx_id")
    block = _required_native_op("deterministic_topology_base_fields")(
        rx_id,
        path_length_m,
        delay_s,
        path_gain,
        int(tx_index),
        int(component_id),
        depth_source,
        int(depth_value),
        primitive_source,
        int(primitive_value),
        edge_source,
        int(edge_value),
    )
    if not isinstance(block, dict):
        raise TypeError(
            "_channel.deterministic_topology_base_fields must return a dict"
        )
    _validate_path_block("deterministic_topology_base_fields", block)
    return block


def deterministic_repeat_range(
    reference: torch.Tensor, *, start: int, end: int, repeats: int
) -> torch.Tensor:
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=1)
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    out = _required_native_op("deterministic_repeat_range")(
        reference, int(start), int(end), int(repeats)
    )
    validate_cuda_tensor("out", out, dtype=torch.int32, ndim=1)
    if out.shape != ((int(end) - int(start)) * int(repeats),):
        raise ValueError(
            "_channel.deterministic_repeat_range returned bad shape"
        )
    return out


def deterministic_face_anchor_points(
    vertices: torch.Tensor, faces: torch.Tensor
) -> torch.Tensor:
    validate_cuda_tensor(
        "vertices", vertices, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("faces", faces, dtype=torch.int32, ndim=2, trailing_shape=(3,))
    if vertices.get_device() != faces.get_device():
        raise ValueError("vertices and faces must share a CUDA device")
    out = _required_native_op("deterministic_face_anchor_points")(vertices, faces)
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.deterministic_face_anchor_points must return a tensor"
        )
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if out.shape != (faces.shape[0], 3):
        raise ValueError(
            "_channel.deterministic_face_anchor_points returned bad shape"
        )
    return out


def deterministic_reflection_epc_input_batch(
    *,
    tx: torch.Tensor,
    rx_positions: torch.Tensor,
    sequences: torch.Tensor,
    tri_a: torch.Tensor,
    normals: torch.Tensor,
    rx_start: int,
    rx_end: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("tx", tx, dtype=torch.float32, ndim=1)
    if tx.shape != (3,):
        raise ValueError("tx must have shape (3,)")
    validate_cuda_tensor(
        "rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "tri_a", tri_a, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "normals", normals, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if normals.shape != tri_a.shape:
        raise ValueError("normals must match tri_a")
    if sequences.dtype not in {torch.int32, torch.int64}:
        raise TypeError("sequences must have dtype torch.int32 or torch.int64")
    if not sequences.is_cuda:
        raise ValueError("sequences must be a CUDA tensor")
    if sequences.ndim != 2:
        raise ValueError("sequences must have shape (S, depth)")
    if not sequences.is_contiguous():
        raise ValueError("sequences must be contiguous")
    if rx_start < 0:
        raise ValueError("rx_start must be non-negative")
    if rx_end < rx_start:
        raise ValueError("rx_end must be greater than or equal to rx_start")
    if rx_end > int(rx_positions.shape[0]):
        raise ValueError("rx_end is out of range")
    exported = _required_native_op("deterministic_reflection_epc_input_batch")(
        tx,
        rx_positions,
        sequences,
        tri_a,
        normals,
        int(rx_start),
        int(rx_end),
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_reflection_epc_input_batch must return a dict"
        )
    expected_fields = {
        "tx_batch",
        "rx_batch",
        "rx_indices",
        "sequence_batch",
        "direct_plane_points",
        "direct_plane_normals",
    }
    if set(exported) != expected_fields:
        raise ValueError(
            "_channel.deterministic_reflection_epc_input_batch returned unexpected fields"
        )
    pair_count = (int(rx_end) - int(rx_start)) * int(sequences.shape[0])
    depth = int(sequences.shape[1])
    validate_cuda_tensor(
        "tx_batch",
        exported["tx_batch"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "rx_batch",
        exported["rx_batch"],
        dtype=torch.float32,
        ndim=2,
        trailing_shape=(3,),
    )
    validate_cuda_tensor(
        "rx_indices", exported["rx_indices"], dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "sequence_batch", exported["sequence_batch"], dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor(
        "direct_plane_points",
        exported["direct_plane_points"],
        dtype=torch.float32,
        ndim=3,
    )
    validate_cuda_tensor(
        "direct_plane_normals",
        exported["direct_plane_normals"],
        dtype=torch.float32,
        ndim=3,
    )
    if exported["tx_batch"].shape != (pair_count, 3):
        raise ValueError(
            "_channel.deterministic_reflection_epc_input_batch returned bad tx_batch shape"
        )
    if exported["rx_batch"].shape != (pair_count, 3):
        raise ValueError(
            "_channel.deterministic_reflection_epc_input_batch returned bad rx_batch shape"
        )
    if exported["rx_indices"].shape != (pair_count,):
        raise ValueError(
            "_channel.deterministic_reflection_epc_input_batch returned bad rx_indices shape"
        )
    if exported["sequence_batch"].shape != (pair_count, depth):
        raise ValueError(
            "_channel.deterministic_reflection_epc_input_batch returned bad sequence_batch shape"
        )
    if exported["direct_plane_points"].shape != (pair_count, depth, 3):
        raise ValueError(
            "_channel.deterministic_reflection_epc_input_batch returned bad direct_plane_points shape"
        )
    if exported["direct_plane_normals"].shape != (pair_count, depth, 3):
        raise ValueError(
            "_channel.deterministic_reflection_epc_input_batch returned bad direct_plane_normals shape"
        )
    return exported


def deterministic_face_sequence_chunk(
    reference: torch.Tensor,
    *,
    face_count: int,
    depth: int,
    start: int,
    end: int,
    adjacent_distinct: bool = False,
) -> torch.Tensor:
    if not isinstance(reference, torch.Tensor):
        raise TypeError("reference must be a torch.Tensor")
    if not reference.is_cuda:
        raise ValueError("reference must be a CUDA tensor")
    if face_count <= 0:
        raise ValueError("face_count must be positive")
    if depth <= 0:
        raise ValueError("depth must be positive")
    if start < 0:
        raise ValueError("start must be non-negative")
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    out = _required_native_op("deterministic_face_sequence_chunk")(
        reference,
        int(face_count),
        int(depth),
        int(start),
        int(end),
        bool(adjacent_distinct),
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.deterministic_face_sequence_chunk must return a tensor"
        )
    validate_cuda_tensor("out", out, dtype=torch.int32, ndim=2)
    if out.shape != (int(end) - int(start), int(depth)):
        raise ValueError(
            "_channel.deterministic_face_sequence_chunk returned bad shape"
        )
    return out


def deterministic_mapped_face_sequence_chunk(
    face_ids: torch.Tensor,
    *,
    depth: int,
    start: int,
    end: int,
    adjacent_distinct: bool = False,
) -> torch.Tensor:
    if not isinstance(face_ids, torch.Tensor):
        raise TypeError("face_ids must be a torch.Tensor")
    if face_ids.dtype not in {torch.int32, torch.int64}:
        raise TypeError("face_ids must have dtype torch.int32 or torch.int64")
    if not face_ids.is_cuda:
        raise ValueError("face_ids must be a CUDA tensor")
    if face_ids.ndim != 1:
        raise ValueError("face_ids must be one-dimensional")
    if not face_ids.is_contiguous():
        raise ValueError("face_ids must be contiguous")
    if depth <= 0:
        raise ValueError("depth must be positive")
    if start < 0:
        raise ValueError("start must be non-negative")
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    out = _required_native_op("deterministic_mapped_face_sequence_chunk")(
        face_ids,
        int(depth),
        int(start),
        int(end),
        bool(adjacent_distinct),
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            "_channel.deterministic_mapped_face_sequence_chunk must return a tensor"
        )
    validate_cuda_tensor("out", out, dtype=torch.int32, ndim=2)
    if out.shape != (int(end) - int(start), int(depth)):
        raise ValueError(
            "_channel.deterministic_mapped_face_sequence_chunk returned bad shape"
        )
    return out


# -------------------------------------------------------------------------
# primitives
# -------------------------------------------------------------------------
def deterministic_component_counts(component_id: torch.Tensor) -> dict[str, int]:
    validate_cuda_tensor("component_id", component_id, dtype=torch.int32, ndim=1)
    exported = _required_native_op("deterministic_component_counts")(component_id)
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel.deterministic_component_counts must return a dict"
        )
    counts: dict[str, int] = {}
    for name in ("los", "reflection", "diffraction"):
        value = exported[name]
        if not isinstance(value, int):
            raise TypeError(
                f"_channel.deterministic_component_counts returned non-int {name}"
            )
        counts[name] = value
    return counts


def deterministic_selected_edge_count(edge_id: torch.Tensor) -> int:
    validate_cuda_tensor("edge_id", edge_id, dtype=torch.int32, ndim=1)
    value = _required_native_op("deterministic_selected_edge_count")(edge_id)
    if not isinstance(value, int):
        raise TypeError(
            "_channel.deterministic_selected_edge_count must return an int"
        )
    return value


def core_pack_int2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("x", x, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("y", y, dtype=torch.int32, ndim=1)
    if y.shape != x.shape:
        raise ValueError("x and y must have the same shape")
    out = _required_native_op("core_pack_int2")(x, y)
    if not isinstance(out, torch.Tensor):
        raise TypeError("_channel.core_pack_int2 must return a tensor")
    validate_cuda_tensor("out", out, dtype=torch.int32, ndim=2, trailing_shape=(2,))
    if out.shape != (x.shape[0], 2):
        raise ValueError("_channel.core_pack_int2 returned an unexpected shape")
    return out


def deterministic_diffraction_state_pack(
    edge_indices: torch.Tensor,
    edge_pos: torch.Tensor,
    edge_dir: torch.Tensor,
    line_min: torch.Tensor,
    line_max: torch.Tensor,
    n0: torch.Tensor,
    n1: torch.Tensor,
    face0: torch.Tensor,
    face1: torch.Tensor,
    exterior_angle: torch.Tensor,
    tx: torch.Tensor,
    tx_power: torch.Tensor,
    tx_power_index: int,
) -> tuple[torch.Tensor, ...]:
    validate_cuda_tensor("edge_indices", edge_indices, dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "edge_pos", edge_pos, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "edge_dir", edge_dir, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("line_min", line_min, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("line_max", line_max, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("n0", n0, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("n1", n1, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("face0", face0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face1", face1, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("exterior_angle", exterior_angle, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("tx", tx, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    if not 0 <= int(tx_power_index) < int(tx_power.shape[0]):
        raise ValueError("tx_power_index is out of range")
    states = _required_native_op("deterministic_diffraction_state_pack")(
        edge_indices,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
        int(tx_power_index),
    )
    if not isinstance(states, tuple) or len(states) != 12:
        raise TypeError(
            "_channel.deterministic_diffraction_state_pack must return 12 tensors"
        )
    validate_cuda_tensor("state_edge_index", states[0], dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "state_edge_pos", states[1], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "state_edge_dir", states[2], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_line_min", states[3], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("state_line_max", states[4], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "state_n0", states[5], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "state_n1", states[6], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_face0", states[7], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_face1", states[8], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_exterior_angle", states[9], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "state_src", states[10], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_tx_power", states[11], dtype=torch.float32, ndim=1)
    return states


def deterministic_diffraction_state_pack_selected(
    selected: torch.Tensor,
    edge_pos: torch.Tensor,
    edge_dir: torch.Tensor,
    line_min: torch.Tensor,
    line_max: torch.Tensor,
    n0: torch.Tensor,
    n1: torch.Tensor,
    face0: torch.Tensor,
    face1: torch.Tensor,
    exterior_angle: torch.Tensor,
    tx: torch.Tensor,
    tx_power: torch.Tensor,
    tx_power_index: int,
) -> tuple[torch.Tensor, ...]:
    validate_cuda_tensor("selected", selected, dtype=torch.bool, ndim=1)
    validate_cuda_tensor(
        "edge_pos", edge_pos, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "edge_dir", edge_dir, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("line_min", line_min, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("line_max", line_max, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("n0", n0, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("n1", n1, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("face0", face0, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("face1", face1, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("exterior_angle", exterior_angle, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("tx", tx, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    if not 0 <= int(tx_power_index) < int(tx_power.shape[0]):
        raise ValueError("tx_power_index is out of range")
    states = _required_native_op("deterministic_diffraction_state_pack_selected")(
        selected,
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
        int(tx_power_index),
    )
    if not isinstance(states, tuple) or len(states) != 12:
        raise TypeError(
            "_channel.deterministic_diffraction_state_pack_selected must return 12 tensors"
        )
    validate_cuda_tensor("state_edge_index", states[0], dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "state_edge_pos", states[1], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "state_edge_dir", states[2], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_line_min", states[3], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("state_line_max", states[4], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "state_n0", states[5], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor(
        "state_n1", states[6], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_face0", states[7], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_face1", states[8], dtype=torch.int32, ndim=1)
    validate_cuda_tensor("state_exterior_angle", states[9], dtype=torch.float32, ndim=1)
    validate_cuda_tensor(
        "state_src", states[10], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("state_tx_power", states[11], dtype=torch.float32, ndim=1)
    return states


def mc_selected_edge_indices(selected: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("selected", selected, dtype=torch.bool, ndim=1)
    indices = _required_native_op("mc_selected_edge_indices")(selected)
    if not isinstance(indices, torch.Tensor):
        raise TypeError("_channel.mc_selected_edge_indices must return a tensor")
    validate_cuda_tensor("indices", indices, dtype=torch.int32, ndim=1)
    return indices


def path_concat_vec3(
    blocks: tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> torch.Tensor:
    if not blocks:
        raise ValueError("blocks must not be empty")
    for index, block in enumerate(blocks):
        validate_cuda_tensor(
            f"blocks[{index}]", block, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )
    out = _required_native_op("path_concat_vec3")(tuple(blocks))
    validate_cuda_tensor("out", out, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    return out


# -------------------------------------------------------------------------
# sampling
# -------------------------------------------------------------------------
def mc_sample_directions(count: int, reference: torch.Tensor) -> torch.Tensor:
    if count < 0:
        raise ValueError("count must be non-negative")
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)

    directions = _required_native_op("mc_sample_directions")(int(count), reference)
    if not isinstance(directions, torch.Tensor):
        raise TypeError("_channel.mc_sample_directions must return a tensor")
    validate_cuda_tensor(
        "directions", directions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    return directions


# -------------------------------------------------------------------------
# compact autograd
# -------------------------------------------------------------------------
COMPACT_CONTINUOUS_FIELDS = (
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


def _compact_autograd_companion(
    native_op: Callable[..., object],
    operation: str,
    valid: torch.Tensor,
    selected_row_index: torch.Tensor,
    continuous_values: tuple[torch.Tensor | None, ...],
    *,
    candidate_count: int,
    sequence_width: int,
) -> dict[str, torch.Tensor]:
    if len(continuous_values) != len(COMPACT_CONTINUOUS_FIELDS):
        raise ValueError("compact autograd companion requires all continuous fields")
    raw = native_op(
        valid,
        selected_row_index,
        *continuous_values,
        int(candidate_count),
        int(sequence_width),
    )
    if not isinstance(raw, dict) or set(raw) != set(COMPACT_CONTINUOUS_FIELDS):
        raise TypeError(f"native compact {operation} returned bad fields")
    return raw


def evaluated_paths_compact_finalize_backward(
    valid: torch.Tensor,
    selected_row_index: torch.Tensor,
    *gradients: torch.Tensor | None,
    candidate_count: int,
    sequence_width: int,
) -> dict[str, torch.Tensor]:
    """Scatter compact continuous cotangents to candidate rows."""

    return _compact_autograd_companion(
        _required_native_op("evaluated_paths_compact_finalize_backward"),
        "backward",
        valid,
        selected_row_index,
        gradients,
        candidate_count=candidate_count,
        sequence_width=sequence_width,
    )


def evaluated_paths_compact_finalize_jvp(
    valid: torch.Tensor,
    selected_row_index: torch.Tensor,
    *tangents: torch.Tensor | None,
    candidate_count: int,
    sequence_width: int,
) -> dict[str, torch.Tensor]:
    """Gather candidate continuous tangents into exact compact rows."""

    return _compact_autograd_companion(
        _required_native_op("evaluated_paths_compact_finalize_jvp"),
        "JVP",
        valid,
        selected_row_index,
        tangents,
        candidate_count=candidate_count,
        sequence_width=sequence_width,
    )


# -------------------------------------------------------------------------
# canonical compact
# -------------------------------------------------------------------------
_CANONICAL_BLOCK_FIELDS = tuple(
    name
    for name, _dtype in (
        *_PATH_BLOCK_SCHEMA,
        *_DETERMINISTIC_TOPOLOGY_EXTRA_SCHEMA,
    )
)
_EXTRA_CONTINUOUS_FIELDS = ("field_xyz", "coefficient")
_FORWARD_BLOCK_FIELDS = (*_CANONICAL_BLOCK_FIELDS, *_EXTRA_CONTINUOUS_FIELDS)
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
        with disable_functorch():
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


# -------------------------------------------------------------------------
# transmission
# -------------------------------------------------------------------------
_TRANSMISSION_BLOCK_FIELDS = (
    "valid",
    "tx_id",
    "rx_id",
    "depth",
    "component_id",
    "primitive_id",
    "edge_id",
    "path_length_m",
    "delay_s",
    "path_gain",
    "path_field",
    "interaction_position",
    "interaction_normal",
    "material_id",
    "primitive_sequence",
    "material_sequence",
    "interaction_positions",
    "interaction_normals",
)
_COUNT_FIELDS = ("device_candidate_count", "device_guardrail_count")
_OUTPUT_FIELDS = (*_TRANSMISSION_BLOCK_FIELDS, *_COUNT_FIELDS)
_CONTINUOUS_OUTPUT_FIELDS = (
    "path_length_m",
    "delay_s",
    "interaction_position",
    "interaction_normal",
    "interaction_positions",
    "interaction_normals",
)


class _EnumeratedTransmissionTopologyPackFunction(torch.autograd.Function):
    """Native fixed-valid topology copy with native VJP/JVP companions."""

    @staticmethod
    def forward(*inputs):
        raw = _required_native_op("enumerated_transmission_topology_pack")(*inputs)
        if not isinstance(raw, dict) or set(raw) != set(_OUTPUT_FIELDS):
            raise TypeError("native transmission topology pack returned bad fields")
        return tuple(raw[name] for name in _OUTPUT_FIELDS)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.set_materialize_grads(False)
        saved = tuple(
            torch.autograd.forward_ad.unpack_dual(value).primal
            for value in (output[0], inputs[1])
        )
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        continuous = set(_CONTINUOUS_OUTPUT_FIELDS)
        ctx.mark_non_differentiable(
            *(
                value
                for name, value in zip(_OUTPUT_FIELDS, output, strict=True)
                if name not in continuous
            )
        )

    @staticmethod
    @_ad_first_order_only
    def backward(ctx, *grad_outputs):
        none_grads = (None,) * 13
        continuous_grads = tuple(
            grad_outputs[_OUTPUT_FIELDS.index(name)]
            for name in _CONTINUOUS_OUTPUT_FIELDS
        )
        if all(value is None for value in continuous_grads):
            return none_grads
        if not any(ctx.needs_input_grad[index] for index in (5, 6, 7)):
            return none_grads
        topology_valid, hit_valid = ctx.saved_tensors
        # Reductions commonly supply expanded stride-zero cotangents.  The
        # typed CUDA companion consumes contiguous row-major buffers, so
        # normalize only live cotangent slots at the dispatch boundary.
        continuous_grads = tuple(
            None if value is None else value.contiguous() for value in continuous_grads
        )
        raw = _required_native_op("enumerated_transmission_topology_pack_backward")(
            topology_valid, hit_valid, *continuous_grads
        )
        expected = {"grad_distance", "grad_position", "grad_normal"}
        if not isinstance(raw, dict) or set(raw) != expected:
            raise TypeError("native transmission topology backward returned bad fields")
        return (
            None,
            None,
            None,
            None,
            None,
            raw["grad_distance"] if ctx.needs_input_grad[5] else None,
            raw["grad_position"] if ctx.needs_input_grad[6] else None,
            raw["grad_normal"] if ctx.needs_input_grad[7] else None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, *tangents):
        native_tangents = tuple(
            _ad_native_tangent_or_none(tangents[index]) for index in (5, 6, 7)
        )
        if all(value is None for value in native_tangents):
            return (None,) * len(_OUTPUT_FIELDS)
        topology_valid, hit_valid = (
            _ad_native_tensor(value) for value in ctx.saved_tensors
        )
        with disable_functorch():
            raw = _required_native_op("enumerated_transmission_topology_pack_jvp")(
                topology_valid, hit_valid, *native_tangents
            )
        if not isinstance(raw, dict) or set(raw) != set(_CONTINUOUS_OUTPUT_FIELDS):
            raise TypeError("native transmission topology JVP returned bad fields")
        return tuple(raw.get(name) for name in _OUTPUT_FIELDS)


@dataclass(frozen=True, slots=True, eq=False)
class TransmissionTopologyCapacity:
    """Pair-major component-5 rows with CUDA-resident actual counts."""

    candidate_capacity: int
    sequence_width: int
    failure_state: CapacityFailureState
    execution: CapacityExecutionCounts
    valid: torch.Tensor
    tx_id: torch.Tensor
    rx_id: torch.Tensor
    depth: torch.Tensor
    component_id: torch.Tensor
    primitive_id: torch.Tensor
    edge_id: torch.Tensor
    path_length_m: torch.Tensor
    delay_s: torch.Tensor
    path_gain: torch.Tensor
    path_field: torch.Tensor
    interaction_position: torch.Tensor
    interaction_normal: torch.Tensor
    material_id: torch.Tensor
    primitive_sequence: torch.Tensor
    material_sequence: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor

    def __post_init__(self) -> None:
        capacity = require_host_count("candidate_capacity", self.candidate_capacity)
        width = require_host_count("sequence_width", self.sequence_width)
        valid = require_tensor(
            "valid",
            self.valid,
            dtype=torch.bool,
            shape=(capacity,),
            cuda=True,
            contiguous=True,
        )
        require_capacity_failure_state(self.failure_state, device=valid.device)
        if not isinstance(self.execution, CapacityExecutionCounts):
            raise TypeError("execution must be CapacityExecutionCounts")
        if self.execution.candidate_capacity != capacity:
            raise ValueError("execution capacity must match candidate_capacity")
        if self.execution.failure_state is not self.failure_state:
            raise ValueError("execution must retain the exact failure_state")
        for name in (
            "tx_id",
            "rx_id",
            "depth",
            "component_id",
            "primitive_id",
            "edge_id",
            "material_id",
        ):
            require_tensor(
                name,
                getattr(self, name),
                dtype=torch.int32,
                shape=(capacity,),
                device=valid.device,
                cuda=True,
                contiguous=True,
            )
        for name in ("path_length_m", "delay_s", "path_gain"):
            require_tensor(
                name,
                getattr(self, name),
                dtype=torch.float32,
                shape=(capacity,),
                device=valid.device,
                cuda=True,
                contiguous=True,
            )
        require_tensor(
            "path_field",
            self.path_field,
            dtype=torch.complex64,
            shape=(capacity,),
            device=valid.device,
            cuda=True,
            contiguous=True,
        )
        for name in ("interaction_position", "interaction_normal"):
            require_tensor(
                name,
                getattr(self, name),
                dtype=torch.float32,
                shape=(capacity, 3),
                device=valid.device,
                cuda=True,
                contiguous=True,
            )
        for name in ("primitive_sequence", "material_sequence"):
            require_tensor(
                name,
                getattr(self, name),
                dtype=torch.int32,
                shape=(capacity, width),
                device=valid.device,
                cuda=True,
                contiguous=True,
            )
        for name in ("interaction_positions", "interaction_normals"):
            require_tensor(
                name,
                getattr(self, name),
                dtype=torch.float32,
                shape=(capacity, width, 3),
                device=valid.device,
                cuda=True,
                contiguous=True,
            )

    @property
    def device(self) -> torch.device:
        return self.valid.device

    def as_block(self) -> dict[str, torch.Tensor]:
        """Return the topology block without copying or reordering tensors."""

        return {
            "valid": self.valid,
            "tx_id": self.tx_id,
            "rx_id": self.rx_id,
            "depth": self.depth,
            "component_id": self.component_id,
            "primitive_id": self.primitive_id,
            "edge_id": self.edge_id,
            "path_length_m": self.path_length_m,
            "delay_s": self.delay_s,
            "path_gain": self.path_gain,
            "path_field": self.path_field,
            "interaction_position": self.interaction_position,
            "interaction_normal": self.interaction_normal,
            "material_id": self.material_id,
            "primitive_sequence": self.primitive_sequence,
            "material_sequence": self.material_sequence,
            "interaction_positions": self.interaction_positions,
            "interaction_normals": self.interaction_normals,
        }


def enumerated_transmission_topology_pack(
    penetration: SegmentPenetrationResult,
    face_material_id: torch.Tensor,
    geometry_mode_id: torch.Tensor,
    *,
    tx_count: int,
    rx_count: int,
) -> TransmissionTopologyCapacity:
    """Pack one inert row per endpoint pair without reading device counts."""

    if not isinstance(penetration, SegmentPenetrationResult):
        raise TypeError("penetration must be a SegmentPenetrationResult")
    validate_cuda_tensor(
        "face_material_id", face_material_id, dtype=torch.int32, ndim=1
    )
    validate_cuda_tensor(
        "geometry_mode_id", geometry_mode_id, dtype=torch.int32, ndim=1
    )
    if face_material_id.device != penetration.device:
        raise ValueError("face_material_id must share the penetration device")
    if geometry_mode_id.device != penetration.device:
        raise ValueError("geometry_mode_id must share the penetration device")
    tx_count = require_host_count("tx_count", tx_count)
    rx_count = require_host_count("rx_count", rx_count)
    candidate_capacity = tx_count * rx_count
    if penetration.segment_count != candidate_capacity:
        raise ValueError("penetration rows must equal tx_count * rx_count")

    values = _EnumeratedTransmissionTopologyPackFunction.apply(
        penetration.failure_state.bits,
        penetration.valid,
        penetration.num_hits,
        penetration.reached_target,
        penetration.overflow,
        penetration.distance,
        penetration.position,
        penetration.normal,
        penetration.global_primitive_id,
        face_material_id,
        geometry_mode_id,
        tx_count,
        rx_count,
    )
    raw = dict(zip(_OUTPUT_FIELDS, values, strict=True))
    block = {name: raw[name] for name in _TRANSMISSION_BLOCK_FIELDS}
    execution = CapacityExecutionCounts(
        candidate_capacity=candidate_capacity,
        failure_state=penetration.failure_state,
        device_candidate_count=raw["device_candidate_count"],
        device_guardrail_count=raw["device_guardrail_count"],
    )
    return TransmissionTopologyCapacity(
        candidate_capacity=candidate_capacity,
        sequence_width=penetration.hit_capacity,
        failure_state=penetration.failure_state,
        execution=execution,
        **block,
    )