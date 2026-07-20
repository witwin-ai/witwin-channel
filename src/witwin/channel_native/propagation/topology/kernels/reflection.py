"""Native fixed-capacity reflection candidates after RayD EPC visibility."""

from __future__ import annotations

import torch

from witwin.channel_native.propagation.models.reflection import (
    ReflectionCandidateCapacity,
)
from witwin.channel_native.runtime.symbols import required_symbol as _required_native_op
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor


_OUTPUT_FIELDS = {
    "valid",
    "candidate_count",
    "overflow",
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


def _require_host_count(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_epc_sequences(epc_sequences: object) -> torch.Tensor:
    if not isinstance(epc_sequences, torch.Tensor):
        raise TypeError("epc_sequences must be a torch.Tensor")
    if epc_sequences.dtype not in {torch.int32, torch.int64}:
        raise TypeError("epc_sequences must have dtype torch.int32 or torch.int64")
    if not epc_sequences.is_cuda:
        raise ValueError("epc_sequences must be a CUDA tensor")
    if epc_sequences.ndim != 2 or not epc_sequences.is_contiguous():
        raise ValueError("epc_sequences must be contiguous with shape (N, depth)")
    return epc_sequences


def _validate_reflection_shapes(
    *,
    visible: torch.Tensor,
    epc_sequences: torch.Tensor,
    epc_hits: torch.Tensor,
    epc_normals: torch.Tensor,
    sequence_batch: torch.Tensor,
    rx_indices: torch.Tensor,
    tx: torch.Tensor,
    face_tensors: tuple[torch.Tensor, ...],
) -> tuple[int, int]:
    if tx.shape != (3,):
        raise ValueError("tx must have shape (3,)")
    input_count = int(visible.shape[0])
    depth = int(epc_sequences.shape[1])
    if depth <= 0 or epc_sequences.shape[0] != input_count:
        raise ValueError("epc_sequences must match visible and have positive depth")
    if epc_hits.shape != (input_count, depth, 3):
        raise ValueError("epc_hits must have shape (N, depth, 3)")
    if epc_normals.shape != epc_hits.shape:
        raise ValueError("epc_normals must match epc_hits")
    if sequence_batch.shape != epc_sequences.shape:
        raise ValueError("sequence_batch must match epc_sequences")
    if rx_indices.shape != visible.shape:
        raise ValueError("rx_indices must match visible")
    if any(tensor.shape != face_tensors[0].shape for tensor in face_tensors[1:]):
        raise ValueError("face material tensors must share shape")
    return input_count, depth


def _validate_shared_device(
    visible: torch.Tensor, named_tensors: tuple[tuple[str, torch.Tensor], ...]
) -> None:
    for name, tensor in named_tensors:
        if tensor.device != visible.device:
            raise ValueError(f"{name} must share visible device")


def deterministic_reflection_candidate_capacity_block(
    *,
    visible: torch.Tensor,
    epc_sequences: torch.Tensor,
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
    candidate_capacity: int,
) -> ReflectionCandidateCapacity:
    """Stably gather visible reflection rows into explicit host capacity.

    The live switch supplies the host-known theoretical EPC batch row count as
    ``candidate_capacity``. A smaller explicit capacity remains supported only
    as a fail-loud contract; it never truncates visible rows.
    """

    validate_cuda_tensor("visible", visible, dtype=torch.bool, ndim=1)
    epc_sequences = _validate_epc_sequences(epc_sequences)
    validate_cuda_tensor("epc_hits", epc_hits, dtype=torch.float32, ndim=3)
    validate_cuda_tensor("epc_normals", epc_normals, dtype=torch.float32, ndim=3)
    validate_cuda_tensor(
        "sequence_batch", sequence_batch, dtype=torch.int32, ndim=2
    )
    validate_cuda_tensor("rx_indices", rx_indices, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("tx", tx, dtype=torch.float32, ndim=1)
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
    _, depth = _validate_reflection_shapes(
        visible=visible,
        epc_sequences=epc_sequences,
        epc_hits=epc_hits,
        epc_normals=epc_normals,
        sequence_batch=sequence_batch,
        rx_indices=rx_indices,
        tx=tx,
        face_tensors=(
            face_eps_r,
            face_sigma_e,
            face_mu_r,
            face_gain,
            face_material_id,
        ),
    )
    tx_index = _require_host_count("tx_index", tx_index)
    if tx_index >= int(tx_power.shape[0]):
        raise ValueError("tx_index is out of range")
    if type(grouped_export) is not bool:
        raise TypeError("grouped_export must be a bool")
    candidate_capacity = _require_host_count(
        "candidate_capacity", candidate_capacity
    )
    _validate_shared_device(visible, (
        ("epc_sequences", epc_sequences),
        ("epc_hits", epc_hits),
        ("epc_normals", epc_normals),
        ("sequence_batch", sequence_batch),
        ("rx_indices", rx_indices),
        ("tx", tx),
        ("rx_positions", rx_positions),
        ("tx_power", tx_power),
        ("face_eps_r", face_eps_r),
        ("face_sigma_e", face_sigma_e),
        ("face_mu_r", face_mu_r),
        ("face_gain", face_gain),
        ("face_material_id", face_material_id),
    ))

    raw = _required_native_op(
        "deterministic_reflection_candidate_capacity_block"
    )(
        visible,
        epc_sequences,
        epc_hits,
        epc_normals,
        sequence_batch,
        rx_indices,
        tx,
        rx_positions,
        tx_power,
        tx_index,
        face_eps_r,
        face_sigma_e,
        face_mu_r,
        face_gain,
        face_material_id,
        grouped_export,
        candidate_capacity,
    )
    if not isinstance(raw, dict) or set(raw) != _OUTPUT_FIELDS:
        raise TypeError("native reflection candidate capacity returned bad fields")
    return ReflectionCandidateCapacity(
        candidate_capacity=candidate_capacity,
        depth=depth,
        **raw,
    )


__all__ = ["deterministic_reflection_candidate_capacity_block"]
