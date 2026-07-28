from __future__ import annotations

import torch

from witwin.channel.runtime import (
    required_symbol as _required_native_op,
    validate_cuda_tensor,
)
from .blocks import _validate_path_reflection_candidates


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


__all__ = ["path_reflection_candidates"]
