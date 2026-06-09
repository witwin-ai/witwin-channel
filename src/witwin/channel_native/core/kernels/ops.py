from __future__ import annotations

import torch

from .extension import native_extension
from .metadata import make_metadata, validate_metadata


def validate_cuda_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    ndim: int,
    trailing_shape: tuple[int, ...] = (),
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if trailing_shape and tuple(tensor.shape[-len(trailing_shape) :]) != trailing_shape:
        raise ValueError(f"{name} must end with shape {trailing_shape}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def noop_metadata(*, accumulation_strategy: str = "none") -> dict[str, bool | float | int | str]:
    return make_metadata(
        primitive="noop_metadata",
        accumulation_strategy=accumulation_strategy,
        scheduling_strategy="none",
        ad_status="none",
    )


def path_los_export(
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    frequency_hz: float,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor("tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("tx_power", tx_power, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("rx_positions", rx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if tx_power.shape[0] != tx_positions.shape[0]:
        raise ValueError("tx_power must have one value per transmitter")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    native = native_extension()
    if native is not None and hasattr(native, "path_los_export"):
        exported = native.path_los_export(tx_positions, tx_power, rx_positions, float(frequency_hz))
        if not isinstance(exported, dict):
            raise TypeError("_channel_native.path_los_export must return a dict")
        return exported

    light_speed_m_per_s = 299_792_458.0
    rx_count = rx_positions.shape[0]
    tx_count = tx_positions.shape[0]
    rx_id = torch.arange(rx_count, device=rx_positions.device, dtype=torch.int32).repeat_interleave(tx_count)
    tx_id = torch.arange(tx_count, device=tx_positions.device, dtype=torch.int32).repeat(rx_count)
    tx_for_path = tx_positions[tx_id.to(dtype=torch.long)]
    rx_for_path = rx_positions[rx_id.to(dtype=torch.long)]
    path_length = torch.linalg.vector_norm(tx_for_path - rx_for_path, dim=-1).clamp_min(1.0e-6)
    delay = path_length / light_speed_m_per_s
    wavelength = light_speed_m_per_s / frequency_hz
    free_space_gain = (wavelength / (4.0 * torch.pi * path_length)).square()
    path_gain = tx_power[tx_id.to(dtype=torch.long)] * free_space_gain
    return {
        "tx_id": tx_id.contiguous(),
        "rx_id": rx_id.contiguous(),
        "path_length_m": path_length.to(dtype=torch.float32).contiguous(),
        "delay_s": delay.to(dtype=torch.float32).contiguous(),
        "path_gain": path_gain.to(dtype=torch.float32).contiguous(),
    }
