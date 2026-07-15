from __future__ import annotations

import torch

from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor


def _validate_layer_csr(
    layer_offset: torch.Tensor,
    layer_count: torch.Tensor,
    layer_thickness_m: torch.Tensor,
    layer_eps_r: torch.Tensor,
    layer_sigma_e: torch.Tensor,
    layer_mu_r: torch.Tensor,
    device: int,
) -> None:
    validate_cuda_tensor("layer_offset", layer_offset, dtype=torch.int32, ndim=1)
    validate_cuda_tensor("layer_count", layer_count, dtype=torch.int32, ndim=1)
    if layer_count.shape != layer_offset.shape:
        raise ValueError("layer_count must match layer_offset length")
    for name, tensor in (
        ("layer_thickness_m", layer_thickness_m),
        ("layer_eps_r", layer_eps_r),
        ("layer_sigma_e", layer_sigma_e),
        ("layer_mu_r", layer_mu_r),
    ):
        validate_cuda_tensor(name, tensor, dtype=torch.float32, ndim=1)
        if tensor.shape != layer_thickness_m.shape:
            raise ValueError(f"{name} must match layer_thickness_m length")
    for name, tensor in (
        ("layer_offset", layer_offset),
        ("layer_count", layer_count),
        ("layer_thickness_m", layer_thickness_m),
        ("layer_eps_r", layer_eps_r),
        ("layer_sigma_e", layer_sigma_e),
        ("layer_mu_r", layer_mu_r),
    ):
        if tensor.get_device() != device:
            raise ValueError(f"{name} must share the op device")


__all__ = ["_validate_layer_csr"]
