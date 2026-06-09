from __future__ import annotations

from dataclasses import dataclass

import torch

from ._validation import require_tensor


@dataclass(frozen=True, slots=True)
class MaterialStore:
    eps_r: torch.Tensor
    mu_r: torch.Tensor
    sigma_e: torch.Tensor
    gain: torch.Tensor
    model_id: torch.Tensor
    model_params: torch.Tensor
    frequency_hz: float
    version: int

    def __post_init__(self) -> None:
        require_tensor("eps_r", self.eps_r, dtype=torch.float32, ndim=1)
        require_tensor("mu_r", self.mu_r, dtype=torch.float32, ndim=1)
        require_tensor("sigma_e", self.sigma_e, dtype=torch.float32, ndim=1)
        require_tensor("gain", self.gain, dtype=torch.float32, ndim=1)
        require_tensor("model_id", self.model_id, dtype=torch.int32, ndim=1)
        require_tensor("model_params", self.model_params, dtype=torch.float32, ndim=2)
        lengths = {
            self.eps_r.shape[0],
            self.mu_r.shape[0],
            self.sigma_e.shape[0],
            self.gain.shape[0],
            self.model_id.shape[0],
            self.model_params.shape[0],
        }
        if len(lengths) != 1:
            raise ValueError("material tensors must have the same length")
        if self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive")
