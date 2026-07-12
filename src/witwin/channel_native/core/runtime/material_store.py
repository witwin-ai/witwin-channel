from __future__ import annotations

from dataclasses import dataclass

import torch

from ._validation import require_tensor


@dataclass(frozen=True, slots=True)
class MaterialStore:
    material_id: torch.Tensor
    eps_r: torch.Tensor
    mu_r: torch.Tensor
    sigma_e: torch.Tensor
    gain: torch.Tensor
    model_id: torch.Tensor
    thickness_m: torch.Tensor
    scattering_coefficient: torch.Tensor
    xpd_coefficient: torch.Tensor
    material_keys: tuple[str, ...]
    frequency_hz: float
    abi_version: int
    cache_token: str
    version: int

    def __post_init__(self) -> None:
        require_tensor("material_id", self.material_id, dtype=torch.int32, ndim=1)
        require_tensor("eps_r", self.eps_r, dtype=torch.float32, ndim=1)
        require_tensor("mu_r", self.mu_r, dtype=torch.float32, ndim=1)
        require_tensor("sigma_e", self.sigma_e, dtype=torch.float32, ndim=1)
        require_tensor("gain", self.gain, dtype=torch.float32, ndim=1)
        require_tensor("model_id", self.model_id, dtype=torch.int32, ndim=1)
        require_tensor("thickness_m", self.thickness_m, dtype=torch.float32, ndim=1)
        require_tensor(
            "scattering_coefficient",
            self.scattering_coefficient,
            dtype=torch.float32,
            ndim=1,
        )
        require_tensor(
            "xpd_coefficient", self.xpd_coefficient, dtype=torch.float32, ndim=1
        )
        lengths = {
            self.material_id.shape[0],
            self.eps_r.shape[0],
            self.mu_r.shape[0],
            self.sigma_e.shape[0],
            self.gain.shape[0],
            self.model_id.shape[0],
            self.thickness_m.shape[0],
            self.scattering_coefficient.shape[0],
            self.xpd_coefficient.shape[0],
            len(self.material_keys),
        }
        if len(lengths) != 1:
            raise ValueError("material tensors must have the same length")
        if self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive")
        if self.abi_version != 2:
            raise ValueError("MaterialStore requires material ABI version 2")
        if self.material_id.numel() and not torch.equal(
            self.material_id, torch.arange(self.material_id.numel(), dtype=torch.int32)
        ):
            raise ValueError(
                "material_id must be dense and stable in [0, material_count)"
            )
        if len(set(self.material_keys)) != len(self.material_keys):
            raise ValueError("material_keys must be unique")
