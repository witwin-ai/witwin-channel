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
    # ABI v3: flat CSR layer stack over all M materials (L total layers).
    layer_offset: torch.Tensor
    layer_count: torch.Tensor
    layer_thickness_m: torch.Tensor
    layer_eps_r: torch.Tensor
    layer_sigma_e: torch.Tensor
    layer_mu_r: torch.Tensor
    # ABI v3: front-surface roughness statistics (sigma_h == 0 means smooth).
    rough_sigma_h_m: torch.Tensor
    rough_corr_x_m: torch.Tensor
    rough_corr_y_m: torch.Tensor
    rough_axis_rad: torch.Tensor
    # ABI v3: 0=thin_sheet, 1=closed_volume / 0=smooth, 1=kirchhoff_ensemble.
    geometry_mode_id: torch.Tensor
    scatter_model_id: torch.Tensor
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
        require_tensor("layer_offset", self.layer_offset, dtype=torch.int32, ndim=1)
        require_tensor("layer_count", self.layer_count, dtype=torch.int32, ndim=1)
        require_tensor(
            "layer_thickness_m", self.layer_thickness_m, dtype=torch.float32, ndim=1
        )
        require_tensor("layer_eps_r", self.layer_eps_r, dtype=torch.float32, ndim=1)
        require_tensor("layer_sigma_e", self.layer_sigma_e, dtype=torch.float32, ndim=1)
        require_tensor("layer_mu_r", self.layer_mu_r, dtype=torch.float32, ndim=1)
        require_tensor(
            "rough_sigma_h_m", self.rough_sigma_h_m, dtype=torch.float32, ndim=1
        )
        require_tensor(
            "rough_corr_x_m", self.rough_corr_x_m, dtype=torch.float32, ndim=1
        )
        require_tensor(
            "rough_corr_y_m", self.rough_corr_y_m, dtype=torch.float32, ndim=1
        )
        require_tensor(
            "rough_axis_rad", self.rough_axis_rad, dtype=torch.float32, ndim=1
        )
        require_tensor(
            "geometry_mode_id", self.geometry_mode_id, dtype=torch.int32, ndim=1
        )
        require_tensor(
            "scatter_model_id", self.scatter_model_id, dtype=torch.int32, ndim=1
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
            self.layer_offset.shape[0],
            self.layer_count.shape[0],
            self.rough_sigma_h_m.shape[0],
            self.rough_corr_x_m.shape[0],
            self.rough_corr_y_m.shape[0],
            self.rough_axis_rad.shape[0],
            self.geometry_mode_id.shape[0],
            self.scatter_model_id.shape[0],
            len(self.material_keys),
        }
        if len(lengths) != 1:
            raise ValueError("material tensors must have the same length")
        layer_lengths = {
            self.layer_thickness_m.shape[0],
            self.layer_eps_r.shape[0],
            self.layer_sigma_e.shape[0],
            self.layer_mu_r.shape[0],
        }
        if len(layer_lengths) != 1:
            raise ValueError("layer tensors must have the same length")
        total_layers = self.layer_thickness_m.shape[0]
        counts = self.layer_count.to(dtype=torch.int64)
        offsets = self.layer_offset.to(dtype=torch.int64)
        if self.layer_count.numel():
            if bool((counts < 1).any()):
                raise ValueError("layer_count must be >= 1 for every material")
            expected_offsets = torch.cumsum(counts, dim=0) - counts
            if not torch.equal(offsets, expected_offsets):
                raise ValueError("layer_offset must be the exclusive scan of layer_count")
        if int(counts.sum()) != total_layers:
            raise ValueError("layer_count must sum to the layer tensor length")
        if self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive")
        if self.abi_version != 3:
            raise ValueError("MaterialStore requires material ABI version 3")
        if self.material_id.numel() and not torch.equal(
            self.material_id, torch.arange(self.material_id.numel(), dtype=torch.int32)
        ):
            raise ValueError(
                "material_id must be dense and stable in [0, material_count)"
            )
        if len(set(self.material_keys)) != len(self.material_keys):
            raise ValueError("material_keys must be unique")
