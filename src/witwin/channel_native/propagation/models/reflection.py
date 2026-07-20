"""Fixed-capacity reflection candidates after RayD EPC visibility."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .capacity import _require_cuda_tensor, _require_host_count


@dataclass(frozen=True, slots=True, eq=False)
class ReflectionCandidateCapacity:
    """Stable visible-candidate rows with device-resident cardinality."""

    candidate_capacity: int
    depth: int
    valid: torch.Tensor
    candidate_count: torch.Tensor
    overflow: torch.Tensor
    selected_sequences: torch.Tensor
    selected_hits: torch.Tensor
    selected_normals: torch.Tensor
    selected_rx_id: torch.Tensor
    selected_tx: torch.Tensor
    selected_rx: torch.Tensor
    tx_power: torch.Tensor
    eps_r: torch.Tensor
    sigma_e: torch.Tensor
    mu_r: torch.Tensor
    gain: torch.Tensor
    first_face: torch.Tensor
    material_id: torch.Tensor
    material_sequence: torch.Tensor
    first_hit: torch.Tensor
    first_normal: torch.Tensor

    def __post_init__(self) -> None:
        capacity = _require_host_count("candidate_capacity", self.candidate_capacity)
        depth = _require_host_count("depth", self.depth)
        if depth == 0:
            raise ValueError("depth must be positive")
        valid = _require_cuda_tensor(
            "valid", self.valid, dtype=torch.bool, shape=(capacity,)
        )
        _require_cuda_tensor(
            "candidate_count",
            self.candidate_count,
            dtype=torch.int32,
            shape=(1,),
            device=valid.device,
        )
        _require_cuda_tensor(
            "overflow",
            self.overflow,
            dtype=torch.bool,
            shape=(1,),
            device=valid.device,
        )
        for name in ("selected_sequences", "material_sequence"):
            _require_cuda_tensor(
                name,
                getattr(self, name),
                dtype=torch.int32,
                shape=(capacity, depth),
                device=valid.device,
            )
        for name in ("selected_hits", "selected_normals"):
            _require_cuda_tensor(
                name,
                getattr(self, name),
                dtype=torch.float32,
                shape=(capacity, depth, 3),
                device=valid.device,
            )
        for name in ("selected_rx_id", "first_face", "material_id"):
            _require_cuda_tensor(
                name,
                getattr(self, name),
                dtype=torch.int32,
                shape=(capacity,),
                device=valid.device,
            )
        for name in ("selected_tx", "selected_rx", "first_hit", "first_normal"):
            _require_cuda_tensor(
                name,
                getattr(self, name),
                dtype=torch.float32,
                shape=(capacity, 3),
                device=valid.device,
            )
        _require_cuda_tensor(
            "tx_power",
            self.tx_power,
            dtype=torch.float32,
            shape=(capacity,),
            device=valid.device,
        )
        for name in ("eps_r", "sigma_e", "mu_r", "gain"):
            _require_cuda_tensor(
                name,
                getattr(self, name),
                dtype=torch.float32,
                shape=(capacity, depth),
                device=valid.device,
            )

    @property
    def device(self) -> torch.device:
        return self.valid.device


__all__ = ["ReflectionCandidateCapacity"]
