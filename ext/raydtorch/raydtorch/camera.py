from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .types import Ray


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fov_x: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera width and height must be positive.")
        if self.fov_x <= 0.0 or self.fov_x >= 180.0:
            raise ValueError("Camera fov_x must be in (0, 180).")

    @property
    def aspect(self) -> float:
        return float(self.width) / float(self.height)

    def _require_sample(self, sample: torch.Tensor) -> torch.Tensor:
        if sample.device.type != "cuda":
            raise TypeError("sample must be CUDA.")
        if sample.dtype != torch.float32:
            raise TypeError("sample must be torch.float32.")
        if sample.ndim != 2 or sample.shape[1] != 2:
            raise ValueError("sample must have shape (N, 2).")
        return sample.contiguous()

    def sample_to_world(self, sample: torch.Tensor, depth: float = 1.0) -> torch.Tensor:
        sample = self._require_sample(sample)
        tan_x = math.tan(math.radians(self.fov_x) * 0.5)
        tan_y = tan_x / self.aspect
        x = (sample[:, 0] * 2.0 - 1.0) * tan_x * depth
        y = (1.0 - sample[:, 1] * 2.0) * tan_y * depth
        z = torch.full_like(x, depth)
        return torch.stack((x, y, z), dim=1).contiguous()

    def world_to_sample(self, point: torch.Tensor) -> torch.Tensor:
        if point.device.type != "cuda":
            raise TypeError("point must be CUDA.")
        if point.dtype != torch.float32:
            raise TypeError("point must be torch.float32.")
        if point.ndim != 2 or point.shape[1] != 3:
            raise ValueError("point must have shape (N, 3).")
        point = point.contiguous()
        tan_x = math.tan(math.radians(self.fov_x) * 0.5)
        tan_y = tan_x / self.aspect
        safe_z = torch.clamp(point[:, 2], min=1.0e-12)
        u = point[:, 0] / (safe_z * tan_x) * 0.5 + 0.5
        v = 0.5 - point[:, 1] / (safe_z * tan_y) * 0.5
        return torch.stack((u, v), dim=1).contiguous()

    def sample_ray(self, sample: torch.Tensor) -> Ray:
        sample = self._require_sample(sample)
        target = self.sample_to_world(sample)
        direction = torch.nn.functional.normalize(target, dim=1)
        origin_base = torch.zeros((sample.shape[0], 3), device=sample.device, dtype=sample.dtype)
        origin = origin_base + sample.sum(dim=1, keepdim=True) * 0.0
        return Ray(origin.contiguous(), direction.contiguous())
