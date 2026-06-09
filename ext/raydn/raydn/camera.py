from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from . import _C
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
        return sample

    def _require_point(self, point: torch.Tensor) -> torch.Tensor:
        if point.device.type != "cuda":
            raise TypeError("point must be CUDA.")
        if point.dtype != torch.float32:
            raise TypeError("point must be torch.float32.")
        if point.ndim != 2 or point.shape[1] != 3:
            raise ValueError("point must have shape (N, 3).")
        return point

    def _tan_xy(self) -> tuple[float, float]:
        tan_x = math.tan(math.radians(self.fov_x) * 0.5)
        return tan_x, tan_x / self.aspect

    def sample_to_world(self, sample: torch.Tensor, depth: float = 1.0) -> torch.Tensor:
        if _C is None:
            raise RuntimeError("RayDN extension is not built yet.")
        sample = self._require_sample(sample)
        tan_x, tan_y = self._tan_xy()
        return _CameraSampleToWorldFunction.apply(sample, tan_x, tan_y, float(depth))

    def world_to_sample(self, point: torch.Tensor) -> torch.Tensor:
        if _C is None:
            raise RuntimeError("RayDN extension is not built yet.")
        point = self._require_point(point)
        tan_x, tan_y = self._tan_xy()
        return _CameraWorldToSampleFunction.apply(point, tan_x, tan_y)

    def sample_ray(self, sample: torch.Tensor) -> Ray:
        if _C is None:
            raise RuntimeError("RayDN extension is not built yet.")
        sample = self._require_sample(sample)
        tan_x, tan_y = self._tan_xy()
        origin, direction = _CameraSampleRayFunction.apply(sample, tan_x, tan_y)
        return Ray(origin, direction)


class _CameraSampleToWorldFunction(torch.autograd.Function):
    @staticmethod
    def forward(sample: torch.Tensor, tan_x: float, tan_y: float, depth: float) -> torch.Tensor:
        return torch.ops.raydn.camera_sample_to_world(sample, float(tan_x), float(tan_y), float(depth))

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        ctx.set_materialize_grads(False)
        sample, tan_x, tan_y, depth = inputs
        ctx.sample_count = int(sample.shape[0])
        ctx.tan_x = float(tan_x)
        ctx.tan_y = float(tan_y)
        ctx.depth = float(depth)

    @staticmethod
    def backward(ctx, grad_world: torch.Tensor | None):
        if grad_world is None:
            return None, None, None, None
        grad_sample = torch.ops.raydn.camera_sample_to_world_backward(
            grad_world,
            ctx.sample_count,
            ctx.tan_x,
            ctx.tan_y,
            ctx.depth,
        )
        return grad_sample, None, None, None


class _CameraWorldToSampleFunction(torch.autograd.Function):
    @staticmethod
    def forward(point: torch.Tensor, tan_x: float, tan_y: float) -> torch.Tensor:
        return torch.ops.raydn.camera_world_to_sample(point, float(tan_x), float(tan_y))

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        ctx.set_materialize_grads(False)
        point, tan_x, tan_y = inputs
        ctx.save_for_backward(point)
        ctx.tan_x = float(tan_x)
        ctx.tan_y = float(tan_y)

    @staticmethod
    def backward(ctx, grad_sample: torch.Tensor | None):
        if grad_sample is None:
            return None, None, None
        (point,) = ctx.saved_tensors
        grad_point = torch.ops.raydn.camera_world_to_sample_backward(
            point,
            grad_sample,
            ctx.tan_x,
            ctx.tan_y,
        )
        return grad_point, None, None


class _CameraSampleRayFunction(torch.autograd.Function):
    @staticmethod
    def forward(sample: torch.Tensor, tan_x: float, tan_y: float):
        return tuple(torch.ops.raydn.camera_sample_ray(sample, float(tan_x), float(tan_y)))

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        ctx.set_materialize_grads(False)
        sample, tan_x, tan_y = inputs
        ctx.save_for_backward(sample)
        ctx.tan_x = float(tan_x)
        ctx.tan_y = float(tan_y)

    @staticmethod
    def backward(ctx, grad_origin: torch.Tensor | None, grad_direction: torch.Tensor | None):
        if grad_direction is None:
            return None, None, None
        (sample,) = ctx.saved_tensors
        grad_sample = torch.ops.raydn.camera_sample_ray_backward(
            sample,
            grad_direction,
            ctx.tan_x,
            ctx.tan_y,
        )
        return grad_sample, None, None
