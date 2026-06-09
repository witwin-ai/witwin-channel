from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
import torch


RayFlags = IntFlag(
    "RayFlags",
    {
        "None": 0x00,
        "Geometric": 0x01,
        "ShadingN": 0x02,
        "UV": 0x04,
        "All": 0x01 | 0x02 | 0x04,
    },
)


def _require_float_cuda_tensor(value: torch.Tensor, name: str, shape_last: int | None) -> None:
    if value.device.type != "cuda":
        raise TypeError(f"{name} must be CUDA.")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32.")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")
    if shape_last is not None and (value.ndim != 2 or value.shape[1] != shape_last):
        raise ValueError(f"{name} must have shape (N, {shape_last}).")


@dataclass(frozen=True)
class Ray:
    o: torch.Tensor
    d: torch.Tensor
    tmax: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _require_float_cuda_tensor(self.o, "Ray.o", 3)
        _require_float_cuda_tensor(self.d, "Ray.d", 3)
        if self.o.shape[0] != self.d.shape[0]:
            raise ValueError("Ray.o and Ray.d must have the same batch size.")
        if self.tmax is None:
            object.__setattr__(
                self,
                "tmax",
                torch.full((self.o.shape[0],), float("inf"), device=self.o.device, dtype=self.o.dtype),
            )
        else:
            _require_float_cuda_tensor(self.tmax, "Ray.tmax", None)
            if self.tmax.ndim != 1 or self.tmax.shape[0] != self.o.shape[0]:
                raise ValueError("Ray.tmax must have shape (N,).")


@dataclass(frozen=True)
class Intersection:
    t: torch.Tensor
    p: torch.Tensor
    n: torch.Tensor
    geo_n: torch.Tensor
    uv: torch.Tensor
    barycentric: torch.Tensor
    shape_id: torch.Tensor
    prim_id: torch.Tensor
    local_prim_id: torch.Tensor
    global_prim_id: torch.Tensor

    def is_valid(self) -> torch.Tensor:
        if self.shape_id.numel() != self.t.numel():
            return torch.isfinite(self.t)
        return self.shape_id >= 0


@dataclass(frozen=True)
class NearestPointEdge:
    distance: torch.Tensor
    edge_point: torch.Tensor
    edge_t: torch.Tensor
    shape_id: torch.Tensor
    edge_id: torch.Tensor
    global_edge_id: torch.Tensor


@dataclass(frozen=True)
class NearestRayEdge:
    distance: torch.Tensor
    ray_t: torch.Tensor
    point: torch.Tensor
    edge_t: torch.Tensor
    edge_point: torch.Tensor
    shape_id: torch.Tensor
    edge_id: torch.Tensor
    global_edge_id: torch.Tensor


@dataclass(frozen=True)
class ReflectionChain:
    valid: torch.Tensor
    t: torch.Tensor
    image_sources: torch.Tensor
    prim_ids: torch.Tensor


@dataclass(frozen=True)
class ReflEpcField:
    field_real: torch.Tensor
    field_imag: torch.Tensor
    path_length: torch.Tensor
    valid: torch.Tensor
    resolved_prim_ids: torch.Tensor


@dataclass(frozen=True)
class DfrGrid:
    axis: int = 2
    position: float = 0.0
    coord0_min: float = -1.0
    coord0_max: float = 1.0
    coord1_min: float = -1.0
    coord1_max: float = 1.0
    resolution0: int = 1
    resolution1: int = 1
    cell_area: float | None = None

    def resolved_cell_area(self) -> float:
        if self.cell_area is not None:
            return float(self.cell_area)
        span0 = float(self.coord0_max) - float(self.coord0_min)
        span1 = float(self.coord1_max) - float(self.coord1_min)
        return abs(span0 * span1) / float(int(self.resolution0) * int(self.resolution1))


@dataclass(frozen=True)
class DfrMaterial:
    eta_r: torch.Tensor
    sigma: torch.Tensor
    mu_r: torch.Tensor
    gain: torch.Tensor
    valid: torch.Tensor

    @staticmethod
    def default(count: int, *, device: torch.device, dtype: torch.dtype = torch.float32) -> "DfrMaterial":
        return DfrMaterial(
            eta_r=torch.ones((count,), device=device, dtype=dtype),
            sigma=torch.zeros((count,), device=device, dtype=dtype),
            mu_r=torch.ones((count,), device=device, dtype=dtype),
            gain=torch.ones((count,), device=device, dtype=dtype),
            valid=torch.ones((count,), device=device, dtype=torch.bool),
        )


@dataclass(frozen=True)
class DfrStates:
    edge_index: torch.Tensor
    edge_pos: torch.Tensor
    edge_dir: torch.Tensor
    edge_t_min: torch.Tensor
    edge_t_max: torch.Tensor
    n0: torch.Tensor
    n1: torch.Tensor
    prim0: torch.Tensor
    prim1: torch.Tensor
    exterior_angle: torch.Tensor
    src: torch.Tensor
    src_power: torch.Tensor
    wi: torch.Tensor | None = None
    d0: torch.Tensor | None = None
    count: int | None = None

    @property
    def state_count(self) -> int:
        return int(self.edge_index.shape[0] if self.count is None else self.count)

    def with_default_vectors(self) -> "DfrStates":
        wi = self.wi
        d0 = self.d0
        if wi is None:
            wi = torch.zeros_like(self.edge_pos)
        if d0 is None:
            d0 = torch.zeros_like(self.edge_pos)
        return DfrStates(
            self.edge_index,
            self.edge_pos,
            self.edge_dir,
            self.edge_t_min,
            self.edge_t_max,
            self.n0,
            self.n1,
            self.prim0,
            self.prim1,
            self.exterior_angle,
            self.src,
            self.src_power,
            wi,
            d0,
            self.count,
        )


@dataclass(frozen=True)
class DfrAccum:
    grid_cell_count: int
    power: torch.Tensor
    field_x_re: torch.Tensor
    field_x_im: torch.Tensor
    field_y_re: torch.Tensor
    field_y_im: torch.Tensor
    field_z_re: torch.Tensor
    field_z_im: torch.Tensor
    direct_count: torch.Tensor
    keller_count: torch.Tensor
    suffix_count: torch.Tensor
    vis_rejects: torch.Tensor
    edge_vis_rejects: torch.Tensor
    utd_rejects: torch.Tensor
    edge_uses: torch.Tensor


@dataclass(frozen=True)
class DfrCoherentAccum:
    grid_cell_count: int
    direct_field_x_re: torch.Tensor
    direct_field_x_im: torch.Tensor
    direct_field_y_re: torch.Tensor
    direct_field_y_im: torch.Tensor
    direct_field_z_re: torch.Tensor
    direct_field_z_im: torch.Tensor
    multi_field_x_re: torch.Tensor
    multi_field_x_im: torch.Tensor
    multi_field_y_re: torch.Tensor
    multi_field_y_im: torch.Tensor
    multi_field_z_re: torch.Tensor
    multi_field_z_im: torch.Tensor
    direct_count: torch.Tensor
    multi_count: torch.Tensor
    visibility_reject_count: torch.Tensor
    utd_reject_count: torch.Tensor


@dataclass(frozen=True)
class DfrPaths:
    capacity: int
    count: torch.Tensor
    valid: torch.Tensor
    tx_id: torch.Tensor
    rx_id: torch.Tensor
    order: torch.Tensor
    edge0: torch.Tensor
    edge1: torch.Tensor
    edge2: torch.Tensor
    delay: torch.Tensor
    field_x_re: torch.Tensor
    field_x_im: torch.Tensor
    field_y_re: torch.Tensor
    field_y_im: torch.Tensor
    field_z_re: torch.Tensor
    field_z_im: torch.Tensor
    p0: torch.Tensor
    p1: torch.Tensor
    p2: torch.Tensor


@dataclass(frozen=True)
class SceneGlobalGeometry:
    vertices: torch.Tensor
    faces: torch.Tensor
    face_normal: torch.Tensor
    shape_id: torch.Tensor
    local_prim_id: torch.Tensor
    global_prim_id: torch.Tensor
