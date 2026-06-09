from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
import torch

from . import _C


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


@dataclass(frozen=True, slots=True)
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
                torch.empty((0,), device=self.o.device, dtype=self.o.dtype),
            )
        else:
            _require_float_cuda_tensor(self.tmax, "Ray.tmax", None)
            if self.tmax.ndim != 1 or (self.tmax.numel() != 0 and self.tmax.shape[0] != self.o.shape[0]):
                raise ValueError("Ray.tmax must be empty or have shape (N,).")


@dataclass(frozen=True, slots=True)
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
        if _C is not None and self.t.device.type == "cuda":
            return torch.ops.raydn.intersection_valid(self.t, self.shape_id)
        if self.shape_id.numel() != self.t.numel():
            return torch.isfinite(self.t)
        return self.shape_id >= 0


class _LazyIntersection:
    __slots__ = ("_load_t", "_load_full", "_t", "_full")

    def __init__(self, load_t, load_full) -> None:
        self._load_t = load_t
        self._load_full = load_full
        self._t: torch.Tensor | None = None
        self._full: Intersection | None = None

    def _ensure_full(self) -> Intersection:
        if self._full is None:
            self._full = self._load_full()
        return self._full

    @property
    def t(self) -> torch.Tensor:
        if self._t is not None:
            return self._t
        if self._full is not None:
            return self._full.t
        self._t = self._load_t()
        return self._t

    @property
    def p(self) -> torch.Tensor:
        return self._ensure_full().p

    @property
    def n(self) -> torch.Tensor:
        return self._ensure_full().n

    @property
    def geo_n(self) -> torch.Tensor:
        return self._ensure_full().geo_n

    @property
    def uv(self) -> torch.Tensor:
        return self._ensure_full().uv

    @property
    def barycentric(self) -> torch.Tensor:
        return self._ensure_full().barycentric

    @property
    def shape_id(self) -> torch.Tensor:
        return self._ensure_full().shape_id

    @property
    def prim_id(self) -> torch.Tensor:
        return self._ensure_full().prim_id

    @property
    def local_prim_id(self) -> torch.Tensor:
        return self._ensure_full().local_prim_id

    @property
    def global_prim_id(self) -> torch.Tensor:
        return self._ensure_full().global_prim_id

    def is_valid(self) -> torch.Tensor:
        return self._ensure_full().is_valid()


class _ReducedIntersection:
    __slots__ = ("_scene", "t", "_fields")

    def __init__(self, scene_handle: int, t: torch.Tensor) -> None:
        self._scene = scene_handle
        self.t = t
        self._fields: tuple[torch.Tensor, ...] | None = None

    def _empty_fields(self) -> tuple[torch.Tensor, ...]:
        if self._fields is None:
            self._fields = torch.ops.raydn.intersection_empty_fields(self._scene, self.t)
        return self._fields

    @property
    def p(self) -> torch.Tensor:
        return self._empty_fields()[0]

    @property
    def n(self) -> torch.Tensor:
        return self._empty_fields()[1]

    @property
    def geo_n(self) -> torch.Tensor:
        return self._empty_fields()[2]

    @property
    def uv(self) -> torch.Tensor:
        return self._empty_fields()[3]

    @property
    def barycentric(self) -> torch.Tensor:
        return self._empty_fields()[4]

    @property
    def shape_id(self) -> torch.Tensor:
        return self._empty_fields()[5]

    @property
    def prim_id(self) -> torch.Tensor:
        return self._empty_fields()[6]

    @property
    def local_prim_id(self) -> torch.Tensor:
        return self._empty_fields()[7]

    @property
    def global_prim_id(self) -> torch.Tensor:
        return self._empty_fields()[8]

    def is_valid(self) -> torch.Tensor:
        return torch.ops.raydn.intersection_valid(self.t, self.shape_id)


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


class ReflectionChain:
    __slots__ = ("_valid", "_t", "_image_sources", "_prim_ids", "_loader")

    def __init__(
        self,
        valid: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        image_sources: torch.Tensor | None = None,
        prim_ids: torch.Tensor | None = None,
        *,
        loader=None,
    ) -> None:
        self._valid = valid
        self._t = t
        self._image_sources = image_sources
        self._prim_ids = prim_ids
        self._loader = loader

    def _ensure_reduced(self) -> None:
        if self._valid is not None and self._t is not None and self._prim_ids is not None:
            return
        if self._loader is None:
            raise RuntimeError("ReflectionChain has no trace data loader.")
        valid, t, image_sources, prim_ids = self._loader(False)
        self._valid = valid
        self._t = t
        self._prim_ids = prim_ids
        if image_sources is not None:
            self._image_sources = image_sources

    def _ensure_full(self) -> None:
        if self._image_sources is not None:
            return
        if self._loader is None:
            raise RuntimeError("ReflectionChain has no image-source data.")
        valid, t, image_sources, prim_ids = self._loader(True)
        self._valid = valid
        self._t = t
        self._image_sources = image_sources
        self._prim_ids = prim_ids

    @property
    def valid(self) -> torch.Tensor:
        self._ensure_reduced()
        return self._valid

    @property
    def t(self) -> torch.Tensor:
        self._ensure_reduced()
        return self._t

    @property
    def image_sources(self) -> torch.Tensor:
        self._ensure_full()
        return self._image_sources

    @property
    def prim_ids(self) -> torch.Tensor:
        self._ensure_reduced()
        return self._prim_ids


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
