"""Wedge data structures, configuration, and backend contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import drjit as dr
import witwin as wt

from ...utils.drjit_ops import ArrayInit


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BoundaryPolicy = Literal["exclude", "half_plane"]
SelectionMode = Literal["all_edges", "vertical_only"]


@dataclass(frozen=True)
class WedgeGeometryConfig:
    boundary_policy: BoundaryPolicy = "exclude"
    wedge_convention: Literal["channel_utd"] = "channel_utd"
    epsilon: float = 1e-6


@dataclass(frozen=True)
class WedgeSelectionConfig:
    mode: SelectionMode = "vertical_only"
    vertical_ratio: float = 0.7
    min_wedge_n: float = 1.0 + 1e-6


@dataclass(frozen=True)
class HeightPlaneAnchorSpec:
    z: float
    clamp_epsilon: float = 1e-6


# ---------------------------------------------------------------------------
# Backend contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EdgeInfoBuffer:
    start: wt.Point3f
    edge: wt.Vector3f
    end: wt.Point3f
    length: wt.Float
    normal0: wt.Vector3f
    normal1: wt.Vector3f
    is_boundary: wt.Bool
    shape_id: wt.Int32
    local_edge_id: wt.Int32
    global_edge_id: wt.Int32

    def size(self) -> int:
        return int(dr.width(self.global_edge_id))

    @staticmethod
    def empty() -> "EdgeInfoBuffer":
        return EdgeInfoBuffer(
            start=ArrayInit.empty_point3(),
            edge=ArrayInit.empty_vector3(),
            end=ArrayInit.empty_point3(),
            length=dr.zeros(wt.Float, 0),
            normal0=ArrayInit.empty_vector3(),
            normal1=ArrayInit.empty_vector3(),
            is_boundary=dr.zeros(wt.Bool, 0),
            shape_id=dr.zeros(wt.Int32, 0),
            local_edge_id=dr.zeros(wt.Int32, 0),
            global_edge_id=dr.zeros(wt.Int32, 0),
        )


@dataclass(frozen=True)
class EdgeTopologyBuffer:
    v0: wt.Int32
    v1: wt.Int32
    face0_local: wt.Int32
    face1_local: wt.Int32
    face0_global: wt.Int32
    face1_global: wt.Int32
    opposite_vertex0: wt.Int32
    opposite_vertex1: wt.Int32

    def size(self) -> int:
        return int(dr.width(self.v0))

    @staticmethod
    def empty() -> "EdgeTopologyBuffer":
        empty_int = dr.zeros(wt.Int32, 0)
        return EdgeTopologyBuffer(
            v0=empty_int,
            v1=empty_int,
            face0_local=empty_int,
            face1_local=empty_int,
            face0_global=empty_int,
            face1_global=empty_int,
            opposite_vertex0=empty_int,
            opposite_vertex1=empty_int,
        )


@runtime_checkable
class EdgeTopologyBackend(Protocol):
    def version(self) -> int: ...
    def edge_version(self) -> int: ...
    def n_triangles(self) -> int: ...
    def edge_info(self) -> EdgeInfoBuffer: ...
    def edge_topology(self) -> EdgeTopologyBuffer: ...
    def triangle_edge_indices(self, prim_id, global_: bool = True): ...
    def edge_adjacent_faces(self, edge_id, global_: bool = True): ...
    def mesh_face_offsets(self): ...
    def mesh_edge_offsets(self): ...


@runtime_checkable
class SurfaceQueryBackend(Protocol):
    def shadow_test(self, ray, active=True): ...
    def intersect(self, ray, active=True, flags=None): ...


@runtime_checkable
class WedgeBackend(EdgeTopologyBackend, SurfaceQueryBackend, Protocol):
    pass


# ---------------------------------------------------------------------------
# Runtime data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WedgeGeometry:
    n_edges: int
    start: wt.Point3f
    end: wt.Point3f
    edge_dir: wt.Vector3f
    length: wt.Float
    n0: wt.Vector3f
    nn: wt.Vector3f
    wedge_n: wt.Float
    exterior_angle: wt.Float
    is_boundary: wt.Bool
    is_valid: wt.Bool
    shape_id: wt.Int32
    local_edge_id: wt.Int32
    global_edge_id: wt.Int32
    face0: wt.Int32
    face1: wt.Int32
    v0: wt.Int32
    v1: wt.Int32

    @staticmethod
    def empty() -> "WedgeGeometry":
        empty_float = dr.zeros(wt.Float, 0)
        empty_bool = dr.zeros(wt.Bool, 0)
        empty_int = dr.zeros(wt.Int32, 0)
        return WedgeGeometry(
            n_edges=0,
            start=ArrayInit.empty_point3(),
            end=ArrayInit.empty_point3(),
            edge_dir=ArrayInit.empty_vector3(),
            length=empty_float,
            n0=ArrayInit.empty_vector3(),
            nn=ArrayInit.empty_vector3(),
            wedge_n=empty_float,
            exterior_angle=empty_float,
            is_boundary=empty_bool,
            is_valid=empty_bool,
            shape_id=empty_int,
            local_edge_id=empty_int,
            global_edge_id=empty_int,
            face0=empty_int,
            face1=empty_int,
            v0=empty_int,
            v1=empty_int,
        )


@dataclass(frozen=True)
class WedgeSelection:
    geometry: WedgeGeometry
    selected_idx: wt.UInt32
    selected_mask: wt.Bool
    summary: dict[str, int] = field(default_factory=dict)

    def size(self) -> int:
        return int(dr.width(self.selected_idx))

    @staticmethod
    def empty(geometry: WedgeGeometry | None = None) -> "WedgeSelection":
        return WedgeSelection(
            geometry=WedgeGeometry.empty() if geometry is None else geometry,
            selected_idx=dr.zeros(wt.UInt32, 0),
            selected_mask=dr.zeros(wt.Bool, 0),
            summary={},
        )


@dataclass(frozen=True)
class WedgeAnchorView:
    wedge_idx: wt.UInt32
    anchor_pos: wt.Point3f
    anchor_t: wt.Float
    is_clamped: wt.Bool
    summary: dict[str, int] = field(default_factory=dict)

    def size(self) -> int:
        return int(dr.width(self.wedge_idx))

    @staticmethod
    def empty() -> "WedgeAnchorView":
        return WedgeAnchorView(
            wedge_idx=dr.zeros(wt.UInt32, 0),
            anchor_pos=ArrayInit.empty_point3(),
            anchor_t=dr.zeros(wt.Float, 0),
            is_clamped=dr.zeros(wt.Bool, 0),
            summary={},
        )


@dataclass(frozen=True)
class TriangleWedgeMap:
    edge0: wt.Int32
    edge1: wt.Int32
    edge2: wt.Int32
    n_triangles: int
    n_wedges: int

    @staticmethod
    def empty() -> "TriangleWedgeMap":
        empty_int = dr.zeros(wt.Int32, 0)
        return TriangleWedgeMap(edge0=empty_int, edge1=empty_int, edge2=empty_int, n_triangles=0, n_wedges=0)


@dataclass(frozen=True)
class WedgePack:
    n_wedges: int
    pos: wt.Point3f
    edge_dir: wt.Vector3f
    length: wt.Float
    line_min: wt.Float
    line_max: wt.Float
    n0: wt.Vector3f
    nn: wt.Vector3f
    wedge_n: wt.Float
    adjacent_face0: wt.Int32
    adjacent_face1: wt.Int32
    global_idx: wt.Int32
    local_idx: wt.UInt32
    is_boundary: wt.Bool
    is_clamped: wt.Bool
    summary: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def empty() -> "WedgePack":
        empty_float = dr.zeros(wt.Float, 0)
        empty_bool = dr.zeros(wt.Bool, 0)
        empty_int = dr.zeros(wt.Int32, 0)
        empty_uint = dr.zeros(wt.UInt32, 0)
        return WedgePack(
            n_wedges=0,
            pos=ArrayInit.empty_point3(),
            edge_dir=ArrayInit.empty_vector3(),
            length=empty_float,
            line_min=empty_float,
            line_max=empty_float,
            n0=ArrayInit.empty_vector3(),
            nn=ArrayInit.empty_vector3(),
            wedge_n=empty_float,
            adjacent_face0=empty_int,
            adjacent_face1=empty_int,
            global_idx=empty_int,
            local_idx=empty_uint,
            is_boundary=empty_bool,
            is_clamped=empty_bool,
            summary={},
        )
