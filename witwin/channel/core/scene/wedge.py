"""Wedge data structures, configuration, and stateless pipeline operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import drjit as dr
from witwin.channel import types as wt

from witwin.channel.core.numerics.arrays import gather, safe_normalize


BoundaryPolicy = Literal["exclude", "half_plane"]


@dataclass(frozen=True)
class WedgeConfig:
    boundary_policy: BoundaryPolicy = "exclude"
    vertical_only: bool = False
    vertical_ratio: float = 0.7
    min_wedge_n: float = 1.0 + 1e-6
    epsilon: float = 1e-6


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


@dataclass(frozen=True)
class WedgeSelection:
    geometry: WedgeGeometry
    selected_idx: wt.UInt32
    selected_mask: wt.Bool

    def size(self) -> int:
        return int(dr.width(self.selected_idx))


@dataclass(frozen=True)
class WedgeAnchorView:
    wedge_idx: wt.UInt32
    anchor_pos: wt.Point3f
    anchor_t: wt.Float
    is_clamped: wt.Bool

    def size(self) -> int:
        return int(dr.width(self.wedge_idx))


@dataclass(frozen=True)
class TriangleWedgeMap:
    edge0: wt.Int32
    edge1: wt.Int32
    edge2: wt.Int32
    n_triangles: int
    n_wedges: int


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


def _unsigned_angle(a: wt.Vector3f, b: wt.Vector3f, axis: wt.Vector3f) -> wt.Float:
    cross = dr.cross(a, b)
    angle = dr.atan2(dr.sign(dr.dot(cross, axis)) * dr.norm(cross), dr.dot(a, b))
    return dr.select(angle < 0.0, angle + 2.0 * dr.pi, angle)


class WedgeOps:
    """Static pipeline operations for wedge construction."""

    @staticmethod
    def build_geometry(edge_info, edge_topology, config: WedgeConfig) -> WedgeGeometry:
        n_edges = int(dr.width(edge_info.global_edge_id))
        if n_edges == 0:
            raise ValueError("Cannot build wedge geometry from empty edge data.")

        eps = config.epsilon
        edge_dir = safe_normalize(edge_info.edge, eps=eps)
        is_boundary = edge_info.is_boundary
        n0_cand = safe_normalize(edge_info.normal0, eps=eps)
        n1_cand = safe_normalize(edge_info.normal1, eps=eps)

        # Canonical normal ordering via exterior-angle comparison.
        to1 = safe_normalize(dr.cross(n0_cand, edge_dir), eps=eps)
        tn1 = safe_normalize(dr.cross(n1_cand, edge_dir), eps=eps)
        to2 = safe_normalize(dr.cross(n1_cand, edge_dir), eps=eps)
        tn2 = safe_normalize(dr.cross(n0_cand, edge_dir), eps=eps)
        choose_first = _unsigned_angle(to1, tn1, edge_dir) < _unsigned_angle(to2, tn2, edge_dir)
        ordered_n0 = dr.select(choose_first, n0_cand, n1_cand)
        ordered_nn = dr.select(choose_first, n1_cand, n0_cand)

        valid_length = edge_info.length > eps
        interior = (~is_boundary) & valid_length
        half_plane_active = config.boundary_policy == "half_plane"
        half_plane = (is_boundary & valid_length) if half_plane_active else dr.zeros(wt.Bool, n_edges)

        ordered_n0 = dr.select(interior, ordered_n0, n0_cand)
        ordered_nn = dr.select(interior, ordered_nn, n1_cand)
        if half_plane_active:
            ordered_n0 = dr.select(is_boundary, n0_cand, ordered_n0)
            ordered_nn = dr.select(is_boundary, -n0_cand, ordered_nn)

        interior_angle = dr.acos(dr.clip(-dr.dot(ordered_n0, ordered_nn), -1.0, 1.0))
        exterior_angle = dr.select(interior, 2.0 * dr.pi - interior_angle, wt.Float(0.0))
        if half_plane_active:
            exterior_angle = dr.select(half_plane, wt.Float(2.0 * dr.pi), exterior_angle)

        return WedgeGeometry(
            n_edges=n_edges,
            start=edge_info.start, end=edge_info.end,
            edge_dir=edge_dir, length=edge_info.length,
            n0=ordered_n0, nn=ordered_nn,
            wedge_n=exterior_angle / dr.pi,
            exterior_angle=exterior_angle,
            is_boundary=is_boundary,
            is_valid=interior | half_plane,
            shape_id=edge_info.shape_id,
            local_edge_id=edge_info.local_edge_id,
            global_edge_id=edge_info.global_edge_id,
            face0=edge_topology.face0_global,
            face1=edge_topology.face1_global,
            v0=wt.Int32(edge_topology.v0_global),
            v1=wt.Int32(edge_topology.v1_global),
        )

    @staticmethod
    def select(geometry: WedgeGeometry, config: WedgeConfig) -> WedgeSelection:
        valid = geometry.is_valid & (geometry.length > 0.0) & (geometry.wedge_n > config.min_wedge_n)
        mask = (valid & (dr.abs(geometry.edge_dir.z) > config.vertical_ratio)) if config.vertical_only else valid
        return WedgeSelection(geometry=geometry, selected_idx=dr.compress(mask), selected_mask=mask)

    @staticmethod
    def build_midpoint_anchors(selection: WedgeSelection) -> WedgeAnchorView:
        idx = selection.selected_idx
        start = gather(selection.geometry.start, idx)
        end = gather(selection.geometry.end, idx)
        n = selection.size()
        return WedgeAnchorView(
            wedge_idx=idx, anchor_pos=(start + end) * 0.5,
            anchor_t=dr.full(wt.Float, 0.5, n),
            is_clamped=dr.full(wt.Bool, False, n),
        )

    @staticmethod
    def build_anchors(selection: WedgeSelection, z: float, clamp_eps: float = 1e-6) -> WedgeAnchorView:
        idx = selection.selected_idx
        start = gather(selection.geometry.start, idx)
        end = gather(selection.geometry.end, idx)
        dz = end.z - start.z
        horizontal = dr.abs(dz) <= clamp_eps
        t = dr.select(horizontal, wt.Float(0.5), dr.clip((wt.Float(z) - start.z) / dz, 0.0, 1.0))
        return WedgeAnchorView(
            wedge_idx=idx, anchor_pos=start + (end - start) * t, anchor_t=t,
            is_clamped=(t <= clamp_eps) | (t >= 1.0 - clamp_eps),
        )

    @staticmethod
    def build_triangle_map(selection: WedgeSelection, n_triangles: int) -> TriangleWedgeMap:
        geometry = selection.geometry
        n_selected = selection.size()
        if n_selected == 0 or n_triangles == 0:
            empty = dr.full(wt.Int32, -1, n_triangles)
            return TriangleWedgeMap(edge0=empty, edge1=empty, edge2=empty, n_triangles=n_triangles, n_wedges=0)

        sel_idx = selection.selected_idx
        local_ids = dr.arange(wt.Int32, n_selected)
        face0 = wt.Int32(dr.gather(type(geometry.face0), geometry.face0, sel_idx))
        face1 = wt.Int32(dr.gather(type(geometry.face1), geometry.face1, sel_idx))
        slots = dr.full(wt.Int32, -1, n_triangles * 3)
        counters = dr.zeros(wt.UInt32, n_triangles)

        valid0, valid1 = face0 >= 0, face1 >= 0
        safe_face0 = wt.UInt32(dr.select(valid0, face0, wt.Int32(0)))
        safe_face1 = wt.UInt32(dr.select(valid1, face1, wt.Int32(0)))
        dr.scatter(slots, local_ids, safe_face0 * 3 + dr.scatter_inc(counters, safe_face0, valid0), valid0)
        dr.scatter(slots, local_ids, safe_face1 * 3 + dr.scatter_inc(counters, safe_face1, valid1), valid1)

        tri_slot = dr.arange(wt.UInt32, n_triangles) * 3
        return TriangleWedgeMap(
            edge0=dr.gather(wt.Int32, slots, tri_slot),
            edge1=dr.gather(wt.Int32, slots, tri_slot + 1),
            edge2=dr.gather(wt.Int32, slots, tri_slot + 2),
            n_triangles=n_triangles, n_wedges=n_selected,
        )

    @staticmethod
    def pack(selection: WedgeSelection, anchors: WedgeAnchorView) -> WedgePack:
        geometry = selection.geometry
        idx = anchors.wedge_idx
        n = anchors.size()
        length = dr.gather(wt.Float, geometry.length, idx)
        t = anchors.anchor_t
        return WedgePack(
            n_wedges=n,
            pos=anchors.anchor_pos,
            edge_dir=gather(geometry.edge_dir, idx),
            length=length,
            line_min=-t * length,
            line_max=(wt.Float(1.0) - t) * length,
            n0=gather(geometry.n0, idx),
            nn=gather(geometry.nn, idx),
            wedge_n=dr.gather(wt.Float, geometry.wedge_n, idx),
            adjacent_face0=wt.Int32(dr.gather(type(geometry.face0), geometry.face0, idx)),
            adjacent_face1=wt.Int32(dr.gather(type(geometry.face1), geometry.face1, idx)),
            global_idx=wt.Int32(dr.gather(type(geometry.global_edge_id), geometry.global_edge_id, idx)),
            local_idx=dr.arange(wt.UInt32, n),
            is_boundary=dr.gather(wt.Bool, geometry.is_boundary, idx),
            is_clamped=anchors.is_clamped,
        )
