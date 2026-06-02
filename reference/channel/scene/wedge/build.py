"""Build raw edge/topology buffers into wedge geometry semantics."""

from __future__ import annotations

import drjit as dr
import witwin as wt

from ...utils.drjit_ops import safe_normalize  # standalone helper
from .types import EdgeInfoBuffer, EdgeTopologyBuffer, WedgeGeometry, WedgeGeometryConfig


def build_wedge_geometry(
    edge_info: EdgeInfoBuffer,
    edge_topology: EdgeTopologyBuffer,
    config: WedgeGeometryConfig | None = None,
) -> WedgeGeometry:
    config = WedgeGeometryConfig() if config is None else config
    n_edges = edge_info.size()
    if n_edges == 0:
        return WedgeGeometry.empty()

    start = edge_info.start
    end = edge_info.end
    edge_vec = edge_info.edge
    length = edge_info.length
    edge_dir = safe_normalize(edge_vec, eps=config.epsilon)
    is_boundary = edge_info.is_boundary

    n0_candidate = safe_normalize(edge_info.normal0, eps=config.epsilon)
    n1_candidate = safe_normalize(edge_info.normal1, eps=config.epsilon)

    to_hat_1 = safe_normalize(dr.cross(n0_candidate, edge_dir), eps=config.epsilon)
    tn_hat_1 = safe_normalize(dr.cross(n1_candidate, edge_dir), eps=config.epsilon)
    to_hat_2 = safe_normalize(dr.cross(n1_candidate, edge_dir), eps=config.epsilon)
    tn_hat_2 = safe_normalize(dr.cross(n0_candidate, edge_dir), eps=config.epsilon)

    cross_1 = dr.cross(to_hat_1, tn_hat_1)
    dot_1 = dr.dot(to_hat_1, tn_hat_1)
    sign_1 = dr.sign(dr.dot(cross_1, edge_dir))
    angle_1 = dr.atan2(sign_1 * dr.norm(cross_1), dot_1)
    angle_1 = dr.select(angle_1 < 0.0, angle_1 + 2.0 * dr.pi, angle_1)

    cross_2 = dr.cross(to_hat_2, tn_hat_2)
    dot_2 = dr.dot(to_hat_2, tn_hat_2)
    sign_2 = dr.sign(dr.dot(cross_2, edge_dir))
    angle_2 = dr.atan2(sign_2 * dr.norm(cross_2), dot_2)
    angle_2 = dr.select(angle_2 < 0.0, angle_2 + 2.0 * dr.pi, angle_2)

    choose_first = angle_1 < angle_2
    ordered_n0 = dr.select(choose_first, n0_candidate, n1_candidate)
    ordered_nn = dr.select(choose_first, n1_candidate, n0_candidate)

    valid_length = length > config.epsilon
    interior_mask = (~is_boundary) & valid_length
    half_plane_mask = is_boundary & valid_length if config.boundary_policy == "half_plane" else dr.zeros(wt.Bool, n_edges)

    ordered_n0 = dr.select(interior_mask, ordered_n0, n0_candidate)
    ordered_nn = dr.select(interior_mask, ordered_nn, n1_candidate)
    if config.boundary_policy == "half_plane":
        ordered_n0 = dr.select(is_boundary, n0_candidate, ordered_n0)
        ordered_nn = dr.select(is_boundary, -n0_candidate, ordered_nn)

    interior_angle = dr.acos(dr.clip(-dr.dot(ordered_n0, ordered_nn), -1.0, 1.0))
    exterior_angle = dr.select(interior_mask, 2.0 * dr.pi - interior_angle, wt.Float(0.0))
    if config.boundary_policy == "half_plane":
        exterior_angle = dr.select(half_plane_mask, wt.Float(2.0 * dr.pi), exterior_angle)

    wedge_n = exterior_angle / dr.pi
    is_valid = interior_mask | half_plane_mask

    return WedgeGeometry(
        n_edges=n_edges,
        start=start,
        end=end,
        edge_dir=edge_dir,
        length=length,
        n0=ordered_n0,
        nn=ordered_nn,
        wedge_n=wedge_n,
        exterior_angle=exterior_angle,
        is_boundary=is_boundary,
        is_valid=is_valid,
        shape_id=edge_info.shape_id,
        local_edge_id=edge_info.local_edge_id,
        global_edge_id=edge_info.global_edge_id,
        face0=edge_topology.face0_global,
        face1=edge_topology.face1_global,
        v0=edge_topology.v0,
        v1=edge_topology.v1,
    )

