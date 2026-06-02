"""Edge angle computation and geometry setup helpers for diffraction."""

import drjit as dr
import witwin as wt

from ....utils.constants import EPS, SMALL_EPS
from ....utils.drjit_ops import Broadcast, broadcast_point, broadcast_vector, repeat_float


def _project_to_wedge_plane(vec, edge_dir):
    return vec - dr.dot(vec, edge_dir) * edge_dir


def _normalize_in_wedge_plane(vec, edge_dir):
    vec_proj = _project_to_wedge_plane(vec, edge_dir)
    return vec_proj / (dr.norm(vec_proj) + EPS)


def _compute_edge_angles(source_pos, edge_pos, edge_dir, n0, target_pos):
    target_width = dr.width(target_pos.x)
    edge_pos_b = broadcast_point(edge_pos, target_width)
    edge_dir_b = broadcast_vector(edge_dir, target_width)
    n0_b = broadcast_vector(n0, target_width)

    source_to_edge = edge_pos - source_pos
    source_to_edge_proj = _project_to_wedge_plane(source_to_edge, edge_dir)
    s_prime = dr.norm(source_to_edge_proj) + EPS
    to_hat = dr.normalize(dr.cross(n0, edge_dir))
    ki_proj = source_to_edge_proj / s_prime

    phi_prime = dr.pi - dr.safe_acos(dr.clip(-dr.dot(ki_proj, to_hat), -1.0, 1.0))
    phi_prime = phi_prime * (-dr.sign(-dr.dot(ki_proj, n0)))
    phi_prime = phi_prime + dr.pi

    edge_to_target = target_pos - edge_pos_b
    edge_to_target_proj = _project_to_wedge_plane(edge_to_target, edge_dir_b)
    s = dr.norm(edge_to_target_proj) + EPS

    to_hat_b = broadcast_vector(to_hat, target_width)
    ko_proj = edge_to_target_proj / s

    phi = dr.pi - dr.safe_acos(dr.clip(dr.dot(ko_proj, to_hat_b), -1.0, 1.0))
    phi = phi * (-dr.sign(dr.dot(ko_proj, n0_b)))
    phi = phi + dr.pi

    return phi, repeat_float(phi_prime, target_width), s, repeat_float(s_prime, target_width)


def _compute_edge_geometry(source_pos, edge_pos, edge_dir, n0, target_pos):
    phi, phi_prime, s_proj, s_prime_proj = _compute_edge_angles(
        source_pos,
        edge_pos,
        edge_dir,
        n0,
        target_pos,
    )
    width = dr.width(target_pos.x)
    edge_pos_b = broadcast_point(edge_pos, width)
    edge_dir_b = broadcast_vector(edge_dir, width)
    source_pos_b = broadcast_point(source_pos, width)
    edge_hat = edge_dir_b / (dr.norm(edge_dir_b) + EPS)
    source_to_edge = edge_pos_b - source_pos_b
    edge_to_target = target_pos - edge_pos_b
    s_prime = dr.norm(source_to_edge) + EPS
    s = dr.norm(edge_to_target) + EPS
    sin_beta_prime = dr.clip(s_prime_proj / s_prime, wt.Float(SMALL_EPS), wt.Float(1.0))
    sin_beta = dr.clip(s_proj / s, wt.Float(SMALL_EPS), wt.Float(1.0))
    # Exact Keller-cone anchors satisfy sin(beta)=sin(beta'). Existing anchors are
    # approximate, so use a symmetric effective beta that preserves the projected-length product.
    sin_beta_eff = dr.sqrt(dr.maximum(sin_beta * sin_beta_prime, wt.Float(SMALL_EPS)))
    return {
        "phi": phi,
        "phi_prime": phi_prime,
        "s_proj": s_proj,
        "s_prime_proj": s_prime_proj,
        "s": s,
        "s_prime": s_prime,
        "sin_beta": sin_beta,
        "sin_beta_prime": sin_beta_prime,
        "sin_beta_eff": sin_beta_eff,
        "edge_hat": edge_hat,
    }


def _compute_incident_edge_geometry(source_pos, edge_pos, edge_dir, n0):
    source_to_edge = edge_pos - source_pos
    source_to_edge_proj = _project_to_wedge_plane(source_to_edge, edge_dir)
    s_prime = dr.norm(source_to_edge_proj) + EPS
    to_hat = dr.normalize(dr.cross(n0, edge_dir))
    ki_proj = source_to_edge_proj / s_prime
    phi_prime = dr.pi - dr.safe_acos(dr.clip(-dr.dot(ki_proj, to_hat), -1.0, 1.0))
    phi_prime = phi_prime * (-dr.sign(-dr.dot(ki_proj, n0)))
    phi_prime = phi_prime + dr.pi
    return phi_prime, s_prime


__all__ = [
    "_project_to_wedge_plane",
    "_normalize_in_wedge_plane",
    "_compute_edge_angles",
    "_compute_edge_geometry",
    "_compute_incident_edge_geometry",
]
