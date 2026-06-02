"""Wedge-frame diffraction geometry: projections, edge angles, pole/exterior masks.

Pure DrJit math with no scene or solver state. Shared by the Monte Carlo and
deterministic diffraction solvers.
"""

from __future__ import annotations

from dataclasses import dataclass

import drjit as dr
from witwin.channel import types as wt

from witwin.channel.core.numerics.arrays import broadcast
from witwin.channel.core.numerics.constants import EPS, SMALL_EPS


@dataclass(slots=True)
class WedgeGeometry:
    """Edge-local geometry for a (source, edge, target) triple."""
    phi: object
    phi_prime: object
    s_proj: object
    s_prime_proj: object
    s: object
    s_prime: object
    sin_beta: object
    sin_beta_prime: object
    sin_beta_eff: object
    edge_hat: object


def project_to_wedge_plane(vec, edge_dir):
    """Project ``vec`` into the plane perpendicular to ``edge_dir``."""
    return vec - dr.dot(vec, edge_dir) * edge_dir


def normalize_in_wedge_plane(vec, edge_dir):
    """Project ``vec`` into the wedge plane and normalize (EPS-floored)."""
    vec_proj = project_to_wedge_plane(vec, edge_dir)
    return vec_proj / (dr.norm(vec_proj) + EPS)


def distance_to_cot_pole(arg):
    """Minimum distance from ``arg`` to the nearest integer multiple of pi."""
    nearest_pole = dr.round(arg / dr.pi) * dr.pi
    return dr.abs(arg - nearest_pole)


def edge_angles(source_pos, edge_pos, edge_dir, n0, target_pos):
    """Return ``(phi, phi_prime, s_proj, s_prime_proj)`` for a wedge edge.

    ``phi`` / ``phi_prime`` are diffraction azimuth angles measured from the
    `
0`` face in the plane perpendicular to ``edge_dir``. ``s_proj`` and
    ``s_prime_proj`` are the in-plane source/target distances."""
    target_width = dr.width(target_pos.x)
    edge_pos_b = broadcast(edge_pos, target_width)
    edge_dir_b = broadcast(edge_dir, target_width)
    n0_b = broadcast(n0, target_width)

    source_to_edge = edge_pos - source_pos
    source_to_edge_proj = project_to_wedge_plane(source_to_edge, edge_dir)
    s_prime = dr.norm(source_to_edge_proj) + EPS
    to_hat = dr.normalize(dr.cross(n0, edge_dir))
    ki_proj = source_to_edge_proj / s_prime

    phi_prime = dr.pi - dr.safe_acos(dr.clip(-dr.dot(ki_proj, to_hat), -1.0, 1.0))
    phi_prime = phi_prime * (-dr.sign(-dr.dot(ki_proj, n0)))
    phi_prime = phi_prime + dr.pi

    edge_to_target = target_pos - edge_pos_b
    edge_to_target_proj = project_to_wedge_plane(edge_to_target, edge_dir_b)
    s = dr.norm(edge_to_target_proj) + EPS

    to_hat_b = broadcast(to_hat, target_width)
    ko_proj = edge_to_target_proj / s

    phi = dr.pi - dr.safe_acos(dr.clip(dr.dot(ko_proj, to_hat_b), -1.0, 1.0))
    phi = phi * (-dr.sign(dr.dot(ko_proj, n0_b)))
    phi = phi + dr.pi

    return phi, broadcast(phi_prime, target_width), s, broadcast(s_prime, target_width)


def wedge_geometry(source_pos, edge_pos, edge_dir, n0, target_pos) -> WedgeGeometry:
    """Full wedge geometry: angles, distances, projected sines, edge unit vector."""
    phi, phi_prime, s_proj, s_prime_proj = edge_angles(source_pos, edge_pos, edge_dir, n0, target_pos)
    width = dr.width(target_pos.x)
    edge_pos_b = broadcast(edge_pos, width)
    edge_dir_b = broadcast(edge_dir, width)
    source_pos_b = broadcast(source_pos, width)
    edge_hat = edge_dir_b / (dr.norm(edge_dir_b) + EPS)
    source_to_edge = edge_pos_b - source_pos_b
    edge_to_target = target_pos - edge_pos_b
    s_prime = dr.norm(source_to_edge) + EPS
    s = dr.norm(edge_to_target) + EPS
    sin_beta_prime = dr.clip(s_prime_proj / s_prime, wt.Float(SMALL_EPS), wt.Float(1.0))
    sin_beta = dr.clip(s_proj / s, wt.Float(SMALL_EPS), wt.Float(1.0))
    sin_beta_eff = dr.sqrt(dr.maximum(sin_beta * sin_beta_prime, wt.Float(SMALL_EPS)))
    return WedgeGeometry(
        phi=phi, phi_prime=phi_prime,
        s_proj=s_proj, s_prime_proj=s_prime_proj,
        s=s, s_prime=s_prime,
        sin_beta=sin_beta, sin_beta_prime=sin_beta_prime,
        sin_beta_eff=sin_beta_eff,
        edge_hat=edge_hat,
    )


def incident_edge_geometry(source_pos, edge_pos, edge_dir, n0):
    """Return ``(phi_prime, s_prime)`` for the source-to-edge ray only."""
    source_to_edge = edge_pos - source_pos
    source_to_edge_proj = project_to_wedge_plane(source_to_edge, edge_dir)
    s_prime = dr.norm(source_to_edge_proj) + EPS
    to_hat = dr.normalize(dr.cross(n0, edge_dir))
    ki_proj = source_to_edge_proj / s_prime
    phi_prime = dr.pi - dr.safe_acos(dr.clip(-dr.dot(ki_proj, to_hat), -1.0, 1.0))
    phi_prime = phi_prime * (-dr.sign(-dr.dot(ki_proj, n0)))
    phi_prime = phi_prime + dr.pi
    return phi_prime, s_prime


def cotangent_pole_safe_mask(phi, phi_prime, wedge_n, pole_guard):
    """True where all four UTD cotangent arguments stay clear of integer*pi
    poles by at least ``pole_guard``."""
    two_n = 2.0 * wedge_n
    dif_phi = phi - phi_prime
    sum_phi = phi + phi_prime
    args = [
        (dr.pi + dif_phi) / two_n,
        (dr.pi - dif_phi) / two_n,
        (dr.pi + sum_phi) / two_n,
        (dr.pi - sum_phi) / two_n,
    ]
    safe = dr.full(wt.Bool, True, dr.width(phi))
    for arg in args:
        safe = safe & (distance_to_cot_pole(arg) > pole_guard)
    return safe


def slope_derivative_safe_mask(phi, phi_prime, wedge_n, step):
    """True where ``phi``/``phi_prime`` are interior to the wedge by ``step``
    and the UTD cotangent arguments are pole-safe for for slope-derivative
    evaluation."""
    n_pi = wedge_n * dr.pi
    interior = (
        (phi >= step) & (phi <= (n_pi - step))
        & (phi_prime >= step) & (phi_prime <= (n_pi - step))
    )
    pole_guard = step / (2.0 * wedge_n)
    return interior & cotangent_pole_safe_mask(phi, phi_prime, wedge_n, pole_guard)


def wedge_exterior_mask(direction_from_edge, edge_dir, n0, nn):
    """Half-space test: does ``direction_from_edge`` lie on or outside the
    wedge boundary defined by faces with normals `
0`` and `
n``?

    Replaces hard angular clipping with a direct sign test against the two
    faces in the plane perpendicular to ``edge_dir``.
    """
    direction_proj = project_to_wedge_plane(direction_from_edge, edge_dir)
    signed_distance_0 = dr.dot(direction_proj, n0)
    signed_distance_n = dr.dot(direction_proj, nn)
    return (
        (dr.norm(direction_proj) > wt.Float(SMALL_EPS))
        & (
            (signed_distance_0 >= -wt.Float(SMALL_EPS))
            | (signed_distance_n >= -wt.Float(SMALL_EPS))
        )
    )


def rotate_vector_around_axis(vec, axis, theta):
    """Rotate ``vec`` around unit ``axis`` by ``theta`` radians (Rodrigues formula)."""
    sin_theta, cos_theta = dr.sincos(theta)
    axis_dot_vec = dr.dot(axis, vec)
    axis_cross_vec = dr.cross(axis, vec)
    return (
        axis * axis_dot_vec
        + cos_theta * dr.cross(axis_cross_vec, axis)
        + sin_theta * axis_cross_vec
    )


__all__ = [
    "WedgeGeometry",
    "cotangent_pole_safe_mask",
    "distance_to_cot_pole",
    "edge_angles",
    "incident_edge_geometry",
    "normalize_in_wedge_plane",
    "project_to_wedge_plane",
    "rotate_vector_around_axis",
    "slope_derivative_safe_mask",
    "wedge_exterior_mask",
    "wedge_geometry",
]
