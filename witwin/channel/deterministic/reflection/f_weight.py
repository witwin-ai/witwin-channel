"""DrJit reflection transition weights for surface-boundary crossings."""

from __future__ import annotations

from dataclasses import dataclass

import drjit as dr

from witwin.channel.core.numerics.constants import EPS, SMALL_EPS
from witwin.channel.core.physics.wave_math import f_utd
from witwin.channel.deterministic import types as wt

from .boundary import ReflectionBoundarySupport
from .secondary_visibility import SecondaryVisibilitySupport


@dataclass(frozen=True)
class ReflectionTransitionWeights:
    primary_weight: wt.Complex2f
    adjacent_weight: wt.Complex2f
    adjacent_plane_point: wt.Point3f
    adjacent_plane_normal: wt.Vector3f
    adjacent_valid: wt.Bool


def _zero_complex(width: int) -> wt.Complex2f:
    return wt.Complex2f(dr.zeros(wt.Float, width), dr.zeros(wt.Float, width))


def _one_complex(width: int) -> wt.Complex2f:
    return wt.Complex2f(dr.ones(wt.Float, width), dr.zeros(wt.Float, width))


def _zero_point(width: int) -> wt.Point3f:
    return dr.zeros(wt.Point3f, width)


def _zero_vector(width: int) -> wt.Vector3f:
    return dr.zeros(wt.Vector3f, width)


def _select_complex(mask: wt.Bool, yes: wt.Complex2f, no: wt.Complex2f) -> wt.Complex2f:
    return wt.Complex2f(
        dr.select(mask, yes.real, no.real),
        dr.select(mask, yes.imag, no.imag),
    )


def _select_point(mask: wt.Bool, yes: wt.Point3f, no: wt.Point3f) -> wt.Point3f:
    return wt.Point3f(
        dr.select(mask, yes.x, no.x),
        dr.select(mask, yes.y, no.y),
        dr.select(mask, yes.z, no.z),
    )


def _select_vector(mask: wt.Bool, yes: wt.Vector3f, no: wt.Vector3f) -> wt.Vector3f:
    return wt.Vector3f(
        dr.select(mask, yes.x, no.x),
        dr.select(mask, yes.y, no.y),
        dr.select(mask, yes.z, no.z),
    )


def _safe_transition(x: wt.Float) -> wt.Complex2f:
    safe_x = dr.maximum(x, wt.Float(SMALL_EPS))
    transition = f_utd(safe_x)
    ramp = dr.minimum(wt.Float(1.0), dr.maximum(x, wt.Float(0.0)) / wt.Float(SMALL_EPS))
    return transition * ramp


def _effective_transition_distance(
    hit_p: wt.Point3f,
    previous_point: wt.Point3f,
    next_point: wt.Point3f,
) -> wt.Float:
    s_prev = dr.norm(hit_p - previous_point) + wt.Float(EPS)
    s_next = dr.norm(next_point - hit_p) + wt.Float(EPS)
    return (s_prev * s_next) / (s_prev + s_next + wt.Float(EPS))


def reflection_transition_weights(
    *,
    hit_p: wt.Point3f,
    previous_point: wt.Point3f,
    next_point: wt.Point3f,
    primary_plane_point: wt.Point3f,
    primary_plane_normal: wt.Vector3f,
    edge_support: ReflectionBoundarySupport,
    wave_k,
    primary_side_mask,
) -> ReflectionTransitionWeights:
    """Compute primary and adjacent F-weights for one reflection boundary.

    ``primary_side_mask`` is the explicit side gate: true means the current
    hit is still on the primary surface side of the boundary, false means it
    has crossed into the adjacent/free-space side. The helper uses
    ``edge_support.valid`` as the fast-path switch; invalid support returns the
    interior hard-mode primary weight ``1 + 0j`` and no adjacent residual.
    """
    del primary_plane_point
    width = int(dr.width(hit_p.x))
    zero = _zero_complex(width)
    one = _one_complex(width)
    primary_side = wt.Bool(primary_side_mask)

    effective_distance = _effective_transition_distance(hit_p, previous_point, next_point)
    transition_x = wt.Float(wave_k) * edge_support.distance * edge_support.distance / (
        effective_distance + wt.Float(EPS)
    )
    transition = _safe_transition(transition_x)

    primary_near = _select_complex(primary_side, transition, zero)
    primary_far = _select_complex(primary_side, one, zero)
    primary_weight = _select_complex(edge_support.valid, primary_near, primary_far)

    primary_n0 = dr.abs(dr.dot(edge_support.n0, primary_plane_normal))
    primary_nn = dr.abs(dr.dot(edge_support.n_face_n, primary_plane_normal))
    n0_is_primary = primary_n0 >= primary_nn
    adjacent_normal = _select_vector(n0_is_primary, edge_support.n_face_n, edge_support.n0)
    has_adjacent_face = (edge_support.adjacent_face0 >= wt.Int32(0)) & (
        edge_support.adjacent_face1 >= wt.Int32(0)
    )
    adjacent_valid = edge_support.valid & (~primary_side) & has_adjacent_face

    return ReflectionTransitionWeights(
        primary_weight=primary_weight,
        adjacent_weight=_select_complex(adjacent_valid, transition, zero),
        adjacent_plane_point=_select_point(adjacent_valid, edge_support.edge_pos, _zero_point(width)),
        adjacent_plane_normal=_select_vector(adjacent_valid, adjacent_normal, _zero_vector(width)),
        adjacent_valid=adjacent_valid,
    )


def reflection_segment_attenuation(
    *,
    support: SecondaryVisibilitySupport,
    wave_k,
) -> wt.Complex2f:
    """Return secondary-visibility segment attenuation for one receiver segment."""
    width = int(dr.width(support.gamma))
    zero = _zero_complex(width)
    one = _one_complex(width)
    gamma = dr.select(support.valid, support.gamma, wt.Float(0.0))
    effective_l = dr.select(support.valid, support.effective_L, wt.Float(1.0))
    transition_x = wt.Float(wave_k) * gamma * gamma / (effective_l + wt.Float(EPS))
    transition = _safe_transition(transition_x)
    occluded_weight = _select_complex(support.valid, transition, zero)
    return _select_complex(support.is_occluded, occluded_weight, one)


__all__ = [
    "ReflectionTransitionWeights",
    "reflection_segment_attenuation",
    "reflection_transition_weights",
]
