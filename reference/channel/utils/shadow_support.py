from __future__ import annotations

import drjit as dr

import witwin as wt


def _smoothstep01(x):
    x_clamped = dr.clip(x, wt.Float(0.0), wt.Float(1.0))
    return x_clamped * x_clamped * (wt.Float(3.0) - wt.Float(2.0) * x_clamped)


def shadow_opening_angle_from_wedge_n(wedge_n):
    boundary_eps = wt.Float(1.0e-3)
    return dr.maximum(
        wt.Float(2.0 * dr.pi) - wedge_n * dr.pi,
        wt.Float(2.0) * boundary_eps,
    )


def shadow_decay_span_from_wedge_n(wedge_n):
    boundary_eps = wt.Float(1.0e-3)
    shadow_opening_angle = shadow_opening_angle_from_wedge_n(wedge_n)
    shadow_angle_ratio = dr.minimum(
        shadow_opening_angle / dr.pi,
        wt.Float(1.0),
    )
    shadow_decay_span = dr.maximum(
        (wt.Float(0.17) + wt.Float(0.12) * shadow_angle_ratio) * shadow_opening_angle,
        wt.Float(8.0) * boundary_eps,
    )
    return dr.minimum(
        shadow_decay_span,
        wt.Float(0.5) * shadow_opening_angle,
    )


def shadow_decay_power_from_wedge_n(wedge_n):
    shadow_angle_ratio = dr.minimum(
        shadow_opening_angle_from_wedge_n(wedge_n) / dr.pi,
        wt.Float(1.0),
    )
    return wt.Float(2.0) + (wt.Float(1.0) - shadow_angle_ratio)


def shadow_completion_weight_from_normalized_distance(u, wedge_n):
    shadow_completion_curve = (
        wt.Float(0.88) * (wt.Float(1.0) - _smoothstep01(u))
        + wt.Float(0.12) * (wt.Float(1.0) - dr.clip(u, wt.Float(0.0), wt.Float(1.0)))
    )
    return dr.power(
        shadow_completion_curve,
        shadow_decay_power_from_wedge_n(wedge_n),
    )


def shadow_completion_weight_from_distance(distance, wedge_n):
    shadow_decay_span = shadow_decay_span_from_wedge_n(wedge_n)
    shadow_boundary_u = dr.clip(
        distance / shadow_decay_span,
        wt.Float(0.0),
        wt.Float(1.0),
    )
    return shadow_completion_weight_from_normalized_distance(
        shadow_boundary_u,
        wedge_n,
    )


def shadow_support_amplitude_threshold(shadow_support_cutoff_db):
    if shadow_support_cutoff_db is None:
        return None
    cutoff_db = max(0.0, float(shadow_support_cutoff_db))
    return dr.power(wt.Float(10.0), wt.Float(-0.05 * cutoff_db))


def shadow_support_angle_from_cutoff_db(wedge_n, shadow_support_cutoff_db):
    shadow_decay_span = shadow_decay_span_from_wedge_n(wedge_n)
    amplitude_threshold = shadow_support_amplitude_threshold(shadow_support_cutoff_db)
    if amplitude_threshold is None:
        return shadow_decay_span
    if float(shadow_support_cutoff_db) <= 0.0:
        return dr.zeros(wt.Float, dr.width(wedge_n))

    low = dr.zeros(wt.Float, dr.width(wedge_n))
    high = dr.ones(wt.Float, dr.width(wedge_n))
    for _ in range(12):
        mid = wt.Float(0.5) * (low + high)
        keep = shadow_completion_weight_from_normalized_distance(mid, wedge_n) >= amplitude_threshold
        low = dr.select(keep, mid, low)
        high = dr.select(keep, high, mid)
    return shadow_decay_span * low


__all__ = [
    "shadow_completion_weight_from_distance",
    "shadow_decay_span_from_wedge_n",
    "shadow_support_amplitude_threshold",
    "shadow_support_angle_from_cutoff_db",
]
