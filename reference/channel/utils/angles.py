from __future__ import annotations

import math

import drjit as dr
import witwin as wt


def spherical_angles(vectors) -> tuple[wt.Float, wt.Float]:
    radius = dr.maximum(dr.norm(vectors), wt.Float(1e-12))
    theta = dr.acos(dr.clip(vectors.z / radius, -1.0, 1.0))
    phi = dr.atan2(vectors.y, vectors.x)
    phi = dr.select(phi < 0.0, phi + wt.Float(2.0 * math.pi), phi)
    return theta, phi


__all__ = ["spherical_angles"]
