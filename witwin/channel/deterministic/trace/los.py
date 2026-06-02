from __future__ import annotations

import drjit as dr
import rayd
from witwin.channel.deterministic import types as wt

from witwin.channel.core.numerics.constants import RAY_EPS
from witwin.channel.core.physics.polarization import (
    project_real_polarization_to_ray,
    scalarize_vector_to_polarization,
    vector_eval,
    vector_from_scalar,
)


def _los_blocked_field(*, scene, runtime):
    tx = runtime.tx
    wave = runtime.wave
    ray_dir = runtime.rx.positions - tx.position
    distance = dr.norm(ray_dir)
    ray_dir_normalized = ray_dir / distance
    rays = rayd.RayAD(tx.position, ray_dir_normalized)
    rays.tmax = distance - RAY_EPS
    with dr.suspend_grad():
        blocked = scene.ray_test(rays)
    field = (
        wave.wavelength
        / (4 * dr.pi * distance)
        * dr.exp(wt.Complex2f(0, -wave.k * distance))
    )
    return dr.select(blocked, wt.Complex2f(0, 0), field), ray_dir, ray_dir_normalized


def trace(*, scene, runtime):
    """Line-of-sight scalar field, shadow-tested against the scene."""
    field, ray_dir, _ = _los_blocked_field(scene=scene, runtime=runtime)
    tx_pol_dir = project_real_polarization_to_ray(runtime.tx.polarization, ray_dir)
    field_vec = vector_from_scalar(field, tx_pol_dir)
    return scalarize_vector_to_polarization(
        field_vec, ray_dir, runtime.rx.effective_polarization(runtime.tx),
    )


def trace_vector(*, scene, runtime):
    """Line-of-sight world-vector field, shadow-tested against the scene."""
    field, _, ray_dir_normalized = _los_blocked_field(scene=scene, runtime=runtime)
    tx_pol_dir = project_real_polarization_to_ray(runtime.tx.polarization, ray_dir_normalized)
    return vector_eval(vector_from_scalar(field, tx_pol_dir))


__all__ = ["trace", "trace_vector"]
