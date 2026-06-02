"""Ray direction generation for GPU-accelerated ray tracing."""

import drjit as dr
import witwin as wt

# Golden ratio for Fibonacci lattice
_PHI = (1 + 5 ** 0.5) / 2


def _coerce_direction3f(direction):
    if isinstance(direction, wt.Vector3f):
        return direction
    if isinstance(direction, wt.Point3f):
        return wt.Vector3f(direction.x, direction.y, direction.z)
    return wt.Vector3f(float(direction[0]), float(direction[1]), float(direction[2]))


def _normalize_direction(direction):
    vec = _coerce_direction3f(direction)
    return vec / (dr.norm(vec) + wt.Float(1e-12))


def _orthonormal_basis_from_axis(axis):
    axis_hat = _normalize_direction(axis)
    helper = dr.select(
        dr.abs(axis_hat.z) < wt.Float(0.9),
        wt.Vector3f(0.0, 0.0, 1.0),
        wt.Vector3f(0.0, 1.0, 0.0),
    )
    tangent = dr.cross(helper, axis_hat)
    tangent = tangent / (dr.norm(tangent) + wt.Float(1e-12))
    bitangent = dr.cross(axis_hat, tangent)
    bitangent = bitangent / (dr.norm(bitangent) + wt.Float(1e-12))
    return tangent, bitangent, axis_hat


def _orient_local_cap_directions(local_x, local_y, local_z, axis):
    tangent, bitangent, axis_hat = _orthonormal_basis_from_axis(axis)
    return wt.Vector3f(
        tangent.x * local_x + bitangent.x * local_y + axis_hat.x * local_z,
        tangent.y * local_x + bitangent.y * local_y + axis_hat.y * local_z,
        tangent.z * local_x + bitangent.z * local_y + axis_hat.z * local_z,
    )


def generate_sphere_directions(n_rays: int):
    """
    Generate uniformly distributed directions on a sphere using Fibonacci lattice.
    Pure DrJit implementation (no PyTorch dependency).

    Args:
        n_rays: Number of ray directions to generate

    Returns:
        wt.Vector3f: Unit direction vectors
    """
    indices = dr.arange(wt.Float, n_rays)
    phi = dr.acos(1 - 2 * (indices + 0.5) / n_rays)
    theta = dr.pi * (1 + _PHI) * indices

    sin_phi = dr.sin(phi)
    dx = sin_phi * dr.cos(theta)
    dy = sin_phi * dr.sin(theta)
    dz = dr.cos(phi)

    return wt.Vector3f(dx, dy, dz)


def generate_hemisphere_directions(n_rays: int, normal):
    """Generate uniformly distributed directions over the hemisphere aligned to ``normal``."""
    indices = dr.arange(wt.Float, n_rays)
    z = (indices + 0.5) / n_rays
    theta = 2.0 * dr.pi * indices / _PHI
    radial = dr.sqrt(dr.maximum(wt.Float(0.0), wt.Float(1.0) - z * z))
    local_x = radial * dr.cos(theta)
    local_y = radial * dr.sin(theta)
    return _orient_local_cap_directions(local_x, local_y, z, normal)


def generate_cone_directions(n_rays: int, axis, half_angle):
    """Generate uniformly distributed directions inside a cone around ``axis``."""
    half_angle = wt.Float(float(half_angle))
    max_cos = dr.cos(half_angle)
    indices = dr.arange(wt.Float, n_rays)
    z = wt.Float(1.0) - (indices + 0.5) / n_rays * (wt.Float(1.0) - max_cos)
    theta = 2.0 * dr.pi * indices / _PHI
    radial = dr.sqrt(dr.maximum(wt.Float(0.0), wt.Float(1.0) - z * z))
    local_x = radial * dr.cos(theta)
    local_y = radial * dr.sin(theta)
    return _orient_local_cap_directions(local_x, local_y, z, axis)


def generate_circle_directions(n_rays: int):
    """
    Generate uniformly distributed directions on a circle (2D, z=0).
    Pure DrJit implementation (no PyTorch dependency).

    Args:
        n_rays: Number of ray directions to generate

    Returns:
        wt.Vector3f: Unit direction vectors (z component is always 0)
    """
    # Create angles from 0 to 2*pi (exclusive of endpoint)
    step = 2 * dr.pi / n_rays
    theta = dr.arange(wt.Float, n_rays) * step

    dx = dr.cos(theta)
    dy = dr.sin(theta)
    dz = dr.zeros(wt.Float, n_rays)

    return wt.Vector3f(dx, dy, dz)

