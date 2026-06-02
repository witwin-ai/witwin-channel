"""Line-of-Sight (LoS) tracing for radio propagation simulation"""

import drjit as dr
import rayd
import witwin as wt

from ..utils.constants import RAY_EPS


def los_blocked(scene, tx_pos, rx_positions):
    """
    Detect LoS occlusion via shadow test (faster than full intersection).

    Args:
        scene: Scene object
        tx_pos: Transmitter position - wt.Point3f (gradient-preserving)
        rx_positions: Receiver positions - Point3f array

    Returns:
        blocked: Boolean array (wt.Bool), True means blocked
    """
    if scene is None:
        return dr.zeros(wt.Bool, dr.width(rx_positions.x))
    if len(getattr(scene, "structures", ())) == 0:
        return dr.zeros(wt.Bool, dr.width(rx_positions.x))
    tri_data = getattr(scene, "tri_data_gpu", None)
    if tri_data is not None:
        n_triangles = int(tri_data.get("n_triangles", 0))
        if n_triangles <= 0:
            return dr.zeros(wt.Bool, dr.width(rx_positions.x))
    ray_dir = rx_positions - tx_pos
    ray_length = dr.norm(ray_dir)
    ray_dir_normalized = ray_dir / ray_length
    if hasattr(scene, "ray_test"):
        rays = rayd.Ray(tx_pos, ray_dir_normalized)
        rays.tmax = ray_length - RAY_EPS
        with dr.suspend_grad():
            return scene.ray_test(rays)

    rays = rayd.Ray(tx_pos, ray_dir_normalized)
    with dr.suspend_grad():
        si = scene.ray_intersect(rays)
        return si.is_valid() & (si.t < ray_length - RAY_EPS)


def compute_los_field(scene, rx_positions_or_x, *args):
    """
    Compute LoS field with mesh occlusion using ray tracing.

    Args:
        scene: Scene object
        rx_positions_or_x: Either a flattened ``Point3f`` receiver array or
            the legacy tangential ``X`` coordinate array.
        *args: ``(tx_pos, wavelength, k)`` for the 3D receiver-position form
            or ``(Y, rx_z, tx_pos, wavelength, k)`` for the historical planar
            signature.

    Returns:
        a_los: Complex field amplitude (DrJit Complex2f)
    """
    if len(args) == 3:
        rx_positions = rx_positions_or_x
        tx_pos, wavelength, k = args
    elif len(args) == 5:
        Y, rx_z, tx_pos, wavelength, k = args
        rx_positions = wt.Point3f(rx_positions_or_x, Y, rx_z)
    else:
        raise TypeError(
            "compute_los_field expects either (scene, rx_positions, tx_pos, wavelength, k) "
            "or the legacy planar signature (scene, X, Y, rx_z, tx_pos, wavelength, k)."
        )

    # Use ray tracing to detect occlusion
    los_blocked_mask = los_blocked(scene, tx_pos, rx_positions)

    # Compute LoS field (3D distance) - tx_pos is wt.Point3f
    ray_dir = rx_positions - tx_pos
    d_los = dr.norm(ray_dir)

    los_coeff = wt.Float(wavelength) / (4 * dr.pi * d_los)
    los_phase = wt.Complex2f(0, -wt.Float(k) * d_los)
    a_los = los_coeff * dr.exp(los_phase)
    a_los = dr.select(los_blocked_mask, wt.Complex2f(0, 0), a_los)

    return a_los

