"""Rotation gradient demo (AD vs FD)."""
from pathlib import Path
import sys

try:
    from ._paths import FIGURES_DIR, maybe_show
    from ._monitor import DEFAULT_MONITOR_NAME, assert_plane_monitor_result, monitor_height
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _paths import FIGURES_DIR, maybe_show
    from _monitor import DEFAULT_MONITOR_NAME, assert_plane_monitor_result, monitor_height

import matplotlib.pyplot as plt
import numpy as np
import witwin as wt
import drjit as dr
from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene
from witwin.channel import DEFAULT_VARIANT, FieldMonitor, Scene, Tracer, draw_scene, to_numpy, to_power_db

_FIELD_COMPONENTS = {
    'a_tot': 'total',
    'a_ref': 'reflection',
}


def compute_field(rotation, center, size, freq, tx_pos, range_x, range_y, grid_size,
                  n_rays=1000000, max_reflections=1, reflection_coef=1.0):
    """Compute total field with given rotation (wt.Float)."""
    scene = build_test_scene(box_drjit_geometry(center=wt.Point3f(*center), size=size, rotation=rotation))
    monitor = FieldMonitor(
        DEFAULT_MONITOR_NAME,
        axis="z",
        position=monitor_height(tx_pos),
        bounds=(range_x, range_y),
        grid_size=grid_size,
    )
    scene.add_monitor(monitor)
    tracer = Tracer(
        frequency=freq,
        scene=scene,
        reflection_n_rays=n_rays,
        reflection_max_bounces=max_reflections,
        reflection_coef=reflection_coef
    )
    result = tracer.trace(tx_pos=tx_pos)
    assert_plane_monitor_result(result, monitor)
    return result, scene


def compute_ad_gradient_rotation(rotation_val, center, size, freq, tx_pos, range_x, range_y, grid_size,
                                  n_rays=1000000, max_reflections=1, reflection_coef=1.0,
                                  target_field='a_tot'):
    """Compute |da/d(theta)| using DrJit forward AD.

    Args:
        target_field: 'a_tot' for total field, 'a_ref' for reflection field
    """

    def compute_component_grad(target_part):
        """Compute gradient of real or imag part w.r.t. rotation."""
        rotation = wt.Float(float(rotation_val))
        dr.enable_grad(rotation)
        dr.set_grad(rotation, wt.Float(1.0))  # d/d(theta)

        result, scene = compute_field(
            rotation, center, size, freq, tx_pos, range_x, range_y, grid_size,
            n_rays, max_reflections, reflection_coef
        )

        a = getattr(result.primary.field, _FIELD_COMPONENTS[target_field])
        field = a.real if target_part == 'real' else a.imag

        dr.forward_to(field, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
        grad = dr.grad(field)

        return result, scene, to_numpy(grad) if grad is not None else np.zeros(grid_size * grid_size)

    # Compute gradients for real and imag parts
    result, scene, grad_re = compute_component_grad('real')
    _, _, grad_im = compute_component_grad('imag')

    # |da/d(theta)| = sqrt((d(Re)/d(theta))^2 + (d(Im)/d(theta))^2)
    grad_mag = np.sqrt(grad_re**2 + grad_im**2)

    return result, scene, grad_mag


def compute_fd_gradient_rotation(rotation_val, center, size, freq, tx_pos, range_x, range_y, grid_size,
                                  n_rays=1000000, max_reflections=1, reflection_coef=1.0, delta=0.01,
                                  target_field='a_tot'):
    """Compute |da/d(theta)| using finite differences.

    Args:
        target_field: 'a_tot' for total field, 'a_ref' for reflection field
    """

    # Base field
    rotation_base = wt.Float(float(rotation_val))
    result_base, scene = compute_field(
        rotation_base, center, size, freq, tx_pos, range_x, range_y, grid_size,
        n_rays, max_reflections, reflection_coef
    )
    a_base = getattr(result_base.primary.field, _FIELD_COMPONENTS[target_field])
    re_base = to_numpy(a_base.real)
    im_base = to_numpy(a_base.imag)

    # Perturbed field (rotation + delta)
    rotation_perturbed = wt.Float(float(rotation_val) + delta)
    result_perturbed, _ = compute_field(
        rotation_perturbed, center, size, freq, tx_pos, range_x, range_y, grid_size,
        n_rays, max_reflections, reflection_coef
    )
    a_perturbed = getattr(result_perturbed.primary.field, _FIELD_COMPONENTS[target_field])
    re_perturbed = to_numpy(a_perturbed.real)
    im_perturbed = to_numpy(a_perturbed.imag)

    # Compute gradients
    grad_re = (re_perturbed - re_base) / delta
    grad_im = (im_perturbed - im_base) / delta

    # |da/d(theta)| = sqrt((d(Re)/d(theta))^2 + (d(Im)/d(theta))^2)
    grad_mag = np.sqrt(grad_re**2 + grad_im**2)

    return result_base, scene, grad_mag


def main():
    print("=" * 65)
    print("Demo: Rotation Gradient |da_tot/d(theta)| (AD vs FD)")
    print("=" * 65)

    # Parameters
    grid_size = 512
    freq = 1e9
    tx_pos = (-5.0, 5.0, 1.5)
    range_x, range_y = (-8, 8), (-8, 8)
    center = (0.0, 0.0, 2.0)
    size = 4.0
    rotation_val = np.deg2rad(15)
    n_rays = 1000000
    max_reflections = 1
    reflection_coef = 1.0

    print(f"\nParameters:")
    print(f"  Grid: {grid_size}x{grid_size}")
    print(f"  Frequency: {freq/1e9:.1f} GHz")
    print(f"  TX position: {tx_pos}")
    print(f"  Cube center: {center}")
    print(f"  Rotation: {np.degrees(rotation_val):.1f} deg ({rotation_val:.4f} rad)")
    print(f"  Includes: LoS + Reflection + Diffraction")

    # Compute AD gradient: |da_tot/d(theta)|
    print("\n[1] Computing AD gradient |da_tot/d(theta)|...")
    result_ad, scene_ad, grad_ad_mag = compute_ad_gradient_rotation(
        rotation_val, center, size, freq, tx_pos, range_x, range_y, grid_size,
        n_rays, max_reflections, reflection_coef, target_field='a_tot'
    )
    print(f"    sum(|da_tot/d(theta)|) = {np.sum(grad_ad_mag):.2f}")

    # Compute FD gradient: |da_tot/d(theta)|
    print("\n[2] Computing FD gradient |da_tot/d(theta)|...")
    result_fd, scene_fd, grad_fd_mag = compute_fd_gradient_rotation(
        rotation_val, center, size, freq, tx_pos, range_x, range_y, grid_size,
        n_rays, max_reflections, reflection_coef, target_field='a_tot'
    )
    print(f"    sum(|da_tot/d(theta)|) = {np.sum(grad_fd_mag):.2f}")

    # Prepare data for visualization
    field = to_numpy(to_power_db(result_fd.primary.field.total)).reshape(grid_size, grid_size)
    grad_ad_2d = grad_ad_mag.reshape(grid_size, grid_size)
    grad_fd_2d = grad_fd_mag.reshape(grid_size, grid_size)

    # Convert to dB scale
    grad_ad_db = 20 * np.log10(grad_ad_2d + 1e-20)
    grad_fd_db = 20 * np.log10(grad_fd_2d + 1e-20)

    # Get edges from scene
    edge_cache = scene_fd.get_edge_data(result_fd.primary.plane_position)
    edges = edge_cache['edges_2d']

    # Plot
    print("\n[3] Generating figure...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    extent = [range_x[0], range_x[1], range_y[0], range_y[1]]
    field_vmin, field_vmax = -60, -20
    grad_vmin, grad_vmax = -80, 0

    # Plot 1: Total Field (dB)
    im1 = axes[0].imshow(field, extent=extent, origin='lower', cmap='jet', vmin=field_vmin, vmax=field_vmax)
    draw_scene(axes[0], edges, tx_pos, range_x, range_y)
    axes[0].set_title(f'Total Field (dB)\nrotation={np.degrees(rotation_val):.1f} deg', fontsize=11)
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # Plot 2: AD |da_tot/d(theta)| (dB)
    im2 = axes[1].imshow(grad_ad_db, extent=extent, origin='lower', cmap='RdBu_r', vmin=grad_vmin, vmax=grad_vmax)
    draw_scene(axes[1], edges, tx_pos, range_x, range_y)
    axes[1].set_title('Automatic Differentiation\n|da_tot/d(theta)| (dB)', fontsize=11)
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # Plot 3: FD |da_tot/d(theta)| (dB)
    im3 = axes[2].imshow(grad_fd_db, extent=extent, origin='lower', cmap='RdBu_r', vmin=grad_vmin, vmax=grad_vmax)
    draw_scene(axes[2], edges, tx_pos, range_x, range_y)
    axes[2].set_title('Finite Difference (GT)\n|da_tot/d(theta)| (dB)', fontsize=11)
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    fig.suptitle(f'Rotation Gradient (Z-axis) - {freq/1e9:.1f} GHz', fontsize=12)
    plt.tight_layout()
    output_path = FIGURES_DIR / "demo_rotation_grad.png"
    plt.savefig(output_path, dpi=150)
    print(f"[OK] Figure saved to {output_path}")

    # Summary
    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)
    print("\n|da_tot/d(theta)|:")
    print(f"  AD sum: {np.sum(grad_ad_mag):.2f}")
    print(f"  FD sum: {np.sum(grad_fd_mag):.2f}")
    print(f"  Difference: {abs(np.sum(grad_ad_mag) - np.sum(grad_fd_mag)):.2f}")
    if np.sum(grad_fd_mag) > 0:
        print(f"  Relative diff: {abs(np.sum(grad_ad_mag) - np.sum(grad_fd_mag)) / np.sum(grad_fd_mag) * 100:.2f}%")

    # Check if gradients are non-zero
    if np.sum(grad_ad_mag) > 0:
        print("\n[OK] Rotation gradients are working!")
    else:
        print("\n[WARN] AD gradients are zero - check gradient propagation")

    maybe_show()


if __name__ == "__main__":
    main()


