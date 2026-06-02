"""Field + AD/FD gradient demo for position."""
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


def compute_field(center, size, freq, tx_pos, range_x, range_y, grid_size,
                  n_rays=1000000, max_reflections=1, reflection_coef=1.0):
    """Compute total field."""
    scene = build_test_scene(box_drjit_geometry(center=center, size=size))
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


def compute_ad_gradient_complex_mag(center_vals, size, freq, tx_pos, range_x, range_y, grid_size,
                                     n_rays=1000000, max_reflections=1, reflection_coef=1.0):
    """Compute |da_tot/dx| using DrJit forward AD.

    |da/dx| = sqrt((d(Re)/dx)^2 + (d(Im)/dx)^2)
    First compute gradient, then take magnitude.
    """
    def compute_component_grad(target_part):
        """Compute gradient of real or imag part of a_tot."""
        center = wt.Point3f(float(center_vals[0]), float(center_vals[1]), float(center_vals[2]))
        dr.enable_grad(center)
        dr.set_grad(center, wt.Vector3f(1.0, 0.0, 0.0))  # d/dx

        scene = build_test_scene(box_drjit_geometry(center=center, size=size))
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

        # Get real or imag part and forward AD (must be done while center is in scope)
        a_tot = result.primary.field.total
        field = a_tot.real if target_part == 'real' else a_tot.imag

        dr.forward_to(field, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
        grad = dr.grad(field)

        return result, scene, to_numpy(grad) if grad is not None else np.zeros(grid_size * grid_size)

    # Compute gradients for real and imag parts
    result, scene, grad_re = compute_component_grad('real')
    _, _, grad_im = compute_component_grad('imag')

    # |da/dx| = sqrt((d(Re)/dx)^2 + (d(Im)/dx)^2)
    grad_mag = np.sqrt(grad_re**2 + grad_im**2)

    return result, scene, grad_mag


def compute_fd_gradient_complex_mag(center_vals, size, freq, tx_pos, range_x, range_y, grid_size,
                                     n_rays=1000000, max_reflections=1, reflection_coef=1.0, delta=0.01):
    """Compute |da_tot/dx| using finite differences.

    |da/dx| = sqrt((d(Re)/dx)^2 + (d(Im)/dx)^2)
    First compute gradient, then take magnitude.
    """
    # Base field
    center_base = wt.Point3f(float(center_vals[0]), float(center_vals[1]), float(center_vals[2]))
    result_base, scene = compute_field(center_base, size, freq, tx_pos, range_x, range_y, grid_size,
                                n_rays, max_reflections, reflection_coef)

    a_tot_base = result_base.primary.field.total
    re_base = to_numpy(a_tot_base.real)
    im_base = to_numpy(a_tot_base.imag)

    # Perturbed field (x + delta)
    center_perturbed = wt.Point3f(float(center_vals[0]) + delta, float(center_vals[1]), float(center_vals[2]))
    result_perturbed, _ = compute_field(center_perturbed, size, freq, tx_pos, range_x, range_y, grid_size,
                                     n_rays, max_reflections, reflection_coef)

    a_tot_perturbed = result_perturbed.primary.field.total
    re_perturbed = to_numpy(a_tot_perturbed.real)
    im_perturbed = to_numpy(a_tot_perturbed.imag)

    # Compute gradients
    grad_re = (re_perturbed - re_base) / delta
    grad_im = (im_perturbed - im_base) / delta

    # |da/dx| = sqrt((d(Re)/dx)^2 + (d(Im)/dx)^2)
    grad_mag = np.sqrt(grad_re**2 + grad_im**2)

    return result_base, scene, grad_mag


def main():
    print("=" * 50)
    print("Demo: Field + |da/dx| (AD vs FD)")
    print("=" * 50)

    # Parameters (matching sim.py exactly)
    grid_size = 512
    freq = 1e9
    tx_pos = (-5.0, 5.0, 1.5)
    range_x, range_y = (-8, 8), (-8, 8)
    center_vals = (0.0, 0.0, 2.0)
    size = 4.0
    n_rays = 1000000
    max_reflections = 1
    reflection_coef = 1.0

    print(f"\nParameters:")
    print(f"  Grid: {grid_size}x{grid_size}")
    print(f"  Frequency: {freq/1e9:.1f} GHz")
    print(f"  Cube center: {center_vals}")
    print(f"  Reflection rays: {n_rays}")
    print(f"  Max reflections: {max_reflections}")
    print(f"  Reflection coef: {reflection_coef}")

    # Compute AD gradient: |da_tot/dx|
    print("\n[1] Computing AD gradient |da_tot/dx|...")
    result_ad, scene_ad, grad_ad_mag = compute_ad_gradient_complex_mag(
        center_vals, size, freq, tx_pos, range_x, range_y, grid_size,
        n_rays, max_reflections, reflection_coef
    )
    print(f"    sum(|da_tot/dx|) = {np.sum(grad_ad_mag):.2f}")

    # Compute FD gradient: |da_tot/dx|
    print("\n[2] Computing FD gradient |da_tot/dx|...")
    result_fd, scene_fd, grad_fd_mag = compute_fd_gradient_complex_mag(
        center_vals, size, freq, tx_pos, range_x, range_y, grid_size,
        n_rays, max_reflections, reflection_coef
    )
    print(f"    sum(|da_tot/dx|) = {np.sum(grad_fd_mag):.2f}")

    # Get field for visualization
    field = to_numpy(to_power_db(result_fd.primary.field.total)).reshape(grid_size, grid_size)
    grad_ad_2d = grad_ad_mag.reshape(grid_size, grid_size)
    grad_fd_2d = grad_fd_mag.reshape(grid_size, grid_size)

    # Convert to dB scale for visualization
    grad_ad_db = 20 * np.log10(grad_ad_2d + 1e-20)
    grad_fd_db = 20 * np.log10(grad_fd_2d + 1e-20)

    # Get edges from scene
    edge_cache = scene_fd.get_edge_data(result_fd.primary.plane_position)
    edges = edge_cache['edges_2d']

    # Plot
    print("\n[3] Generating figure...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Field visualization settings
    extent = [range_x[0], range_x[1], range_y[0], range_y[1]]
    field_vmin, field_vmax = -60, -20
    grad_vmin, grad_vmax = -80, -20  # dB scale for gradient magnitude

    # Plot 1: Total Field (dB)
    im1 = axes[0].imshow(field, extent=extent, origin='lower', cmap='jet', vmin=field_vmin, vmax=field_vmax)
    draw_scene(axes[0], edges, tx_pos, range_x, range_y)
    axes[0].set_title('Total Field (dB)', fontsize=12)
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # Plot 2: AD |da_tot/dx| (dB)
    im2 = axes[1].imshow(grad_ad_db, extent=extent, origin='lower', cmap='RdBu_r', vmin=grad_vmin, vmax=grad_vmax)
    draw_scene(axes[1], edges, tx_pos, range_x, range_y)
    axes[1].set_title('Automatic Differentiation\n|da_tot/dx| (dB)', fontsize=12)
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # Plot 3: FD |da_tot/dx| (dB)
    im3 = axes[2].imshow(grad_fd_db, extent=extent, origin='lower', cmap='RdBu_r', vmin=grad_vmin, vmax=grad_vmax)
    draw_scene(axes[2], edges, tx_pos, range_x, range_y)
    axes[2].set_title('Finite Difference (GT)\n|da_tot/dx| (dB)', fontsize=12)
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    fig.suptitle(f'UTD Ray Tracing Gradient - {freq/1e9:.1f} GHz\n'
                 f'|da/dx| = sqrt((dRe/dx)^2 + (dIm/dx)^2)', fontsize=12)
    plt.tight_layout()
    output_path = FIGURES_DIR / "demo.png"
    plt.savefig(output_path, dpi=150)
    print(f"[OK] Figure saved to {output_path}")

    # Summary
    print("\n" + "=" * 50)
    print("Summary: |da_tot/dx|")
    print("=" * 50)
    print(f"  AD sum: {np.sum(grad_ad_mag):.2f}")
    print(f"  FD sum: {np.sum(grad_fd_mag):.2f}")
    print(f"  Difference: {abs(np.sum(grad_ad_mag) - np.sum(grad_fd_mag)):.2f}")
    print(f"  Relative diff: {abs(np.sum(grad_ad_mag) - np.sum(grad_fd_mag)) / np.sum(grad_fd_mag) * 100:.2f}%")

    maybe_show()


if __name__ == "__main__":
    main()


