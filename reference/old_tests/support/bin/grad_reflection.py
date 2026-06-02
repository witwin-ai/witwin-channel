"""Reflection gradients: DrJit AD vs FD."""
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
from witwin.channel import (
    DEFAULT_VARIANT,
    FieldMonitor,
    Tracer,
    plot_field_with_edges,
    plot_gradient_with_edges,
    scalar,
    to_numpy,
    to_numpy_2d,
    to_power_db,
)

# Toggle cube rotation (set to None to disable)
CUBE_ROTATION = np.pi / 5  # or None to disable


def compute_reflection_ad_gradient(center_coords, size, freq, tx_pos, range_x, range_y,
                                    grid_size, n_rays, max_reflections, reflection_coef,
                                    grad_axis=0):
    """
    Compute reflection field and its gradient using DrJit forward AD.

    Args:
        center_coords: tuple (cx, cy, cz) - center coordinates
        grad_axis: 0 for x, 1 for y, 2 for z

    Returns:
        ref_field_db: Reflection field in dB (numpy)
        ref_grad_db: Gradient of reflection field in dB (numpy)
        result: Full trace result dict
    """
    cx, cy, cz = center_coords

    # Set tangent for forward AD
    if grad_axis == 0:
        tangent = wt.Vector3f(1.0, 0.0, 0.0)
    elif grad_axis == 1:
        tangent = wt.Vector3f(0.0, 1.0, 0.0)
    else:
        tangent = wt.Vector3f(0.0, 0.0, 1.0)

    # Helper function to run trace with AD and compute gradient
    # IMPORTANT: forward_to must be called inside the function while center is in scope
    def run_trace_and_get_grad(target='ref_db'):
        # Create fresh center with gradient enabled
        center = wt.Point3f(cx, cy, cz)
        dr.enable_grad(center)
        dr.set_grad(center, tangent)

        # Create mesh and tracer
        scene = build_test_scene(box_drjit_geometry(center=center, size=size, rotation=CUBE_ROTATION))
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

        # Trace
        result = tracer.trace(tx_pos=tx_pos)
        assert_plane_monitor_result(result, monitor)
        payload = result.primary

        # Forward AD must be done while center is in scope
        if target == 'ref_db':
            ref_db = to_power_db(payload.field.reflection)
            dr.forward_to(ref_db, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
            grad = dr.grad(ref_db)
            if grad is None:
                grad = dr.zeros(wt.Float, dr.width(ref_db))
            return result, scene, grad, ref_db
        elif target == 'ref_real':
            dr.forward_to(payload.field.reflection.real, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
            grad = dr.grad(payload.field.reflection.real)
            if grad is None:
                grad = dr.zeros(wt.Float, dr.width(payload.field.reflection.real))
            return result, scene, grad, None
        elif target == 'ref_imag':
            dr.forward_to(payload.field.reflection.imag, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
            grad = dr.grad(payload.field.reflection.imag)
            if grad is None:
                grad = dr.zeros(wt.Float, dr.width(payload.field.reflection.imag))
            return result, scene, grad, None

    # Run trace for dB gradient
    result, scene, ref_grad, ref_db = run_trace_and_get_grad('ref_db')

    # Run trace again for real part gradient
    _, _, ref_real_grad, _ = run_trace_and_get_grad('ref_real')

    # Run trace again for imag part gradient
    _, _, ref_imag_grad, _ = run_trace_and_get_grad('ref_imag')

    # Convert to numpy
    ref_field_db = to_numpy_2d(ref_db, grid_size)
    ref_grad_db = to_numpy_2d(ref_grad, grid_size)
    ref_real_grad_np = to_numpy_2d(ref_real_grad, grid_size)
    ref_imag_grad_np = to_numpy_2d(ref_imag_grad, grid_size)

    return {
        'ref_db': ref_field_db,
        'ref_grad_db': ref_grad_db,
        'ref_real_grad': ref_real_grad_np,
        'ref_imag_grad': ref_imag_grad_np,
        'result': result,
        'scene': scene
    }


def compute_reflection_fd_gradient(center_np, size, freq, tx_pos, range_x, range_y,
                                    grid_size, n_rays, max_reflections, reflection_coef,
                                    grad_axis=0, delta=0.01):
    """
    Compute reflection field gradient using finite difference.

    Args:
        center_np: numpy array [x, y, z]
        grad_axis: 0 for x, 1 for y, 2 for z
        delta: Perturbation size

    Returns:
        ref_field_db: Base reflection field in dB (numpy)
        ref_grad_db: Gradient of reflection field in dB (numpy)
        ref_real_grad: Gradient of real part (numpy)
        ref_imag_grad: Gradient of imag part (numpy)
    """
    def compute_field(c_np):
        center = wt.Point3f(float(c_np[0]), float(c_np[1]), float(c_np[2]))
        scene = build_test_scene(box_drjit_geometry(center=center, size=size, rotation=CUBE_ROTATION))
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

    # Base
    result_base, scene_base = compute_field(center_np)
    ref_db_base = to_numpy_2d(to_power_db(result_base.primary.field.reflection), grid_size)
    ref_real_base = to_numpy_2d(result_base.primary.field.reflection.real, grid_size)
    ref_imag_base = to_numpy_2d(result_base.primary.field.reflection.imag, grid_size)

    # Perturbed
    center_perturbed = center_np.copy()
    center_perturbed[grad_axis] += delta
    result_perturbed, _ = compute_field(center_perturbed)
    ref_db_perturbed = to_numpy_2d(to_power_db(result_perturbed.primary.field.reflection), grid_size)
    ref_real_perturbed = to_numpy_2d(result_perturbed.primary.field.reflection.real, grid_size)
    ref_imag_perturbed = to_numpy_2d(result_perturbed.primary.field.reflection.imag, grid_size)

    # Finite difference
    ref_grad_db = (ref_db_perturbed - ref_db_base) / delta
    ref_real_grad = (ref_real_perturbed - ref_real_base) / delta
    ref_imag_grad = (ref_imag_perturbed - ref_imag_base) / delta

    return {
        'ref_db': ref_db_base,
        'ref_grad_db': ref_grad_db,
        'ref_real_grad': ref_real_grad,
        'ref_imag_grad': ref_imag_grad,
        'result': result_base,
        'scene': scene_base
    }


def main():
    print("=" * 60)
    print("Reflection Gradient Visualization: DrJit AD vs FD")
    print("=" * 60)

    # Parameters
    grid_size = 128
    freq = 1e9
    tx_pos = (-5.0, 5.0, 1.5)
    range_x, range_y = (-8, 8), (-8, 8)
    n_rays = 5000
    max_reflections = 2
    reflection_coef = 0.7

    center_np = np.array([0.0, 0.0, 2.0])
    size = 4.0

    print(f"\nParameters:")
    print(f"  grid_size: {grid_size}")
    print(f"  n_rays: {n_rays}")
    print(f"  max_reflections: {max_reflections}")
    print(f"  frequency: {freq/1e9:.1f} GHz")
    print(f"  center: {center_np.tolist()}")
    print(f"  rotation: {CUBE_ROTATION} rad" if CUBE_ROTATION else "  rotation: None")

    # [1] Compute AD gradient (d/dx)
    print("\n[1] Computing reflection AD gradient (d/d(center_x))...")
    center_tuple = (float(center_np[0]), float(center_np[1]), float(center_np[2]))
    ad_result_x = compute_reflection_ad_gradient(
        center_tuple, size, freq, tx_pos, range_x, range_y,
        grid_size, n_rays, max_reflections, reflection_coef,
        grad_axis=0
    )
    print(f"    sum(d(ref_dB)/d(cx)) AD = {ad_result_x['ref_grad_db'].sum():.4f}")
    print(f"    sum(d(ref_real)/d(cx)) AD = {ad_result_x['ref_real_grad'].sum():.6f}")

    # [2] Compute AD gradient (d/dy)
    print("\n[2] Computing reflection AD gradient (d/d(center_y))...")
    ad_result_y = compute_reflection_ad_gradient(
        center_tuple, size, freq, tx_pos, range_x, range_y,
        grid_size, n_rays, max_reflections, reflection_coef,
        grad_axis=1
    )
    print(f"    sum(d(ref_dB)/d(cy)) AD = {ad_result_y['ref_grad_db'].sum():.4f}")
    print(f"    sum(d(ref_real)/d(cy)) AD = {ad_result_y['ref_real_grad'].sum():.6f}")

    # [3] Compute FD gradient (d/dx)
    print("\n[3] Computing reflection FD gradient (d/d(center_x))...")
    fd_result_x = compute_reflection_fd_gradient(
        center_np, size, freq, tx_pos, range_x, range_y,
        grid_size, n_rays, max_reflections, reflection_coef,
        grad_axis=0, delta=0.01
    )
    print(f"    sum(d(ref_dB)/d(cx)) FD = {fd_result_x['ref_grad_db'].sum():.4f}")
    print(f"    sum(d(ref_real)/d(cx)) FD = {fd_result_x['ref_real_grad'].sum():.6f}")

    # [4] Compute FD gradient (d/dy)
    print("\n[4] Computing reflection FD gradient (d/d(center_y))...")
    fd_result_y = compute_reflection_fd_gradient(
        center_np, size, freq, tx_pos, range_x, range_y,
        grid_size, n_rays, max_reflections, reflection_coef,
        grad_axis=1, delta=0.01
    )
    print(f"    sum(d(ref_dB)/d(cy)) FD = {fd_result_y['ref_grad_db'].sum():.4f}")
    print(f"    sum(d(ref_real)/d(cy)) FD = {fd_result_y['ref_real_grad'].sum():.6f}")

    # Get edges for plotting
    edge_cache = ad_result_x['scene'].get_edge_data(ad_result_x['result'].primary.plane_position)
    edges = edge_cache['edges_2d']

    # Print summary
    print("\n" + "=" * 60)
    print("Gradient Comparison Summary")
    print("=" * 60)
    print("\nd(ref)/d(center_x):")
    print(f"  dB scale - AD: {ad_result_x['ref_grad_db'].sum():.4f}, FD: {fd_result_x['ref_grad_db'].sum():.4f}")
    print(f"  Real part - AD: {ad_result_x['ref_real_grad'].sum():.6f}, FD: {fd_result_x['ref_real_grad'].sum():.6f}")
    print("\nd(ref)/d(center_y):")
    print(f"  dB scale - AD: {ad_result_y['ref_grad_db'].sum():.4f}, FD: {fd_result_y['ref_grad_db'].sum():.4f}")
    print(f"  Real part - AD: {ad_result_y['ref_real_grad'].sum():.6f}, FD: {fd_result_y['ref_real_grad'].sum():.6f}")

    # ==========================================================================
    # Visualization
    # ==========================================================================
    print("\n[5] Generating visualization...")

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))

    # Field colorbar range
    field_vmax = -20
    field_vmin = field_vmax - 40

    # Gradient scale (auto from FD which is more stable for visualization)
    grad_db_scale = max(np.abs(fd_result_x['ref_grad_db']).max(),
                        np.abs(fd_result_y['ref_grad_db']).max())
    grad_db_scale = min(grad_db_scale, 100)  # Cap for visualization
    grad_db_scale= 5

    grad_real_scale = max(np.abs(fd_result_x['ref_real_grad']).max(),
                          np.abs(fd_result_y['ref_real_grad']).max())
    grad_real_scale = min(grad_real_scale, 0.1)  # Cap for visualization

    # Row 1: Reflection Field (dB), d(ref_dB)/d(cx) AD, d(ref_dB)/d(cx) FD, Diff
    im = plot_field_with_edges(axes[0, 0], ad_result_x['ref_db'],
                    'Reflection Field (dB)', edges, tx_pos, range_x, range_y,
                    field_vmin, field_vmax)
    plt.colorbar(im, ax=axes[0, 0], shrink=0.7)

    im = plot_gradient_with_edges(axes[0, 1], ad_result_x['ref_grad_db'],
                       'd(ref_dB)/d(cx) AD', edges, tx_pos, range_x, range_y,
                       -grad_db_scale, grad_db_scale)
    plt.colorbar(im, ax=axes[0, 1], shrink=0.7)

    im = plot_gradient_with_edges(axes[0, 2], fd_result_x['ref_grad_db'],
                       'd(ref_dB)/d(cx) FD', edges, tx_pos, range_x, range_y,
                       -grad_db_scale, grad_db_scale)
    plt.colorbar(im, ax=axes[0, 2], shrink=0.7)

    diff_x_db = ad_result_x['ref_grad_db'] - fd_result_x['ref_grad_db']
    im = plot_gradient_with_edges(axes[0, 3], diff_x_db,
                       'd(ref_dB)/d(cx) (AD - FD)', edges, tx_pos, range_x, range_y)
    plt.colorbar(im, ax=axes[0, 3], shrink=0.7)

    # Row 2: d(ref_dB)/d(cy) - AD, FD, Diff, and per-bounce
    im = plot_gradient_with_edges(axes[1, 0], ad_result_y['ref_grad_db'],
                       'd(ref_dB)/d(cy) AD', edges, tx_pos, range_x, range_y,
                       -grad_db_scale, grad_db_scale)
    plt.colorbar(im, ax=axes[1, 0], shrink=0.7)

    im = plot_gradient_with_edges(axes[1, 1], fd_result_y['ref_grad_db'],
                       'd(ref_dB)/d(cy) FD', edges, tx_pos, range_x, range_y,
                       -grad_db_scale, grad_db_scale)
    plt.colorbar(im, ax=axes[1, 1], shrink=0.7)

    diff_y_db = ad_result_y['ref_grad_db'] - fd_result_y['ref_grad_db']
    im = plot_gradient_with_edges(axes[1, 2], diff_y_db,
                       'd(ref_dB)/d(cy) (AD - FD)', edges, tx_pos, range_x, range_y)
    plt.colorbar(im, ax=axes[1, 2], shrink=0.7)

    # Per-bounce reflection no longer available, show empty
    axes[1, 3].axis('off')

    # Row 3: Complex field gradients (Real part)
    im = plot_gradient_with_edges(axes[2, 0], ad_result_x['ref_real_grad'],
                       'd(ref_real)/d(cx) AD', edges, tx_pos, range_x, range_y,
                       -grad_real_scale, grad_real_scale)
    plt.colorbar(im, ax=axes[2, 0], shrink=0.7)

    im = plot_gradient_with_edges(axes[2, 1], fd_result_x['ref_real_grad'],
                       'd(ref_real)/d(cx) FD', edges, tx_pos, range_x, range_y,
                       -grad_real_scale, grad_real_scale)
    plt.colorbar(im, ax=axes[2, 1], shrink=0.7)

    im = plot_gradient_with_edges(axes[2, 2], ad_result_y['ref_real_grad'],
                       'd(ref_real)/d(cy) AD', edges, tx_pos, range_x, range_y,
                       -grad_real_scale, grad_real_scale)
    plt.colorbar(im, ax=axes[2, 2], shrink=0.7)

    im = plot_gradient_with_edges(axes[2, 3], fd_result_y['ref_real_grad'],
                       'd(ref_real)/d(cy) FD', edges, tx_pos, range_x, range_y,
                       -grad_real_scale, grad_real_scale)
    plt.colorbar(im, ax=axes[2, 3], shrink=0.7)

    fig.suptitle(f'Reflection Gradient Analysis - {freq/1e9:.1f} GHz, {n_rays} rays, {max_reflections} bounces\n'
                 f'Row1: Field & d/dx | Row2: d/dy & Bounce1 | Row3: Complex Real gradients',
                 fontsize=10)
    plt.tight_layout()
    output_path = FIGURES_DIR / "reflection_grad_vis.png"
    plt.savefig(output_path, dpi=150)
    print(f"[OK] Figure saved to {output_path}")

maybe_show()


if __name__ == "__main__":
    main()


