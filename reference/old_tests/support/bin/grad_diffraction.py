"""Diffraction gradients: DrJit AD vs FD."""
from pathlib import Path
import sys

try:
    from ._paths import FIGURES_DIR, maybe_show
    from ._monitor import (
        DEFAULT_MONITOR_NAME,
        assert_boundary_point_sampling,
        assert_plane_monitor_result,
        monitor_height,
    )
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _paths import FIGURES_DIR, maybe_show
    from _monitor import (
        DEFAULT_MONITOR_NAME,
        assert_boundary_point_sampling,
        assert_plane_monitor_result,
        monitor_height,
    )

import matplotlib.pyplot as plt
import numpy as np
import witwin as wt
import drjit as dr
from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene
from witwin.channel import (
    DEFAULT_VARIANT,
    FieldMonitor,
    Tracer,
    compute_diffraction_field,
    plot_field_with_edges,
    plot_gradient_with_edges,
    scalar,
    to_numpy,
    to_numpy_2d,
    to_power_db,
)

# Toggle cube rotation (set to None to disable)
CUBE_ROTATION = 0  # np.pi / 5  # or None to disable


def compute_diffraction_with_components(center, size, freq, tx_pos, range_x, range_y,
                                         grid_size):
    """
    Compute diffraction field and components using DrJit.
    Returns dict with dif_total and components (all differentiable).
    """
    c = 3e8
    wavelength = c / freq
    k = 2 * np.pi / wavelength

    scene = build_test_scene(box_drjit_geometry(center=center, size=size, rotation=CUBE_ROTATION))
    tracer = Tracer(
        frequency=freq,
        scene=scene,
        reflection_n_rays=1000,
        reflection_max_bounces=1,
        reflection_coef=1.0
    )

    # Get TX position
    tx_x = scalar(tx_pos.x) if isinstance(tx_pos, wt.Point3f) else float(tx_pos[0])
    tx_y = scalar(tx_pos.y) if isinstance(tx_pos, wt.Point3f) else float(tx_pos[1])
    tx_z = scalar(tx_pos.z) if isinstance(tx_pos, wt.Point3f) else float(tx_pos[2])
    tx_pos_tuple = (tx_x, tx_y, tx_z)
    rx_z = tx_z
    monitor = FieldMonitor(
        DEFAULT_MONITOR_NAME,
        axis="z",
        position=rx_z,
        bounds=(range_x, range_y),
        grid_size=grid_size,
    )

    # Get edge data
    edge_cache = scene.get_edge_data(rx_z)
    edge_data = edge_cache['edge_data']
    diff_points = edge_cache['diffraction_points']

    if len(diff_points) == 0 or edge_data is None:
        n_total = grid_size * grid_size
        return {
            'dif_total': dr.zeros(wt.Float, n_total),
            'dif_real': dr.zeros(wt.Float, n_total),
            'dif_imag': dr.zeros(wt.Float, n_total),
            'direct_db': dr.zeros(wt.Float, n_total),
            'multi_db': dr.zeros(wt.Float, n_total),
            'direct_real': dr.zeros(wt.Float, n_total),
            'direct_imag': dr.zeros(wt.Float, n_total),
            'multi_real': dr.zeros(wt.Float, n_total),
            'multi_imag': dr.zeros(wt.Float, n_total),
            'edges': edge_cache['edges_2d'],
            'corners': edge_cache['corners_2d'],
            'tx_pos_3d': tx_pos_tuple,
        }

    field = monitor.to_field(1.0)
    coords = field.get_coordinates()
    assert_boundary_point_sampling(
        coords["x_coords"],
        coords["y_coords"],
        bounds=monitor.bounds,
        grid_size=monitor.grid_shape,
    )
    X = coords["X"]
    Y = coords["Y"]

    # Ensure tx_pos is wt.Point3f for gradient preservation
    if not isinstance(tx_pos, wt.Point3f):
        tx_pos_3f = wt.Point3f(float(tx_pos[0]), float(tx_pos[1]), float(tx_pos[2]))
    else:
        tx_pos_3f = tx_pos

    # Compute diffraction with components
    dif_result = compute_diffraction_field(
        X, Y, rx_z, tx_pos_3f,
        scene, wavelength, k,
        return_components=True,
        return_per_edge=False
    )

    dif_real, dif_imag, per_edge_list, components = dif_result

    # Compute power in dB
    power = dif_real * dif_real + dif_imag * dif_imag
    log10 = dr.log(wt.Float(10))
    dif_total_db = 10 * dr.log(power + 1e-20) / log10
    direct_db = to_power_db(components['a_direct'])
    multi_db = to_power_db(components['a_multi'])

    return {
        'dif_total': dif_total_db,
        'dif_real': dif_real,
        'dif_imag': dif_imag,
        'direct_db': direct_db,
        'multi_db': multi_db,
        'direct_real': components['a_direct'].real,
        'direct_imag': components['a_direct'].imag,
        'multi_real': components['a_multi'].real,
        'multi_imag': components['a_multi'].imag,
        'edges': edge_cache['edges_2d'],
        'corners': edge_cache['corners_2d'],
        'tx_pos_3d': tx_pos_tuple,
    }


def compute_total_field(center, size, freq, tx_pos, range_x, range_y, grid_size):
    """Compute total field (LoS + Ref + Dif) using DrJit."""
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
        reflection_n_rays=1000,
        reflection_max_bounces=1,
        reflection_coef=1.0
    )
    result = tracer.trace(tx_pos=tx_pos)
    assert_plane_monitor_result(result, monitor)
    return result


def compute_diffraction_ad_gradient(center_coords, size, freq, tx_pos, range_x, range_y,
                                     grid_size, grad_axis=0):
    """
    Compute diffraction field gradient using DrJit forward AD.

    Computes |da/d(x)| = sqrt((d(Re)/d(x))^2 + (d(Im)/d(x))^2)

    Args:
        center_coords: tuple (cx, cy, cz) - center coordinates
        grad_axis: 0 for x, 1 for y, 2 for z

    Returns:
        dict with dif_db, dif_grad_mag (magnitude of complex gradient), component gradients, and result
    """
    cx, cy, cz = center_coords

    # Set tangent for forward AD
    if grad_axis == 0:
        tangent = wt.Vector3f(1.0, 0.0, 0.0)
    elif grad_axis == 1:
        tangent = wt.Vector3f(0.0, 1.0, 0.0)
    else:
        tangent = wt.Vector3f(0.0, 0.0, 1.0)

    def run_trace_and_get_grad(target='dif_real'):
        """Run trace with AD and compute gradient for target."""
        # Create fresh center with gradient enabled
        center = wt.Point3f(cx, cy, cz)
        dr.enable_grad(center)
        dr.set_grad(center, tangent)

        # Compute diffraction
        result = compute_diffraction_with_components(
            center, size, freq, tx_pos, range_x, range_y, grid_size
        )

        # Forward AD with AllowNoGrad flag for components that may not have gradients
        flags = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad

        if target == 'dif_real':
            dr.forward_to(result['dif_real'], flags=flags)
            grad = dr.grad(result['dif_real'])
        elif target == 'dif_imag':
            dr.forward_to(result['dif_imag'], flags=flags)
            grad = dr.grad(result['dif_imag'])
        elif target == 'direct_real':
            dr.forward_to(result['direct_real'], flags=flags)
            grad = dr.grad(result['direct_real'])
        elif target == 'direct_imag':
            dr.forward_to(result['direct_imag'], flags=flags)
            grad = dr.grad(result['direct_imag'])
        elif target == 'multi_real':
            dr.forward_to(result['multi_real'], flags=flags)
            grad = dr.grad(result['multi_real'])
        elif target == 'multi_imag':
            dr.forward_to(result['multi_imag'], flags=flags)
            grad = dr.grad(result['multi_imag'])
        else:
            raise ValueError(f"Unknown target: {target}")

        # Return zeros if no gradient
        if grad is None:
            grad = dr.zeros(wt.Float, dr.width(result['dif_real']))

        return result, to_numpy(grad)

    # Compute gradients of real and imag parts
    result, dif_real_grad = run_trace_and_get_grad('dif_real')
    _, dif_imag_grad = run_trace_and_get_grad('dif_imag')

    # |da/d(x)| = sqrt((d(Re)/d(x))^2 + (d(Im)/d(x))^2)
    dif_grad_mag = np.sqrt(dif_real_grad**2 + dif_imag_grad**2)

    # Direct and mixed component gradients
    _, direct_real_grad = run_trace_and_get_grad('direct_real')
    _, direct_imag_grad = run_trace_and_get_grad('direct_imag')
    _, multi_real_grad = run_trace_and_get_grad('multi_real')
    _, multi_imag_grad = run_trace_and_get_grad('multi_imag')
    direct_grad_mag = np.sqrt(direct_real_grad**2 + direct_imag_grad**2)
    multi_grad_mag = np.sqrt(multi_real_grad**2 + multi_imag_grad**2)

    # Convert to numpy 2D
    return {
        'dif_db': to_numpy_2d(result['dif_total'], grid_size),
        'dif_grad_mag': dif_grad_mag.reshape(grid_size, grid_size),
        'dif_real_grad': dif_real_grad.reshape(grid_size, grid_size),
        'dif_imag_grad': dif_imag_grad.reshape(grid_size, grid_size),
        'direct_db': to_numpy_2d(result['direct_db'], grid_size),
        'multi_db': to_numpy_2d(result['multi_db'], grid_size),
        'direct_grad_mag': direct_grad_mag.reshape(grid_size, grid_size),
        'multi_grad_mag': multi_grad_mag.reshape(grid_size, grid_size),
        'result': result
    }


def compute_diffraction_fd_gradient(center_np, size, freq, tx_pos, range_x, range_y,
                                     grid_size, grad_axis=0, delta=0.01):
    """
    Compute diffraction field gradient using finite difference.

    Computes |da/d(x)| = sqrt((d(Re)/d(x))^2 + (d(Im)/d(x))^2)

    Args:
        center_np: numpy array [x, y, z]
        grad_axis: 0 for x, 1 for y, 2 for z
        delta: Perturbation size
    """
    def compute_field(c_np):
        center = wt.Point3f(float(c_np[0]), float(c_np[1]), float(c_np[2]))
        return compute_diffraction_with_components(
            center, size, freq, tx_pos, range_x, range_y, grid_size
        )

    # Base
    result_base = compute_field(center_np)

    # Perturbed
    center_perturbed = center_np.copy()
    center_perturbed[grad_axis] += delta
    result_perturbed = compute_field(center_perturbed)

    # Compute gradients
    def fd_grad(base, perturbed):
        return (to_numpy(perturbed) - to_numpy(base)) / delta

    # Compute gradients for real and imag parts
    dif_real_base = to_numpy(result_base['dif_real'])
    dif_imag_base = to_numpy(result_base['dif_imag'])
    dif_real_perturbed = to_numpy(result_perturbed['dif_real'])
    dif_imag_perturbed = to_numpy(result_perturbed['dif_imag'])

    dif_real_grad = (dif_real_perturbed - dif_real_base) / delta
    dif_imag_grad = (dif_imag_perturbed - dif_imag_base) / delta

    # |da/d(x)| = sqrt((d(Re)/d(x))^2 + (d(Im)/d(x))^2)
    dif_grad_mag = np.sqrt(dif_real_grad**2 + dif_imag_grad**2)

    direct_real_grad = fd_grad(result_base['direct_real'], result_perturbed['direct_real'])
    direct_imag_grad = fd_grad(result_base['direct_imag'], result_perturbed['direct_imag'])
    multi_real_grad = fd_grad(result_base['multi_real'], result_perturbed['multi_real'])
    multi_imag_grad = fd_grad(result_base['multi_imag'], result_perturbed['multi_imag'])

    return {
        'dif_db': to_numpy_2d(result_base['dif_total'], grid_size),
        'dif_grad_mag': dif_grad_mag.reshape(grid_size, grid_size),
        'dif_real_grad': dif_real_grad.reshape(grid_size, grid_size),
        'dif_imag_grad': dif_imag_grad.reshape(grid_size, grid_size),
        'direct_db': to_numpy_2d(result_base['direct_db'], grid_size),
        'multi_db': to_numpy_2d(result_base['multi_db'], grid_size),
        'direct_grad_mag': np.sqrt(direct_real_grad**2 + direct_imag_grad**2).reshape(grid_size, grid_size),
        'multi_grad_mag': np.sqrt(multi_real_grad**2 + multi_imag_grad**2).reshape(grid_size, grid_size),
        'result': result_base
    }


def main():
    print("=" * 60)
    print("Diffraction Gradient Visualization: DrJit AD vs FD")
    print("=" * 60)

    # Parameters
    grid_size = 256
    freq = 1e9
    tx_pos = (-5.0, 5.0, 1.5)
    range_x, range_y = (-8, 8), (-8, 8)

    center_np = np.array([0.0, 0.0, 2.0])
    size = 4.0

    print(f"\nParameters:")
    print(f"  grid_size: {grid_size}")
    print(f"  frequency: {freq/1e9:.1f} GHz")
    print(f"  center: {center_np.tolist()}")
    print(f"  rotation: {CUBE_ROTATION} rad" if CUBE_ROTATION else "  rotation: None")

    # [1] Compute total field
    print("\n[1] Computing total field...")
    center_pt = wt.Point3f(float(center_np[0]), float(center_np[1]), float(center_np[2]))
    result_total = compute_total_field(
        center_pt, size, freq, tx_pos, range_x, range_y, grid_size
    )
    total_field = to_numpy_2d(to_power_db(result_total.primary.field.total), grid_size)

    # [2] Compute diffraction AD gradient
    print("\n[2] Computing diffraction AD gradient (d/d(center_x))...")
    center_tuple = (float(center_np[0]), float(center_np[1]), float(center_np[2]))
    ad_result = compute_diffraction_ad_gradient(
        center_tuple, size, freq, tx_pos, range_x, range_y, grid_size, grad_axis=0
    )
    print(f"    sum(|da_dif/d(cx)|) AD = {ad_result['dif_grad_mag'].sum():.4f}")

    # [3] Compute diffraction FD gradient
    print("\n[3] Computing diffraction FD gradient (d/d(center_x))...")
    fd_result = compute_diffraction_fd_gradient(
        center_np, size, freq, tx_pos, range_x, range_y, grid_size, grad_axis=0
    )
    print(f"    sum(|da_dif/d(cx)|) FD = {fd_result['dif_grad_mag'].sum():.4f}")

    # Get edges for plotting
    edges = ad_result['result']['edges']

    # Print summary
    print("\n" + "=" * 60)
    print("Gradient Comparison Summary")
    print("=" * 60)
    print(f"\n|da_dif/d(cx)| (Complex Field Gradient Magnitude):")
    print(f"  AD: {ad_result['dif_grad_mag'].sum():.4f}")
    print(f"  FD: {fd_result['dif_grad_mag'].sum():.4f}")
    print(f"\nDirect and mixed diffraction components:")
    print(f"  direct: AD={ad_result['direct_grad_mag'].sum():.4f}, FD={fd_result['direct_grad_mag'].sum():.4f}")
    print(f"  multi:  AD={ad_result['multi_grad_mag'].sum():.4f}, FD={fd_result['multi_grad_mag'].sum():.4f}")

    # ==========================================================================
    # Visualization
    # ==========================================================================
    print("\n[4] Generating visualization...")

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))

    # Field colorbar range
    field_vmax = -20
    field_vmin = field_vmax - 40

    # Convert gradient magnitude to dB for display
    ad_grad_db = 20 * np.log10(ad_result['dif_grad_mag'] + 1e-20)
    fd_grad_db = 20 * np.log10(fd_result['dif_grad_mag'] + 1e-20)
    grad_vmin, grad_vmax = -80, -20  # dB scale for gradient magnitude

    # Row 1: Total Field, Diffraction Field, Dif AD Grad, Dif FD Grad
    im = plot_field_with_edges(axes[0, 0], total_field, 'Total Field (dB)',
                               edges, tx_pos, range_x, range_y, field_vmin, field_vmax)
    plt.colorbar(im, ax=axes[0, 0], shrink=0.6)

    im = plot_field_with_edges(axes[0, 1], ad_result['dif_db'], 'Diffraction Field (dB)',
                               edges, tx_pos, range_x, range_y, field_vmin, field_vmax)
    plt.colorbar(im, ax=axes[0, 1], shrink=0.6)

    im = plot_field_with_edges(axes[0, 2], ad_grad_db, '|da_dif/d(cx)| AD (dB)',
                               edges, tx_pos, range_x, range_y, grad_vmin, grad_vmax)
    plt.colorbar(im, ax=axes[0, 2], shrink=0.6)

    im = plot_field_with_edges(axes[0, 3], fd_grad_db, '|da_dif/d(cx)| FD (dB)',
                               edges, tx_pos, range_x, range_y, grad_vmin, grad_vmax)
    plt.colorbar(im, ax=axes[0, 3], shrink=0.6)

    # Row 2: Direct and mixed field components
    component_fields = [
        (fd_result['direct_db'], 'Direct Diffraction (dB)', 'inferno', field_vmin, field_vmax),
        (fd_result['multi_db'], 'Higher-Order / Mixed Diffraction (dB)', 'inferno', field_vmin, field_vmax),
        (20 * np.log10(ad_result['direct_grad_mag'] + 1e-20), '|d a_direct / d(cx)| AD (dB)', 'RdBu_r', grad_vmin, grad_vmax),
        (20 * np.log10(fd_result['direct_grad_mag'] + 1e-20), '|d a_direct / d(cx)| FD (dB)', 'RdBu_r', grad_vmin, grad_vmax),
    ]

    for i, (field_data, title, cmap, vmin_val, vmax_val) in enumerate(component_fields):
        im = plot_field_with_edges(
            axes[1, i], field_data, title, edges, tx_pos, range_x, range_y, vmin_val, vmax_val, cmap=cmap
        )
        plt.colorbar(im, ax=axes[1, i], shrink=0.6)

    # Row 3: Mixed component gradients and AD/FD residuals
    residual_total = ad_grad_db - fd_grad_db
    residual_direct = 20 * np.log10(ad_result['direct_grad_mag'] + 1e-20) - 20 * np.log10(fd_result['direct_grad_mag'] + 1e-20)
    residual_multi = 20 * np.log10(ad_result['multi_grad_mag'] + 1e-20) - 20 * np.log10(fd_result['multi_grad_mag'] + 1e-20)
    row3_entries = [
        (20 * np.log10(ad_result['multi_grad_mag'] + 1e-20), '|d a_multi / d(cx)| AD (dB)', grad_vmin, grad_vmax),
        (20 * np.log10(fd_result['multi_grad_mag'] + 1e-20), '|d a_multi / d(cx)| FD (dB)', grad_vmin, grad_vmax),
        (residual_total, 'AD-FD Total Gradient (dB)', -20, 20),
        (residual_direct + residual_multi, 'AD-FD Component Residual (dB)', -20, 20),
    ]

    for i, (grad_data, title, vmin_val, vmax_val) in enumerate(row3_entries):
        im = plot_gradient_with_edges(axes[2, i], grad_data, title, edges, tx_pos, range_x, range_y,
                                      vmin_val, vmax_val)
        plt.colorbar(im, ax=axes[2, i], shrink=0.6)

    fig.suptitle(f'Diffraction Gradient Analysis - {freq/1e9:.1f} GHz\n'
                 f'|da/dx| = sqrt((dRe/dx)^2 + (dIm/dx)^2) | Row1: Total | Row2: Direct | Row3: Mixed/Residuals',
                 fontsize=10)
    plt.tight_layout()
    output_path = FIGURES_DIR / "diffraction_grad_vis.png"
    plt.savefig(output_path, dpi=150)
    print(f"[OK] Figure saved to {output_path}")

maybe_show()


if __name__ == "__main__":
    main()


