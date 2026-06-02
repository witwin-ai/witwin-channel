"""Visualize gradient flow from mesh center to field outputs."""
import matplotlib.pyplot as plt
import numpy as np
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

import witwin as wt
import drjit as dr
from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene
from witwin.channel import (
    DEFAULT_VARIANT,
    FieldMonitor,
    Tracer,
    corner_xy,
    edge_xy,
    to_numpy,
    to_power_db,
)

_FIELD_COMPONENTS = {
    'a_los': 'los',
    'a_ref': 'reflection',
    'a_dif': 'diffraction',
    'a_tot': 'total',
}


def _field_component(payload, field_name):
    try:
        return getattr(payload.field, _FIELD_COMPONENTS[field_name])
    except KeyError as exc:
        raise ValueError(f"Unsupported field component: {field_name}") from exc


def compute_all_fields(center, size, freq, tx_pos, range_x, range_y,
                       grid_size, n_rays, max_reflections, reflection_coef):
    """Compute all fields and return the full result dict.

    Args:
        center: wt.Point3f - cube center position (may have gradient enabled)
        size: float - cube size
        freq: float - frequency in Hz
        tx_pos: tuple - transmitter position (x, y, z)
        range_x, range_y: tuples - grid extent
        grid_size: int - grid resolution
        n_rays: int - number of rays for reflection
        max_reflections: int - max reflection bounces
        reflection_coef: float - reflection coefficient

    Returns:
        dict with field results (DrJit arrays)
    """
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

    trace_result = tracer.trace(tx_pos=tx_pos, verbose=False)
    assert_plane_monitor_result(trace_result, monitor)

    payload = trace_result.primary
    edge_cache = scene.get_edge_data(payload.plane_position)
    return {
        'payload': payload,
        'tracer': tracer,
        'scene': scene,
        'edges': edge_cache['edges_2d'],
        'corners': edge_cache['corners_2d'],
    }


def compute_gradient_drjit_forward_ad(center_vals, size, freq, tx_pos, range_x, range_y,
                                       grid_size, n_rays, max_reflections, reflection_coef,
                                       target='a_dif'):
    """Compute gradient using DrJit forward AD.

    IMPORTANT: dr.forward_to() must be called inside the function where center is in scope.

    Args:
        center_vals: tuple (cx, cy, cz) - center coordinates
        size: float - cube size
        target: str - complex field name ('a_dif', 'a_ref', 'a_tot', 'a_los')
        ...

    Returns:
        output: primal field power (numpy array)
        grad_maps: list of [grad_x, grad_y] numpy arrays
    """
    def compute_with_ad(tangent):
        """Compute field and gradient with AD - forward_to inside to preserve gradient."""
        # Create center with gradient enabled
        center = wt.Point3f(float(center_vals[0]), float(center_vals[1]), float(center_vals[2]))
        dr.enable_grad(center)
        dr.set_grad(center, tangent)

        # Create mesh and tracer
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

        # Trace
        trace_result = tracer.trace(tx_pos=tx_pos, verbose=False)
        assert_plane_monitor_result(trace_result, monitor)
        payload = trace_result.primary

        # Get complex field and compute power (magnitude squared)
        a_field = _field_component(payload, target)
        power = a_field.real * a_field.real + a_field.imag * a_field.imag
        dr.forward_to(power, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
        grad = dr.grad(power)

        return payload, to_numpy(power), to_numpy(grad) if grad is not None else np.zeros_like(to_numpy(power))

    grad_maps = []
    output = None
    result = None

    for axis in range(2):
        axis_name = ['x', 'y'][axis]
        print(f"  Computing DrJit forward AD for center[{axis}] ({axis_name}-axis)...")

        # Set tangent: direction of differentiation
        if axis == 0:
            tangent = wt.Vector3f(1.0, 0.0, 0.0)
        else:
            tangent = wt.Vector3f(0.0, 1.0, 0.0)

        result, primal_np, jvp_np = compute_with_ad(tangent)

        if output is None:
            output = primal_np

        grad_maps.append(jvp_np)
        print(f"    sum(d(|{target}|^2)/d(center_{axis_name})) = {np.sum(jvp_np):.6f}")

    return output, grad_maps, result


def compute_gradient_drjit_complex_mag(center_vals, size, freq, tx_pos, range_x, range_y,
                                        grid_size, n_rays, max_reflections, reflection_coef,
                                        field_name='a_dif'):
    """Compute |d(a)/dx| using DrJit forward AD on complex field.

    IMPORTANT: dr.forward_to() must be called inside the function where center is in scope.
    """
    print(f"  Computing DrJit forward AD for |d({field_name})/d(cx)|...")

    def compute_complex_grad(target_part):
        """Compute gradient of real or imag part."""
        center = wt.Point3f(float(center_vals[0]), float(center_vals[1]), float(center_vals[2]))
        dr.enable_grad(center)
        dr.set_grad(center, wt.Vector3f(1.0, 0.0, 0.0))

        # Create mesh and tracer
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

        trace_result = tracer.trace(tx_pos=tx_pos, verbose=False)
        assert_plane_monitor_result(trace_result, monitor)
        payload = trace_result.primary

        # Get field component and forward AD (must be done while center is in scope)
        a_field = _field_component(payload, field_name)
        if target_part == 'real':
            field = a_field.real
        else:
            field = a_field.imag

        dr.forward_to(field, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
        grad = dr.grad(field)
        return to_numpy(grad) if grad is not None else np.zeros_like(to_numpy(field))

    # Compute gradients for real and imag parts
    jvp_re_np = compute_complex_grad('real')
    jvp_im_np = compute_complex_grad('imag')

    # Compute magnitude: |d(a)/dx| = sqrt((d(Re)/dx)^2 + (d(Im)/dx)^2)
    grad_mag = np.sqrt(jvp_re_np**2 + jvp_im_np**2)

    print(f"    sum(|d({field_name})/d(cx)|) = {np.sum(grad_mag):.6f}")

    return grad_mag


def compute_gradient_finite_diff_all(center_vals, size, freq, tx_pos, range_x, range_y,
                                      grid_size, n_rays, max_reflections, reflection_coef,
                                      delta=0.01):
    """Compute gradient using finite differences for all field types."""
    print("  Computing base field...")

    # Base center (no gradient)
    center_base = wt.Point3f(float(center_vals[0]), float(center_vals[1]), float(center_vals[2]))
    result_base = compute_all_fields(center_base, size, freq, tx_pos, range_x, range_y,
                                      grid_size, n_rays, max_reflections, reflection_coef)

    # Power fields (magnitude squared from complex)
    payload_base = result_base['payload']
    a_los_base = payload_base.field.los
    a_ref_base = payload_base.field.reflection
    a_dif_base = payload_base.field.diffraction
    a_tot_base = payload_base.field.total

    fields_base = {
        'los': to_numpy(a_los_base.real * a_los_base.real + a_los_base.imag * a_los_base.imag).flatten(),
        'ref': to_numpy(a_ref_base.real * a_ref_base.real + a_ref_base.imag * a_ref_base.imag).flatten(),
        'dif': to_numpy(a_dif_base.real * a_dif_base.real + a_dif_base.imag * a_dif_base.imag).flatten(),
        'total': to_numpy(a_tot_base.real * a_tot_base.real + a_tot_base.imag * a_tot_base.imag).flatten()
    }

    # Complex field components
    a_los = payload_base.field.los
    a_ref = payload_base.field.reflection
    a_dif = payload_base.field.diffraction
    a_tot = payload_base.field.total

    complex_base = {
        'los_re': to_numpy(a_los.real).flatten(),
        'ref_re': to_numpy(a_ref.real).flatten(),
        'dif_re': to_numpy(a_dif.real).flatten(),
        'tot_re': to_numpy(a_tot.real).flatten()
    }

    complex_im_base = {
        'los_im': to_numpy(a_los.imag).flatten(),
        'ref_im': to_numpy(a_ref.imag).flatten(),
        'dif_im': to_numpy(a_dif.imag).flatten(),
        'tot_im': to_numpy(a_tot.imag).flatten()
    }

    mag_base = {
        'los_mag': np.abs(to_numpy(a_los.real) + 1j * to_numpy(a_los.imag)).flatten(),
        'ref_mag': np.abs(to_numpy(a_ref.real) + 1j * to_numpy(a_ref.imag)).flatten(),
        'dif_mag': np.abs(to_numpy(a_dif.real) + 1j * to_numpy(a_dif.imag)).flatten(),
        'tot_mag': np.abs(to_numpy(a_tot.real) + 1j * to_numpy(a_tot.imag)).flatten()
    }

    grad_maps = {key: [] for key in fields_base.keys()}
    grad_complex = {key: [] for key in complex_base.keys()}
    grad_complex_im = {key: [] for key in complex_im_base.keys()}
    grad_mag = {key: [] for key in mag_base.keys()}

    for axis in range(2):
        axis_name = ['x', 'y'][axis]
        print(f"  Computing finite diff for center[{axis}] ({axis_name}-axis)...")

        # Perturbed center
        center_list = [float(center_vals[0]), float(center_vals[1]), float(center_vals[2])]
        center_list[axis] += delta
        center_perturbed = wt.Point3f(center_list[0], center_list[1], center_list[2])

        result_perturbed = compute_all_fields(center_perturbed, size, freq, tx_pos,
                                               range_x, range_y, grid_size, n_rays,
                                               max_reflections, reflection_coef)

        payload_perturbed = result_perturbed['payload']
        a_los_p = payload_perturbed.field.los
        a_ref_p = payload_perturbed.field.reflection
        a_dif_p = payload_perturbed.field.diffraction
        a_tot_p = payload_perturbed.field.total

        fields_perturbed = {
            'los': to_numpy(a_los_p.real * a_los_p.real + a_los_p.imag * a_los_p.imag).flatten(),
            'ref': to_numpy(a_ref_p.real * a_ref_p.real + a_ref_p.imag * a_ref_p.imag).flatten(),
            'dif': to_numpy(a_dif_p.real * a_dif_p.real + a_dif_p.imag * a_dif_p.imag).flatten(),
            'total': to_numpy(a_tot_p.real * a_tot_p.real + a_tot_p.imag * a_tot_p.imag).flatten()
        }

        complex_perturbed = {
            'los_re': to_numpy(a_los_p.real).flatten(),
            'ref_re': to_numpy(a_ref_p.real).flatten(),
            'dif_re': to_numpy(a_dif_p.real).flatten(),
            'tot_re': to_numpy(a_tot_p.real).flatten()
        }

        complex_im_perturbed = {
            'los_im': to_numpy(a_los_p.imag).flatten(),
            'ref_im': to_numpy(a_ref_p.imag).flatten(),
            'dif_im': to_numpy(a_dif_p.imag).flatten(),
            'tot_im': to_numpy(a_tot_p.imag).flatten()
        }

        mag_perturbed = {
            'los_mag': np.abs(to_numpy(a_los_p.real) + 1j * to_numpy(a_los_p.imag)).flatten(),
            'ref_mag': np.abs(to_numpy(a_ref_p.real) + 1j * to_numpy(a_ref_p.imag)).flatten(),
            'dif_mag': np.abs(to_numpy(a_dif_p.real) + 1j * to_numpy(a_dif_p.imag)).flatten(),
            'tot_mag': np.abs(to_numpy(a_tot_p.real) + 1j * to_numpy(a_tot_p.imag)).flatten()
        }

        for key in fields_base.keys():
            grad = (fields_perturbed[key] - fields_base[key]) / delta
            grad_maps[key].append(grad)

        for key in complex_base.keys():
            grad = (complex_perturbed[key] - complex_base[key]) / delta
            grad_complex[key].append(grad)

        for key in complex_im_base.keys():
            grad = (complex_im_perturbed[key] - complex_im_base[key]) / delta
            grad_complex_im[key].append(grad)

        for key in mag_base.keys():
            grad = (mag_perturbed[key] - mag_base[key]) / delta
            grad_mag[key].append(grad)

        print(f"    [dB] d(los)/d(c{axis_name})={np.sum(grad_maps['los'][-1]):.2f}, "
              f"d(ref)/d(c{axis_name})={np.sum(grad_maps['ref'][-1]):.2f}, "
              f"d(dif)/d(c{axis_name})={np.sum(grad_maps['dif'][-1]):.2f}, "
              f"d(total)/d(c{axis_name})={np.sum(grad_maps['total'][-1]):.2f}")
        print(f"    [Re] d(los)/d(c{axis_name})={np.sum(grad_complex['los_re'][-1]):.2f}, "
              f"d(ref)/d(c{axis_name})={np.sum(grad_complex['ref_re'][-1]):.2f}, "
              f"d(dif)/d(c{axis_name})={np.sum(grad_complex['dif_re'][-1]):.2f}, "
              f"d(tot)/d(c{axis_name})={np.sum(grad_complex['tot_re'][-1]):.2f}")

    return result_base, grad_maps, grad_complex, grad_complex_im, grad_mag


def plot_field(ax, data, title, result, range_x, range_y, vmin, vmax):
    """Plot field with edges and corners like plot_mesh_2d."""
    payload = result['payload']
    tx_x, tx_y, tx_z = payload.tx_pos
    grid_shape = payload.grid_shape

    # Reshape flat array to 2D
    data_2d = to_numpy(data).reshape(grid_shape)

    im = ax.imshow(data_2d, extent=[range_x[0], range_x[1], range_y[0], range_y[1]],
                   origin='lower', cmap='jet', vmin=vmin, vmax=vmax)

    ax.scatter([tx_x], [tx_y], c='white', s=100, marker='*', edgecolors='black', zorder=5)

    for edge in result['edges']:
        p0_x, p0_y, p1_x, p1_y = edge_xy(edge)
        ax.plot([p0_x, p1_x], [p0_y, p1_y], 'k-', lw=2, alpha=0.7)

    for corner in result['corners']:
        c_x, c_y = corner_xy(corner)
        ax.plot(c_x, c_y, 'ro', markersize=6, alpha=0.7, zorder=7)

    ax.set_xlim(range_x)
    ax.set_ylim(range_y)
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    return im


def plot_gradient(ax, data, title, result, range_x, range_y, vmin, vmax):
    """Plot gradient field with edges."""
    payload = result['payload']
    tx_x, tx_y, tx_z = payload.tx_pos
    grid_shape = payload.grid_shape

    # Reshape flat array to 2D
    data_2d = to_numpy(data).reshape(grid_shape)

    im = ax.imshow(data_2d, extent=[range_x[0], range_x[1], range_y[0], range_y[1]],
                   origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)

    ax.scatter([tx_x], [tx_y], c='black', s=80, marker='*', zorder=5)

    for edge in result['edges']:
        p0_x, p0_y, p1_x, p1_y = edge_xy(edge)
        ax.plot([p0_x, p1_x], [p0_y, p1_y], 'k-', lw=2, alpha=0.7)

    ax.set_xlim(range_x)
    ax.set_ylim(range_y)
    ax.set_title(title, fontsize=9)
    ax.set_aspect('equal')
    return im


def main():
    print("=" * 60)
    print("Gradient Visualization: DrJit Forward AD vs Finite Difference")
    print("=" * 60)

    # Parameters
    grid_size = 512
    freq = 1e9
    tx_pos = (-5.0, 5.0, 1.5)
    range_x, range_y = (-8, 8), (-8, 8)
    n_rays = 10000
    max_reflections = 1
    reflection_coef = 1.0

    # Mesh parameters
    center_vals = (0.0, 0.0, 2.0)
    size = 4.0

    print(f"\nParameters:")
    print(f"  grid_size: {grid_size}")
    print(f"  n_rays: {n_rays}")
    print(f"  frequency: {freq/1e9:.1f} GHz")
    print(f"  center: {center_vals}")

    # Compute gradients using DrJit forward AD - Diffraction
    print("\n[1] Computing gradients via DrJit Forward AD - Diffraction...")
    output_ad_dif, grad_ad_dif, _ = compute_gradient_drjit_forward_ad(
        center_vals, size, freq, tx_pos, range_x, range_y,
        grid_size, n_rays, max_reflections, reflection_coef,
        target='a_dif'
    )

    # Compute gradients using DrJit forward AD - Reflection
    print("\n[1a] Computing gradients via DrJit Forward AD - Reflection...")
    output_ad_ref, grad_ad_ref, _ = compute_gradient_drjit_forward_ad(
        center_vals, size, freq, tx_pos, range_x, range_y,
        grid_size, n_rays, max_reflections, reflection_coef,
        target='a_ref'
    )

    # Compute |d(a_dif)/dx| using DrJit AD
    print("\n[1b] Computing |d(a_dif)/d(cx)| via DrJit Forward AD...")
    grad_ad_complex_mag_dif = compute_gradient_drjit_complex_mag(
        center_vals, size, freq, tx_pos, range_x, range_y,
        grid_size, n_rays, max_reflections, reflection_coef,
        field_name='a_dif'
    )

    # Compute |d(a_ref)/dx| using DrJit AD
    print("\n[1c] Computing |d(a_ref)/d(cx)| via DrJit Forward AD...")
    grad_ad_complex_mag_ref = compute_gradient_drjit_complex_mag(
        center_vals, size, freq, tx_pos, range_x, range_y,
        grid_size, n_rays, max_reflections, reflection_coef,
        field_name='a_ref'
    )

    # Compute |d(a_tot)/dx| using DrJit AD
    print("\n[1d] Computing |d(a_tot)/d(cx)| via DrJit Forward AD...")
    grad_ad_complex_mag_tot = compute_gradient_drjit_complex_mag(
        center_vals, size, freq, tx_pos, range_x, range_y,
        grid_size, n_rays, max_reflections, reflection_coef,
        field_name='a_tot'
    )

    # Compute gradients using finite differences
    print("\n[2] Computing gradients via Finite Differences - All fields...")
    result, grad_fd, grad_complex, grad_complex_im, grad_mag = compute_gradient_finite_diff_all(
        center_vals, size, freq, tx_pos, range_x, range_y,
        grid_size, n_rays, max_reflections, reflection_coef,
        delta=0.01
    )

    # Reshape gradients
    grad_ad_dif_x = grad_ad_dif[0].reshape(grid_size, grid_size)
    grad_ad_dif_y = grad_ad_dif[1].reshape(grid_size, grid_size)
    grad_ad_ref_x = grad_ad_ref[0].reshape(grid_size, grid_size)
    grad_ad_ref_y = grad_ad_ref[1].reshape(grid_size, grid_size)
    grad_ad_complex_mag_dif_x = grad_ad_complex_mag_dif.reshape(grid_size, grid_size)
    grad_ad_complex_mag_ref_x = grad_ad_complex_mag_ref.reshape(grid_size, grid_size)
    grad_ad_complex_mag_tot_x = grad_ad_complex_mag_tot.reshape(grid_size, grid_size)

    grad_fd_all = {}
    for key in ['los', 'ref', 'dif', 'total']:
        grad_fd_all[key] = [
            grad_fd[key][0].reshape(grid_size, grid_size),
            grad_fd[key][1].reshape(grid_size, grid_size)
        ]

    grad_complex_all = {}
    for key in ['los_re', 'ref_re', 'dif_re', 'tot_re']:
        grad_complex_all[key] = [
            grad_complex[key][0].reshape(grid_size, grid_size),
            grad_complex[key][1].reshape(grid_size, grid_size)
        ]

    grad_complex_im_all = {}
    for key in ['los_im', 'ref_im', 'dif_im', 'tot_im']:
        grad_complex_im_all[key] = [
            grad_complex_im[key][0].reshape(grid_size, grid_size),
            grad_complex_im[key][1].reshape(grid_size, grid_size)
        ]

    grad_mag_all = {}
    for key in ['los_mag', 'ref_mag', 'dif_mag', 'tot_mag']:
        grad_mag_all[key] = [
            grad_mag[key][0].reshape(grid_size, grid_size),
            grad_mag[key][1].reshape(grid_size, grid_size)
        ]

    empty_grad = np.zeros((grid_size, grid_size))

    # Compute sum of component gradients
    grad_fd_sum_x = grad_fd_all['los'][0] + grad_fd_all['ref'][0] + grad_fd_all['dif'][0]
    grad_complex_sum_x = grad_complex_all['los_re'][0] + grad_complex_all['ref_re'][0] + grad_complex_all['dif_re'][0]
    grad_mag_sum_x = grad_mag_all['los_mag'][0] + grad_mag_all['ref_mag'][0] + grad_mag_all['dif_mag'][0]

    grad_complex_sum_re_x = grad_complex_all['los_re'][0] + grad_complex_all['ref_re'][0] + grad_complex_all['dif_re'][0]
    grad_complex_sum_im_x = grad_complex_im_all['los_im'][0] + grad_complex_im_all['ref_im'][0] + grad_complex_im_all['dif_im'][0]
    grad_complex_sum_mag_x = np.sqrt(grad_complex_sum_re_x**2 + grad_complex_sum_im_x**2)
    grad_tot_mag_x = np.sqrt(grad_complex_all['tot_re'][0]**2 + grad_complex_im_all['tot_im'][0]**2)

    # Visualization
    print("\n[3] Generating visualization...")

    fig, axes = plt.subplots(5, 5, figsize=(17, 15))

    field_vmax = -20
    field_vmin = field_vmax - 40
    grad_vmin, grad_vmax = -40, 40

    # Compute power dB from complex fields for visualization
    payload = result['payload']
    los_db = to_power_db(payload.field.los)
    ref_db = to_power_db(payload.field.reflection)
    dif_db = to_power_db(payload.field.diffraction)
    tot_db = to_power_db(payload.field.total)

    # Row 1: Field values
    fields = [
        (los_db, 'LoS Field (dB)'),
        (ref_db, 'Reflection Field (dB)'),
        (dif_db, 'Diffraction Field (dB)'),
        (tot_db, 'Total Field (dB)'),
    ]

    for i, (field, title) in enumerate(fields):
        im = plot_field(axes[0, i], field, title, result, range_x, range_y, field_vmin, field_vmax)
        plt.colorbar(im, ax=axes[0, i], shrink=0.6)

    # Row 1 Col 5: AD |d(a_tot)/dx| in dB (total field gradient)
    mag_db_vmin, mag_db_vmax = -80, -20
    tx_x, tx_y, tx_z = payload.tx_pos
    grad_ad_mag_tot_db = 20 * np.log10(grad_ad_complex_mag_tot_x + 1e-20)
    im = axes[0, 4].imshow(grad_ad_mag_tot_db, extent=[range_x[0], range_x[1], range_y[0], range_y[1]],
                           origin='lower', cmap='RdBu_r', vmin=mag_db_vmin, vmax=mag_db_vmax)
    axes[0, 4].scatter([tx_x], [tx_y], c='white', s=80, marker='*', edgecolors='black', zorder=5)
    for edge in result['edges']:
        p0_x, p0_y, p1_x, p1_y = edge_xy(edge)
        axes[0, 4].plot([p0_x, p1_x], [p0_y, p1_y], 'k-', lw=2, alpha=0.7)
    axes[0, 4].set_xlim(range_x)
    axes[0, 4].set_ylim(range_y)
    axes[0, 4].set_title('Automatic Differentiation\n|da_tot/dx| dB', fontsize=9)
    axes[0, 4].set_aspect('equal')
    plt.colorbar(im, ax=axes[0, 4], shrink=0.6)

    # Row 2: DrJit AD gradients (now includes Reflection)
    # Sum of AD gradients (Ref + Dif, LoS not differentiable through mesh)
    grad_ad_sum_x = grad_ad_ref_x + grad_ad_dif_x
    autograd_data = [
        (empty_grad, 'AD: d(LoS)/d(cx)\n(N/A)', True),
        (grad_ad_ref_x, 'AD: d(Ref)/d(cx)', False),
        (grad_ad_dif_x, 'AD: d(Dif)/d(cx)', False),
        (grad_ad_sum_x, 'AD: d(Ref+Dif)/d(cx)', False),
        (grad_ad_sum_x, 'AD: Sum(R+D)', False)
    ]

    scales = [(-1, 1), (-1, 1), (-40, 40), (-40, 40), (-40, 40)]
    for i, (grad, title, is_empty) in enumerate(autograd_data):
        if is_empty:
            axes[1, i].text(0.5, 0.5, 'N/A',
                           transform=axes[1, i].transAxes,
                           ha='center', va='center', fontsize=10, color='gray')
            axes[1, i].set_facecolor('#f0f0f0')
            for edge in result['edges']:
                p0_x, p0_y, p1_x, p1_y = edge_xy(edge)
                axes[1, i].plot([p0_x, p1_x], [p0_y, p1_y], 'k-', lw=1.5, alpha=0.5)
            axes[1, i].set_xlim(range_x)
            axes[1, i].set_ylim(range_y)
            axes[1, i].set_aspect('equal')
            axes[1, i].set_title(title, fontsize=9)
        else:
            im = plot_gradient(axes[1, i], grad, title, result, range_x, range_y, scales[i][0], scales[i][1])
            plt.colorbar(im, ax=axes[1, i], shrink=0.6)

    # Row 3: Finite difference gradients (dB scale)
    fd_data = [
        (grad_fd_all['los'][0], 'FD dB: d(LoS)/d(cx)', (-1, 1)),
        (grad_fd_all['ref'][0], 'FD dB: d(Ref)/d(cx)', (-1, 1)),
        (grad_fd_all['dif'][0], 'FD dB: d(Dif)/d(cx)', (-40, 40)),
        (grad_fd_all['total'][0], 'FD dB: d(Total)/d(cx)', (-40, 40)),
        (grad_fd_sum_x, 'FD dB: Sum(L+R+D)', (-40, 40))
    ]

    for i, (grad, title, scale) in enumerate(fd_data):
        im = plot_gradient(axes[2, i], grad, title, result, range_x, range_y, scale[0], scale[1])
        plt.colorbar(im, ax=axes[2, i], shrink=0.6)

    # Row 4: Finite difference gradients (complex Re)
    complex_scale = 0.05
    fd_complex_data = [
        (grad_complex_all['los_re'][0], 'FD Re: d(LoS)/d(cx)', (-complex_scale, complex_scale)),
        (grad_complex_all['ref_re'][0], 'FD Re: d(Ref)/d(cx)', (-complex_scale, complex_scale)),
        (grad_complex_all['dif_re'][0], 'FD Re: d(Dif)/d(cx)', (-complex_scale, complex_scale)),
        (grad_complex_all['tot_re'][0], 'FD Re: d(Tot)/d(cx)', (-complex_scale, complex_scale)),
        (grad_complex_sum_x, 'FD Re: Sum(L+R+D)', (-complex_scale, complex_scale))
    ]

    for i, (grad, title, scale) in enumerate(fd_complex_data):
        im = plot_gradient(axes[3, i], grad, title, result, range_x, range_y, scale[0], scale[1])
        plt.colorbar(im, ax=axes[3, i], shrink=0.6)

    # Row 5: |d(a)/dx| gradients
    los_grad_mag = np.sqrt(grad_complex_all['los_re'][0]**2 + grad_complex_im_all['los_im'][0]**2)
    ref_grad_mag = np.sqrt(grad_complex_all['ref_re'][0]**2 + grad_complex_im_all['ref_im'][0]**2)
    dif_grad_mag = np.sqrt(grad_complex_all['dif_re'][0]**2 + grad_complex_im_all['dif_im'][0]**2)

    fd_mag_data = [
        (los_grad_mag, 'FD: |da_los/dx| dB'),
        (ref_grad_mag, 'FD: |da_ref/dx| dB'),
        (dif_grad_mag, 'FD: |da_dif/dx| dB'),
        (grad_tot_mag_x, 'FD: |da_tot/dx| dB'),
        (grad_complex_sum_mag_x, 'FD: |Sum(da/dx)| dB')
    ]

    for i, (grad, title) in enumerate(fd_mag_data):
        grad_db = 20 * np.log10(grad + 1e-20)
        im = axes[4, i].imshow(grad_db, extent=[range_x[0], range_x[1], range_y[0], range_y[1]],
                               origin='lower', cmap='RdBu_r', vmin=mag_db_vmin, vmax=mag_db_vmax)
        axes[4, i].scatter([tx_x], [tx_y], c='white', s=80, marker='*', edgecolors='black', zorder=5)
        for edge in result['edges']:
            p0_x, p0_y, p1_x, p1_y = edge_xy(edge)
            axes[4, i].plot([p0_x, p1_x], [p0_y, p1_y], 'k-', lw=2, alpha=0.7)
        axes[4, i].set_xlim(range_x)
        axes[4, i].set_ylim(range_y)
        axes[4, i].set_title(title, fontsize=9)
        axes[4, i].set_aspect('equal')
        plt.colorbar(im, ax=axes[4, i], shrink=0.6)

    fig.suptitle(f'Gradient Analysis - {freq/1e9:.1f} GHz\n'
                 f'Row1: Fields | Row2: AD (Dif) | Row3: FD (dB) | Row4: FD (Re) | Row5: FD (|da/dx|)', fontsize=11)
    plt.tight_layout()
    output_path = FIGURES_DIR / "gradient_map.png"
    plt.savefig(output_path, dpi=150)
    print(f"[OK] Figure saved to {output_path}")

    # Print comparison summary
    print("\n" + "=" * 60)
    print("Gradient Comparison Summary")
    print("=" * 60)

    print("\n[DrJit AD vs FD - Diffraction]")
    print(f"  AD  d(dif_dB)/d(cx): {np.sum(grad_ad_dif[0]):.2f}")
    print(f"  FD  d(dif_dB)/d(cx): {np.sum(grad_fd['dif'][0]):.2f}")

    print("\n[DrJit AD vs FD - Reflection]")
    print(f"  AD  d(ref_dB)/d(cx): {np.sum(grad_ad_ref[0]):.2f}")
    print(f"  FD  d(ref_dB)/d(cx): {np.sum(grad_fd['ref'][0]):.2f}")

    print("\n[Power dB scale - NOT additive due to coherent sum + log]")
    print(f"  FD dB: d(los)/d(cx) = {np.sum(grad_fd['los'][0]):.4f}")
    print(f"  FD dB: d(ref)/d(cx) = {np.sum(grad_fd['ref'][0]):.4f}")
    print(f"  FD dB: d(dif)/d(cx) = {np.sum(grad_fd['dif'][0]):.4f}")
    print(f"  FD dB: Sum(L+R+D)   = {np.sum(grad_fd_sum_x):.4f}")
    print(f"  FD dB: d(total)/d(cx) = {np.sum(grad_fd_all['total'][0]):.4f}")
    print(f"  Difference: {(np.sum(grad_fd_sum_x) - np.sum(grad_fd_all['total'][0])):.4f}")

    print("\n[Complex Re scale - SHOULD be additive: a_tot = a_los + a_ref + a_dif]")
    print(f"  FD Re: d(los)/d(cx) = {np.sum(grad_complex['los_re'][0]):.6f}")
    print(f"  FD Re: d(ref)/d(cx) = {np.sum(grad_complex['ref_re'][0]):.6f}")
    print(f"  FD Re: d(dif)/d(cx) = {np.sum(grad_complex['dif_re'][0]):.6f}")
    print(f"  FD Re: Sum(L+R+D)   = {np.sum(grad_complex_sum_x):.6f}")
    print(f"  FD Re: d(tot)/d(cx) = {np.sum(grad_complex['tot_re'][0]):.6f}")
    print(f"  Difference: {(np.sum(grad_complex_sum_x) - np.sum(grad_complex_all['tot_re'][0])):.6f}")

    print("\n[|da/dx| - SHOULD be additive when summing complex gradients first]")
    print(f"  FD: |da_los/dx| sum = {np.sum(los_grad_mag):.6f}")
    print(f"  FD: |da_ref/dx| sum = {np.sum(ref_grad_mag):.6f}")
    print(f"  FD: |da_dif/dx| sum = {np.sum(dif_grad_mag):.6f}")
    print(f"  FD: |da_tot/dx| sum = {np.sum(grad_tot_mag_x):.6f}")
    print(f"  FD: |Sum(da/dx)| sum = {np.sum(grad_complex_sum_mag_x):.6f}")
    print(f"  Difference (|da_tot/dx| vs |Sum(da/dx)|): {(np.sum(grad_tot_mag_x) - np.sum(grad_complex_sum_mag_x)):.6f}")

maybe_show()


if __name__ == "__main__":
    main()


