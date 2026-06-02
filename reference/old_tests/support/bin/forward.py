"""Radio propagation simulation using mesh-based UTD"""

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

import numpy as np
import torch
import matplotlib.pyplot as plt
import witwin as wt
import time
from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene

from witwin.channel import (
    DEFAULT_VARIANT,
    FieldMonitor,
    Tracer,
    draw_corners,
    draw_edges_with_normals,
    draw_tx,
    to_numpy,
    to_power_db,
)


def plot_mesh_2d(result, scene, frequency, range_x=(-8, 8), range_y=(-8, 8)):
    """
    Plot 2D field heatmap from mesh with all components.

    Args:
        result: dict returned from Tracer.trace()
        scene: Scene object (for edges/corners)
        frequency: Signal frequency in Hz
        range_x, range_y: Plot bounds

    Returns:
        fig: matplotlib figure
    """
    payload = result.primary
    grid_shape = payload.grid_shape
    calculation_height = payload.plane_position

    # Get edges and corners from scene
    edge_cache = scene.get_edge_data(calculation_height)
    edges = edge_cache['edges_2d']
    corners = edge_cache['corners_2d']
    n_edges = len(edges)
    n_corners = len(corners)

    def reshape_2d(arr):
        return to_numpy(arr).reshape(grid_shape)

    # Compute dB values from complex fields
    total_db = to_power_db(payload.field.total)
    los_db = to_power_db(payload.field.los)
    ref_db = to_power_db(payload.field.reflection)
    dif_db = to_power_db(payload.field.diffraction)

    # Compute ref+dif (no LoS)
    a_ref_dif = wt.Complex2f(
        payload.field.reflection.real + payload.field.diffraction.real,
        payload.field.reflection.imag + payload.field.diffraction.imag
    )
    ref_dif_db = to_power_db(a_ref_dif)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    vmax = -20
    vmin = vmax - 40

    tx_pos = payload.tx_pos

    def plot_field(ax, data, title):
        im = ax.imshow(reshape_2d(data), extent=[range_x[0], range_x[1], range_y[0], range_y[1]],
                       origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
        draw_tx(ax, tx_pos)
        draw_edges_with_normals(ax, edges)
        draw_corners(ax, corners)
        ax.set_xlim(range_x)
        ax.set_ylim(range_y)
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, shrink=0.7)

    # Row 1: Total, LoS, Reflection
    plot_field(axes[0, 0], total_db, 'Total Field (dB)')
    plot_field(axes[0, 1], los_db, 'Direct (LoS) (dB)')
    plot_field(axes[0, 2], ref_db, 'Reflection (dB)')

    # Row 2: Diffraction, Ref+Dif, empty
    plot_field(axes[1, 0], dif_db, 'Diffraction (dB)')
    plot_field(axes[1, 1], ref_dif_db, 'Reflection + Diffraction (dB)')
    axes[1, 2].axis('off')

    fig.suptitle(f'UTD Field Simulation - {frequency/1e9:.1f} GHz\n'
                 f'Height: {calculation_height:.1f}m, '
                 f'Edges: {n_edges}, Corners: {n_corners}', fontsize=14)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # === Mesh-based propagation test ===
    print("=" * 60)
    print("Running mesh-based propagation simulation")
    print("=" * 60)

    # Mesh parameters (torch tensors)
    center = torch.tensor([0.0, 0.0, 2.0], device='cuda')
    size = torch.tensor(4.0, device='cuda')

    # Setup simulation parameters
    freq = 1e9  # 1 GHz
    tx_pos = torch.tensor([-5.0, 5.0, 1.5], device='cuda')
    range_x, range_y = (-8, 8), (-8, 8)
    grid_size = 512
    n_rays = 10000000
    max_reflections = 1
    reflection_coef = 1.0

    # Create mesh
    scene = build_test_scene(box_drjit_geometry(center=center, size=size))
    monitor = FieldMonitor(
        DEFAULT_MONITOR_NAME,
        axis="z",
        position=monitor_height(tx_pos),
        bounds=(range_x, range_y),
        grid_size=grid_size,
    )
    scene.add_monitor(monitor)
    print(f"Mesh: vertices and faces created")

    # Create tracer
    tracer = Tracer(
        frequency=freq,
        scene=scene,
        reflection_n_rays=n_rays,
        reflection_max_bounces=max_reflections,
        reflection_coef=reflection_coef
    )

    start_time = time.time()

    # Compute field distribution
    result = tracer.trace(tx_pos=tx_pos)
    assert_plane_monitor_result(result, monitor)
    end_time = time.time()
    print(f"[OK] Computed field distribution in {end_time - start_time:.5f} seconds")

    # Plot results
    fig = plot_mesh_2d(result, scene, freq, range_x, range_y)
    output_path = FIGURES_DIR / "mesh_2d_detailed.png"
    fig.savefig(output_path, dpi=150)
    print(f"[OK] Figure saved to {output_path}")
    maybe_show()


