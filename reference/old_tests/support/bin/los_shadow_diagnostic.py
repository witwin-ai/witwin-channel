"""Diagnostic: extract 1D cuts across the LoS shadow boundary to characterize the jump."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import drjit as dr
import numpy as np
import witwin as wt

from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene
from witwin.channel import (
    ChannelConfig,
    DEFAULT_VARIANT,
    DiffractionExecutionConfig,
    Material,
    FieldMonitor,
    TraceConfig,
    Tracer,
    to_numpy,
    to_power_db,
)

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_diagnostic():
    grid_size = 512
    freq = 1e9
    range_x = (-8.0, 8.0)
    range_y = (-8.0, 8.0)
    center = wt.Point3f(0.0, 0.0, 2.0)
    size = 4.0
    tx_pos = wt.Point3f(-5.0, 5.0, 1.5)
    rotation = wt.Float(float(np.deg2rad(-5.0)))
    eps_r = 1.0e4

    scene = build_test_scene(
        box_drjit_geometry(center=center, size=size, rotation=rotation),
        material=Material(eps_r=eps_r),
    )
    monitor = FieldMonitor(
        "diag",
        axis="z",
        position=1.5,
        bounds=(range_x, range_y),
        grid_size=grid_size,
    )
    scene.add_monitor(monitor)
    config = ChannelConfig(
        trace=TraceConfig(
            diffraction_execution=DiffractionExecutionConfig(suffix_dda="symbolic"),
        )
    )
    tracer = Tracer(
        frequency=freq,
        scene=scene,
        config=config,
        reflection_n_rays=20_000,
        reflection_max_bounces=1,
        reflection_coef=1.0,
        use_scene_materials_for_reflection=True,
        use_scene_materials_for_diffraction=True,
        enable_rd_diffraction=True,
        max_diffractions=2,
        tx_polarization=(1.0, 0.0, 0.0),
    )
    result = tracer.trace(tx_pos=tx_pos)

    # Extract fields as numpy
    def to_complex(f):
        return np.asarray(to_numpy(f.real), dtype=np.float64) + 1j * np.asarray(to_numpy(f.imag), dtype=np.float64)

    los = to_complex(result.primary.field.los).reshape(grid_size, grid_size)
    ref = to_complex(result.primary.field.reflection).reshape(grid_size, grid_size)
    dif = to_complex(result.primary.field.diffraction).reshape(grid_size, grid_size)
    total = to_complex(result.primary.field.total).reshape(grid_size, grid_size)

    los_power_db = 20 * np.log10(np.abs(los) + 1e-20)
    ref_power_db = 20 * np.log10(np.abs(ref) + 1e-20)
    dif_power_db = 20 * np.log10(np.abs(dif) + 1e-20)
    total_power_db = 20 * np.log10(np.abs(total) + 1e-20)

    # Find LoS shadow boundary: where los goes from nonzero to zero
    los_mask = np.abs(los) > 1e-15
    boundary_mask = np.zeros_like(los_mask, dtype=bool)
    boundary_mask[:, 1:] |= los_mask[:, 1:] != los_mask[:, :-1]
    boundary_mask[:, :-1] |= los_mask[:, 1:] != los_mask[:, :-1]
    boundary_mask[1:, :] |= los_mask[1:, :] != los_mask[:-1, :]
    boundary_mask[:-1, :] |= los_mask[1:, :] != los_mask[:-1, :]

    x = np.linspace(range_x[0], range_x[1], grid_size)
    y = np.linspace(range_y[0], range_y[1], grid_size)

    # Extract several horizontal cuts across the shadow boundary
    # Find rows that cross the shadow boundary
    boundary_rows = np.where(np.any(boundary_mask, axis=1))[0]
    if len(boundary_rows) == 0:
        print("No shadow boundary found!")
        return

    # Pick a few representative cut rows
    cut_indices = boundary_rows[::max(1, len(boundary_rows) // 6)][:6]

    fig, axes = plt.subplots(len(cut_indices) + 2, 4, figsize=(24, 4 * (len(cut_indices) + 2)))

    # Row 0: 2D overview
    extent = [range_x[0], range_x[1], range_y[0], range_y[1]]
    for col, (data, title) in enumerate([
        (total_power_db, "Total (dB)"),
        (los_power_db, "LoS (dB)"),
        (dif_power_db, "Diffraction (dB)"),
        (ref_power_db, "Reflection (dB)"),
    ]):
        ax = axes[0, col]
        im = ax.imshow(data, extent=extent, origin="lower", cmap="inferno", vmin=-80, vmax=-10)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.7)
        # Mark the cut lines
        for ci in cut_indices:
            ax.axhline(y[ci], color="cyan", linewidth=0.5, alpha=0.6)

    # Row 1: Total field power with LoS boundary overlay, and phase
    ax = axes[1, 0]
    im = ax.imshow(total_power_db, extent=extent, origin="lower", cmap="inferno", vmin=-80, vmax=-10)
    ax.set_title("Total (dB) + LoS boundary")
    # Draw LoS boundary contour
    ax.contour(x, y, los_mask.astype(float), levels=[0.5], colors="lime", linewidths=0.8)
    plt.colorbar(im, ax=ax, shrink=0.7)

    # Total field jump magnitude
    ax = axes[1, 1]
    total_db = total_power_db
    h_jump = np.abs(total_db[:, 1:] - total_db[:, :-1])
    v_jump = np.abs(total_db[1:, :] - total_db[:-1, :])
    max_jump = np.zeros_like(total_db)
    max_jump[:, 1:] = np.maximum(max_jump[:, 1:], h_jump)
    max_jump[:, :-1] = np.maximum(max_jump[:, :-1], h_jump)
    max_jump[1:, :] = np.maximum(max_jump[1:, :], v_jump)
    max_jump[:-1, :] = np.maximum(max_jump[:-1, :], v_jump)
    im = ax.imshow(max_jump, extent=extent, origin="lower", cmap="hot", vmin=0, vmax=10)
    ax.set_title("Cell-to-cell dB jump")
    ax.contour(x, y, los_mask.astype(float), levels=[0.5], colors="lime", linewidths=0.8)
    plt.colorbar(im, ax=ax, shrink=0.7)

    # LoS + diffraction compensation check
    ax = axes[1, 2]
    los_plus_dif = los + dif
    compensation_db = 20 * np.log10(np.abs(los_plus_dif) + 1e-20)
    im = ax.imshow(compensation_db, extent=extent, origin="lower", cmap="inferno", vmin=-80, vmax=-10)
    ax.set_title("LoS + Diffraction (dB)")
    ax.contour(x, y, los_mask.astype(float), levels=[0.5], colors="lime", linewidths=0.8)
    plt.colorbar(im, ax=ax, shrink=0.7)

    # Phase of LoS + diffraction near boundary
    ax = axes[1, 3]
    phase_data = np.angle(los_plus_dif)
    im = ax.imshow(phase_data, extent=extent, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax.set_title("Phase(LoS + Dif)")
    ax.contour(x, y, los_mask.astype(float), levels=[0.5], colors="lime", linewidths=0.8)
    plt.colorbar(im, ax=ax, shrink=0.7)

    # Horizontal cuts
    for i, ci in enumerate(cut_indices):
        row_idx = i + 2

        # Cut: amplitude in dB
        ax = axes[row_idx, 0]
        ax.plot(x, 20 * np.log10(np.abs(total[ci, :]) + 1e-20), "k-", label="Total", linewidth=1.2)
        ax.plot(x, 20 * np.log10(np.abs(los[ci, :]) + 1e-20), "b--", label="LoS", linewidth=0.8)
        ax.plot(x, 20 * np.log10(np.abs(dif[ci, :]) + 1e-20), "r--", label="Diffraction", linewidth=0.8)
        ax.plot(x, 20 * np.log10(np.abs(ref[ci, :]) + 1e-20), "g--", label="Reflection", linewidth=0.8)
        ax.set_title(f"Cut y={y[ci]:.2f}: Amplitude (dB)")
        ax.set_ylim(-80, -10)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        # Cut: LoS + diffraction vs total
        ax = axes[row_idx, 1]
        ax.plot(x, 20 * np.log10(np.abs(total[ci, :]) + 1e-20), "k-", label="Total", linewidth=1.2)
        ax.plot(x, 20 * np.log10(np.abs(los[ci, :] + dif[ci, :]) + 1e-20), "m--", label="LoS+Dif", linewidth=1)
        ax.plot(x, 20 * np.log10(np.abs(los[ci, :] + dif[ci, :] + ref[ci, :]) + 1e-20), "c:", label="LoS+Dif+Ref", linewidth=1)
        ax.set_title(f"Cut y={y[ci]:.2f}: Compensation check")
        ax.set_ylim(-80, -10)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        # Cut: real & imag parts near boundary
        ax = axes[row_idx, 2]
        ax.plot(x, np.real(los[ci, :] + dif[ci, :]), "b-", label="Re(LoS+Dif)", linewidth=0.8)
        ax.plot(x, np.imag(los[ci, :] + dif[ci, :]), "r-", label="Im(LoS+Dif)", linewidth=0.8)
        # Mark shadow boundary positions
        boundary_x = x[boundary_mask[ci, :]]
        for bx in boundary_x:
            ax.axvline(bx, color="gray", linewidth=0.5, alpha=0.5, linestyle=":")
        ax.set_title(f"Cut y={y[ci]:.2f}: Re/Im(LoS+Dif)")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        # Cut: diffraction only amplitude and phase
        ax = axes[row_idx, 3]
        ax2 = ax.twinx()
        ax.plot(x, 20 * np.log10(np.abs(dif[ci, :]) + 1e-20), "r-", label="|Dif| (dB)", linewidth=0.8)
        ax2.plot(x, np.angle(dif[ci, :]), "b.", markersize=0.5, alpha=0.3, label="Phase(Dif)")
        ax.set_title(f"Cut y={y[ci]:.2f}: Diffraction detail")
        for bx in boundary_x:
            ax.axvline(bx, color="gray", linewidth=0.5, alpha=0.5, linestyle=":")
        ax.set_ylim(-80, -10)
        ax.legend(fontsize=6, loc="upper left")
        ax2.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "los_shadow_diagnostic.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Quantify the jump at the shadow boundary
    # For each boundary pixel, measure the total field jump
    near_boundary = boundary_mask.copy()
    total_mag = np.abs(total)
    dif_mag = np.abs(dif)

    boundary_total_jumps = []
    boundary_losdif_jumps = []
    for ci in range(1, grid_size - 1):
        for cj in range(1, grid_size - 1):
            if not near_boundary[ci, cj]:
                continue
            neighbors = [total_mag[ci-1, cj], total_mag[ci+1, cj], total_mag[ci, cj-1], total_mag[ci, cj+1]]
            jump = max(abs(total_mag[ci, cj] - n) for n in neighbors)
            boundary_total_jumps.append(jump)

            los_dif = np.abs(los + dif)
            neighbors_ld = [los_dif[ci-1, cj], los_dif[ci+1, cj], los_dif[ci, cj-1], los_dif[ci, cj+1]]
            jump_ld = max(abs(los_dif[ci, cj] - n) for n in neighbors_ld)
            boundary_losdif_jumps.append(jump_ld)

    boundary_total_jumps = np.array(boundary_total_jumps)
    boundary_losdif_jumps = np.array(boundary_losdif_jumps)

    print(f"\nShadow boundary statistics:")
    print(f"  Total field boundary jump:   median={np.median(boundary_total_jumps):.4e}  "
          f"p95={np.percentile(boundary_total_jumps, 95):.4e}  max={np.max(boundary_total_jumps):.4e}")
    print(f"  LoS+Dif boundary jump:       median={np.median(boundary_losdif_jumps):.4e}  "
          f"p95={np.percentile(boundary_losdif_jumps, 95):.4e}  max={np.max(boundary_losdif_jumps):.4e}")

    # Check if LoS+Dif is continuous across boundary (should be if UTD works correctly)
    # Compute ratio of LoS+Dif boundary jump to total field amplitude
    field_at_boundary = total_mag[near_boundary]
    rel_total_jump = boundary_total_jumps / (field_at_boundary[:len(boundary_total_jumps)] + 1e-20)
    rel_losdif_jump = boundary_losdif_jumps / (field_at_boundary[:len(boundary_losdif_jumps)] + 1e-20)

    print(f"  Relative total jump:         median={np.median(rel_total_jump):.4f}  "
          f"p95={np.percentile(rel_total_jump, 95):.4f}")
    print(f"  Relative LoS+Dif jump:       median={np.median(rel_losdif_jump):.4f}  "
          f"p95={np.percentile(rel_losdif_jump, 95):.4f}")


if __name__ == "__main__":
    run_diagnostic()
