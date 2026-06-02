"""ISB geometry diagnostic: compare LoS occlusion boundary vs UTD shadow boundary.

This script checks whether the binary LoS step (from ray-mesh intersection)
aligns with the UTD transition function's compensating singularity at the
Incident Shadow Boundary.

The diagnostic computes, for each receiver on a 1D cross-section:
  1. LoS blocked mask from ray tracing
  2. Per-edge phi angle from UTD geometry
  3. The ISB condition: phi crosses pi (shadow boundary of the direct wave)
  4. Per-edge d2 transition function value (the compensating UTD term)

If the LoS step and UTD ISB are misaligned, this is the primary cause of
the ISB field discontinuity.

Usage:
    python -m tests.support.bin.isb_geometry_diagnostic
    WITWIN_CHANNEL_MAIN_SHOW=1 python -m tests.support.bin.isb_geometry_diagnostic
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import drjit as dr
import numpy as np
import witwin as wt
from witwin.channel import DEFAULT_VARIANT

from tests._scene_helpers import box_drjit_geometry, build_scene
from witwin.channel import Material, FieldMonitor, Tracer
from witwin.channel.trace.los import los_blocked
from witwin.channel.trace.diffraction.geometry import (
    _compute_edge_angles,
)
from witwin.channel.scene.builder import _ensure_edge_runtime

if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"


def _scalar(v):
    """Extract a Python float from a DrJit scalar or 1-element array."""
    arr = np.array(v.numpy()) if hasattr(v, "numpy") else np.asarray(v)
    return float(arr.flat[0])


def run(
    *,
    y_cut: float = -4.0,
    n_points: int = 2048,
    eps_r: float = 1e4,
    freq: float = 1e9,
):
    wavelength = 299792458.0 / freq
    k = 2.0 * np.pi / wavelength
    center = wt.Point3f(0.0, 0.0, 2.0)
    tx_pos = wt.Point3f(-5.0, 5.0, 1.5)
    rotation = wt.Float(float(np.deg2rad(-5.0)))
    rx_z = 1.5

    scene = build_scene(
        box_drjit_geometry(center=center, size=4.0, rotation=rotation),
        material=Material(eps_r=eps_r),
    )

    # Ensure edge runtime is initialized
    _ensure_edge_runtime(scene)

    # Build 1D receiver line at y_cut
    x_coords = np.linspace(-8.0, 8.0, n_points)
    X = wt.Float(x_coords)
    Y = wt.Float(np.full(n_points, y_cut))
    rx_positions = wt.Point3f(X, Y, wt.Float(rx_z))

    # --- Step 1: LoS blocked mask from ray tracing ---
    los_mask = los_blocked(scene, tx_pos, rx_positions)
    los_blocked_np = np.array(los_mask.numpy()).astype(bool)

    # --- Step 2: Per-edge UTD geometry ---
    gpu = scene._diffraction_edge_gpu
    if gpu is None:
        print("No diffraction edges found!")
        return
    n_edges = gpu["n_edges"]
    print(f"Scene has {n_edges} diffraction edges")

    edge_pos_np = np.stack([
        np.array(gpu["pos"].x.numpy()),
        np.array(gpu["pos"].y.numpy()),
        np.array(gpu["pos"].z.numpy()),
    ], axis=-1)
    edge_dir_np = np.stack([
        np.array(gpu["edge_dir"].x.numpy()),
        np.array(gpu["edge_dir"].y.numpy()),
        np.array(gpu["edge_dir"].z.numpy()),
    ], axis=-1)
    n0_np = np.stack([
        np.array(gpu["n0"].x.numpy()),
        np.array(gpu["n0"].y.numpy()),
        np.array(gpu["n0"].z.numpy()),
    ], axis=-1)
    wedge_n_np = np.array(gpu["wedge_n"].numpy())

    tx_x, tx_y, tx_z = _scalar(tx_pos.x), _scalar(tx_pos.y), _scalar(tx_pos.z)

    # For each edge, compute phi at every receiver point
    edge_phi_arrays = []
    edge_phi_prime_arrays = []

    for ei in range(n_edges):
        src = wt.Point3f(
            dr.full(wt.Float, tx_x, n_points),
            dr.full(wt.Float, tx_y, n_points),
            dr.full(wt.Float, tx_z, n_points),
        )
        ep_b = wt.Point3f(
            dr.full(wt.Float, edge_pos_np[ei, 0], n_points),
            dr.full(wt.Float, edge_pos_np[ei, 1], n_points),
            dr.full(wt.Float, edge_pos_np[ei, 2], n_points),
        )
        ed_b = wt.Vector3f(
            dr.full(wt.Float, edge_dir_np[ei, 0], n_points),
            dr.full(wt.Float, edge_dir_np[ei, 1], n_points),
            dr.full(wt.Float, edge_dir_np[ei, 2], n_points),
        )
        n0_b = wt.Vector3f(
            dr.full(wt.Float, n0_np[ei, 0], n_points),
            dr.full(wt.Float, n0_np[ei, 1], n_points),
            dr.full(wt.Float, n0_np[ei, 2], n_points),
        )

        phi, phi_prime, s_proj, s_prime_proj = _compute_edge_angles(
            src, ep_b, ed_b, n0_b, rx_positions,
        )
        edge_phi_arrays.append(np.array(phi.numpy()))
        edge_phi_prime_arrays.append(np.array(phi_prime.numpy()))

    edge_phi_arrays = np.array(edge_phi_arrays)  # (n_edges, n_points)
    edge_phi_prime_arrays = np.array(edge_phi_prime_arrays)
    wedge_n_list = wedge_n_np

    # --- Step 2b: Print all edges ---
    for ei in range(n_edges):
        phi_at_center = edge_phi_arrays[ei, n_points // 2]
        phi_prime_at_center = edge_phi_prime_arrays[ei, n_points // 2]
        print(f"  Edge {ei}: pos=({edge_pos_np[ei,0]:.2f},{edge_pos_np[ei,1]:.2f},{edge_pos_np[ei,2]:.2f})"
              f" dir=({edge_dir_np[ei,0]:.2f},{edge_dir_np[ei,1]:.2f},{edge_dir_np[ei,2]:.2f})"
              f" wedge_n={wedge_n_list[ei]:.3f}"
              f" phi_center={phi_at_center:.3f} phi'={phi_prime_at_center:.3f}")

    # --- Step 3: Summary ---
    # ISB for the direct wave (d2 term) occurs when phi - phi' = pi
    n_blocked = int(los_blocked_np.sum())
    print(f"LoS blocked: {n_blocked}/{n_points} ({100*n_blocked/n_points:.1f}%)")
    los_transitions = []
    for i in range(len(x_coords) - 1):
        if los_blocked_np[i] != los_blocked_np[i + 1]:
            los_transitions.append(i)
    print(f"LoS transitions found: {len(los_transitions)}")
    for t_idx, t_pos in enumerate(los_transitions):
        x_trans = x_coords[t_pos]
        lit_side = "left" if not los_blocked_np[t_pos] else "right"
        print(f"\n  Transition {t_idx}: x={x_trans:.4f} (lit side={lit_side})")

        # Print all edges' phi - phi' at the transition
        print(f"    Per-edge phi-phi' at transition (ISB at phi-phi'=pi={np.pi:.4f}):")
        for ei in range(n_edges):
            diff = edge_phi_arrays[ei, t_pos] - edge_phi_prime_arrays[ei, t_pos]
            deficit = diff - np.pi
            marker = " <-- CLOSEST" if ei == int(np.argmin(np.abs(edge_phi_arrays[:, t_pos] - edge_phi_prime_arrays[:, t_pos] - np.pi))) else ""
            diffracting = " (diffracting)" if wedge_n_list[ei] > 1.01 else " (flat)"
            print(f"      Edge {ei}: phi-phi'={diff:.4f} (deficit={deficit:+.4f})"
                  f" wedge_n={wedge_n_list[ei]:.3f}{diffracting}{marker}")

        # Find best diffracting edge (wedge_n > 1)
        diffracting_mask = wedge_n_list > 1.01
        if not np.any(diffracting_mask):
            print(f"    No diffracting edges!")
            continue

        diffs_at_trans = edge_phi_arrays[:, t_pos] - edge_phi_prime_arrays[:, t_pos]
        diffs_at_trans_masked = np.where(diffracting_mask, np.abs(diffs_at_trans - np.pi), 999.0)
        best_edge = int(np.argmin(diffs_at_trans_masked))
        print(f"    Best diffracting edge: {best_edge} (phi-phi' deficit={diffs_at_trans[best_edge]-np.pi:+.4f})")

        # Search for phi - phi' = pi crossing for this edge
        diff_line = edge_phi_arrays[best_edge] - edge_phi_prime_arrays[best_edge]
        x_utd_cross = None
        search_lo = max(0, t_pos - 200)
        search_hi = min(n_points - 1, t_pos + 200)
        for j in range(search_lo, search_hi):
            if (diff_line[j] - np.pi) * (diff_line[j + 1] - np.pi) < 0:
                frac = (np.pi - diff_line[j]) / (diff_line[j + 1] - diff_line[j])
                x_utd_cross = x_coords[j] + frac * (x_coords[j + 1] - x_coords[j])
                break

        dx = x_coords[1] - x_coords[0]
        if x_utd_cross is not None:
            offset = x_utd_cross - x_trans
            print(f"    UTD ISB (phi-phi'=pi) at x={x_utd_cross:.4f}")
            print(f"    *** OFFSET: {offset:.5f} m = {offset/dx:.1f} grid cells ***")
        else:
            print(f"    UTD ISB crossing: NOT FOUND within å?00 samples")
            print(f"    phi-phi' range: [{diff_line[search_lo]:.4f}, {diff_line[search_hi]:.4f}]")

    # --- Step 4: Plot ---
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

    # Panel 0: LoS blocked mask
    ax = axes[0]
    ax.set_title(f"LoS blocked mask (y={y_cut:.1f})")
    ax.fill_between(x_coords, 0, 1, where=los_blocked_np, color="gray", alpha=0.3, label="LoS blocked")
    ax.fill_between(x_coords, 0, 1, where=~los_blocked_np, color="yellow", alpha=0.2, label="LoS lit")
    # Mark transitions
    for i in range(len(x_coords) - 1):
        if los_blocked_np[i] != los_blocked_np[i + 1]:
            ax.axvline(x_coords[i], color="red", linewidth=1.5, linestyle="-", alpha=0.8)
    ax.set_ylabel("LoS region")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 1: Per-edge (phi - phi') with ISB line at pi
    ax = axes[1]
    ax.set_title("Per-edge (phi - phi')  --  ISB at (phi - phi') = pi")
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_edges, 10)))
    for ei in range(n_edges):
        diff_line = edge_phi_arrays[ei] - edge_phi_prime_arrays[ei]
        wn = wedge_n_list[ei]
        style = "-" if wn > 1.01 else ":"
        label = f"Edge {ei} (wn={wn:.2f})"
        ax.plot(x_coords, diff_line, color=colors[ei % 10], linewidth=0.8 if wn > 1.01 else 0.5,
                alpha=0.8, linestyle=style, label=label)
    ax.axhline(np.pi, color="black", linewidth=1.5, linestyle="--", label="phi-phi' = pi (ISB)")
    for i in range(len(x_coords) - 1):
        if los_blocked_np[i] != los_blocked_np[i + 1]:
            ax.axvline(x_coords[i], color="red", linewidth=1.0, linestyle=":", alpha=0.6,
                       label="LoS step" if i == los_transitions[0] else None)
    ax.set_ylabel("phi - phi' (rad)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: Zoom on ISB region - phi vs pi for the most relevant edge
    # Find which edge has phi crossing pi near the LoS transitions
    los_transitions = []
    for i in range(len(x_coords) - 1):
        if los_blocked_np[i] != los_blocked_np[i + 1]:
            los_transitions.append(i)

    ax = axes[2]
    ax.set_title("ISB alignment: LoS transition vs UTD phi=pi crossing")
    for t_idx, t_pos in enumerate(los_transitions):
        x_trans = x_coords[t_pos]
        # Find which edge has phi closest to pi at this x
        phi_at_trans = edge_phi_arrays[:, t_pos]
        best_edge = np.argmin(np.abs(phi_at_trans - np.pi))
        phi_line = edge_phi_arrays[best_edge]

        # Window around transition
        win = max(0, t_pos - 50), min(n_points, t_pos + 50)
        x_win = x_coords[win[0]:win[1]]
        phi_win = phi_line[win[0]:win[1]]

        ax.plot(x_win, phi_win - np.pi, color=colors[best_edge % 10], linewidth=1.5,
                label=f"Trans {t_idx}: edge {best_edge}, phi-pi")
        ax.axvline(x_trans, color="red", linewidth=1.5, linestyle="-", alpha=0.6,
                   label=f"LoS step x={x_trans:.3f}")

        # Find where phi crosses pi
        for j in range(win[0], min(win[1] - 1, n_points - 1)):
            if (phi_line[j] - np.pi) * (phi_line[j + 1] - np.pi) < 0:
                # Linear interpolation
                frac = (np.pi - phi_line[j]) / (phi_line[j + 1] - phi_line[j])
                x_utd = x_coords[j] + frac * (x_coords[j + 1] - x_coords[j])
                ax.axvline(x_utd, color="blue", linewidth=1.5, linestyle="--", alpha=0.6,
                           label=f"UTD ISB x={x_utd:.3f}")
                offset = x_utd - x_trans
                dx = x_coords[1] - x_coords[0]
                print(f"\nISB transition {t_idx} at x={x_trans:.4f}:")
                print(f"  Best edge: {best_edge}")
                print(f"  LoS step at x = {x_trans:.4f}")
                print(f"  UTD phi=pi at x = {x_utd:.4f}")
                print(f"  Offset = {offset:.4f} m = {offset/dx:.1f} grid cells")
                print(f"  phi' = {edge_phi_prime_arrays[best_edge, t_pos]:.4f} rad")
                print(f"  wedge_n = {wedge_n_list[best_edge]:.4f}")
                break

    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax.set_ylabel("phi - pi (rad)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: Per-edge phi - phi_prime (shadow deficit angle)
    ax = axes[3]
    ax.set_title("phi - phi_prime per edge (ISB at phi - phi_prime = 0 for direct shadow)")
    for ei in range(n_edges):
        diff = edge_phi_arrays[ei] - edge_phi_prime_arrays[ei]
        ax.plot(x_coords, diff, color=colors[ei % 10], linewidth=0.8, alpha=0.8, label=f"Edge {ei}")
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
    for t_pos in los_transitions:
        ax.axvline(x_coords[t_pos], color="red", linewidth=1.0, linestyle=":", alpha=0.6)
    ax.set_ylabel("phi - phi' (rad)")
    ax.set_xlabel("x (m)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "isb_geometry_diagnostic.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    print(f"\nSaved to {out_path}")

    if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") == "1":
        plt.show()


if __name__ == "__main__":
    run()

