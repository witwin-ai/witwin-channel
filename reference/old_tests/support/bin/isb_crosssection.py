"""Cross-section diagnostic for ISB (Incident Shadow Boundary) analysis.

Plots LoS, diffraction, reflection, and total field magnitudes along a
horizontal line at a fixed y-coordinate, highlighting ISB transitions.

Usage:
    python -m tests.support.bin.isb_crosssection
    WITWIN_CHANNEL_MAIN_SHOW=1 python -m tests.support.bin.isb_crosssection
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
from witwin.channel import (
    ChannelConfig,
    DiffractionExecutionConfig,
    Material,
    FieldMonitor,
    TraceConfig,
    Tracer,
)

if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"


def run(
    *,
    y_cut: float = -4.0,
    grid_size: int = 512,
    eps_r: float = 1e4,
    freq: float = 1e9,
):
    wavelength = 299792458.0 / freq
    center = wt.Point3f(0.0, 0.0, 2.0)
    tx_pos = wt.Point3f(-5.0, 5.0, 1.5)
    rotation = wt.Float(float(np.deg2rad(-5.0)))

    scene = build_scene(
        box_drjit_geometry(center=center, size=4.0, rotation=rotation),
        material=Material(eps_r=eps_r),
    )
    monitor = FieldMonitor(
        "isb_diag",
        axis="z",
        position=1.5,
        bounds=((-8.0, 8.0), (-8.0, 8.0)),
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
        max_diffractions=2,
        tx_polarization=(1.0, 0.0, 0.0),
        use_scene_materials_for_diffraction=True,
    )
    result = tracer.trace(tx_pos=tx_pos)

    n = grid_size
    x = np.linspace(-8, 8, n)
    row = int((y_cut + 8) / 16 * n)
    actual_y = -8 + row * 16 / n
    print(f"y_cut={y_cut} -> row={row}, actual y={actual_y:.3f}")

    def _mag_row(field):
        re = np.array(field.real.numpy()).reshape(n, n)[row]
        im = np.array(field.imag.numpy()).reshape(n, n)[row]
        return np.sqrt(re**2 + im**2)

    los_line = _mag_row(result.primary.field.los)
    dif_line = _mag_row(result.primary.field.diffraction)
    ref_line = _mag_row(result.primary.field.reflection)
    tot_line = _mag_row(result.primary.field.total)

    floor = -100.0
    to_db = lambda a: np.maximum(20.0 * np.log10(a + 1e-20), floor)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    ax.set_title(f"y = {actual_y:.1f}  cross-section  (linear magnitude,  eps_r={eps_r})")
    ax.plot(x, los_line, "b-", label="LoS", alpha=0.8, linewidth=1)
    ax.plot(x, dif_line, "r-", label="Diffraction", alpha=0.8, linewidth=1)
    ax.plot(x, ref_line, "g-", label="Reflection", alpha=0.6, linewidth=1)
    ax.plot(x, tot_line, "k-", label="Total", alpha=0.9, linewidth=1.5)
    ax.plot(x, los_line / 2, "b--", label="LoS/2 (ideal ISB)", alpha=0.4, linewidth=1)
    ax.set_ylabel("|E|")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-8, 8)
    ax.grid(True, alpha=0.3)

    for i in range(len(x) - 1):
        if (los_line[i] < 1e-10) != (los_line[i + 1] < 1e-10):
            ax.axvline(x[i], color="orange", alpha=0.5, linestyle="--", linewidth=0.8)

    ax = axes[1]
    ax.set_title(f"y = {actual_y:.1f}  cross-section  (dB)")
    ax.plot(x, to_db(los_line), "b-", label="LoS", alpha=0.8, linewidth=1)
    ax.plot(x, to_db(dif_line), "r-", label="Diffraction", alpha=0.8, linewidth=1)
    ax.plot(x, to_db(ref_line), "g-", label="Reflection", alpha=0.6, linewidth=1)
    ax.plot(x, to_db(tot_line), "k-", label="Total", alpha=0.9, linewidth=1.5)
    ax.set_ylabel("dB")
    ax.set_xlabel("x (m)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-80, -30)
    ax.grid(True, alpha=0.3)

    for i in range(len(x) - 1):
        if (los_line[i] < 1e-10) != (los_line[i + 1] < 1e-10):
            ax.axvline(x[i], color="orange", alpha=0.5, linestyle="--", linewidth=0.8)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "isb_y_minus4_crosssection.png"
    plt.savefig(str(out_path), dpi=150)
    print(f"Saved to {out_path}")

    if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") == "1":
        plt.show()

    # Print ISB transitions
    print("\nISB transitions:")
    for i in range(len(x) - 1):
        if (los_line[i] < 1e-10) != (los_line[i + 1] < 1e-10):
            lit = i + 1 if los_line[i + 1] > 1e-10 else i
            shd = i if los_line[i] < 1e-10 else i + 1
            print(f"  x={x[i]:.3f}:")
            print(f"    los_lit={los_line[lit]:.3e}  dif_lit={dif_line[lit]:.3e}  dif/los={dif_line[lit]/los_line[lit]:.4f}")
            print(f"    dif_shd={dif_line[shd]:.3e}  dif_shd/los={dif_line[shd]/los_line[lit]:.4f}")
            print(f"    total_lit={tot_line[lit]:.3e}  total_shd={tot_line[shd]:.3e}  jump={abs(tot_line[lit]-tot_line[shd])/max(tot_line[lit],tot_line[shd])*100:.1f}%")


if __name__ == "__main__":
    run()
