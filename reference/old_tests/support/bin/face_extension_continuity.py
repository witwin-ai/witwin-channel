"""Continuity diagnostic across wedge-face extension lines.

This script isolates the dominant first-order direct diffraction state in the
rotated high-permittivity box scene and samples the diffraction field across
the two wedge-face extension lines attached to that state.

For each face line it reports:
  1. The total diffraction field across a 2D scan in line-coordinate/offset
  2. The dominant-state field across the same scan
  3. One-sided jump metrics around the face extension line
  4. The selected state's local geometry/exterior-mask diagnostics

Usage:
    python -m tests.support.bin.face_extension_continuity
    WITWIN_CHANNEL_MAIN_SHOW=1 python -m tests.support.bin.face_extension_continuity
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
    draw_scene,
)
from witwin.channel.trace.diffraction import compute_diffraction_field
from witwin.channel.trace.diffraction.builders import _prepare_diffraction_state_arrays
from witwin.channel.trace.diffraction.constants import SOURCE_TYPE_DIRECT_TX
from witwin.channel.trace.diffraction.field import _edge_state_field_to_targets
from witwin.channel.trace.diffraction.geometry import _compute_edge_geometry, _wedge_exterior_region_mask
from witwin.channel.kernels.trace.packed_state import gather_state_arrays
from witwin.channel.trace.reflection import compute_reflection_field

if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"
OUTPUT_PATH = OUTPUT_DIR / "face_extension_continuity.png"


def _scalar(value) -> float:
    if hasattr(value, "numpy"):
        return float(np.array(value.numpy()).reshape(-1)[0])
    return float(np.asarray(value).reshape(-1)[0])


def _complex_numpy(field) -> np.ndarray:
    return np.asarray(field.real.numpy(), dtype=np.float64) + 1j * np.asarray(field.imag.numpy(), dtype=np.float64)


def _vector_norm_numpy(vector_field) -> np.ndarray:
    x = _complex_numpy(vector_field["x"])
    y = _complex_numpy(vector_field["y"])
    z = _complex_numpy(vector_field["z"])
    return np.sqrt(np.abs(x) ** 2 + np.abs(y) ** 2 + np.abs(z) ** 2)


def _wrapped_phase(field_complex: np.ndarray) -> np.ndarray:
    return np.angle(field_complex)


def _relative_jump(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), 1e-12)
    return np.abs(lhs - rhs) / denom


def _plane_point_on_edge(edge_pos, edge_dir, plane_z: float) -> np.ndarray:
    edge_pos_np = np.array([_scalar(edge_pos.x), _scalar(edge_pos.y), _scalar(edge_pos.z)], dtype=np.float64)
    edge_dir_np = np.array([_scalar(edge_dir.x), _scalar(edge_dir.y), _scalar(edge_dir.z)], dtype=np.float64)
    if abs(edge_dir_np[2]) < 1e-9:
        return np.array([edge_pos_np[0], edge_pos_np[1], plane_z], dtype=np.float64)
    t = (plane_z - edge_pos_np[2]) / edge_dir_np[2]
    return edge_pos_np + t * edge_dir_np


def _horizontal_direction_from_face_normal(face_normal) -> tuple[np.ndarray, np.ndarray]:
    face_normal_np = np.array([_scalar(face_normal.x), _scalar(face_normal.y), _scalar(face_normal.z)], dtype=np.float64)
    line_normal = face_normal_np.copy()
    line_normal[2] = 0.0
    line_normal /= max(np.linalg.norm(line_normal), 1e-12)
    z_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    line_dir = np.cross(z_hat, line_normal)
    line_dir /= max(np.linalg.norm(line_dir), 1e-12)
    if line_dir[0] < 0.0:
        line_dir = -line_dir
        line_normal = -line_normal
    return line_dir, line_normal


def _dominant_direct_state(
    *,
    tracer: Tracer,
    scene,
    reflection_detail,
    reference_x: float,
    reference_y: float,
    rx_z: float,
):
    edge_cache, edge_data, state_arrays, _ = _prepare_diffraction_state_arrays(
        tracer._coerce_tx_pos((-5.0, 5.0, 1.5)),
        rx_z,
        scene,
        tracer.wavelength,
        tracer.k,
        reflection_detail,
        tracer.diffraction_material,
        tracer.reflection_n_rays,
        tracer.reflection_max_bounces,
        tracer.reflection_coef,
        "2d",
        tracer.max_diffractions,
        use_scene_materials=tracer.use_scene_materials_for_diffraction,
        total_state_budget_per_order=None,
        inserted_state_budget_per_order=None,
        max_inserted_reflections_per_path=max(0, tracer.max_diffractions - 1),
        tx_polarization=tracer.tx_polarization,
    )
    del edge_cache
    if edge_data is None or int(state_arrays["n_states"]) == 0:
        raise RuntimeError("No diffraction states available for continuity diagnostic.")

    order = np.asarray(state_arrays["order"].numpy(), dtype=np.int64)
    prefix_depth = np.asarray(state_arrays["prefix_reflection_depth"].numpy(), dtype=np.int64)
    source_type = np.asarray(state_arrays["source_type_code"].numpy(), dtype=np.int64)
    candidate_idx = np.where(
        (order == 1)
        & (prefix_depth == 0)
        & (source_type == SOURCE_TYPE_DIRECT_TX)
    )[0]
    if candidate_idx.size == 0:
        raise RuntimeError("No first-order direct diffraction states found.")

    target = wt.Point3f(wt.Float([reference_x]), wt.Float([reference_y]), wt.Float([rx_z]))
    best_idx = None
    best_mag = -1.0
    for idx in candidate_idx.tolist():
        state = gather_state_arrays(state_arrays, wt.UInt32(idx))
        _, vector_field = _edge_state_field_to_targets(
            state,
            target,
            tracer.k,
            return_vector=True,
            wavelength=tracer.wavelength,
            material_detail=tracer.diffraction_material,
        )
        field_mag = float(_vector_norm_numpy(vector_field)[0])
        if field_mag > best_mag:
            best_mag = field_mag
            best_idx = idx

    best_state = gather_state_arrays(state_arrays, wt.UInt32(int(best_idx)))
    return best_state, int(best_idx), float(best_mag)


def _sample_face_scan(
    *,
    tracer: Tracer,
    scene,
    reflection_detail,
    state,
    rx_z: float,
    face_label: str,
    line_coord_values: np.ndarray,
    offset_values: np.ndarray,
):
    face_normal = state["n0"] if face_label == "face0" else state["n_face_n"]
    line_dir, line_normal = _horizontal_direction_from_face_normal(face_normal)
    line_origin = _plane_point_on_edge(state["edge_pos"], state["edge_dir"], rx_z)

    grid_u, grid_delta = np.meshgrid(line_coord_values, offset_values, indexing="xy")
    points = line_origin[None, None, :] + grid_u[..., None] * line_dir[None, None, :] + grid_delta[..., None] * line_normal[None, None, :]
    X = wt.Float(points[..., 0].reshape(-1).tolist())
    Y = wt.Float(points[..., 1].reshape(-1).tolist())
    target_pos = wt.Point3f(X, Y, wt.Float(rx_z))

    total_real, total_imag, _, components = compute_diffraction_field(
        X,
        Y,
        rx_z,
        tracer._coerce_tx_pos((-5.0, 5.0, 1.5)),
        scene,
        tracer.wavelength,
        tracer.k,
        reflection_detail=reflection_detail,
        max_diffractions=tracer.max_diffractions,
        reflection_n_rays=tracer.reflection_n_rays,
        reflection_max_bounces=tracer.reflection_max_bounces,
        reflection_coef=tracer.reflection_coef,
        reflection_mode="2d",
        grid=None,
        grid_data=None,
        return_components=True,
        return_per_edge=False,
        return_state_audit=False,
        diffraction_material=tracer.diffraction_material,
        use_scene_materials=tracer.use_scene_materials_for_diffraction,
        total_state_budget_per_order=None,
        inserted_state_budget_per_order=None,
        max_inserted_reflections_per_path=max(0, tracer.max_diffractions - 1),
        tx_polarization=tracer.tx_polarization,
        rx_polarization=tracer.rx_polarization,
        execution=tracer.config.trace.diffraction_execution,
    )
    total_scalar = np.asarray(total_real.numpy(), dtype=np.float64) + 1j * np.asarray(total_imag.numpy(), dtype=np.float64)
    total_scalar = total_scalar.reshape(offset_values.size, line_coord_values.size)
    total_vector = {
        axis: _complex_numpy(components["polarization_direct"][axis] + components["polarization_multi"][axis]).reshape(
            offset_values.size, line_coord_values.size
        )
        for axis in ("x", "y", "z")
    }
    total_vector_norm = np.sqrt(sum(np.abs(total_vector[axis]) ** 2 for axis in ("x", "y", "z")))

    dominant_scalar, dominant_vector = _edge_state_field_to_targets(
        state,
        target_pos,
        tracer.k,
        return_vector=True,
        wavelength=tracer.wavelength,
        material_detail=tracer.diffraction_material,
    )
    dominant_scalar = _complex_numpy(dominant_scalar).reshape(offset_values.size, line_coord_values.size)
    dominant_vector_norm = _vector_norm_numpy(dominant_vector).reshape(offset_values.size, line_coord_values.size)

    geometry = _compute_edge_geometry(
        state["source_pos"],
        state["edge_pos"],
        state["edge_dir"],
        state["n0"],
        target_pos,
    )
    phi = np.asarray(geometry["phi"].numpy(), dtype=np.float64).reshape(offset_values.size, line_coord_values.size)
    target_exterior = np.asarray(
        _wedge_exterior_region_mask(
            target_pos - state["edge_pos"],
            state["edge_dir"],
            state["n0"],
            state["n_face_n"],
        ).numpy(),
        dtype=bool,
    ).reshape(offset_values.size, line_coord_values.size)

    mid = offset_values.size // 2
    neg_idx = mid - 1
    pos_idx = mid + 1
    total_scalar_jump = _relative_jump(total_scalar[neg_idx], total_scalar[pos_idx])
    dominant_scalar_jump = _relative_jump(dominant_scalar[neg_idx], dominant_scalar[pos_idx])
    total_vector_jump = _relative_jump(total_vector_norm[neg_idx], total_vector_norm[pos_idx])
    dominant_vector_jump = _relative_jump(dominant_vector_norm[neg_idx], dominant_vector_norm[pos_idx])
    total_phase_jump = np.abs(np.angle(np.exp(1j * (_wrapped_phase(total_scalar[pos_idx]) - _wrapped_phase(total_scalar[neg_idx])))))
    dominant_phase_jump = np.abs(
        np.angle(np.exp(1j * (_wrapped_phase(dominant_scalar[pos_idx]) - _wrapped_phase(dominant_scalar[neg_idx]))))
    )
    exterior_flip = target_exterior[neg_idx] != target_exterior[pos_idx]
    stable_mask = ~exterior_flip

    def _masked_peak(values, mask):
        if not np.any(mask):
            return np.nan, np.nan
        masked = np.where(mask, values, np.nan)
        idx = int(np.nanargmax(masked))
        return float(masked[idx]), float(line_coord_values[idx])

    far_mask = stable_mask & (np.abs(line_coord_values) >= 0.5)

    max_jump_idx = int(np.argmax(total_scalar_jump))
    max_u = float(line_coord_values[max_jump_idx])
    phi_neg = float(phi[neg_idx, max_jump_idx])
    phi_pos = float(phi[pos_idx, max_jump_idx])
    stable_total_peak, stable_total_u = _masked_peak(total_scalar_jump, stable_mask)
    stable_state_peak, stable_state_u = _masked_peak(dominant_scalar_jump, stable_mask)
    far_total_peak, far_total_u = _masked_peak(total_scalar_jump, far_mask)
    far_state_peak, far_state_u = _masked_peak(dominant_scalar_jump, far_mask)
    far_total_vector_peak, far_total_vector_u = _masked_peak(total_vector_jump, far_mask)
    far_state_vector_peak, far_state_vector_u = _masked_peak(dominant_vector_jump, far_mask)

    return {
        "face_label": face_label,
        "line_origin": line_origin,
        "line_dir": line_dir,
        "line_normal": line_normal,
        "line_coord_values": line_coord_values,
        "offset_values": offset_values,
        "total_scalar": total_scalar,
        "total_vector_norm": total_vector_norm,
        "dominant_scalar": dominant_scalar,
        "dominant_vector_norm": dominant_vector_norm,
        "phi": phi,
        "target_exterior": target_exterior,
        "total_scalar_jump": total_scalar_jump,
        "dominant_scalar_jump": dominant_scalar_jump,
        "total_vector_jump": total_vector_jump,
        "dominant_vector_jump": dominant_vector_jump,
        "total_phase_jump": total_phase_jump,
        "dominant_phase_jump": dominant_phase_jump,
        "exterior_flip": exterior_flip,
        "stable_mask": stable_mask,
        "max_jump_u": max_u,
        "max_jump_value": float(total_scalar_jump[max_jump_idx]),
        "max_jump_phi_neg": phi_neg,
        "max_jump_phi_pos": phi_pos,
        "max_jump_exterior_flip": bool(exterior_flip[max_jump_idx]),
        "stable_total_peak": stable_total_peak,
        "stable_total_u": stable_total_u,
        "stable_state_peak": stable_state_peak,
        "stable_state_u": stable_state_u,
        "far_total_peak": far_total_peak,
        "far_total_u": far_total_u,
        "far_state_peak": far_state_peak,
        "far_state_u": far_state_u,
        "far_total_vector_peak": far_total_vector_peak,
        "far_total_vector_u": far_total_vector_u,
        "far_state_vector_peak": far_state_vector_peak,
        "far_state_vector_u": far_state_vector_u,
    }


def run(
    *,
    eps_r: float = 1.0e4,
    freq: float = 1e9,
    rx_z: float = 1.5,
    reference_x: float = -1.25,
    reference_y: float = -4.0,
    n_line_points: int = 161,
    n_offset_points: int = 121,
    line_half_span: float = 5.0,
    offset_half_span: float = 0.12,
):
    tx_pos = wt.Point3f(-5.0, 5.0, 1.5)
    scene = build_scene(
        box_drjit_geometry(
            center=wt.Point3f(0.0, 0.0, 2.0),
            size=4.0,
            rotation=wt.Float(float(np.deg2rad(-5.0))),
        ),
        material=Material(eps_r=eps_r),
    )
    tracer = Tracer(
        frequency=freq,
        scene=scene,
        config=ChannelConfig(
            trace=TraceConfig(
                diffraction_execution=DiffractionExecutionConfig(suffix_dda="symbolic"),
            )
        ),
        reflection_n_rays=20_000,
        reflection_max_bounces=1,
        reflection_coef=1.0,
        max_diffractions=2,
        tx_polarization=(1.0, 0.0, 0.0),
        use_scene_materials_for_diffraction=True,
    )

    reflection_monitor = FieldMonitor(
        "reflection_detail_probe",
        axis="z",
        position=rx_z,
        bounds=((-8.0, 8.0), (-8.0, 8.0)),
        grid_size=32,
    )
    probe_field = reflection_monitor.to_field(tracer.wavelength, default_resolution=tracer.resolution_wavelength)
    probe_coords = probe_field.get_coordinates()
    _, _, reflection_detail = compute_reflection_field(
        grid=probe_field,
        rx_z=rx_z,
        tx_pos=tx_pos,
        scene=scene,
        wavelength=tracer.wavelength,
        k=tracer.k,
        n_rays=tracer.reflection_n_rays,
        max_reflections=tracer.reflection_max_bounces,
        mode="2d",
        reflection_coef=tracer.reflection_coef,
        tx_polarization=tracer.tx_polarization,
        reflection_relative_permittivity=tracer.reflection_relative_permittivity,
        reflection_conductivity=tracer.reflection_conductivity,
        reflection_material=tracer.reflection_material,
        use_scene_materials=tracer.use_scene_materials_for_reflection,
        rx_polarization=tracer.rx_polarization,
        return_per_bounce=False,
        grid_data=probe_coords,
    )

    state, state_idx, state_mag = _dominant_direct_state(
        tracer=tracer,
        scene=scene,
        reflection_detail=reflection_detail,
        reference_x=reference_x,
        reference_y=reference_y,
        rx_z=rx_z,
    )
    edge_idx = int(_scalar(state["edge_idx"]))
    print(
        f"Selected dominant direct state: idx={state_idx}, edge={edge_idx}, "
        f"|E_state(reference)|={state_mag:.3e}, reference=({reference_x:.3f}, {reference_y:.3f}, {rx_z:.3f})"
    )

    line_coord_values = np.linspace(-line_half_span, line_half_span, n_line_points)
    offset_values = np.linspace(-offset_half_span, offset_half_span, n_offset_points)
    face_scans = [
        _sample_face_scan(
            tracer=tracer,
            scene=scene,
            reflection_detail=reflection_detail,
            state=state,
            rx_z=rx_z,
            face_label=face_label,
            line_coord_values=line_coord_values,
            offset_values=offset_values,
        )
        for face_label in ("face0", "face1")
    ]

    for scan in face_scans:
        print(
            f"{scan['face_label']}: max total scalar jump={scan['max_jump_value']:.3e} "
            f"at line_u={scan['max_jump_u']:.3f}, phi(-h)={scan['max_jump_phi_neg']:.4f}, "
            f"phi(+h)={scan['max_jump_phi_pos']:.4f}, exterior_flip={scan['max_jump_exterior_flip']}"
        )
        if np.isfinite(scan["stable_total_peak"]):
            print(
                f"  stable-side residual: total={scan['stable_total_peak']:.3e} at line_u={scan['stable_total_u']:.3f}, "
                f"state={scan['stable_state_peak']:.3e} at line_u={scan['stable_state_u']:.3f}"
            )
        if np.isfinite(scan["far_total_peak"]):
            print(
                f"  extension-only residual (|u|>=0.5): total={scan['far_total_peak']:.3e} at line_u={scan['far_total_u']:.3f}, "
                f"state={scan['far_state_peak']:.3e} at line_u={scan['far_state_u']:.3f}, "
                f"total_vec={scan['far_total_vector_peak']:.3e} at line_u={scan['far_total_vector_u']:.3f}, "
                f"state_vec={scan['far_state_vector_peak']:.3e} at line_u={scan['far_state_vector_u']:.3f}"
            )

    edges_2d = scene.get_edge_data(rx_z)["edges_2d"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    extent = [-line_half_span, line_half_span, -offset_half_span, offset_half_span]

    for row_idx, scan in enumerate(face_scans):
        overview_ax = axes[row_idx, 0]
        draw_scene(overview_ax, edges_2d, (-5.0, 5.0, rx_z), (-8.0, 8.0), (-8.0, 8.0))
        line_start = scan["line_origin"] - line_half_span * scan["line_dir"]
        line_end = scan["line_origin"] + line_half_span * scan["line_dir"]
        overview_ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], color="cyan", linewidth=1.5)
        overview_ax.scatter([scan["line_origin"][0]], [scan["line_origin"][1]], color="yellow", s=24)
        jump_point = scan["line_origin"] + scan["max_jump_u"] * scan["line_dir"]
        overview_ax.scatter([jump_point[0]], [jump_point[1]], color="red", s=18)
        overview_ax.set_title(
            f"{scan['face_label']} extension line\nselected edge {edge_idx}, red=max jump",
            fontsize=10,
        )

        mag_ax = axes[row_idx, 1]
        mag_db = 20.0 * np.log10(np.abs(scan["total_scalar"]) + 1e-20)
        mag_im = mag_ax.imshow(
            mag_db,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="inferno",
            vmin=max(float(np.percentile(mag_db, 5.0)), -120.0),
            vmax=float(np.percentile(mag_db, 99.5)),
        )
        mag_ax.axhline(0.0, color="cyan", linewidth=0.8, linestyle="--")
        mag_ax.axvline(scan["max_jump_u"], color="white", linewidth=0.8, linestyle=":")
        mag_ax.set_title(f"{scan['face_label']} total diffraction |E| (dB)", fontsize=10)
        mag_ax.set_xlabel("line coordinate u (m)")
        mag_ax.set_ylabel("cross-line offset (m)")
        plt.colorbar(mag_im, ax=mag_ax, shrink=0.84)

        phase_ax = axes[row_idx, 2]
        phase_im = phase_ax.imshow(
            _wrapped_phase(scan["dominant_scalar"]),
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
        )
        phase_ax.axhline(0.0, color="cyan", linewidth=0.8, linestyle="--")
        phase_ax.axvline(scan["max_jump_u"], color="white", linewidth=0.8, linestyle=":")
        phase_ax.set_title(f"{scan['face_label']} dominant-state phase", fontsize=10)
        phase_ax.set_xlabel("line coordinate u (m)")
        phase_ax.set_ylabel("cross-line offset (m)")
        plt.colorbar(phase_im, ax=phase_ax, shrink=0.84)

        metric_ax = axes[row_idx, 3]
        metric_ax.plot(scan["line_coord_values"], scan["total_scalar_jump"], label="total scalar jump", linewidth=1.4)
        metric_ax.plot(
            scan["line_coord_values"],
            np.where(scan["stable_mask"], scan["total_scalar_jump"], np.nan),
            label="stable-side total jump",
            linewidth=1.6,
            color="navy",
        )
        metric_ax.plot(scan["line_coord_values"], scan["dominant_scalar_jump"], label="state scalar jump", linewidth=1.2)
        metric_ax.plot(scan["line_coord_values"], scan["total_vector_jump"], label="total vector-norm jump", linewidth=1.0)
        metric_ax.plot(scan["line_coord_values"], scan["dominant_vector_jump"], label="state vector-norm jump", linewidth=1.0)
        metric_ax.plot(
            scan["line_coord_values"],
            0.25 * scan["exterior_flip"].astype(np.float64),
            label="exterior flip x0.25",
            linewidth=1.0,
            linestyle="--",
        )
        metric_ax.axvline(scan["max_jump_u"], color="black", linewidth=0.8, linestyle=":")
        metric_ax.set_title(
            f"{scan['face_label']} one-sided continuity metrics\n"
            f"max total jump={scan['max_jump_value']:.3e}",
            fontsize=10,
        )
        metric_ax.set_xlabel("line coordinate u (m)")
        metric_ax.set_ylabel("relative jump")
        metric_ax.grid(True, alpha=0.3)
        metric_ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(
        "Face-Extension Continuity Diagnostic: total diffraction and dominant-state scans across wedge-face extension lines",
        fontsize=14,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"Saved to {OUTPUT_PATH}")

    if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") == "1":
        plt.show()


if __name__ == "__main__":
    run()
