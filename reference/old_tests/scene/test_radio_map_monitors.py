from __future__ import annotations

import numpy as np
import pytest
import torch
import drjit as dr
import rayd
import witwin as wt
from tests._scene_helpers import box_drjit_geometry, box_geometry, build_scene
from tests.main.plot_multipath_components import CUBE1_BASE_CENTER, build_scene_for_cube1_x
from witwin.channel import (
    Material,
    PathMonitor,
    RadioMapMonitor,
    RadioMapResult,
    Scene,
    Tracer,
    native_extension_available,
    to_numpy,
)
from witwin.channel.monitors.radio_map.grid import AxisAlignedRadioMapNativeGrid, RadioMapGrid
from witwin.channel.monitors.radio_map.deterministic.cell_accumulation import (
    _shadow_boundary_edge_line_bounds,
    _shadow_boundary_transition_responses,
)
from witwin.channel.monitors.radio_map.deterministic.trace import (
    _empty_radio_map_diagnostics,
    trace_radio_map_monitor,
)
import witwin.channel.monitors.radio_map.monte_carlo.trace as radio_map_monte_carlo_trace_module
import witwin.channel.trace.reflection.api as reflection_field
from witwin.channel.trace.diffraction.builders import _prepare_diffraction_state_arrays
from witwin.channel.monitors import resolve_radio_map_monitor
pytestmark = pytest.mark.gpu


def _radio_map_complex_grid(payload: RadioMapResult, name: str) -> np.ndarray:
    return np.asarray(payload.coherent[name], dtype=np.complex64).reshape(payload.tensor_shape)


def _path_amplitudes(paths) -> np.ndarray:
    return np.asarray(paths.a, dtype=np.complex64).reshape(paths.path_shape)


def _runtime_scene():
    return build_scene(box_geometry(center=(0.0, 0.0, -2.0), size=0.5))


def _runtime_wall_scene():
    return build_scene(box_geometry(center=(0.0, 0.0, 1.5), size=(0.25, 6.0, 3.0)))


def _rotated_isb_scene():
    return build_scene(
        box_geometry(
            center=(0.0, 0.0, 2.0),
            size=4.0,
            rotation=float(np.deg2rad(-5.0)),
        ),
        material=Material(eps_r=1.0e4),
    )


def _runtime_three_cube_scene():
    return build_scene(
        box_drjit_geometry(center=(-2.5, -3.0, 1.5), size=2.0, rotation=None).to_mesh(),
        box_drjit_geometry(center=(2.0, 0.5, 1.5), size=2.0, rotation=None).to_mesh(),
        box_drjit_geometry(center=(-0.5, 3.5, 1.5), size=2.0, rotation=None).to_mesh(),
        material=Material(eps_r=1.0e4, sigma_e=0.0),
        edge_selection_mode="vertical_only",
    )


_MC_AD_FLAGS = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad


def _monte_carlo_ad_monitor(
    *,
    accumulation_backend: str = "auto",
    grid_shape=(32, 32),
    samples_per_tx: int = 96,
    ad: bool | None = None,
    max_diffractions: int = 1,
):
    return RadioMapMonitor(
        "radio_map_monte_carlo_ad",
        axis="z",
        position=1.0,
        bounds=((-10.0, 10.0), (-10.0, 10.0)),
        grid_shape=grid_shape,
        combine_mode="incoherent",
        receiver_model="matched_isotropic",
        accumulation_backend=accumulation_backend,
        sampling_mode="monte_carlo",
        ad=ad,
        samples_per_tx=int(samples_per_tx),
        seed=7,
        max_diffractions=int(max_diffractions),
    )


def _monte_carlo_ad_tracer(scene):
    return Tracer(
        frequency=1.0e9,
        scene=scene,
        reflection_n_rays=256,
        reflection_max_bounces=1,
        reflection_coef=0.8,
        enable_rd_diffraction=True,
        max_diffractions=1,
    )


def _mc_gradient_abs_sum(value) -> float:
    return float(np.sum(np.abs(np.asarray(value, dtype=np.float64))))


def _trace_three_cube_monte_carlo_ad_gradients(parameter: str):
    if parameter not in {"cube1_x", "tx_x"}:
        raise ValueError(f"Unsupported parameter: {parameter}")

    def _build_result():
        cube1_x = wt.Float(CUBE1_BASE_CENTER[0])
        tx_x = wt.Float(0.0)
        if parameter == "cube1_x":
            dr.enable_grad(cube1_x)
        else:
            dr.enable_grad(tx_x)
        scene = build_scene_for_cube1_x(cube1_x)
        tracer = _monte_carlo_ad_tracer(scene)
        monitor = _monte_carlo_ad_monitor()
        result = tracer.trace(
            wt.Point3f(tx_x, -5.0, 4.0),
            monitor=monitor,
            verbose=False,
        )
        return result, cube1_x, tx_x

    result, cube1_x, tx_x = _build_result()
    dr.set_grad(cube1_x if parameter == "cube1_x" else tx_x, 1.0)
    path_gain_grad = dr.forward_to(result.path_gain, flags=_MC_AD_FLAGS)

    component_result, cube1_x, tx_x = _build_result()
    dr.set_grad(cube1_x if parameter == "cube1_x" else tx_x, 1.0)
    reflection_grad, diffraction_grad = dr.forward_to(
        component_result.incoherent["reflection"],
        component_result.incoherent["diffraction"],
        flags=_MC_AD_FLAGS,
    )
    return result, {
        "path_gain": _mc_gradient_abs_sum(path_gain_grad),
        "reflection": _mc_gradient_abs_sum(reflection_grad),
        "diffraction": _mc_gradient_abs_sum(diffraction_grad),
    }


def _monte_carlo_diffraction_replay_alignment_summary(*, samples_per_tx: int = 512):
    tracer = _monte_carlo_ad_tracer(_runtime_three_cube_scene())
    monitor = resolve_radio_map_monitor(
        _monte_carlo_ad_monitor(
            grid_shape=(32, 32),
            samples_per_tx=int(samples_per_tx),
            ad=True,
        )
    )
    solver_controls = tracer._resolve_monitor_solver_controls(
        monitor,
        execution_intent="radio_map_incoherent",
    )
    primal_state = radio_map_monte_carlo_trace_module._trace_monte_carlo_primal(
        wt.Point3f(0.0, -5.0, 4.0),
        monitor,
        tracer.scene,
        tracer._resolved_trace_config,
        solver_controls,
        reflection_detail=None,
        radio_map_accumulation_backend="auto",
        return_timing=False,
        resolved_ad_mode=True,
        ad_backend="test",
        loop_mode="symbolic",
        collect_ad_tapes=True,
    )
    tape = primal_state.diffraction_tape
    if tape is None:
        raise AssertionError("Expected Monte Carlo diffraction tape for replay alignment test.")
    width = int(dr.width(tape.cell_idx))
    if width <= 0:
        raise AssertionError("Expected non-empty Monte Carlo diffraction tape.")

    mc_ad = radio_map_monte_carlo_trace_module.mc_ad
    mc_diff = radio_map_monte_carlo_trace_module.mc_diff
    mc_common = radio_map_monte_carlo_trace_module.mc_common
    config = tracer._resolved_trace_config
    grid = primal_state.grid
    support = mc_ad.diffraction_edge_support_arrays(tracer.scene)
    edge_data = mc_diff._gather_diffraction_edge_subset(
        tracer.scene,
        tape.edge_index,
        valid_mask=tape.edge_index >= 0,
    )
    safe_edge_idx = wt.UInt32(dr.select(tape.edge_index >= 0, tape.edge_index, wt.Int32(0)))
    edge_v0_idx = dr.gather(wt.Int32, support["edge_v0"], safe_edge_idx)
    edge_v1_idx = dr.gather(wt.Int32, support["edge_v1"], safe_edge_idx)
    face0_prim_idx = dr.gather(wt.Int32, support["face0_prim"], safe_edge_idx)
    face1_prim_idx = dr.gather(wt.Int32, support["face1_prim"], safe_edge_idx)
    edge_v0 = mc_ad.local_vertex_point(tracer.scene, edge_v0_idx)
    edge_v1 = mc_ad.local_vertex_point(tracer.scene, edge_v1_idx)
    edge_vec = edge_v1 - edge_v0
    edge_length = dr.norm(edge_vec) + wt.Float(1.0e-6)
    edge_dir = edge_vec / edge_length
    edge_pos = wt.Point3f(
        wt.Float(0.5) * (edge_v0.x + edge_v1.x),
        wt.Float(0.5) * (edge_v0.y + edge_v1.y),
        wt.Float(0.5) * (edge_v0.z + edge_v1.z),
    )
    edge_length_scale = edge_length / dr.maximum(edge_data["length"], wt.Float(1.0e-6))
    line_min = edge_data["line_min"] * edge_length_scale
    line_max = edge_data["line_max"] * edge_length_scale
    line_length = dr.maximum(line_max - line_min, wt.Float(0.0))
    diff_point = edge_pos + edge_dir * (line_min + line_length * tape.edge_fraction)
    source_pos = wt.Point3f(
        dr.full(wt.Float, 0.0, width),
        dr.full(wt.Float, -5.0, width),
        dr.full(wt.Float, 4.0, width),
    )
    incident_dir = diff_point - source_pos
    face0_material = mc_ad.local_material_support(
        face0_prim_idx,
        scene=tracer.scene,
        override_material=config.diffraction_material,
        use_scene_materials=bool(config.use_scene_materials_for_diffraction),
        default_eta_r=5.0,
        default_sigma=0.0,
        gain=float(config.reflection_coef),
    )
    face1_material = mc_ad.local_material_support(
        face1_prim_idx,
        scene=tracer.scene,
        override_material=config.diffraction_material,
        use_scene_materials=bool(config.use_scene_materials_for_diffraction),
        default_eta_r=5.0,
        default_sigma=0.0,
        gain=float(config.reflection_coef),
    )
    flip = dr.dot(incident_dir, edge_data["n0"]) > 0.0
    oriented_edge_dir = dr.select(flip, -edge_dir, edge_dir)
    oriented_n0 = dr.select(flip, edge_data["n_face_n"], edge_data["n0"])
    oriented_nn = dr.select(flip, edge_data["n0"], edge_data["n_face_n"])
    oriented_face0_eta_r = dr.select(flip, face1_material["eta_r"], face0_material["eta_r"])
    oriented_face0_sigma = dr.select(flip, face1_material["sigma"], face0_material["sigma"])
    oriented_face0_gain = dr.select(flip, face1_material["gain"], face0_material["gain"])
    oriented_face0_use_fresnel = dr.select(
        flip,
        face1_material["use_fresnel"],
        face0_material["use_fresnel"],
    )
    oriented_face1_eta_r = dr.select(flip, face0_material["eta_r"], face1_material["eta_r"])
    oriented_face1_sigma = dr.select(flip, face0_material["sigma"], face1_material["sigma"])
    oriented_face1_gain = dr.select(flip, face0_material["gain"], face1_material["gain"])
    oriented_face1_use_fresnel = dr.select(
        flip,
        face0_material["use_fresnel"],
        face1_material["use_fresnel"],
    )
    face_sum = oriented_n0 + oriented_nn
    face_sum_norm = dr.norm(face_sum)
    offset_normal = dr.select(
        face_sum_norm > wt.Float(1.0e-6),
        face_sum / face_sum_norm,
        wt.Vector3f(0.0, 0.0, 0.0),
    )
    ko = mc_diff._sample_keller_cone(
        oriented_edge_dir,
        oriented_n0,
        oriented_nn,
        tape.cone_sample,
        incident_dir,
        lit_region=True,
    )
    ray_origin = mc_common._spawn_offset_ray_origin(diff_point, ko, offset_normal)
    plane_hit = mc_common._plane_hit_from_segment(
        ray_origin=ray_origin,
        ray_dir=ko,
        blocker_dist=dr.full(wt.Float, 1.0e10, width),
        grid=grid,
        active=dr.full(wt.Bool, True, width),
    )
    integration_weight = mc_diff._diffraction_integration_weight(
        edge_origin=edge_pos,
        edge_dir=oriented_edge_dir,
        n0=oriented_n0,
        source_pos=source_pos,
        diff_point=diff_point,
        k_world=ko,
        target_pos=plane_hit["target_pos"],
        plane_normal=mc_common._axis_unit_normal(str(grid.axis)),
    )
    field_power = mc_diff.diff_field._sampled_edge_diffraction_power_to_targets_mc(
        source_pos=source_pos,
        edge_dir=oriented_edge_dir,
        n0=oriented_n0,
        nn=oriented_nn,
        wedge_n=edge_data["wedge_n"],
        face0_eta_r=oriented_face0_eta_r,
        face0_sigma=oriented_face0_sigma,
        face0_gain=oriented_face0_gain,
        face0_use_fresnel=oriented_face0_use_fresnel,
        face1_eta_r=oriented_face1_eta_r,
        face1_sigma=oriented_face1_sigma,
        face1_gain=oriented_face1_gain,
        face1_use_fresnel=oriented_face1_use_fresnel,
        sampled_edge_pos=diff_point,
        target_pos=plane_hit["target_pos"],
        k=config.k,
        wavelength=config.wavelength,
        support_override={
            "field_valid": tape.field_valid,
            "pole_safe": tape.pole_safe,
            "dif_n_p": tape.dif_n_p,
            "dif_n_m": tape.dif_n_m,
            "sum_n_p": tape.sum_n_p,
            "sum_n_m": tape.sum_n_m,
        },
    )
    cell_area = float(grid.cell_size[0] * grid.cell_size[1])
    diffraction_scale = (float(config.wavelength) / (4.0 * np.pi)) ** 2 / cell_area
    contribution = np.asarray(
        field_power
        * wt.Float(diffraction_scale)
        * integration_weight
        * wt.Float(primal_state.diffraction_total_length_weight)
        * edge_data["wedge_n"]
        * wt.Float(np.pi),
        dtype=np.float64,
    )
    replay = np.zeros(int(grid.n_cells), dtype=np.float64)
    np.add.at(replay, np.asarray(tape.cell_idx, dtype=np.uint32), contribution)
    primal = np.asarray(primal_state.component_power["diffraction"], dtype=np.float64)
    diff = replay - primal
    return {
        "tape_width": width,
        "sum_abs_diff": float(np.sum(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "replay_sum": float(np.sum(replay)),
        "primal_sum": float(np.sum(primal)),
    }


def test_shadow_boundary_edge_bounds_require_explicit_finite_segments():
    with pytest.raises(RuntimeError, match="line_min and line_max"):
        _shadow_boundary_edge_line_bounds(
            None,
            {"length": wt.Float(2.0)},
            wt.UInt32(0),
        )


def _native_coherent_gradient_monitor(*, grid_shape=(8, 8)):
    return RadioMapMonitor(
        "radio_map_native_grad",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=grid_shape,
        combine_mode="coherent",
        receiver_model="projected_polarized",
        accumulation_backend="native_coherent",
        ray_mode="3d",
        quadrature_mode="center",
        max_diffractions=1,
    )


def _native_coherent_gradient_weights(monitor: RadioMapMonitor):
    n_cells = int(np.prod(monitor.grid_shape))
    return wt.Float(np.linspace(0.5, 1.5, n_cells, dtype=np.float32))


def _trace_native_coherent_radio_map(*, tracer: Tracer, monitor: RadioMapMonitor, tx_pos, return_timing: bool = False):
    solver_controls = tracer._resolve_monitor_solver_controls(
        monitor,
        execution_intent="radio_map_coherent",
    )
    return trace_radio_map_monitor(
        tx_pos,
        monitor,
        tracer.scene,
        tracer._resolved_trace_config,
        solver_controls,
        radio_map_accumulation_backend="native_coherent",
        return_timing=return_timing,
    )


def _point_grad_abs_sum(point) -> float:
    grad_components = [
        np.asarray(to_numpy(dr.grad(getattr(point, axis))), dtype=np.float32).reshape(-1)
        for axis in ("x", "y", "z")
    ]
    return float(sum(np.sum(np.abs(component)) for component in grad_components))


def _cut_jump_db(path_gain: np.ndarray, los_mask: np.ndarray) -> float:
    transition_idx = np.where(los_mask[:-1] != los_mask[1:])[0]
    assert int(transition_idx.size) >= 1
    idx = int(transition_idx[0])
    left = 10.0 * np.log10(max(float(path_gain[idx]), 1.0e-20))
    right = 10.0 * np.log10(max(float(path_gain[idx + 1]), 1.0e-20))
    return float(right - left)


def _run_projected_isb_completion_case(
    *,
    tracer: Tracer,
    tx_pos,
    plane_z: float,
    grid_shape: tuple[int, int],
):
    raw_monitor = RadioMapMonitor(
        "radio_map_projected_isb_raw",
        axis="z",
        position=float(plane_z),
        bounds=((-8.0, 8.0), (-8.0, 8.0)),
        grid_shape=grid_shape,
        combine_mode="coherent",
        receiver_model="projected_polarized",
        ray_mode="3d",
    )
    completion_monitor = RadioMapMonitor(
        "radio_map_projected_isb_completion",
        axis="z",
        position=float(plane_z),
        bounds=((-8.0, 8.0), (-8.0, 8.0)),
        grid_shape=grid_shape,
        combine_mode="coherent",
        receiver_model="projected_polarized",
        ray_mode="3d",
        shadow_boundary_mode="projected_isb_completion",
    )

    raw_payload = tracer.trace(tx_pos, monitor=raw_monitor, verbose=False)
    completion_payload = tracer.trace(tx_pos, monitor=completion_monitor, verbose=False)

    x_coords = np.asarray(raw_payload.coords.x, dtype=np.float32)
    y_coords = np.asarray(raw_payload.coords.y, dtype=np.float32)
    raw_path_gain = np.asarray(raw_payload.path_gain, dtype=np.float32)
    completion_path_gain = np.asarray(completion_payload.path_gain, dtype=np.float32)
    raw_los = _radio_map_complex_grid(raw_payload, "los")

    col = int(np.argmin(np.abs(x_coords - 4.0)))
    row = int(np.argmin(np.abs(y_coords + 4.0)))
    x4_los_mask = (np.abs(raw_los[:, col]) ** 2) > 1.0e-14
    yneg4_los_mask = (np.abs(raw_los[row, :]) ** 2) > 1.0e-14
    raw_x4_jump = _cut_jump_db(raw_path_gain[:, col], x4_los_mask)
    completion_x4_jump = _cut_jump_db(completion_path_gain[:, col], x4_los_mask)
    raw_yneg4_jump = _cut_jump_db(raw_path_gain[row, :], yneg4_los_mask)
    completion_yneg4_jump = _cut_jump_db(completion_path_gain[row, :], yneg4_los_mask)

    completion_surrogate_total = np.asarray(
        completion_payload.coherent_power["projected_isb_surrogate_total"],
        dtype=np.float32,
    )
    completion_weight = np.asarray(
        completion_payload.incoherent["projected_isb_completion_weight"],
        dtype=np.float32,
    )
    completion_deficiency = np.asarray(
        completion_payload.incoherent["projected_isb_completion_deficiency"],
        dtype=np.float32,
    )
    completion_only = np.asarray(
        completion_payload.coherent_power["projected_isb_completion"],
        dtype=np.float32,
    )
    geometry = tracer.scene.structures[0].geometry
    geometry_device = geometry.position.device
    grid_y, grid_x = torch.meshgrid(
        torch.tensor(y_coords, dtype=torch.float32, device=geometry_device),
        torch.tensor(x_coords, dtype=torch.float32, device=geometry_device),
        indexing="ij",
    )
    grid_z = torch.full_like(grid_x, float(plane_z))
    inside_mask = geometry.signed_distance(grid_x, grid_y, grid_z) < -1.0e-4
    inside_mask_np = inside_mask.detach().cpu().numpy()

    return {
        "raw_payload": raw_payload,
        "completion_payload": completion_payload,
        "raw_path_gain": raw_path_gain,
        "completion_path_gain": completion_path_gain,
        "completion_surrogate_total": completion_surrogate_total,
        "completion_weight": completion_weight,
        "completion_deficiency": completion_deficiency,
        "completion_only": completion_only,
        "inside_mask_np": inside_mask_np,
        "raw_x4_jump": raw_x4_jump,
        "completion_x4_jump": completion_x4_jump,
        "raw_yneg4_jump": raw_yneg4_jump,
        "completion_yneg4_jump": completion_yneg4_jump,
    }


def _run_matched_isb_completion_case(
    *,
    tracer: Tracer,
    tx_pos,
    plane_z: float,
    grid_shape: tuple[int, int],
):
    raw_monitor = RadioMapMonitor(
        "radio_map_matched_isb_raw",
        axis="z",
        position=float(plane_z),
        bounds=((-8.0, 8.0), (-8.0, 8.0)),
        grid_shape=grid_shape,
        combine_mode="coherent",
        receiver_model="matched_isotropic",
        ray_mode="3d",
    )
    completion_monitor = RadioMapMonitor(
        "radio_map_matched_isb_completion",
        axis="z",
        position=float(plane_z),
        bounds=((-8.0, 8.0), (-8.0, 8.0)),
        grid_shape=grid_shape,
        combine_mode="coherent",
        receiver_model="matched_isotropic",
        ray_mode="3d",
        shadow_boundary_mode="matched_isb_completion",
    )

    raw_payload = tracer.trace(tx_pos, monitor=raw_monitor, verbose=False)
    completion_payload = tracer.trace(tx_pos, monitor=completion_monitor, verbose=False)

    x_coords = np.asarray(raw_payload.coords.x, dtype=np.float32)
    y_coords = np.asarray(raw_payload.coords.y, dtype=np.float32)
    raw_path_gain = np.asarray(raw_payload.path_gain, dtype=np.float32)
    completion_path_gain = np.asarray(completion_payload.path_gain, dtype=np.float32)
    raw_los_power = np.asarray(raw_payload.coherent_power["los"], dtype=np.float32).reshape(
        raw_payload.tensor_shape
    )

    col = int(np.argmin(np.abs(x_coords - 4.0)))
    row = int(np.argmin(np.abs(y_coords + 4.0)))
    x4_los_mask = raw_los_power[:, col] > 1.0e-14
    yneg4_los_mask = raw_los_power[row, :] > 1.0e-14
    raw_x4_jump = _cut_jump_db(raw_path_gain.reshape(raw_payload.tensor_shape)[:, col], x4_los_mask)
    completion_x4_jump = _cut_jump_db(
        completion_path_gain.reshape(completion_payload.tensor_shape)[:, col],
        x4_los_mask,
    )
    raw_yneg4_jump = _cut_jump_db(raw_path_gain.reshape(raw_payload.tensor_shape)[row, :], yneg4_los_mask)
    completion_yneg4_jump = _cut_jump_db(
        completion_path_gain.reshape(completion_payload.tensor_shape)[row, :],
        yneg4_los_mask,
    )

    completion_surrogate_total = np.asarray(
        completion_payload.coherent_power["matched_isb_surrogate_total"],
        dtype=np.float32,
    )
    completion_weight = np.asarray(
        completion_payload.incoherent["matched_isb_completion_weight"],
        dtype=np.float32,
    )
    completion_hard_visibility = np.asarray(
        completion_payload.incoherent["matched_isb_hard_visibility"],
        dtype=np.float32,
    )
    completion_transition_magnitude = np.asarray(
        completion_payload.incoherent["matched_isb_transition_magnitude"],
        dtype=np.float32,
    )
    completion_transition_phase = np.asarray(
        completion_payload.incoherent["matched_isb_transition_phase"],
        dtype=np.float32,
    )
    completion_only = np.asarray(
        completion_payload.coherent_power["matched_isb_completion"],
        dtype=np.float32,
    )
    geometry = tracer.scene.structures[0].geometry
    geometry_device = geometry.position.device
    grid_y, grid_x = torch.meshgrid(
        torch.tensor(y_coords, dtype=torch.float32, device=geometry_device),
        torch.tensor(x_coords, dtype=torch.float32, device=geometry_device),
        indexing="ij",
    )
    grid_z = torch.full_like(grid_x, float(plane_z))
    inside_mask = geometry.signed_distance(grid_x, grid_y, grid_z) < -1.0e-4
    inside_mask_np = inside_mask.detach().cpu().numpy()

    return {
        "raw_payload": raw_payload,
        "completion_payload": completion_payload,
        "raw_path_gain": raw_path_gain,
        "completion_path_gain": completion_path_gain,
        "completion_surrogate_total": completion_surrogate_total,
        "completion_weight": completion_weight,
        "completion_hard_visibility": completion_hard_visibility,
        "completion_transition_magnitude": completion_transition_magnitude,
        "completion_transition_phase": completion_transition_phase,
        "completion_only": completion_only,
        "inside_mask_np": inside_mask_np,
        "raw_x4_jump": raw_x4_jump,
        "completion_x4_jump": completion_x4_jump,
        "raw_yneg4_jump": raw_yneg4_jump,
        "completion_yneg4_jump": completion_yneg4_jump,
    }


def _cell_center_positions(grid) -> np.ndarray:
    return np.stack(
        [
            np.asarray(grid.cell_centers.x, dtype=np.float32),
            np.asarray(grid.cell_centers.y, dtype=np.float32),
            np.asarray(grid.cell_centers.z, dtype=np.float32),
        ],
        axis=-1,
    )


def _path_gain_from_paths(paths, *, tensor_shape: tuple[int, int]) -> np.ndarray:
    coeff = _path_amplitudes(paths)
    valid = np.asarray(paths.valid, dtype=np.bool_)
    power = (np.abs(coeff) ** 2) * valid
    return power.sum(axis=1, dtype=np.float32).reshape(tensor_shape)


def _coherent_power_from_paths(paths, *, tensor_shape: tuple[int, int]) -> np.ndarray:
    coeff = _path_amplitudes(paths)
    valid = np.asarray(paths.valid, dtype=np.bool_)
    coherent = (coeff * valid).sum(axis=1, dtype=np.complex64)
    return (np.abs(coherent) ** 2).astype(np.float32).reshape(tensor_shape)


def _build_tracer(scene: Scene) -> Tracer:
    return Tracer(
        frequency=1.0e9,
        scene=scene,
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=0,
    )


def _build_multipath_tracer(scene: Scene) -> Tracer:
    return Tracer(
        frequency=1.0e9,
        scene=scene,
        reflection_n_rays=256,
        reflection_max_bounces=1,
        max_diffractions=1,
    )


def test_scene_accepts_radio_map_monitors_and_grid_is_cell_centered():
    monitor = RadioMapMonitor(
        "radio_map_xy",
        axis="z",
        position=1.5,
        bounds=((-2.0, 2.0), (-3.0, 3.0)),
        grid_shape=(4, 3),
    )
    scene = Scene(monitors=[monitor])
    grid = RadioMapGrid.from_monitor(monitor, default_cell_size=None)

    assert scene.monitors == [monitor]
    assert scene.resolved_monitors() == [monitor]
    assert scene.clone().monitors == [monitor]
    np.testing.assert_allclose(np.asarray(grid.x_coords, dtype=np.float32), [-1.5, -0.5, 0.5, 1.5])
    np.testing.assert_allclose(np.asarray(grid.y_coords, dtype=np.float32), [-2.0, 0.0, 2.0])
    assert grid.grid_shape == (4, 3)
    assert grid.tensor_shape == (3, 4)
    assert grid.surface_descriptor()["surface_mode"] == "axis_aligned"


def test_axis_aligned_native_grid_adapter_matches_radio_map_sample_positions():
    monitor = RadioMapMonitor(
        "radio_map_native_adapter",
        axis="z",
        position=1.5,
        bounds=((-2.0, 2.0), (-2.0, 2.0)),
        grid_shape=(4, 4),
    )
    grid = RadioMapGrid.from_monitor(monitor, default_cell_size=None)
    native_grid = AxisAlignedRadioMapNativeGrid.from_grid(grid, sample_index=0)
    sample_positions = grid.sample_sets[0].positions

    np.testing.assert_allclose(np.asarray(native_grid.X, dtype=np.float32), np.asarray(sample_positions.x, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(native_grid.Y, dtype=np.float32), np.asarray(sample_positions.y, dtype=np.float32))
    np.testing.assert_allclose(
        np.asarray(native_grid.receivers.z, dtype=np.float32),
        np.asarray(sample_positions.z, dtype=np.float32),
    )
    assert native_grid.sample_index == 0


def test_radio_map_monitor_defaults_to_first_order_diffraction():
    tracer = Tracer(
        frequency=1.0e9,
        scene=_runtime_wall_scene(),
        reflection_n_rays=256,
        reflection_max_bounces=1,
        max_diffractions=3,
    )
    monitor = RadioMapMonitor(
        "radio_map_default_diffraction",
        axis="z",
        position=1.5,
        bounds=((-2.0, 2.0), (-2.0, 2.0)),
        grid_shape=(4, 4),
    )
    solver_controls = tracer._resolve_monitor_solver_controls(
        monitor,
        execution_intent="radio_map_incoherent",
    )

    assert monitor.max_diffractions == 1
    assert int(solver_controls["effective"]["max_diffractions"]) == 1


def test_radio_map_monitor_rejects_removed_2x2_quadrature_mode():
    with pytest.raises(ValueError, match="quadrature_mode"):
        RadioMapMonitor(
            "radio_map_removed_2x2",
            axis="z",
            position=1.5,
            bounds=((-2.0, 2.0), (-2.0, 2.0)),
            grid_shape=(4, 4),
            quadrature_mode="2x2",
        )


def test_radio_map_monitor_resolves_receiver_model_by_contract():
    axis_aligned_incoherent = RadioMapMonitor(
        "radio_map_default_receiver_model",
        axis="z",
        position=1.5,
        bounds=((-2.0, 2.0), (-2.0, 2.0)),
        grid_shape=(4, 4),
    )
    coherent = RadioMapMonitor(
        "radio_map_default_receiver_model_coherent",
        axis="z",
        position=1.5,
        bounds=((-2.0, 2.0), (-2.0, 2.0)),
        grid_shape=(4, 4),
        combine_mode="coherent",
    )
    oriented = RadioMapMonitor(
        "radio_map_default_receiver_model_oriented",
        center=(0.0, 0.0, 1.5),
        orientation=(0.1, 0.2, 0.3),
        size=(2.0, 2.0),
        grid_shape=(2, 2),
    )

    assert axis_aligned_incoherent.receiver_model == "matched_isotropic"
    assert coherent.receiver_model == "projected_polarized"
    assert oriented.receiver_model == "projected_polarized"


def test_radio_map_monitor_validates_monte_carlo_contract():
    monitor = RadioMapMonitor(
        "radio_map_monte_carlo_defaults",
        axis="z",
        position=1.5,
        bounds=((-2.0, 2.0), (-2.0, 2.0)),
        grid_shape=(4, 4),
        sampling_mode="monte_carlo",
    )

    assert monitor.sampling_mode == "monte_carlo"
    assert monitor.samples_per_tx == 65536
    assert monitor.rr_prob == 1.0
    assert monitor.receiver_model == "matched_isotropic"
    assert monitor.accumulation_backend == "auto"

    with pytest.raises(ValueError, match="combine_mode='incoherent'"):
        RadioMapMonitor(
            "radio_map_monte_carlo_invalid_combine",
            axis="z",
            position=1.5,
            bounds=((-2.0, 2.0), (-2.0, 2.0)),
            grid_shape=(4, 4),
            sampling_mode="monte_carlo",
            combine_mode="coherent",
        )

    with pytest.raises(ValueError, match="receiver_model='matched_isotropic'"):
        RadioMapMonitor(
            "radio_map_monte_carlo_invalid_receiver",
            axis="z",
            position=1.5,
            bounds=((-2.0, 2.0), (-2.0, 2.0)),
            grid_shape=(4, 4),
            sampling_mode="monte_carlo",
            receiver_model="projected_polarized",
        )

    with pytest.raises(ValueError, match="max_diffractions <= 1"):
        RadioMapMonitor(
            "radio_map_monte_carlo_invalid_diffraction_order",
            axis="z",
            position=1.5,
            bounds=((-2.0, 2.0), (-2.0, 2.0)),
            grid_shape=(4, 4),
            sampling_mode="monte_carlo",
            max_diffractions=2,
        )


def test_radio_map_monitor_rejects_shadow_boundary_surrogate_without_matched_incoherent_receiver():
    with pytest.raises(
        ValueError,
        match="shadow_boundary_mode='utd_cross_term_surrogate' requires combine_mode='incoherent' and receiver_model='matched_isotropic'",
    ):
        RadioMapMonitor(
            "radio_map_shadow_boundary_invalid",
            axis="z",
            position=1.5,
            bounds=((-2.0, 2.0), (-2.0, 2.0)),
            grid_shape=(4, 4),
            receiver_model="projected_polarized",
            shadow_boundary_mode="utd_cross_term_surrogate",
        )


def test_radio_map_monitor_rejects_projected_isb_completion_without_projected_coherent_center_contract():
    with pytest.raises(
        ValueError,
        match="shadow_boundary_mode='projected_isb_completion' requires combine_mode='coherent', receiver_model='projected_polarized', and quadrature_mode='center'",
    ):
        RadioMapMonitor(
            "radio_map_projected_isb_invalid",
            axis="z",
            position=1.5,
            bounds=((-2.0, 2.0), (-2.0, 2.0)),
            grid_shape=(4, 4),
            combine_mode="incoherent",
            receiver_model="projected_polarized",
            shadow_boundary_mode="projected_isb_completion",
        )


def test_radio_map_monitor_rejects_matched_isb_completion_without_matched_coherent_center_contract():
    with pytest.raises(
        ValueError,
        match="shadow_boundary_mode='matched_isb_completion' requires combine_mode='coherent', receiver_model='matched_isotropic', and quadrature_mode='center'",
    ):
        RadioMapMonitor(
            "radio_map_matched_isb_invalid",
            axis="z",
            position=1.5,
            bounds=((-2.0, 2.0), (-2.0, 2.0)),
            grid_shape=(4, 4),
            combine_mode="coherent",
            receiver_model="projected_polarized",
            shadow_boundary_mode="matched_isb_completion",
        )


def test_radio_map_trace_matches_path_monitor_baseline_for_cell_centers():
    tracer = _build_tracer(_runtime_scene())
    monitor = RadioMapMonitor(
        "radio_map_xy",
        axis="z",
        position=1.5,
        bounds=((-2.0, 2.0), (-1.0, 1.0)),
        grid_shape=(4, 2),
        tx_power=2.5,
        receiver_model="projected_polarized",
    )
    grid = RadioMapGrid.from_monitor(monitor, default_cell_size=tracer.cell_size)
    path_monitor = PathMonitor(
        "radio_map_baseline",
        positions=torch.as_tensor(_cell_center_positions(grid), dtype=torch.float32),
        max_diffractions=0,
    )
    tx_pos = wt.Point3f(0.0, -3.0, 1.5)

    payload = tracer.trace(tx_pos, monitor=monitor, verbose=False)
    paths = tracer.trace(tx_pos, monitor=path_monitor, verbose=False)

    assert isinstance(payload, RadioMapResult)
    np.testing.assert_allclose(
        np.asarray(payload.path_gain, dtype=np.float32),
        _path_gain_from_paths(paths, tensor_shape=payload.tensor_shape),
        rtol=1e-5,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(payload.rss, dtype=np.float32),
        np.asarray(payload.path_gain, dtype=np.float32) * 2.5,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_array_equal(np.asarray(payload.tx_association(), dtype=np.int32), 0)
    assert payload.values is payload.path_gain
    assert payload.combine_mode == "incoherent"


def test_radio_map_trace_reports_matched_isotropic_receiver_model_on_default_axis_aligned_incoherent_trace():
    tracer = _build_tracer(_runtime_scene())
    monitor = RadioMapMonitor(
        "radio_map_default_receiver_model_trace",
        axis="z",
        position=1.5,
        bounds=((-2.0, 2.0), (-1.0, 1.0)),
        grid_shape=(4, 2),
    )

    payload = tracer.trace(
        wt.Point3f(0.0, -3.0, 1.5),
        monitor=monitor,
        verbose=False,
    )

    assert payload.receiver_model == "matched_isotropic"
    assert payload.metadata["receiver_model"] == "matched_isotropic"
    assert payload.metadata["execution_intent"]["kind"] == "radio_map_incoherent"
    assert payload.metadata["accumulation_backend"]["resolved"] == "cell_accumulation"


def test_radio_map_center_sampling_and_trace_many_fill_sinr_and_association():
    tracer = _build_tracer(_runtime_scene())
    monitor = RadioMapMonitor(
        "radio_map_sinr",
        axis="z",
        position=1.5,
        bounds=((-2.0, 2.0), (-1.0, 1.0)),
        grid_shape=(4, 2),
        tx_power=2.0,
        metric="sinr",
    )
    results = tracer.trace_many(
        [
            {"tx_pos": wt.Point3f(-2.0, -3.0, 1.5), "tx_label": "left"},
            {"tx_pos": wt.Point3f(2.0, -3.0, 1.5), "tx_label": "right"},
        ],
        monitor=monitor,
        verbose=False,
    )

    left_map = results[0]
    right_map = results[1]

    assert len(left_map.sample_positions()) == 1
    assert tuple(left_map.metadata["aggregate_tx_labels"]) == ("left", "right")
    assert left_map.metadata["tx_stack_execution"]["mode"] == "trace_many_streaming_post_aggregation"
    assert bool(left_map.metadata["tx_stack_execution"]["native"]) is False
    assert bool(left_map.metadata["tx_stack_execution"]["rss_stack_materialized"]) is False
    association = np.asarray(left_map.tx_association(), dtype=np.int32)
    np.testing.assert_array_equal(association, np.asarray(right_map.tx_association(), dtype=np.int32))
    np.testing.assert_array_equal(association[:, :2], 0)
    np.testing.assert_array_equal(association[:, 2:], 1)

    left_rss = np.asarray(left_map.rss, dtype=np.float32)
    right_rss = np.asarray(right_map.rss, dtype=np.float32)
    expected_left_sinr = left_rss / np.maximum(right_rss, np.finfo(np.float32).tiny)
    expected_right_sinr = right_rss / np.maximum(left_rss, np.finfo(np.float32).tiny)
    np.testing.assert_allclose(
        np.asarray(left_map.sinr, dtype=np.float32),
        expected_left_sinr,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(right_map.sinr, dtype=np.float32),
        expected_right_sinr,
        rtol=1e-5,
        atol=1e-6,
    )


def test_radio_map_result_samples_metric_positions_with_association_filter():
    tracer = _build_tracer(_runtime_scene())
    monitor = RadioMapMonitor(
        "radio_map_sampling",
        axis="z",
        position=1.5,
        bounds=((-2.0, 2.0), (-1.0, 1.0)),
        grid_shape=(4, 2),
        tx_power=2.0,
        metric="rss",
    )
    payload = tracer.trace_many(
        [
            {"tx_pos": wt.Point3f(-2.0, -3.0, 1.5), "tx_label": "left"},
            {"tx_pos": wt.Point3f(2.0, -3.0, 1.5), "tx_label": "right"},
        ],
        monitor=monitor,
        verbose=False,
    )[0]

    positions, cell_indices = payload.sample_metric_positions(
        4,
        tx_association="left",
        replacement=False,
        jitter=False,
        seed=12,
        return_cell_indices=True,
    )

    assert tuple(positions.shape) == (4, 3)
    np.testing.assert_array_less(positions[:, 0].detach().cpu().numpy(), 0.0)
    np.testing.assert_allclose(
        positions[:, 2].detach().cpu().numpy(),
        np.full((4,), 1.5, dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_array_less(cell_indices[:, 1].detach().cpu().numpy(), 2)


def test_oriented_radio_map_metric_sampling_stays_on_surface_plane():
    tracer = _build_tracer(_runtime_scene())
    monitor = RadioMapMonitor(
        "radio_map_tilted_sampling",
        center=(0.0, 0.0, 1.5),
        orientation=(0.2, 0.1, 0.3),
        size=(2.0, 2.0),
        grid_shape=(2, 2),
        tx_power=1.0,
        receiver_model="projected_polarized",
    )
    payload = tracer.trace(wt.Point3f(-1.0, -3.0, 1.5), monitor=monitor, verbose=False)

    positions = payload.sample_metric_positions(12, seed=5, jitter=True)
    center = torch.tensor(payload.surface["center"], dtype=torch.float32, device=positions.device)
    normal = torch.tensor(payload.surface["normal"], dtype=torch.float32, device=positions.device)
    signed_distance = torch.sum((positions - center.unsqueeze(0)) * normal.unsqueeze(0), dim=1)

    np.testing.assert_allclose(
        signed_distance.detach().cpu().numpy(),
        np.zeros((12,), dtype=np.float32),
        atol=1e-5,
    )


def test_oriented_radio_map_matches_path_monitor_baseline():
    tracer = _build_tracer(_runtime_scene())
    monitor = RadioMapMonitor(
        "radio_map_tilted",
        center=(0.0, 0.0, 1.5),
        orientation=(0.2, 0.1, 0.0),
        size=(2.0, 2.0),
        grid_shape=(2, 2),
        tx_power=1.0,
        receiver_model="projected_polarized",
    )
    grid = RadioMapGrid.from_monitor(monitor, default_cell_size=tracer.cell_size)
    path_monitor = PathMonitor(
        "radio_map_tilted_baseline",
        positions=torch.as_tensor(_cell_center_positions(grid), dtype=torch.float32),
        max_diffractions=0,
    )
    tx_pos = wt.Point3f(-1.0, -3.0, 1.5)

    payload = tracer.trace(tx_pos, monitor=monitor, verbose=False)
    paths = tracer.trace(tx_pos, monitor=path_monitor, verbose=False)

    np.testing.assert_allclose(
        np.asarray(payload.path_gain, dtype=np.float32),
        _path_gain_from_paths(paths, tensor_shape=payload.tensor_shape),
        rtol=1e-5,
        atol=1e-7,
    )
    assert payload.surface["surface_mode"] == "oriented"
    assert payload.tangential_axes == ("u", "v")


def test_radio_map_trace_runs_with_reflection_and_diffraction_enabled():
    tracer = _build_multipath_tracer(_runtime_scene())
    monitor = RadioMapMonitor(
        "radio_map_multipath",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-4.0, 4.0)),
        grid_shape=(8, 8),
    )
    payload = tracer.trace(wt.Point3f(-5.0, 5.0, 1.5), monitor=monitor, verbose=False)

    assert isinstance(payload, RadioMapResult)
    assert np.asarray(payload.path_gain, dtype=np.float32).shape == payload.tensor_shape
    assert np.isfinite(np.asarray(payload.path_gain, dtype=np.float32)).all()


def test_empty_radio_map_diagnostics_allocates_independent_component_buffers():
    diagnostics = _empty_radio_map_diagnostics(4)

    dr.scatter_reduce(
        dr.ReduceOp.Add,
        diagnostics["incoherent"]["reflection"],
        wt.Float(1.0),
        wt.UInt32(2),
    )
    dr.scatter_reduce(
        dr.ReduceOp.Add,
        diagnostics["coherent"]["reflection"].real,
        wt.Float(2.0),
        wt.UInt32(1),
    )
    dr.eval(
        diagnostics["incoherent"]["reflection"],
        diagnostics["coherent"]["reflection"].real,
    )

    np.testing.assert_allclose(
        np.asarray(diagnostics["incoherent"]["los"], dtype=np.float32),
        np.zeros(4, dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics["incoherent"]["diffraction"], dtype=np.float32),
        np.zeros(4, dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics["coherent"]["los"], dtype=np.complex64),
        np.zeros(4, dtype=np.complex64),
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics["coherent"]["diffraction"], dtype=np.complex64),
        np.zeros(4, dtype=np.complex64),
    )


def test_radio_map_coherent_combine_matches_coherent_path_sum_for_cell_centers():
    tracer = _build_multipath_tracer(_runtime_scene())
    monitor = RadioMapMonitor(
        "radio_map_coherent",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-4.0, 4.0)),
        grid_shape=(4, 4),
        combine_mode="coherent",
        receiver_model="projected_polarized",
    )
    grid = RadioMapGrid.from_monitor(monitor, default_cell_size=tracer.cell_size)
    path_monitor = PathMonitor(
        "radio_map_coherent_baseline",
        positions=torch.as_tensor(_cell_center_positions(grid), dtype=torch.float32),
        max_diffractions=1,
    )
    tx_pos = wt.Point3f(-5.0, 5.0, 1.5)

    payload = tracer.trace(tx_pos, monitor=monitor, verbose=False)
    paths = tracer.trace(tx_pos, monitor=path_monitor, verbose=False)

    np.testing.assert_allclose(
        np.asarray(payload.path_gain, dtype=np.float32),
        _coherent_power_from_paths(paths, tensor_shape=payload.tensor_shape),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(payload.coherent_power["total"], dtype=np.float32),
        np.asarray(payload.path_gain, dtype=np.float32),
        rtol=1e-6,
        atol=1e-7,
    )
    assert payload.combine_mode == "coherent"


def test_radio_map_trace_reuses_diffraction_state_cache_between_traces():
    tracer = Tracer(
        frequency=1.0e9,
        scene=_runtime_scene(),
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=1,
    )
    monitor = RadioMapMonitor(
        "radio_map_cache",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-4.0, 4.0)),
        grid_shape=(8, 8),
    )
    tx_pos = wt.Point3f(-5.0, 5.0, 1.5)

    first_payload = tracer.trace(tx_pos, monitor=monitor, verbose=False)
    second_payload = tracer.trace(tx_pos, monitor=monitor, verbose=False)

    first_reuse = first_payload.metadata["runtime_reuse"]["diffraction_state_prep_cache"]
    second_reuse = second_payload.metadata["runtime_reuse"]["diffraction_state_prep_cache"]

    assert first_reuse["mode"] == "persistent"
    assert int(first_reuse["misses"]) >= 1
    assert second_reuse["mode"] == "persistent"
    assert int(second_reuse["hits"]) >= 1
    assert int(second_reuse["misses"]) == 0


def test_trace_shares_reflection_discovery_between_radio_map_and_path_monitors(monkeypatch):
    scene = build_scene(
        box_geometry(center=(0.0, 0.0, 1.5), size=(0.25, 6.0, 3.0)),
    )
    radio_map = RadioMapMonitor(
        "radio_map_shared_reflection",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=(8, 8),
        ray_mode="3d",
    )
    path = PathMonitor(
        "rx",
        positions=torch.tensor([[-3.0, 5.0, 1.5]], dtype=torch.float32),
        ray_mode="3d",
        max_diffractions=0,
    )
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        reflection_n_rays=4096,
        reflection_max_bounces=1,
        max_diffractions=0,
    )

    original = reflection_field._trace_reflection_paths
    call_count = 0

    def _counted_trace(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(reflection_field, "_trace_reflection_paths", _counted_trace)

    result = tracer.trace((-3.0, -5.0, 1.5), monitor=[radio_map, path], verbose=False)

    assert call_count == 1
    assert int(np.asarray(result["rx"].num_paths, dtype=np.int32)[0]) >= 2


def test_radio_map_path_gain_keeps_tx_gradient_flow_for_multipath_trace():
    tracer = _build_multipath_tracer(_runtime_scene())
    monitor = RadioMapMonitor(
        "radio_map_grad",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-4.0, 4.0)),
        grid_shape=(8, 8),
        receiver_model="projected_polarized",
    )
    tx_pos = wt.Point3f(-5.0, 5.0, 1.5)
    dr.enable_grad(tx_pos)
    dr.set_grad(tx_pos, wt.Vector3f(1.0, 0.0, 0.0))
    solver_controls = tracer._resolve_monitor_solver_controls(
        monitor,
        execution_intent="radio_map_incoherent",
    )

    payload = trace_radio_map_monitor(
        tx_pos,
        monitor,
        tracer.scene,
        tracer._resolved_trace_config,
        solver_controls,
        verbose=False,
        return_timing=False,
    )
    path_gain = payload["metrics"]["path_gain"]
    dr.forward_to(path_gain, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
    grad = np.asarray(to_numpy(dr.grad(path_gain)), dtype=np.float64)

    assert float(np.sum(np.abs(grad))) > 0.0
    assert payload["metadata"]["accumulation_backend"]["resolved"] == "baseline"


def test_radio_map_native_coherent_backend_matches_baseline_for_reflection_and_diffraction():
    if not native_extension_available():
        pytest.skip("Native radio-map parity test requires the bundled native extension.")

    tracer = Tracer(
        frequency=1.0e9,
        scene=_runtime_wall_scene(),
        reflection_n_rays=4096,
        reflection_max_bounces=1,
        max_diffractions=1,
    )
    monitor = RadioMapMonitor(
        "radio_map_native_parity",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=(8, 8),
        combine_mode="coherent",
        receiver_model="projected_polarized",
        ray_mode="3d",
    )
    tx_pos = wt.Point3f(-3.0, -5.0, 1.5)
    solver_controls = tracer._resolve_monitor_solver_controls(
        monitor,
        execution_intent="radio_map_coherent",
    )

    baseline = trace_radio_map_monitor(
        tx_pos,
        monitor,
        tracer.scene,
        tracer._resolved_trace_config,
        solver_controls,
        radio_map_accumulation_backend="baseline",
    )
    native = trace_radio_map_monitor(
        tx_pos,
        monitor,
        tracer.scene,
        tracer._resolved_trace_config,
        solver_controls,
        radio_map_accumulation_backend="native_coherent",
    )

    assert int(baseline["metadata"]["path_counts"]["reflection"]) > 0
    assert int(baseline["metadata"]["path_counts"]["diffraction"]) > 0
    assert int(native["metadata"]["path_counts"]["reflection"]) > 0
    assert int(native["metadata"]["path_counts"]["diffraction"]) > 0
    np.testing.assert_array_less(
        0.0,
        np.asarray(native["diagnostics"]["coherent_power"]["reflection"], dtype=np.float32).max(),
    )
    np.testing.assert_array_less(
        0.0,
        np.asarray(native["diagnostics"]["coherent_power"]["diffraction"], dtype=np.float32).max(),
    )
    np.testing.assert_allclose(
        np.asarray(native["metrics"]["path_gain"], dtype=np.float32),
        np.asarray(baseline["metrics"]["path_gain"], dtype=np.float32),
        rtol=1e-3,
        atol=5e-6,
    )
    assert native["metadata"]["accumulation_backend"]["requested"] == "native_coherent"
    assert native["metadata"]["accumulation_backend"]["resolved"] == "native_coherent"
    assert native["metadata"]["accumulation_backend"]["cell_accumulation_mode"] == "sample_reduction"
    assert native["metadata"]["runtime_backends"]["reflection"]["implementation"] == "native_cuda_custom_op"
    assert native["metadata"]["runtime_backends"]["reflection"]["policy"] == "fresh_trace_native_requested_backend"
    assert native["metadata"]["runtime_backends"]["diffraction"]["implementation"] == "native_cuda_custom_op"


def test_radio_map_native_coherent_component_metrics_keep_tx_gradient_flow():
    if not native_extension_available():
        pytest.skip("Native radio-map gradient test requires the bundled native extension.")

    tracer = Tracer(
        frequency=1.0e9,
        scene=_runtime_wall_scene(),
        reflection_n_rays=2048,
        reflection_max_bounces=1,
        max_diffractions=1,
    )
    monitor = _native_coherent_gradient_monitor(grid_shape=(8, 8))
    weights = _native_coherent_gradient_weights(monitor)

    for target_name in ("path_gain", "reflection", "diffraction", "total"):
        tx_pos = wt.Point3f(-3.0, -5.0, 1.5)
        dr.enable_grad(tx_pos)
        payload = _trace_native_coherent_radio_map(
            tracer=tracer,
            monitor=monitor,
            tx_pos=tx_pos,
        )
        target = (
            payload["metrics"]["path_gain"]
            if target_name == "path_gain"
            else payload["diagnostics"]["coherent_power"][target_name]
        )
        loss = dr.sum(target * weights)
        dr.backward(loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)

        assert _point_grad_abs_sum(tx_pos) > 0.0
        assert payload["metadata"]["accumulation_backend"]["resolved"] == "native_coherent"
        assert payload["metadata"]["runtime_backends"]["reflection"]["implementation"] == "native_cuda_custom_op"
        assert payload["metadata"]["runtime_backends"]["diffraction"]["implementation"] == (
            "native_cuda_custom_op"
        )


def test_radio_map_native_coherent_path_gain_keeps_scene_material_and_geometry_grad_flow():
    if not native_extension_available():
        pytest.skip("Native radio-map gradient test requires the bundled native extension.")

    monitor = _native_coherent_gradient_monitor(grid_shape=(8, 8))
    weights = _native_coherent_gradient_weights(monitor)

    material_scene = build_scene(
        box_geometry(center=(0.0, 0.0, 1.5), size=(0.25, 6.0, 3.0)),
        material=Material(eps_r=4.0),
    )
    tracer_material = Tracer(
        frequency=1.0e9,
        scene=material_scene,
        reflection_n_rays=2048,
        reflection_max_bounces=1,
        max_diffractions=1,
        use_scene_materials_for_reflection=True,
        use_scene_materials_for_diffraction=True,
    )
    dr.enable_grad(material_scene.tri_data_gpu["material_eps_r"])
    material_payload = _trace_native_coherent_radio_map(
        tracer=tracer_material,
        monitor=monitor,
        tx_pos=wt.Point3f(-3.0, -5.0, 1.5),
    )
    material_loss = dr.sum(material_payload["metrics"]["path_gain"] * weights)
    dr.backward(material_loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
    material_grad = np.asarray(
        to_numpy(dr.grad(material_scene.tri_data_gpu["material_eps_r"])),
        dtype=np.float32,
    )

    geometry_scene = _runtime_wall_scene()
    tracer_geometry = Tracer(
        frequency=1.0e9,
        scene=geometry_scene,
        reflection_n_rays=2048,
        reflection_max_bounces=1,
        max_diffractions=1,
        use_scene_materials_for_reflection=True,
        use_scene_materials_for_diffraction=True,
    )
    dr.enable_grad(
        geometry_scene.tri_data_gpu["v0"].x,
        geometry_scene.tri_data_gpu["v0"].y,
        geometry_scene.tri_data_gpu["v0"].z,
    )
    geometry_payload = _trace_native_coherent_radio_map(
        tracer=tracer_geometry,
        monitor=monitor,
        tx_pos=wt.Point3f(-3.0, -5.0, 1.5),
    )
    geometry_loss = dr.sum(geometry_payload["metrics"]["path_gain"] * weights)
    dr.backward(geometry_loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
    geometry_grad = np.asarray(
        to_numpy(dr.grad(geometry_scene.tri_data_gpu["v0"].x)),
        dtype=np.float32,
    )

    assert float(np.sum(np.abs(material_grad))) > 0.0
    assert float(np.sum(np.abs(geometry_grad))) > 0.0
    assert material_payload["metadata"]["runtime_backends"]["reflection"]["implementation"] == (
        "native_cuda_custom_op"
    )
    assert material_payload["metadata"]["runtime_backends"]["diffraction"]["implementation"] == (
        "native_cuda_custom_op"
    )
    assert geometry_payload["metadata"]["runtime_backends"]["reflection"]["implementation"] == (
        "native_cuda_custom_op"
    )
    assert geometry_payload["metadata"]["runtime_backends"]["diffraction"]["implementation"] == (
        "native_cuda_custom_op"
    )


def test_radio_map_native_coherent_timing_reports_state_preparation_seconds():
    if not native_extension_available():
        pytest.skip("Native radio-map timing test requires the bundled native extension.")

    tracer = Tracer(
        frequency=1.0e9,
        scene=_runtime_wall_scene(),
        reflection_n_rays=2048,
        reflection_max_bounces=1,
        max_diffractions=1,
    )
    monitor = _native_coherent_gradient_monitor(grid_shape=(8, 8))
    payload = _trace_native_coherent_radio_map(
        tracer=tracer,
        monitor=monitor,
        tx_pos=wt.Point3f(-3.0, -5.0, 1.5),
        return_timing=True,
    )

    sample_timing = payload["metadata"]["sample_evaluations"][0]["timing"]

    assert "state_preparation_seconds" in sample_timing
    assert float(sample_timing["state_preparation_seconds"]) >= 0.0
    assert float(payload["timing"]["total_seconds"]) >= float(sample_timing["state_preparation_seconds"])


def test_radio_map_cell_accumulation_backend_matches_baseline_for_reflection_and_diffraction():
    tracer = Tracer(
        frequency=1.0e9,
        scene=_runtime_wall_scene(),
        reflection_n_rays=4096,
        reflection_max_bounces=1,
        max_diffractions=1,
    )
    monitor = RadioMapMonitor(
        "radio_map_cell_accumulation_parity",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=(8, 8),
        combine_mode="incoherent",
        receiver_model="projected_polarized",
        ray_mode="3d",
    )
    tx_pos = wt.Point3f(-3.0, -5.0, 1.5)
    solver_controls = tracer._resolve_monitor_solver_controls(
        monitor,
        execution_intent="radio_map_incoherent",
    )

    baseline = trace_radio_map_monitor(
        tx_pos,
        monitor,
        tracer.scene,
        tracer._resolved_trace_config,
        solver_controls,
        radio_map_accumulation_backend="baseline",
    )
    native = trace_radio_map_monitor(
        tx_pos,
        monitor,
        tracer.scene,
        tracer._resolved_trace_config,
        solver_controls,
        radio_map_accumulation_backend="cell_accumulation",
    )

    assert int(baseline["metadata"]["path_counts"]["reflection"]) > 0
    assert int(baseline["metadata"]["path_counts"]["diffraction"]) > 0
    assert int(native["metadata"]["path_counts"]["los"]) == int(
        baseline["metadata"]["path_counts"]["los"]
    )
    assert int(native["metadata"]["path_counts"]["reflection"]) == int(
        baseline["metadata"]["path_counts"]["reflection"]
    )
    assert 0 < int(native["metadata"]["path_counts"]["diffraction"]) <= int(
        baseline["metadata"]["path_counts"]["diffraction"]
    )
    assert int(native["metadata"]["path_counts"]["total"]) == int(
        native["metadata"]["path_counts"]["los"]
        + native["metadata"]["path_counts"]["reflection"]
        + native["metadata"]["path_counts"]["diffraction"]
    )
    np.testing.assert_array_less(
        0.0,
        np.asarray(native["diagnostics"]["incoherent"]["reflection"], dtype=np.float32).max(),
    )
    np.testing.assert_array_less(
        0.0,
        np.asarray(native["diagnostics"]["incoherent"]["diffraction"], dtype=np.float32).max(),
    )
    assert not np.array_equal(
        np.asarray(native["diagnostics"]["incoherent"]["reflection"], dtype=np.float32),
        np.asarray(native["diagnostics"]["incoherent"]["diffraction"], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(native["metrics"]["path_gain"], dtype=np.float32),
        np.asarray(baseline["metrics"]["path_gain"], dtype=np.float32),
        rtol=3e-4,
        atol=5e-6,
    )
    assert native["metadata"]["accumulation_backend"]["requested"] == "cell_accumulation"
    assert native["metadata"]["accumulation_backend"]["resolved"] == "cell_accumulation"
    assert native["metadata"]["accumulation_backend"]["cell_accumulation_mode"] == "direct_in_loop_scatter"


def test_radio_map_baseline_matched_isotropic_matches_cell_accumulation_for_reflection_and_diffraction():
    tracer = Tracer(
        frequency=1.0e9,
        scene=_runtime_wall_scene(),
        reflection_n_rays=4096,
        reflection_max_bounces=1,
        max_diffractions=1,
    )
    monitor = RadioMapMonitor(
        "radio_map_matched_isotropic_parity",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=(8, 8),
        combine_mode="incoherent",
        receiver_model="matched_isotropic",
        ray_mode="3d",
    )
    tx_pos = wt.Point3f(-3.0, -5.0, 1.5)
    solver_controls = tracer._resolve_monitor_solver_controls(
        monitor,
        execution_intent="radio_map_incoherent",
    )

    baseline = trace_radio_map_monitor(
        tx_pos,
        monitor,
        tracer.scene,
        tracer._resolved_trace_config,
        solver_controls,
        radio_map_accumulation_backend="baseline",
    )
    native = trace_radio_map_monitor(
        tx_pos,
        monitor,
        tracer.scene,
        tracer._resolved_trace_config,
        solver_controls,
        radio_map_accumulation_backend="cell_accumulation",
    )

    baseline_runtime_backends = baseline["metadata"]["runtime_backends"]
    assert baseline["metadata"]["accumulation_backend"]["resolved"] == "baseline"
    assert baseline["metadata"]["receiver_model"] == "matched_isotropic"
    assert baseline_runtime_backends["reflection"]["radio_map_power_backend"] == "baseline_vector_power_replay"
    assert baseline_runtime_backends["reflection"]["pair_replay_backend"] == "direct_replay_vector_power"
    assert baseline_runtime_backends["diffraction"]["radio_map_power_backend"] == "baseline_vector_power_replay"
    assert baseline_runtime_backends["diffraction"]["pair_replay_backend"] == "direct_state_vector_power"
    assert baseline_runtime_backends["suffix"]["implementation"] == "disabled"
    assert int(native["metadata"]["path_counts"]["los"]) == int(
        baseline["metadata"]["path_counts"]["los"]
    )
    assert int(native["metadata"]["path_counts"]["reflection"]) == int(
        baseline["metadata"]["path_counts"]["reflection"]
    )
    assert 0 < int(native["metadata"]["path_counts"]["diffraction"]) <= int(
        baseline["metadata"]["path_counts"]["diffraction"]
    )
    assert int(native["metadata"]["path_counts"]["total"]) == int(
        native["metadata"]["path_counts"]["los"]
        + native["metadata"]["path_counts"]["reflection"]
        + native["metadata"]["path_counts"]["diffraction"]
    )
    np.testing.assert_allclose(
        np.asarray(native["metrics"]["path_gain"], dtype=np.float32),
        np.asarray(baseline["metrics"]["path_gain"], dtype=np.float32),
        rtol=3e-4,
        atol=5e-6,
    )


def test_radio_map_three_cube_forward_baseline_matches_cell_accumulation_for_diffraction():
    tracer = Tracer(
        frequency=1.0e9,
        scene=build_scene_for_cube1_x(float(CUBE1_BASE_CENTER[0])),
        reflection_n_rays=512,
        reflection_max_bounces=3,
        reflection_coef=0.8,
        enable_rd_diffraction=True,
        max_diffractions=2,
    )
    monitor = RadioMapMonitor(
        "radio_map_three_cube_matched_isb_parity",
        axis="z",
        position=1.0,
        bounds=((-10.0, 10.0), (-10.0, 10.0)),
        grid_shape=(32, 32),
        combine_mode="coherent",
        receiver_model="matched_isotropic",
        ray_mode="3d",
        shadow_boundary_mode="matched_isb_completion",
        max_diffractions=2,
    )
    tx_pos = wt.Point3f(0.0, -5.0, 4.0)
    solver_controls = tracer._resolve_monitor_solver_controls(
        monitor,
        execution_intent="radio_map_coherent",
    )

    baseline = trace_radio_map_monitor(
        tx_pos,
        monitor,
        tracer.scene,
        tracer._resolved_trace_config,
        solver_controls,
        radio_map_accumulation_backend="baseline",
    )
    cell = trace_radio_map_monitor(
        tx_pos,
        monitor,
        tracer.scene,
        tracer._resolved_trace_config,
        solver_controls,
        radio_map_accumulation_backend="cell_accumulation",
    )

    np.testing.assert_allclose(
        np.asarray(cell["metrics"]["path_gain"], dtype=np.float64),
        np.asarray(baseline["metrics"]["path_gain"], dtype=np.float64),
        rtol=1.0e-6,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        np.asarray(cell["diagnostics"]["coherent_power"]["diffraction"], dtype=np.float64),
        np.asarray(baseline["diagnostics"]["coherent_power"]["diffraction"], dtype=np.float64),
        rtol=1.0e-6,
        atol=1.0e-9,
    )


def test_radio_map_three_cube_baseline_reports_diffraction_diagnostic_counts():
    tracer = Tracer(
        frequency=1.0e9,
        scene=build_scene_for_cube1_x(float(CUBE1_BASE_CENTER[0])),
        reflection_n_rays=256,
        reflection_max_bounces=3,
        reflection_coef=0.8,
        enable_rd_diffraction=True,
        max_diffractions=2,
    )
    monitor = RadioMapMonitor(
        "radio_map_three_cube_diagnostics",
        axis="z",
        position=1.0,
        bounds=((-10.0, 10.0), (-10.0, 10.0)),
        grid_shape=(16, 16),
        combine_mode="coherent",
        receiver_model="matched_isotropic",
        ray_mode="3d",
        shadow_boundary_mode="matched_isb_completion",
        accumulation_backend="baseline",
        max_diffractions=2,
    )

    payload = tracer.trace(
        wt.Point3f(0.0, -5.0, 4.0),
        monitor=monitor,
        verbose=False,
    )

    expected_keys = {
        "prepared_state_count",
        "visible_pair_count",
        "support_pair_count",
        "pair_valid_count",
        "shadow_completion_count",
        "interior_count",
        "hard_visibility_zero_count",
    }
    counts = payload.metadata["diffraction_diagnostics"]
    assert set(counts.keys()) == expected_keys
    assert int(counts["prepared_state_count"]) > 0
    assert int(counts["visible_pair_count"]) >= int(counts["support_pair_count"])
    assert int(counts["support_pair_count"]) >= int(counts["pair_valid_count"])
    assert set(payload.metadata["sample_evaluations"][0]["diffraction_diagnostics"].keys()) == expected_keys
    surrogate_parameters = payload.metadata["shadow_boundary_surrogate"]["parameters"]
    assert surrogate_parameters["incident_weight_aggregation"] == "clamped_sum_incident_weight"
    assert surrogate_parameters["max_incident_weight_usage"] == "diagnostic_only"
    assert "raw_diffraction" in payload.coherent_power
    assert "matched_isb_completion_only" in payload.coherent_power
    assert "folded_diffraction" in payload.coherent_power
    for key in (
        "matched_isb_sum_incident_weight",
        "matched_isb_max_incident_weight",
        "matched_isb_argmax_margin",
        "matched_isb_support_edge_count",
        "matched_isb_argmax_edge_idx",
    ):
        assert key in payload.incoherent
    completion_weight = np.asarray(payload.incoherent["matched_isb_completion_weight"], dtype=np.float32)
    sum_incident_weight = np.asarray(payload.incoherent["matched_isb_sum_incident_weight"], dtype=np.float32)
    np.testing.assert_allclose(
        completion_weight,
        np.clip(sum_incident_weight, 0.0, 1.0),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_radio_map_trace_supports_explicit_baseline_matched_isotropic_backend():
    tracer = Tracer(
        frequency=1.0e9,
        scene=_runtime_wall_scene(),
        reflection_n_rays=4096,
        reflection_max_bounces=1,
        max_diffractions=1,
    )
    monitor = RadioMapMonitor(
        "radio_map_baseline_matched_isotropic_public",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=(8, 8),
        combine_mode="incoherent",
        receiver_model="matched_isotropic",
        accumulation_backend="baseline",
        ray_mode="3d",
    )

    payload = tracer.trace(
        wt.Point3f(-3.0, -5.0, 1.5),
        monitor=monitor,
        verbose=False,
    )

    assert payload.metadata["accumulation_backend"]["resolved"] == "baseline"
    assert payload.receiver_model == "matched_isotropic"
    assert payload.metadata["runtime_backends"]["reflection"]["radio_map_power_backend"] == "baseline_vector_power_replay"
    assert payload.metadata["runtime_backends"]["diffraction"]["pair_replay_backend"] == "direct_state_vector_power"


def test_radio_map_monte_carlo_los_matches_deterministic_reference_on_empty_scene():
    if not native_extension_available():
        pytest.skip("Monte Carlo radio-map test requires the bundled native extension.")

    tracer = Tracer(
        frequency=1.0e9,
        scene=build_scene(),
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=0,
    )
    baseline_monitor = RadioMapMonitor(
        "radio_map_monte_carlo_los_reference",
        axis="z",
        position=1.5,
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        grid_shape=(4, 4),
        combine_mode="incoherent",
        receiver_model="matched_isotropic",
        accumulation_backend="baseline",
        quadrature_mode="stratified_fixed_n",
        samples_per_cell=16,
        max_diffractions=0,
    )
    monte_carlo_monitor = RadioMapMonitor(
        "radio_map_monte_carlo_los",
        axis="z",
        position=1.5,
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        grid_shape=(4, 4),
        combine_mode="incoherent",
        receiver_model="matched_isotropic",
        accumulation_backend="native_monte_carlo",
        sampling_mode="monte_carlo",
        samples_per_tx=262144,
        max_diffractions=0,
        seed=7,
    )
    tx_pos = wt.Point3f(0.25, -0.15, 3.0)

    baseline = tracer.trace(tx_pos, monitor=baseline_monitor, verbose=False)
    monte_carlo = tracer.trace(tx_pos, monitor=monte_carlo_monitor, verbose=False)

    np.testing.assert_allclose(
        np.asarray(monte_carlo.path_gain, dtype=np.float32),
        np.asarray(baseline.path_gain, dtype=np.float32),
        rtol=0.6,
        atol=2e-6,
    )
    assert len(monte_carlo.sample_positions()) == 0
    assert monte_carlo.metadata["sampling_mode"] == "monte_carlo"
    assert monte_carlo.metadata["accumulation_backend"]["resolved"] == "native_monte_carlo"
    assert monte_carlo.metadata["monte_carlo"]["accepted_hit_counts"]["los"] > 0
    assert monte_carlo.metadata["ray_sampling"]["batch_size"] == 262144
    assert monte_carlo.metadata["ray_sampling"]["batch_count"] == 1
    assert monte_carlo.metadata["path_counts"]["reflection"] == 0
    assert monte_carlo.metadata["path_counts"]["diffraction"] == 0


def test_radio_map_monte_carlo_uses_single_full_batch_without_diffraction():
    if not native_extension_available():
        pytest.skip("Monte Carlo radio-map test requires the bundled native extension.")

    tracer = Tracer(
        frequency=1.0e9,
        scene=build_scene(),
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=0,
    )
    monitor = RadioMapMonitor(
        "radio_map_monte_carlo_full_batch_no_diffraction",
        axis="z",
        position=1.5,
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        grid_shape=(4, 4),
        combine_mode="incoherent",
        receiver_model="matched_isotropic",
        accumulation_backend="native_monte_carlo",
        sampling_mode="monte_carlo",
        samples_per_tx=1024,
        max_diffractions=0,
        seed=11,
    )

    payload = tracer.trace(
        wt.Point3f(0.25, -0.15, 3.0),
        monitor=monitor,
        verbose=False,
    )

    ray_sampling = payload.metadata["ray_sampling"]
    monte_carlo = payload.metadata["monte_carlo"]
    assert ray_sampling["batch_size"] == 1024
    assert ray_sampling["batch_count"] == 1
    assert str(ray_sampling["batch_policy"]).startswith("full_batch")
    assert monte_carlo["samples_per_tx"] == 1024
    assert ray_sampling["diffraction_batch_size"] == 0
    assert ray_sampling["diffraction_batch_count"] == 0


def test_radio_map_monte_carlo_reports_reflection_and_diffraction_hits_on_wall_scene():
    if not native_extension_available():
        pytest.skip("Monte Carlo radio-map test requires the bundled native extension.")

    tracer = Tracer(
        frequency=1.0e9,
        scene=_runtime_wall_scene(),
        reflection_n_rays=4096,
        reflection_max_bounces=1,
        max_diffractions=1,
    )
    monitor = RadioMapMonitor(
        "radio_map_monte_carlo_wall",
        axis="z",
        position=1.0,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=(8, 8),
        combine_mode="incoherent",
        receiver_model="matched_isotropic",
        accumulation_backend="native_monte_carlo",
        sampling_mode="monte_carlo",
        samples_per_tx=65536,
        seed=3,
    )

    payload = tracer.trace(
        wt.Point3f(-3.0, -5.0, 1.5),
        monitor=monitor,
        verbose=False,
    )

    assert payload.metadata["accumulation_backend"]["resolved"] == "native_monte_carlo"
    assert payload.metadata["runtime_backends"]["reflection"]["implementation"] == (
        "tx_emitted_rays_drjit_symbolic_loop_plus_scatter_reduce"
    )
    assert payload.metadata["runtime_backends"]["reflection"]["cell_scatter_backend"] == (
        "drjit_scatter_reduce"
    )
    assert payload.metadata["runtime_backends"]["diffraction"]["cell_scatter_backend"] == (
        "drjit_scatter_reduce"
    )
    assert payload.metadata["monte_carlo"]["accepted_hit_counts"]["reflection"] > 0
    assert payload.metadata["monte_carlo"]["accepted_hit_counts"]["diffraction"] > 0
    np.testing.assert_array_less(
        0.0,
        np.asarray(payload.incoherent["reflection"], dtype=np.float32).max(),
    )
    np.testing.assert_array_less(
        0.0,
        np.asarray(payload.incoherent["diffraction"], dtype=np.float32).max(),
    )


def test_radio_map_monte_carlo_diffraction_uses_in_loop_state_store(monkeypatch):
    if not native_extension_available():
        pytest.skip("Monte Carlo radio-map test requires the bundled native extension.")

    tracer = Tracer(
        frequency=1.0e9,
        scene=_runtime_wall_scene(),
        reflection_n_rays=4096,
        reflection_max_bounces=1,
        max_diffractions=1,
    )
    monitor = RadioMapMonitor(
        "radio_map_monte_carlo_wall_in_loop_state_store",
        axis="z",
        position=1.0,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=(8, 8),
        combine_mode="incoherent",
        receiver_model="matched_isotropic",
        accumulation_backend="native_monte_carlo",
        sampling_mode="monte_carlo",
        samples_per_tx=65536,
        seed=3,
    )

    original_build = (
        radio_map_monte_carlo_trace_module.mc_diff.DirectTxDiffractionStates.from_edge_indices
    )
    call_count = 0

    def _counted_build(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(
        radio_map_monte_carlo_trace_module.mc_diff.DirectTxDiffractionStates,
        "from_edge_indices",
        _counted_build,
    )

    payload = tracer.trace(
        wt.Point3f(-3.0, -5.0, 1.5),
        monitor=monitor,
        verbose=False,
    )

    assert call_count == 0
    assert payload.metadata["runtime_reuse"]["diffraction_state_prep_cache"]["state_layout"] == (
        "depth0_in_loop_state_store"
    )
    assert payload.metadata["runtime_backends"]["diffraction"]["implementation"] == (
        "depth0_in_loop_wedge_state_store_plus_keller_cone_symbolic_loop_scatter_reduce"
    )
    assert payload.metadata["monte_carlo"]["accepted_hit_counts"]["diffraction"] > 0


def test_radio_map_monte_carlo_three_cube_reflection_preserves_specular_support():
    if not native_extension_available():
        pytest.skip("Monte Carlo radio-map test requires the bundled native extension.")

    tracer = Tracer(
        frequency=1.0e9,
        scene=_runtime_three_cube_scene(),
        reflection_n_rays=20_000,
        reflection_max_bounces=1,
        reflection_coef=0.8,
        enable_rd_diffraction=False,
        max_diffractions=0,
    )
    monitor = RadioMapMonitor(
        "radio_map_monte_carlo_three_cube_reflection",
        axis="z",
        position=1.0,
        bounds=((-10.0, 10.0), (-10.0, 10.0)),
        grid_shape=(256, 256),
        combine_mode="incoherent",
        receiver_model="matched_isotropic",
        accumulation_backend="native_monte_carlo",
        ray_mode="3d",
        max_diffractions=0,
        sampling_mode="monte_carlo",
        samples_per_tx=20_000,
        seed=7,
    )

    payload = tracer.trace(
        wt.Point3f(0.0, -5.0, 4.0),
        monitor=monitor,
        verbose=False,
    )

    reflection = np.asarray(payload.incoherent["reflection"], dtype=np.float32)
    assert payload.metadata["monte_carlo"]["accepted_hit_counts"]["reflection"] >= 500
    assert int(np.count_nonzero(reflection > 0.0)) >= 500


def test_radio_map_monte_carlo_auto_grad_sensitive_uses_native_backend():
    if not native_extension_available():
        pytest.skip("Monte Carlo AD radio-map test requires the bundled native extension.")

    tx_x = wt.Float(0.0)
    dr.enable_grad(tx_x)
    tracer = _monte_carlo_ad_tracer(_runtime_three_cube_scene())
    monitor = _monte_carlo_ad_monitor(grid_shape=(16, 16), samples_per_tx=32)

    payload = tracer.trace(
        wt.Point3f(tx_x, -5.0, 4.0),
        monitor=monitor,
        verbose=False,
    )

    assert payload.metadata["accumulation_backend"]["resolved"] == "native_monte_carlo"
    assert payload.metadata["receiver_sampling"]["strategy"] == "tx_emitted_rays"
    assert payload.metadata["monte_carlo"]["ad_mode"] is True
    assert payload.metadata["monte_carlo"]["ad_backend"] == "outer_custom_op_native_sparse_coeff_cuda"
    assert (
        payload.metadata["monte_carlo"]["tape_layout_version"]
        == "single_solver_native_sparse_coeff_tape_v3"
    )
    assert (
        payload.metadata["monte_carlo"]["tx_accumulation_transport_mode"]
        == "fixed_tape_full_replay_central_difference"
    )
    assert float(payload.metadata["monte_carlo"]["tx_accumulation_transport_step"]) > 0.0
    assert len(payload.sample_positions()) == 0
    assert np.isfinite(np.asarray(payload.path_gain, dtype=np.float32)).all()


def test_radio_map_monte_carlo_symbolic_intersection_preserves_geometry_gradients():
    cube1_x = wt.Float(CUBE1_BASE_CENTER[0])
    dr.enable_grad(cube1_x)
    scene = build_scene_for_cube1_x(cube1_x)
    si = scene.ray_intersect(
        rayd.Ray(
            wt.Point3f(wt.Float(-6.0), wt.Float(-3.0), wt.Float(1.5)),
            wt.Vector3f(wt.Float(1.0), wt.Float(0.0), wt.Float(0.0)),
        ),
        active=wt.Bool(True),
    )

    assert bool(np.asarray(si.is_valid(), dtype=np.bool_).reshape(-1)[0])
    assert dr.grad_enabled(si.t)
    assert dr.grad_enabled(si.p.x)
    assert dr.grad_enabled(si.p.y)
    assert dr.grad_enabled(si.p.z)

    dr.backward(dr.sum(si.t))

    assert abs(float(np.asarray(dr.grad(cube1_x), dtype=np.float64).reshape(-1)[0])) > 0.0


def test_rayd_direct_ad_mesh_preserves_symbolic_reflection_gradients():
    verts = wt.Point3f(
        wt.Float([0.0, 1.0, 0.0]),
        wt.Float([0.0, 0.0, 1.0]),
        wt.Float([0.0, 0.0, 0.0]),
    )
    dr.enable_grad(verts.x, verts.y, verts.z)
    mesh = rayd.Mesh(
        verts,
        wt.Vector3u(wt.UInt32([0]), wt.UInt32([1]), wt.UInt32([2])),
    )
    rayd_scene = rayd.Scene()
    rayd_scene.add_mesh(mesh, dynamic=True)
    rayd_scene.build()
    rayd_scene.sync()

    ray = rayd.Ray(
        wt.Point3f(wt.Float([0.25]), wt.Float([0.25]), wt.Float([-1.0])),
        wt.Vector3f(wt.Float([0.0]), wt.Float([0.0]), wt.Float([1.0])),
    )
    si = rayd_scene.intersect(ray)
    chain = rayd_scene.trace_reflections(ray, 1, False, True)
    bounce = chain.bounce(0)

    assert dr.grad_enabled(si.t)
    assert dr.grad_enabled(bounce.image_sources.x)
    assert dr.grad_enabled(bounce.image_sources.y)
    assert dr.grad_enabled(bounce.image_sources.z)

    loss = (
        dr.sum(si.t)
        + dr.sum(bounce.image_sources.x)
        + dr.sum(bounce.image_sources.y)
        + dr.sum(bounce.image_sources.z)
    )
    dr.backward(loss)

    grad_z = np.asarray(dr.grad(verts.z), dtype=np.float64)
    assert float(np.sum(np.abs(grad_z))) > 0.0


def test_radio_map_monte_carlo_native_forward_gradients_for_tx_x():
    if not native_extension_available():
        pytest.skip("Monte Carlo AD radio-map test requires the bundled native extension.")

    payload, gradients = _trace_three_cube_monte_carlo_ad_gradients("tx_x")

    assert payload.metadata["accumulation_backend"]["resolved"] == "native_monte_carlo"
    assert payload.metadata["monte_carlo"]["ad_mode"] is True
    assert payload.metadata["monte_carlo"]["ad_backend"] == "outer_custom_op_native_sparse_coeff_cuda"
    assert np.isfinite(np.asarray(payload.path_gain, dtype=np.float32)).all()
    assert gradients["path_gain"] > 0.0
    assert gradients["reflection"] > 0.0
    assert gradients["diffraction"] > 0.0


def test_radio_map_monte_carlo_native_forward_gradients_for_cube1_x():
    if not native_extension_available():
        pytest.skip("Monte Carlo AD radio-map test requires the bundled native extension.")

    payload, gradients = _trace_three_cube_monte_carlo_ad_gradients("cube1_x")

    assert payload.metadata["accumulation_backend"]["resolved"] == "native_monte_carlo"
    assert payload.metadata["monte_carlo"]["ad_mode"] is True
    assert payload.metadata["monte_carlo"]["ad_backend"] == "outer_custom_op_native_sparse_coeff_cuda"
    assert np.isfinite(np.asarray(payload.path_gain, dtype=np.float32)).all()
    assert gradients["path_gain"] > 0.0
    assert np.isfinite(float(gradients["reflection"]))
    assert gradients["diffraction"] > 0.0


def test_radio_map_monte_carlo_explicit_ad_false_keeps_native_solver():
    if not native_extension_available():
        pytest.skip("Monte Carlo AD radio-map test requires the bundled native extension.")

    tx_x = wt.Float(0.0)
    dr.enable_grad(tx_x)
    tracer = _monte_carlo_ad_tracer(_runtime_three_cube_scene())
    monitor = _monte_carlo_ad_monitor(
        grid_shape=(16, 16),
        samples_per_tx=32,
        ad=False,
    )

    payload = tracer.trace(
        wt.Point3f(tx_x, -5.0, 4.0),
        monitor=monitor,
        verbose=False,
    )

    assert payload.metadata["accumulation_backend"]["resolved"] == "native_monte_carlo"
    assert payload.metadata["monte_carlo"]["ad_mode"] is False
    assert payload.metadata["monte_carlo"]["ad_backend"] == "disabled"
    assert payload.metadata["monte_carlo"]["tape_layout_version"] == "disabled"
    assert payload.metadata["receiver_sampling"]["strategy"] == "tx_emitted_rays"
    assert np.isfinite(np.asarray(payload.path_gain, dtype=np.float32)).all()


def test_radio_map_monte_carlo_ad_true_max_diffractions_zero_keeps_los_reflection_gradients():
    if not native_extension_available():
        pytest.skip("Monte Carlo AD radio-map test requires the bundled native extension.")

    tx_x = wt.Float(0.0)
    dr.enable_grad(tx_x)
    tracer = Tracer(
        frequency=1.0e9,
        scene=_runtime_three_cube_scene(),
        reflection_n_rays=256,
        reflection_max_bounces=1,
        reflection_coef=0.8,
        enable_rd_diffraction=True,
        max_diffractions=0,
    )
    monitor = _monte_carlo_ad_monitor(
        grid_shape=(16, 16),
        samples_per_tx=96,
        ad=True,
        max_diffractions=0,
    )

    payload = tracer.trace(
        wt.Point3f(tx_x, -5.0, 4.0),
        monitor=monitor,
        verbose=False,
    )
    dr.set_grad(tx_x, 1.0)
    path_gain_grad = dr.forward_to(payload.path_gain, flags=_MC_AD_FLAGS)

    assert payload.metadata["accumulation_backend"]["resolved"] == "native_monte_carlo"
    assert payload.metadata["monte_carlo"]["ad_mode"] is True
    assert payload.metadata["monte_carlo"]["ad_backend"] == "outer_custom_op_native_sparse_coeff_cuda"
    assert payload.metadata["path_counts"]["diffraction"] == 0
    assert float(np.sum(np.abs(np.asarray(path_gain_grad, dtype=np.float64)))) > 0.0


def test_radio_map_monte_carlo_diffraction_replay_matches_primal_forward():
    summary = _monte_carlo_diffraction_replay_alignment_summary(samples_per_tx=512)

    assert summary["tape_width"] > 0
    assert summary["max_abs_diff"] < 1.0e-7
    assert summary["sum_abs_diff"] < 1.0e-6
    assert abs(summary["replay_sum"] - summary["primal_sum"]) < 1.0e-7


def test_radio_map_monte_carlo_native_backend_accepts_grad_sensitive_workload():
    tx_x = wt.Float(0.0)
    dr.enable_grad(tx_x)
    tracer = _monte_carlo_ad_tracer(_runtime_three_cube_scene())
    monitor = _monte_carlo_ad_monitor(
        accumulation_backend="native_monte_carlo",
        grid_shape=(16, 16),
        samples_per_tx=32,
    )

    payload = tracer.trace(
        wt.Point3f(tx_x, -5.0, 4.0),
        monitor=monitor,
        verbose=False,
    )

    assert payload.metadata["accumulation_backend"]["resolved"] == "native_monte_carlo"
    assert payload.metadata["monte_carlo"]["ad_mode"] is True
    assert payload.metadata["monte_carlo"]["ad_backend"] == "outer_custom_op_native_sparse_coeff_cuda"


def test_radio_map_cell_accumulation_respects_zero_diffraction_override():
    tracer = Tracer(
        frequency=1.0e9,
        scene=_runtime_wall_scene(),
        reflection_n_rays=0,
        reflection_max_bounces=0,
        max_diffractions=0,
    )
    monitor = RadioMapMonitor(
        "radio_map_cell_accumulation_no_diffraction",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=(8, 8),
        combine_mode="incoherent",
        receiver_model="projected_polarized",
        accumulation_backend="cell_accumulation",
        max_diffractions=0,
    )

    payload = tracer.trace(
        wt.Point3f(-3.0, -5.0, 1.5),
        monitor=monitor,
        verbose=False,
    )

    assert payload.metadata["accumulation_backend"]["resolved"] == "cell_accumulation"
    assert payload.metadata["path_counts"]["reflection"] == 0
    assert payload.metadata["path_counts"]["diffraction"] == 0
    assert payload.metadata["path_counts"]["total"] == payload.metadata["path_counts"]["los"]


def test_radio_map_shadow_boundary_surrogate_reports_adjusted_path_gain():
    tracer = Tracer(
        frequency=1.0e9,
        scene=_runtime_wall_scene(),
        reflection_n_rays=4096,
        reflection_max_bounces=1,
        max_diffractions=1,
    )
    tx_pos = wt.Point3f(-3.0, -5.0, 1.5)
    raw_monitor = RadioMapMonitor(
        "radio_map_shadow_boundary_raw",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=(16, 16),
        combine_mode="incoherent",
        receiver_model="matched_isotropic",
        ray_mode="3d",
    )
    surrogate_monitor = RadioMapMonitor(
        "radio_map_shadow_boundary_surrogate",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=(16, 16),
        combine_mode="incoherent",
        receiver_model="matched_isotropic",
        ray_mode="3d",
        shadow_boundary_mode="utd_cross_term_surrogate",
    )

    raw_payload = tracer.trace(tx_pos, monitor=raw_monitor, verbose=False)
    surrogate_payload = tracer.trace(tx_pos, monitor=surrogate_monitor, verbose=False)

    surrogate_total = np.asarray(
        surrogate_payload.incoherent["utd_surrogate_total"],
        dtype=np.float32,
    )
    raw_total = np.asarray(
        surrogate_payload.incoherent["total"],
        dtype=np.float32,
    )
    incident_cross = np.asarray(
        surrogate_payload.incoherent["utd_surrogate_incident_cross"],
        dtype=np.float32,
    )
    reflection_cross = np.asarray(
        surrogate_payload.incoherent["utd_surrogate_reflection_cross"],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        np.asarray(surrogate_payload.path_gain, dtype=np.float32),
        surrogate_total,
        rtol=1e-6,
        atol=1e-7,
    )
    assert surrogate_payload.metadata["shadow_boundary_surrogate"]["enabled"] is True
    assert surrogate_payload.metadata["shadow_boundary_surrogate"]["mode"] == "utd_cross_term_surrogate"
    assert float(np.max(np.abs(incident_cross))) > 0.0
    assert float(np.max(np.abs(reflection_cross))) > 0.0
    assert float(np.max(np.abs(surrogate_total - raw_total))) > 0.0
    assert float(
        np.max(
            np.abs(
                np.asarray(surrogate_payload.path_gain, dtype=np.float32)
                - np.asarray(raw_payload.path_gain, dtype=np.float32)
            )
        )
    ) > 0.0


def test_radio_map_matched_isotropic_coherent_uses_vector_coherent_path_gain():
    tracer = Tracer(
        frequency=1.0e9,
        scene=_runtime_wall_scene(),
        reflection_n_rays=4096,
        reflection_max_bounces=1,
        max_diffractions=1,
    )
    tx_pos = wt.Point3f(-3.0, -5.0, 1.5)
    incoherent_monitor = RadioMapMonitor(
        "radio_map_matched_incoherent",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=(16, 16),
        combine_mode="incoherent",
        receiver_model="matched_isotropic",
        ray_mode="3d",
    )
    coherent_monitor = RadioMapMonitor(
        "radio_map_matched_coherent",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=(16, 16),
        combine_mode="coherent",
        receiver_model="matched_isotropic",
        ray_mode="3d",
    )

    incoherent_payload = tracer.trace(tx_pos, monitor=incoherent_monitor, verbose=False)
    coherent_payload = tracer.trace(tx_pos, monitor=coherent_monitor, verbose=False)

    coherent_path_gain = np.asarray(coherent_payload.path_gain, dtype=np.float32)
    coherent_total = np.asarray(coherent_payload.coherent_power["total"], dtype=np.float32)
    incoherent_path_gain = np.asarray(incoherent_payload.path_gain, dtype=np.float32)

    np.testing.assert_allclose(
        coherent_path_gain,
        coherent_total,
        rtol=1e-6,
        atol=1e-7,
    )
    assert (
        coherent_payload.metadata["metric_contract"]["path_gain"]
        == "squared_norm_of_matched_isotropic_vector_coherent_sum_weighted_by_fixed_cell_quadrature"
    )
    assert (
        coherent_payload.metadata["accumulation_backend"]["resolved"]
        == "cell_accumulation"
    )
    assert (
        coherent_payload.metadata["runtime_backends"]["diffraction"]["pair_replay_backend"]
        == "direct_state_vector_power"
    )
    assert float(np.max(np.abs(coherent_path_gain - incoherent_path_gain))) > 0.0


def test_radio_map_matched_isb_completion_reduces_x4_jump_without_mesh_leak():
    tracer = Tracer(
        frequency=1.0e9,
        scene=_rotated_isb_scene(),
        reflection_n_rays=8192,
        reflection_max_bounces=1,
        reflection_coef=1.0,
        enable_rd_diffraction=True,
        max_diffractions=2,
        use_scene_materials_for_reflection=True,
        use_scene_materials_for_diffraction=True,
    )
    case = _run_matched_isb_completion_case(
        tracer=tracer,
        tx_pos=wt.Point3f(-5.0, 5.0, 1.5),
        plane_z=1.5,
        grid_shape=(128, 128),
    )

    np.testing.assert_allclose(
        case["completion_path_gain"],
        case["completion_surrogate_total"],
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        case["completion_path_gain"],
        np.asarray(case["completion_payload"].coherent_power["total"], dtype=np.float32),
        rtol=1e-6,
        atol=1e-7,
    )
    assert case["completion_payload"].metadata["shadow_boundary_surrogate"]["enabled"] is True
    assert case["completion_payload"].metadata["shadow_boundary_surrogate"]["mode"] == "matched_isb_completion"
    assert case["completion_payload"].metadata["shadow_boundary_surrogate"]["completion_model"] == (
        "matched_isotropic_isb_scene_edge_complex_transition_residual_completion"
    )
    assert case["completion_payload"].metadata["shadow_boundary_surrogate"]["visibility_model"] == (
        "scene_edge_incident_transition_weighted_average_with_incident_gated_direct_mode_residual_matching"
    )
    assert (
        case["completion_payload"].metadata["metric_contract"]["path_gain"]
        == "squared_norm_of_matched_isotropic_vector_coherent_sum_plus_isb_visibility_completion_weighted_by_fixed_cell_quadrature"
    )
    assert float(np.max(case["completion_weight"])) > 0.0
    assert float(np.max(case["completion_only"][~case["inside_mask_np"]])) > 0.0
    assert float(np.max(case["completion_only"][case["inside_mask_np"]])) < 1.0e-14
    assert float(np.min(case["completion_hard_visibility"])) >= 0.0
    assert float(np.max(case["completion_hard_visibility"])) <= 1.0
    assert float(np.min(case["completion_transition_magnitude"])) >= 0.0
    # The finite-wedge completion diagnostic stores the magnitude of the
    # aggregated incident transition response. It is not normalized to [0, 1].
    assert float(np.max(case["completion_transition_magnitude"])) > 0.0
    assert np.isfinite(case["completion_transition_phase"]).all()
    assert abs(float(case["completion_x4_jump"])) < abs(float(case["raw_x4_jump"])) - 2.0
    assert abs(float(case["completion_yneg4_jump"])) < abs(float(case["raw_yneg4_jump"])) - 1.0


def test_radio_map_matched_isb_completion_uses_finite_wedge_bounds():
    tracer_kwargs = {
        "frequency": 1.0e9,
        "reflection_n_rays": 2048,
        "reflection_max_bounces": 1,
        "reflection_coef": 1.0,
        "enable_rd_diffraction": True,
        "max_diffractions": 2,
        "use_scene_materials_for_reflection": True,
        "use_scene_materials_for_diffraction": True,
    }
    tracer = Tracer(
        scene=_rotated_isb_scene(),
        **tracer_kwargs,
    )

    case = _run_matched_isb_completion_case(
        tracer=tracer,
        tx_pos=wt.Point3f(-5.0, 5.0, 0.5),
        plane_z=0.5,
        grid_shape=(64, 64),
    )

    np.testing.assert_allclose(
        case["completion_path_gain"],
        case["completion_surrogate_total"],
        rtol=1e-6,
        atol=1e-7,
    )
    assert np.isfinite(case["completion_path_gain"]).all()
    assert float(np.max(case["completion_only"][case["inside_mask_np"]])) < 1.0e-14
    assert abs(float(case["completion_x4_jump"])) < abs(float(case["raw_x4_jump"])) - 1.0
    assert float(np.max(case["completion_weight"])) <= 1.0 + 1.0e-6
    assert float(np.max(case["completion_weight"])) > 0.0
    assert float(np.max(case["completion_only"][~case["inside_mask_np"]])) > 1.0e-10


def test_shadow_boundary_transition_responses_require_visible_exterior_wedge_support():
    batch_state = {
        "edge_dir": wt.Vector3f(wt.Float(0.0), wt.Float(0.0), wt.Float(1.0)),
        "edge_pos": wt.Point3f(wt.Float(0.0), wt.Float(0.0), wt.Float(0.0)),
        "source_pos": wt.Point3f(-1.0, 1.0, 0.0),
        "n0": wt.Vector3f(wt.Float(1.0), wt.Float(0.0), wt.Float(0.0)),
        "n_face_n": wt.Vector3f(wt.Float(0.0), wt.Float(1.0), wt.Float(0.0)),
        "wedge_n": wt.Float(1.5),
        "edge_line_min": wt.Float(-10.0),
        "edge_line_max": wt.Float(10.0),
        "source_visible": wt.Bool(True),
    }
    exterior_rx = wt.Point3f(wt.Float(1.0), wt.Float(-1.0), wt.Float(0.0))
    interior_rx = wt.Point3f(wt.Float(-1.0), wt.Float(-1.0), wt.Float(0.0))

    incident_transition, _, incident_weight, _ = _shadow_boundary_transition_responses(
        batch_state,
        exterior_rx,
        k=wt.Float(2.0 * np.pi),
    )
    hidden_transition, _, hidden_weight, _ = _shadow_boundary_transition_responses(
        dict(batch_state, source_visible=wt.Bool(False)),
        exterior_rx,
        k=wt.Float(2.0 * np.pi),
    )
    interior_transition, _, interior_weight, _ = _shadow_boundary_transition_responses(
        batch_state,
        interior_rx,
        k=wt.Float(2.0 * np.pi),
    )

    assert float(np.max(np.asarray(incident_weight, dtype=np.float32))) > 0.0
    assert float(np.max(np.abs(np.asarray(incident_transition, dtype=np.complex64)))) > 0.0
    assert float(np.max(np.asarray(hidden_weight, dtype=np.float32))) == 0.0
    assert float(np.max(np.abs(np.asarray(hidden_transition, dtype=np.complex64)))) == 0.0
    assert float(np.max(np.asarray(interior_weight, dtype=np.float32))) == 0.0
    assert float(np.max(np.abs(np.asarray(interior_transition, dtype=np.complex64)))) == 0.0


def test_radio_map_projected_isb_completion_reduces_x4_jump_without_blowing_up_yneg4():
    tracer = Tracer(
        frequency=1.0e9,
        scene=_rotated_isb_scene(),
        reflection_n_rays=8192,
        reflection_max_bounces=1,
        reflection_coef=1.0,
        enable_rd_diffraction=True,
        max_diffractions=2,
        use_scene_materials_for_reflection=True,
        use_scene_materials_for_diffraction=True,
    )
    case = _run_projected_isb_completion_case(
        tracer=tracer,
        tx_pos=wt.Point3f(-5.0, 5.0, 1.5),
        plane_z=1.5,
        grid_shape=(128, 128),
    )

    np.testing.assert_allclose(
        case["completion_path_gain"],
        case["completion_surrogate_total"],
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        case["completion_path_gain"],
        np.asarray(case["completion_payload"].coherent_power["total"], dtype=np.float32),
        rtol=1e-6,
        atol=1e-7,
    )
    assert case["completion_payload"].metadata["shadow_boundary_surrogate"]["enabled"] is True
    assert case["completion_payload"].metadata["shadow_boundary_surrogate"]["mode"] == "projected_isb_completion"
    assert case["completion_payload"].metadata["shadow_boundary_surrogate"]["completion_model"] == (
        "projected_shadow_side_direct_continuation"
    )
    assert float(np.max(case["completion_weight"])) > 0.0
    assert float(np.max(case["completion_deficiency"])) > 0.0
    assert float(np.max(case["completion_only"][~case["inside_mask_np"]])) > 0.0
    assert float(np.max(case["completion_only"][case["inside_mask_np"]])) < 1.0e-14
    assert abs(float(case["completion_x4_jump"])) < abs(float(case["raw_x4_jump"])) - 2.5
    assert abs(float(case["completion_yneg4_jump"])) < 4.0
    assert abs(float(case["completion_yneg4_jump"])) < abs(float(case["raw_yneg4_jump"])) + 2.5


def test_radio_map_projected_isb_completion_folds_into_diffraction_component_without_mesh_leak():
    tracer = Tracer(
        frequency=1.0e9,
        scene=_rotated_isb_scene(),
        reflection_n_rays=8192,
        reflection_max_bounces=1,
        reflection_coef=1.0,
        enable_rd_diffraction=True,
        max_diffractions=2,
        use_scene_materials_for_reflection=True,
        use_scene_materials_for_diffraction=True,
    )
    case = _run_projected_isb_completion_case(
        tracer=tracer,
        tx_pos=wt.Point3f(-5.0, 5.0, 1.5),
        plane_z=1.5,
        grid_shape=(128, 128),
    )

    raw_payload = case["raw_payload"]
    completion_payload = case["completion_payload"]
    raw_diffraction = _radio_map_complex_grid(raw_payload, "diffraction")
    completion_only = _radio_map_complex_grid(
        completion_payload,
        "projected_isb_completion",
    )
    completion_diffraction = _radio_map_complex_grid(
        completion_payload,
        "diffraction",
    )
    raw_los = _radio_map_complex_grid(raw_payload, "los")
    raw_diffraction_power = np.asarray(
        raw_payload.coherent_power["diffraction"],
        dtype=np.float32,
    ).reshape(raw_payload.tensor_shape)
    completion_diffraction_power = np.asarray(
        completion_payload.coherent_power["diffraction"],
        dtype=np.float32,
    ).reshape(completion_payload.tensor_shape)

    outside_mask = ~case["inside_mask_np"]
    shadow_outside_mask = outside_mask & ((np.abs(raw_los) ** 2) < 1.0e-14)
    boosted_shadow_outside_mask = (
        shadow_outside_mask
        & (completion_diffraction_power > raw_diffraction_power * 4.0 + 1.0e-20)
        & (completion_diffraction_power > 1.0e-12)
    )

    np.testing.assert_allclose(
        completion_diffraction,
        raw_diffraction + completion_only,
        rtol=1e-6,
        atol=1e-7,
    )
    assert float(np.max(np.abs(completion_only[outside_mask]))) > 0.0
    assert float(np.max(np.abs(completion_only[case["inside_mask_np"]]))) < 1.0e-14
    assert float(
        np.max(
            np.abs(
                completion_diffraction_power[case["inside_mask_np"]]
                - raw_diffraction_power[case["inside_mask_np"]]
            )
        )
    ) < 1.0e-14
    assert int(np.count_nonzero(boosted_shadow_outside_mask)) > 0
    assert float(
        np.max(
            completion_diffraction_power[boosted_shadow_outside_mask]
            - raw_diffraction_power[boosted_shadow_outside_mask]
        )
    ) > 1.0e-6


def test_radio_map_projected_isb_completion_stays_finite_and_leak_free_across_3d_plane_tx_sweep():
    tracer = Tracer(
        frequency=1.0e9,
        scene=_rotated_isb_scene(),
        reflection_n_rays=8192,
        reflection_max_bounces=1,
        reflection_coef=1.0,
        enable_rd_diffraction=True,
        max_diffractions=2,
        use_scene_materials_for_reflection=True,
        use_scene_materials_for_diffraction=True,
    )
    for plane_z in (0.5, 1.5, 2.5, 3.5):
        for tx_z in (0.5, 1.5, 2.5, 3.5):
            case = _run_projected_isb_completion_case(
                tracer=tracer,
                tx_pos=wt.Point3f(-5.0, 5.0, float(tx_z)),
                plane_z=float(plane_z),
                grid_shape=(96, 96),
            )
            np.testing.assert_allclose(
                case["completion_path_gain"],
                case["completion_surrogate_total"],
                rtol=1e-6,
                atol=1e-7,
            )
            assert np.isfinite(case["raw_path_gain"]).all()
            assert np.isfinite(case["completion_path_gain"]).all()
            assert np.isfinite(case["completion_only"]).all()
            assert (
                case["completion_payload"].metadata["shadow_boundary_surrogate"]["mode"]
                == "projected_isb_completion"
            )
            assert float(np.max(case["completion_only"][case["inside_mask_np"]])) < 1.0e-14
            assert float(np.max(case["completion_only"][~case["inside_mask_np"]])) > 0.0
            assert abs(float(case["completion_x4_jump"])) < abs(float(case["raw_x4_jump"])) - 2.0


def test_radio_map_trace_auto_selects_native_coherent_backend_for_public_trace():
    if not native_extension_available():
        pytest.skip("Native radio-map backend test requires the bundled native extension.")

    tracer = Tracer(
        frequency=1.0e9,
        scene=_runtime_wall_scene(),
        reflection_n_rays=4096,
        reflection_max_bounces=1,
        max_diffractions=1,
    )
    monitor = RadioMapMonitor(
        "radio_map_native_auto",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=(32, 32),
        combine_mode="coherent",
        receiver_model="projected_polarized",
        ray_mode="3d",
    )

    payload = tracer.trace(
        wt.Point3f(-3.0, -5.0, 1.5),
        monitor=monitor,
        verbose=False,
    )
    runtime_backends = payload.metadata["runtime_backends"]
    cache_report = payload.metadata["runtime_reuse"]["diffraction_state_prep_cache"]

    assert payload.metadata["accumulation_backend"]["resolved"] == "native_coherent"
    assert runtime_backends["reflection"]["requested_backend"] == "native"
    assert runtime_backends["diffraction"]["implementation"] == "native_cuda_custom_op"
    assert runtime_backends["diffraction"]["planner_backend"] == "shadow_band_halfspace_drjit"
    assert runtime_backends["suffix"]["implementation"] == "native_cuda_custom_op"
    assert cache_report["state_layout"] == "full"


def test_radio_map_trace_uses_explicit_cell_accumulation_backend_for_public_trace():
    tracer = Tracer(
        frequency=1.0e9,
        scene=_runtime_wall_scene(),
        reflection_n_rays=4096,
        reflection_max_bounces=1,
        max_diffractions=1,
    )
    monitor = RadioMapMonitor(
        "radio_map_cell_accumulation_auto",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=(8, 8),
        combine_mode="incoherent",
        receiver_model="projected_polarized",
        accumulation_backend="cell_accumulation",
        ray_mode="3d",
    )

    payload = tracer.trace(
        wt.Point3f(-3.0, -5.0, 1.5),
        monitor=monitor,
        verbose=False,
    )
    runtime_backends = payload.metadata["runtime_backends"]
    cache_report = payload.metadata["runtime_reuse"]["diffraction_state_prep_cache"]

    assert payload.metadata["accumulation_backend"]["resolved"] == "cell_accumulation"
    assert payload.metadata["accumulation_backend"]["cell_accumulation_mode"] == "direct_in_loop_scatter"
    assert runtime_backends["reflection"]["radio_map_scalar_power_backend"] == "direct_in_loop_cell_scatter"
    assert runtime_backends["reflection"]["pair_replay_backend"] == "direct_replay_scalar_power"
    assert isinstance(runtime_backends["reflection"]["selected_reason"], str)
    assert runtime_backends["diffraction"]["radio_map_scalar_power_backend"] == "direct_in_loop_cell_scatter"
    assert runtime_backends["diffraction"]["pair_replay_backend"] == "direct_state_scalar_power"
    assert isinstance(runtime_backends["diffraction"]["selected_reason"], str)
    assert runtime_backends["suffix"]["implementation"] == "disabled"
    assert cache_report["state_layout"] == "reduced_v2"


def test_radio_map_trace_auto_selects_cell_accumulation_backend_for_public_trace():
    tracer = Tracer(
        frequency=1.0e9,
        scene=_runtime_wall_scene(),
        reflection_n_rays=4096,
        reflection_max_bounces=1,
        max_diffractions=1,
    )
    monitor = RadioMapMonitor(
        "radio_map_cell_accumulation_auto_default",
        axis="z",
        position=1.5,
        bounds=((-4.0, 4.0), (-6.0, 6.0)),
        grid_shape=(8, 8),
        combine_mode="incoherent",
        ray_mode="3d",
    )

    payload = tracer.trace(
        wt.Point3f(-3.0, -5.0, 1.5),
        monitor=monitor,
        verbose=False,
    )

    assert payload.metadata["accumulation_backend"]["resolved"] == "cell_accumulation"
    assert payload.metadata["execution_intent"]["kind"] == "radio_map_incoherent"


def _three_cube_higher_order_topology_summary(cube1_x: float):
    wavelength = 0.3
    tx_pos = wt.Point3f(0.0, -5.0, 4.0)
    scene = build_scene_for_cube1_x(float(cube1_x))
    _, _, state_arrays, budget_report = _prepare_diffraction_state_arrays(
        tx_pos,
        1.0,
        scene,
        wavelength,
        2.0 * np.pi / wavelength,
        None,
        None,
        64,
        1,
        0.8,
        "3d",
        2,
        retain_cold_metadata=True,
        preserve_higher_order_candidate_topology=True,
    )
    per_order = budget_report["per_order"]
    assert len(per_order) >= 2
    order2_report = per_order[1]
    order_values = np.asarray(state_arrays["order"], dtype=np.uint32).reshape(-1)
    return {
        "direct_states": int(order2_report["direct_states"]),
        "candidate_unique_count": int(order2_report["higher_order_builder"]["candidate_unique_count"]),
        "topology_preserved": bool(order2_report["higher_order_builder"]["topology_preserved"]),
        "order2_state_count": int(np.count_nonzero(order_values == 2)),
    }


def test_radio_map_higher_order_diffraction_topology_stays_fixed_for_three_cube_geometry_fd_probe():
    step = 1.0e-3
    summaries = [
        _three_cube_higher_order_topology_summary(CUBE1_BASE_CENTER[0] - step),
        _three_cube_higher_order_topology_summary(CUBE1_BASE_CENTER[0]),
        _three_cube_higher_order_topology_summary(CUBE1_BASE_CENTER[0] + step),
    ]

    direct_counts = {summary["direct_states"] for summary in summaries}
    candidate_counts = {summary["candidate_unique_count"] for summary in summaries}
    order2_counts = {summary["order2_state_count"] for summary in summaries}

    assert all(summary["topology_preserved"] for summary in summaries)
    assert len(candidate_counts) == 1
    assert len(direct_counts) == 1
    assert len(order2_counts) == 1
