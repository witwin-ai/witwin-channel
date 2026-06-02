"""Integrated visual test for position, rotation, and TX gradients."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import drjit as dr
import numpy as np
import pytest
import torch
import witwin as wt
from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene
from witwin.channel import DEFAULT_VARIANT, Material, FieldMonitor, Tracer, draw_scene, to_numpy, to_power_db
if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

pytestmark = pytest.mark.gpu

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_PATH = OUTPUT_DIR / "position_rotation_tx.png"
POSITION_LEGACY_COMPARE_OUTPUT_PATH = OUTPUT_DIR / "position_rotation_tx_cube_position_legacy_compare.png"
POSITION_MATERIAL_MODEL_COMPARE_OUTPUT_PATH = (
    OUTPUT_DIR / "position_rotation_tx_cube_position_material_model_compare.png"
)
POSITION_SCALARIZATION_COMPARE_OUTPUT_PATH = (
    OUTPUT_DIR / "position_rotation_tx_cube_position_scalarization_compare.png"
)
DEFAULT_MONITOR_NAME = "main_grad_grid"


def _sync_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if hasattr(dr, "sync_thread"):
        dr.sync_thread()


def _flush_gpu_caches() -> None:
    _sync_gpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if hasattr(dr, "flush_malloc_cache"):
        dr.flush_malloc_cache()
    _sync_gpu()


def _capture_trace_summary(
    result,
    scene=None,
    *,
    grid_size: int | None = None,
    include_field_db: bool = False,
    include_edges: bool = False,
    include_metadata: bool = False,
    include_material_flags: bool = False,
    include_jones_x_rel: bool = False,
) -> dict[str, object]:
    summary: dict[str, object] = {}
    if include_field_db:
        if grid_size is None:
            raise ValueError("grid_size is required when include_field_db=True")
        summary["field_db"] = np.asarray(
            to_numpy(to_power_db(result.primary.field.total)),
            dtype=np.float64,
        ).reshape(grid_size, grid_size)
    if include_edges:
        if scene is None:
            raise ValueError("scene is required when include_edges=True")
        summary["edges_2d"] = scene.get_edge_data(result.primary.plane_position)["edges_2d"]
    if include_metadata:
        summary["metadata"] = dict(result.primary.metadata)
    if include_material_flags:
        if scene is None:
            raise ValueError("scene is required when include_material_flags=True")
        summary["material_has_specified_materials"] = bool(scene.tri_data_gpu["material_has_specified_materials"])
        summary["material_n_default_material_triangles"] = int(
            scene.tri_data_gpu["material_n_default_material_triangles"]
        )
    if include_jones_x_rel:
        summary["jones_x_rel_l2"] = _jones_x_rel_l2(result)
    return summary


def _scalar_height(value) -> float:
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            pass
    try:
        return float(value[0])
    except (TypeError, ValueError, IndexError, KeyError):
        return float(value)


def _monitor_height(tx_pos) -> float:
    if hasattr(tx_pos, "z"):
        return _scalar_height(tx_pos.z)
    return _scalar_height(tx_pos[2])


def _assert_plane_monitor_result(result, monitor: FieldMonitor) -> None:
    payload = result
    if hasattr(result, "monitor"):
        payload = result.monitor(monitor.name)
    sampling = payload.metadata["receiver_sampling"]
    assert sampling["sample_positions"] == "boundary_points"
    assert sampling["index_partitioning"] == "span_over_n_bins"
    assert tuple(payload.grid_shape) == tuple(monitor.grid_shape)
    assert tuple(payload.range_x) == monitor.bounds[0]
    assert tuple(payload.range_y) == monitor.bounds[1]
    assert abs(float(payload.plane_position) - float(monitor.position)) < 1e-6


def _uniform_override_material_dict(*, eps_r: float, gain: float, sigma_e: float = 0.0) -> dict[str, float]:
    return {
        "relative_permittivity": float(eps_r),
        "conductivity": float(sigma_e),
        "gain": float(gain),
    }


def _complex_component_to_numpy(component) -> np.ndarray:
    return np.asarray(to_numpy(component.real), dtype=np.float64) + 1j * np.asarray(
        to_numpy(component.imag),
        dtype=np.float64,
    )


def _jones_x_rel_l2(result) -> float:
    scalar_total = _complex_component_to_numpy(result.primary.field.total)
    jones_x_total = _complex_component_to_numpy(result.primary.jones.total["x"])
    return _relative_l2_error(scalar_total, jones_x_total)


def _field_component(result, field_mode: str):
    if field_mode == "total":
        return result.primary.field.total
    if field_mode == "reflection":
        return result.primary.field.reflection
    if field_mode == "diffraction":
        return result.primary.field.diffraction
    raise ValueError(f"Unsupported field_mode: {field_mode}")


def _trace_scene(
    *,
    center,
    size,
    freq,
    tx_pos,
    range_x,
    range_y,
    grid_size,
    n_rays,
    max_reflections,
    reflection_coef,
    rotation=None,
    material=None,
    reflection_material_override=None,
    diffraction_material_override=None,
    use_scene_materials_for_reflection=False,
    use_scene_materials_for_diffraction=False,
    rx_polarization=None,
):
    scene = build_test_scene(
        box_drjit_geometry(center=center, size=size, rotation=rotation),
        material=material,
    )
    monitor = FieldMonitor(
        DEFAULT_MONITOR_NAME,
        axis="z",
        position=_monitor_height(tx_pos),
        bounds=(range_x, range_y),
        grid_size=grid_size,
    )
    scene.add_monitor(monitor)
    tracer = Tracer(
        frequency=freq,
        scene=scene,
        reflection_n_rays=n_rays,
        reflection_max_bounces=max_reflections,
        reflection_coef=reflection_coef,
        reflection_material=reflection_material_override,
        diffraction_material=diffraction_material_override,
        use_scene_materials_for_reflection=use_scene_materials_for_reflection,
        use_scene_materials_for_diffraction=use_scene_materials_for_diffraction,
        rx_polarization=rx_polarization,
    )
    result = tracer.trace(tx_pos=tx_pos)
    _assert_plane_monitor_result(result, monitor)
    return result, scene


def _position_ad_gradient(
    *,
    center_vals,
    size,
    freq,
    tx_vals,
    range_x,
    range_y,
    grid_size,
    n_rays,
    max_reflections,
    reflection_coef,
    field_mode="total",
    material=None,
    reflection_material_override=None,
    diffraction_material_override=None,
    use_scene_materials_for_reflection=False,
    use_scene_materials_for_diffraction=False,
    rx_polarization=None,
    **_,
):
    def _component(part: str):
        center = wt.Point3f(*map(float, center_vals))
        tx_pos = wt.Point3f(*map(float, tx_vals))
        dr.enable_grad(center)
        dr.set_grad(center, wt.Vector3f(1.0, 0.0, 0.0))
        result, scene = _trace_scene(
            center=center,
            size=size,
            freq=freq,
            tx_pos=tx_pos,
            range_x=range_x,
            range_y=range_y,
            grid_size=grid_size,
            n_rays=n_rays,
            max_reflections=max_reflections,
            reflection_coef=reflection_coef,
            material=material,
            reflection_material_override=reflection_material_override,
            diffraction_material_override=diffraction_material_override,
            use_scene_materials_for_reflection=use_scene_materials_for_reflection,
            use_scene_materials_for_diffraction=use_scene_materials_for_diffraction,
            rx_polarization=rx_polarization,
        )
        field_component = _field_component(result, field_mode)
        field = field_component.real if part == "real" else field_component.imag
        dr.forward_to(field, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
        grad = dr.grad(field)
        return result, scene, to_numpy(grad) if grad is not None else np.zeros(grid_size * grid_size)

    result, scene, grad_re = _component("real")
    imag_result, imag_scene, grad_im = _component("imag")
    del imag_result, imag_scene
    return result, scene, np.sqrt(grad_re**2 + grad_im**2)


def _position_fd_gradient(**kwargs):
    delta = 0.01
    center_vals = kwargs["center_vals"]
    base_result, scene = _trace_scene(
        center=wt.Point3f(*map(float, center_vals)),
        size=kwargs["size"],
        freq=kwargs["freq"],
        tx_pos=wt.Point3f(*map(float, kwargs["tx_vals"])),
        range_x=kwargs["range_x"],
        range_y=kwargs["range_y"],
        grid_size=kwargs["grid_size"],
        n_rays=kwargs["n_rays"],
        max_reflections=kwargs["max_reflections"],
        reflection_coef=kwargs["reflection_coef"],
        material=kwargs.get("material"),
        reflection_material_override=kwargs.get("reflection_material_override"),
        diffraction_material_override=kwargs.get("diffraction_material_override"),
        use_scene_materials_for_reflection=kwargs.get("use_scene_materials_for_reflection", False),
        use_scene_materials_for_diffraction=kwargs.get("use_scene_materials_for_diffraction", False),
        rx_polarization=kwargs.get("rx_polarization"),
    )
    perturbed_result, perturbed_scene = _trace_scene(
        center=wt.Point3f(float(center_vals[0]) + delta, float(center_vals[1]), float(center_vals[2])),
        size=kwargs["size"],
        freq=kwargs["freq"],
        tx_pos=wt.Point3f(*map(float, kwargs["tx_vals"])),
        range_x=kwargs["range_x"],
        range_y=kwargs["range_y"],
        grid_size=kwargs["grid_size"],
        n_rays=kwargs["n_rays"],
        max_reflections=kwargs["max_reflections"],
        reflection_coef=kwargs["reflection_coef"],
        material=kwargs.get("material"),
        reflection_material_override=kwargs.get("reflection_material_override"),
        diffraction_material_override=kwargs.get("diffraction_material_override"),
        use_scene_materials_for_reflection=kwargs.get("use_scene_materials_for_reflection", False),
        use_scene_materials_for_diffraction=kwargs.get("use_scene_materials_for_diffraction", False),
        rx_polarization=kwargs.get("rx_polarization"),
    )
    base = base_result.primary.field.total
    perturbed = perturbed_result.primary.field.total
    grad_re = (to_numpy(perturbed.real) - to_numpy(base.real)) / delta
    grad_im = (to_numpy(perturbed.imag) - to_numpy(base.imag)) / delta
    del perturbed_result, perturbed_scene
    return base_result, scene, np.sqrt(grad_re**2 + grad_im**2)


def _rotation_ad_gradient(
    *,
    rotation_val,
    center_vals,
    size,
    freq,
    tx_vals,
    range_x,
    range_y,
    grid_size,
    n_rays,
    max_reflections,
    reflection_coef,
    material=None,
    reflection_material_override=None,
    diffraction_material_override=None,
    use_scene_materials_for_reflection=False,
    use_scene_materials_for_diffraction=False,
    **_,
):
    def _component(part: str):
        center = wt.Point3f(*map(float, center_vals))
        tx_pos = wt.Point3f(*map(float, tx_vals))
        rotation = wt.Float(float(rotation_val))
        dr.enable_grad(rotation)
        dr.set_grad(rotation, wt.Float(1.0))
        result, scene = _trace_scene(
            center=center,
            size=size,
            freq=freq,
            tx_pos=tx_pos,
            range_x=range_x,
            range_y=range_y,
            grid_size=grid_size,
            n_rays=n_rays,
            max_reflections=max_reflections,
            reflection_coef=reflection_coef,
            rotation=rotation,
            material=material,
            reflection_material_override=reflection_material_override,
            diffraction_material_override=diffraction_material_override,
            use_scene_materials_for_reflection=use_scene_materials_for_reflection,
            use_scene_materials_for_diffraction=use_scene_materials_for_diffraction,
        )
        field = result.primary.field.total.real if part == "real" else result.primary.field.total.imag
        dr.forward_to(field, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
        grad = dr.grad(field)
        return result, scene, to_numpy(grad) if grad is not None else np.zeros(grid_size * grid_size)

    result, scene, grad_re = _component("real")
    imag_result, imag_scene, grad_im = _component("imag")
    del imag_result, imag_scene
    return result, scene, np.sqrt(grad_re**2 + grad_im**2)


def _rotation_fd_gradient(**kwargs):
    delta = 0.01
    tx_pos = wt.Point3f(*map(float, kwargs["tx_vals"]))
    center = wt.Point3f(*map(float, kwargs["center_vals"]))
    base_result, scene = _trace_scene(
        center=center,
        size=kwargs["size"],
        freq=kwargs["freq"],
        tx_pos=tx_pos,
        range_x=kwargs["range_x"],
        range_y=kwargs["range_y"],
        grid_size=kwargs["grid_size"],
        n_rays=kwargs["n_rays"],
        max_reflections=kwargs["max_reflections"],
        reflection_coef=kwargs["reflection_coef"],
        rotation=wt.Float(float(kwargs["rotation_val"])),
        material=kwargs.get("material"),
        reflection_material_override=kwargs.get("reflection_material_override"),
        diffraction_material_override=kwargs.get("diffraction_material_override"),
        use_scene_materials_for_reflection=kwargs.get("use_scene_materials_for_reflection", False),
        use_scene_materials_for_diffraction=kwargs.get("use_scene_materials_for_diffraction", False),
    )
    perturbed_result, perturbed_scene = _trace_scene(
        center=center,
        size=kwargs["size"],
        freq=kwargs["freq"],
        tx_pos=tx_pos,
        range_x=kwargs["range_x"],
        range_y=kwargs["range_y"],
        grid_size=kwargs["grid_size"],
        n_rays=kwargs["n_rays"],
        max_reflections=kwargs["max_reflections"],
        reflection_coef=kwargs["reflection_coef"],
        rotation=wt.Float(float(kwargs["rotation_val"]) + delta),
        material=kwargs.get("material"),
        reflection_material_override=kwargs.get("reflection_material_override"),
        diffraction_material_override=kwargs.get("diffraction_material_override"),
        use_scene_materials_for_reflection=kwargs.get("use_scene_materials_for_reflection", False),
        use_scene_materials_for_diffraction=kwargs.get("use_scene_materials_for_diffraction", False),
    )
    base = base_result.primary.field.total
    perturbed = perturbed_result.primary.field.total
    grad_re = (to_numpy(perturbed.real) - to_numpy(base.real)) / delta
    grad_im = (to_numpy(perturbed.imag) - to_numpy(base.imag)) / delta
    del perturbed_result, perturbed_scene
    return base_result, scene, np.sqrt(grad_re**2 + grad_im**2)


def _tx_ad_gradient(
    *,
    tx_vals,
    center_vals,
    size,
    freq,
    range_x,
    range_y,
    grid_size,
    n_rays,
    max_reflections,
    reflection_coef,
    material=None,
    reflection_material_override=None,
    diffraction_material_override=None,
    use_scene_materials_for_reflection=False,
    use_scene_materials_for_diffraction=False,
    **_,
):
    def _component(part: str):
        center = wt.Point3f(*map(float, center_vals))
        tx_pos = wt.Point3f(*map(float, tx_vals))
        dr.enable_grad(tx_pos)
        dr.set_grad(tx_pos, wt.Vector3f(1.0, 0.0, 0.0))
        result, scene = _trace_scene(
            center=center,
            size=size,
            freq=freq,
            tx_pos=tx_pos,
            range_x=range_x,
            range_y=range_y,
            grid_size=grid_size,
            n_rays=n_rays,
            max_reflections=max_reflections,
            reflection_coef=reflection_coef,
            material=material,
            reflection_material_override=reflection_material_override,
            diffraction_material_override=diffraction_material_override,
            use_scene_materials_for_reflection=use_scene_materials_for_reflection,
            use_scene_materials_for_diffraction=use_scene_materials_for_diffraction,
        )
        field = result.primary.field.total.real if part == "real" else result.primary.field.total.imag
        dr.forward_to(field, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
        grad = dr.grad(field)
        return result, scene, to_numpy(grad) if grad is not None else np.zeros(grid_size * grid_size)

    result, scene, grad_re = _component("real")
    imag_result, imag_scene, grad_im = _component("imag")
    del imag_result, imag_scene
    return result, scene, np.sqrt(grad_re**2 + grad_im**2)


def _tx_fd_gradient(**kwargs):
    delta = 0.01
    center = wt.Point3f(*map(float, kwargs["center_vals"]))
    tx_vals = kwargs["tx_vals"]
    base_result, scene = _trace_scene(
        center=center,
        size=kwargs["size"],
        freq=kwargs["freq"],
        tx_pos=wt.Point3f(*map(float, tx_vals)),
        range_x=kwargs["range_x"],
        range_y=kwargs["range_y"],
        grid_size=kwargs["grid_size"],
        n_rays=kwargs["n_rays"],
        max_reflections=kwargs["max_reflections"],
        reflection_coef=kwargs["reflection_coef"],
        material=kwargs.get("material"),
        reflection_material_override=kwargs.get("reflection_material_override"),
        diffraction_material_override=kwargs.get("diffraction_material_override"),
        use_scene_materials_for_reflection=kwargs.get("use_scene_materials_for_reflection", False),
        use_scene_materials_for_diffraction=kwargs.get("use_scene_materials_for_diffraction", False),
    )
    perturbed_result, perturbed_scene = _trace_scene(
        center=center,
        size=kwargs["size"],
        freq=kwargs["freq"],
        tx_pos=wt.Point3f(float(tx_vals[0]) + delta, float(tx_vals[1]), float(tx_vals[2])),
        range_x=kwargs["range_x"],
        range_y=kwargs["range_y"],
        grid_size=kwargs["grid_size"],
        n_rays=kwargs["n_rays"],
        max_reflections=kwargs["max_reflections"],
        reflection_coef=kwargs["reflection_coef"],
        material=kwargs.get("material"),
        reflection_material_override=kwargs.get("reflection_material_override"),
        diffraction_material_override=kwargs.get("diffraction_material_override"),
        use_scene_materials_for_reflection=kwargs.get("use_scene_materials_for_reflection", False),
        use_scene_materials_for_diffraction=kwargs.get("use_scene_materials_for_diffraction", False),
    )
    base = base_result.primary.field.total
    perturbed = perturbed_result.primary.field.total
    grad_re = (to_numpy(perturbed.real) - to_numpy(base.real)) / delta
    grad_im = (to_numpy(perturbed.imag) - to_numpy(base.imag)) / delta
    del perturbed_result, perturbed_scene
    return base_result, scene, np.sqrt(grad_re**2 + grad_im**2)


def _to_db(array: np.ndarray, *, floor: float = -120.0) -> np.ndarray:
    return np.maximum(20.0 * np.log10(np.asarray(array, dtype=np.float64) + 1e-20), floor)


def _relative_l2_error(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_arr = np.asarray(lhs)
    rhs_arr = np.asarray(rhs)
    return float(np.linalg.norm(lhs_arr - rhs_arr) / max(float(np.linalg.norm(rhs_arr)), 1e-12))


def _render_row(axes, *, field_db, grad_ad_db, grad_fd_db, edges, tx_pos, range_x, range_y, row_title):
    extent = [range_x[0], range_x[1], range_y[0], range_y[1]]
    field_vmin, field_vmax = -90.0, -20.0
    grad_vmax = max(float(np.percentile(grad_ad_db, 99.5)), float(np.percentile(grad_fd_db, 99.5)))
    grad_vmin = grad_vmax - 60.0
    diff_db = grad_ad_db - grad_fd_db
    diff_vmax = max(float(np.percentile(np.abs(diff_db), 99.5)), 3.0)
    images = (
        axes[0].imshow(field_db, extent=extent, origin="lower", cmap="inferno", vmin=field_vmin, vmax=field_vmax),
        axes[1].imshow(grad_ad_db, extent=extent, origin="lower", cmap="RdBu_r", vmin=grad_vmin, vmax=grad_vmax),
        axes[2].imshow(grad_fd_db, extent=extent, origin="lower", cmap="RdBu_r", vmin=grad_vmin, vmax=grad_vmax),
        axes[3].imshow(diff_db, extent=extent, origin="lower", cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax),
    )
    titles = (
        f"{row_title}: Total Field (dB)",
        f"{row_title}: AD Gradient (dB)",
        f"{row_title}: FD Gradient (dB)",
        f"{row_title}: AD - FD (dB)",
    )
    for ax, image, title in zip(axes, images, titles):
        draw_scene(ax, edges, tx_pos, range_x, range_y)
        ax.set_title(title, fontsize=10)
        plt.colorbar(image, ax=ax, shrink=0.8)


def test_position_rotation_tx_visual_grid():
    _flush_gpu_caches()
    params = {
        "grid_size": 64,
        "freq": 1e9,
        "range_x": (-8.0, 8.0),
        "range_y": (-8.0, 8.0),
        "center_vals": (0.0, 0.0, 2.0),
        "size": 4.0,
        "tx_vals": (-5.0, 5.0, 1.5),
        "rotation_val": float(np.deg2rad(15.0)),
        "n_rays": 20_000,
        "max_reflections": 1,
        "reflection_coef": 1.0,
    }
    grid_size = params["grid_size"]

    legacy_pos_result, legacy_pos_scene, legacy_pos_ad = _position_ad_gradient(**params)
    legacy_pos_summary = _capture_trace_summary(
        legacy_pos_result,
        legacy_pos_scene,
        include_edges=True,
        include_metadata=True,
        include_jones_x_rel=True,
    )
    del legacy_pos_result, legacy_pos_scene
    _flush_gpu_caches()

    pos_result, pos_scene, pos_ad = _position_ad_gradient(**params)
    pos_summary = _capture_trace_summary(
        pos_result,
        pos_scene,
        grid_size=grid_size,
        include_field_db=True,
        include_edges=True,
        include_metadata=True,
        include_material_flags=True,
    )
    del pos_result, pos_scene
    _flush_gpu_caches()

    pos_fd_result, pos_fd_scene, pos_fd = _position_fd_gradient(**params)
    del pos_fd_result, pos_fd_scene
    _flush_gpu_caches()

    rot_result, rot_scene, rot_ad = _rotation_ad_gradient(**params)
    rot_summary = _capture_trace_summary(
        rot_result,
        rot_scene,
        grid_size=grid_size,
        include_field_db=True,
        include_edges=True,
    )
    del rot_result, rot_scene
    _flush_gpu_caches()

    rot_fd_result, rot_fd_scene, rot_fd = _rotation_fd_gradient(**params)
    del rot_fd_result, rot_fd_scene
    _flush_gpu_caches()

    tx_result, tx_scene, tx_ad = _tx_ad_gradient(**params)
    tx_summary = _capture_trace_summary(
        tx_result,
        tx_scene,
        grid_size=grid_size,
        include_field_db=True,
        include_edges=True,
    )
    del tx_result, tx_scene
    _flush_gpu_caches()

    tx_fd_result, tx_fd_scene, tx_fd = _tx_fd_gradient(**params)
    del tx_fd_result, tx_fd_scene
    _flush_gpu_caches()

    scene_material_kwargs = {
        "material": Material(eps_r=2.0, sigma_e=0.0),
        "use_scene_materials_for_reflection": True,
        "use_scene_materials_for_diffraction": True,
    }
    override_material_kwargs = {
        "material": Material(),
        "reflection_material_override": _uniform_override_material_dict(
            eps_r=2.0,
            gain=params["reflection_coef"],
        ),
        "diffraction_material_override": _uniform_override_material_dict(
            eps_r=2.0,
            gain=params["reflection_coef"],
        ),
    }
    scene_total_result, scene_total_scene, scene_total_grad = _position_ad_gradient(
        field_mode="total",
        **scene_material_kwargs,
        **params,
    )
    scene_total_summary = _capture_trace_summary(
        scene_total_result,
        scene_total_scene,
        include_metadata=True,
        include_jones_x_rel=True,
    )
    del scene_total_result, scene_total_scene
    _flush_gpu_caches()

    explicit_total_result, _, explicit_total_grad = _position_ad_gradient(
        field_mode="total",
        rx_polarization=(1.0, 0.0, 0.0),
        **scene_material_kwargs,
        **params,
    )
    explicit_total_summary = _capture_trace_summary(explicit_total_result, include_jones_x_rel=True)
    del explicit_total_result
    _flush_gpu_caches()

    scene_ref_result, scene_ref_scene, scene_ref_grad = _position_ad_gradient(
        field_mode="reflection",
        **scene_material_kwargs,
        **params,
    )
    del scene_ref_result, scene_ref_scene
    _flush_gpu_caches()

    scene_dif_result, scene_dif_scene, scene_dif_grad = _position_ad_gradient(
        field_mode="diffraction",
        **scene_material_kwargs,
        **params,
    )
    del scene_dif_result, scene_dif_scene
    _flush_gpu_caches()

    override_total_result, override_total_scene, override_total_grad = _position_ad_gradient(
        field_mode="total",
        **override_material_kwargs,
        **params,
    )
    override_total_summary = _capture_trace_summary(override_total_result, include_metadata=True)
    del override_total_result, override_total_scene
    _flush_gpu_caches()

    legacy_ref_result, legacy_ref_scene, legacy_ref_grad = _position_ad_gradient(field_mode="reflection", **params)
    del legacy_ref_result, legacy_ref_scene
    _flush_gpu_caches()

    legacy_dif_result, legacy_dif_scene, legacy_dif_grad = _position_ad_gradient(field_mode="diffraction", **params)
    del legacy_dif_result, legacy_dif_scene
    _flush_gpu_caches()

    tx_vals = params["tx_vals"]
    range_x = params["range_x"]
    range_y = params["range_y"]

    fig, axes = plt.subplots(3, 4, figsize=(19, 14), constrained_layout=True)

    _render_row(
        axes[0],
        field_db=pos_summary["field_db"],
        grad_ad_db=_to_db(pos_ad.reshape(grid_size, grid_size)),
        grad_fd_db=_to_db(pos_fd.reshape(grid_size, grid_size)),
        edges=pos_summary["edges_2d"],
        tx_pos=tx_vals,
        range_x=range_x,
        range_y=range_y,
        row_title="Position",
    )

    _render_row(
        axes[1],
        field_db=rot_summary["field_db"],
        grad_ad_db=_to_db(rot_ad.reshape(grid_size, grid_size)),
        grad_fd_db=_to_db(rot_fd.reshape(grid_size, grid_size)),
        edges=rot_summary["edges_2d"],
        tx_pos=tx_vals,
        range_x=range_x,
        range_y=range_y,
        row_title="Rotation",
    )

    _render_row(
        axes[2],
        field_db=tx_summary["field_db"],
        grad_ad_db=_to_db(tx_ad.reshape(grid_size, grid_size)),
        grad_fd_db=_to_db(tx_fd.reshape(grid_size, grid_size)),
        edges=tx_summary["edges_2d"],
        tx_pos=tx_vals,
        range_x=range_x,
        range_y=range_y,
        row_title="TX Position",
    )

    fig.suptitle("Integrated Gradient Visual Test: Position, Rotation, TX", fontsize=14)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)

    pos_grad_db = _to_db(pos_ad.reshape(grid_size, grid_size))
    legacy_pos_grad_db = _to_db(legacy_pos_ad.reshape(grid_size, grid_size))
    legacy_pos_diff = pos_ad.reshape(grid_size, grid_size) - legacy_pos_ad.reshape(grid_size, grid_size)
    legacy_compare_vmax = max(
        float(np.percentile(pos_grad_db, 99.5)),
        float(np.percentile(legacy_pos_grad_db, 99.5)),
    )
    legacy_compare_vmin = legacy_compare_vmax - 60.0
    legacy_diff_vmax = max(float(np.percentile(np.abs(legacy_pos_diff), 99.5)), 1e-12)
    legacy_pos_rel_l2_error = _relative_l2_error(pos_ad, legacy_pos_ad)
    legacy_pos_metadata = pos_summary["metadata"]

    compare_fig, compare_axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    compare_panels = (
        {
            "ax": compare_axes[0],
            "image": legacy_pos_grad_db,
            "cmap": "inferno",
            "vmin": legacy_compare_vmin,
            "vmax": legacy_compare_vmax,
            "title": "Legacy Cube Position Total-Field Gradient (dB)",
            "note": "legacy scalar reflection / PEC diffraction",
        },
        {
            "ax": compare_axes[1],
            "image": pos_grad_db,
            "cmap": "inferno",
            "vmin": legacy_compare_vmin,
            "vmax": legacy_compare_vmax,
            "title": "Current Cube Position Total-Field Gradient (dB)",
            "note": (
                f"reflection_model_source={legacy_pos_metadata['reflection_model_source']}\n"
                f"diffraction_face_material_source={legacy_pos_metadata['diffraction_face_material_source']}"
            ),
        },
        {
            "ax": compare_axes[2],
            "image": legacy_pos_diff,
            "cmap": "RdBu_r",
            "vmin": -legacy_diff_vmax,
            "vmax": legacy_diff_vmax,
            "title": "Current Minus Legacy (Linear)",
            "note": f"rel-L2={legacy_pos_rel_l2_error:.2e}",
        },
    )
    extent = [range_x[0], range_x[1], range_y[0], range_y[1]]
    legacy_edges = legacy_pos_summary["edges_2d"]
    for panel in compare_panels:
        image = panel["ax"].imshow(
            panel["image"],
            extent=extent,
            origin="lower",
            cmap=panel["cmap"],
            vmin=panel["vmin"],
            vmax=panel["vmax"],
        )
        draw_scene(panel["ax"], legacy_edges, tx_vals, range_x, range_y)
        panel["ax"].set_title(panel["title"], fontsize=10)
        panel["ax"].text(
            0.02,
            0.02,
            panel["note"],
            transform=panel["ax"].transAxes,
            fontsize=8,
            color="white",
            va="bottom",
            ha="left",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3.0},
        )
        plt.colorbar(image, ax=panel["ax"], shrink=0.84)

    compare_fig.suptitle(
        "Cube Position Total-Field Gradient: Legacy vs Current test_position_rotation_tx.py Path",
        fontsize=14,
    )
    compare_fig.savefig(POSITION_LEGACY_COMPARE_OUTPUT_PATH, dpi=180)
    plt.close(compare_fig)

    scene_total_db = _to_db(scene_total_grad.reshape(grid_size, grid_size))
    legacy_total_db = _to_db(legacy_pos_ad.reshape(grid_size, grid_size))
    override_total_db = _to_db(override_total_grad.reshape(grid_size, grid_size))
    legacy_ref_db = _to_db(legacy_ref_grad.reshape(grid_size, grid_size))
    scene_ref_db = _to_db(scene_ref_grad.reshape(grid_size, grid_size))
    legacy_dif_db = _to_db(legacy_dif_grad.reshape(grid_size, grid_size))
    scene_dif_db = _to_db(scene_dif_grad.reshape(grid_size, grid_size))
    scene_vs_legacy_total_diff = scene_total_grad.reshape(grid_size, grid_size) - legacy_pos_ad.reshape(grid_size, grid_size)
    scene_vs_legacy_ref_diff = scene_ref_grad.reshape(grid_size, grid_size) - legacy_ref_grad.reshape(grid_size, grid_size)
    scene_vs_legacy_dif_diff = scene_dif_grad.reshape(grid_size, grid_size) - legacy_dif_grad.reshape(grid_size, grid_size)
    scene_vs_legacy_total_rel = _relative_l2_error(scene_total_grad, legacy_pos_ad)
    scene_vs_legacy_ref_rel = _relative_l2_error(scene_ref_grad, legacy_ref_grad)
    scene_vs_legacy_dif_rel = _relative_l2_error(scene_dif_grad, legacy_dif_grad)
    scene_vs_override_total_rel = _relative_l2_error(scene_total_grad, override_total_grad)
    explicit_total_db = _to_db(explicit_total_grad.reshape(grid_size, grid_size))
    explicit_vs_legacy_diff = explicit_total_grad.reshape(grid_size, grid_size) - legacy_pos_ad.reshape(grid_size, grid_size)
    explicit_vs_legacy_rel = _relative_l2_error(explicit_total_grad, legacy_pos_ad)
    scene_vs_explicit_diff = scene_total_grad.reshape(grid_size, grid_size) - explicit_total_grad.reshape(grid_size, grid_size)
    scene_vs_explicit_rel = _relative_l2_error(scene_total_grad, explicit_total_grad)
    legacy_jones_x_rel = legacy_pos_summary["jones_x_rel_l2"]
    explicit_jones_x_rel = explicit_total_summary["jones_x_rel_l2"]
    scene_jones_x_rel = scene_total_summary["jones_x_rel_l2"]

    material_compare_fig, material_compare_axes = plt.subplots(
        3,
        3,
        figsize=(15, 14),
        constrained_layout=True,
    )
    material_compare_rows = (
        {
            "row_idx": 0,
            "legacy_db": legacy_total_db,
            "scene_db": scene_total_db,
            "diff": scene_vs_legacy_total_diff,
            "title": "Total Field Gradient",
            "rel": scene_vs_legacy_total_rel,
            "note": (
                f"scene-vs-override rel-L2={scene_vs_override_total_rel:.2e}\n"
                f"legacy source={legacy_pos_summary['metadata']['reflection_model_source']}/"
                f"{legacy_pos_summary['metadata']['diffraction_face_material_source']}\n"
                f"opt-in scene source={scene_total_summary['metadata']['reflection_model_source']}/"
                f"{scene_total_summary['metadata']['diffraction_face_material_source']}"
            ),
        },
        {
            "row_idx": 1,
            "legacy_db": legacy_ref_db,
            "scene_db": scene_ref_db,
            "diff": scene_vs_legacy_ref_diff,
            "title": "Reflection Gradient",
            "rel": scene_vs_legacy_ref_rel,
            "note": "Legacy scalar bounce weight vs Fresnel scene material",
        },
        {
            "row_idx": 2,
            "legacy_db": legacy_dif_db,
            "scene_db": scene_dif_db,
            "diff": scene_vs_legacy_dif_diff,
            "title": "Diffraction Gradient",
            "rel": scene_vs_legacy_dif_rel,
            "note": "Legacy PEC wedge-face coefficients vs Fresnel face coefficients",
        },
    )
    for row in material_compare_rows:
        legacy_vmax = max(
            float(np.percentile(row["legacy_db"], 99.5)),
            float(np.percentile(row["scene_db"], 99.5)),
        )
        legacy_vmin = legacy_vmax - 60.0
        diff_vmax = max(float(np.percentile(np.abs(row["diff"]), 99.5)), 1e-12)
        panels = (
            {
                "ax": material_compare_axes[row["row_idx"], 0],
                "image": row["legacy_db"],
                "cmap": "inferno",
                "vmin": legacy_vmin,
                "vmax": legacy_vmax,
                "title": f"Legacy {row['title']} (dB)",
                "note": "ground truth baseline",
            },
            {
                "ax": material_compare_axes[row["row_idx"], 1],
                "image": row["scene_db"],
                "cmap": "inferno",
                "vmin": legacy_vmin,
                "vmax": legacy_vmax,
                "title": f"Opt-In Scene-Material {row['title']} (dB)",
                "note": row["note"],
            },
            {
                "ax": material_compare_axes[row["row_idx"], 2],
                "image": row["diff"],
                "cmap": "RdBu_r",
                "vmin": -diff_vmax,
                "vmax": diff_vmax,
                "title": f"Scene Minus Legacy {row['title']} (Linear)",
                "note": f"rel-L2={row['rel']:.2e}",
            },
        )
        for panel in panels:
            image = panel["ax"].imshow(
                panel["image"],
                extent=extent,
                origin="lower",
                cmap=panel["cmap"],
                vmin=panel["vmin"],
                vmax=panel["vmax"],
            )
            draw_scene(panel["ax"], legacy_edges, tx_vals, range_x, range_y)
            panel["ax"].set_title(panel["title"], fontsize=10)
            panel["ax"].text(
                0.02,
                0.02,
                panel["note"],
                transform=panel["ax"].transAxes,
                fontsize=8,
                color="white",
                va="bottom",
                ha="left",
                bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3.0},
            )
            plt.colorbar(image, ax=panel["ax"], shrink=0.84)

    material_compare_fig.suptitle(
        "Cube Position Gradient: Legacy Ground Truth vs Opt-In Material-Aware Scene Path",
        fontsize=14,
    )
    material_compare_fig.savefig(POSITION_MATERIAL_MODEL_COMPARE_OUTPUT_PATH, dpi=180)
    plt.close(material_compare_fig)

    scalar_compare_vmax = max(
        float(np.percentile(legacy_total_db, 99.5)),
        float(np.percentile(explicit_total_db, 99.5)),
        float(np.percentile(scene_total_db, 99.5)),
    )
    scalar_compare_vmin = scalar_compare_vmax - 60.0
    explicit_diff_vmax = max(float(np.percentile(np.abs(explicit_vs_legacy_diff), 99.5)), 1e-12)
    scene_diff_vmax = max(float(np.percentile(np.abs(scene_vs_legacy_total_diff), 99.5)), 1e-12)
    scene_explicit_diff_vmax = max(float(np.percentile(np.abs(scene_vs_explicit_diff), 99.5)), 1e-12)
    scalarization_fig, scalarization_axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    scalarization_panels = (
        {
            "ax": scalarization_axes[0, 0],
            "image": legacy_total_db,
            "cmap": "inferno",
            "vmin": scalar_compare_vmin,
            "vmax": scalar_compare_vmax,
            "title": "Legacy Total-Field Gradient (dB)",
            "note": f"scalar/PEC baseline\nfield-vs-Jones-x rel-L2={legacy_jones_x_rel:.2e}",
        },
        {
            "ax": scalarization_axes[0, 1],
            "image": explicit_total_db,
            "cmap": "inferno",
            "vmin": scalar_compare_vmin,
            "vmax": scalar_compare_vmax,
            "title": "Explicit Co-Polar Jones Projection (dB)",
            "note": (
                "rx_polarization=(1, 0, 0)\n"
                f"field-vs-Jones-x rel-L2={explicit_jones_x_rel:.2e}"
            ),
        },
        {
            "ax": scalarization_axes[0, 2],
            "image": scene_total_db,
            "cmap": "inferno",
            "vmin": scalar_compare_vmin,
            "vmax": scalar_compare_vmax,
            "title": "Default Implicit Co-Polar Scalar Gradient (dB)",
            "note": (
                "rx_polarization=None, implicit tx co-polar projection\n"
                f"field-vs-Jones-x rel-L2={scene_jones_x_rel:.2e}"
            ),
        },
        {
            "ax": scalarization_axes[1, 0],
            "image": explicit_vs_legacy_diff,
            "cmap": "RdBu_r",
            "vmin": -explicit_diff_vmax,
            "vmax": explicit_diff_vmax,
            "title": "Explicit Co-Polar Minus Legacy (Linear)",
            "note": f"rel-L2={explicit_vs_legacy_rel:.2e}",
        },
        {
            "ax": scalarization_axes[1, 1],
            "image": scene_vs_legacy_total_diff,
            "cmap": "RdBu_r",
            "vmin": -scene_diff_vmax,
            "vmax": scene_diff_vmax,
            "title": "Default Material-Aware Scalar Minus Legacy (Linear)",
            "note": f"rel-L2={scene_vs_legacy_total_rel:.2e}",
        },
        {
            "ax": scalarization_axes[1, 2],
            "image": scene_vs_explicit_diff,
            "cmap": "RdBu_r",
            "vmin": -scene_explicit_diff_vmax,
            "vmax": scene_explicit_diff_vmax,
            "title": "Default Implicit Minus Explicit Co-Polar (Linear)",
            "note": f"rel-L2={scene_vs_explicit_rel:.2e}",
        },
    )
    for panel in scalarization_panels:
        image = panel["ax"].imshow(
            panel["image"],
            extent=extent,
            origin="lower",
            cmap=panel["cmap"],
            vmin=panel["vmin"],
            vmax=panel["vmax"],
        )
        draw_scene(panel["ax"], legacy_edges, tx_vals, range_x, range_y)
        panel["ax"].set_title(panel["title"], fontsize=10)
        panel["ax"].text(
            0.02,
            0.02,
            panel["note"],
            transform=panel["ax"].transAxes,
            fontsize=8,
            color="white",
            va="bottom",
            ha="left",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 3.0},
        )
        plt.colorbar(image, ax=panel["ax"], shrink=0.84)

    scalarization_fig.suptitle(
        "Cube Position Gradient: Legacy vs Explicit Co-Polar Jones Projection vs Default Implicit Co-Polar Field",
        fontsize=14,
    )
    scalarization_fig.savefig(POSITION_SCALARIZATION_COMPARE_OUTPUT_PATH, dpi=180)
    plt.close(scalarization_fig)

    assert OUTPUT_PATH.exists()
    assert OUTPUT_PATH.stat().st_size > 0
    assert POSITION_LEGACY_COMPARE_OUTPUT_PATH.exists()
    assert POSITION_LEGACY_COMPARE_OUTPUT_PATH.stat().st_size > 0
    assert POSITION_MATERIAL_MODEL_COMPARE_OUTPUT_PATH.exists()
    assert POSITION_MATERIAL_MODEL_COMPARE_OUTPUT_PATH.stat().st_size > 0
    assert POSITION_SCALARIZATION_COMPARE_OUTPUT_PATH.exists()
    assert POSITION_SCALARIZATION_COMPARE_OUTPUT_PATH.stat().st_size > 0
    assert float(np.sum(pos_ad)) > 0.0
    assert float(np.sum(rot_ad)) > 0.0
    assert float(np.sum(tx_ad)) > 0.0
    assert not pos_summary["material_has_specified_materials"]
    assert pos_summary["material_n_default_material_triangles"] > 0
    assert pos_summary["metadata"]["reflection_model_source"] == "default"
    assert pos_summary["metadata"]["diffraction_face_material_source"] == "default"
    assert legacy_pos_rel_l2_error < 2e-7
    assert scene_total_summary["metadata"]["reflection_model_source"] == "scene"
    assert scene_total_summary["metadata"]["diffraction_face_material_source"] == "scene"
    assert override_total_summary["metadata"]["reflection_model_source"] == "override"
    assert override_total_summary["metadata"]["diffraction_face_material_source"] == "override"
    assert scene_vs_override_total_rel < 1e-5
    assert explicit_vs_legacy_rel > 0.1
    assert scene_vs_explicit_rel < 1e-5
    assert explicit_jones_x_rel < 1e-5
    assert scene_jones_x_rel < 1e-5
    assert scene_vs_legacy_ref_rel > 0.5
    assert scene_vs_legacy_dif_rel > 0.2
