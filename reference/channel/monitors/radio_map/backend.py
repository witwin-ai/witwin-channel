from __future__ import annotations

import drjit as dr

from .monitor import RadioMapMonitor
from ..._native import native_extension_available
from ...scene import Scene
from ..orchestration import ResolvedTraceConfig


def _normalize_radio_map_accumulation_backend(value: str) -> str:
    resolved = str(value).lower()
    if resolved not in {
        "auto",
        "baseline",
        "native_coherent",
        "cell_accumulation",
        "native_monte_carlo",
    }:
        raise ValueError(
            "radio_map_accumulation_backend must be 'auto', 'baseline', "
            "'native_coherent', 'cell_accumulation', or 'native_monte_carlo'."
        )
    return resolved


def _point_grad_enabled(point) -> bool:
    if point is None:
        return False
    for axis in ("x", "y", "z"):
        component = getattr(point, axis, None)
        if component is None:
            continue
        try:
            if bool(dr.grad_enabled(component)):
                return True
        except Exception:
            continue
    return False


def _scene_geometry_grad_enabled(scene) -> bool:
    if scene is None:
        return False
    vertices = getattr(scene, "vertices", None)
    if vertices is not None and _point_grad_enabled(vertices):
        return True
    tri_data = getattr(scene, "tri_data_gpu", None)
    if isinstance(tri_data, dict):
        for key in ("v0", "v1", "v2"):
            value = tri_data.get(key)
            if value is not None and _point_grad_enabled(value):
                return True
    return False


def _scene_material_grad_enabled(scene) -> bool:
    if scene is None:
        return False
    tri_data = getattr(scene, "tri_data_gpu", None)
    if isinstance(tri_data, dict):
        for key in ("material_eps_r", "material_sigma_e"):
            value = tri_data.get(key)
            if value is None:
                continue
            try:
                if bool(dr.grad_enabled(value)):
                    return True
            except Exception:
                continue
    return False


def _radio_map_grad_sensitive_workload(
    config: ResolvedTraceConfig,
    *,
    tx_pos,
    scene: Scene,
) -> bool:
    if _point_grad_enabled(tx_pos):
        return True
    if _scene_geometry_grad_enabled(scene):
        return True
    if (
        bool(config.use_scene_materials_for_reflection)
        or bool(config.use_scene_materials_for_diffraction)
    ) and _scene_material_grad_enabled(scene):
        return True
    return False


def _radio_map_native_coherent_supported(
    monitor: RadioMapMonitor,
    grid,
    config: ResolvedTraceConfig,
    *,
    scene: Scene | None = None,
) -> bool:
    return (
        grid.surface_mode == "axis_aligned"
        and monitor.combine_mode == "coherent"
        and str(monitor.receiver_model) == "projected_polarized"
        and native_extension_available()
        and str(config.reflection_field_backend) == "native"
        and str(config.diffraction_execution.suffix_backend) == "native"
    )


def _radio_map_cell_accumulation_supported(
    monitor: RadioMapMonitor,
    grid,
    config: ResolvedTraceConfig,
    *,
    tx_pos,
    scene: Scene,
) -> bool:
    matched_isotropic_vector_coherent = (
        str(monitor.combine_mode) == "coherent"
        and str(monitor.receiver_model) == "matched_isotropic"
    )
    if (
        grid.surface_mode != "axis_aligned"
        or (
            str(monitor.combine_mode) != "incoherent"
            and not matched_isotropic_vector_coherent
        )
    ):
        return False
    if str(monitor.receiver_model) not in {"projected_polarized", "matched_isotropic"}:
        return False
    if _point_grad_enabled(tx_pos):
        return False
    if _scene_geometry_grad_enabled(scene):
        return False
    if (
        bool(config.use_scene_materials_for_reflection)
        or bool(config.use_scene_materials_for_diffraction)
    ) and _scene_material_grad_enabled(scene):
        return False
    return True


def _radio_map_native_monte_carlo_supported(
    monitor: RadioMapMonitor,
    grid,
    config: ResolvedTraceConfig,
    *,
    tx_pos,
    scene: Scene,
) -> bool:
    if (
        str(getattr(monitor, "sampling_mode", "deterministic")) != "monte_carlo"
        or grid.surface_mode != "axis_aligned"
        or str(monitor.combine_mode) != "incoherent"
        or str(monitor.receiver_model) != "matched_isotropic"
        or not native_extension_available()
    ):
        return False
    return True


def _resolve_radio_map_accumulation_backend(
    *,
    requested_backend: str,
    monitor: RadioMapMonitor,
    grid,
    config: ResolvedTraceConfig,
    tx_pos,
    scene: Scene,
) -> str:
    resolved_requested = _normalize_radio_map_accumulation_backend(requested_backend)
    if str(getattr(monitor, "sampling_mode", "deterministic")) == "monte_carlo":
        if resolved_requested == "auto":
            if _radio_map_native_monte_carlo_supported(
                monitor,
                grid,
                config,
                tx_pos=tx_pos,
                scene=scene,
            ):
                return "native_monte_carlo"
            raise RuntimeError(
                "sampling_mode='monte_carlo' requires the bundled native extension and an "
                "axis-aligned matched-isotropic radio-map workload."
            )
        if resolved_requested != "native_monte_carlo":
            raise RuntimeError(
                "sampling_mode='monte_carlo' only supports "
                "radio_map_accumulation_backend='auto' or 'native_monte_carlo'."
            )
        if not _radio_map_native_monte_carlo_supported(
            monitor,
            grid,
            config,
            tx_pos=tx_pos,
                scene=scene,
            ):
            raise RuntimeError(
                "radio_map_accumulation_backend='native_monte_carlo' requires the bundled "
                "native extension and an axis-aligned matched-isotropic Monte Carlo "
                "radio-map workload."
            )
        return resolved_requested
    if resolved_requested == "auto":
        if _radio_map_native_coherent_supported(monitor, grid, config, scene=scene):
            return "native_coherent"
        if _radio_map_cell_accumulation_supported(
            monitor,
            grid,
            config,
            tx_pos=tx_pos,
            scene=scene,
        ):
            return "cell_accumulation"
        return "baseline"
    if resolved_requested == "native_coherent" and not _radio_map_native_coherent_supported(
        monitor,
        grid,
        config,
        scene=scene,
    ):
        raise RuntimeError(
            "radio_map_accumulation_backend='native_coherent' requires an axis-aligned "
            "RadioMapMonitor with combine_mode='coherent', native_extension_available()==True, "
            "and trace.reflection_field_backend='native', diffraction_execution.suffix_backend='native'."
        )
    if resolved_requested == "cell_accumulation" and not _radio_map_cell_accumulation_supported(
        monitor,
        grid,
        config,
        tx_pos=tx_pos,
        scene=scene,
    ):
        raise RuntimeError(
            "radio_map_accumulation_backend='cell_accumulation' requires an axis-aligned "
            "RadioMapMonitor with either combine_mode='incoherent' or "
            "combine_mode='coherent' plus receiver_model='matched_isotropic', and a "
            "gradient-disabled workload."
        )
    return resolved_requested


__all__ = [
    "_normalize_radio_map_accumulation_backend",
    "_point_grad_enabled",
    "_radio_map_grad_sensitive_workload",
    "_radio_map_native_coherent_supported",
    "_radio_map_cell_accumulation_supported",
    "_radio_map_native_monte_carlo_supported",
    "_resolve_radio_map_accumulation_backend",
    "_scene_geometry_grad_enabled",
    "_scene_material_grad_enabled",
]
