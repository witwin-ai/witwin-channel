from __future__ import annotations

from typing import Any

import torch

from witwin.channel_native import Scene
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.core.kernels.metadata import make_metadata
from witwin.channel_native.core.kernels.ops import deterministic_component_counts

from .accumulation import accumulate_path_result, build_path_table
from .config import Config
from .result import Result
from .topology import export_topology, receiver_positions_and_layout


def _validate_requested_components(config: Config) -> None:
    if "reflection" in config.components and config.max_depth < 1:
        raise RuntimeError("deterministic reflection requires max_depth >= 1")
    if "diffraction" in config.components:
        if config.max_depth < 1:
            raise RuntimeError("deterministic diffraction requires max_depth >= 1")
        if config.max_diffraction_order < 1:
            raise RuntimeError("deterministic diffraction requires max_diffraction_order >= 1")


def _metadata(
    *,
    config: Config,
    native_info: dict[str, Any],
    path_count: int,
    component_counts: dict[str, int],
    launch_count: int,
) -> dict[str, Any]:
    capability = {
        "raydn_native": bool(native_info["uses_raydn_native"]),
        "path_native": bool(native_info.get("uses_path_native", False)),
        "cuda_available": bool(native_info["cuda_available"]),
        "optix_available": bool(native_info["optix_available"]),
    }
    components = {
        "los": "enabled" if "los" in config.components else "not_requested",
        "reflection": "not_requested",
        "diffraction": "not_requested",
    }
    if "reflection" in config.components:
        if not capability["raydn_native"]:
            raise RuntimeError("deterministic reflection requires RayDN native capability")
        components["reflection"] = "enabled"
    if "diffraction" in config.components:
        if not capability["raydn_native"]:
            raise RuntimeError("deterministic diffraction requires RayDN native capability")
        components["diffraction"] = "enabled"
    raydn_component_enabled = components["reflection"] == "enabled" or components["diffraction"] == "enabled"
    return {
        "max_depth": config.max_depth,
        "max_diffraction_order": config.max_diffraction_order,
        "coherent": config.coherent,
        "return_field": config.return_field,
        "export_paths": config.export_paths,
        "max_paths": config.max_paths,
        "sort_key": config.sort_key,
        "accumulation_strategy": "coherent" if config.coherent else "incoherent",
        "components": components,
        "counts": {
            "path_count": path_count,
            "valid_path_count": path_count,
            "components": component_counts,
        },
        "capability": capability,
        "kernel": make_metadata(
            primitive="deterministic_solver",
            forward_launch_count=launch_count,
            accumulation_strategy="atomic_add",
            scheduling_strategy="native_fused" if raydn_component_enabled else "native_cuda",
            raydn_native=capability["raydn_native"],
            ad_status="none",
        ),
    }


def solve(scene: Scene, config: Config) -> Result:
    if not torch.cuda.is_available():
        raise RuntimeError("witwin.channel_native.deterministic requires CUDA")

    native_info = build_info()
    _validate_requested_components(config)
    if "reflection" in config.components and not native_info["uses_raydn_native"]:
        raise RuntimeError("deterministic reflection requires RayDN native capability")
    if "diffraction" in config.components and not native_info["uses_raydn_native"]:
        raise RuntimeError("deterministic diffraction requires RayDN native capability")
    has_grid = any(receiver.__class__.__name__ == "ReceiverGrid" for receiver in scene.receivers)
    if has_grid and len(scene.receivers) > 1 and not config.export_paths:
        raise RuntimeError("mixed point/grid receivers require export_paths=True")

    device = torch.device("cuda")
    _, layout = receiver_positions_and_layout(scene, device=device)
    path_result = export_topology(scene, config)
    path_count = int(path_result.valid.numel())
    component_counts = deterministic_component_counts(path_result.component_id)
    path_gain, field, component_power, component_fields = accumulate_path_result(
        path_result,
        frequency_hz=float(scene.frequency),
        num_tx=len(scene.transmitters),
        num_rx=layout.receiver_count,
        layout=layout,
        coherent=config.coherent,
        return_field=config.return_field,
    )
    metadata = _metadata(
        config=config,
        native_info=native_info,
        path_count=path_count,
        component_counts=component_counts,
        launch_count=path_result.launch_count,
    )
    diagnostics = None
    if config.diagnostics:
        candidate_count = int(path_result.candidate_count)
        diagnostics = {
            "path_gain_shape": tuple(path_gain.shape),
            "field_shape": tuple(field.shape),
            "path_count": path_count,
            "component_counts": component_counts,
            "coherent": config.coherent,
            "accumulation_mode": "coherent" if config.coherent else "incoherent",
            "native_launch_count": int(path_result.launch_count),
            "visibility_rejection_count": int(path_result.visibility_rejection_count),
            "selected_edge_count": int(path_result.selected_edge_count),
            "path_planning": {
                "max_paths": config.max_paths,
                "candidate_count": candidate_count,
                "guardrail_count": int(path_result.guardrail_count),
                "truncated": config.max_paths is not None and candidate_count > int(config.max_paths),
            },
        }
    return Result(
        path_gain=path_gain,
        field=field,
        component_power=component_power,
        component_fields=component_fields,
        paths=(
            build_path_table(
                path_result,
                frequency_hz=float(scene.frequency),
                include_fields=config.return_field,
            )
            if config.export_paths
            else None
        ),
        metadata=metadata,
        diagnostics=diagnostics,
    )
