from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any

import torch


from witwin.channel_native.core.antenna import validate_scalar_endpoint_features
from witwin.channel_native.capabilities import (
    capabilities,
    config_metadata,
    serialize_config,
)
from witwin.channel_native import build_info
from witwin.channel_native.core.kernels.metadata import make_metadata
from witwin.channel_native.core.field_state import PHASE_CONVENTION
from witwin.channel_native.scene.models import ReceiverGrid
from witwin.channel_native.scene.tensors import _frequency_scalar
from witwin.channel_native.propagation.geometry.endpoints import (
    apply_receiver_layout,
    receiver_positions_and_layout,
)
from witwin.channel_native.propagation.topology.kernels.primitives import (
    deterministic_component_counts,
)
from witwin.channel_native.propagation.enumerated.engine import (
    evaluate_enumerated_paths,
)
from witwin.channel_native.propagation.enumerated.scattering import (
    append_scattering_evaluated_paths,
)
from witwin.channel_native.core.components import component_availability_status

from .accumulation import (
    _OPTIONAL_COMPONENTS,
    accumulate_path_result,
    build_path_table,
)
from .config import Config
from .result import Result

if TYPE_CHECKING:
    from witwin.channel_native.scene.models import Scene


def _validate_requested_components(config: Config) -> None:
    if "reflection" in config.components and config.max_depth < 1:
        raise RuntimeError("deterministic reflection requires max_depth >= 1")
    if "diffraction" in config.components:
        if config.max_depth < 1:
            raise RuntimeError("deterministic diffraction requires max_depth >= 1")
        if config.max_diffraction_order < 1:
            raise RuntimeError(
                "deterministic diffraction requires max_diffraction_order >= 1"
            )


def _coupled_paths_metadata(config: Config) -> dict[str, Any]:
    """Coupled reflection-diffraction metadata block (ADR-011).

    Mirrors the path solver's coupled_paths block.
    """

    if not config.coupled_paths:
        return {
            "requested": False,
            "geometry": "not_requested",
            "coefficient": "not_requested",
        }
    return {
        "requested": True,
        "geometry": "native_1r1d_reciprocal",
        "coefficient": "unified_complex3_jones",
    }


def _register_coupled_component(
    config: Config,
    topology: Any,
    component_counts: dict[str, int],
    extra_components: tuple[str, ...],
) -> tuple[str, ...]:
    """Record the coupled component count and export name (ADR-011).

    Coupled rows carry component ids 3 (R->D) and 4 (D->R); both accumulate
    into the single coupled slot. "coupled" is not a public component name, so
    it is enabled by the coupled_paths gate rather than the components set.
    """

    if not config.coupled_paths:
        return extra_components
    component_counts["coupled"] = int(
        ((topology.component_id == 3) | (topology.component_id == 4)).sum().item()
    )
    return extra_components + ("coupled",)


def _metadata(
    *,
    config: Config,
    native_info: dict[str, Any],
    path_count: int,
    component_counts: dict[str, int],
    launch_count: int,
    ad_companion_launches: int = 0,
    ad_tape_bytes: int = 0,
    forward_time_ms: float = 0.0,
    peak_memory_bytes: int = 0,
    scattering_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    capability = {
        "raydn_native": bool(native_info["uses_raydn_native"]),
        "path_native": bool(native_info.get("uses_path_native", False)),
        "cuda_available": bool(native_info["cuda_available"]),
        "optix_available": bool(native_info["optix_available"]),
    }
    components = component_availability_status(
        config.components,
        reflection_available=capability["raydn_native"],
        diffraction_available=capability["raydn_native"],
        reflection_error="deterministic reflection requires RayDN native capability",
        diffraction_error="deterministic diffraction requires RayDN native capability",
    )
    # transmission carries specular wall-penetration paths since wave 2 and
    # scattering carries Kirchhoff rough-surface patch paths since wave 3.
    # Both keep the truthful requested-but-empty status when no paths were
    # found (e.g. every surface in the scene is smooth).
    for name in ("transmission", "scattering"):
        if name not in config.components:
            components[name] = "not_requested"
        elif component_counts.get(name, 0) > 0:
            components[name] = "enabled"
        else:
            components[name] = "enabled_no_paths"
    if "transmission" in config.components:
        if not capability["raydn_native"]:
            raise RuntimeError(
                "deterministic transmission requires RayDN native capability"
            )
        # Endpoint-connection thin_sheet contract (plan 05 section 4).
        metadata_transmission = {
            "thin_sheet_straight_path_approximation": True,
            "group_delay": "geometric",
        }
    else:
        metadata_transmission = None
    raydn_component_enabled = (
        components["reflection"] == "enabled" or components["diffraction"] == "enabled"
    )
    requested_config = serialize_config(config)
    effective_config = dict(requested_config)
    metadata = {
        "max_depth": config.max_depth,
        "max_diffraction_order": config.max_diffraction_order,
        "coherent": config.coherent,
        "return_field": config.return_field,
        "export_paths": config.export_paths,
        "max_paths": config.max_paths,
        "max_paths_scope": config.max_paths_scope,
        "sort_key": config.sort_key,
        "accumulation_strategy": "coherent" if config.coherent else "incoherent",
        "components": components,
        "counts": {
            "path_count": path_count,
            "valid_path_count": path_count,
            "components": component_counts,
        },
        "capability": capability,
        # Plan 07 AD-4: the real registered-companion accounting. vjp retains
        # tape and schedules its companions on the user's later backward; jvp
        # runs its dual companions inside this forward and retains no tape.
        "kernel": make_metadata(
            primitive="deterministic_solver",
            forward_launch_count=launch_count,
            backward_launch_count=(
                ad_companion_launches if config.ad_mode == "vjp" else 0
            ),
            jvp_launch_count=(
                ad_companion_launches if config.ad_mode == "jvp" else 0
            ),
            tape_bytes=ad_tape_bytes if config.ad_mode == "vjp" else 0,
            accumulation_strategy="atomic_add",
            scheduling_strategy="native_fused"
            if raydn_component_enabled
            else "native_cuda",
            raydn_native=capability["raydn_native"],
            ad_status=config.ad_mode,
            forward_time_ms=forward_time_ms,
            peak_memory_bytes=peak_memory_bytes,
        ),
        "field_abi": "complex3_v1",
        "phase_convention": dict(PHASE_CONVENTION),
        "coefficient_semantics": "unit_excitation_dimensionless_receiver_projection",
        # Coupled reflection-diffraction compensator (ADR-011); mirrors the
        # path solver's coupled_paths metadata block.
        "coupled_paths": _coupled_paths_metadata(config),
    }
    if metadata_transmission is not None:
        metadata["transmission"] = metadata_transmission
    if scattering_info is not None:
        # Incoherent Kirchhoff patch quadrature (plan 05 wave 3); the flag
        # documents that per-path phases are NOT physical for ensemble rows.
        metadata["scattering"] = dict(scattering_info)
    metadata.update(
        config_metadata(
            requested=requested_config,
            effective=effective_config,
            component_max_depth={
                "los": 0 if "los" in config.components else -1,
                "reflection": config.max_depth
                if "reflection" in config.components
                else -1,
                "diffraction": 1 if "diffraction" in config.components else -1,
                # transmission chains are capped like reflection; scattering is
                # single-bounce in v1.
                "transmission": config.max_depth
                if "transmission" in config.components
                else -1,
                "scattering": 1 if "scattering" in config.components else -1,
            },
        )
    )
    metadata["semantic_capabilities"] = capabilities()["solvers"]["deterministic"]
    return metadata


def solve(scene: Scene, config: Config) -> Result:
    validate_scalar_endpoint_features(
        scene.transmitters, scene.receivers, solver="deterministic"
    )
    if not torch.cuda.is_available():
        raise RuntimeError("witwin.channel_native.deterministic requires CUDA")
    # Solve-level wall time and CUDA high-water-mark delta for the kernel
    # metadata (plan 07 AD-4). This is AD instrumentation only: the leading
    # synchronize would stall the host on the caller's queued work and the
    # trailing one drains the solve before returning, which an optimization
    # loop over ad_mode="none" must not pay. none-mode reports zeros and takes
    # no sync, preserving the byte-identical zero-overhead primal contract.
    ad_instrumented = config.ad_mode != "none"
    solve_start = 0.0
    peak_before = 0
    if ad_instrumented:
        torch.cuda.synchronize()
        solve_start = perf_counter()
        peak_before = torch.cuda.max_memory_allocated()

    native_info = build_info()
    _validate_requested_components(config)
    if "reflection" in config.components and not native_info["uses_raydn_native"]:
        raise RuntimeError("deterministic reflection requires RayDN native capability")
    if "diffraction" in config.components and not native_info["uses_raydn_native"]:
        raise RuntimeError("deterministic diffraction requires RayDN native capability")
    has_grid = any(
        receiver.__class__.__name__ == "ReceiverGrid" for receiver in scene.receivers
    )
    if has_grid and len(scene.receivers) > 1 and not config.export_paths:
        raise RuntimeError("mixed point/grid receivers require export_paths=True")

    device = torch.device("cuda")
    _, layout = receiver_positions_and_layout(scene, device=device)
    # One host read of a tensor frequency for the whole solve: topology
    # export, field evaluation, accumulation and path export share it
    # (audit M3). Scene.compile() keeps its own read: the material cache
    # token must see the live value to stay correct under in-place
    # frequency mutation.
    frequency_hz = _frequency_scalar(scene)
    evaluated, sidecars = evaluate_enumerated_paths(
        scene,
        config,
        frequency_value=frequency_hz,
        # Stream coupled discovery over receiver blocks so a full grid solve
        # stays under the per-block candidate budget (ADR-011).
        coupled_rx_streaming=config.coupled_paths,
    )
    scattering_info = None
    if "scattering" in config.components:
        evaluated, sidecars, scattering_info = append_scattering_evaluated_paths(
            scene,
            config,
            evaluated,
            sidecars,
        )
    topology = evaluated.topology
    path_count = evaluated.row_count
    component_counts = deterministic_component_counts(topology.component_id)
    # The native counter materializes only los/reflection/diffraction slots.
    for name, cid in (("transmission", 5), ("scattering", 6)):
        if name in config.components:
            component_counts[name] = int(
                (topology.component_id == cid).sum().item()
            )
    extra_components = tuple(
        name for name in _OPTIONAL_COMPONENTS if name in config.components
    )
    extra_components = _register_coupled_component(
        config, topology, component_counts, extra_components
    )
    path_gain, field, component_power, component_fields = accumulate_path_result(
        evaluated,
        frequency_hz=frequency_hz,
        num_tx=len(scene.transmitters),
        num_rx=layout.receiver_count,
        layout=layout,
        coherent=config.coherent,
        return_field=config.return_field,
        extra_components=extra_components,
        # AD modes run the same native accumulator inside its dispatch-only
        # autograd.Function so Result.path_gain/field/component_power carry
        # the complete graph; none-mode keeps the bare zero-overhead kernel.
        differentiable=config.ad_mode != "none",
    )
    exact_diffraction = None
    if (
        sidecars.diffraction_vector_field is not None
        and len(scene.receivers) == 1
        and isinstance(scene.receivers[0], ReceiverGrid)
    ):
        vector_field = sidecars.diffraction_vector_field
        exact_diffraction_flat = vector_field.abs().square().sum(dim=-1)
        exact_diffraction = apply_receiver_layout(exact_diffraction_flat, layout)
        previous_diffraction = component_power["diffraction"]
        component_power["diffraction"] = exact_diffraction
        if not config.coherent:
            path_gain = path_gain - previous_diffraction + exact_diffraction
    if ad_instrumented:
        torch.cuda.synchronize()
        forward_time_ms = (perf_counter() - solve_start) * 1.0e3
        peak_memory_bytes = max(0, torch.cuda.max_memory_allocated() - peak_before)
    else:
        forward_time_ms = 0.0
        peak_memory_bytes = 0
    metadata = _metadata(
        config=config,
        native_info=native_info,
        path_count=path_count,
        component_counts=component_counts,
        launch_count=sidecars.execution.launch_count,
        ad_companion_launches=sidecars.execution.ad_companion_launches,
        ad_tape_bytes=sidecars.execution.ad_tape_bytes,
        forward_time_ms=forward_time_ms,
        peak_memory_bytes=peak_memory_bytes,
        scattering_info=scattering_info,
    )
    diagnostics = None
    if config.diagnostics:
        candidate_count = sidecars.execution.candidate_count
        diagnostics = {
            "path_gain_shape": tuple(path_gain.shape),
            "field_shape": tuple(field.shape),
            "path_count": path_count,
            "component_counts": component_counts,
            "coherent": config.coherent,
            "accumulation_mode": "coherent" if config.coherent else "incoherent",
            "native_launch_count": sidecars.execution.launch_count,
            "diffraction_accumulation": (
                "exact_vector_coherent_paths"
                if exact_diffraction is not None
                else "scalar_path_accumulation"
            ),
            "visibility_rejection_count": (
                sidecars.execution.visibility_rejection_count
            ),
            "selected_edge_count": sidecars.execution.selected_edge_count,
            "path_planning": {
                "max_paths": config.max_paths,
                "max_paths_scope": config.max_paths_scope,
                "candidate_count": candidate_count,
                "guardrail_count": sidecars.execution.guardrail_count,
                "truncated": config.max_paths is not None
                and path_count < candidate_count,
            },
        }
    return Result(
        path_gain=path_gain,
        field=field,
        component_power=component_power,
        component_fields=component_fields,
        paths=(
            build_path_table(
                evaluated,
                frequency_hz=frequency_hz,
                include_fields=config.return_field,
            )
            if config.export_paths
            else None
        ),
        metadata=metadata,
        diagnostics=diagnostics,
    )
