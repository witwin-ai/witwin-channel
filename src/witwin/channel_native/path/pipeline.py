from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch

from witwin.channel_native import Scene, build_info
from witwin.channel_native.propagation.enumerated.engine import (
    evaluate_enumerated_paths,
)
from witwin.channel_native.propagation.enumerated.capacity import (
    sanitize_enumerated_capacity_transaction,
)
from witwin.channel_native.propagation.enumerated.scattering import (
    append_scattering_evaluated_paths,
)
from witwin.channel_native.propagation.models.evaluated import EvaluatedPaths
from witwin.channel_native.propagation.models.fields import PathFields
from witwin.channel_native.propagation.models.geometry import PathGeometry
from witwin.channel_native.propagation.models.topology import PathTopology
from witwin.channel_native.runtime.capacity import SolveCapacityTransaction
from witwin.channel_native.scene.tensors import (
    receiver_positions as _shared_receiver_positions,
    transmitter_positions as _shared_transmitter_positions,
)

from .arrays import (
    explicit_array_scene,
    pack_explicit_arrays,
    pack_synthetic_arrays,
    validate_synthetic_array_scene,
)
from .config import Config
from .metadata import _metadata
from .result import PathResult, from_evaluated_paths


_COMPONENT_ID = {
    "los": 0,
    "reflection": 1,
    "diffraction": 2,
    "reflection_diffraction": 3,
    "diffraction_reflection": 4,
    # transmission exports specular wall-penetration paths since wave 2;
    # scattering exports incoherent Kirchhoff patch paths since wave 3.
    "transmission": 5,
    "scattering": 6,
}


@dataclass(frozen=True, slots=True)
class _DeferredPathResult:
    """Internal result plus the one terminal check deferred past array packing."""

    result: PathResult
    capacity_transaction: SolveCapacityTransaction | None


def _compact_valid_evaluated_paths_for_legacy_result(
    paths: EvaluatedPaths,
) -> EvaluatedPaths:
    """Select valid rows before the legacy Ragged Path result pack.

    This structural bridge is deleted by the ADR-029 Phase D capacity-result
    switch.  It performs no geometry or RF computation; it only prevents inert
    failure rows from reaching the legacy endpoint-id gathers.
    """

    selected = torch.nonzero(paths.topology.valid, as_tuple=False).reshape(-1)
    topology = PathTopology(
        **{
            name: getattr(paths.topology, name).index_select(0, selected).contiguous()
            for name in (
                "valid",
                "tx_id",
                "rx_id",
                "depth",
                "component_id",
                "primitive_id",
                "edge_id",
                "material_id",
                "primitive_sequence",
                "material_sequence",
                "interaction_type",
            )
        }
    )
    geometry = PathGeometry(
        row_identity=topology.row_identity,
        **{
            name: getattr(paths.geometry, name).index_select(0, selected).contiguous()
            for name in (
                "path_length_m",
                "delay_s",
                "field_direction",
                "interaction_position",
                "interaction_normal",
                "interaction_positions",
                "interaction_normals",
            )
        },
    )
    fields = PathFields(
        row_identity=topology.row_identity,
        **{
            name: getattr(paths.fields, name).index_select(0, selected).contiguous()
            for name in ("path_gain", "path_field", "field_xyz", "coefficient")
        },
    )
    return EvaluatedPaths(topology=topology, geometry=geometry, fields=fields)


def _transmitter_tensors(scene: Scene) -> tuple[torch.Tensor, torch.Tensor]:
    return _shared_transmitter_positions(scene, device=torch.device("cuda"))


def _receiver_positions(scene: Scene, *, reference: torch.Tensor) -> torch.Tensor:
    return _shared_receiver_positions(
        scene, device=reference.device, reference=reference
    )


def _validate_runtime(config: Config) -> tuple[bool, bool, bool]:
    if not torch.cuda.is_available():
        raise RuntimeError("witwin.channel_native.path solver requires CUDA")
    if config.isb_boundary_taper and config.ad_mode != "none":
        # ISB boundary taper (ADR-017) gate 3: the C1 clearance-factor AD
        # companion is a documented follow-up; reject taper + AD loudly rather
        # than returning a silently incomplete gradient.
        raise RuntimeError(
            "isb_boundary_taper does not support ad_mode != 'none' yet "
            "(ADR-017 gate 3 C1 clearance companion is a follow-up)"
        )
    info = build_info()
    reflection_available = bool(info["uses_rayd_native"])
    diffraction_available = bool(info["uses_rayd_native"])
    path_native_available = bool(info.get("uses_path_native", False))
    if not path_native_available:
        raise RuntimeError(
            "path solver requires _channel_native path native CUDA kernels"
        )
    if "reflection" in config.components and not reflection_available:
        raise RuntimeError("reflection paths require RayD native capability")
    if "diffraction" in config.components and not diffraction_available:
        raise RuntimeError("diffraction paths require RayD native capability")
    if config.max_depth < 1 and (
        "reflection" in config.components or "diffraction" in config.components
    ):
        raise RuntimeError("requested scattering paths require max_depth >= 1")
    return reflection_available, diffraction_available, path_native_available


def _solve_base(
    scene: Scene,
    config: Config,
    *,
    validate_runtime: Callable[[Config], tuple[bool, bool, bool]] = _validate_runtime,
    evaluate_enumerated_paths: Callable[..., Any] = evaluate_enumerated_paths,
    append_scattering_evaluated_paths: Callable[..., Any] = (
        append_scattering_evaluated_paths
    ),
    metadata: Callable[..., dict[str, Any]] = _metadata,
    transmitter_tensors: Callable[..., tuple[torch.Tensor, torch.Tensor]] = (
        _transmitter_tensors
    ),
    receiver_positions: Callable[..., torch.Tensor] = _receiver_positions,
    pack_evaluated_paths: Callable[..., PathResult] = from_evaluated_paths,
) -> _DeferredPathResult:
    reflection_available, diffraction_available, path_native_available = (
        validate_runtime(config)
    )
    # Solve-level wall time and CUDA high-water-mark delta for the kernel
    # metadata (plan 07 AD-4). AD instrumentation only: the syncs would break
    # host/device overlap for a caller looping over ad_mode="none" solves, so
    # none-mode reports zeros and takes no sync (zero-overhead primal contract).
    ad_instrumented = config.ad_mode != "none"
    solve_start = 0.0
    peak_before = 0
    if ad_instrumented:
        torch.cuda.synchronize()
        solve_start = perf_counter()
        peak_before = torch.cuda.max_memory_allocated()
    evaluated, sidecars = evaluate_enumerated_paths(
        scene, config, defer_capacity_terminal=True
    )
    scattering_info = None
    if "scattering" in config.components:
        evaluated, sidecars, scattering_info = append_scattering_evaluated_paths(
            scene,
            config,
            evaluated,
            sidecars,
        )
    evaluated, sidecars = sanitize_enumerated_capacity_transaction(evaluated, sidecars)
    evaluated = _compact_valid_evaluated_paths_for_legacy_result(evaluated)
    path_count = evaluated.row_count
    if ad_instrumented:
        torch.cuda.synchronize()
        forward_time_ms = (perf_counter() - solve_start) * 1.0e3
        peak_memory_bytes = max(0, torch.cuda.max_memory_allocated() - peak_before)
    else:
        forward_time_ms = 0.0
        peak_memory_bytes = 0
    result_metadata = metadata(
        config=config,
        path_count=path_count,
        reflection_available=reflection_available,
        diffraction_available=diffraction_available,
        path_native_available=path_native_available,
        transmission_path_count=int(
            (evaluated.topology.component_id == _COMPONENT_ID["transmission"])
            .sum()
            .item()
        ),
        scattering_path_count=int(
            (evaluated.topology.component_id == _COMPONENT_ID["scattering"])
            .sum()
            .item()
        ),
        ad_companion_launches=sidecars.execution.ad_companion_launches,
        ad_tape_bytes=sidecars.execution.ad_tape_bytes,
        forward_time_ms=forward_time_ms,
        peak_memory_bytes=peak_memory_bytes,
        scattering_info=scattering_info,
    )
    tx_positions, _tx_power = transmitter_tensors(scene)
    rx_positions = receiver_positions(scene, reference=tx_positions)
    return _DeferredPathResult(
        result=pack_evaluated_paths(
            evaluated,
            num_rx=int(rx_positions.shape[0]),
            num_tx=int(tx_positions.shape[0]),
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            metadata=result_metadata,
        ),
        capacity_transaction=sidecars.capacity_transaction,
    )


def solve(
    scene: Scene,
    config: Config,
    *,
    solve_base: Callable[[Scene, Config], _DeferredPathResult] = _solve_base,
) -> PathResult:
    """Solve canonical paths and pack synthetic or explicit antenna arrays."""

    endpoints = [*scene.transmitters, *scene.receivers]
    if any(not endpoint.synthetic_array for endpoint in endpoints):
        expanded_scene, num_rx_ant, num_tx_ant = explicit_array_scene(scene)
        deferred = solve_base(expanded_scene, config)
        result = pack_explicit_arrays(
            deferred.result,
            scene=scene,
            num_rx_ant=num_rx_ant,
            num_tx_ant=num_tx_ant,
        )
        if deferred.capacity_transaction is not None:
            deferred.capacity_transaction.terminal_check()
        return result
    validate_synthetic_array_scene(scene)
    deferred = solve_base(scene, config)
    result = pack_synthetic_arrays(
        deferred.result,
        frequency_hz=scene.frequency,
        transmitters=scene.transmitters,
        receivers=scene.receivers,
    )
    if deferred.capacity_transaction is not None:
        deferred.capacity_transaction.terminal_check()
    return result
