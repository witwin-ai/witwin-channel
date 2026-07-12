from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from witwin.channel_native import Scene
from witwin.channel_native import ReceiverGrid
from witwin.channel_native.core.edge_selection import resolve_scene_edge_policy
from witwin.channel_native.core.field_state import (
    receiver_polarizations,
    transmitter_polarizations,
)
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.core.material_runtime import face_material_field_bundle
from witwin.channel_native.core.memory_budget import (
    MemoryEstimate,
    enforce_memory_budget,
)
from witwin.channel_native.core.path_topology import export_topology
from witwin.channel_native.path import Config as PathConfig
from witwin.channel_native.core.kernels.ops import (
    bdpt_accumulate_connection_samples,
    bdpt_compact_connection_samples,
    bdpt_concat_connection_samples,
    bdpt_count_valid_connection_samples,
    bdpt_diffraction_accumulation_forward,
    bdpt_diffraction_connection_samples_from_tape,
    bdpt_diffraction_point_connection_samples,
    bdpt_diffraction_state_pack,
    bdpt_diffraction_state_wi,
    bdpt_endpoint_subpath_state,
    bdpt_endpoint_connection_samples,
    bdpt_endpoint_connection_visibility_inputs,
    bdpt_finalize_component_maps,
    bdpt_finalize_point_components,
    bdpt_filter_connection_samples,
    bdpt_connection_variance,
    bdpt_los_component_maps_from_matrix,
    bdpt_selected_edge_indices,
    bdpt_zero_matrix,
    mc_component_map_buffer,
    mc_store_component_map,
    raydn_visibility_forward,
)
from witwin.channel_native.montecarlo.basic.raydn_components import (
    _cached_diffraction_edge_geometry,
)
from .connections import (
    first_receiver_grid,
    receiver_positions,
    transmitter_tensors,
)

from .config import Config
from .metadata import make_solver_metadata, select_accumulation_strategy
from .result import BDPTPathSamples, Result
from .sampling import make_launch_state


_EXPORT_BYTES_PER_PATH = 96
_CONNECTION_BYTES_PER_ROW = 57
_VISIBILITY_BYTES_PER_ROW = 25
_LIGHT_SPEED_M_PER_S = 299_792_458.0


@dataclass(frozen=True, slots=True)
class _GridSpec:
    axis: int
    position: float
    coord0_min: float
    coord0_max: float
    coord1_min: float
    coord1_max: float
    resolution0: int
    resolution1: int
    cell_area: float


def _estimate_workspace_bytes(
    config: Config, *, tx_count: int, grid_cells: int, rx_count: int
) -> int:
    launch_entries = (
        max(0, int(tx_count)) * int(config.samples) * int(config.sample_streams)
    )
    bytes_estimate = launch_entries * 32
    if grid_cells > 0:
        bytes_estimate += tx_count * grid_cells * 3 * 4
    if config.export_paths:
        exported = (
            config.max_exported_paths
            if config.max_exported_paths is not None
            else launch_entries
        )
        bytes_estimate += int(exported) * _EXPORT_BYTES_PER_PATH

    # Dense endpoint-connection tables materialize light_count x rx_count rows
    # (contribution fields + visibility inputs). The LoS term connects one
    # deterministic endpoint per transmitter, so only the sampled reflection
    # and diffraction connection paths scale with the sample budget.
    row_bytes = _CONNECTION_BYTES_PER_ROW + _VISIBILITY_BYTES_PER_ROW
    rx_rows = max(0, int(rx_count))
    dense_rows = 0
    if "los" in config.components:
        dense_rows += max(0, int(tx_count)) * rx_rows
    point_receivers = grid_cells <= 0
    if "reflection" in config.components and (point_receivers or config.export_paths):
        dense_rows += launch_entries * rx_rows * max(1, int(config.max_depth))
    if "diffraction" in config.components and point_receivers:
        dense_rows += launch_entries * rx_rows
    bytes_estimate += dense_rows * row_bytes
    return int(bytes_estimate)


def _check_workspace_guardrail(config: Config, workspace_bytes: int) -> None:
    if config.workspace_limit_bytes is None:
        return
    enforce_memory_budget(
        MemoryEstimate(temporary_bytes=workspace_bytes),
        budget_bytes=config.workspace_limit_bytes,
        workload="workspace for BDPT",
    )


def _path_counts(config: Config, *, tx_count: int) -> dict[str, int]:
    return {
        component: int(config.samples) * int(config.sample_streams) * int(tx_count)
        if component in config.components
        else 0
        for component in ("los", "reflection", "diffraction")
    }


def _effective_native_samples(config: Config) -> int:
    return int(config.samples) * int(config.sample_streams)


def _effective_native_depth(config: Config) -> int:
    return min(
        int(config.max_depth),
        int(config.max_light_depth) + int(config.max_sensor_depth),
    )


def _vector3_values(vector: torch.Tensor) -> tuple[float, float, float]:
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _axis_index(values: tuple[float, float, float], *, name: str) -> tuple[int, float]:
    nonzero = [idx for idx, value in enumerate(values) if abs(value) > 1.0e-6]
    if len(nonzero) != 1:
        raise ValueError(f"{name} must be axis-aligned")
    index = nonzero[0]
    value = values[index]
    sign = 1.0 if value > 0.0 else -1.0
    if abs(abs(value) - 1.0) > 1.0e-5:
        raise ValueError(f"{name} must be a unit axis vector")
    return index, sign


def _grid_spec(grid: ReceiverGrid) -> _GridSpec:
    rows, cols = grid.shape
    origin = _vector3_values(grid.origin)
    axis0, sign0 = _axis_index(_vector3_values(grid.x_axis), name="ReceiverGrid.x_axis")
    axis1, sign1 = _axis_index(_vector3_values(grid.y_axis), name="ReceiverGrid.y_axis")
    if axis0 == axis1:
        raise ValueError("ReceiverGrid axes must be orthogonal")
    axis = ({0, 1, 2} - {axis0, axis1}).pop()
    expected = (1, 2) if axis == 0 else (0, 2) if axis == 1 else (0, 1)
    if (axis0, axis1) != expected:
        raise ValueError("ReceiverGrid axes must match RayDN grid coordinate order")

    step0 = float(grid.spacing[0]) * sign0
    step1 = float(grid.spacing[1]) * sign1
    first0 = origin[axis0]
    first1 = origin[axis1]
    last0 = first0 + step0 * float(rows - 1)
    last1 = first1 + step1 * float(cols - 1)
    half0 = abs(float(grid.spacing[0])) * 0.5
    half1 = abs(float(grid.spacing[1])) * 0.5
    coord0_min = min(first0, last0) - half0
    coord0_max = max(first0, last0) + half0
    coord1_min = min(first1, last1) - half1
    coord1_max = max(first1, last1) + half1
    return _GridSpec(
        axis=axis,
        position=origin[axis],
        coord0_min=coord0_min,
        coord0_max=coord0_max,
        coord1_min=coord1_min,
        coord1_max=coord1_max,
        resolution0=rows,
        resolution1=cols,
        cell_area=abs((coord0_max - coord0_min) * (coord1_max - coord1_min))
        / float(rows * cols),
    )


def _face_material_tensors(
    scene: Scene, *, device: torch.device
) -> tuple[torch.Tensor, ...]:
    bundle = face_material_field_bundle(scene, device=device)
    return (
        bundle["eps_r"],
        bundle["sigma_e"],
        bundle["mu_r"],
        bundle["gain"],
        bundle["valid"],
        bundle["thickness"],
    )


def _component_maps_from_matrices(
    component_matrices: dict[str, torch.Tensor],
    *,
    rows: int,
    cols: int,
) -> dict[str, torch.Tensor]:
    return {
        name: bdpt_los_component_maps_from_matrix(component, rows=rows, cols=cols)
        for name, component in component_matrices.items()
    }


def _path_samples_from_connection_export(
    exported: dict[str, torch.Tensor],
) -> BDPTPathSamples:
    return BDPTPathSamples(
        topology=exported["topology"],
        contribution=exported["contribution"],
        pdf=exported["pdf"],
        mis_weight=exported["mis_weight"],
        component_id=exported["component_id"],
        valid=exported["valid"],
        tx_id=exported["tx_id"],
        rx_id=exported["rx_id"],
        grid_linear_id=exported["grid_linear_id"],
        light_depth=exported["light_depth"],
        sensor_depth=exported["sensor_depth"],
        path_length_m=exported["path_length_m"],
    )


def _empty_path_samples(reference: torch.Tensor) -> BDPTPathSamples:
    device = reference.device
    empty_float = torch.empty((0,), device=device, dtype=torch.float32)
    empty_int = torch.empty((0,), device=device, dtype=torch.int32)
    return BDPTPathSamples(
        topology=torch.empty((0, 4), device=device, dtype=torch.int32),
        contribution=empty_float,
        pdf=empty_float.clone(),
        mis_weight=empty_float.clone(),
        component_id=empty_int,
        valid=torch.empty((0,), device=device, dtype=torch.bool),
        tx_id=empty_int.clone(),
        rx_id=empty_int.clone(),
        grid_linear_id=empty_int.clone(),
        light_depth=empty_int.clone(),
        sensor_depth=empty_int.clone(),
        path_length_m=empty_float.clone(),
    )


def _reduced_light_endpoint_state(
    tx_reference: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_positions: torch.Tensor,
    rx_polarization: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """One light endpoint per transmitter for the deterministic LoS term.

    All depth-0 light samples share the transmitter position, so N samples
    per tx are N identical rows weighted 1/N. Connecting the T unique
    endpoints with samples_per_tx=1 yields the identical estimate while the
    connection table shrinks from T*N*R rows to T*R (audit P-1/P-5).
    """

    device = tx_reference.device
    tx_count = int(tx_reference.shape[0])
    tx_ids = torch.arange(tx_count, device=device, dtype=torch.int32)
    seeds = torch.zeros((tx_count,), device=device, dtype=torch.int64)
    return bdpt_endpoint_subpath_state(
        tx_reference,
        tx_power,
        tx_polarization,
        rx_positions,
        rx_polarization,
        tx_ids,
        seeds,
    )["light"]


def _native_los_connection_samples(
    raydn: Any,
    light: dict[str, torch.Tensor],
    sensor: dict[str, torch.Tensor],
    *,
    scene_has_structures: bool,
    frequency_hz: float,
    mis: str,
    beta: float,
    strategy_count: int,
) -> dict[str, torch.Tensor]:
    samples = bdpt_endpoint_connection_samples(
        light,
        sensor,
        frequency_hz=frequency_hz,
        samples_per_tx=1,
        max_paths=None,
        mis=mis,
        beta=beta,
        strategy_count=strategy_count,
    )
    if not scene_has_structures:
        return samples
    visibility_inputs = bdpt_endpoint_connection_visibility_inputs(
        light,
        sensor,
        sample_count=int(samples["valid"].shape[0]),
    )
    visible = raydn_visibility_forward(
        raydn.require_handle(),
        visibility_inputs["start"],
        visibility_inputs["end"],
        visibility_inputs["active"],
    )[0]
    return bdpt_filter_connection_samples(samples, visible)


def _diffraction_sample_split(
    sample_count: int, *, mis: str
) -> tuple[int, int, int]:
    if mis == "none":
        return int(sample_count), 0, 0
    direct = (int(sample_count) + 2) // 3
    keller = (int(sample_count) + 1) // 3
    return direct, keller, 0


def _diffraction_strategy_count(direct_samples: int, keller_samples: int) -> int:
    return (1 if direct_samples > 0 else 0) + (1 if keller_samples > 0 else 0)


def _native_diffraction_component_maps(
    scene: Scene,
    raydn: Any,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    grid: ReceiverGrid,
    material_tensors: tuple[torch.Tensor, ...],
    *,
    samples: int,
    seed: int,
    export_samples: bool,
    mis: str,
    beta: float,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    _eps_r, _sigma_e, _mu_r, material_gain, material_valid, _thickness = material_tensors
    spec = _grid_spec(grid)
    dim0, dim1 = grid.shape[1], grid.shape[0]
    maps = mc_component_map_buffer(
        tx_positions, tx_count=tx_positions.shape[0], dim0=dim0, dim1=dim1
    )
    edge_geometry = _cached_diffraction_edge_geometry(raydn)
    (
        selected,
        edge_pos,
        edge_dir,
        _lengths,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
    ) = edge_geometry
    edge_indices = bdpt_selected_edge_indices(selected)
    wavelength = _LIGHT_SPEED_M_PER_S / float(scene.frequency)
    sample_blocks: list[dict[str, torch.Tensor]] = []

    for tx_index in range(int(tx_positions.shape[0])):
        states = bdpt_diffraction_state_pack(
            edge_indices,
            edge_pos,
            edge_dir,
            line_min,
            line_max,
            n0,
            n1,
            face0,
            face1,
            exterior_angle,
            tx_positions[tx_index],
            tx_power[tx_index],
        )
        state_count = int(states[0].shape[0])
        if state_count <= 0:
            continue
        state_wi = bdpt_diffraction_state_wi(states[1], states[10])
        cell_count = int(spec.resolution0) * int(spec.resolution1)
        direct_samples = int(samples) if int(samples) >= state_count * cell_count else 0
        keller_samples = 0 if direct_samples else int(samples)
        suffix_samples = 0
        out = bdpt_diffraction_accumulation_forward(
            raydn.require_handle(),
            None,
            *states,
            state_wi,
            state_wi,
            _eps_r,
            _sigma_e,
            _mu_r,
            material_gain,
            material_valid,
            state_count,
            int(spec.axis),
            float(spec.position),
            float(spec.coord0_min),
            float(spec.coord0_max),
            float(spec.coord1_min),
            float(spec.coord1_max),
            int(spec.resolution0),
            int(spec.resolution1),
            float(spec.cell_area),
            float(wavelength),
            int(direct_samples),
            int(keller_samples),
            int(suffix_samples),
            int(seed),
            1,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            1 if export_samples else 0,
            None,
            None,
        )
        diffraction_map = out[0] * float(cell_count) if direct_samples else out[0]
        maps = mc_store_component_map(
            maps,
            diffraction_map,
            tx_index=tx_index,
        )
        if export_samples:
            tape = {
                "active": out[14],
                "state_idx": out[15],
                "cell": out[16],
                "material_idx": out[17],
                "edge_u": out[18],
            }
            sample_blocks.append(
                bdpt_diffraction_connection_samples_from_tape(
                    tape,
                    states,
                    material_gain,
                    material_valid,
                    tx_index=tx_index,
                    state_count=state_count,
                    grid_axis=int(spec.axis),
                    grid_position=float(spec.position),
                    grid_coord0_min=float(spec.coord0_min),
                    grid_coord0_max=float(spec.coord0_max),
                    grid_coord1_min=float(spec.coord1_min),
                    grid_coord1_max=float(spec.coord1_max),
                    grid_resolution0=int(spec.resolution0),
                    grid_resolution1=int(spec.resolution1),
                    grid_cell_area=float(spec.cell_area),
                    wavelength=float(wavelength),
                    direct_samples=int(direct_samples),
                    keller_samples=int(keller_samples),
                    mis=mis,
                    beta=beta,
                    strategy_count=_diffraction_strategy_count(
                        direct_samples, keller_samples
                    ),
                )
            )
    return maps, sample_blocks


def _native_diffraction_point_connection_samples(
    scene: Scene,
    raydn: Any,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    material_tensors: tuple[torch.Tensor, ...],
    *,
    samples: int,
    seed: int,
    mis: str,
    beta: float,
) -> Iterator[dict[str, torch.Tensor]]:
    _eps_r, _sigma_e, _mu_r, material_gain, material_valid, _thickness = material_tensors
    edge_geometry = _cached_diffraction_edge_geometry(raydn)
    (
        selected,
        edge_pos,
        edge_dir,
        _lengths,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
    ) = edge_geometry
    edge_indices = bdpt_selected_edge_indices(selected)
    wavelength = _LIGHT_SPEED_M_PER_S / float(scene.frequency)
    direct_samples, keller_samples, _suffix_samples = _diffraction_sample_split(
        int(samples), mis=mis
    )
    for tx_index in range(int(tx_positions.shape[0])):
        states = bdpt_diffraction_state_pack(
            edge_indices,
            edge_pos,
            edge_dir,
            line_min,
            line_max,
            n0,
            n1,
            face0,
            face1,
            exterior_angle,
            tx_positions[tx_index],
            tx_power[tx_index],
        )
        state_count = int(states[0].shape[0])
        if state_count <= 0:
            continue
        for rx_start in range(0, int(rx_positions.shape[0]), 64):
            rx_end = min(rx_start + 64, int(rx_positions.shape[0]))
            exported = bdpt_diffraction_point_connection_samples(
                rx_positions[rx_start:rx_end],
                states,
                material_gain,
                material_valid,
                tx_index=tx_index,
                state_count=state_count,
                direct_samples=int(direct_samples),
                keller_samples=int(keller_samples),
                seed=int(seed) + tx_index * 104729,
                wavelength=float(wavelength),
                mis=mis,
                beta=beta,
                strategy_count=_diffraction_strategy_count(
                    direct_samples, keller_samples
                ),
            )
            samples_out = exported["samples"]
            if not isinstance(samples_out, dict):
                raise RuntimeError(
                    "native BDPT diffraction point sampler returned invalid samples"
                )
            if rx_start:
                samples_out["rx_id"].add_(rx_start)
                samples_out["grid_linear_id"].add_(rx_start)
                samples_out["topology"][:, 1].add_(rx_start)
            visible_source = raydn_visibility_forward(
                raydn.require_handle(),
                exported["source_start"],
                exported["source_end"],
                exported["visibility_active"],
            )[0]
            filtered = bdpt_filter_connection_samples(samples_out, visible_source)
            visible_target = raydn_visibility_forward(
                raydn.require_handle(),
                exported["target_start"],
                exported["target_end"],
                exported["visibility_active"],
            )[0]
            yield bdpt_filter_connection_samples(filtered, visible_target)


def _topology_connection_samples(
    topology: Any,
    selected: torch.Tensor,
    *,
    component_out: int,
) -> dict[str, torch.Tensor] | None:
    if int(selected.numel()) == 0:
        return None
    tx_id = topology.tx_id.index_select(0, selected).to(torch.int32)
    rx_id = topology.rx_id.index_select(0, selected).to(torch.int32)
    depth = topology.depth.index_select(0, selected).to(torch.int32)
    contribution = topology.path_gain.index_select(0, selected)
    count = int(selected.numel())
    component_id = torch.full_like(tx_id, int(component_out))
    one = torch.ones((count,), device=tx_id.device, dtype=torch.float32)
    valid = torch.ones((count,), device=tx_id.device, dtype=torch.bool)
    zero_depth = torch.zeros_like(depth)
    return {
        "topology": torch.stack((tx_id, rx_id, component_id, depth), dim=1),
        "contribution": contribution,
        "pdf": one,
        "mis_weight": one,
        "component_id": component_id,
        "valid": valid,
        "tx_id": tx_id,
        "rx_id": rx_id,
        "grid_linear_id": rx_id.clone(),
        "light_depth": depth,
        "sensor_depth": zero_depth,
        "path_length_m": topology.path_length_m.index_select(0, selected),
    }


def _reflection_discrete_connection_samples(
    scene: Scene, config: Config
) -> dict[str, torch.Tensor] | None:
    """Enumerate delta-specular paths with unit forward/reverse discrete mass."""

    topology = export_topology(
        scene,
        PathConfig(
            max_depth=int(config.max_depth),
            components={"reflection"},
        ),
    )
    selected = torch.nonzero(topology.component_id == 1, as_tuple=False).flatten()
    return _topology_connection_samples(topology, selected, component_out=1)


def _coupled_discrete_connection_samples(
    scene: Scene, config: Config
) -> dict[str, torch.Tensor] | None:
    """Enumerate mixed delta/UTD paths with unit bidirectional discrete mass."""

    topology = export_topology(
        scene,
        PathConfig(
            max_depth=int(config.max_depth),
            components={"reflection", "diffraction"},
            coupled_paths=True,
            coupled_candidate_limit=int(config.coupled_candidate_limit),
        ),
    )
    selected = torch.nonzero(topology.component_id >= 3, as_tuple=False).flatten()
    return _topology_connection_samples(topology, selected, component_out=2)


def solve(scene: Scene, config: Config) -> Result:
    grid = first_receiver_grid(scene)
    if grid is not None and config.receiver_strategy != "grid_area":
        raise RuntimeError("receiver_strategy='point_sphere' requires point receivers")

    grid_cells = 0 if grid is None else int(grid.shape[0] * grid.shape[1])
    native_samples = _effective_native_samples(config)
    native_max_depth = _effective_native_depth(config)
    selected_accumulation = select_accumulation_strategy(
        config,
        grid_cells=grid_cells,
        estimated_valid_ratio=1.0 if "los" in config.components else 0.05,
    )
    rx_count_estimate = grid_cells if grid is not None else len(scene.receivers)
    workspace_bytes = _estimate_workspace_bytes(
        config,
        tx_count=len(scene.transmitters),
        grid_cells=grid_cells,
        rx_count=rx_count_estimate,
    )
    _check_workspace_guardrail(config, workspace_bytes)

    if not torch.cuda.is_available():
        raise RuntimeError("witwin.channel_native.montecarlo.bdpt requires CUDA")

    info = build_info()
    raydn = scene.raydn_scene()
    raydn_available = bool(info["uses_raydn_native"]) and raydn.available
    reflection_available = raydn_available
    diffraction_available = raydn_available and config.max_diffraction_order > 0
    if (
        "reflection" in config.components
        and scene.structures
        and not reflection_available
    ):
        raise RuntimeError("reflection requires RayDN native capability")
    if (
        "diffraction" in config.components
        and scene.structures
        and not diffraction_available
    ):
        raise RuntimeError("diffraction requires RayDN native capability")

    tx_reference, tx_power = transmitter_tensors(scene)
    rx_positions = receiver_positions(scene, reference=tx_reference, grid=grid)
    topology_scene = (
        scene
        if grid is None
        else Scene(
            structures=scene.structures,
            transmitters=scene.transmitters,
            receivers=[grid],
            frequency=scene.frequency,
            metadata=scene.metadata,
        )
    )
    tx_polarization = transmitter_polarizations(scene, device=tx_reference.device)
    rx_polarization = receiver_polarizations(
        scene, device=tx_reference.device, grid=grid
    )
    launch_state = make_launch_state(
        tx_reference, tx_count=len(scene.transmitters), config=config
    )
    endpoint_subpaths = bdpt_endpoint_subpath_state(
        tx_reference,
        tx_power,
        tx_polarization,
        rx_positions,
        rx_polarization,
        launch_state["tx_id"],
        launch_state["light_seed"],
    )
    los_light_state = (
        _reduced_light_endpoint_state(
            tx_reference,
            tx_power,
            tx_polarization,
            rx_positions,
            rx_polarization,
        )
        if "los" in config.components
        else None
    )
    endpoint_connection_samples = None
    endpoint_accumulation = None
    if los_light_state is not None and not scene.structures:
        endpoint_connection_samples = bdpt_endpoint_connection_samples(
            los_light_state,
            endpoint_subpaths["sensor"],
            frequency_hz=float(scene.frequency),
            samples_per_tx=1,
            max_paths=None,
            mis=config.mis,
            beta=config.power_heuristic_beta,
            strategy_count=1,
        )
        endpoint_accumulation = bdpt_accumulate_connection_samples(
            endpoint_connection_samples,
            tx_count=len(scene.transmitters),
            rx_count=int(rx_positions.shape[0]),
            accumulation_strategy=selected_accumulation,
        )
    launch_count = 1

    tx_count = len(scene.transmitters)
    rx_count = int(rx_positions.shape[0])
    endpoint_only = (
        endpoint_accumulation is not None and config.components == frozenset({"los"})
    )

    def zero_component_matrix() -> torch.Tensor:
        return bdpt_zero_matrix(tx_reference, rows=tx_count, cols=rx_count)

    estimate_samples: dict[str, torch.Tensor] | None = None
    if endpoint_only:
        component_matrices = {
            "los": endpoint_accumulation["los"],
            "reflection": zero_component_matrix(),
            "diffraction": zero_component_matrix(),
        }
        estimate_samples = endpoint_connection_samples
    else:
        sample_blocks: list[dict[str, torch.Tensor]] = []
        streamed_matrices = {
            "los": zero_component_matrix(),
            "reflection": zero_component_matrix(),
            "diffraction": zero_component_matrix(),
        }
        material_tensors = (
            _face_material_tensors(scene, device=tx_reference.device)
            if scene.structures
            and (
                ("reflection" in config.components)
                or ("diffraction" in config.components)
            )
            else None
        )
        if los_light_state is not None:
            sample_blocks.append(
                _native_los_connection_samples(
                    raydn,
                    los_light_state,
                    endpoint_subpaths["sensor"],
                    scene_has_structures=bool(scene.structures),
                    frequency_hz=float(scene.frequency),
                    mis=config.mis,
                    beta=config.power_heuristic_beta,
                    strategy_count=1,
                )
            )
            launch_count += 2 if scene.structures else 1
        reflection_requested = (
            "reflection" in config.components
            and reflection_available
            and material_tensors is not None
            and native_max_depth >= 1
        )
        diffraction_requested = (
            "diffraction" in config.components
            and diffraction_available
            and material_tensors is not None
            and native_max_depth >= 1
        )
        if reflection_requested:
            reflection_samples = _reflection_discrete_connection_samples(
                topology_scene, config
            )
            if reflection_samples is not None:
                sample_blocks.append(reflection_samples)
                launch_count += 1
        if diffraction_requested:
            retain_diffraction_samples = (
                grid is None or config.export_paths or config.diagnostics
            )
            for diffraction_samples in _native_diffraction_point_connection_samples(
                scene,
                raydn,
                tx_reference,
                tx_power,
                rx_positions,
                material_tensors,
                samples=native_samples,
                seed=int(config.seed),
                mis=config.mis,
                beta=config.power_heuristic_beta,
            ):
                if retain_diffraction_samples:
                    sample_blocks.append(diffraction_samples)
                    continue
                chunk = bdpt_accumulate_connection_samples(
                    diffraction_samples,
                    tx_count=tx_count,
                    rx_count=rx_count,
                    accumulation_strategy=selected_accumulation,
                )
                for component in streamed_matrices:
                    streamed_matrices[component].add_(chunk[component])
            launch_count += 3
        if config.coupled_paths:
            coupled_samples = _coupled_discrete_connection_samples(
                topology_scene, config
            )
            if coupled_samples is not None:
                sample_blocks.append(coupled_samples)
                launch_count += 1
        if sample_blocks:
            estimate_samples = (
                sample_blocks[0]
                if len(sample_blocks) == 1
                else bdpt_concat_connection_samples(tuple(sample_blocks))
            )
            accumulated = bdpt_accumulate_connection_samples(
                estimate_samples,
                tx_count=tx_count,
                rx_count=rx_count,
                accumulation_strategy=selected_accumulation,
            )
            component_matrices = {
                component: accumulated[component] + streamed_matrices[component]
                for component in streamed_matrices
            }
        else:
            component_matrices = streamed_matrices

    component_maps: dict[str, torch.Tensor] | None = None
    point_component_matrices: dict[str, torch.Tensor] | None = None
    if grid is not None:
        component_maps = _component_maps_from_matrices(
            component_matrices,
            rows=grid.shape[0],
            cols=grid.shape[1],
        )
        finalized = bdpt_finalize_component_maps(
            component_maps["los"],
            component_maps["reflection"],
            component_maps["diffraction"],
        )
    else:
        point_component_matrices = component_matrices
        finalized = bdpt_finalize_point_components(
            point_component_matrices["los"],
            point_component_matrices["reflection"],
            point_component_matrices["diffraction"],
        )
    path_gain = finalized["path_gain"]
    component_power = {
        "los": finalized["los_power"],
        "reflection": finalized["reflection_power"],
        "diffraction": finalized["diffraction_power"],
    }

    variance = None
    if config.diagnostics:
        if estimate_samples is None:
            variance = bdpt_zero_matrix(tx_reference, rows=tx_count, cols=rx_count)
        else:
            variance = bdpt_connection_variance(
                estimate_samples,
                tx_count=tx_count,
                rx_count=rx_count,
                # LoS-only estimates are one deterministic row per (tx, rx)
                # connection, not native_samples Monte Carlo draws.
                samples_per_tx=1 if endpoint_only else native_samples,
            )
        if grid is not None:
            variance = bdpt_los_component_maps_from_matrix(
                variance,
                rows=grid.shape[0],
                cols=grid.shape[1],
            )
    path_samples = None
    if config.export_paths:
        if endpoint_only:
            exported_endpoint_samples = bdpt_endpoint_connection_samples(
                los_light_state,
                endpoint_subpaths["sensor"],
                frequency_hz=float(scene.frequency),
                samples_per_tx=1,
                max_paths=config.max_exported_paths,
                mis=config.mis,
                beta=config.power_heuristic_beta,
                strategy_count=1,
            )
            path_samples = _path_samples_from_connection_export(
                exported_endpoint_samples
            )
        else:
            if estimate_samples is None:
                path_samples = _empty_path_samples(tx_reference)
            else:
                exported_path_samples = bdpt_compact_connection_samples(
                    estimate_samples,
                    max_paths=config.max_exported_paths,
                )
                path_samples = _path_samples_from_connection_export(
                    exported_path_samples
                )
    valid_contribution_count = (
        bdpt_count_valid_connection_samples(estimate_samples)
        if estimate_samples is not None
        else (int(path_gain.numel()) if config.components else 0)
    )
    path_counts_by_strategy = _path_counts(config, tx_count=len(scene.transmitters))
    metadata = make_solver_metadata(
        config=config,
        selected_accumulation_strategy=selected_accumulation,
        path_counts_by_strategy=path_counts_by_strategy,
        valid_contribution_count=valid_contribution_count,
        reflection_available=reflection_available,
        diffraction_available=diffraction_available,
        cuda_available=bool(info.get("cuda_available", True)),
        optix_available=bool(info.get("optix_available", False)),
        workspace_bytes=workspace_bytes,
        variance_enabled=variance is not None,
        launch_count=launch_count,
        effective_max_depth=native_max_depth,
    )
    edge_policy = resolve_scene_edge_policy(scene)
    metadata["edge_policy"] = {
        "edge_selection_mode": edge_policy.edge_selection_mode,
        "edge_diffraction": bool(edge_policy.edge_diffraction),
        "boundary_edge_policy": edge_policy.boundary_edge_policy,
    }
    diagnostics: dict[str, Any] | None = None
    if config.diagnostics:
        diagnostics = {
            "path_gain_shape": tuple(path_gain.shape),
            "device": str(path_gain.device),
            "launch_state_shape": tuple(launch_state["tx_id"].shape),
            "endpoint_subpath_shapes": {
                "light": tuple(endpoint_subpaths["light"]["origin"].shape),
                "sensor": tuple(endpoint_subpaths["sensor"]["origin"].shape),
            },
            "component_map_shapes": (
                None
                if component_maps is None
                else {key: tuple(value.shape) for key, value in component_maps.items()}
            ),
        }
    return Result(
        path_gain=path_gain,
        component_power=component_power,
        metadata=metadata,
        diagnostics=diagnostics,
        component_maps=component_maps,
        variance=variance,
        path_samples=path_samples,
    )
