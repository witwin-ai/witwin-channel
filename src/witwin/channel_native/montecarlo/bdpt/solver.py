from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from witwin.channel_native import Scene
from witwin.channel_native import ReceiverGrid
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.core.materials import effective_sigma_e
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
    bdpt_face_material_tensors_from_host,
    bdpt_finalize_component_maps,
    bdpt_finalize_point_components,
    bdpt_filter_connection_samples,
    bdpt_intersect_forward,
    bdpt_reflection_accumulation_forward,
    bdpt_reflection_launch_inputs,
    bdpt_reflected_light_subpath_state,
    bdpt_connection_variance,
    bdpt_los_component_maps_from_matrix,
    bdpt_sample_directions,
    bdpt_selected_edge_indices,
    bdpt_subpath_intersection_inputs,
    bdpt_zero_matrix,
    mc_component_map_buffer,
    mc_sample_directions,
    mc_store_component_map,
    mc_store_scaled_component_map,
    raydn_visibility_forward,
)
from witwin.channel_native.path.raydn_export import _diffraction_edge_geometry
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


def _estimate_workspace_bytes(config: Config, *, tx_count: int, grid_cells: int, rx_count: int) -> int:
    launch_entries = max(0, int(tx_count)) * int(config.samples) * int(config.sample_streams)
    bytes_estimate = launch_entries * 32
    if grid_cells > 0:
        bytes_estimate += tx_count * grid_cells * 3 * 4
    if config.export_paths:
        exported = config.max_exported_paths if config.max_exported_paths is not None else launch_entries
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
    if workspace_bytes > config.workspace_limit_bytes:
        raise RuntimeError(
            "workspace limit exceeded for BDPT: "
            f"estimated {workspace_bytes} bytes exceeds {config.workspace_limit_bytes}. "
            "Reduce samples or the receiver grid resolution, or raise "
            "Config.workspace_limit_bytes if the device has enough memory."
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
        cell_area=abs((coord0_max - coord0_min) * (coord1_max - coord1_min)) / float(rows * cols),
    )


def _face_material_tensors(scene: Scene) -> tuple[torch.Tensor, ...]:
    material_eps_r: list[float] = []
    material_sigma_e: list[float] = []
    material_mu_r: list[float] = []
    face_material_id: list[int] = []
    for material_id, structure in enumerate(scene.structures):
        params = structure.material.parameters()
        material_eps_r.append(float(params["eps_r"]))
        material_sigma_e.append(effective_sigma_e(params))
        material_mu_r.append(float(params["mu_r"]))
        face_material_id.extend([material_id] * int(structure.faces.shape[0]))
    if not material_eps_r:
        material_eps_r = [1.0]
        material_sigma_e = [0.0]
        material_mu_r = [1.0]
    exported = bdpt_face_material_tensors_from_host(
        tuple(material_eps_r),
        tuple(material_sigma_e),
        tuple(material_mu_r),
        tuple(face_material_id),
    )
    return (
        exported["eps_r"],
        exported["sigma_e"],
        exported["mu_r"],
        exported["gain"],
        exported["valid"],
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


def _path_samples_from_connection_export(exported: dict[str, torch.Tensor]) -> BDPTPathSamples:
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


def _native_reflection_connection_samples(
    raydn: Any,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    endpoint_subpaths: dict[str, dict[str, torch.Tensor]],
    rx_positions: torch.Tensor,
    material_tensors: tuple[torch.Tensor, ...],
    *,
    frequency_hz: float,
    samples: int,
    seed: int,
    mis: str,
    beta: float,
    strategy_count: int,
    max_depth: int,
) -> list[dict[str, torch.Tensor]]:
    eps_r, sigma_e, mu_r, material_gain, material_valid = material_tensors
    sensor = endpoint_subpaths["sensor"]
    sample_blocks: list[dict[str, torch.Tensor]] = []
    for tx_index in range(int(tx_positions.shape[0])):
        launch_inputs = bdpt_reflection_launch_inputs(tx_positions, tx_index=tx_index, sample_count=int(samples))
        ray_d = bdpt_sample_directions(int(samples), tx_positions, seed=int(seed) + tx_index * 65537)
        state = bdpt_endpoint_subpath_state(
            tx_positions,
            tx_power,
            rx_positions,
            launch_inputs["tx_id"],
            launch_inputs["light_seed"],
        )["light"]
        state["direction"] = ray_d
        ray_inputs = {
            "ray_o": launch_inputs["ray_o"],
            "ray_d": ray_d,
            "ray_tmax": launch_inputs["ray_tmax"],
            "active": launch_inputs["active"],
        }
        for _bounce in range(max(1, int(max_depth))):
            hit = bdpt_intersect_forward(
                raydn.require_handle(),
                ray_inputs["ray_o"],
                ray_inputs["ray_d"],
                ray_inputs["ray_tmax"],
                ray_inputs["active"],
            )
            reflected = bdpt_reflected_light_subpath_state(
                state,
                hit,
                material_gain=material_gain,
                material_valid=material_valid,
                material_eps_r=eps_r,
                material_sigma_e=sigma_e,
                material_mu_r=mu_r,
                frequency_hz=frequency_hz,
            )
            samples_out = bdpt_endpoint_connection_samples(
                reflected,
                sensor,
                frequency_hz=frequency_hz,
                samples_per_tx=int(samples),
                max_paths=None,
                mis=mis,
                beta=beta,
                strategy_count=strategy_count,
            )
            visibility_inputs = bdpt_endpoint_connection_visibility_inputs(
                reflected,
                sensor,
                sample_count=int(samples_out["valid"].shape[0]),
            )
            visible = raydn_visibility_forward(
                raydn.require_handle(),
                visibility_inputs["start"],
                visibility_inputs["end"],
                visibility_inputs["active"],
            )[0]
            sample_blocks.append(bdpt_filter_connection_samples(samples_out, visible))
            if not bool(reflected["valid"].any()):
                break
            state = reflected
            ray_inputs = bdpt_subpath_intersection_inputs(reflected)
    return sample_blocks


def _reduced_light_endpoint_state(
    tx_reference: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
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
    return bdpt_endpoint_subpath_state(tx_reference, tx_power, rx_positions, tx_ids, seeds)["light"]


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


def _native_reflection_component_maps(
    scene: Scene,
    raydn: Any,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    material_tensors: tuple[torch.Tensor, ...],
    grid: ReceiverGrid,
    *,
    samples: int,
    max_depth: int,
) -> torch.Tensor:
    material_eta_r, material_sigma, material_mu_r, material_gain, material_valid = material_tensors
    spec = _grid_spec(grid)
    dim0, dim1 = grid.shape[1], grid.shape[0]
    maps = mc_component_map_buffer(tx_positions, tx_count=tx_positions.shape[0], dim0=dim0, dim1=dim1)
    wavelength = _LIGHT_SPEED_M_PER_S / float(scene.frequency)
    solid_angle_per_ray = float(4.0 * math.pi / max(1, int(samples)))
    for tx_index in range(int(tx_positions.shape[0])):
        launch_inputs = bdpt_reflection_launch_inputs(tx_positions, tx_index=tx_index, sample_count=int(samples))
        ray_d = mc_sample_directions(int(samples), tx_positions)
        out = bdpt_reflection_accumulation_forward(
            raydn.require_handle(),
            launch_inputs["ray_o"],
            ray_d,
            launch_inputs["ray_tmax"],
            launch_inputs["active"],
            launch_inputs["ray_o"],
            launch_inputs["tx_pol"],
            material_eta_r,
            material_sigma,
            material_mu_r,
            material_gain,
            material_valid,
            int(max_depth),
            int(spec.axis),
            float(spec.position),
            float(spec.coord0_min),
            float(spec.coord0_max),
            float(spec.coord1_min),
            float(spec.coord1_max),
            int(spec.resolution0),
            int(spec.resolution1),
            float(wavelength),
            float(solid_angle_per_ray),
            False,
            False,
            0,
            1,
            1,
            262144,
            64,
            0,
            False,
        )
        maps = mc_store_scaled_component_map(
            maps,
            out[0],
            tx_power,
            tx_index=tx_index,
            scale_index=tx_index,
        )
    return maps


def _diffraction_sample_split(sample_count: int) -> tuple[int, int, int]:
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
    _eps_r, _sigma_e, _mu_r, material_gain, material_valid = material_tensors
    spec = _grid_spec(grid)
    dim0, dim1 = grid.shape[1], grid.shape[0]
    maps = mc_component_map_buffer(tx_positions, tx_count=tx_positions.shape[0], dim0=dim0, dim1=dim1)
    edge_geometry = _diffraction_edge_geometry(raydn)
    selected, edge_pos, edge_dir, _lengths, line_min, line_max, n0, n1, face0, face1, exterior_angle = edge_geometry
    edge_indices = bdpt_selected_edge_indices(selected)
    wavelength = _LIGHT_SPEED_M_PER_S / float(scene.frequency)
    # Grid maps use the Keller-cone strategy exclusively (audit MC-5/2.5,
    # aligned with the basic solver): the deterministic direct cell scan
    # cannot cover state_count x cell_count pairs within the sample budget,
    # and the unweighted direct+keller sum double counts covered cells.
    direct_samples, keller_samples, suffix_samples = 0, int(samples), 0
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
        maps = mc_store_component_map(maps, out[0], tx_index=tx_index)
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
                    strategy_count=_diffraction_strategy_count(direct_samples, keller_samples),
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
) -> list[dict[str, torch.Tensor]]:
    _eps_r, _sigma_e, _mu_r, material_gain, material_valid = material_tensors
    edge_geometry = _diffraction_edge_geometry(raydn)
    selected, edge_pos, edge_dir, _lengths, line_min, line_max, n0, n1, face0, face1, exterior_angle = edge_geometry
    edge_indices = bdpt_selected_edge_indices(selected)
    wavelength = _LIGHT_SPEED_M_PER_S / float(scene.frequency)
    direct_samples, keller_samples, _suffix_samples = _diffraction_sample_split(int(samples))
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
        exported = bdpt_diffraction_point_connection_samples(
            rx_positions,
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
            strategy_count=_diffraction_strategy_count(direct_samples, keller_samples),
        )
        samples_out = exported["samples"]
        if not isinstance(samples_out, dict):
            raise RuntimeError("native BDPT diffraction point sampler returned invalid samples")
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
        sample_blocks.append(bdpt_filter_connection_samples(filtered, visible_target))
    return sample_blocks


def solve(scene: Scene, config: Config) -> Result:
    if not torch.cuda.is_available():
        raise RuntimeError("witwin.channel_native.montecarlo.bdpt requires CUDA")

    info = build_info()
    raydn = scene.raydn_scene()
    raydn_available = bool(info["uses_raydn_native"]) and raydn.available
    reflection_available = raydn_available
    diffraction_available = raydn_available and config.max_diffraction_order > 0
    if "diffraction" in config.components and config.max_diffraction_order <= 0:
        raise RuntimeError("diffraction requires max_diffraction_order > 0")
    if "diffraction" in config.components and config.mis == "none":
        direct_samples, keller_samples, _suffix = _diffraction_sample_split(_effective_native_samples(config))
        if _diffraction_strategy_count(direct_samples, keller_samples) > 1:
            raise RuntimeError(
                "mis='none' double counts the direct+keller diffraction strategies; "
                "use mis='balance' or 'power_heuristic'"
            )
    if "reflection" in config.components and scene.structures and not reflection_available:
        raise RuntimeError("reflection requires RayDN native capability")
    if "diffraction" in config.components and scene.structures and not diffraction_available:
        raise RuntimeError("diffraction requires RayDN native capability")

    device = torch.device("cuda")
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

    tx_reference, tx_power = transmitter_tensors(scene)
    rx_positions = receiver_positions(scene, reference=tx_reference, grid=grid)
    launch_state = make_launch_state(tx_reference, tx_count=len(scene.transmitters), config=config)
    endpoint_subpaths = bdpt_endpoint_subpath_state(
        tx_reference,
        tx_power,
        rx_positions,
        launch_state["tx_id"],
        launch_state["light_seed"],
    )
    los_light_state = (
        _reduced_light_endpoint_state(tx_reference, tx_power, rx_positions)
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
    endpoint_only = endpoint_accumulation is not None and config.components == frozenset({"los"})

    def zero_component_matrix() -> torch.Tensor:
        return bdpt_zero_matrix(tx_reference, rows=tx_count, cols=rx_count)

    estimate_samples: dict[str, torch.Tensor] | None = None
    reflection_component_map: torch.Tensor | None = None
    diffraction_component_map: torch.Tensor | None = None
    if endpoint_only:
        component_matrices = {
            "los": endpoint_accumulation["los"],
            "reflection": zero_component_matrix(),
            "diffraction": zero_component_matrix(),
        }
        estimate_samples = endpoint_connection_samples
    else:
        sample_blocks: list[dict[str, torch.Tensor]] = []
        material_tensors = (
            _face_material_tensors(scene)
            if scene.structures and (("reflection" in config.components) or ("diffraction" in config.components))
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
            # The plane-capture accumulation kernel is the reflection
            # estimator for grids; path export runs the connection sampler in
            # addition, never instead (the map must not change with export).
            if grid is not None:
                reflection_component_map = _native_reflection_component_maps(
                    scene,
                    raydn,
                    tx_reference,
                    tx_power,
                    material_tensors,
                    grid,
                    samples=native_samples,
                    max_depth=native_max_depth,
                )
            if grid is None or config.export_paths:
                sample_blocks.extend(
                    _native_reflection_connection_samples(
                        raydn,
                        tx_reference,
                        tx_power,
                        endpoint_subpaths,
                        rx_positions,
                        material_tensors,
                        frequency_hz=float(scene.frequency),
                        samples=native_samples,
                        seed=int(config.seed),
                        mis=config.mis,
                        beta=config.power_heuristic_beta,
                        strategy_count=1,
                        max_depth=native_max_depth,
                    )
                )
            launch_count += 4
        if diffraction_requested:
            if grid is None:
                sample_blocks.extend(
                    _native_diffraction_point_connection_samples(
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
                    )
                )
            else:
                diffraction_component_map, diffraction_sample_blocks = _native_diffraction_component_maps(
                    scene,
                    raydn,
                    tx_reference,
                    tx_power,
                    grid,
                    material_tensors,
                    samples=native_samples,
                    seed=int(config.seed),
                    export_samples=bool(config.export_paths),
                    mis=config.mis,
                    beta=config.power_heuristic_beta,
                )
                sample_blocks.extend(diffraction_sample_blocks)
            launch_count += 3
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
                "los": accumulated["los"],
                "reflection": accumulated["reflection"],
                "diffraction": accumulated["diffraction"],
            }
        else:
            component_matrices = {
                "los": zero_component_matrix(),
                "reflection": zero_component_matrix(),
                "diffraction": zero_component_matrix(),
            }

    component_maps: dict[str, torch.Tensor] | None = None
    point_component_matrices: dict[str, torch.Tensor] | None = None
    if grid is not None:
        component_maps = _component_maps_from_matrices(
            component_matrices,
            rows=grid.shape[0],
            cols=grid.shape[1],
        )
        if diffraction_component_map is not None:
            component_maps["diffraction"] = diffraction_component_map
        if reflection_component_map is not None:
            component_maps["reflection"] = reflection_component_map
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
            path_samples = _path_samples_from_connection_export(exported_endpoint_samples)
        else:
            if estimate_samples is None:
                raise RuntimeError("BDPT path export requires native connection samples")
            exported_path_samples = bdpt_compact_connection_samples(
                estimate_samples,
                max_paths=config.max_exported_paths,
            )
            path_samples = _path_samples_from_connection_export(exported_path_samples)
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
    )
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
                None if component_maps is None else {key: tuple(value.shape) for key, value in component_maps.items()}
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
