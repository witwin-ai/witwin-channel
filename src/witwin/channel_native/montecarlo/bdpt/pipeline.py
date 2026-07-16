from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch

from witwin.channel_native.scene.models import ReceiverGrid
from witwin.channel_native.core.antenna import validate_scalar_endpoint_features
from witwin.channel_native.core.edge_selection import resolve_scene_edge_policy
from witwin.channel_native.core.field_state import (
    receiver_polarizations,
    transmitter_polarizations,
)
from witwin.channel_native.materials.encoding import face_material_field_bundle
from witwin.channel_native.core.memory_budget import (
    MemoryEstimate,
    enforce_memory_budget,
)
from witwin.channel_native.core.receiver_geometry import (
    axis_aligned_grid_spec as _grid_spec,
    first_receiver_grid,
)
from witwin.channel_native.core.path_topology import export_topology

from witwin.channel_native.montecarlo.bdpt.kernels.sampling import (
    bdpt_diffraction_state_pack,
    bdpt_diffraction_state_wi,
    bdpt_reflection_launch_inputs,
    bdpt_sample_directions,
    bdpt_selected_edge_indices,
)
from witwin.channel_native.scene.models import Scene
from witwin.channel_native.montecarlo.bdpt.kernels.paths import (
    bdpt_accumulate_connection_samples,
    bdpt_compact_connection_samples,
    bdpt_concat_connection_samples,
    bdpt_connection_variance,
    bdpt_count_valid_connection_samples,
    bdpt_diffraction_connection_samples_from_tape,
    bdpt_diffraction_point_connection_samples,
    bdpt_endpoint_connection_samples,
    bdpt_endpoint_connection_visibility_inputs,
    bdpt_endpoint_subpath_state,
    bdpt_filter_connection_samples,
    bdpt_reflected_light_subpath_state,
    bdpt_subpath_intersection_inputs,
    bdpt_transmitted_light_subpath_state,
)
from witwin.channel_native.montecarlo.bdpt.kernels.maps import (
    bdpt_component_map_buffer,
    bdpt_finalize_component_maps,
    bdpt_finalize_point_components,
    bdpt_los_component_maps_from_matrix,
    bdpt_store_component_map,
    bdpt_zero_matrix,
)
from witwin.channel_native.materials.kernels.functional import em_layer_stack_eval
from witwin.channel_native.core.diffraction_geometry import (
    cached_diffraction_edge_geometry as _cached_diffraction_edge_geometry,
)
from witwin.channel_native.propagation.geometry.kernels import bridge as geometry_bridge
from witwin.channel_native.montecarlo.scattering_events import (
    local_frames,
    rough_material_runtimes,
    scatter_direction_uniforms,
    scattered_subpath_state,
    scattering_nee_connection_samples,
    te_tm_incident_power,
    three_way_rough_probabilities,
    world_to_local,
)
from witwin.channel_native.montecarlo.transmission import (
    event_uniforms,
    layer_csr_view,
    scene_diagonal_m,
    straight_transmission_chains,
    transmission_event_probability,
    unpolarized_power_budgets,
)
from .connections import (
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
class _BDPTTopologyOptions:
    max_depth: int
    components: frozenset[str]
    max_paths: int | None = None
    max_paths_scope: str = "per_pair"
    ad_mode: str = "none"
    coupled_paths: bool = False
    coupled_candidate_limit: int = 1_000_000

    def __post_init__(self) -> None:
        if self.max_depth > 5 and self.components & {
            "reflection",
            "transmission",
        }:
            raise RuntimeError("path reflection/transmission support max_depth <= 5")


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
    if "transmission" in config.components:
        # Straight endpoint chains: one deterministic row per (tx, rx) pair.
        dense_rows += max(0, int(tx_count)) * rx_rows
        # Sampled mixed reflection+transmission chains retain a connection
        # block per light depth.
        dense_rows += launch_entries * rx_rows * max(1, int(config.max_depth))
    if "scattering" in config.components:
        # NEE rows from scatter-selected vertices: bounded by one block of
        # (scatter events x receivers) per light depth.
        dense_rows += launch_entries * rx_rows * max(1, int(config.max_depth))
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
        for component in (
            "los",
            "reflection",
            "diffraction",
            "transmission",
            "scattering",
        )
    }


def _effective_native_samples(config: Config) -> int:
    return int(config.samples) * int(config.sample_streams)


def _effective_native_depth(config: Config) -> int:
    return min(int(config.max_depth), int(config.max_light_depth))


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
    visible = geometry_bridge.raydn_visibility_forward(
        raydn.require_handle(),
        visibility_inputs["start"],
        visibility_inputs["end"],
        visibility_inputs["active"],
    )[0]
    return bdpt_filter_connection_samples(samples, visible)


def _diffraction_sample_split(sample_count: int, *, mis: str) -> tuple[int, int, int]:
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
    _eps_r, _sigma_e, _mu_r, material_gain, material_valid, _thickness = (
        material_tensors
    )
    spec = _grid_spec(grid)
    dim0, dim1 = grid.shape[1], grid.shape[0]
    maps = bdpt_component_map_buffer(
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
        out = geometry_bridge.bdpt_diffraction_accumulation_forward(
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
        maps = bdpt_store_component_map(
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
    _eps_r, _sigma_e, _mu_r, material_gain, material_valid, _thickness = (
        material_tensors
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
            visible_source = geometry_bridge.raydn_visibility_forward(
                raydn.require_handle(),
                exported["source_start"],
                exported["source_end"],
                exported["visibility_active"],
            )[0]
            filtered = bdpt_filter_connection_samples(samples_out, visible_source)
            visible_target = geometry_bridge.raydn_visibility_forward(
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
        _BDPTTopologyOptions(
            max_depth=int(config.max_depth),
            components=frozenset({"reflection"}),
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
        _BDPTTopologyOptions(
            max_depth=int(config.max_depth),
            components=frozenset({"reflection", "diffraction"}),
            coupled_paths=True,
            coupled_candidate_limit=int(config.coupled_candidate_limit),
        ),
    )
    selected = torch.nonzero(topology.component_id >= 3, as_tuple=False).flatten()
    return _topology_connection_samples(topology, selected, component_out=2)


_TRANSMISSION_COMPONENT_ID = 5
_MASK_REFLECTION = 2
_MASK_TRANSMISSION = 8


def _merge_event_states(
    reflected: dict[str, torch.Tensor],
    transmitted: dict[str, torch.Tensor],
    choose_transmit: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Row-wise merge of the two event kernels' outputs.

    Both kernels are evaluated on the full batch and the per-row winner is
    selected, which preserves the tensor layout exactly (equivalent to
    partitioning the hit indices, running each kernel on its partition, and
    scattering back by original index).
    """

    wide = choose_transmit[:, None]
    merged: dict[str, torch.Tensor] = {}
    for key, reflected_value in reflected.items():
        condition = wide if reflected_value.dim() == 2 else choose_transmit
        merged[key] = torch.where(condition, transmitted[key], reflected_value)
    return merged


def _transmission_straight_connection_samples(
    raydn: Any,
    light: dict[str, torch.Tensor],
    sensor: dict[str, torch.Tensor],
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    material_bundle: dict[str, torch.Tensor],
    *,
    frequency_hz: float,
    max_depth: int,
    scene_diagonal: float,
    mis: str,
    beta: float,
) -> tuple[dict[str, torch.Tensor] | None, int]:
    """Exact pure-transmission Tx->Rx chains (endpoint-connection context).

    Specular thin_sheet transmission never bends the ray (parallel-plate
    exit), so every pure-transmission path topology IS the straight Tx->Rx
    segment. Marching that segment yields the per-pair power transmittance
    product, which scales the analytic endpoint-connection LoS contribution
    and is reclassified as the exclusive transmission path class (component
    id 5). Like the discrete reflection enumeration, these chains carry unit
    bidirectional mass (mis weight 1). Pairs whose segment crosses no wall
    belong to the los class and are filtered out here; a vacuum wall has unit
    power transmittance, so the transmission component reproduces the
    unobstructed LoS value exactly.
    """

    rx_count = int(rx_positions.shape[0])
    if int(tx_positions.shape[0]) == 0 or rx_count == 0:
        return None, 0
    samples = bdpt_endpoint_connection_samples(
        light,
        sensor,
        frequency_hz=frequency_hz,
        samples_per_tx=1,
        max_paths=None,
        mis=mis,
        beta=beta,
        strategy_count=1,
    )
    layer_csr = layer_csr_view(material_bundle)
    transmittance_rows = []
    penetrated_rows = []
    wall_rows = []
    for tx_index in range(int(tx_positions.shape[0])):
        origins = tx_positions[tx_index].unsqueeze(0).repeat(rx_count, 1)
        chain = straight_transmission_chains(
            raydn,
            origins,
            rx_positions,
            face_material_id=material_bundle["material_id"],
            layer_csr=layer_csr,
            frequency_hz=frequency_hz,
            max_depth=max_depth,
            scene_diagonal=scene_diagonal,
        )
        transmittance_rows.append(chain["transmittance"])
        penetrated_rows.append(chain["penetrated"])
        wall_rows.append(chain["wall_count"])
    # Connection rows are light-major (light_index * sensor_count + sensor)
    # and the reduced light state carries exactly one row per transmitter, so
    # the concatenated per-tx march aligns 1:1 with the connection table.
    transmittance = torch.cat(transmittance_rows, dim=0)
    penetrated = torch.cat(penetrated_rows, dim=0)
    wall_count = torch.cat(wall_rows, dim=0)
    samples["contribution"] = samples["contribution"] * transmittance
    samples = bdpt_filter_connection_samples(samples, penetrated)
    chain_count = int(samples["valid"].sum())
    if chain_count == 0:
        return None, 0
    component_id = torch.where(
        samples["valid"],
        torch.full_like(samples["component_id"], _TRANSMISSION_COMPONENT_ID),
        samples["component_id"],
    )
    light_depth = torch.where(samples["valid"], wall_count, samples["light_depth"])
    topology = samples["topology"].clone()
    topology[:, 2] = component_id
    topology[:, 3] = light_depth
    samples["component_id"] = component_id
    samples["light_depth"] = light_depth
    samples["topology"] = topology
    return samples, chain_count


def _merge_scattered_state(
    merged: dict[str, torch.Tensor],
    scattered: dict[str, torch.Tensor],
    choose_scatter: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Row-wise overlay of the scattered branch onto the reflect/transmit
    merge (same evaluate-everywhere-select-per-row pattern as
    :func:`_merge_event_states`)."""

    wide = choose_scatter[:, None]
    out: dict[str, torch.Tensor] = {}
    for key, merged_value in merged.items():
        condition = wide if merged_value.dim() == 2 else choose_scatter
        out[key] = torch.where(condition, scattered[key], merged_value)
    return out


def _transmission_sampled_connection_samples(
    raydn: Any,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_positions: torch.Tensor,
    rx_polarization: torch.Tensor,
    sensor: dict[str, torch.Tensor],
    material_bundle: dict[str, torch.Tensor],
    *,
    frequency_hz: float,
    samples: int,
    max_depth: int,
    seed: int,
    mis: str,
    beta: float,
    scattering_runtimes: dict[int, Any] | None = None,
    emit_mixed_transmission: bool = True,
    scene_diagonal: float = 0.0,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, int]]:
    """Shooting-context light subpaths with three-way event selection.

    Implements plan section 7.1. At every surface hit a seeded, reproducible
    uniform selects among the delta specular reflection, the continuous
    Kirchhoff scattering (rough faces only, when ``scattering_runtimes`` is
    provided) and the delta transmission events; the selected branch's field
    is divided by sqrt(p_event) so the power estimator stays unbiased (see
    the inline algebra note).

    Event probabilities: smooth faces keep the wave-2 two-way split
    p_t = T/(R+T) from the native stack budgets BIT-IDENTICALLY (their
    scatter probability is exactly zero, so the same uniform partitions the
    same way); rough faces use the (R_coh, R_diff, T_bar) budgets from
    scattering.energy with the same floor pattern. The rough reflect branch
    additionally multiplies the field by the coherent attenuation C_r so its
    amplitude represents sqrt(R_coh), matching the budget that selected it.

    Contribution routing (never double counts):
    - MIXED reflection+transmission chains connect through the native
      endpoint kernel (component 5), as in wave 2; emitted only when
      ``emit_mixed_transmission`` (the transmission component is requested).
    - Scatter-selected vertices emit torch-side NEE rows (component 6) and
      then TERMINATE (v1 single-bounce rule; reflection/transmission never
      follow a scattering event).
    - Pure reflection stays with the discrete enumeration and pure
      transmission with the straight endpoint chains.
    """

    device = tx_positions.device
    layer_csr = layer_csr_view(material_bundle)
    face_material_id = material_bundle["material_id"]
    material_axis_rad = material_bundle["rough_axis_rad"]
    runtimes = scattering_runtimes or {}
    sensor_count = int(sensor["origin"].shape[0])
    sample_blocks: list[dict[str, torch.Tensor]] = []
    transmit_events = 0
    reflect_events = 0
    scatter_events = 0
    nee_rows = 0
    for tx_index in range(int(tx_positions.shape[0])):
        launch_inputs = bdpt_reflection_launch_inputs(
            tx_positions, tx_index=tx_index, sample_count=int(samples)
        )
        ray_d = bdpt_sample_directions(
            int(samples), tx_positions, seed=int(seed) + tx_index * 65537
        )
        state = bdpt_endpoint_subpath_state(
            tx_positions,
            tx_power,
            tx_polarization,
            rx_positions,
            rx_polarization,
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
        for bounce in range(max(1, int(max_depth))):
            hit = geometry_bridge.bdpt_intersect_forward(
                raydn.require_handle(),
                ray_inputs["ray_o"],
                ray_inputs["ray_d"],
                ray_inputs["ray_tmax"],
                ray_inputs["active"],
            )
            prim = hit["global_prim_id"]
            material_id = face_material_id.index_select(
                0, prim.clamp_min(0).to(torch.int64)
            )
            hit_ok = (
                state["valid"] & (prim >= 0) & (hit["t"] >= 0.0) & (material_id >= 0)
            )
            cos_theta = (
                (state["direction"] * hit["n"]).sum(dim=-1).abs().clamp(1.0e-6, 1.0)
            )
            stack = em_layer_stack_eval(
                cos_theta,
                material_id.clamp_min(0),
                layer_csr["layer_offset"],
                layer_csr["layer_count"],
                layer_csr["layer_thickness_m"],
                layer_csr["layer_eps_r"],
                layer_csr["layer_sigma_e"],
                layer_csr["layer_mu_r"],
                frequency_hz=frequency_hz,
            )
            r_eff, t_eff = unpolarized_power_budgets(stack)
            p_transmit = transmission_event_probability(r_eff, t_eff)
            uniforms = event_uniforms(
                int(samples), seed=seed, tx_index=tx_index, depth=bounce, device=device
            )
            if runtimes:
                rough_probs = three_way_rough_probabilities(
                    cos_theta,
                    material_id,
                    material_bundle,
                    stack,
                    frequency_hz=float(frequency_hz),
                )
                rough = rough_probs["rough"] & hit_ok
                p_scatter = torch.where(
                    rough, rough_probs["p_scatter"], torch.zeros_like(p_transmit)
                )
                # Smooth rows keep the exact wave-2 two-way probability;
                # rough rows switch to the three-way budget split.
                p_transmit = torch.where(rough, rough_probs["p_transmit"], p_transmit)
                coherent_amplitude = torch.where(
                    rough,
                    rough_probs["r_coh_amplitude"],
                    torch.ones_like(p_transmit),
                )
            else:
                rough = torch.zeros_like(hit_ok)
                p_scatter = torch.zeros_like(p_transmit)
                coherent_amplitude = torch.ones_like(p_transmit)
            # One uniform partitions the three events: [0, p_s) scatter,
            # [p_s, p_s + p_t) transmit, else reflect. Smooth faces have
            # p_s = 0 exactly, so their transmit test u < p_t is unchanged.
            choose_scatter = hit_ok & (uniforms < p_scatter)
            choose_transmit = (
                hit_ok & ~choose_scatter & (uniforms < p_scatter + p_transmit)
            )
            reflected = bdpt_reflected_light_subpath_state(
                state,
                hit,
                material_gain=material_bundle["gain"],
                material_valid=material_bundle["valid"],
                material_eps_r=material_bundle["eps_r"],
                material_sigma_e=material_bundle["sigma_e"],
                material_mu_r=material_bundle["mu_r"],
                material_thickness=material_bundle["thickness"],
                frequency_hz=frequency_hz,
            )
            transmitted = bdpt_transmitted_light_subpath_state(
                state,
                hit,
                face_material_id=face_material_id,
                layer_offset=layer_csr["layer_offset"],
                layer_count=layer_csr["layer_count"],
                layer_thickness_m=layer_csr["layer_thickness_m"],
                layer_eps_r=layer_csr["layer_eps_r"],
                layer_sigma_e=layer_csr["layer_sigma_e"],
                layer_mu_r=layer_csr["layer_mu_r"],
                frequency_hz=frequency_hz,
            )
            merged = _merge_event_states(reflected, transmitted, choose_transmit)
            # Unbiased event split: every contribution downstream has the form
            # source_power * |field|^2 * (geometry terms), and branch e was
            # selected with probability p_e, so the POWER must be divided by
            # p_e. Dividing the FIELD (and the real amplitude proxy) by
            # sqrt(p_e) achieves exactly that:
            #   E[|field_e / sqrt(p_e)|^2] = sum_e p_e * |field_e|^2 / p_e
            #                              = sum_e |field_e|^2.
            # source_power is deliberately untouched; scaling it too would
            # double count the correction. The reflect probability is
            # 1 - p_s - p_t (p_s = 0 on smooth faces, reproducing wave 2).
            p_event = torch.where(
                choose_transmit, p_transmit, 1.0 - p_scatter - p_transmit
            )
            inv_amplitude = torch.where(
                merged["valid"],
                torch.rsqrt(p_event.clamp_min(1.0e-4)),
                torch.ones_like(p_event),
            )
            # Rough reflect branch: the native kernel applied the SMOOTH
            # stack Jones (amplitude sqrt(R_bar)); multiplying by C_r turns
            # it into the coherent amplitude sqrt(R_coh) that matches the
            # budget driving its selection probability (contract 6.2).
            reflect_scale = torch.where(
                rough & ~choose_transmit & ~choose_scatter,
                coherent_amplitude,
                torch.ones_like(coherent_amplitude),
            )
            amplitude_scale = inv_amplitude * reflect_scale
            for key in ("throughput_real", "throughput_imag"):
                merged[key] = merged[key] * amplitude_scale
            for key in ("field_real", "field_imag"):
                merged[key] = merged[key] * amplitude_scale[:, None]
            # Reflection/transmission directions are delta events, but the
            # sampled event class still has a discrete probability mass. Keep
            # that mass in the proposal density; canonical enumerated delta
            # paths use a separate unit-mass block.
            for key in ("pdf_forward", "pdf_reverse"):
                merged[key] = torch.where(
                    merged["valid"], merged[key] * p_event, torch.zeros_like(p_event)
                )

            scattered_valid = torch.zeros_like(choose_scatter)
            if runtimes and bool(choose_scatter.any()):
                # Local roughness frames from the shading normal flipped
                # toward the incident side (roughness applies to whichever
                # side is illuminated in v1; the store carries front-surface
                # statistics only).
                direction = state["direction"]
                hit_normal = hit["n"]
                normal_flipped = torch.where(
                    ((direction * hit_normal).sum(dim=-1) > 0.0)[:, None],
                    -hit_normal,
                    hit_normal,
                )
                axis_rad = material_axis_rad.index_select(
                    0, material_id.clamp_min(0).to(torch.int64)
                )
                frame_t1, frame_t2 = local_frames(normal_flipped, axis_rad)
                wi_world = -direction
                wi_local = world_to_local(wi_world, frame_t1, frame_t2, normal_flipped)
                p_te, p_tm = te_tm_incident_power(
                    state["field_real"],
                    state["field_imag"],
                    direction,
                    normal_flipped,
                )
                direction_uniforms = scatter_direction_uniforms(
                    int(samples),
                    seed=seed,
                    tx_index=tx_index,
                    depth=bounce,
                    device=device,
                )
                scattered = scattered_subpath_state(
                    state,
                    hit,
                    choose_scatter=choose_scatter,
                    normal=normal_flipped,
                    frame_t1=frame_t1,
                    frame_t2=frame_t2,
                    wi_local=wi_local,
                    p_te=p_te,
                    p_tm=p_tm,
                    p_scatter=p_scatter,
                    material_id=material_id,
                    runtimes=runtimes,
                    uniforms=direction_uniforms,
                    scene_diagonal=scene_diagonal,
                )
                scattered_valid = scattered["valid"]
                merged = _merge_scattered_state(merged, scattered, choose_scatter)
                rows = torch.nonzero(scattered_valid, as_tuple=False).flatten()
                if int(rows.numel()):
                    nee_block = scattering_nee_connection_samples(
                        raydn,
                        sensor,
                        runtimes,
                        position=hit["p"].index_select(0, rows),
                        normal=normal_flipped.index_select(0, rows),
                        frame_t1=frame_t1.index_select(0, rows),
                        frame_t2=frame_t2.index_select(0, rows),
                        wi_local=wi_local.index_select(0, rows),
                        p_te=p_te.index_select(0, rows),
                        p_tm=p_tm.index_select(0, rows),
                        p_scatter=p_scatter.index_select(0, rows),
                        material_id=material_id.index_select(0, rows),
                        source_power=state["source_power"].index_select(0, rows),
                        tx_id=state["tx_id"].index_select(0, rows),
                        light_depth=scattered["depth"].index_select(0, rows),
                        path_length_at_vertex=scattered["path_length"].index_select(
                            0, rows
                        ),
                        frequency_hz=float(frequency_hz),
                        samples=int(samples),
                        scene_diagonal=scene_diagonal,
                    )
                    if nee_block is not None:
                        sample_blocks.append(nee_block)
                        nee_rows += int(nee_block["valid"].sum())
            transmit_events += int((choose_transmit & merged["valid"]).sum())
            scatter_events += int(scattered_valid.sum())
            reflect_events += int(
                (~choose_transmit & ~choose_scatter & merged["valid"]).sum()
            )
            mask = merged["component_mask"]
            mixed = (
                merged["valid"]
                & ~choose_scatter
                & ((mask & _MASK_REFLECTION) != 0)
                & ((mask & _MASK_TRANSMISSION) != 0)
            )
            if emit_mixed_transmission and bool(mixed.any()):
                samples_out = bdpt_endpoint_connection_samples(
                    merged,
                    sensor,
                    frequency_hz=frequency_hz,
                    samples_per_tx=int(samples),
                    max_paths=None,
                    mis=mis,
                    beta=beta,
                    strategy_count=1,
                )
                visibility_inputs = bdpt_endpoint_connection_visibility_inputs(
                    merged,
                    sensor,
                    sample_count=int(samples_out["valid"].shape[0]),
                )
                visible = geometry_bridge.raydn_visibility_forward(
                    raydn.require_handle(),
                    visibility_inputs["start"],
                    visibility_inputs["end"],
                    visibility_inputs["active"],
                )[0]
                keep = visible & mixed.repeat_interleave(sensor_count)
                sample_blocks.append(bdpt_filter_connection_samples(samples_out, keep))
            # v1 single-bounce rule: scattered subpaths connected above and
            # terminate here; reflection/transmission never follow them.
            merged["valid"] = merged["valid"] & ~choose_scatter
            if not bool(merged["valid"].any()):
                break
            state = merged
            ray_inputs = bdpt_subpath_intersection_inputs(merged)
    return sample_blocks, {
        "transmit": transmit_events,
        "reflect": reflect_events,
        "scatter": scatter_events,
        "scattering_nee_rows": nee_rows,
    }


def solve(
    scene: Scene,
    config: Config,
    *,
    build_info_fn: Callable[[], dict[str, object]],
    transmitter_tensors_fn: Callable[
        [Scene], tuple[torch.Tensor, torch.Tensor]
    ] = transmitter_tensors,
) -> Result:
    grid = first_receiver_grid(scene)
    if grid is not None and config.receiver_strategy != "grid_area":
        raise RuntimeError("receiver_strategy='point_sphere' requires point receivers")
    validate_scalar_endpoint_features(
        scene.transmitters, scene.receivers, solver="BDPT"
    )

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

    info = build_info_fn()
    raydn = scene.raydn_scene()
    raydn_available = bool(info["uses_raydn_native"]) and raydn.available
    reflection_available = raydn_available
    diffraction_available = raydn_available and config.max_diffraction_order > 0
    transmission_available = raydn_available
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
    if (
        "transmission" in config.components
        and scene.structures
        and not transmission_available
    ):
        raise RuntimeError("transmission requires RayDN native capability")

    tx_reference, tx_power = transmitter_tensors_fn(scene)
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
    transmission_chain_count = 0
    event_counts = {"transmit": 0, "reflect": 0, "scatter": 0, "scattering_nee_rows": 0}
    scattering_runtimes: dict[int, Any] = {}
    if endpoint_only:
        component_matrices = {
            "los": endpoint_accumulation["los"],
            "reflection": zero_component_matrix(),
            "diffraction": zero_component_matrix(),
            "transmission": zero_component_matrix(),
            "scattering": zero_component_matrix(),
        }
        estimate_samples = endpoint_connection_samples
    else:
        sample_blocks: list[dict[str, torch.Tensor]] = []
        streamed_matrices = {
            "los": zero_component_matrix(),
            "reflection": zero_component_matrix(),
            "diffraction": zero_component_matrix(),
            "transmission": zero_component_matrix(),
            "scattering": zero_component_matrix(),
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
        transmission_requested = (
            "transmission" in config.components
            and scene.structures
            and transmission_available
            and native_max_depth >= 1
        )
        scattering_requested = (
            "scattering" in config.components
            and scene.structures
            and transmission_available
            and native_max_depth >= 1
        )
        if scattering_requested:
            # Kirchhoff tables per rough material (scatter_model_id == 1);
            # raises kirchhoff_domain_exceeded for out-of-domain roughness.
            scattering_runtimes = rough_material_runtimes(scene.compile())
        sampler_requested = transmission_requested or bool(scattering_runtimes)
        if sampler_requested:
            material_bundle = face_material_field_bundle(
                scene, device=tx_reference.device
            )
            scene_diagonal = scene_diagonal_m(scene)
        if transmission_requested:
            transmission_light_state = _reduced_light_endpoint_state(
                tx_reference,
                tx_power,
                tx_polarization,
                rx_positions,
                rx_polarization,
            )
            straight_samples, transmission_chain_count = (
                _transmission_straight_connection_samples(
                    raydn,
                    transmission_light_state,
                    endpoint_subpaths["sensor"],
                    tx_reference,
                    rx_positions,
                    material_bundle,
                    frequency_hz=float(scene.frequency),
                    max_depth=native_max_depth,
                    scene_diagonal=scene_diagonal,
                    mis=config.mis,
                    beta=config.power_heuristic_beta,
                )
            )
            if straight_samples is not None:
                sample_blocks.append(straight_samples)
            launch_count += 2
        if sampler_requested:
            mixed_blocks, event_counts = _transmission_sampled_connection_samples(
                raydn,
                tx_reference,
                tx_power,
                tx_polarization,
                rx_positions,
                rx_polarization,
                endpoint_subpaths["sensor"],
                material_bundle,
                frequency_hz=float(scene.frequency),
                samples=native_samples,
                max_depth=native_max_depth,
                seed=int(config.seed),
                mis=config.mis,
                beta=config.power_heuristic_beta,
                scattering_runtimes=scattering_runtimes,
                emit_mixed_transmission=transmission_requested,
                scene_diagonal=scene_diagonal,
            )
            sample_blocks.extend(mixed_blocks)
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
            component_maps["transmission"],
            component_maps["scattering"],
        )
    else:
        point_component_matrices = component_matrices
        finalized = bdpt_finalize_point_components(
            point_component_matrices["los"],
            point_component_matrices["reflection"],
            point_component_matrices["diffraction"],
            point_component_matrices["transmission"],
            point_component_matrices["scattering"],
        )
    path_gain = finalized["path_gain"]
    component_power = {
        "los": finalized["los_power"],
        "reflection": finalized["reflection_power"],
        "diffraction": finalized["diffraction_power"],
    }
    # transmission and scattering maps are always finalized (zeros when the
    # component is off) but only exposed when requested.
    for optional in ("transmission", "scattering"):
        if optional in config.components:
            component_power[optional] = finalized[f"{optional}_power"]
        elif component_maps is not None:
            component_maps.pop(optional)

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
    if "transmission" in config.components:
        # component_mask bit 8 marks transmitted subpaths (contract section 1).
        metadata["transmission"] = {
            "straight_chain_paths": int(transmission_chain_count),
            "event_counts": {
                "transmit": int(event_counts["transmit"]),
                "reflect": int(event_counts["reflect"]),
            },
            "component_mask_bit": 8,
        }
    if "scattering" in config.components:
        # component_mask bit 16 marks scattered subpaths (contract section 1).
        metadata["scattering"] = {
            "event_counts": {"scatter": int(event_counts["scatter"])},
            "nee_connection_rows": int(event_counts["scattering_nee_rows"]),
            "rough_material_count": len(scattering_runtimes),
            "component_mask_bit": 16,
            # v1 depth rule: one Kirchhoff event per path; scattered
            # subpaths connect to the receivers and terminate.
            "max_scattering_order": 1,
        }
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
