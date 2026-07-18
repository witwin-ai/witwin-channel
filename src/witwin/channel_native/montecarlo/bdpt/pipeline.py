from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

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
    first_receiver_grid,
)
from witwin.channel_native.propagation import EvaluatedPaths, evaluate_enumerated_paths
from witwin.channel_native.scene.models import ReceiverGrid, Scene
from witwin.channel_native.montecarlo.bdpt.kernels.maps import (
    bdpt_finalize_component_maps,
    bdpt_finalize_point_components,
    bdpt_los_component_maps_from_matrix,
    bdpt_zero_matrix,
)
from witwin.channel_native.montecarlo.bdpt.kernels.paths import (
    bdpt_accumulate_connection_samples,
    bdpt_compact_connection_samples,
    bdpt_concat_connection_samples,
    bdpt_connection_variance,
    bdpt_count_valid_connection_samples,
    bdpt_endpoint_connection_samples,
    bdpt_endpoint_subpath_state,
)
from witwin.channel_native.montecarlo.events.scattering import rough_material_runtimes
from witwin.channel_native.montecarlo.events.transmission import scene_diagonal_m

from .accumulation import (
    _component_maps_from_matrices,
    _empty_path_samples,
    _path_samples_from_connection_export,
)
from .config import Config
from .connections import (
    _native_los_connection_samples,
    _transmission_sampled_connection_samples,
    _transmission_straight_connection_samples,
)
from .endpoints import (
    receiver_positions,
    transmitter_tensors,
)
from .metadata import make_solver_metadata, select_accumulation_strategy
from .result import Result
from .sampling import make_launch_state


_EXPORT_BYTES_PER_PATH = 96
_CONNECTION_BYTES_PER_ROW = 57
_VISIBILITY_BYTES_PER_ROW = 25


@dataclass(frozen=True, slots=True)
class _SolvePrep:
    """Workspace-sizing and native-capability results for one solve()."""

    native_samples: int
    native_max_depth: int
    selected_accumulation: str
    workspace_bytes: int
    info: dict[str, object]
    raydn: Any
    reflection_available: bool
    diffraction_available: bool
    transmission_available: bool


@dataclass(frozen=True, slots=True)
class _EndpointWorkspace:
    """Endpoint subpath tensors and derived counts shared across stages."""

    tx_reference: torch.Tensor
    tx_power: torch.Tensor
    rx_positions: torch.Tensor
    topology_scene: Scene
    tx_polarization: torch.Tensor
    rx_polarization: torch.Tensor
    launch_state: dict[str, torch.Tensor]
    endpoint_subpaths: dict[str, Any]
    los_light_state: dict[str, torch.Tensor] | None
    endpoint_connection_samples: dict[str, torch.Tensor] | None
    endpoint_accumulation: dict[str, torch.Tensor] | None
    launch_count: int
    tx_count: int
    rx_count: int
    endpoint_only: bool


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


def _evaluated_connection_samples(
    paths: EvaluatedPaths, selected: torch.Tensor, *, component_out: int
) -> dict[str, torch.Tensor] | None:
    if int(selected.numel()) == 0:
        return None
    topology = paths.topology
    geometry = paths.geometry
    fields = paths.fields
    tx_id = topology.tx_id.index_select(0, selected).to(torch.int32)
    rx_id = topology.rx_id.index_select(0, selected).to(torch.int32)
    depth = topology.depth.index_select(0, selected).to(torch.int32)
    contribution = fields.path_gain.index_select(0, selected)
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
        "path_length_m": geometry.path_length_m.index_select(0, selected),
    }


def _single_class_discrete_connection_samples(
    scene: Scene,
    config: Config,
    *,
    component: str,
    component_id: int,
) -> dict[str, torch.Tensor] | None:
    """Enumerate one delta-like path class as unit-mass discrete connections.

    Shared by the reflection and standalone-diffraction BDPT connection builders:
    both consume the public ``evaluate_enumerated_paths`` for a single delta/UTD
    component (ADR-008), select the matching enumerated rows, and pack them with
    unit forward/reverse discrete mass. The selected ``component_id`` also names
    the accumulation bucket (reflection -> 1, diffraction -> 2). The coupled
    builder differs (mixed-order discovery plus a >=3 class selection) and keeps
    its own body.
    """

    paths, _ = evaluate_enumerated_paths(
        scene,
        _BDPTTopologyOptions(
            max_depth=int(config.max_depth),
            components=frozenset({component}),
        ),
    )
    selected = torch.nonzero(
        paths.topology.component_id == component_id, as_tuple=False
    ).flatten()
    return _evaluated_connection_samples(paths, selected, component_out=component_id)


def _reflection_discrete_connection_samples(
    scene: Scene, config: Config
) -> dict[str, torch.Tensor] | None:
    """Enumerate delta-specular paths with unit forward/reverse discrete mass."""

    return _single_class_discrete_connection_samples(
        scene, config, component="reflection", component_id=1
    )


def _diffraction_discrete_connection_samples(
    scene: Scene, config: Config
) -> dict[str, torch.Tensor] | None:
    """Enumerate first-order UTD diffraction paths with unit discrete mass.

    Standalone diffraction is a delta-like discrete path: first-order UTD emits
    one deterministic edge-diffraction connection per (tx, edge, rx) triple with
    the same field the deterministic solver evaluates. BDPT consumes it as an
    opaque discrete-path oracle (ADR-008/ADR-018) exactly as it does reflection,
    so the standalone diffraction component reproduces the deterministic
    reference instead of the retired crude power heuristic. The enumerated engine
    only implements order-1 diffraction, so max_diffraction_order stays at its
    default of 1.
    """

    return _single_class_discrete_connection_samples(
        scene, config, component="diffraction", component_id=2
    )


def _coupled_discrete_connection_samples(
    scene: Scene, config: Config
) -> dict[str, torch.Tensor] | None:
    """Enumerate mixed delta/UTD paths with unit bidirectional discrete mass."""

    paths, _ = evaluate_enumerated_paths(
        scene,
        _BDPTTopologyOptions(
            max_depth=int(config.max_depth),
            components=frozenset({"reflection", "diffraction"}),
            coupled_paths=True,
            coupled_candidate_limit=int(config.coupled_candidate_limit),
        ),
    )
    selected = torch.nonzero(paths.topology.component_id >= 3, as_tuple=False).flatten()
    return _evaluated_connection_samples(paths, selected, component_out=2)


def _validate(scene: Scene, config: Config) -> ReceiverGrid | None:
    grid = first_receiver_grid(scene)
    if grid is not None and config.receiver_strategy != "grid_area":
        raise RuntimeError("receiver_strategy='point_sphere' requires point receivers")
    validate_scalar_endpoint_features(
        scene.transmitters, scene.receivers, solver="BDPT"
    )
    return grid



def _prepare_workspace_and_capabilities(
    scene: Scene,
    config: Config,
    *,
    grid: ReceiverGrid | None,
    build_info_fn: Callable[[], dict[str, object]],
) -> _SolvePrep:
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
    return _SolvePrep(
        native_samples=native_samples,
        native_max_depth=native_max_depth,
        selected_accumulation=selected_accumulation,
        workspace_bytes=workspace_bytes,
        info=info,
        raydn=raydn,
        reflection_available=reflection_available,
        diffraction_available=diffraction_available,
        transmission_available=transmission_available,
    )



def _build_endpoint_subpaths(
    scene: Scene,
    config: Config,
    *,
    grid: ReceiverGrid | None,
    transmitter_tensors_fn: Callable[[Scene], tuple[torch.Tensor, torch.Tensor]],
    selected_accumulation: str,
) -> _EndpointWorkspace:
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
    return _EndpointWorkspace(
        tx_reference=tx_reference,
        tx_power=tx_power,
        rx_positions=rx_positions,
        topology_scene=topology_scene,
        tx_polarization=tx_polarization,
        rx_polarization=rx_polarization,
        launch_state=launch_state,
        endpoint_subpaths=endpoint_subpaths,
        los_light_state=los_light_state,
        endpoint_connection_samples=endpoint_connection_samples,
        endpoint_accumulation=endpoint_accumulation,
        launch_count=launch_count,
        tx_count=tx_count,
        rx_count=rx_count,
        endpoint_only=endpoint_only,
    )



def _collect_connection_samples(
    scene: Scene,
    config: Config,
    *,
    prep: _SolvePrep,
    workspace: _EndpointWorkspace,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor] | None,
    int,
    dict[str, int],
    dict[int, Any],
    int,
]:
    tx_reference = workspace.tx_reference
    tx_power = workspace.tx_power
    rx_positions = workspace.rx_positions
    topology_scene = workspace.topology_scene
    tx_polarization = workspace.tx_polarization
    rx_polarization = workspace.rx_polarization
    endpoint_subpaths = workspace.endpoint_subpaths
    los_light_state = workspace.los_light_state
    endpoint_connection_samples = workspace.endpoint_connection_samples
    endpoint_accumulation = workspace.endpoint_accumulation
    endpoint_only = workspace.endpoint_only
    tx_count = workspace.tx_count
    rx_count = workspace.rx_count
    launch_count = workspace.launch_count
    raydn = prep.raydn
    native_samples = prep.native_samples
    native_max_depth = prep.native_max_depth
    selected_accumulation = prep.selected_accumulation
    reflection_available = prep.reflection_available
    diffraction_available = prep.diffraction_available
    transmission_available = prep.transmission_available
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
            and bool(scene.structures)
            and native_max_depth >= 1
        )
        diffraction_requested = (
            "diffraction" in config.components
            and diffraction_available
            and bool(scene.structures)
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
            # Standalone first-order diffraction routes through the shared
            # enumerated engine as a unit-mass discrete connection, exactly like
            # reflection above (ADR-008/ADR-018), replacing the retired crude
            # native power heuristic.
            diffraction_samples = _diffraction_discrete_connection_samples(
                topology_scene, config
            )
            if diffraction_samples is not None:
                sample_blocks.append(diffraction_samples)
                launch_count += 1
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
                component: accumulated[component]
                for component in (
                    "los",
                    "reflection",
                    "diffraction",
                    "transmission",
                    "scattering",
                )
            }
        else:
            component_matrices = {
                component: zero_component_matrix()
                for component in (
                    "los",
                    "reflection",
                    "diffraction",
                    "transmission",
                    "scattering",
                )
            }
    return (
        component_matrices,
        estimate_samples,
        transmission_chain_count,
        event_counts,
        scattering_runtimes,
        launch_count,
    )



def _accumulate_and_finalize(
    config: Config,
    *,
    grid: ReceiverGrid | None,
    prep: _SolvePrep,
    workspace: _EndpointWorkspace,
    component_matrices: dict[str, torch.Tensor],
    estimate_samples: dict[str, torch.Tensor] | None,
) -> tuple[
    dict[str, torch.Tensor] | None,
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor | None,
]:
    tx_reference = workspace.tx_reference
    tx_count = workspace.tx_count
    rx_count = workspace.rx_count
    endpoint_only = workspace.endpoint_only
    native_samples = prep.native_samples
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
    return component_maps, path_gain, component_power, variance



def _export_paths(
    scene: Scene,
    config: Config,
    *,
    workspace: _EndpointWorkspace,
    estimate_samples: dict[str, torch.Tensor] | None,
) -> ReceiverGrid | None:
    endpoint_only = workspace.endpoint_only
    los_light_state = workspace.los_light_state
    endpoint_subpaths = workspace.endpoint_subpaths
    tx_reference = workspace.tx_reference
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
    return path_samples



def _build_metadata(
    scene: Scene,
    config: Config,
    *,
    prep: _SolvePrep,
    workspace: _EndpointWorkspace,
    estimate_samples: dict[str, torch.Tensor] | None,
    path_gain: torch.Tensor,
    variance: torch.Tensor | None,
    launch_count: int,
    transmission_chain_count: int,
    event_counts: dict[str, int],
    scattering_runtimes: dict[int, Any],
    component_maps: dict[str, torch.Tensor] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    launch_state = workspace.launch_state
    endpoint_subpaths = workspace.endpoint_subpaths
    selected_accumulation = prep.selected_accumulation
    reflection_available = prep.reflection_available
    diffraction_available = prep.diffraction_available
    info = prep.info
    workspace_bytes = prep.workspace_bytes
    native_max_depth = prep.native_max_depth
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
    return metadata, diagnostics


def solve(
    scene: Scene,
    config: Config,
    *,
    build_info_fn: Callable[[], dict[str, object]],
    transmitter_tensors_fn: Callable[
        [Scene], tuple[torch.Tensor, torch.Tensor]
    ] = transmitter_tensors,
) -> Result:
    grid = _validate(scene, config)
    prep = _prepare_workspace_and_capabilities(
        scene, config, grid=grid, build_info_fn=build_info_fn
    )
    workspace = _build_endpoint_subpaths(
        scene,
        config,
        grid=grid,
        transmitter_tensors_fn=transmitter_tensors_fn,
        selected_accumulation=prep.selected_accumulation,
    )
    (
        component_matrices,
        estimate_samples,
        transmission_chain_count,
        event_counts,
        scattering_runtimes,
        launch_count,
    ) = _collect_connection_samples(
        scene, config, prep=prep, workspace=workspace
    )
    component_maps, path_gain, component_power, variance = _accumulate_and_finalize(
        config,
        grid=grid,
        prep=prep,
        workspace=workspace,
        component_matrices=component_matrices,
        estimate_samples=estimate_samples,
    )
    path_samples = _export_paths(
        scene, config, workspace=workspace, estimate_samples=estimate_samples
    )
    metadata, diagnostics = _build_metadata(
        scene,
        config,
        prep=prep,
        workspace=workspace,
        estimate_samples=estimate_samples,
        path_gain=path_gain,
        variance=variance,
        launch_count=launch_count,
        transmission_chain_count=transmission_chain_count,
        event_counts=event_counts,
        scattering_runtimes=scattering_runtimes,
        component_maps=component_maps,
    )
    return Result(
        path_gain=path_gain,
        component_power=component_power,
        metadata=metadata,
        diagnostics=diagnostics,
        component_maps=component_maps,
        variance=variance,
        path_samples=path_samples,
    )
