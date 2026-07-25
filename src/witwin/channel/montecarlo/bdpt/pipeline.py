from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from witwin.channel.scene.antenna import validate_scalar_endpoint_features
from witwin.channel.scene.edge_selection import resolve_scene_edge_policy
from witwin.channel.runtime.kernel_metadata import AdLaunchLedger
from witwin.channel.runtime.memory_budget import (
    MemoryEstimate,
    enforce_memory_budget,
)
from witwin.channel.scene.receiver_geometry import (
    first_receiver_grid,
)
from witwin.channel.materials.encoding import face_material_field_bundle
from witwin.channel.runtime.autograd_contracts import _ad_geometry_live
from witwin.channel.scene.endpoints import ReceiverGrid, SolverScene as Scene, require_compiled
from witwin.channel.montecarlo.events.transmission import scene_diagonal_m
from witwin.channel.propagation import EvaluatedPaths, evaluate_enumerated_paths
from witwin.channel.montecarlo.bdpt.autograd_accumulate import (
    bdpt_finalize_component_maps_ad,
    bdpt_finalize_point_components_ad,
)
from witwin.channel.montecarlo.bdpt.kernels.maps import (
    bdpt_finalize_component_maps,
    bdpt_finalize_point_components,
    bdpt_los_component_maps_from_matrix,
    bdpt_zero_matrix,
)
from witwin.channel.montecarlo.bdpt.kernels.paths import (
    bdpt_compact_connection_samples,
    bdpt_concat_connection_samples,
    bdpt_connection_variance,
    bdpt_count_valid_connection_samples,
    bdpt_endpoint_connection_samples,
)
from witwin.channel.montecarlo.events.scattering import rough_material_runtimes

from .accumulation import (
    _component_maps_from_matrices,
    _empty_path_samples,
    _path_samples_from_connection_export,
)
from .config import Config
from .connections import (
    _native_los_connection_samples,
    _transmission_sampled_connection_samples,
)
from .endpoints import (
    transmitter_tensors,
)
from .metadata import make_solver_metadata, select_accumulation_strategy
from .result import Result
from .workspace import (
    _EndpointWorkspace,
    _SolvePrep,
    _accumulate_connection_samples,
    _build_endpoint_subpaths,
)


_EXPORT_BYTES_PER_PATH = 96
_CONNECTION_BYTES_PER_ROW = 57
_VISIBILITY_BYTES_PER_ROW = 25


def _host_frequency(scene: Scene) -> float:
    value = scene.frequency
    return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)


@dataclass(frozen=True, slots=True)
class _BDPTTopologyOptions:
    max_depth: int
    components: frozenset[str]
    max_paths: int | None = None
    _detach_field_tx_power: bool = True
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


def _evaluated_connection_samples(
    paths: EvaluatedPaths,
    selected: torch.Tensor,
    *,
    component_out: int,
    tx_power: torch.Tensor | None = None,
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
    if tx_power is not None:
        # ADR-022 tx_power threading (ADR-014 coefficient precedent). The
        # enumerated engine applies per-tx power as a frozen host scale, so each
        # contribution row is EXACTLY LINEAR in P[tx_id]: path_gain = P * base,
        # with base (geometry x |field|^2 / P) independent of P. Reattach the
        # live power's gradient by the exact-primal ratio P_live / P_live.detach()
        # -- x / x.detach() is exactly 1.0 in IEEE for finite nonzero P, so the
        # primal is bitwise unchanged, and d(contribution)/dP = path_gain / P =
        # base is exact by linearity. Any material/frequency gradient already on
        # ``contribution`` (from the enumerated oracle) is preserved because the
        # factor is 1.0. Zero-power rows (path_gain == 0) pass through untouched
        # (denominator clamped, gradient there is 0 anyway).
        power_rows = tx_power.index_select(0, tx_id.to(torch.int64))
        power_detached = power_rows.detach()
        safe_denominator = torch.where(
            power_detached != 0.0, power_detached, torch.ones_like(power_detached)
        )
        contribution = torch.where(
            power_detached != 0.0,
            contribution * (power_rows / safe_denominator),
            contribution,
        )
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
    tx_power: torch.Tensor | None = None,
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
            # ADR-022: thread ad_mode read-only through the ADR-008 oracle so
            # the enumerated discrete block inherits the enumerated engine's
            # fixed-winner geometry/material AD. 'none' is a no-op.
            ad_mode=config.ad_mode,
        ),
    )
    selected = torch.nonzero(
        paths.topology.component_id == component_id, as_tuple=False
    ).flatten()
    return _evaluated_connection_samples(
        paths, selected, component_out=component_id, tx_power=tx_power
    )


def _reflection_discrete_connection_samples(
    scene: Scene, config: Config, *, tx_power: torch.Tensor | None = None
) -> dict[str, torch.Tensor] | None:
    """Enumerate delta-specular paths with unit forward/reverse discrete mass."""

    return _single_class_discrete_connection_samples(
        scene, config, component="reflection", component_id=1, tx_power=tx_power
    )


def _diffraction_discrete_connection_samples(
    scene: Scene, config: Config, *, tx_power: torch.Tensor | None = None
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
        scene, config, component="diffraction", component_id=2, tx_power=tx_power
    )


def _transmission_discrete_connection_samples(
    scene: Scene, config: Config, *, tx_power: torch.Tensor | None = None
) -> dict[str, torch.Tensor] | None:
    """Enumerate pure straight-segment transmission paths with unit discrete mass.

    ADR-020. Pure specular transmission is delta-like exactly as reflection is,
    so BDPT consumes it as an opaque enumerated discrete-path oracle (ADR-008),
    reproducing the deterministic full-Jones layer-stack field
    (``field_transmission_sequence``) instead of the retired straight-chain TE/TM
    mean. Mixed reflection+transmission chains are not enumerable and stay with
    the event-selected shooting sampler (native full-Jones field).
    """

    return _single_class_discrete_connection_samples(
        scene, config, component="transmission", component_id=5, tx_power=tx_power
    )


def _coupled_discrete_connection_samples(
    scene: Scene, config: Config, *, tx_power: torch.Tensor | None = None
) -> dict[str, torch.Tensor] | None:
    """Enumerate mixed delta/UTD paths with unit bidirectional discrete mass."""

    paths, _ = evaluate_enumerated_paths(
        scene,
        _BDPTTopologyOptions(
            max_depth=int(config.max_depth),
            components=frozenset({"reflection", "diffraction"}),
            coupled_paths=True,
            coupled_candidate_limit=int(config.coupled_candidate_limit),
            ad_mode=config.ad_mode,
        ),
    )
    selected = torch.nonzero(paths.topology.component_id >= 3, as_tuple=False).flatten()
    return _evaluated_connection_samples(
        paths, selected, component_out=2, tx_power=tx_power
    )


_COHERENT_ENUMERATED_COMPONENTS = (
    ("los", 0),
    ("reflection", 1),
    ("diffraction", 2),
)


def _enumerated_component_block_with_field(
    scene: Scene,
    config: Config,
    *,
    component: str,
    component_id: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor] | None:
    """Enumerate one delta/UTD class as a discrete block plus its complex field.

    ADR-019 coherent path. Returns the same unit-mass discrete connection block
    the power-domain path uses, together with the per-row complex projected
    field coefficient (``path_field``) selected in row order so it aligns with
    the block. ``path_field`` is the natively evaluated complex field the
    deterministic coherent accumulator sums, so BDPT coherent reproduces the
    deterministic per-component coherent power.
    """

    paths, _ = evaluate_enumerated_paths(
        scene,
        _BDPTTopologyOptions(
            max_depth=int(config.max_depth),
            components=frozenset({component}),
            ad_mode=config.ad_mode,
        ),
    )
    selected = torch.nonzero(
        paths.topology.component_id == component_id, as_tuple=False
    ).flatten()
    block = _evaluated_connection_samples(paths, selected, component_out=component_id)
    if block is None:
        return None
    field = paths.fields.path_field.index_select(0, selected)
    return block, field.real, field.imag


def _coupled_component_block_with_field(
    scene: Scene, config: Config
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor] | None:
    """Coupled reflection-diffraction discrete block plus its complex field."""

    paths, _ = evaluate_enumerated_paths(
        scene,
        _BDPTTopologyOptions(
            max_depth=int(config.max_depth),
            components=frozenset({"reflection", "diffraction"}),
            coupled_paths=True,
            coupled_candidate_limit=int(config.coupled_candidate_limit),
            ad_mode=config.ad_mode,
        ),
    )
    selected = torch.nonzero(paths.topology.component_id >= 3, as_tuple=False).flatten()
    block = _evaluated_connection_samples(paths, selected, component_out=2)
    if block is None:
        return None
    field = paths.fields.path_field.index_select(0, selected)
    return block, field.real, field.imag


def _collect_coherent_connection_samples(
    scene: Scene,
    config: Config,
    *,
    prep: _SolvePrep,
    workspace: _EndpointWorkspace,
    ledger: AdLaunchLedger | None = None,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor] | None,
    int,
    dict[str, int],
    dict[int, Any],
    int,
]:
    """ADR-019 coherent collection over the enumerable delta/UTD family.

    Every coherent-eligible component (los / reflection / diffraction, plus the
    coupled compensator folded into diffraction) routes through the shared
    enumerated engine as a unit-mass discrete block carrying its complex field.
    The blocks concatenate through the native connection-sample concat (12-field
    schema, unchanged) while the per-row field coefficients concatenate in the
    same block order, then the coherent accumulate op sums the phasor per
    (tx, rx, component) and finalizes ``|sum|^2``. Config validation already
    guarantees ``components`` is a subset of {los, reflection, diffraction}, so
    transmission/scattering never reach here.
    """

    topology_scene = workspace.topology_scene
    tx_count = workspace.tx_count
    rx_count = workspace.rx_count
    launch_count = workspace.launch_count

    def zero_component_matrix() -> torch.Tensor:
        return bdpt_zero_matrix(
            workspace.tx_reference, rows=tx_count, cols=rx_count
        )

    sample_blocks: list[dict[str, torch.Tensor]] = []
    coeff_reals: list[torch.Tensor] = []
    coeff_imags: list[torch.Tensor] = []
    for component, component_id in _COHERENT_ENUMERATED_COMPONENTS:
        if component not in config.components:
            continue
        built = _enumerated_component_block_with_field(
            topology_scene, config, component=component, component_id=component_id
        )
        if built is not None:
            block, coeff_real, coeff_imag = built
            sample_blocks.append(block)
            coeff_reals.append(coeff_real)
            coeff_imags.append(coeff_imag)
            launch_count += 1
    if config.coupled_paths:
        built = _coupled_component_block_with_field(topology_scene, config)
        if built is not None:
            block, coeff_real, coeff_imag = built
            sample_blocks.append(block)
            coeff_reals.append(coeff_real)
            coeff_imags.append(coeff_imag)
            launch_count += 1

    empty_event_counts = {
        "transmit": 0,
        "reflect": 0,
        "scatter": 0,
        "scattering_nee_rows": 0,
    }
    if not sample_blocks:
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
        return component_matrices, None, 0, empty_event_counts, {}, launch_count

    estimate_samples = (
        sample_blocks[0]
        if len(sample_blocks) == 1
        else bdpt_concat_connection_samples(tuple(sample_blocks))
    )
    # Concatenate the field coefficients in the identical block order the
    # native concat uses so the coefficient rows align 1:1 with the samples.
    coeff_real = coeff_reals[0] if len(coeff_reals) == 1 else torch.cat(coeff_reals, dim=0)
    coeff_imag = coeff_imags[0] if len(coeff_imags) == 1 else torch.cat(coeff_imags, dim=0)
    accumulated = _accumulate_connection_samples(
        config,
        estimate_samples,
        tx_count=tx_count,
        rx_count=rx_count,
        accumulation_strategy=prep.selected_accumulation,
        combine_domain="coherent",
        coeff_real=coeff_real,
        coeff_imag=coeff_imag,
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
    return (
        component_matrices,
        estimate_samples,
        0,
        empty_event_counts,
        {},
        launch_count,
    )


def _validate(scene: Scene, config: Config) -> ReceiverGrid | None:
    grid = first_receiver_grid(scene)
    if grid is not None and config.receiver_strategy != "grid_area":
        raise RuntimeError("receiver_strategy='point_sphere' requires point receivers")
    validate_scalar_endpoint_features(
        scene.transmitters, scene.receivers, solver="BDPT"
    )
    _reject_live_geometry_through_sampler(scene, config)
    return grid


def _reject_live_geometry_through_sampler(scene: Scene, config: Config) -> None:
    """ADR-022: mesh-vertex geometry is frozen for the stochastic sampler
    (``ad_geometry='enumerated_blocks_only'``).

    Reflection/diffraction/transmission (pure) and LoS route through the shared
    enumerated engine, which owns geometry adjoints; but the mixed-transmission
    shooting walk and the scattering NEE draw hit points from the stochastic
    sampler, whose geometry is not differentiable in v1. If a mesh-vertex leaf
    participates in AD and one of those stochastic blocks will run, refuse
    loudly instead of silently detaching. Purely-enumerated component sets keep
    their geometry gradients untouched."""

    if config.ad_mode == "none":
        return
    sampler_active = bool(scene.structures) and bool(
        {"transmission", "scattering"} & set(config.components)
    )
    if not sampler_active:
        return
    if _ad_geometry_live(*(structure.vertices for structure in scene.structures)):
        raise RuntimeError(
            "BDPT ad_geometry='enumerated_blocks_only' (ADR-022): a mesh-vertex "
            "gradient would reach the stochastic transmission/scattering sampler, "
            "whose hit-point geometry is frozen in v1; the geometry gradient is "
            "refused loudly rather than silently detached. Restrict AD to a "
            "purely-enumerated component set or to material/frequency/tx_power "
            "leaves."
        )


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
        raise RuntimeError("witwin.channel.montecarlo.bdpt requires CUDA")

    info = build_info_fn()
    rayd = require_compiled(scene).rayd
    rayd_available = bool(info["uses_rayd_native"]) and rayd.available
    reflection_available = rayd_available
    diffraction_available = rayd_available and config.max_diffraction_order > 0
    transmission_available = rayd_available
    if (
        "reflection" in config.components
        and scene.structures
        and not reflection_available
    ):
        raise RuntimeError("reflection requires RayD native capability")
    if (
        "diffraction" in config.components
        and scene.structures
        and not diffraction_available
    ):
        raise RuntimeError("diffraction requires RayD native capability")
    if (
        "transmission" in config.components
        and scene.structures
        and not transmission_available
    ):
        raise RuntimeError("transmission requires RayD native capability")
    return _SolvePrep(
        native_samples=native_samples,
        native_max_depth=native_max_depth,
        selected_accumulation=selected_accumulation,
        workspace_bytes=workspace_bytes,
        info=info,
        rayd=rayd,
        reflection_available=reflection_available,
        diffraction_available=diffraction_available,
        transmission_available=transmission_available,
    )


def _collect_connection_samples(
    scene: Scene,
    config: Config,
    *,
    prep: _SolvePrep,
    workspace: _EndpointWorkspace,
    ledger: AdLaunchLedger | None = None,
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
    if config.coherent:
        # ADR-019 opt-in coherent combine. Config validation guarantees the
        # component set is coherent-eligible; route the whole solve through the
        # enumerated delta/UTD collector with phasor accumulation.
        return _collect_coherent_connection_samples(
            scene, config, prep=prep, workspace=workspace, ledger=ledger
        )
    rayd = prep.rayd
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
        ad = config.ad_mode != "none"
        # Live per-tx power under ad reattaches the tx_power gradient onto the
        # enumerated discrete blocks (linear coefficient) and the LoS companion.
        enumerated_tx_power = tx_power if ad else None
        sample_blocks: list[dict[str, torch.Tensor]] = []
        if los_light_state is not None:
            sample_blocks.append(
                _native_los_connection_samples(
                    rayd,
                    los_light_state,
                    endpoint_subpaths["sensor"],
                    scene_has_structures=bool(scene.structures),
                    frequency_hz=scene.frequency if ad else _host_frequency(scene),
                    mis=config.mis,
                    beta=config.power_heuristic_beta,
                    strategy_count=1,
                    ad=ad,
                    tx_power=enumerated_tx_power,
                    frequency_value=_host_frequency(scene),
                    ledger=ledger,
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
                topology_scene, config, tx_power=enumerated_tx_power
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
                topology_scene, config, tx_power=enumerated_tx_power
            )
            if diffraction_samples is not None:
                sample_blocks.append(diffraction_samples)
                launch_count += 1
        if config.coupled_paths:
            coupled_samples = _coupled_discrete_connection_samples(
                topology_scene, config, tx_power=enumerated_tx_power
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
            scattering_runtimes = rough_material_runtimes(require_compiled(scene))
        sampler_requested = transmission_requested or bool(scattering_runtimes)
        if sampler_requested:
            material_bundle = face_material_field_bundle(
                scene, device=tx_reference.device
            )
            scene_diagonal = scene_diagonal_m(scene)
        if transmission_requested:
            # ADR-020: pure straight-segment transmission routes through the
            # shared enumerated engine as a unit-mass discrete connection, like
            # reflection/diffraction (ADR-008/ADR-018), replacing the
            # polarization-agnostic straight-chain TE/TM mean with the full-Jones
            # layer-stack field. Mixed reflection+transmission chains are handled
            # separately by the event-selected shooting sampler below.
            transmission_samples = _transmission_discrete_connection_samples(
                topology_scene, config, tx_power=enumerated_tx_power
            )
            if transmission_samples is not None:
                sample_blocks.append(transmission_samples)
                transmission_chain_count = int(transmission_samples["valid"].sum())
            launch_count += 1
        if sampler_requested:
            ad = config.ad_mode != "none"
            mixed_blocks, event_counts = _transmission_sampled_connection_samples(
                rayd,
                tx_reference,
                tx_power,
                tx_polarization,
                rx_positions,
                rx_polarization,
                endpoint_subpaths["sensor"],
                material_bundle,
                # ADR-015 Part A / ADR-022: under AD the carrier stays a live
                # tensor so lambda/frequency chains carry a gradient; the primal
                # reads the detached host scalar and is bitwise unchanged.
                frequency_hz=scene.frequency if ad else _host_frequency(scene),
                samples=native_samples,
                max_depth=native_max_depth,
                seed=int(config.seed),
                mis=config.mis,
                beta=config.power_heuristic_beta,
                scattering_runtimes=scattering_runtimes,
                emit_mixed_transmission=transmission_requested,
                scene_diagonal=scene_diagonal,
                ad=ad,
                ledger=ledger,
            )
            sample_blocks.extend(mixed_blocks)
            launch_count += 3
        if sample_blocks:
            estimate_samples = (
                sample_blocks[0]
                if len(sample_blocks) == 1
                else bdpt_concat_connection_samples(tuple(sample_blocks))
            )
            accumulated = _accumulate_connection_samples(
                config,
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
    # ADR-022: the finalize is a linear map with native backward/jvp companions;
    # dispatch the differentiable twin under ad_mode != 'none' so component
    # gradients reach path_gain and the per-component powers. 'none' calls the
    # identical primal symbol (bitwise).
    ad = config.ad_mode != "none"
    component_maps: dict[str, torch.Tensor] | None = None
    point_component_matrices: dict[str, torch.Tensor] | None = None
    if grid is not None:
        component_maps = _component_maps_from_matrices(
            component_matrices,
            rows=grid.shape[0],
            cols=grid.shape[1],
        )
        finalize_maps = (
            bdpt_finalize_component_maps_ad if ad else bdpt_finalize_component_maps
        )
        finalized = finalize_maps(
            component_maps["los"],
            component_maps["reflection"],
            component_maps["diffraction"],
            component_maps["transmission"],
            component_maps["scattering"],
        )
    else:
        point_component_matrices = component_matrices
        finalize_points = (
            bdpt_finalize_point_components_ad
            if ad
            else bdpt_finalize_point_components
        )
        finalized = finalize_points(
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
                frequency_hz=_host_frequency(scene),
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
    ledger: AdLaunchLedger | None = None,
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
        ad_ledger=ledger,
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
    # ADR-022 per-solve companion accounting (mirrors montecarlo.basic). Under
    # ad_mode='none' no companion registers, so the ledger stays empty and the
    # metadata reports zero backward/jvp launches and zero tape.
    ledger = AdLaunchLedger()
    workspace = _build_endpoint_subpaths(
        scene,
        config,
        grid=grid,
        transmitter_tensors_fn=transmitter_tensors_fn,
        selected_accumulation=prep.selected_accumulation,
        ledger=ledger,
    )
    (
        component_matrices,
        estimate_samples,
        transmission_chain_count,
        event_counts,
        scattering_runtimes,
        launch_count,
    ) = _collect_connection_samples(
        scene, config, prep=prep, workspace=workspace, ledger=ledger
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
        ledger=ledger,
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
