"""Canonical typed engine for enumerated propagation paths."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
from witwin.channel.scene.endpoints import require_compiled

from witwin.channel.materials.evaluation import (
    _require_frequency_ad_constant_materials,
)
from witwin.channel.propagation.enumerated.coupled import (
    coupled_reflection_diffraction_topology,
)
from witwin.channel.propagation.enumerated.capacity import (
    sanitize_enumerated_capacity_transaction,
)
from witwin.channel.propagation.enumerated.diffraction import (
    _diffraction_topology_order1,
)
from witwin.channel.propagation.enumerated.los import _los_topology
from witwin.channel.propagation.enumerated.reflection import (
    _reflection_topology_multibounce,
    _reflection_topology_order1,
)
from witwin.channel.propagation.enumerated.transmission import (
    _transmission_topology,
)
from witwin.channel.propagation.fields.evaluation import evaluate_path_fields
from witwin.channel.propagation.geometry.endpoints import (
    receiver_positions_and_layout,
    transmitter_tensors,
)
from witwin.channel.propagation.models.capacity import CapacityExecutionCounts
from witwin.channel.propagation.models.contracts import TopologyConfig
from witwin.channel.propagation.models.evaluated import EvaluatedPaths
from witwin.channel.propagation.topology.concatenate import (
    _pad_topology_sequences,
    concatenate_path_blocks,
)
from witwin.channel.propagation.topology.export import (
    EvaluatedPathSidecars,
    evaluated_paths_from_block,
    evaluated_paths_from_result,
)
from witwin.channel.propagation.topology.kernels import (
    primitives as topology_primitives,
)
from witwin.channel.runtime.capacity import (
    SolveCapacityTransaction,
    create_solve_capacity_transaction,
)
from witwin.channel.scene.tensors import (
    _frequency_scalar,
    transmitter_polarizations,
)

if TYPE_CHECKING:
    from witwin.channel.scene.endpoints import SolverScene as Scene


def _path_components(config: TopologyConfig) -> set[str]:
    components = set(config.components)
    if config.max_depth == 0:
        # Every non-LoS component is a surface interaction that needs at least
        # one bounce. transmission is a wall penetration event and scattering is
        # a single-bounce rough-surface event, so both drop out at depth 0.
        components.discard("reflection")
        components.discard("diffraction")
        components.discard("transmission")
        components.discard("scattering")
    if int(getattr(config, "max_diffraction_order", 1)) == 0:
        components.discard("diffraction")
    return components


def _resolve_isb_taper(config: TopologyConfig) -> tuple[bool, float, float]:
    """Resolve the ADR-017 ISB boundary taper flag and per-stage widths.

    DEFAULT-OFF: when the taper is disabled the field/diffraction stages receive
    width 0.0 (every existing call path untouched) and the LoS membership stage
    receives the default 0.5, preserving the pre-ADR-017 calls byte-for-byte.
    """

    enabled = bool(getattr(config, "isb_boundary_taper", False))
    configured_width = float(getattr(config, "isb_boundary_taper_width", 0.5))
    field_width = configured_width if enabled else 0.0
    los_width = configured_width if enabled else 0.5
    return enabled, field_width, los_width


def _require_defer_capacity_terminal(value: bool) -> None:
    if type(value) is not bool:
        raise TypeError("defer_capacity_terminal must be a bool")


def _create_transmission_capacity_transaction(
    scene: Scene,
    config: TopologyConfig,
    components: set[str],
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
) -> SolveCapacityTransaction | None:
    has_work = (
        "transmission" in components
        and config.max_depth >= 1
        and bool(scene.structures)
        and int(tx_positions.shape[0]) > 0
        and int(rx_positions.shape[0]) > 0
    )
    if not has_work:
        return None
    return create_solve_capacity_transaction(tx_positions)


def _capacity_execution_summary(
    execution: CapacityExecutionCounts | None,
) -> tuple[CapacityExecutionCounts | None, int]:
    if execution is None:
        return None, 0
    return execution, execution.candidate_capacity


def _finish_capacity_boundary(
    evaluated: EvaluatedPaths,
    sidecars: EvaluatedPathSidecars,
    transaction: SolveCapacityTransaction | None,
    *,
    defer_terminal: bool,
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars]:
    if transaction is None or defer_terminal:
        return evaluated, sidecars
    evaluated, sidecars = sanitize_enumerated_capacity_transaction(evaluated, sidecars)
    transaction.terminal_check()
    return evaluated, replace(sidecars, capacity_transaction=None)


def evaluate_enumerated_paths(
    scene: Scene,
    config: TopologyConfig,
    *,
    frequency_value: float | None = None,
    coupled_rx_streaming: bool = False,
    defer_capacity_terminal: bool = False,
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars]:
    """Discover, select, and evaluate canonical enumerated propagation rows.

    ``coupled_rx_streaming`` streams coupled reflection-diffraction discovery
    over receiver blocks so a full grid solve stays under the per-block
    candidate budget (ADR-011). The deterministic grid solver sets it; the path
    and Monte Carlo callers keep the single-shot total-cap discovery, so their
    coupled behavior is unchanged.

    ``defer_capacity_terminal`` is reserved for Path and Deterministic outer
    solvers, which must sanitize scattering-appended rows and enqueue the one
    terminal observer only after their public result/array packing. ADR-008
    opaque oracle callers keep the default and complete the transaction here.
    """

    _require_defer_capacity_terminal(defer_capacity_terminal)
    device = torch.device("cuda")
    tx_positions, tx_power = transmitter_tensors(scene, device=device)
    field_tx_power = (
        tx_power.detach()
        if bool(getattr(config, "_detach_field_tx_power", False))
        else tx_power
    )
    tx_polarizations = transmitter_polarizations(scene, device=device)
    rx_positions, _ = receiver_positions_and_layout(scene, device=device)
    compiled = require_compiled(scene)
    # One host read of a tensor frequency for the whole export: discovery and
    # the field seam below share this detached scalar (audit M3). Callers
    # that already read it (the solver seams) pass it in.
    frequency_hz = (
        _frequency_scalar(scene) if frequency_value is None else float(frequency_value)
    )
    ad_mode = str(getattr(config, "ad_mode", "none"))
    if ad_mode != "none":
        _require_frequency_ad_constant_materials(scene, compiled, ad_mode=ad_mode)
    components = _path_components(config)
    # ISB boundary taper (ADR-017): DEFAULT-OFF. When off, width 0.0 flows to the
    # field/diffraction stages and every existing call path is untouched.
    isb_boundary_taper, isb_boundary_taper_width, isb_los_width = _resolve_isb_taper(
        config
    )
    coupled_paths = bool(getattr(config, "coupled_paths", False))
    sequence_width = max(int(config.max_depth), 2 if coupled_paths else 0)
    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    visibility_rejection_count = 0
    candidate_count = 0
    guardrail_count = 0
    diffraction_vector_field = None
    capacity_execution: CapacityExecutionCounts | None = None
    capacity_transaction: SolveCapacityTransaction | None = None

    capacity_transaction = _create_transmission_capacity_transaction(
        scene, config, components, tx_positions, rx_positions
    )

    if "los" in components:
        los_block, los_launches, los_candidates, los_visibility_rejections = (
            _los_topology(
                scene,
                compiled,
                tx_positions,
                tx_power,
                rx_positions,
                tx_polarizations,
                frequency_hz=frequency_hz,
                sequence_width=sequence_width,
                isb_boundary_taper=isb_boundary_taper,
                isb_boundary_taper_width=isb_los_width,
            )
        )
        launch_count += los_launches
        candidate_count += los_candidates
        visibility_rejection_count += los_visibility_rejections
        blocks.append(los_block)
    if "reflection" in components and config.max_depth >= 1:
        block, reflection_launches = _reflection_topology_order1(
            scene,
            compiled,
            tx_positions,
            tx_power,
            rx_positions,
            frequency_hz=frequency_hz,
        )
        launch_count += reflection_launches
        candidate_count += int(block["valid"].numel())
        blocks.append(block)
    if "reflection" in components and config.max_depth >= 2:
        block, reflection_launches, reflection_candidates = (
            _reflection_topology_multibounce(
                scene,
                compiled,
                tx_positions,
                tx_power,
                rx_positions,
                frequency_hz=frequency_hz,
                min_depth=2,
                max_depth=int(config.max_depth),
                max_paths=config.max_paths,
            )
        )
        launch_count += reflection_launches
        candidate_count += int(reflection_candidates)
        blocks.append(block)
    if "diffraction" in components and config.max_depth >= 1:
        block, diffraction_launches, diffraction_vector_field = (
            _diffraction_topology_order1(
                scene,
                compiled,
                tx_positions,
                tx_power,
                rx_positions,
                frequency_hz=frequency_hz,
                isb_boundary_taper_width=isb_boundary_taper_width,
            )
        )
        launch_count += diffraction_launches
        candidate_count += int(block["valid"].numel())
        blocks.append(block)
    if "transmission" in components and config.max_depth >= 1:
        block, transmission_launches, transmission_execution = _transmission_topology(
            scene,
            compiled,
            tx_positions,
            rx_positions,
            max_depth=int(config.max_depth),
            ad_mode=ad_mode,
            failure_state=(
                capacity_transaction.failure_state
                if capacity_transaction is not None
                else None
            ),
        )
        launch_count += transmission_launches
        capacity_execution, transmission_candidates = _capacity_execution_summary(
            transmission_execution
        )
        candidate_count += transmission_candidates
        blocks.append(block)
    if (
        coupled_paths
        and config.max_depth >= 2
        and {"reflection", "diffraction"}.issubset(components)
    ):
        block, coupled_launches, coupled_candidates = (
            coupled_reflection_diffraction_topology(
                scene,
                compiled,
                tx_positions,
                rx_positions,
                candidate_limit=int(
                    getattr(config, "coupled_candidate_limit", 1_000_000)
                ),
                rx_streamed=coupled_rx_streaming,
            )
        )
        launch_count += coupled_launches
        candidate_count += coupled_candidates
        blocks.append(block)
    if len(blocks) == 1 and components == {"los"} and config.max_paths is None:
        evaluated, sidecars = evaluated_paths_from_result(
            SimpleNamespace(
                **blocks[0],
                launch_count=launch_count,
                visibility_rejection_count=visibility_rejection_count,
                selected_edge_count=0,
                candidate_count=candidate_count,
                guardrail_count=guardrail_count,
            )
        )
        evaluated, execution = evaluate_path_fields(
            scene,
            compiled,
            evaluated,
            sidecars.execution,
            tx_positions,
            field_tx_power,
            rx_positions,
            components=components,
            ad_mode=ad_mode,
            frequency_value=frequency_hz,
            isb_boundary_taper_width=isb_boundary_taper_width,
        )
        sidecars = replace(
            sidecars,
            execution=execution,
            capacity_execution=capacity_execution,
            capacity_transaction=capacity_transaction,
        )
        return _finish_capacity_boundary(
            evaluated,
            sidecars,
            capacity_transaction,
            defer_terminal=defer_capacity_terminal,
        )
    padded_blocks = [
        block
        if "primitive_sequence" in block
        and int(block["primitive_sequence"].shape[1]) == sequence_width
        else _pad_topology_sequences(block, width=sequence_width)
        for block in blocks
    ]
    paths = concatenate_path_blocks(padded_blocks, device=device)
    selected_edge_count = topology_primitives.deterministic_selected_edge_count(
        paths["edge_id"]
    )
    evaluated, sidecars = evaluated_paths_from_block(
        paths,
        max_paths=config.max_paths,
        max_paths_scope=str(getattr(config, "max_paths_scope", "global")),
        tx_count=len(scene.transmitters),
        max_depth=config.max_depth,
        launch_count=launch_count,
        visibility_rejection_count=visibility_rejection_count,
        selected_edge_count=selected_edge_count,
        candidate_count=candidate_count,
        guardrail_count=guardrail_count,
    )
    sidecars = replace(
        sidecars,
        diffraction_vector_field=diffraction_vector_field,
        capacity_execution=capacity_execution,
        capacity_transaction=capacity_transaction,
    )
    evaluated, execution = evaluate_path_fields(
        scene,
        compiled,
        evaluated,
        sidecars.execution,
        tx_positions,
        field_tx_power,
        rx_positions,
        components=components,
        ad_mode=ad_mode,
        frequency_value=frequency_hz,
        isb_boundary_taper_width=isb_boundary_taper_width,
    )
    sidecars = replace(sidecars, execution=execution)
    return _finish_capacity_boundary(
        evaluated,
        sidecars,
        capacity_transaction,
        defer_terminal=defer_capacity_terminal,
    )
