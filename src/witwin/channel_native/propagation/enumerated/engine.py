"""Canonical typed engine for enumerated propagation paths."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from witwin.channel_native.materials.evaluation import (
    _require_frequency_ad_constant_materials,
)
from witwin.channel_native.propagation.enumerated.coupled import (
    _coupled_reflection_diffraction_topology_order2,
)
from witwin.channel_native.propagation.enumerated.diffraction import (
    _diffraction_topology_order1,
)
from witwin.channel_native.propagation.enumerated.los import _los_topology
from witwin.channel_native.propagation.enumerated.reflection import (
    _reflection_topology_multibounce,
    _reflection_topology_order1,
)
from witwin.channel_native.propagation.enumerated.transmission import (
    _transmission_topology,
)
from witwin.channel_native.propagation.fields.evaluation import evaluate_path_fields
from witwin.channel_native.propagation.geometry.endpoints import (
    receiver_positions_and_layout,
    transmitter_tensors,
)
from witwin.channel_native.propagation.models.contracts import TopologyConfig
from witwin.channel_native.propagation.models.evaluated import EvaluatedPaths
from witwin.channel_native.propagation.topology.concatenate import (
    _pad_topology_sequences,
    concatenate_path_blocks,
)
from witwin.channel_native.propagation.topology.export import (
    EvaluatedPathSidecars,
    evaluated_paths_from_block,
    evaluated_paths_from_result,
)
from witwin.channel_native.propagation.topology.kernels import (
    primitives as topology_primitives,
)
from witwin.channel_native.core.scene_tensors import _frequency_scalar

if TYPE_CHECKING:
    from witwin.channel_native.core.scene import Scene


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


def evaluate_enumerated_paths(
    scene: Scene,
    config: TopologyConfig,
    *,
    frequency_value: float | None = None,
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars]:
    """Discover, select, and evaluate canonical enumerated propagation rows."""

    device = torch.device("cuda")
    tx_positions, tx_power = transmitter_tensors(scene, device=device)
    rx_positions, _ = receiver_positions_and_layout(scene, device=device)
    compiled = scene.compile()
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
    coupled_paths = bool(getattr(config, "coupled_paths", False))
    sequence_width = max(int(config.max_depth), 2 if coupled_paths else 0)
    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    visibility_rejection_count = 0
    candidate_count = 0
    guardrail_count = 0
    diffraction_vector_field = None

    if "los" in components:
        los_block, los_launches, los_candidates, los_visibility_rejections = (
            _los_topology(
                scene,
                compiled,
                tx_positions,
                tx_power,
                rx_positions,
                frequency_hz=frequency_hz,
                sequence_width=sequence_width,
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
            )
        )
        launch_count += diffraction_launches
        candidate_count += int(block["valid"].numel())
        blocks.append(block)
    if "transmission" in components and config.max_depth >= 1:
        block, transmission_launches, transmission_candidates, transmission_guardrails = (
            _transmission_topology(
                scene,
                compiled,
                tx_positions,
                rx_positions,
                max_depth=int(config.max_depth),
            )
        )
        launch_count += transmission_launches
        candidate_count += transmission_candidates
        guardrail_count += transmission_guardrails
        blocks.append(block)
    if (
        coupled_paths
        and config.max_depth >= 2
        and {"reflection", "diffraction"}.issubset(components)
    ):
        block, coupled_launches, coupled_candidates = (
            _coupled_reflection_diffraction_topology_order2(
                scene,
                compiled,
                tx_positions,
                rx_positions,
                candidate_limit=int(
                    getattr(config, "coupled_candidate_limit", 1_000_000)
                ),
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
            tx_power,
            rx_positions,
            components=components,
            ad_mode=ad_mode,
            frequency_value=frequency_hz,
        )
        return evaluated, replace(sidecars, execution=execution)
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
    if diffraction_vector_field is not None:
        sidecars = replace(
            sidecars, diffraction_vector_field=diffraction_vector_field
        )
    evaluated, execution = evaluate_path_fields(
        scene,
        compiled,
        evaluated,
        sidecars.execution,
        tx_positions,
        tx_power,
        rx_positions,
        components=components,
        ad_mode=ad_mode,
        frequency_value=frequency_hz,
    )
    return evaluated, replace(sidecars, execution=execution)
