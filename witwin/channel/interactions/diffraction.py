# Copyright Xingyu Chen.
# The diffraction interaction: discovery planning, geometry, and enumerated orchestration.

"""The diffraction interaction: discovery planning, geometry, and enumerated orchestration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from witwin.channel.scene.endpoints import transmitter_polarizations_f32
from witwin.channel.materials import face_material_field_bundle, face_material_tensors
from witwin.channel.scene.compiler import (
    LIGHT_SPEED_M_PER_S as _LIGHT_SPEED_M_PER_S,
)
from witwin.channel.propagation.geometry import (
    cached_diffraction_edge_geometry as _cached_diffraction_edge_geometry,
    diffraction_edge_geometry as _diffraction_edge_geometry,
)
from witwin.channel.propagation.topology import (
    concatenate_path_blocks,
)
from witwin.channel.interactions.reflection import (
    _MULTIBOUNCE_PAIR_CHUNK_SIZE,
)
from witwin.channel.propagation.topology import _ensure_topology_fields
from witwin.channel.kernels import fields as field_kernels
from witwin.channel.kernels import geometry as geometry_kernels
from witwin.channel.kernels import topology as topology_kernels
from witwin.channel.runtime import (
    CudaProfileMark,
    CudaProfileRange,
    cuda_profile_mark,
    cuda_profile_range,
    profiled_cuda_range,
)

if TYPE_CHECKING:
    from witwin.channel.scene.compiler import CompiledScene
    from witwin.channel.scene.endpoints import SolverScene as Scene
    from witwin.channel.scene.resources import RayDSceneResource


@dataclass(frozen=True, slots=True)
class DiffractionOrder1Plan:
    preserve_imported_edges: bool
    tx_count: int
    rx_count: int


@dataclass(frozen=True, slots=True)
class DiffractionRxChunkRequest:
    rx_start: int
    rx_end: int
    capacity: int


def prepare_diffraction_order1_plan(
    *, metadata: Mapping[str, object], tx_count: int, rx_count: int,
) -> DiffractionOrder1Plan:
    mitsuba_metadata = metadata.get("mitsuba", {})
    # Channel's merge_shapes import keeps the selected boundary-edge table
    # intact. The synthetic-scene path instead merges coincident structure
    # boundaries into one physical wedge.
    preserve_imported_edges = isinstance(mitsuba_metadata, dict) and bool(
        mitsuba_metadata.get("merge_shapes", False)
    )
    return DiffractionOrder1Plan(
        preserve_imported_edges=preserve_imported_edges,
        tx_count=int(tx_count),
        rx_count=int(rx_count),
    )


def iter_diffraction_rx_chunk_requests(
    plan: DiffractionOrder1Plan, *, state_count: int,
) -> Iterator[DiffractionRxChunkRequest]:
    rx_chunk_size = max(1, _MULTIBOUNCE_PAIR_CHUNK_SIZE // state_count)
    for rx_start in range(0, plan.rx_count, rx_chunk_size):
        rx_end = min(rx_start + rx_chunk_size, plan.rx_count)
        yield DiffractionRxChunkRequest(
            rx_start=rx_start,
            rx_end=rx_end,
            capacity=(rx_end - rx_start) * state_count,
        )


@dataclass(frozen=True, slots=True)
class DiffractionEdgeGeometry:
    selected: torch.Tensor
    edge_position: torch.Tensor
    edge_direction: torch.Tensor
    edge_length: torch.Tensor
    line_min: torch.Tensor
    line_max: torch.Tensor
    n0: torch.Tensor
    n1: torch.Tensor
    face0: torch.Tensor
    face1: torch.Tensor
    exterior_angle: torch.Tensor


def query_diffraction_edges(
    rayd: RayDSceneResource, *, preserve_imported_edges: bool,
) -> DiffractionEdgeGeometry:
    raw = (
        _diffraction_edge_geometry(rayd.edge_records())
        if preserve_imported_edges
        else _cached_diffraction_edge_geometry(rayd)
    )
    return DiffractionEdgeGeometry(
        selected=raw[0],
        edge_position=raw[1],
        edge_direction=raw[2],
        edge_length=raw[3],
        line_min=raw[4],
        line_max=raw[5],
        n0=raw[6],
        n1=raw[7],
        face0=raw[8],
        face1=raw[9],
        exterior_angle=raw[10],
    )


def selected_diffraction_edges(
    rayd: RayDSceneResource, *, preserve_imported_edges: bool,
) -> tuple[DiffractionEdgeGeometry, torch.Tensor]:
    """The edge table and its selected edge indices, queried once for every transmitter."""

    edges = query_diffraction_edges(rayd, preserve_imported_edges=preserve_imported_edges)
    return edges, topology_kernels.mc_selected_edge_indices(edges.selected)


@dataclass(frozen=True, slots=True)
class DiffractionStateGeometry:
    edge_index: torch.Tensor
    edge_position: torch.Tensor
    edge_direction: torch.Tensor
    edge_t_min: torch.Tensor
    edge_t_max: torch.Tensor
    n0: torch.Tensor
    n1: torch.Tensor
    prim0: torch.Tensor
    prim1: torch.Tensor
    exterior_angle: torch.Tensor
    source: torch.Tensor
    source_power: torch.Tensor


def name_diffraction_states(states: tuple[torch.Tensor, ...]) -> DiffractionStateGeometry:
    if len(states) != 12:
        raise ValueError("diffraction state tuple must contain exactly 12 tensors")
    return DiffractionStateGeometry(
        edge_index=states[0],
        edge_position=states[1],
        edge_direction=states[2],
        edge_t_min=states[3],
        edge_t_max=states[4],
        n0=states[5],
        n1=states[6],
        prim0=states[7],
        prim1=states[8],
        exterior_angle=states[9],
        source=states[10],
        source_power=states[11],
    )


@dataclass(frozen=True, slots=True)
class DiffractionVisibleStatePlan:
    edge_index: torch.Tensor
    edge_position: torch.Tensor
    edge_direction: torch.Tensor
    edge_t_min: torch.Tensor
    edge_t_max: torch.Tensor
    n0: torch.Tensor
    n1: torch.Tensor
    prim0: torch.Tensor
    prim1: torch.Tensor
    exterior_angle: torch.Tensor
    source: torch.Tensor
    source_power: torch.Tensor
    active: torch.Tensor


def plan_tx_visible_diffraction_states(
    rayd: RayDSceneResource, states: tuple[torch.Tensor, ...], tx: torch.Tensor, *,
    per_state_source: bool = False,
) -> DiffractionVisibleStatePlan:
    """Transmitter-visible states; ``per_state_source`` checks each state against its own source."""

    named = name_diffraction_states(states)
    active = geometry_kernels.diffraction_tx_visible_state_plan(
        rayd.require_resource(),
        tx,
        named.edge_index,
        named.edge_position,
        named.edge_direction,
        named.edge_t_min,
        named.edge_t_max,
        named.n0,
        named.n1,
        named.prim0,
        named.prim1,
        named.exterior_angle,
        named.source,
        named.source_power,
        per_state_source=per_state_source,
    )
    return DiffractionVisibleStatePlan(
        edge_index=named.edge_index,
        edge_position=named.edge_position,
        edge_direction=named.edge_direction,
        edge_t_min=named.edge_t_min,
        edge_t_max=named.edge_t_max,
        n0=named.n0,
        n1=named.n1,
        prim0=named.prim0,
        prim1=named.prim1,
        exterior_angle=named.exterior_angle,
        source=named.source,
        source_power=named.source_power,
        active=active,
    )


@dataclass(frozen=True, slots=True)
class DiffractionOrder1Query:
    handle: object
    tx_position: torch.Tensor
    tx_polarization: torch.Tensor
    rx_positions: torch.Tensor
    active: torch.Tensor
    states: DiffractionVisibleStatePlan
    material_eta_r: torch.Tensor
    material_sigma: torch.Tensor
    material_mu_r: torch.Tensor
    material_gain: torch.Tensor
    material_valid: torch.Tensor
    state_count: int
    capacity: int
    wavelength: float
    # ISB boundary taper, D member. 0.0 (default) reproduces the hard
    # RayD GO step; > 0 notches the incident-boundary odd part over the congruent
    # window inside the shared UTD header (pair.isbTaperWidthScale).
    isb_taper_width_scale: float = 0.0
    # Per-state transmitter index (int32): every state then carries its own transmitter, the
    # launch has one transmitter lane, and the exported tx_id is that index.
    state_source_index: torch.Tensor | None = None
    # Slab / layer-stack reflection of the face terms (per face: thickness, layer-store row; then
    # the layer store), so a face diffracts with the coefficient its specular reflection uses.
    face_layers: tuple[torch.Tensor, ...] | None = None

    def __post_init__(self) -> None:
        if self.active is not self.states.active:
            raise ValueError("diffraction query active must alias the visible state plan")
        if self.state_count != int(self.states.edge_index.shape[0]):
            raise ValueError("diffraction query state_count must equal plan capacity N")


@dataclass(frozen=True, slots=True)
class DiffractionOrder1Geometry:
    valid: torch.Tensor
    tx_id: torch.Tensor
    rx_id: torch.Tensor
    depth: torch.Tensor
    edge_id: torch.Tensor
    delay_s: torch.Tensor
    x_re: torch.Tensor
    x_im: torch.Tensor
    y_re: torch.Tensor
    y_im: torch.Tensor
    z_re: torch.Tensor
    z_im: torch.Tensor
    interaction_position: torch.Tensor


def query_diffraction_order1(query: DiffractionOrder1Query) -> DiffractionOrder1Geometry:
    states = query.states
    cuda_profile_mark(CudaProfileMark.OPTIX_TRAVERSAL)
    cuda_profile_mark(CudaProfileMark.DIFFRACTION_EXPORTER_REQUEST)
    raw = geometry_kernels.rayd_diffraction_paths_order1_forward(
        query.handle,
        query.tx_position,
        query.tx_polarization,
        query.rx_positions,
        query.active,
        states.edge_index,
        states.edge_position,
        states.edge_direction,
        states.edge_t_min,
        states.edge_t_max,
        states.n0,
        states.n1,
        states.prim0,
        states.prim1,
        states.exterior_angle,
        states.source,
        states.source_power,
        query.material_eta_r,
        query.material_sigma,
        query.material_mu_r,
        query.material_gain,
        query.material_valid,
        query.state_count,
        query.capacity,
        query.wavelength,
        query.isb_taper_width_scale,
        query.state_source_index,
        list(query.face_layers) if query.face_layers is not None else None,
    )
    return DiffractionOrder1Geometry(
        valid=raw[1],
        tx_id=raw[2],
        rx_id=raw[3],
        depth=raw[4],
        edge_id=raw[5],
        delay_s=raw[8],
        x_re=raw[9],
        x_im=raw[10],
        y_re=raw[11],
        y_im=raw[12],
        z_re=raw[13],
        z_im=raw[14],
        interaction_position=raw[15],
    )


def _deterministic_diffraction_states(
    rayd: RayDSceneResource, tx: torch.Tensor, tx_power: torch.Tensor, tx_index: int, *,
    preserve_imported_edges: bool = False,
    edges: tuple[DiffractionEdgeGeometry, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Per-transmitter diffraction states; ``edges`` reuses one solve's selected edge table."""

    if edges is None:
        edges = selected_diffraction_edges(rayd, preserve_imported_edges=preserve_imported_edges)
    edge_table, selected_indices = edges
    return topology_kernels.deterministic_diffraction_state_pack(
        selected_indices,
        edge_table.edge_position,
        edge_table.edge_direction,
        edge_table.line_min,
        edge_table.line_max,
        edge_table.n0,
        edge_table.n1,
        edge_table.face0,
        edge_table.face1,
        edge_table.exterior_angle,
        tx,
        tx_power,
        int(tx_index),
    )


def _diffraction_block(
    out: DiffractionOrder1Geometry, *, rx_start: int,
) -> dict[str, torch.Tensor] | None:
    """Compact one batched exporter output into a diffraction topology block.

    The rows carry the exporter's UTD vector field as ``field_xyz`` and no scalar fields; the
    field stage evaluates those.
    """

    with cuda_profile_range(CudaProfileRange.DIFFRACTION_TOPOLOGY_PACKING):
        block = topology_kernels.deterministic_diffraction_topology_compact(
            valid=out.valid,
            tx_id=out.tx_id,
            rx_id=out.rx_id,
            rx_offset=rx_start,
            depth=out.depth,
            edge_id=out.edge_id,
            delay_s=out.delay_s,
            x_re=out.x_re,
            x_im=out.x_im,
            y_re=out.y_re,
            y_im=out.y_im,
            z_re=out.z_re,
            z_im=out.z_im,
            interaction_position=out.interaction_position,
        )
    if int(block["valid"].shape[0]) == 0:
        return None
    return _ensure_topology_fields(block)


def _batched_diffraction_states(
    edges: tuple[DiffractionEdgeGeometry, torch.Tensor], tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Diffraction states of every transmitter (transmitter-major) and their transmitter index."""

    geometry, selected_indices = edges
    states = topology_kernels.deterministic_diffraction_state_pack(
        selected_indices,
        geometry.edge_position,
        geometry.edge_direction,
        geometry.line_min,
        geometry.line_max,
        geometry.n0,
        geometry.n1,
        geometry.face0,
        geometry.face1,
        geometry.exterior_angle,
        tx_positions.reshape(-1).contiguous(),
        tx_power.to(dtype=torch.float32).contiguous(),
        0,
    )
    edge_states = int(selected_indices.shape[0])
    source_index = torch.arange(
        int(tx_positions.shape[0]), device=tx_positions.device, dtype=torch.int32
    ).repeat_interleave(edge_states)
    return states, source_index


@profiled_cuda_range(CudaProfileRange.DIFFRACTION_TOTAL_STAGE)
def _diffraction_topology_order1(
    scene: Scene, compiled: CompiledScene, tx_positions: torch.Tensor, tx_power: torch.Tensor,
    rx_positions: torch.Tensor, *, frequency_hz: float, isb_boundary_taper_width: float = 0.0,
    tx_polarizations: torch.Tensor | None = None, transmission_crossings: int = 0,
) -> tuple[dict[str, torch.Tensor], int]:
    """Order-1 diffraction topology for every transmitter and receiver.

    ``tx_polarizations`` carries explicit per-transmitter polarizations when the
    caller binds endpoint batches instead of scene transmitters. ``transmission_crossings`` is the
    wall-crossing limit of the solve's straight-through transmission paths (0 when it has none);
    the incident-boundary terms then compensate only the step those paths leave.
    """


    device = tx_positions.device
    rayd = compiled.rayd
    if not scene.structures or tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return (
            _ensure_topology_fields(
                {
                    "valid": torch.empty((0,), device=device, dtype=torch.bool),
                    "tx_id": torch.empty((0,), device=device, dtype=torch.int32),
                    "rx_id": torch.empty((0,), device=device, dtype=torch.int32),
                    "depth": torch.empty((0,), device=device, dtype=torch.int32),
                    "component_id": torch.empty((0,), device=device, dtype=torch.int32),
                    "primitive_id": torch.empty((0,), device=device, dtype=torch.int32),
                    "edge_id": torch.empty((0,), device=device, dtype=torch.int32),
                    "path_length_m": torch.empty(
                        (0,), device=device, dtype=torch.float32
                    ),
                    "delay_s": torch.empty((0,), device=device, dtype=torch.float32),
                    "path_gain": torch.empty((0,), device=device, dtype=torch.float32),
                }
            ),
            0,
        )
    if not rayd.available:
        raise RuntimeError(
            "diffraction discovery requires RayD native scene capability"
        )

    face_eps_r, face_sigma_e, face_mu_r, material_gain, material_valid = (
        face_material_tensors(compiled, device=device)
    )
    bundle = face_material_field_bundle(compiled, device=device)
    face_layers = (
        bundle["thickness"], bundle["material_id"], bundle["layer_offset"], bundle["layer_count"],
        bundle["layer_thickness_m"], bundle["layer_eps_r"], bundle["layer_sigma_e"],
        bundle["layer_mu_r"],
        *field_kernels.wedge_transmission_layers(
            bundle["transmission_row"], crossings=transmission_crossings),
    )
    # Per-transmitter polarization threaded into the RayD UTD op: the
    # incident field basis must use the real scene polarization, not a
    # fabricated z-axis vector.
    if tx_polarizations is None:
        tx_polarizations = transmitter_polarizations_f32(scene, device=device)
    else:
        tx_polarizations = tx_polarizations.to(device=device, dtype=torch.float32).contiguous()
    wavelength = _LIGHT_SPEED_M_PER_S / float(frequency_hz)
    handle = rayd.require_resource()
    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    rx_count = int(rx_positions.shape[0])
    plan = prepare_diffraction_order1_plan(
        metadata=scene.metadata,
        tx_count=int(tx_positions.shape[0]),
        rx_count=rx_count,
    )
    edges = selected_diffraction_edges(
        rayd, preserve_imported_edges=plan.preserve_imported_edges
    )
    # Every transmitter's states are packed, visibility-planned and exported together: each state
    # carries its own source and transmitter index, so one launch per stage serves every
    # transmitter, and receivers are chunked to bound the rx x state workspace.
    states, source_index = _batched_diffraction_states(edges, tx_positions, tx_power)
    visible_plan = plan_tx_visible_diffraction_states(
        rayd, states, tx_positions[0], per_state_source=True
    )
    state_count = int(visible_plan.edge_index.shape[0])
    if state_count > 0:
        for rx_request in iter_diffraction_rx_chunk_requests(plan, state_count=state_count):
            rx_start = rx_request.rx_start
            rx_end = rx_request.rx_end
            with cuda_profile_range(CudaProfileRange.DIFFRACTION_EXPORTER):
                out = query_diffraction_order1(
                    DiffractionOrder1Query(
                        handle=handle,
                        tx_position=tx_positions.contiguous(),
                        tx_polarization=tx_polarizations.contiguous(),
                        rx_positions=rx_positions[rx_start:rx_end].contiguous(),
                        active=visible_plan.active,
                        states=visible_plan,
                        material_eta_r=face_eps_r,
                        material_sigma=face_sigma_e,
                        material_mu_r=face_mu_r,
                        material_gain=material_gain,
                        material_valid=material_valid,
                        state_count=state_count,
                        capacity=rx_request.capacity,
                        wavelength=float(wavelength),
                        # ISB boundary taper, D member. 0.0 when the
                        # switch is off keeps the RayD export bit-identical; the
                        # width notches the incident-boundary odd part in the header.
                        isb_taper_width_scale=float(isb_boundary_taper_width),
                        state_source_index=source_index,
                        face_layers=face_layers,
                    )
                )
            launch_count += 1
            block = _diffraction_block(out, rx_start=rx_start)
            if block is not None:
                blocks.append(block)
    return (
        _ensure_topology_fields(concatenate_path_blocks(blocks, device=device)),
        launch_count,
    )
