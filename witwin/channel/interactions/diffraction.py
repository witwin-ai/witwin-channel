# Copyright Xingyu Chen.
# The diffraction interaction: discovery planning, geometry, and enumerated orchestration.

"""The diffraction interaction: discovery planning, geometry, and enumerated orchestration.

This module gathers what used to be three files - the lazy first-order
discovery plan (``propagation.topology.discovery.diffraction``), the typed edge
and first-order path geometry queries (``propagation.geometry.diffraction``),
and the enumerated topology owner (``propagation.enumerated.diffraction``) - so
one file holds the whole concept. The native facades it dispatches through stay
in :mod:`witwin.channel.kernels`; the shared edge-state helpers stay in
:mod:`witwin.channel.propagation.geometry`, and the shared receiver
chunk size stays with its owner in
:mod:`witwin.channel.interactions.reflection`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from witwin.channel.scene.endpoints import transmitter_polarizations_f32
from witwin.channel.materials import face_material_tensors
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
    from witwin.channel.scene.endpoints import SolverScene as Scene


@dataclass(frozen=True, slots=True)
class DiffractionOrder1Plan:
    preserve_imported_edges: bool
    tx_count: int
    rx_count: int


@dataclass(frozen=True, slots=True)
class DiffractionTxRequest:
    tx_index: int
    tx: torch.Tensor


@dataclass(frozen=True, slots=True)
class DiffractionRxChunkRequest:
    rx_start: int
    rx_end: int
    capacity: int


def prepare_diffraction_order1_plan(
    *,
    metadata: Mapping[str, object],
    tx_count: int,
    rx_count: int,
) -> DiffractionOrder1Plan:
    mitsuba_metadata = metadata.get("mitsuba", {})
    # Channel's merge_shapes import keeps the selected boundary-edge table
    # intact. The synthetic-scene path instead merges coincident structure
    # boundaries into one physical wedge (the single-wedge test contract).
    preserve_imported_edges = isinstance(mitsuba_metadata, dict) and bool(
        mitsuba_metadata.get("merge_shapes", False)
    )
    return DiffractionOrder1Plan(
        preserve_imported_edges=preserve_imported_edges,
        tx_count=int(tx_count),
        rx_count=int(rx_count),
    )


def iter_diffraction_tx_requests(
    plan: DiffractionOrder1Plan,
    *,
    tx_positions: torch.Tensor,
) -> Iterator[DiffractionTxRequest]:
    for tx_index in range(plan.tx_count):
        yield DiffractionTxRequest(tx_index=tx_index, tx=tx_positions[tx_index])


def iter_diffraction_rx_chunk_requests(
    plan: DiffractionOrder1Plan,
    *,
    state_count: int,
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
    rayd: object,
    *,
    preserve_imported_edges: bool,
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


def name_diffraction_states(
    states: tuple[torch.Tensor, ...],
) -> DiffractionStateGeometry:
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
    rayd: object,
    states: tuple[torch.Tensor, ...],
    tx: torch.Tensor,
) -> DiffractionVisibleStatePlan:
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
    # ISB boundary taper (ADR-017), D member. 0.0 (default) reproduces the hard
    # RayD GO step; > 0 notches the incident-boundary odd part over the congruent
    # window inside the shared UTD header (pair.isbTaperWidthScale).
    isb_taper_width_scale: float = 0.0

    def __post_init__(self) -> None:
        if self.active is not self.states.active:
            raise ValueError("diffraction query active must alias the visible state plan")
        if self.state_count != int(self.states.edge_index.shape[0]):
            raise ValueError("diffraction query state_count must equal plan capacity N")


@dataclass(frozen=True, slots=True)
class DiffractionOrder1Geometry:
    valid: torch.Tensor
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


def query_diffraction_order1(
    query: DiffractionOrder1Query,
) -> DiffractionOrder1Geometry:
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
    )
    return DiffractionOrder1Geometry(
        valid=raw[1],
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
    rayd: object,
    tx: torch.Tensor,
    tx_power: torch.Tensor,
    tx_index: int,
    *,
    preserve_imported_edges: bool = False,
) -> tuple[torch.Tensor, ...]:
    edges = query_diffraction_edges(
        rayd,
        preserve_imported_edges=preserve_imported_edges,
    )
    return topology_kernels.deterministic_diffraction_state_pack(
        topology_kernels.mc_selected_edge_indices(edges.selected),
        edges.edge_position,
        edges.edge_direction,
        edges.line_min,
        edges.line_max,
        edges.n0,
        edges.n1,
        edges.face0,
        edges.face1,
        edges.exterior_angle,
        tx,
        tx_power,
        int(tx_index),
    )


@profiled_cuda_range(CudaProfileRange.DIFFRACTION_TOTAL_STAGE)
def _diffraction_topology_order1(
    scene: Scene,
    compiled: object,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    frequency_hz: float,
    isb_boundary_taper_width: float = 0.0,
) -> tuple[dict[str, torch.Tensor], int, torch.Tensor]:
    from witwin.channel.kernels import fields as field_kernels

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
            torch.zeros(
                (int(tx_positions.shape[0]), int(rx_positions.shape[0]), 3),
                device=device,
                dtype=torch.complex64,
            ),
        )
    if not rayd.available:
        raise RuntimeError(
            "deterministic diffraction requires RayD native scene capability"
        )

    face_eps_r, face_sigma_e, face_mu_r, material_gain, material_valid = (
        face_material_tensors(compiled, device=device)
    )
    # Per-transmitter polarization threaded into the RayD UTD op (R5 fix): the
    # incident field basis must use the real scene polarization, not a
    # fabricated z-axis vector.
    tx_polarizations = transmitter_polarizations_f32(scene, device=device)
    wavelength = _LIGHT_SPEED_M_PER_S / float(frequency_hz)
    handle = rayd.require_resource()
    blocks: list[dict[str, torch.Tensor]] = []
    vector_field = torch.zeros(
        (int(tx_positions.shape[0]), int(rx_positions.shape[0]), 3),
        device=device,
        dtype=torch.complex64,
    )
    launch_count = 0
    rx_count = int(rx_positions.shape[0])
    plan = prepare_diffraction_order1_plan(
        metadata=scene.metadata,
        tx_count=int(tx_positions.shape[0]),
        rx_count=rx_count,
    )
    for tx_request in iter_diffraction_tx_requests(
        plan,
        tx_positions=tx_positions,
    ):
        tx_index = tx_request.tx_index
        tx = tx_request.tx
        states = _deterministic_diffraction_states(
            rayd,
            tx,
            tx_power,
            tx_index,
            preserve_imported_edges=plan.preserve_imported_edges,
        )
        visible_plan = plan_tx_visible_diffraction_states(rayd, states, tx)
        state_count = int(visible_plan.edge_index.shape[0])
        if state_count <= 0:
            continue
        # Chunk receivers so the rx x edge-state workspace stays bounded on
        # city-scale scenes (audit P-2); the reflection paths already chunk.
        for rx_request in iter_diffraction_rx_chunk_requests(
            plan,
            state_count=state_count,
        ):
            rx_start = rx_request.rx_start
            rx_end = rx_request.rx_end
            rx_chunk = rx_positions[rx_start:rx_end].contiguous()
            with cuda_profile_range(CudaProfileRange.DIFFRACTION_EXPORTER):
                out = query_diffraction_order1(
                    DiffractionOrder1Query(
                        handle=handle,
                        tx_position=tx.reshape(1, 3).contiguous(),
                        tx_polarization=tx_polarizations[tx_index]
                        .reshape(1, 3)
                        .contiguous(),
                        rx_positions=rx_chunk,
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
                        # ISB boundary taper (ADR-017), D member. 0.0 when the
                        # switch is off keeps the RayD export bit-identical; the
                        # width notches the incident-boundary odd part in the header.
                        isb_taper_width_scale=float(isb_boundary_taper_width),
                    )
                )
            launch_count += 1
            with cuda_profile_range(CudaProfileRange.DIFFRACTION_TOPOLOGY_PACKING):
                compacted = (
                    topology_kernels.deterministic_diffraction_order1_compact(
                        valid=out.valid,
                        rx_id=out.rx_id,
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
                )
            if int(compacted["rx_id"].numel()) == 0:
                continue
            if rx_start > 0:
                compacted["rx_id"] = compacted["rx_id"] + rx_start
            # The path kernel has already evaluated the full UTD vector field.
            # Keep its xyz components until after summing all edges; reducing
            # each path to an equivalent scalar first loses vector coherence.
            receiver_index = compacted["rx_id"].to(dtype=torch.long)
            for axis, real_index in enumerate(("x_re", "y_re", "z_re")):
                imag_index = real_index.replace("_re", "_im")
                vector_field[tx_index, :, axis].index_add_(
                    0,
                    receiver_index,
                    torch.complex(compacted[real_index], compacted[imag_index]),
                )
            field_result = field_kernels.deterministic_diffraction_vector_field(
                x_re=compacted["x_re"],
                x_im=compacted["x_im"],
                y_re=compacted["y_re"],
                y_im=compacted["y_im"],
                z_re=compacted["z_re"],
                z_im=compacted["z_im"],
            )
            field_power = field_result["path_gain"]
            path_field = field_kernels.deterministic_pack_complex(
                field_result["field_real"], field_result["field_imag"]
            )
            field_xyz = torch.stack(
                (
                    torch.complex(compacted["x_re"], compacted["x_im"]),
                    torch.complex(compacted["y_re"], compacted["y_im"]),
                    torch.complex(compacted["z_re"], compacted["z_im"]),
                ),
                dim=1,
            ).contiguous()
            delay = compacted["delay_s"]
            path_length = field_kernels.deterministic_delay_to_path_length(delay)
            empty_i32 = torch.empty((0,), device=device, dtype=torch.int32)
            blocks.append(
                _ensure_topology_fields(
                    topology_kernels.deterministic_topology_base_fields(
                        rx_id=compacted["rx_id"],
                        path_length_m=path_length,
                        delay_s=delay,
                        path_gain=field_power,
                        tx_index=tx_index,
                        component_id=2,
                        depth_source=compacted["depth"],
                        depth_value=0,
                        primitive_source=empty_i32,
                        primitive_value=-1,
                        edge_source=compacted["edge_id"],
                        edge_value=-1,
                    ),
                    interaction_position=compacted["interaction_position"],
                    path_field=path_field,
                    field_xyz=field_xyz,
                )
            )
    return (
        _ensure_topology_fields(concatenate_path_blocks(blocks, device=device)),
        launch_count,
        vector_field,
    )