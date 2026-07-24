"""Enumerated first-order diffraction topology discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from witwin.channel.core.field_state import transmitter_polarizations
from witwin.channel.materials.encoding import face_material_tensors
from witwin.channel.scene.tensors import (
    LIGHT_SPEED_M_PER_S as _LIGHT_SPEED_M_PER_S,
)
from witwin.channel.propagation.geometry.diffraction import (
    DiffractionOrder1Query,
    plan_tx_visible_diffraction_states,
    query_diffraction_edges,
    query_diffraction_order1,
)
from witwin.channel.propagation.topology.concatenate import (
    concatenate_path_blocks,
)
from witwin.channel.propagation.topology.discovery.diffraction import (
    iter_diffraction_rx_chunk_requests,
    iter_diffraction_tx_requests,
    prepare_diffraction_order1_plan,
)
from witwin.channel.propagation.topology.export import _ensure_topology_fields
from witwin.channel.propagation.topology.kernels import (
    compaction as topology_compaction,
)
from witwin.channel.propagation.topology.kernels import (
    construction as topology_construction,
)
from witwin.channel.propagation.topology.kernels import (
    primitives as topology_primitives,
)
from witwin.channel.runtime.profiling import (
    CudaProfileRange,
    cuda_profile_range,
    profiled_cuda_range,
)

if TYPE_CHECKING:
    from witwin.channel.scene.endpoints import SolverScene as Scene


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
    return topology_primitives.deterministic_diffraction_state_pack(
        topology_primitives.mc_selected_edge_indices(edges.selected),
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
    from witwin.channel.propagation.fields.kernels import (
        deterministic as field_kernels,
    )

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
    tx_polarizations = transmitter_polarizations(scene, device=device)
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
                    topology_compaction.deterministic_diffraction_order1_compact(
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
                    topology_construction.deterministic_topology_base_fields(
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
