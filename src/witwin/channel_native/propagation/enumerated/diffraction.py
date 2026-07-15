"""Enumerated first-order diffraction topology discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from witwin.channel_native.core.diffraction_geometry import (
    cached_diffraction_edge_geometry as _cached_diffraction_edge_geometry,
    diffraction_edge_geometry as _diffraction_edge_geometry,
)
from witwin.channel_native.core.material_runtime import face_material_tensors
from witwin.channel_native.core.scene_tensors import (
    LIGHT_SPEED_M_PER_S as _LIGHT_SPEED_M_PER_S,
)
from witwin.channel_native.propagation.geometry.kernels import bridge as geometry_bridge
from witwin.channel_native.propagation.topology.concatenate import (
    concatenate_path_blocks,
)
from witwin.channel_native.propagation.topology.discovery.reflection import (
    _MULTIBOUNCE_PAIR_CHUNK_SIZE,
)
from witwin.channel_native.propagation.topology.export import _ensure_topology_fields
from witwin.channel_native.propagation.topology.kernels import (
    compaction as topology_compaction,
)
from witwin.channel_native.propagation.topology.kernels import (
    construction as topology_construction,
)
from witwin.channel_native.propagation.topology.kernels import (
    primitives as topology_primitives,
)

if TYPE_CHECKING:
    from witwin.channel_native.core.scene import Scene


def _deterministic_diffraction_states(
    raydn: object,
    tx: torch.Tensor,
    tx_power: torch.Tensor,
    tx_index: int,
    *,
    preserve_imported_edges: bool = False,
) -> tuple[torch.Tensor, ...]:
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
    ) = (
        _diffraction_edge_geometry(raydn.edge_records())
        if preserve_imported_edges
        else _cached_diffraction_edge_geometry(raydn)
    )
    return topology_primitives.deterministic_diffraction_state_pack(
        topology_primitives.mc_selected_edge_indices(selected),
        edge_pos,
        edge_dir,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
        tx,
        tx_power,
        int(tx_index),
    )


_DIFFRACTION_PREFILTER_EDGE_FRACTIONS = (0.02, 1.0 / 3.0, 2.0 / 3.0, 0.98)


def _tx_visible_diffraction_states(
    raydn: object,
    states: tuple[torch.Tensor, ...],
    tx: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Drop edge states that are occluded from the transmitter.

    The UTD kernel checks visibility at the per-receiver stationary point, so
    a state is only culled when the transmitter cannot see the edge at any of
    several sample points along it. This shrinks the rx x state workspace and
    pair launches on city-scale scenes (mirrors the original tx_first
    pruning) while keeping states whose midpoint happens to be occluded.
    """

    state_count = int(states[0].shape[0])
    if state_count <= 0:
        return states
    edge_anchor = states[1]
    edge_dir = states[2]
    line_min = states[3]
    line_max = states[4]
    starts = tx.reshape(1, 3).expand(state_count, 3).contiguous()
    visible = torch.zeros((state_count,), device=edge_anchor.device, dtype=torch.bool)
    for fraction in _DIFFRACTION_PREFILTER_EDGE_FRACTIONS:
        t = line_min + fraction * (line_max - line_min)
        point = (edge_anchor + t.unsqueeze(1) * edge_dir).contiguous()
        visible |= geometry_bridge.raydn_visibility_forward(
            raydn.require_handle(), starts, point, None
        )[0]
    if bool(visible.all()):
        return states
    return tuple(
        tensor[visible] if tensor.shape[:1] == (state_count,) else tensor
        for tensor in states
    )


def _diffraction_topology_order1(
    scene: Scene,
    compiled: object,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    frequency_hz: float,
) -> tuple[dict[str, torch.Tensor], int, torch.Tensor]:
    from witwin.channel_native.propagation.fields.kernels import (
        deterministic as field_kernels,
    )

    device = tx_positions.device
    raydn = compiled.raydn
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
    if not raydn.available:
        raise RuntimeError(
            "deterministic diffraction requires RayDN native scene capability"
        )

    face_eps_r, face_sigma_e, face_mu_r, material_gain, material_valid = (
        face_material_tensors(compiled, device=device)
    )
    wavelength = _LIGHT_SPEED_M_PER_S / float(frequency_hz)
    handle = raydn.require_handle()
    blocks: list[dict[str, torch.Tensor]] = []
    vector_field = torch.zeros(
        (int(tx_positions.shape[0]), int(rx_positions.shape[0]), 3),
        device=device,
        dtype=torch.complex64,
    )
    launch_count = 0
    rx_count = int(rx_positions.shape[0])
    mitsuba_metadata = scene.metadata.get("mitsuba", {})
    # Channel's merge_shapes import keeps the selected boundary-edge table
    # intact.  The synthetic-scene path instead merges coincident structure
    # boundaries into one physical wedge (the single-wedge test contract).
    preserve_imported_edges = isinstance(mitsuba_metadata, dict) and bool(
        mitsuba_metadata.get("merge_shapes", False)
    )
    for tx_index, tx in enumerate(tx_positions):
        states = _deterministic_diffraction_states(
            raydn,
            tx,
            tx_power,
            tx_index,
            preserve_imported_edges=preserve_imported_edges,
        )
        states = _tx_visible_diffraction_states(raydn, states, tx)
        state_count = int(states[0].shape[0])
        if state_count <= 0:
            continue
        # Chunk receivers so the rx x edge-state workspace stays bounded on
        # city-scale scenes (audit P-2); the reflection paths already chunk.
        rx_chunk_size = max(1, _MULTIBOUNCE_PAIR_CHUNK_SIZE // state_count)
        for rx_start in range(0, rx_count, rx_chunk_size):
            rx_end = min(rx_start + rx_chunk_size, rx_count)
            rx_chunk = rx_positions[rx_start:rx_end].contiguous()
            capacity = int(rx_chunk.shape[0]) * state_count
            out = geometry_bridge.raydn_diffraction_paths_order1_forward(
                handle,
                tx.reshape(1, 3).contiguous(),
                rx_chunk,
                None,
                *states,
                face_eps_r,
                face_sigma_e,
                face_mu_r,
                material_gain,
                material_valid,
                state_count,
                capacity,
                float(wavelength),
            )
            launch_count += 1
            compacted = topology_compaction.deterministic_diffraction_order1_compact(
                valid=out[1],
                rx_id=out[3],
                depth=out[4],
                edge_id=out[5],
                delay_s=out[8],
                x_re=out[9],
                x_im=out[10],
                y_re=out[11],
                y_im=out[12],
                z_re=out[13],
                z_im=out[14],
                interaction_position=out[15],
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
