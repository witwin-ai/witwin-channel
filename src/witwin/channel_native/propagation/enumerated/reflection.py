"""Enumerated first-order reflection topology discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from witwin.channel_native.core.material_runtime import face_material_tensors
from witwin.channel_native.propagation.geometry.kernels import bridge as geometry_bridge
from witwin.channel_native.propagation.geometry.kernels import (
    primitives as geometry_primitives,
)
from witwin.channel_native.propagation.geometry.reevaluate import (
    _cached_coplanar_face_groups,
)
from witwin.channel_native.propagation.topology.concatenate import (
    concatenate_path_blocks,
)
from witwin.channel_native.propagation.topology.discovery.reflection import (
    _MULTIBOUNCE_DISCOVERY_RAYS,
    iter_reflection_order1_epc_requests,
    prepare_reflection_order1_plan,
)
from witwin.channel_native.propagation.topology.export import _ensure_topology_fields
from witwin.channel_native.propagation.topology.kernels import (
    compaction as topology_compaction,
)
from witwin.channel_native.propagation.topology.kernels import (
    construction as topology_construction,
)
from witwin.channel_native.propagation.topology.kernels.sampling import (
    mc_sample_directions,
)

if TYPE_CHECKING:
    from witwin.channel_native.core.scene import Scene


def _reflection_topology_order1(
    scene: Scene,
    compiled: object,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    frequency_hz: float,
) -> tuple[dict[str, torch.Tensor], int]:
    from witwin.channel_native.propagation.fields.kernels import (
        deterministic as field_kernels,
    )

    device = tx_positions.device
    raydn = compiled.raydn
    if not scene.structures or tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return _ensure_topology_fields(
            {
                "valid": torch.empty((0,), device=device, dtype=torch.bool),
                "tx_id": torch.empty((0,), device=device, dtype=torch.int32),
                "rx_id": torch.empty((0,), device=device, dtype=torch.int32),
                "depth": torch.empty((0,), device=device, dtype=torch.int32),
                "component_id": torch.empty((0,), device=device, dtype=torch.int32),
                "primitive_id": torch.empty((0,), device=device, dtype=torch.int32),
                "edge_id": torch.empty((0,), device=device, dtype=torch.int32),
                "path_length_m": torch.empty((0,), device=device, dtype=torch.float32),
                "delay_s": torch.empty((0,), device=device, dtype=torch.float32),
                "path_gain": torch.empty((0,), device=device, dtype=torch.float32),
            }
        ), 0
    if not raydn.available:
        raise RuntimeError(
            "deterministic reflection requires RayDN native scene capability"
        )

    records = raydn.edge_records()
    vertices = records.vertices
    faces = records.faces.contiguous()
    normals = geometry_primitives.deterministic_normalize_vec3(
        records.face_normals.contiguous(), eps=1.0e-6
    )
    if faces.shape[0] == 0:
        return _ensure_topology_fields(
            {
                "valid": torch.empty((0,), device=device, dtype=torch.bool),
                "tx_id": torch.empty((0,), device=device, dtype=torch.int32),
                "rx_id": torch.empty((0,), device=device, dtype=torch.int32),
                "depth": torch.empty((0,), device=device, dtype=torch.int32),
                "component_id": torch.empty((0,), device=device, dtype=torch.int32),
                "primitive_id": torch.empty((0,), device=device, dtype=torch.int32),
                "edge_id": torch.empty((0,), device=device, dtype=torch.int32),
                "path_length_m": torch.empty((0,), device=device, dtype=torch.float32),
                "delay_s": torch.empty((0,), device=device, dtype=torch.float32),
                "path_gain": torch.empty((0,), device=device, dtype=torch.float32),
            }
        ), 0

    tri_a = topology_construction.deterministic_face_anchor_points(
        vertices.contiguous(), faces
    )
    face_eps_r, face_sigma_e, face_mu_r, face_gain, _face_valid = face_material_tensors(
        compiled, device=device
    )
    face_material_id = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int32
    ).contiguous()
    empty_i32 = torch.empty((0,), device=device, dtype=torch.int32)

    # Enumerate one candidate per coplanar face group so that a wall meshed
    # from several coplanar triangles yields exactly one specular path, and
    # every planar facade (not one representative per structure) is covered.
    # The EPC kernel resolves the actual containing triangle per path.
    groups = _cached_coplanar_face_groups(
        raydn,
        tri_a,
        normals,
        compiled.geometry.face_surface_id.to(
            device=device, dtype=torch.long
        ).contiguous(),
    )
    grouped_export = True
    group_count = int(groups["group_count"])
    representative_faces = groups["representative_faces"].contiguous()
    plan = prepare_reflection_order1_plan(
        group_count=group_count,
        representative_faces=representative_faces,
        face_group_id=groups["face_group_id"],
    )

    blocks: list[dict[str, torch.Tensor]] = []
    launch_count = 0
    rx_count = int(rx_positions.shape[0])
    if group_count <= 0 or rx_count <= 0:
        return _ensure_topology_fields(
            concatenate_path_blocks(blocks, device=device)
        ), launch_count

    def trace_group_chains(
        tx: torch.Tensor, *, face_group_id: torch.Tensor, max_depth: int
    ) -> torch.Tensor:
        nonlocal launch_count
        chains = _discovered_group_chains(
            raydn, tx, face_group_id=face_group_id, max_depth=max_depth
        )
        launch_count += 1
        return chains

    for request in iter_reflection_order1_epc_requests(
        plan,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        tri_a=tri_a,
        normals=normals,
        trace_group_chains=trace_group_chains,
    ):
        tx_index = request.tx_index
        tx = request.tx
        epc_inputs = request.epc_inputs
        epc = geometry_bridge.raydn_reflection_epc_paths_forward(
            raydn.require_handle(),
            epc_inputs["tx_batch"],
            epc_inputs["rx_batch"],
            None,
            epc_inputs["sequence_batch"],
            epc_inputs["direct_plane_points"],
            epc_inputs["direct_plane_normals"],
            groups["surface_group_id"],
            groups["surface_group_size"],
            groups["surface_group_members"],
            1,
            1,
        )
        launch_count += 1
        selected = topology_compaction.deterministic_reflection_order1_compact(
            visible=epc[0],
            epc_faces=epc[2],
            epc_hits=epc[4],
            epc_normals=epc[5],
            sequence_batch=epc_inputs["sequence_batch"],
            rx_indices=epc_inputs["rx_indices"],
            tx=tx,
            rx_positions=rx_positions,
            tx_power=tx_power.to(dtype=torch.float32).contiguous(),
            tx_index=tx_index,
            face_eps_r=face_eps_r,
            face_sigma_e=face_sigma_e,
            face_mu_r=face_mu_r,
            face_gain=face_gain,
            face_material_id=face_material_id,
            grouped_export=grouped_export,
        )
        if int(selected["selected_faces"].numel()) == 0:
            continue

        field_result = field_kernels.deterministic_reflection_field(
            tx_position=selected["tx_keep"],
            rx_position=selected["rx_keep"],
            hit_position=selected["selected_points"],
            normal=selected["selected_normals"],
            tx_power=selected["tx_power"],
            eps_r=selected["eps_r"],
            sigma_e=selected["sigma_e"],
            mu_r=selected["mu_r"],
            gain=selected["gain"],
            frequency_hz=frequency_hz,
        )
        path_gain = field_result["path_gain"].to(dtype=torch.float32).contiguous()
        path_field = field_kernels.deterministic_pack_complex(
            field_result["field_real"], field_result["field_imag"]
        )
        path_length = field_result["path_length_m"].to(dtype=torch.float32).contiguous()
        delay = field_result["delay_s"].to(dtype=torch.float32).contiguous()
        blocks.append(
            _ensure_topology_fields(
                topology_construction.deterministic_topology_base_fields(
                    rx_id=selected["selected_rx_id"],
                    path_length_m=path_length.to(dtype=torch.float32).contiguous(),
                    delay_s=delay,
                    path_gain=path_gain.to(dtype=torch.float32).contiguous(),
                    tx_index=tx_index,
                    component_id=1,
                    depth_source=empty_i32,
                    depth_value=1,
                    primitive_source=selected["selected_faces"],
                    primitive_value=-1,
                    edge_source=empty_i32,
                    edge_value=-1,
                ),
                interaction_position=selected["selected_points"],
                interaction_normal=selected["selected_normals"],
                material_id=selected["material_id"],
                path_field=path_field,
            )
        )
    return _ensure_topology_fields(
        concatenate_path_blocks(blocks, device=device)
    ), launch_count


def _discovered_group_chains(
    raydn: object,
    tx: torch.Tensor,
    *,
    face_group_id: torch.Tensor,
    max_depth: int,
    ray_count: int = _MULTIBOUNCE_DISCOVERY_RAYS,
) -> torch.Tensor:
    """Trace specular chains from the transmitter and map them to plane groups.

    Returns an (N, max_depth) long tensor of plane-group ids per bounce with
    -1 past each ray's last hit. Only chains reachable from the transmitter
    can host a valid specular path, so validating the unique chains found here
    replaces the exhaustive plane-sequence product on large scenes.
    """

    device = face_group_id.device
    ray_o = tx.reshape(1, 3).expand(ray_count, 3).contiguous()
    ray_d = mc_sample_directions(ray_count, tx.reshape(1, 3))
    ray_tmax = torch.empty((0,), device=device, dtype=torch.float32)
    out = geometry_bridge.raydn_trace_reflections_forward(
        raydn.require_handle(),
        ray_o,
        ray_d,
        ray_tmax,
        None,
        int(max_depth),
    )
    prim_chain = out[2].to(dtype=torch.long).reshape(ray_count, int(max_depth))
    chains = torch.full_like(prim_chain, -1)
    hit = prim_chain >= 0
    chains[hit] = face_group_id[prim_chain[hit]]
    return chains
