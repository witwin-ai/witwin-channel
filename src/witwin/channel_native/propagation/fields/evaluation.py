from __future__ import annotations

import math
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import torch

from witwin.channel_native.core import ad_geometry
from witwin.channel_native.core.diffraction_geometry import (
    cached_diffraction_edge_geometry as _cached_diffraction_edge_geometry,
    diffraction_edge_geometry as _diffraction_edge_geometry,
)
from witwin.channel_native.core.field_state import (
    receiver_polarizations,
    transmitter_polarizations,
)
from witwin.channel_native.core.kernels.metadata import AdLaunchLedger
from witwin.channel_native.core.material_runtime import (
    face_material_field_bundle,
    face_material_tensors,
)
from witwin.channel_native.core.scene_tensors import (
    LIGHT_SPEED_M_PER_S as _LIGHT_SPEED_M_PER_S,
    _frequency_scalar,
)
from witwin.channel_native.propagation.fields.kernels import (
    autograd as field_autograd,
)
from witwin.channel_native.propagation.fields.kernels import (
    functional as field_functional,
)
from witwin.channel_native.propagation.geometry.kernels import (
    autograd as geometry_autograd,
)
from witwin.channel_native.propagation.geometry.kernels import (
    primitives as geometry_primitives,
)
from witwin.channel_native.propagation.geometry.reevaluate import (
    _geometry_participates_in_ad,
    _opposite_vertex_ids,
    _reflection_geometry_ad,
    _vertices_participate_in_ad,
)
from witwin.channel_native.propagation.topology.kernels import (
    construction as topology_construction,
)
from witwin.channel_native.runtime import autograd_contracts as ops

if TYPE_CHECKING:
    from witwin.channel_native.core.path_topology import TopologyBatch
    from witwin.channel_native.core.scene import Scene


def _rough_reflection_factor(
    compiled: object,
    topology: TopologyBatch,
    rows: torch.Tensor,
    depth_value: int,
    source: torch.Tensor,
    material: dict[str, torch.Tensor],
    positions: torch.Tensor,
    normals: torch.Tensor,
    *,
    frequency_hz: float | torch.Tensor,
    scattering_active: bool,
) -> torch.Tensor | None:
    """Per-row field scale for rough-surface specular reflection rows.

    Kirchhoff-rough faces (scatter_model_id == 1) attenuate their coherent
    specular Jones by C_r = exp(-2*(k0*cos_theta_i*sigma_h)^2) per bounce
    (plan 05 section 6.2, contract section 6). C_r is a real positive scalar,
    so this first-order Python-side attenuation of the native smooth-stack
    field is exact for the field magnitude and phase-neutral. Under AD the
    seam passes the 0-d tensor frequency so k0 (and hence dC_r/df) stays on
    the autograd graph; a float frequency keeps the primal path.

    When the scattering component is active, single-bounce specular rows on a
    surface carrying a realization_coherent phase screen are zeroed: the
    coherent patch integral REPLACES the delta specular for that surface
    (contract 6.7.3, never summed).
    """

    device = topology.valid.device
    # Early exits stay on CPU (the MaterialStore tensors live on the host):
    # smooth scenes must not pay any extra GPU sync in the reflection loop.
    rough_face_any = bool((compiled.materials.scatter_model_id == 1).any())
    realization_ids: list[int] = []
    if scattering_active:
        screens = getattr(compiled.assignments, "structure_phase_screens", {})
        realization_ids = [
            index
            for index, screen in screens.items()
            if getattr(screen, "mode", None) == "realization_coherent"
        ]
    if not rough_face_any and not realization_ids:
        return None

    face_material = material["material_id"].to(dtype=torch.int64)
    rough_face = material["scatter_model_id"] == 1
    realization_face = None
    if realization_ids:
        face_structure = compiled.geometry.face_structure_id.to(
            device=device, dtype=torch.int64
        )
        realization_face = torch.zeros(
            (int(face_structure.numel()),), device=device, dtype=torch.bool
        )
        for index in realization_ids:
            realization_face |= face_structure == index

    face_id = topology.primitive_sequence[rows, :depth_value].to(dtype=torch.int64)
    factor = torch.ones((int(rows.numel()),), device=device, dtype=torch.float32)
    if rough_face_any:
        sigma_face = material["rough_sigma_h_m"][face_material]
        sigma_b = sigma_face[face_id]
        rough_b = rough_face[face_material][face_id]
        prev = torch.cat(
            (source[rows].unsqueeze(1), positions[:, : depth_value - 1]), dim=1
        )
        seg = positions - prev
        seg_dir = seg / torch.linalg.vector_norm(seg, dim=-1, keepdim=True).clamp_min(
            1.0e-9
        )
        cos_b = (seg_dir * normals).sum(-1).abs()
        k0 = 2.0 * math.pi * frequency_hz / _LIGHT_SPEED_M_PER_S
        attenuation = torch.exp(-2.0 * (k0 * cos_b * sigma_b).square())
        c_r = torch.where(rough_b, attenuation, torch.ones_like(attenuation))
        factor = c_r.prod(dim=1)
    if realization_face is not None and depth_value == 1:
        replaced = realization_face[face_id[:, 0]]
        factor = torch.where(replaced, torch.zeros_like(factor), factor)
    return factor


def _evaluate_shared_fields(
    scene: Scene,
    compiled: object,
    topology: TopologyBatch,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    components: frozenset[str] | set[str] = frozenset(),
    ad_mode: str = "none",
    frequency_value: float | None = None,
) -> TopologyBatch:
    """Evaluate selected canonical rows with the shared complex3 ABI.

    ``ad_mode == "none"`` keeps the exact primal behavior (plain facades,
    float frequency, no autograd graph). ``jvp``/``vjp`` route LoS,
    reflection and transmission through the differentiable field Functions:
    material gathers stay on the torch graph and the frequency is forwarded
    as a 0-d tensor when the scene stores one, alongside one precomputed
    host scalar (``frequency_value``) threaded into every field Function so
    a tensor-frequency solve pays a single device-to-host frequency read
    (audit M3). Hit geometry stays detached under the fixed-topology
    contract unless a geometry leaf participates in AD, in which case it
    comes from RayD's fixed-winner geometry companions (see
    ``_reflection_geometry_ad``). The rough-surface C_r attenuation is pure
    torch and receives the same live frequency so dC_r/df is kept.
    """

    count = int(topology.valid.shape[0])
    if count == 0:
        return topology
    ad_enabled = ad_mode != "none"
    # Plan 07 AD-4 metadata: one ledger entry per registered differentiable
    # Function, tape bytes mirroring each Function's save_for_backward set.
    ledger = AdLaunchLedger() if ad_enabled else None
    if ad_enabled:
        frequency = (
            scene.frequency
            if isinstance(scene.frequency, torch.Tensor)
            else float(scene.frequency)
        )
        if frequency_value is None:
            frequency_value = ops._ad_frequency_value(frequency)
        frequency_value = float(frequency_value)
        los_field_op = partial(
            field_autograd.field_free_space_ad,
            frequency=frequency,
            frequency_value=frequency_value,
        )
        reflection_field_op = partial(
            field_autograd.field_reflection_sequence_ad,
            frequency=frequency,
            frequency_value=frequency_value,
        )
        transmission_field_op = partial(
            field_autograd.field_transmission_sequence_ad,
            frequency=frequency,
            frequency_value=frequency_value,
        )
    else:
        frequency = (
            _frequency_scalar(scene)
            if frequency_value is None
            else float(frequency_value)
        )
        los_field_op = partial(field_functional.field_free_space, frequency_hz=frequency)
        reflection_field_op = partial(
            field_functional.field_reflection_sequence, frequency_hz=frequency
        )
        transmission_field_op = partial(
            field_functional.field_transmission_sequence, frequency_hz=frequency
        )
    device = topology.valid.device
    # AD-2: when a geometry leaf is on the graph, the endpoints come from the
    # live scene tensors and the hit geometry comes from RayD's fixed-winner
    # chain companions (a native re-launch of the EPC discovery on the frozen
    # winner sequence plus its backward/jvp CUDA kernels), so the field
    # kernels see geometry with a gradient without any torch-side re-solve.
    # Otherwise the detached native discovery output is used unchanged (AD-1).
    geometry_ad = ad_enabled and _geometry_participates_in_ad(scene)
    if geometry_ad:
        vertices = ad_geometry.scene_vertex_table(scene, compiled)
        tx_positions = ad_geometry.transmitter_positions_ad(
            scene, tx_positions, device=device
        )
        rx_positions = ad_geometry.receiver_positions_ad(
            scene, rx_positions, device=device
        )
    tx_id = topology.tx_id.to(dtype=torch.int64)
    rx_id = topology.rx_id.to(dtype=torch.int64)
    source = tx_positions[tx_id].contiguous()
    target = rx_positions[rx_id].contiguous()
    source_power = tx_power[tx_id].to(dtype=torch.float32).contiguous()
    tx_pol = transmitter_polarizations(scene, device=device)[tx_id].contiguous()
    rx_pol = receiver_polarizations(scene, device=device)[rx_id].contiguous()

    field_xyz = topology.field_xyz.clone()
    coefficient = topology.coefficient.clone()
    path_field = topology.path_field.clone()
    path_gain = topology.path_gain.clone()
    path_length = topology.path_length_m.clone()
    delay = topology.delay_s.clone()
    direction = topology.field_direction.clone()
    launch_count = topology.launch_count

    los_rows = torch.nonzero(topology.component_id == 0, as_tuple=False).reshape(-1)
    if int(los_rows.shape[0]) > 0:
        los_args = (
            source[los_rows].contiguous(),
            target[los_rows].contiguous(),
            source_power[los_rows].contiguous(),
            tx_pol[los_rows].contiguous(),
            rx_pol[los_rows].contiguous(),
        )
        if ledger is not None:
            ledger.add(*los_args)
        evaluated = los_field_op(*los_args)
        field_xyz.index_copy_(0, los_rows, evaluated["field_vector"])
        coefficient.index_copy_(0, los_rows, evaluated["coefficient"])
        path_field.index_copy_(0, los_rows, evaluated["path_field"])
        path_gain.index_copy_(0, los_rows, evaluated["path_gain"])
        path_length.index_copy_(0, los_rows, evaluated["path_length_m"])
        delay.index_copy_(0, los_rows, evaluated["delay_s"])
        direction.index_copy_(0, los_rows, evaluated["direction"])
        launch_count += 1

    material: dict[str, torch.Tensor] | None = None
    for depth_value in range(1, 6):
        rows = torch.nonzero(
            (topology.component_id == 1) & (topology.depth == depth_value),
            as_tuple=False,
        ).reshape(-1)
        if int(rows.shape[0]) == 0:
            continue
        if material is None:
            material = face_material_field_bundle(compiled, device=device)
        face_id = topology.primitive_sequence[rows, :depth_value].to(dtype=torch.int64)
        if geometry_ad:
            positions, normals = _reflection_geometry_ad(
                compiled,
                vertices,
                source[rows].contiguous(),
                target[rows].contiguous(),
                face_id,
                depth_value,
            )
        else:
            positions = topology.interaction_positions[
                rows, :depth_value
            ].contiguous()
            normals = topology.interaction_normals[rows, :depth_value].contiguous()
        reflection_args = (
            source[rows].contiguous(),
            target[rows].contiguous(),
            positions,
            normals,
            source_power[rows].contiguous(),
            tx_pol[rows].contiguous(),
            rx_pol[rows].contiguous(),
            material["eps_r"][face_id].contiguous(),
            material["sigma_e"][face_id].contiguous(),
            material["mu_r"][face_id].contiguous(),
            material["gain"][face_id].contiguous(),
            material["thickness"][face_id].contiguous(),
        )
        if ledger is not None:
            if geometry_ad:
                # The fixed-winner EPC re-solve registers its own companion.
                ledger.add(vertices, source[rows], target[rows], face_id)
            ledger.add(*reflection_args)
        evaluated = reflection_field_op(*reflection_args)
        # Rough-surface coherent attenuation / realization delta replacement
        # (see _rough_reflection_factor). Applied Python-side on the native
        # smooth-stack result; C_r is real, so field magnitude scaling is
        # exact and phase-neutral.
        rough_factor = _rough_reflection_factor(
            compiled,
            topology,
            rows,
            depth_value,
            source,
            material,
            positions,
            normals,
            frequency_hz=frequency,
            scattering_active="scattering" in components,
        )
        if rough_factor is not None:
            field_scale = rough_factor.to(torch.float32)
            evaluated = dict(evaluated)
            evaluated["field_vector"] = evaluated["field_vector"] * field_scale[:, None]
            evaluated["coefficient"] = evaluated["coefficient"] * field_scale
            evaluated["path_field"] = evaluated["path_field"] * field_scale
            evaluated["path_gain"] = evaluated["path_gain"] * field_scale.square()
        field_xyz.index_copy_(0, rows, evaluated["field_vector"])
        coefficient.index_copy_(0, rows, evaluated["coefficient"])
        path_field.index_copy_(0, rows, evaluated["path_field"])
        path_gain.index_copy_(0, rows, evaluated["path_gain"])
        path_length.index_copy_(0, rows, evaluated["path_length_m"])
        delay.index_copy_(0, rows, evaluated["delay_s"])
        direction.index_copy_(0, rows, evaluated["direction"])
        launch_count += 1

    transmission_rows = torch.nonzero(
        topology.component_id == 5, as_tuple=False
    ).reshape(-1)
    if int(transmission_rows.shape[0]) > 0:
        if material is None:
            material = face_material_field_bundle(compiled, device=device)
        rows = transmission_rows
        width = int(topology.interaction_positions.shape[1])
        slots = torch.arange(width, device=device).reshape(1, -1)
        event_valid = (
            slots < topology.depth[rows].to(dtype=torch.int64).reshape(-1, 1)
        ).contiguous()
        positions = topology.interaction_positions[rows].contiguous()
        if geometry_ad:
            # The transmission kernel reads only the straight source-target
            # ray and the wall normals, so the crossing points carry exactly
            # zero gradient and stay the detached discovery values. The
            # normals come from RayD's differentiable face-normal table
            # gathered by the frozen winner prim; the kernel orients them
            # against the incident ray internally, so the table's sign
            # convention does not matter. Invalid slots gather face 0 but are
            # skipped by the kernel, so they receive a zero cotangent.
            records = compiled.raydn.edge_records()
            face_normal_table = geometry_autograd.raydn_face_normals_ad(
                compiled.raydn.require_handle(),
                vertices,
                records.face_normals.contiguous(),
            )
            prim_sequence = topology.primitive_sequence[rows].to(dtype=torch.int64)
            normals = face_normal_table[prim_sequence.clamp_min(0)]
        else:
            normals = topology.interaction_normals[rows].contiguous()
        transmission_args = (
            source[rows].contiguous(),
            target[rows].contiguous(),
            positions,
            normals,
            topology.material_sequence[rows].contiguous(),
            event_valid,
            source_power[rows].contiguous(),
            tx_pol[rows].contiguous(),
            rx_pol[rows].contiguous(),
            material["layer_offset"],
            material["layer_count"],
            material["layer_thickness_m"],
            material["layer_eps_r"],
            material["layer_sigma_e"],
            material["layer_mu_r"],
        )
        if ledger is not None:
            if geometry_ad:
                # The differentiable face-normal table registers a companion.
                ledger.add(vertices)
            ledger.add(*transmission_args)
        evaluated = transmission_field_op(*transmission_args)
        field_xyz.index_copy_(0, rows, evaluated["field_vector"])
        coefficient.index_copy_(0, rows, evaluated["coefficient"])
        path_field.index_copy_(0, rows, evaluated["path_field"])
        path_gain.index_copy_(0, rows, evaluated["path_gain"])
        path_length.index_copy_(0, rows, evaluated["path_length_m"])
        delay.index_copy_(0, rows, evaluated["delay_s"])
        direction.index_copy_(0, rows, evaluated["direction"])
        launch_count += 1

    diffraction_rows = torch.nonzero(
        topology.component_id == 2, as_tuple=False
    ).reshape(-1)
    if int(diffraction_rows.shape[0]) > 0:
        if ad_enabled:
            # Plan 07 AD-4: RayD's order-1 path export is detached, so the
            # wedge field is re-evaluated from the frozen topology (edge id,
            # edge tables, wedge-face materials) with the differentiable
            # kernel, and the projection runs through its own Function so the
            # arrival-direction chain stays on the graph. Forward parity with
            # topology.field_xyz is pinned by tests/ad.
            if material is None:
                material = face_material_field_bundle(compiled, device=device)
            preserve_imported_edges = bool(
                isinstance(scene.metadata.get("mitsuba", {}), dict)
                and scene.metadata.get("mitsuba", {}).get("merge_shapes", False)
            )
            edge_geometry = (
                _diffraction_edge_geometry(compiled.raydn.edge_records())
                if preserve_imported_edges
                else _cached_diffraction_edge_geometry(compiled.raydn)
            )
            edge_id = topology.edge_id[diffraction_rows].to(dtype=torch.int64)
            face0 = edge_geometry[8].to(dtype=torch.int64)[edge_id]
            face1 = edge_geometry[9].to(dtype=torch.int64)[edge_id]
            face_count = int(material["eps_r"].shape[0])
            # RayD's export treats a face as absent when its id is out of
            # range or its material entry is not exported (face_material
            # validity is discrete winner state).
            table_valid = face_material_tensors(compiled, device=device)[4]
            face0_c = face0.clamp(min=0, max=max(face_count - 1, 0))
            face1_c = face1.clamp(min=0, max=max(face_count - 1, 0))
            valid0 = (face0 >= 0) & (face0 < face_count) & table_valid[face0_c]
            valid1 = (face1 >= 0) & (face1 < face_count) & table_valid[face1_c]
            wedge_vertices = None
            if geometry_ad and _vertices_participate_in_ad(scene):
                # Mesh-vertex x diffraction (plan 07 section 9.3): hand the
                # winner edge vertices to the wedge kernel, which rebuilds
                # the edge tables from them on the dual row. The integer
                # winner extraction below runs on detached tables; only the
                # vertex gathers touch the live table.
                records = compiled.raydn.edge_records()
                edge_v0_ids = records.edge_v0.to(dtype=torch.int64)[edge_id]
                edge_v1_ids = records.edge_v1.to(dtype=torch.int64)[edge_id]
                faces_table = records.faces.to(dtype=torch.int64)
                tri0 = faces_table[face0.clamp(min=0)]
                tri1 = faces_table[face1.clamp(min=0)]
                opp0_ids = _opposite_vertex_ids(tri0, edge_v0_ids, edge_v1_ids)
                opp1_ids = _opposite_vertex_ids(tri1, edge_v0_ids, edge_v1_ids)
                edge_boundary = (face1 < 0).contiguous()
                wedge_vertices = (
                    vertices[edge_v0_ids].contiguous(),
                    vertices[edge_v1_ids].contiguous(),
                    vertices[opp0_ids].contiguous(),
                    vertices[opp1_ids].contiguous(),
                    edge_boundary,
                )
            wedge_args = (
                source[diffraction_rows].contiguous(),
                target[diffraction_rows].contiguous(),
                edge_geometry[1][edge_id].contiguous(),
                edge_geometry[2][edge_id].contiguous(),
                edge_geometry[4][edge_id].contiguous(),
                edge_geometry[5][edge_id].contiguous(),
                edge_geometry[6][edge_id].contiguous(),
                edge_geometry[7][edge_id].contiguous(),
                edge_geometry[10][edge_id].contiguous(),
                valid0.contiguous(),
                material["eps_r"][face0_c].contiguous(),
                material["sigma_e"][face0_c].contiguous(),
                material["mu_r"][face0_c].contiguous(),
                material["gain"][face0_c].contiguous(),
                valid1.contiguous(),
                material["eps_r"][face1_c].contiguous(),
                material["sigma_e"][face1_c].contiguous(),
                material["mu_r"][face1_c].contiguous(),
                material["gain"][face1_c].contiguous(),
                source_power[diffraction_rows].contiguous(),
            )
            if ledger is not None:
                ledger.add(
                    *wedge_args,
                    *(wedge_vertices if wedge_vertices is not None else ()),
                )
            evaluated = field_autograd.field_diffraction_wedge_ad(
                *wedge_args,
                frequency=frequency,
                frequency_value=frequency_value,
                vertices=wedge_vertices,
            )
            powered_xyz = evaluated["field_vector"]
            arrival = evaluated["direction"]
            if ledger is not None:
                ledger.add(powered_xyz, arrival, rx_pol[diffraction_rows])
            projected = field_autograd.field_project_complex3_ad(
                powered_xyz,
                arrival,
                rx_pol[diffraction_rows].contiguous(),
            )
        else:
            arrival = geometry_primitives.deterministic_normalize_vec3(
                (
                    target[diffraction_rows]
                    - topology.interaction_positions[diffraction_rows, 0]
                ).contiguous(),
                eps=1.0e-6,
            )
            powered_xyz = topology.field_xyz[diffraction_rows].contiguous()
            projected = field_functional.field_project_complex3(
                powered_xyz,
                arrival,
                rx_pol[diffraction_rows].contiguous(),
            )
        amplitude = source_power[diffraction_rows].clamp_min(1.0e-30).sqrt()
        field_xyz.index_copy_(0, diffraction_rows, powered_xyz / amplitude[:, None])
        path_field.index_copy_(0, diffraction_rows, projected["coefficient"])
        coefficient.index_copy_(
            0, diffraction_rows, projected["coefficient"] / amplitude
        )
        path_gain.index_copy_(0, diffraction_rows, projected["path_gain"])
        direction.index_copy_(0, diffraction_rows, arrival)
        launch_count += 1

    coupled_rows = torch.nonzero(
        (topology.component_id == 3) | (topology.component_id == 4),
        as_tuple=False,
    ).reshape(-1)
    if int(coupled_rows.shape[0]) > 0:
        if material is None:
            material = face_material_field_bundle(compiled, device=device)
        raydn = compiled.raydn
        records = raydn.edge_records()
        preserve_imported_edges = bool(
            isinstance(scene.metadata.get("mitsuba", {}), dict)
            and scene.metadata.get("mitsuba", {}).get("merge_shapes", False)
        )
        edge_geometry = (
            _diffraction_edge_geometry(records)
            if preserve_imported_edges
            else _cached_diffraction_edge_geometry(raydn)
        )
        edge_n0 = edge_geometry[6]
        edge_n1 = edge_geometry[7]
        edge_face0 = edge_geometry[8].to(dtype=torch.int64)
        edge_face1 = edge_geometry[9].to(dtype=torch.int64)
        edge_exterior = edge_geometry[10]
        coupled_geometry_ad = ad_enabled and geometry_ad
        if coupled_geometry_ad and _vertices_participate_in_ad(scene):
            # The coupled chain consumes the wall plane and the edge tables
            # through the stationary re-solve and the coupled field kernel,
            # whose adjoints take them as frozen winners; a vertex gradient
            # through the coupled rows would therefore be silently
            # incomplete. Fail loudly instead (plan 07 section 9.4).
            raise NotImplementedError(
                "coupled reflection-diffraction paths do not support mesh "
                "vertex gradients: the coupled stationary re-solve and the "
                "coupled field adjoints treat the wall plane and the edge "
                "tables as frozen winners, so d(coupled)/d(vertices) would "
                "be silently missing. Drop the vertices requires_grad/"
                "tangent or disable coupled_paths."
            )
        if coupled_geometry_ad:
            tri_a = topology_construction.deterministic_face_anchor_points(
                records.vertices.contiguous(), records.faces.contiguous()
            )
            normals_table = geometry_primitives.deterministic_normalize_vec3(
                records.face_normals.contiguous(), eps=1.0e-6
            )
        for component_id, reverse_order in ((3, False), (4, True)):
            rows = torch.nonzero(
                topology.component_id == component_id, as_tuple=False
            ).reshape(-1)
            if int(rows.shape[0]) == 0:
                continue
            edge_id = topology.edge_id[rows].to(dtype=torch.int64)
            reflection_face = topology.primitive_id[rows].to(dtype=torch.int64)
            face0 = edge_face0[edge_id]
            raw_face1 = edge_face1[edge_id]
            face1 = torch.where(raw_face1 >= 0, raw_face1, face0)
            reflection_slot = 1 if reverse_order else 0
            edge_slot = 0 if reverse_order else 1
            reflection_position = topology.interaction_positions[
                rows, reflection_slot
            ].contiguous()
            edge_position = topology.interaction_positions[
                rows, edge_slot
            ].contiguous()
            if coupled_geometry_ad:
                # Plan 07 AD-4: the coupled interaction points move with the
                # endpoints (Fresnel angles are not stationary), so re-solve
                # the frozen winner's stationary geometry differentiably (the
                # same image-source math as the discovery prepare kernel) and
                # feed the live points into the field kernel. D->R is the
                # reciprocal problem with the endpoints exchanged.
                epc_source = target[rows] if reverse_order else source[rows]
                epc_receiver = source[rows] if reverse_order else target[rows]
                resolved = field_autograd.coupled_rd_prepare_ad(
                    epc_source.contiguous(),
                    epc_receiver.contiguous(),
                    tri_a[reflection_face].contiguous(),
                    normals_table[reflection_face].contiguous(),
                    edge_geometry[1][edge_id].contiguous(),
                    edge_geometry[2][edge_id].contiguous(),
                    edge_geometry[4][edge_id].contiguous(),
                    edge_geometry[5][edge_id].contiguous(),
                )
                if not bool(resolved["active"].all()):
                    raise RuntimeError(
                        "fixed-winner coupled stationary re-solve no longer "
                        "reproduces the discovered coupled paths; the winner "
                        "topology moved under the current scene tensors"
                    )
                for name, reference in (
                    ("edge_point", edge_position),
                    ("reflection_point", reflection_position),
                ):
                    if not bool(
                        torch.isclose(
                            resolved[name].detach(), reference, atol=1.0e-3
                        ).all()
                    ):
                        raise RuntimeError(
                            "fixed-winner coupled stationary re-solve moved the "
                            f"{name} away from the discovered topology"
                        )
                edge_position = resolved["edge_point"]
                reflection_position = resolved["reflection_point"]

            def material_tuple(face: torch.Tensor) -> tuple[torch.Tensor, ...]:
                return tuple(
                    material[name][face].contiguous()
                    for name in ("eps_r", "sigma_e", "mu_r", "gain", "thickness")
                )

            coupled_field_op = (
                partial(
                    field_autograd.field_coupled_rd_ad,
                    frequency=frequency,
                    frequency_value=frequency_value,
                )
                if ad_enabled
                else partial(field_functional.field_coupled_rd, frequency_hz=frequency)
            )
            reflection_materials = material_tuple(reflection_face)
            wedge_materials0 = material_tuple(face0)
            wedge_materials1 = material_tuple(face1)
            coupled_args = (
                source[rows].contiguous(),
                target[rows].contiguous(),
                reflection_position,
                topology.interaction_normals[
                    rows, reflection_slot
                ].contiguous(),
                edge_position,
                edge_geometry[2][edge_id].contiguous(),
                edge_n0[edge_id].contiguous(),
                edge_n1[edge_id].contiguous(),
                edge_exterior[edge_id].contiguous(),
                source_power[rows].contiguous(),
                tx_pol[rows].contiguous(),
                rx_pol[rows].contiguous(),
            )
            if ledger is not None:
                if coupled_geometry_ad:
                    # Fixed-winner coupled stationary re-solve companion.
                    ledger.add(source[rows], target[rows], reflection_position)
                ledger.add(
                    *coupled_args,
                    *reflection_materials,
                    *wedge_materials0,
                    *wedge_materials1,
                )
            evaluated = coupled_field_op(
                *coupled_args,
                reflection_materials,
                wedge_materials0,
                wedge_materials1,
                reverse=reverse_order,
            )
            field_xyz.index_copy_(0, rows, evaluated["field_vector"])
            coefficient.index_copy_(0, rows, evaluated["coefficient"])
            path_field.index_copy_(0, rows, evaluated["path_field"])
            path_gain.index_copy_(0, rows, evaluated["path_gain"])
            direction.index_copy_(0, rows, evaluated["direction"])
            launch_count += 1

    return replace(
        topology,
        path_length_m=path_length,
        delay_s=delay,
        path_gain=path_gain,
        path_field=path_field,
        field_xyz=field_xyz,
        coefficient=coefficient,
        field_direction=direction,
        launch_count=launch_count,
        ad_companion_launches=(
            topology.ad_companion_launches + ledger.launches
            if ledger is not None
            else topology.ad_companion_launches
        ),
        ad_tape_bytes=(
            topology.ad_tape_bytes + ledger.tape_bytes
            if ledger is not None
            else topology.ad_tape_bytes
        ),
    )
