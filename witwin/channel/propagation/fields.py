# Copyright Xingyu Chen.
# Implements fields.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import torch

from witwin.channel.kernels.fields import (
    _FieldCoupledChainAdFunction,
    _FieldDiffractionWedgeAdFunction,
    _FieldFreeSpaceAdFunction,
    _FieldProjectComplex3AdFunction,
    _FieldReflectionSequenceAdFunction,
    _FieldTransmissionSequenceAdFunction,
    field_coupled_chain,
    field_coupled_chain_ad,
    field_diffraction_wedge,
    field_diffraction_wedge_ad,
    field_free_space,
    field_free_space_ad,
    field_free_space_backward,
    field_free_space_jvp,
    field_project_complex3,
    field_project_complex3_ad,
    field_reflection_sequence,
    field_reflection_sequence_ad,
    field_reflection_sequence_backward,
    field_reflection_sequence_jvp,
    field_transmission_sequence,
    field_transmission_sequence_ad,
    field_transmission_sequence_backward,
    field_transmission_sequence_jvp,
)
from witwin.channel.runtime import AdLaunchLedger, _ad_frequency_value
from witwin.channel.kernels import fields as field_kernels
from witwin.channel.propagation.rows import (
    COMPONENT_ROW_GROUPS,
    EvaluatedPaths,
    PathFields,
    PathGeometry,
    PathRowGroups,
    PathTopology,
)
from witwin.channel.scene import endpoints as scene_endpoints
from witwin.channel.propagation.geometry import (
    cached_diffraction_edge_geometry as _cached_diffraction_edge_geometry,
    diffraction_edge_geometry as _diffraction_edge_geometry,
)
from witwin.channel.scene.endpoints import (
    receiver_polarization_rows,
    transmitter_polarizations_f32,
)
from witwin.channel.materials import (
    face_geometry_id_table,
    face_material_field_bundle,
    face_material_tensors,
    has_multilayer_material,
)
from witwin.channel.scene.compiler import (
    _frequency_scalar,
)
from witwin.channel.scene.resources import diffraction_edge_vertex_ids
from witwin.channel.kernels import geometry as geometry_kernels
from witwin.channel.propagation.geometry import (
    apply_los_taper,
    los_clearance_factor,
    occluder_boxes,
)
from witwin.channel.propagation.geometry import (
    _geometry_participates_in_ad,
    _opposite_vertex_ids,
    _participates_in_ad,
    _reflection_geometry_ad,
    _vertices_participate_in_ad,
)
from witwin.channel.propagation.topology import (
    PathExecutionStats,
)

if TYPE_CHECKING:
    from witwin.channel.scene.compiler import CompiledScene
    from witwin.channel.scene.endpoints import SolverScene as Scene


_LOS_GROUP = 0
_DIFFRACTION_GROUP = 2
# Single-diffraction chains: the diffraction last (with the interior-diffraction chains, whose
# component id is _CHAIN_INTERIOR_BASE + slot) and the diffraction first.
_CHAIN_LAST_GROUP = 3
_CHAIN_FIRST_GROUP = 4
_CHAIN_INTERIOR_BASE = 8
_TRANSMISSION_GROUP = 5
_DD_GROUP = 7


def _take(values: torch.Tensor, rows: torch.Tensor | None) -> torch.Tensor:
    """Gather one row group; ``rows is None`` means the group is every row in order."""

    return values.contiguous() if rows is None else values[rows].contiguous()


class _FieldOutputs:
    """The seven per-row field outputs, cloned from the inputs only on a partial write."""

    __slots__ = ("values", "_owned")

    def __init__(self, values: dict[str, torch.Tensor], *, owned: bool = False) -> None:
        self.values = dict(values)
        self._owned: set[str] = set(values) if owned else set()

    def write(self, name: str, rows: torch.Tensor | None, value: torch.Tensor) -> None:
        if rows is None:
            self.values[name] = value
            self._owned.add(name)
            return
        if name not in self._owned:
            self.values[name] = self.values[name].clone()
            self._owned.add(name)
        self.values[name].index_copy_(0, rows, value)

    def write_evaluated(
        self, rows: torch.Tensor | None, evaluated: dict[str, torch.Tensor], *,
        with_length: bool = True,
    ) -> None:
        self.write("field_xyz", rows, evaluated["field_vector"])
        self.write("coefficient", rows, evaluated["coefficient"])
        self.write("path_field", rows, evaluated["path_field"])
        self.write("path_gain", rows, evaluated["path_gain"])
        if with_length:
            self.write("path_length", rows, evaluated["path_length_m"])
            self.write("delay", rows, evaluated["delay_s"])
        self.write("direction", rows, evaluated["direction"])


def _chain_rows(groups: PathRowGroups) -> torch.Tensor | None:
    """Rows of every single-diffraction chain group, or None when they are every row."""

    present = [group for group in (_CHAIN_LAST_GROUP, _CHAIN_FIRST_GROUP) if groups.count(group) > 0]
    if len(present) == 1:
        return groups.rows(present[0])
    return torch.cat([groups.rows(group) for group in present])


def _chain_diffraction_slot(component_id: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """The diffraction slot of each chain row, as its component id encodes it."""

    return torch.where(
        component_id == _CHAIN_LAST_GROUP,
        depth - 1,
        torch.where(component_id == _CHAIN_FIRST_GROUP, torch.zeros_like(depth), component_id - _CHAIN_INTERIOR_BASE),
    )


def _evaluate_chain_rows(
    compiled: CompiledScene, topology: PathTopology, geometry: PathGeometry, source: torch.Tensor,
    target: torch.Tensor, source_power: torch.Tensor, tx_pol: torch.Tensor, rx_pol: torch.Tensor,
    material: dict[str, torch.Tensor], edge_geometry: tuple[torch.Tensor, ...],
    edge_n0: torch.Tensor, edge_n1: torch.Tensor, edge_exterior: torch.Tensor,
    edge_face0: torch.Tensor, edge_face1: torch.Tensor, ad_enabled: bool,
    frequency: float | torch.Tensor, frequency_value: float | None, ledger: AdLaunchLedger | None,
    groups: PathRowGroups, outputs: _FieldOutputs, launch_count: int,
) -> int:
    """Evaluate the interaction-chain rows (reflections, one diffraction, reflections) in one launch.

    Each row's slot materials are gathered per face: the reflection slots from the row's
    sequence, the two wedge faces from its edge. The discovered interaction points are frozen;
    under AD the endpoints move them through the native re-solve of the frozen winner.
    """

    if not groups.has_rows(_CHAIN_LAST_GROUP, _CHAIN_FIRST_GROUP):
        return launch_count
    rows = _chain_rows(groups)
    component_id = _take(topology.component_id, rows).to(dtype=torch.int32)
    depth = _take(topology.depth, rows).to(dtype=torch.int32)
    sequence = _take(topology.primitive_sequence, rows).to(dtype=torch.int64)
    positions = _take(geometry.interaction_positions, rows)
    normals = _take(geometry.interaction_normals, rows)
    width = int(sequence.shape[1])
    edge_id = _take(topology.edge_id, rows).to(dtype=torch.int64)
    slot = _chain_diffraction_slot(component_id, depth).to(dtype=torch.int64)
    columns = torch.arange(width, device=sequence.device).reshape(1, -1)
    reflection = (columns != slot.reshape(-1, 1)) & (columns < depth.reshape(-1, 1).to(dtype=torch.int64))
    face0 = edge_face0[edge_id]
    raw_face1 = edge_face1[edge_id]
    # Boundary edges have no second face; reuse face0.
    face1 = torch.where(raw_face1 >= 0, raw_face1, face0)
    slot_faces = torch.where(reflection, sequence, face0.reshape(-1, 1))
    face_table = torch.cat((slot_faces, face0.reshape(-1, 1), face1.reshape(-1, 1)), dim=1)
    chain_material = tuple(
        material[name][face_table].contiguous() for name in ("eps_r", "sigma_e", "mu_r", "gain", "thickness")
    )
    # Edge-segment bounds relative to the discovered edge (Keller) point along the normalized
    # edge axis, so the native stationary machinery can truncate and corner-mend the
    # diffraction leg. Frozen: chain rows carry no edge-geometry gradient.
    edge_point = positions.gather(1, slot.reshape(-1, 1, 1).expand(-1, 1, 3)).reshape(-1, 3)
    edge_ref = edge_geometry[1][edge_id]
    edge_axis = edge_geometry[2][edge_id]
    edge_axis = edge_axis / edge_axis.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    t_keller = ((edge_point.detach() - edge_ref) * edge_axis).sum(dim=-1)
    edge_line_min = (edge_geometry[4][edge_id] - t_keller).contiguous()
    edge_line_max = (edge_geometry[5][edge_id] - t_keller).contiguous()
    chain_args = (
        _take(source, rows),
        _take(target, rows),
        positions.detach().contiguous(),
        normals.detach().contiguous(),
        component_id.contiguous(),
        depth.contiguous(),
        edge_geometry[2][edge_id].contiguous(),
        edge_n0[edge_id].contiguous(),
        edge_n1[edge_id].contiguous(),
        edge_exterior[edge_id].contiguous(),
        edge_line_min,
        edge_line_max,
        _take(source_power, rows),
        _take(tx_pol, rows),
        _take(rx_pol, rows),
    )
    if ledger is not None:
        ledger.add(*chain_args, *chain_material)
    if ad_enabled:
        if has_multilayer_material(compiled):
            raise NotImplementedError(
                "coupled reflection-diffraction derivatives with a multilayer reflecting material are "
                "not supported; disable coupled_paths or use ad_mode='none'"
            )
        evaluated = field_kernels.field_coupled_chain_ad(
            *chain_args, chain_material, frequency=frequency, frequency_value=frequency_value
        )
    else:
        layers = (
            field_kernels.reflection_layers(material, face_table, multilayer=True)
            if has_multilayer_material(compiled)
            else None
        )
        evaluated = field_kernels.field_coupled_chain(
            *chain_args, chain_material, frequency_hz=frequency, layers=layers
        )
    outputs.write_evaluated(rows, evaluated, with_length=False)
    return launch_count + 1


def _evaluate_coupled_dd_rows(
    topology: PathTopology, geometry: PathGeometry, source: torch.Tensor, target: torch.Tensor,
    source_power: torch.Tensor, tx_pol: torch.Tensor, rx_pol: torch.Tensor,
    material: dict[str, torch.Tensor], edge_geometry: tuple[torch.Tensor, ...],
    edge_n0: torch.Tensor, edge_n1: torch.Tensor, edge_exterior: torch.Tensor,
    edge_face0: torch.Tensor, edge_face1: torch.Tensor, ad_enabled: bool,
    frequency: float | torch.Tensor, frequency_value: float | None, ledger: AdLaunchLedger | None,
    groups: PathRowGroups, outputs: _FieldOutputs, launch_count: int,
) -> int:
    """Evaluate the coupled double-diffraction rows (cid 7, coupled double diffraction).

    Pure lift of the cid-7 branch out of ``_evaluate_coupled_fields``:
    identical gather math, tensor ordering, row identity, aliasing and error
    behavior. Both interactions are diffraction (type 2), so both
    primitive_sequence slots carry edge ids; interaction_positions carry the
    frozen Fermat seeds [Q1, Q2]. Results land in the same coherent coupled
    slot as cid 3/4 via in-place ``index_copy_`` on the shared output tensors,
    so their storage/aliasing is preserved. Returns the updated launch_count.
    """

    if groups.count(_DD_GROUP) > 0:
        dd_rows = groups.rows(_DD_GROUP)
        dd_sequence = _take(topology.primitive_sequence, dd_rows)
        dd_positions = _take(geometry.interaction_positions, dd_rows)
        edge1_id = dd_sequence[:, 0].to(dtype=torch.int64)
        edge2_id = dd_sequence[:, 1].to(dtype=torch.int64)
        q1 = dd_positions[:, 0].contiguous()
        q2 = dd_positions[:, 1].contiguous()

        def _dd_line_bounds(
            edge_id: torch.Tensor, keller_point: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # /coupled double diffraction: edge-segment bounds relative to the passed Keller
            # point along the normalized edge axis, so the native stationary
            # machinery truncates and corner-mends each DD leg. Frozen
            # (detached): DD rows carry no edge-geometry gradient; tx/rx
            # gradient flows through source/target inside the native
            # re-anchoring (same lookup path as the cid 3/4 single edge).
            edge_ref = edge_geometry[1][edge_id]
            edge_axis = edge_geometry[2][edge_id]
            edge_axis = edge_axis / edge_axis.norm(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-12)
            t_keller = (
                (keller_point.detach() - edge_ref) * edge_axis
            ).sum(dim=-1)
            line_min = (edge_geometry[4][edge_id] - t_keller).contiguous()
            line_max = (edge_geometry[5][edge_id] - t_keller).contiguous()
            return line_min, line_max

        edge1_line_min, edge1_line_max = _dd_line_bounds(edge1_id, q1)
        edge2_line_min, edge2_line_max = _dd_line_bounds(edge2_id, q2)

        def _dd_wedge_faces(edge_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            face0 = edge_face0[edge_id]
            raw_face1 = edge_face1[edge_id]
            # Boundary edges have no second face; reuse face0 like cid 3/4.
            face1 = torch.where(raw_face1 >= 0, raw_face1, face0)
            return face0, face1

        edge1_face0, edge1_face1 = _dd_wedge_faces(edge1_id)
        edge2_face0, edge2_face1 = _dd_wedge_faces(edge2_id)

        def _dd_material_tuple(face: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return tuple(
                material[name][face].contiguous()
                for name in ("eps_r", "sigma_e", "mu_r", "gain", "thickness")
            )

        wedge1_material0 = _dd_material_tuple(edge1_face0)
        wedge1_material1 = _dd_material_tuple(edge1_face1)
        wedge2_material0 = _dd_material_tuple(edge2_face0)
        wedge2_material1 = _dd_material_tuple(edge2_face1)

        coupled_dd_field_op = (
            partial(
                field_kernels.field_coupled_dd_ad,
                frequency=frequency,
                frequency_value=frequency_value,
            )
            if ad_enabled
            else partial(
                field_kernels.field_coupled_dd, frequency_hz=frequency
            )
        )
        dd_args = (
            _take(source, dd_rows),
            _take(target, dd_rows),
            q1,
            edge_geometry[2][edge1_id].contiguous(),
            edge_n0[edge1_id].contiguous(),
            edge_n1[edge1_id].contiguous(),
            edge_exterior[edge1_id].contiguous(),
            q2,
            edge_geometry[2][edge2_id].contiguous(),
            edge_n0[edge2_id].contiguous(),
            edge_n1[edge2_id].contiguous(),
            edge_exterior[edge2_id].contiguous(),
            _take(source_power, dd_rows),
            _take(tx_pol, dd_rows),
            _take(rx_pol, dd_rows),
        )
        if ledger is not None:
            ledger.add(
                *dd_args,
                *wedge1_material0,
                *wedge1_material1,
                *wedge2_material0,
                *wedge2_material1,
                edge1_line_min,
                edge1_line_max,
                edge2_line_min,
                edge2_line_max,
            )
        evaluated = coupled_dd_field_op(
            *dd_args,
            wedge1_material0,
            wedge1_material1,
            wedge2_material0,
            wedge2_material1,
            edge1_line_min,
            edge1_line_max,
            edge2_line_min,
            edge2_line_max,
        )
        outputs.write_evaluated(dd_rows, evaluated, with_length=False)
        launch_count += 1
    return launch_count


def _los_taper_frequency(
    frequency_value: float | None, frequency: float | torch.Tensor,
) -> float | torch.Tensor:
    """Host frequency for the boundary taper LoS taper clearance kernel.

    The LoS taper only ever runs on the ad_mode="none" primal (taper + AD is
    rejected upstream until the C1 clearance companion lands, the boundary taper 3),
    so ``frequency_value`` is the host float scalar when set; fall back to the
    resolved frequency otherwise.
    """

    return float(frequency_value) if frequency_value is not None else frequency


def _rough_face_tables(
    compiled: CompiledScene, material: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Per-face rough-surface sigma and Kirchhoff-rough flag, or None on a smooth scene.

    The gate reads the host material store, so a smooth scene pays no GPU work.
    """

    if not bool((compiled.materials.scatter_model_id == 1).any()):
        return None
    face_material = material["material_id"].to(dtype=torch.int64)
    sigma_face = material["rough_sigma_h_m"][face_material].to(torch.float32).contiguous()
    rough_face = (material["scatter_model_id"] == 1)[face_material].contiguous()
    return sigma_face, rough_face


def _realization_structures(compiled: CompiledScene, *, scattering_active: bool) -> list[int]:
    """Structures whose realization_coherent phase screen replaces their specular reflection.

    When the scattering component is active, the coherent patch integral of such a surface
    REPLACES its single-bounce delta specular (contract 6.7.3, never summed).
    """

    if not scattering_active:
        return []
    screens = getattr(compiled.assignments, "structure_phase_screens", {})
    return [
        index
        for index, screen in screens.items()
        if getattr(screen, "mode", None) == "realization_coherent"
    ]


def _realization_face_mask(
    compiled: CompiledScene, realization_ids: list[int], device: torch.device,
) -> torch.Tensor:
    """Per-face mask of the faces of the given realization structures."""

    face_structure = face_geometry_id_table(compiled, "face_structure_id", device=device)
    realization_face = torch.zeros(
        (int(face_structure.numel()),), device=device, dtype=torch.bool
    )
    for structure_index in realization_ids:
        structure_id = int(compiled.assignments.structure_id[structure_index])
        realization_face |= face_structure == structure_id
    return realization_face


def _rough_scale_inputs(
    compiled: CompiledScene, face_id: torch.Tensor, depth_value: int,
    material: dict[str, torch.Tensor], *, scattering_active: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Per-bounce inputs of the native rough-reflection C_r op for one reflection depth.

    Kirchhoff-rough faces (scatter_model_id == 1) attenuate their coherent
    specular Jones by C_r = exp(-2*(k0*cos_theta_i*sigma_h)^2) per bounce. The native op
    ``field_rough_reflection_scale_ad`` owns the factor math and its application; this helper
    gathers the per-bounce ``sigma_b`` / ``rough_b`` and the realization ``replaced`` mask.
    Returns ``None`` for smooth scenes so the reflection loop pays no extra GPU work.
    """

    device = face_id.device
    rough_tables = _rough_face_tables(compiled, material)
    realization_ids = _realization_structures(compiled, scattering_active=scattering_active)
    if rough_tables is None and not realization_ids:
        return None
    row_count = int(face_id.shape[0])
    if rough_tables is not None:
        sigma_face, rough_face = rough_tables
        sigma_b = sigma_face[face_id].contiguous()
        rough_b = rough_face[face_id].contiguous()
    else:
        sigma_b = torch.zeros(
            (row_count, depth_value), device=device, dtype=torch.float32
        )
        rough_b = torch.zeros(
            (row_count, depth_value), device=device, dtype=torch.bool
        )
    replaced = torch.zeros((row_count,), device=device, dtype=torch.bool)
    if realization_ids and depth_value == 1:
        realization_face = _realization_face_mask(compiled, realization_ids, device)
        replaced = realization_face[face_id[:, 0]].contiguous()
    return sigma_b, rough_b, replaced


def _canonical_face_tables(
    compiled: CompiledScene, groups: PathRowGroups, components: frozenset[str] | set[str],
    device: torch.device,
) -> field_kernels.CanonicalFaceTables:
    """Per-face material, layer and rough-surface tables of the fused canonical launch."""

    if not groups.reflection_depths():
        return field_kernels.CanonicalFaceTables.empty(device)
    material = face_material_field_bundle(compiled, device=device)
    rough_tables = _rough_face_tables(compiled, material)
    realization_ids = _realization_structures(
        compiled, scattering_active="scattering" in components
    )
    return field_kernels.CanonicalFaceTables(
        eps_r=material["eps_r"],
        sigma_e=material["sigma_e"],
        mu_r=material["mu_r"],
        gain=material["gain"],
        thickness=material["thickness"],
        material=material["material_id"],
        layers=tuple(
            material[name]
            for name in (
                "layer_offset", "layer_count", "layer_thickness_m", "layer_eps_r",
                "layer_sigma_e", "layer_mu_r",
            )
        ),
        multilayer=has_multilayer_material(compiled),
        rough_sigma=None if rough_tables is None else rough_tables[0],
        rough=None if rough_tables is None else rough_tables[1],
        replaced=(
            _realization_face_mask(compiled, realization_ids, device) if realization_ids else None
        ),
    )


def _evaluate_los_fields(
    groups: PathRowGroups, source: torch.Tensor, target: torch.Tensor, source_power: torch.Tensor,
    tx_pol: torch.Tensor, rx_pol: torch.Tensor,
    los_field_op: Callable[..., dict[str, torch.Tensor]], ledger: AdLaunchLedger | None,
    outputs: _FieldOutputs, launch_count: int, compiled: CompiledScene, frequency_hz: float,
    isb_boundary_taper_width: float,
) -> int:
    if groups.count(_LOS_GROUP) > 0:
        los_rows = groups.rows(_LOS_GROUP)
        los_args = (
            _take(source, los_rows),
            _take(target, los_rows),
            _take(source_power, los_rows),
            _take(tx_pol, los_rows),
            _take(rx_pol, los_rows),
        )
        if ledger is not None:
            ledger.add(*los_args)
        evaluated = los_field_op(*los_args)
        # ISB boundary taper, LoS member. When on (width > 0), scale the
        # LoS field bundle by the C1 clearance factor tau re-derived on these rows
        # from the same native kernel that set the survival gate in discovery, so
        # the tapered rows carry a smoothly-decaying amplitude across the shadow
        # boundary. Off (width == 0, the default) leaves the bundle untouched.
        if isb_boundary_taper_width > 0.0:
            taper_boxes = occluder_boxes(compiled)
            if taper_boxes is not None:
                box_min, box_max = taper_boxes
                tau = los_clearance_factor(
                    los_args[0],
                    los_args[1],
                    box_min,
                    box_max,
                    frequency_hz=frequency_hz,
                    width=isb_boundary_taper_width,
                )
                scaled = apply_los_taper(
                    evaluated["field_vector"],
                    evaluated["coefficient"],
                    evaluated["path_field"],
                    evaluated["path_gain"],
                    tau,
                )
                evaluated = {**evaluated, **scaled}
                launch_count += 2
        outputs.write_evaluated(los_rows, evaluated)
        launch_count += 1
    return launch_count


def _evaluate_reflection_fields(
    compiled: CompiledScene, topology: PathTopology, geometry: PathGeometry, groups: PathRowGroups,
    source: torch.Tensor, target: torch.Tensor, source_power: torch.Tensor, tx_pol: torch.Tensor,
    rx_pol: torch.Tensor, components: frozenset[str] | set[str], device: torch.device,
    frequency: float | torch.Tensor, geometry_ad: bool, vertices: torch.Tensor | None,
    reflection_field_op: Callable[..., dict[str, torch.Tensor]], ledger: AdLaunchLedger | None,
    material: dict[str, torch.Tensor] | None, frequency_value: float | None, outputs: _FieldOutputs,
    launch_count: int,
) -> tuple[dict[str, torch.Tensor] | None, int]:
    for depth_value in groups.reflection_depths():
        rows = groups.rows(COMPONENT_ROW_GROUPS + depth_value)
        if material is None:
            material = face_material_field_bundle(compiled, device=device)
        face_id = _take(topology.primitive_sequence, rows)[:, :depth_value].to(dtype=torch.int64)
        row_source = _take(source, rows)
        row_target = _take(target, rows)
        if geometry_ad:
            positions, normals = _reflection_geometry_ad(
                compiled,
                vertices,
                row_source,
                row_target,
                face_id,
                depth_value,
            )
        else:
            positions = _take(geometry.interaction_positions, rows)[:, :depth_value].contiguous()
            normals = _take(geometry.interaction_normals, rows)[:, :depth_value].contiguous()
        reflection_args = (
            row_source,
            row_target,
            positions,
            normals,
            _take(source_power, rows),
            _take(tx_pol, rows),
            _take(rx_pol, rows),
            material["eps_r"][face_id].contiguous(),
            material["sigma_e"][face_id].contiguous(),
            material["mu_r"][face_id].contiguous(),
            material["gain"][face_id].contiguous(),
            material["thickness"][face_id].contiguous(),
        )
        if ledger is not None:
            if geometry_ad:
                # The fixed-winner EPC re-solve registers its own companion.
                ledger.add(vertices, row_source, row_target, face_id)
            ledger.add(*reflection_args)
        layers = (
            field_kernels.reflection_layers(material, face_id, multilayer=True)
            if has_multilayer_material(compiled)
            else None
        )
        evaluated = reflection_field_op(*reflection_args, layers=layers)
        # Rough-surface coherent attenuation / realization delta replacement
        # (rough-surface scattering). The native field_rough_reflection_scale kernel owns
        # the C_r factor and its application onto the four field outputs; C_r
        # is real, so magnitude scaling is exact and phase-neutral. The
        # host-side material gating stays here (no added GPU sync for smooth
        # scenes).
        rough_inputs = _rough_scale_inputs(
            compiled,
            face_id,
            depth_value,
            material,
            scattering_active="scattering" in components,
        )
        if rough_inputs is not None:
            sigma_b, rough_b, replaced = rough_inputs
            scale_args = (
                evaluated["field_vector"],
                evaluated["coefficient"],
                evaluated["path_field"],
                evaluated["path_gain"],
                positions,
                normals,
                row_source,
                sigma_b,
                rough_b,
                replaced,
            )
            if ledger is not None:
                ledger.add(*scale_args)
            scaled = field_kernels.field_rough_reflection_scale_ad(
                *scale_args,
                frequency=frequency,
                frequency_value=frequency_value,
            )
            evaluated = {
                **evaluated,
                "field_vector": scaled["field_vector"],
                "coefficient": scaled["coefficient"],
                "path_field": scaled["path_field"],
                "path_gain": scaled["path_gain"],
            }
            launch_count += 1
        outputs.write_evaluated(rows, evaluated)
        launch_count += 1
    return material, launch_count


def _evaluate_transmission_fields(
    compiled: CompiledScene, topology: PathTopology, geometry: PathGeometry, source: torch.Tensor,
    target: torch.Tensor, source_power: torch.Tensor, tx_pol: torch.Tensor, rx_pol: torch.Tensor,
    device: torch.device, geometry_ad: bool, vertices: torch.Tensor | None,
    transmission_field_op: Callable[..., dict[str, torch.Tensor]], ledger: AdLaunchLedger | None,
    material: dict[str, torch.Tensor] | None, groups: PathRowGroups, outputs: _FieldOutputs,
    launch_count: int,
) -> tuple[dict[str, torch.Tensor] | None, int]:
    if groups.count(_TRANSMISSION_GROUP) > 0:
        if material is None:
            material = face_material_field_bundle(compiled, device=device)
        rows = groups.rows(_TRANSMISSION_GROUP)
        width = int(geometry.interaction_positions.shape[1])
        slots = torch.arange(width, device=device).reshape(1, -1)
        event_valid = (
            slots < _take(topology.depth, rows).to(dtype=torch.int64).reshape(-1, 1)
        ).contiguous()
        positions = _take(geometry.interaction_positions, rows)
        if geometry_ad:
            # The transmission kernel reads only the straight source-target
            # ray and the wall normals, so the crossing points carry exactly
            # zero gradient and stay the detached discovery values. The
            # normals come from RayD's differentiable face-normal table
            # gathered by the frozen winner prim; the kernel orients them
            # against the incident ray internally, so the table's sign
            # convention does not matter. Invalid slots gather face 0 but are
            # skipped by the kernel, so they receive a zero cotangent.
            records = compiled.rayd.edge_records()
            face_normal_table = geometry_kernels.rayd_face_normals_ad(
                compiled.rayd.require_resource(),
                vertices,
                records.face_normals.contiguous(),
            )
            prim_sequence = _take(topology.primitive_sequence, rows).to(dtype=torch.int64)
            normals = face_normal_table[prim_sequence.clamp_min(0)]
        else:
            normals = _take(geometry.interaction_normals, rows)
        transmission_args = (
            _take(topology.valid, rows),
            _take(source, rows),
            _take(target, rows),
            positions,
            normals,
            _take(topology.material_sequence, rows),
            event_valid,
            _take(source_power, rows),
            _take(tx_pol, rows),
            _take(rx_pol, rows),
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
        outputs.write_evaluated(rows, evaluated)
        launch_count += 1
    return material, launch_count


def _evaluate_diffraction_fields(
    scene: Scene, compiled: CompiledScene, topology: PathTopology, source: torch.Tensor,
    target: torch.Tensor, source_power: torch.Tensor, rx_pol: torch.Tensor, device: torch.device,
    frequency: float | torch.Tensor, frequency_value: float | None, geometry_ad: bool,
    vertices: torch.Tensor | None, ledger: AdLaunchLedger | None,
    material: dict[str, torch.Tensor] | None, groups: PathRowGroups, outputs: _FieldOutputs,
    launch_count: int,
) -> tuple[dict[str, torch.Tensor] | None, int]:
    if groups.count(_DIFFRACTION_GROUP) > 0:
        diffraction_rows = groups.rows(_DIFFRACTION_GROUP)
        # diffraction AD: RayD's order-1 path export is detached, so the
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
            _diffraction_edge_geometry(compiled.rayd.edge_records())
            if preserve_imported_edges
            else _cached_diffraction_edge_geometry(compiled.rayd)
        )
        edge_id = _take(topology.edge_id, diffraction_rows).to(dtype=torch.int64)
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
            # Mesh-vertex x diffraction: hand the
            # winner edge vertices to the wedge kernel, which rebuilds
            # the edge tables from them on the dual row. The integer
            # winner extraction below runs on detached tables; only the
            # vertex gathers touch the live table.
            records = compiled.rayd.edge_records()
            record_v0 = records.edge_v0.to(dtype=torch.int64)[edge_id]
            record_v1 = records.edge_v1.to(dtype=torch.int64)[edge_id]
            first_ids, last_ids = diffraction_edge_vertex_ids(compiled.rayd, edge_geometry[2])
            edge_v0_ids = first_ids.to(dtype=torch.int64)[edge_id]
            edge_v1_ids = last_ids.to(dtype=torch.int64)[edge_id]
            faces_table = records.faces.to(dtype=torch.int64)
            tri0 = faces_table[face0.clamp(min=0)]
            tri1 = faces_table[face1.clamp(min=0)]
            opp0_ids = _opposite_vertex_ids(tri0, record_v0, record_v1)
            opp1_ids = _opposite_vertex_ids(tri1, record_v0, record_v1)
            # A half-plane edge carries its one sheet face on both sides.
            edge_boundary = ((face1 < 0) | (face1 == face0)).contiguous()
            wedge_vertices = (
                # The caller passes the vertex table whenever geometry AD is on.
                vertices[edge_v0_ids].contiguous(),  # type: ignore[index]
                vertices[edge_v1_ids].contiguous(),  # type: ignore[index]
                vertices[opp0_ids].contiguous(),  # type: ignore[index]
                vertices[opp1_ids].contiguous(),  # type: ignore[index]
                edge_boundary,
            )
        wedge_args = (
            _take(topology.valid, diffraction_rows),
            _take(source, diffraction_rows),
            _take(target, diffraction_rows),
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
            _take(source_power, diffraction_rows),
        )
        if ledger is not None:
            ledger.add(
                *wedge_args,
                *(wedge_vertices if wedge_vertices is not None else ()),
            )
        evaluated = field_kernels.field_diffraction_wedge_ad(
            *wedge_args,
            frequency=frequency,
            frequency_value=frequency_value,
            vertices=wedge_vertices,
            face_layers=field_kernels.wedge_face_layers(material, face0_c, face1_c),
        )
        powered_xyz = evaluated["field_vector"]
        arrival = evaluated["direction"]
        if ledger is not None:
            ledger.add(powered_xyz, arrival, _take(rx_pol, diffraction_rows))
        projected = field_kernels.field_project_complex3_ad(
            powered_xyz,
            arrival,
            _take(rx_pol, diffraction_rows),
        )
        amplitude = _take(source_power, diffraction_rows).clamp_min(1.0e-30).sqrt()
        outputs.write("field_xyz", diffraction_rows, powered_xyz / amplitude[:, None])
        outputs.write("path_field", diffraction_rows, projected["coefficient"])
        outputs.write("coefficient", diffraction_rows, projected["coefficient"] / amplitude)
        outputs.write("path_gain", diffraction_rows, projected["path_gain"])
        outputs.write("direction", diffraction_rows, arrival)
        launch_count += 1
    return material, launch_count


def _evaluate_coupled_fields(
    scene: Scene, compiled: CompiledScene, topology: PathTopology, geometry: PathGeometry,
    source: torch.Tensor, target: torch.Tensor, source_power: torch.Tensor, tx_pol: torch.Tensor,
    rx_pol: torch.Tensor, device: torch.device, frequency: float | torch.Tensor,
    frequency_value: float | None, ad_enabled: bool, geometry_ad: bool,
    ledger: AdLaunchLedger | None, material: dict[str, torch.Tensor] | None, groups: PathRowGroups,
    outputs: _FieldOutputs, launch_count: int,
) -> tuple[dict[str, torch.Tensor] | None, int]:
    if groups.has_rows(_CHAIN_LAST_GROUP, _CHAIN_FIRST_GROUP, _DD_GROUP):
        if material is None:
            material = face_material_field_bundle(compiled, device=device)
        rayd = compiled.rayd
        records = rayd.edge_records()
        preserve_imported_edges = bool(
            isinstance(scene.metadata.get("mitsuba", {}), dict)
            and scene.metadata.get("mitsuba", {}).get("merge_shapes", False)
        )
        edge_geometry = (
            _diffraction_edge_geometry(records)
            if preserve_imported_edges
            else _cached_diffraction_edge_geometry(rayd)
        )
        edge_n0 = edge_geometry[6]
        edge_n1 = edge_geometry[7]
        edge_face0 = edge_geometry[8].to(dtype=torch.int64)
        edge_face1 = edge_geometry[9].to(dtype=torch.int64)
        edge_exterior = edge_geometry[10]
        if ad_enabled and geometry_ad and _vertices_participate_in_ad(scene):
            # Chain and double-diffraction rows take the reflection planes and the edge tables as
            # frozen winners, so a vertex gradient through them would be silently incomplete.
            raise NotImplementedError(
                "coupled reflection-diffraction and double-diffraction paths do "
                "not support mesh vertex gradients: the coupled stationary "
                "re-solve and the coupled field adjoints treat the wall plane "
                "and the edge tables as frozen winners, so d(coupled)/d(vertices)"
                " would be silently missing. Drop the vertices requires_grad/"
                "tangent or disable coupled_paths."
            )
        launch_count = _evaluate_chain_rows(
            compiled,
            topology,
            geometry,
            source,
            target,
            source_power,
            tx_pol,
            rx_pol,
            material,
            edge_geometry,
            edge_n0,
            edge_n1,
            edge_exterior,
            edge_face0,
            edge_face1,
            ad_enabled,
            frequency,
            frequency_value,
            ledger,
            groups,
            outputs,
            launch_count,
        )
        launch_count = _evaluate_coupled_dd_rows(
            topology,
            geometry,
            source,
            target,
            source_power,
            tx_pol,
            rx_pol,
            material,
            edge_geometry,
            edge_n0,
            edge_n1,
            edge_exterior,
            edge_face0,
            edge_face1,
            ad_enabled,
            frequency,
            frequency_value,
            ledger,
            groups,
            outputs,
            launch_count,
        )
    return material, launch_count


def _row_endpoints(
    topology: PathTopology, tx_positions: torch.Tensor, tx_power: torch.Tensor,
    rx_positions: torch.Tensor, tx_pol_batch: torch.Tensor, rx_pol_batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row source, target, source power and polarizations gathered by tx/rx id."""

    tx_id = topology.tx_id.to(dtype=torch.int64)
    rx_id = topology.rx_id.to(dtype=torch.int64)
    return (
        tx_positions[tx_id].contiguous(),
        rx_positions[rx_id].contiguous(),
        tx_power[tx_id].to(dtype=torch.float32).contiguous(),
        tx_pol_batch[tx_id].contiguous(),
        rx_pol_batch[rx_id].contiguous(),
    )


def _evaluate_differentiable_rows(
    scene: Scene, compiled: CompiledScene, topology: PathTopology, geometry: PathGeometry,
    input_fields: PathFields, row_groups: PathRowGroups, endpoint_rows: tuple[torch.Tensor, ...],
    components: frozenset[str] | set[str], device: torch.device, frequency: float | torch.Tensor,
    frequency_value: float | None, geometry_ad: bool, vertices: torch.Tensor | None,
    los_field_op: Callable[..., dict[str, torch.Tensor]],
    reflection_field_op: Callable[..., dict[str, torch.Tensor]],
    transmission_field_op: Callable[..., dict[str, torch.Tensor]], ledger: AdLaunchLedger | None,
    outputs: _FieldOutputs, launch_count: int, isb_boundary_taper_width: float,
) -> int:
    """Evaluate every row group through its differentiable field Function."""

    source, target, source_power, tx_pol, rx_pol = endpoint_rows
    launch_count = _evaluate_los_fields(
        row_groups, source, target, source_power, tx_pol, rx_pol, los_field_op, ledger, outputs,
        launch_count, compiled, _los_taper_frequency(frequency_value, frequency),
        isb_boundary_taper_width,
    )
    material, launch_count = _evaluate_reflection_fields(
        compiled, topology, geometry, row_groups, source, target, source_power, tx_pol, rx_pol,
        components, device, frequency, geometry_ad, vertices, reflection_field_op, ledger, None,
        frequency_value, outputs, launch_count,
    )
    material, launch_count = _evaluate_transmission_fields(
        compiled, topology, geometry, source, target, source_power, tx_pol, rx_pol, device,
        geometry_ad, vertices, transmission_field_op, ledger, material, row_groups, outputs,
        launch_count,
    )
    material, launch_count = _evaluate_diffraction_fields(
        scene, compiled, topology, source, target, source_power, rx_pol, device, frequency,
        frequency_value, geometry_ad, vertices, ledger, material, row_groups, outputs,
        launch_count,
    )
    _, launch_count = _evaluate_coupled_fields(
        scene, compiled, topology, geometry, source, target, source_power, tx_pol, rx_pol, device,
        frequency, frequency_value, True, geometry_ad, ledger, material, row_groups, outputs,
        launch_count,
    )
    return launch_count


def _evaluate_canonical_rows(
    scene: Scene, compiled: CompiledScene, topology: PathTopology, geometry: PathGeometry,
    input_columns: dict[str, torch.Tensor], row_groups: PathRowGroups, tx_positions: torch.Tensor,
    tx_power: torch.Tensor, rx_positions: torch.Tensor, tx_pol_batch: torch.Tensor,
    rx_pol_batch: torch.Tensor, components: frozenset[str] | set[str], device: torch.device,
    frequency: float, frequency_value: float | None,
    los_field_op: Callable[..., dict[str, torch.Tensor]],
    transmission_field_op: Callable[..., dict[str, torch.Tensor]], launch_count: int,
    isb_boundary_taper_width: float,
) -> tuple[_FieldOutputs, int]:
    """Primal field stage: one canonical-row launch, then the groups other native units own."""

    # One native launch evaluates every line-of-sight, reflection and diffraction row in
    # place; only the boundary-taper LoS member and the operators owned by other native
    # units (transmission, coupled chains) still run on their row groups afterwards.
    taper_los = (
        isb_boundary_taper_width > 0.0
        and row_groups.count(_LOS_GROUP) > 0
        and occluder_boxes(compiled) is not None
    )
    canonical = field_kernels.field_canonical_rows(
        order=row_groups.order,
        component_id=topology.component_id.contiguous(),
        depth=topology.depth.contiguous(),
        tx_id=topology.tx_id.contiguous(),
        rx_id=topology.rx_id.contiguous(),
        primitive_sequence=topology.primitive_sequence.contiguous(),
        tx_positions=tx_positions.contiguous(),
        tx_power=tx_power.to(dtype=torch.float32).contiguous(),
        tx_polarization=tx_pol_batch.contiguous(),
        rx_positions=rx_positions.contiguous(),
        rx_polarization=rx_pol_batch.contiguous(),
        interaction_positions=geometry.interaction_positions.contiguous(),
        interaction_normals=geometry.interaction_normals.contiguous(),
        fields=input_columns,
        faces=_canonical_face_tables(compiled, row_groups, components, device),
        evaluate_los=not taper_los,
        frequency_hz=frequency,
    )
    launch_count += 1
    outputs = _FieldOutputs(canonical, owned=True)
    if taper_los or row_groups.has_rows(
        _TRANSMISSION_GROUP, _CHAIN_LAST_GROUP, _CHAIN_FIRST_GROUP, _DD_GROUP
    ):
        endpoint_rows = _row_endpoints(
            topology, tx_positions, tx_power, rx_positions, tx_pol_batch, rx_pol_batch
        )
        source, target, source_power, tx_pol, rx_pol = endpoint_rows
        if taper_los:
            launch_count = _evaluate_los_fields(
                row_groups, source, target, source_power, tx_pol, rx_pol, los_field_op,
                None, outputs, launch_count, compiled,
                _los_taper_frequency(frequency_value, frequency), isb_boundary_taper_width,
            )
        material, launch_count = _evaluate_transmission_fields(
            compiled, topology, geometry, source, target, source_power, tx_pol, rx_pol,
            device, False, None, transmission_field_op, None, None, row_groups, outputs,
            launch_count,
        )
        _, launch_count = _evaluate_coupled_fields(
            scene, compiled, topology, geometry, source, target, source_power, tx_pol,
            rx_pol, device, frequency, frequency_value, False, False, None, material,
            row_groups, outputs, launch_count,
        )
    return outputs, launch_count


def evaluate_path_fields(
    scene: Scene, compiled: CompiledScene, paths: EvaluatedPaths, execution: PathExecutionStats,
    tx_positions: torch.Tensor, tx_power: torch.Tensor, rx_positions: torch.Tensor, *,
    row_groups: PathRowGroups, components: frozenset[str] | set[str] = frozenset(),
    ad_mode: str = "none", frequency_value: float | None = None,
    isb_boundary_taper_width: float = 0.0, endpoint_tx_polarizations: torch.Tensor | None = None,
    endpoint_rx_polarizations: torch.Tensor | None = None, explicit_endpoint_geometry: bool = False,
) -> tuple[EvaluatedPaths, PathExecutionStats]:
    """Evaluate selected canonical rows with the shared complex3 ABI.

    ``ad_mode == "none"`` evaluates line-of-sight, reflection and diffraction rows in one
    native launch over the canonical rows (float frequency, no autograd graph); transmission
    and coupled rows then run their own operators on their row groups. ``jvp``/``vjp`` route LoS,
    reflection and transmission through the differentiable field Functions:
    material gathers stay on the torch graph and the frequency is forwarded
    as a 0-d tensor when the scene stores one, alongside one precomputed
    host scalar (``frequency_value``) threaded into every field Function so
    a tensor-frequency solve pays a single device-to-host frequency read. Hit geometry stays detached under the fixed-topology
    contract unless a geometry leaf participates in AD, in which case it
    comes from RayD's fixed-winner geometry companions (see
    ``_reflection_geometry_ad``). The rough-surface C_r attenuation is pure
    torch and receives the same live frequency so dC_r/df is kept.
    """

    topology = paths.topology
    geometry = paths.geometry
    input_fields = paths.fields
    count = paths.row_count
    if count == 0:
        return paths, execution
    row_groups.require_row_count(count)
    ad_enabled = ad_mode != "none"
    # diffraction AD metadata: one ledger entry per registered differentiable
    # Function, tape bytes mirroring each Function's save_for_backward set.
    ledger = AdLaunchLedger() if ad_enabled else None
    if ad_enabled:
        frequency = (
            scene.frequency
            if isinstance(scene.frequency, torch.Tensor)
            else float(scene.frequency)
        )
        if frequency_value is None:
            frequency_value = _ad_frequency_value(frequency)
        frequency_value = float(frequency_value)
        los_field_op = partial(
            field_kernels.field_free_space_ad,
            frequency=frequency,
            frequency_value=frequency_value,
        )
        reflection_field_op = partial(
            field_kernels.field_reflection_sequence_ad,
            frequency=frequency,
            frequency_value=frequency_value,
        )
        transmission_field_op = partial(
            field_kernels.field_transmission_sequence_ad,
            frequency=frequency,
            frequency_value=frequency_value,
        )
    else:
        frequency = (
            _frequency_scalar(scene)
            if frequency_value is None
            else float(frequency_value)
        )
        los_field_op = partial(field_kernels.field_free_space, frequency_hz=frequency)
        transmission_field_op = partial(
            field_kernels.field_transmission_sequence, frequency_hz=frequency
        )
    device = paths.device
    # AD: when a geometry leaf is on the graph, the endpoints come from the
    # live scene tensors and the hit geometry comes from RayD's fixed-winner
    # chain companions (a native re-launch of the EPC discovery on the frozen
    # winner sequence plus its backward/jvp CUDA kernels), so the field
    # kernels see geometry with a gradient without any torch-side re-solve.
    # Otherwise the detached native discovery output is used unchanged (AD).
    if (endpoint_tx_polarizations is None) != (endpoint_rx_polarizations is None):
        raise ValueError("explicit endpoint polarizations must be provided together")
    explicit_geometry_ad = explicit_endpoint_geometry and any(
        _participates_in_ad(value) for value in (tx_positions, rx_positions)
    )
    geometry_ad = ad_enabled and (
        explicit_geometry_ad
        or _vertices_participate_in_ad(scene)
        or (not explicit_endpoint_geometry and _geometry_participates_in_ad(scene))
    )
    if geometry_ad:
        vertices = scene_endpoints.scene_vertex_table(scene, compiled)
        if not explicit_endpoint_geometry:
            tx_positions = scene_endpoints.transmitter_positions_ad(
                scene, tx_positions, device=device
            )
            rx_positions = scene_endpoints.receiver_positions_ad(
                scene, rx_positions, device=device
            )
    tx_pol_batch = (
        transmitter_polarizations_f32(scene, device=device)
        if endpoint_tx_polarizations is None
        else endpoint_tx_polarizations
    )
    rx_pol_batch = (
        receiver_polarization_rows(scene, device=device)
        if endpoint_rx_polarizations is None
        else endpoint_rx_polarizations
    )
    input_columns = {
        "field_xyz": input_fields.field_xyz,
        "coefficient": input_fields.coefficient,
        "path_field": input_fields.path_field,
        "path_gain": input_fields.path_gain,
        "path_length": geometry.path_length_m,
        "delay": geometry.delay_s,
        "direction": geometry.field_direction,
    }
    launch_count = execution.launch_count
    if ad_enabled:
        outputs = _FieldOutputs(input_columns)
        endpoint_rows = _row_endpoints(
            topology, tx_positions, tx_power, rx_positions, tx_pol_batch, rx_pol_batch
        )
        launch_count = _evaluate_differentiable_rows(
            scene, compiled, topology, geometry, input_fields, row_groups, endpoint_rows,
            components, device, frequency, frequency_value, geometry_ad,
            vertices if geometry_ad else None, los_field_op, reflection_field_op,
            transmission_field_op, ledger, outputs, launch_count, isb_boundary_taper_width,
        )
    else:
        outputs, launch_count = _evaluate_canonical_rows(
            scene, compiled, topology, geometry, input_columns, row_groups, tx_positions,
            tx_power, rx_positions, tx_pol_batch, rx_pol_batch, components, device, frequency,
            frequency_value, los_field_op, transmission_field_op, launch_count,
            isb_boundary_taper_width,
        )
    field_xyz = outputs.values["field_xyz"]
    coefficient = outputs.values["coefficient"]
    path_field = outputs.values["path_field"]
    path_gain = outputs.values["path_gain"]
    path_length = outputs.values["path_length"]
    delay = outputs.values["delay"]
    direction = outputs.values["direction"]

    updated_geometry = PathGeometry(
        row_identity=paths.row_identity,
        path_length_m=path_length,
        delay_s=delay,
        field_direction=direction,
        interaction_position=geometry.interaction_position,
        interaction_normal=geometry.interaction_normal,
        interaction_positions=geometry.interaction_positions,
        interaction_normals=geometry.interaction_normals,
    )
    updated_fields = PathFields(
        row_identity=paths.row_identity,
        path_gain=path_gain,
        path_field=path_field,
        field_xyz=field_xyz,
        coefficient=coefficient,
    )
    evaluated = EvaluatedPaths(
        topology=topology,
        geometry=updated_geometry,
        fields=updated_fields,
    )
    updated_execution = replace(
        execution,
        launch_count=launch_count,
        ad_companion_launches=(
            execution.ad_companion_launches + ledger.launches
            if ledger is not None
            else execution.ad_companion_launches
        ),
        ad_tape_bytes=(
            execution.ad_tape_bytes + ledger.tape_bytes
            if ledger is not None
            else execution.ad_tape_bytes
        ),
    )
    return evaluated, updated_execution


__all__ = [
    "_FieldCoupledChainAdFunction",
    "_FieldDiffractionWedgeAdFunction",
    "_FieldFreeSpaceAdFunction",
    "_FieldProjectComplex3AdFunction",
    "_FieldReflectionSequenceAdFunction",
    "_FieldTransmissionSequenceAdFunction",
    "field_coupled_chain",
    "field_coupled_chain_ad",
    "field_diffraction_wedge",
    "field_diffraction_wedge_ad",
    "field_free_space",
    "field_free_space_ad",
    "field_free_space_backward",
    "field_free_space_jvp",
    "field_project_complex3",
    "field_project_complex3_ad",
    "field_reflection_sequence",
    "field_reflection_sequence_ad",
    "field_reflection_sequence_backward",
    "field_reflection_sequence_jvp",
    "field_transmission_sequence",
    "field_transmission_sequence_ad",
    "field_transmission_sequence_backward",
    "field_transmission_sequence_jvp",
]
