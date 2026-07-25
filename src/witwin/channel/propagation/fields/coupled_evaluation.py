from __future__ import annotations

from functools import partial

import torch

from witwin.channel.runtime.kernel_metadata import AdLaunchLedger
from witwin.channel.propagation.fields.kernels import (
    autograd as field_autograd,
)
from witwin.channel.propagation.fields.kernels import (
    autograd_coupled_dd as field_autograd_dd,
)
from witwin.channel.propagation.fields.kernels import (
    functional as field_functional,
)
from witwin.channel.propagation.models.geometry import PathGeometry
from witwin.channel.propagation.models.topology import PathTopology


def _resolve_coupled_rd_stationary(
    source: torch.Tensor,
    target: torch.Tensor,
    rows: torch.Tensor,
    reverse_order: bool,
    tri_a: torch.Tensor,
    normals_table: torch.Tensor,
    reflection_face: torch.Tensor,
    edge_geometry: tuple[torch.Tensor, ...],
    edge_id: torch.Tensor,
    edge_position: torch.Tensor,
    reflection_position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable fixed-winner coupled stationary re-solve (cid 3/4, AD-4).

    Pure lift of the cid-3/4 ``coupled_geometry_ad`` branch out of
    ``_evaluate_coupled_fields``, with identical math and error behavior.
    Plan 07 AD-4: the coupled interaction points move with the endpoints
    (Fresnel angles are not stationary), so re-solve the frozen winner's
    stationary geometry differentiably (the same image-source math as the
    discovery prepare kernel) and feed the live points into the field kernel.
    D->R is the reciprocal problem with the endpoints exchanged. Returns the
    live (edge_point, reflection_point); the passed positions are the frozen
    discovery references used only to guard that the winner did not move.
    """

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
    return resolved["edge_point"], resolved["reflection_point"]


def _evaluate_coupled_dd_rows(
    topology: PathTopology,
    geometry: PathGeometry,
    source: torch.Tensor,
    target: torch.Tensor,
    source_power: torch.Tensor,
    tx_pol: torch.Tensor,
    rx_pol: torch.Tensor,
    material: dict[str, torch.Tensor],
    edge_geometry: tuple[torch.Tensor, ...],
    edge_n0: torch.Tensor,
    edge_n1: torch.Tensor,
    edge_exterior: torch.Tensor,
    edge_face0: torch.Tensor,
    edge_face1: torch.Tensor,
    ad_enabled: bool,
    frequency: float | torch.Tensor,
    frequency_value: float | None,
    ledger: AdLaunchLedger | None,
    field_xyz: torch.Tensor,
    coefficient: torch.Tensor,
    path_field: torch.Tensor,
    path_gain: torch.Tensor,
    direction: torch.Tensor,
    launch_count: int,
) -> int:
    """Evaluate the coupled double-diffraction rows (cid 7, ADR-013).

    Pure lift of the cid-7 branch out of ``_evaluate_coupled_fields``:
    identical gather math, tensor ordering, row identity, aliasing and error
    behavior. Both interactions are diffraction (type 2), so both
    primitive_sequence slots carry edge ids; interaction_positions carry the
    frozen Fermat seeds [Q1, Q2]. Results land in the same coherent coupled
    slot as cid 3/4 via in-place ``index_copy_`` on the shared output tensors,
    so their storage/aliasing is preserved. Returns the updated launch_count.
    """

    dd_rows = torch.nonzero(
        topology.component_id == 7, as_tuple=False
    ).reshape(-1)
    if int(dd_rows.shape[0]) > 0:
        edge1_id = topology.primitive_sequence[dd_rows, 0].to(dtype=torch.int64)
        edge2_id = topology.primitive_sequence[dd_rows, 1].to(dtype=torch.int64)
        q1 = geometry.interaction_positions[dd_rows, 0].contiguous()
        q2 = geometry.interaction_positions[dd_rows, 1].contiguous()

        def _dd_line_bounds(
            edge_id: torch.Tensor, keller_point: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # G4/ADR-013: edge-segment bounds relative to the passed Keller
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

        def _dd_wedge_faces(
            edge_id: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            face0 = edge_face0[edge_id]
            raw_face1 = edge_face1[edge_id]
            # Boundary edges have no second face; reuse face0 like cid 3/4.
            face1 = torch.where(raw_face1 >= 0, raw_face1, face0)
            return face0, face1

        edge1_face0, edge1_face1 = _dd_wedge_faces(edge1_id)
        edge2_face0, edge2_face1 = _dd_wedge_faces(edge2_id)

        def _dd_material_tuple(
            face: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
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
                field_autograd_dd.field_coupled_dd_ad,
                frequency=frequency,
                frequency_value=frequency_value,
            )
            if ad_enabled
            else partial(
                field_functional.field_coupled_dd, frequency_hz=frequency
            )
        )
        dd_args = (
            source[dd_rows].contiguous(),
            target[dd_rows].contiguous(),
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
            source_power[dd_rows].contiguous(),
            tx_pol[dd_rows].contiguous(),
            rx_pol[dd_rows].contiguous(),
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
        field_xyz.index_copy_(0, dd_rows, evaluated["field_vector"])
        coefficient.index_copy_(0, dd_rows, evaluated["coefficient"])
        path_field.index_copy_(0, dd_rows, evaluated["path_field"])
        path_gain.index_copy_(0, dd_rows, evaluated["path_gain"])
        direction.index_copy_(0, dd_rows, evaluated["direction"])
        launch_count += 1
    return launch_count
