"""ADR-021 D1 enumerated scatter-chain append path (Deterministic + Path).

Owns everything reachable only when ``scattering_chain_max_depth >= 1``: the
chain-sample building, discovery invocation, native chain-ensemble facade
dispatch, and multi-slot chain row assembly onto canonical typed path
contracts. ``scattering.append_scattering_evaluated_paths`` calls the single
public entry ``append_chain_scattering_paths`` from here; the single-bounce
scattering path stays in ``scattering.py``.

Discovery itself is owned by the sibling ``scattering_chain`` module; this
module consumes its ``ScatterChainDiscovery`` contract and appends the joined,
budgeted rows. The vertex-frame geometry mirrors the single-bounce ensemble
convention and reuses ``_unit``/``_stable_tangent`` plus the scene-owned
phase-screen assignment resolver so the two paths never diverge.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from witwin.channel.field_state import (
    receiver_polarizations,
    transmitter_polarizations,
)
from witwin.channel.materials import (
    face_material_tensors,
    face_material_thickness,
)
from witwin.core import PhaseScreen
from witwin.channel.kernels import scattering as scattering_kernels
from witwin.channel.propagation.enumerated.contracts import TopologyConfig
from witwin.channel.propagation.enumerated.scattering import (
    _stable_tangent,
    _unit,
)
from witwin.channel.scene.resources import (
    realization_phase_screens,
)
from witwin.channel.scene.endpoints import require_compiled
from witwin.channel.propagation.enumerated.scattering_chain import (
    KMAX_AD_DEPTH,
    ScatterChainDiscovery,
    build_chain_samples,
    discover_scatter_chains,
)
from witwin.channel.propagation.enumerated.scattering_scalars import (
    ensemble_coef_scale,
)
from witwin.channel.propagation.geometry.endpoints import (
    receiver_positions_and_layout,
    transmitter_tensors,
)
from witwin.channel.propagation.rows import (
    EvaluatedPaths,
    PathFields,
    PathGeometry,
    PathTopology,
)
from witwin.channel.propagation.topology.export import EvaluatedPathSidecars
from witwin.channel.constants import C0

if TYPE_CHECKING:
    from witwin.channel.scene.endpoints import SolverScene as Scene

__all__ = ["append_chain_scattering_paths"]

# ADR-021 D5 has never been reachable from this append path. Before the kernel
# facades were lifted into ``witwin.channel.kernels``, the Op A ``_ad`` probe
# below read the single-bounce ``scattering/kernels/autograd.py`` namespace,
# which never carried a chain symbol, so ``ad_mode != "none"`` always refused
# here and the AD dispatch a few lines further down was unreachable. The
# facades are one module now, so that refusal is stated instead of implied.
# Turning it on is an ADR-021 D5 numerical decision with its own acceptance
# evidence, not a layout move.
_ADR021_D5_CHAIN_AD_WIRED = False


def _ensemble_scatter_faces(
    compiled: object, screens: dict[int, PhaseScreen], *, device: torch.device
) -> torch.Tensor:
    """Ensemble (non-realization) rough scatter faces (single-bounce parity)."""

    face_material = compiled.assignments.face_material_id.to(
        device=device, dtype=torch.int64
    )
    scatter_face = (
        compiled.materials.scatter_model_id.to(device=device)[face_material] == 1
    )
    face_structure = compiled.geometry.face_structure_id.to(
        device=device, dtype=torch.int64
    )
    realization_face = torch.zeros_like(scatter_face)
    for structure_index in screens:
        structure_id = int(compiled.assignments.structure_id[structure_index])
        realization_face |= face_structure == structure_id
    return torch.nonzero(scatter_face & ~realization_face, as_tuple=False).reshape(-1)


def _chain_bounce_material(
    face_param: torch.Tensor, primitive: torch.Tensor, pad: float
) -> torch.Tensor:
    """Gather a per-face Fresnel parameter onto a padded ``[R, Dmax]`` leg block.

    Padded slots (``primitive < 0``) take the identity ``pad`` value so the
    native chain transport treats them as no-ops (ADR-021 section 1 padding).
    """

    valid = primitive >= 0
    idx = primitive.clamp_min(0).to(torch.int64)
    gathered = face_param.index_select(0, idx.reshape(-1)).reshape(primitive.shape)
    return torch.where(valid, gathered, torch.full_like(gathered, pad)).contiguous()


def _chain_vertex_frame(
    discovery: ScatterChainDiscovery, rough_axis_rad: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Vertex roughness frame + incident local coords (single-bounce parity).

    Mirrors ``_ensemble_rows``: the mean-plane normal is flipped toward the
    incident side, the principal roughness axes ``t1r``/``t2r`` are rotated by the
    material azimuth, and the incident direction is expressed in the local table
    frame (``wi_local``).
    """

    d_i = discovery.d_i
    v_material = discovery.v_material.to(torch.int64)
    side = torch.sign((-d_i * discovery.v_normal).sum(-1))
    side = torch.where(side == 0.0, torch.ones_like(side), side)
    n_o = _unit(discovery.v_normal * side[:, None])
    backup_axis = _stable_tangent(n_o)
    angle = rough_axis_rad.to(device=n_o.device, dtype=torch.float32)[v_material]
    t1 = backup_axis
    t2 = torch.cross(n_o, t1, dim=-1)
    t1r = t1 * torch.cos(angle)[:, None] + t2 * torch.sin(angle)[:, None]
    t2r = torch.cross(n_o, t1r, dim=-1)
    wi_hat = -d_i
    cos_i = (wi_hat * n_o).sum(-1)
    wi_local = torch.stack(
        ((wi_hat * t1r).sum(-1), (wi_hat * t2r).sum(-1), cos_i), dim=-1
    )
    return {
        "n_o": n_o.contiguous(),
        "t1r": t1r.contiguous(),
        "t2r": t2r.contiguous(),
        "backup_axis": backup_axis.contiguous(),
        "cos_i": cos_i.contiguous(),
        "wi_local": wi_local.contiguous(),
    }


def _chain_ensemble_evaluate(
    scene: Scene,
    compiled: object,
    config: TopologyConfig,
    discovery: ScatterChainDiscovery,
    *,
    tx_pol: torch.Tensor,
    rx_pol: torch.Tensor,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    device: torch.device,
    ad_mode: str,
) -> dict[str, torch.Tensor]:
    """Dispatch the native ADR-021 Op A chain-ensemble kernel per transmitter.

    Builds the AS-BUILT native Op A argument set from the discovery contract plus
    the resident Kirchhoff tables and per-face material tensors, then dispatches
    the native ``scattering_chain_ensemble_eval`` (or its ``_ad`` companion) once
    per transmitter so the AD-live radiometric ``coef`` (which carries the per-tx
    power) crosses the ABI as a scalar exactly as the single-bounce op-1 path.
    The endpoint positions ``source``/``vertex``/``target`` feed the C1/C2
    transport and ``weights`` (per-vertex ``A_patch``) follows the op-1
    convention (with the ``1/(L1^2 L2^2)`` spreading applied in-kernel).
    """

    op_a = getattr(scattering_kernels, "scattering_chain_ensemble_eval", None)
    op_a_ad = (
        getattr(scattering_kernels, "scattering_chain_ensemble_eval_ad", None)
        if _ADR021_D5_CHAIN_AD_WIRED
        else None
    )
    if op_a is None or (ad_mode != "none" and op_a_ad is None):
        raise RuntimeError(
            "ADR-021 D1 requires the native Op A chain facade "
            "'scattering_chain_ensemble_eval' (and its '_ad' companion under "
            "ad_mode != 'none'); it is owned by the ADR-021 D2 native chain "
            "kernels (witwin.channel.kernels.scattering) and is not yet wired "
            "into this append path. Enable scattering_chain_max_depth only "
            "once the D5 wave has landed."
        )

    ad_enabled = ad_mode != "none"
    face_eps, face_sigma, face_mu, face_gain, _valid = face_material_tensors(
        compiled, device=device
    )
    face_thickness = face_material_thickness(compiled, device=device)

    def leg_material(primitive: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "eps_r": _chain_bounce_material(face_eps, primitive, 1.0),
            "sigma_e": _chain_bounce_material(face_sigma, primitive, 0.0),
            "mu_r": _chain_bounce_material(face_mu, primitive, 1.0),
            "gain": _chain_bounce_material(face_gain, primitive, 1.0),
            "thickness": _chain_bounce_material(face_thickness, primitive, 0.0),
        }

    c1 = leg_material(discovery.c1_primitive)
    c2 = leg_material(discovery.c2_primitive)
    frame = _chain_vertex_frame(discovery, compiled.materials.rough_axis_rad)

    stack = compiled.kirchhoff_resources.stack
    material_id = discovery.v_material.to(torch.int32).contiguous()
    rows = discovery.row_count

    frequency = float(scene.frequency)
    wavelength = C0 / frequency
    power_scale = wavelength**2 / (4.0 * math.pi) ** 2
    coef_scale_t = ensemble_coef_scale(scene, device, ad_enabled=ad_enabled)
    threshold = max(float(getattr(config, "scattering_power_threshold", 0.0)), 0.0)
    tx_id = discovery.tx_id.to(torch.int64)

    gain = torch.zeros((rows,), device=device, dtype=torch.float32)
    amplitude = torch.zeros((rows,), device=device, dtype=torch.float32)
    length = torch.zeros((rows,), device=device, dtype=torch.float32)
    keep = torch.zeros((rows,), device=device, dtype=torch.bool)

    # Discovery freezes rows in tx-major order. Reuse each existing device
    # selection mask as the RayD row-valid contract: narrowing the contiguous
    # tx block is a view (no allocation or kernel), while manufacturing an
    # all-true tensor here would violate the caller-owned validity contract.
    row_offset = 0
    for tx_index in torch.unique(tx_id).tolist():
        mask = tx_id == tx_index
        idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
        row_count = int(idx.numel())
        if row_count == 0:
            continue
        row_valid = mask.narrow(0, row_offset, row_count)
        row_offset += row_count
        row_tx = discovery.tx_id[idx].to(torch.int64)
        row_rx = discovery.rx_id[idx].to(torch.int64)
        args = (
            row_valid,
            tx_pol.index_select(0, row_tx).contiguous(),
            rx_pol.index_select(0, row_rx).contiguous(),
            tx_positions.index_select(0, row_tx).contiguous(),
            discovery.v_pos[idx].contiguous(),
            rx_positions.index_select(0, row_rx).contiguous(),
            discovery.c1_positions[idx].contiguous(),
            discovery.c1_normals[idx].contiguous(),
            c1["eps_r"][idx],
            c1["sigma_e"][idx],
            c1["mu_r"][idx],
            c1["gain"][idx],
            c1["thickness"][idx],
            discovery.d1[idx].contiguous(),
            discovery.c2_positions[idx].contiguous(),
            discovery.c2_normals[idx].contiguous(),
            c2["eps_r"][idx],
            c2["sigma_e"][idx],
            c2["mu_r"][idx],
            c2["gain"][idx],
            c2["thickness"][idx],
            discovery.d2[idx].contiguous(),
            frame["n_o"][idx],
            frame["t1r"][idx],
            frame["t2r"][idx],
            frame["backup_axis"][idx],
            frame["wi_local"][idx],
            discovery.cos_i[idx].contiguous(),
            discovery.cos_o[idx].contiguous(),
            discovery.d_i[idx].contiguous(),
            discovery.d_o[idx].contiguous(),
            discovery.L1[idx].contiguous(),
            discovery.L2[idx].contiguous(),
            discovery.weight[idx].contiguous(),
            material_id[idx],
            stack.f_te_flat,
            stack.f_tm_flat,
            stack.table_offset,
            stack.table_dims,
            stack.material_slot,
        )
        if ad_enabled:
            evaluated = op_a_ad(
                *args,
                coef=float(tx_power[tx_index]) * coef_scale_t,
                threshold=threshold,
                frequency=frequency,
            )
        else:
            evaluated = op_a(
                *args,
                coef=float(tx_power[tx_index]) * power_scale,
                threshold=threshold,
                frequency_hz=frequency,
            )
        gain[idx] = evaluated["gain"].to(torch.float32)
        amplitude[idx] = evaluated["amplitude"].to(torch.float32)
        length[idx] = evaluated["length"].to(torch.float32)
        keep[idx] = evaluated["keep"]

    if row_offset != rows:
        raise RuntimeError("scatter-chain tx-major row coverage contract was violated")

    return {
        "gain": gain,
        "amplitude": amplitude,
        "length": length,
        "keep": keep,
        "n_o": frame["n_o"],
    }


def _chain_topology_slots(
    discovery: ScatterChainDiscovery,
    vertex_face: torch.Tensor,
    vertex_normal: torch.Tensor,
    width: int,
) -> dict[str, torch.Tensor]:
    """Build the multi-slot interaction sequence of every chain row.

    Slot layout per row (ADR-021 D1): ``[REFLECTION]*d1 + [SCATTERING] +
    [REFLECTION]*d2``; C1 fills the leading ``d1`` slots, the diffuse vertex sits
    at slot ``d1``, and C2 fills slots ``d1 + 1 .. d1 + d2``. Padded slots carry
    ``primitive/material = -1`` and ``interaction_type = 0`` (inactive).
    """

    device = discovery.device
    rows = discovery.row_count
    dmax = KMAX_AD_DEPTH
    d1 = discovery.d1.to(torch.int64)
    d2 = discovery.d2.to(torch.int64)
    prim = torch.full((rows, width), -1, device=device, dtype=torch.int32)
    mat = torch.full((rows, width), -1, device=device, dtype=torch.int32)
    itype = torch.zeros((rows, width), device=device, dtype=torch.int32)
    pos = torch.zeros((rows, width, 3), device=device, dtype=torch.float32)
    nrm = torch.zeros((rows, width, 3), device=device, dtype=torch.float32)
    ones = torch.ones((rows,), device=device, dtype=torch.int32)

    span = min(dmax, width)
    # C1 leg: leading slots 0..d1-1.
    for k in range(span):
        active = k < d1
        col = torch.where(active, discovery.c1_primitive[:, k], prim[:, k])
        prim[:, k] = col
        mat[:, k] = torch.where(active, discovery.c1_material[:, k], mat[:, k])
        itype[:, k] = torch.where(active, ones, itype[:, k])
        pos[:, k] = torch.where(
            active[:, None], discovery.c1_positions[:, k], pos[:, k]
        )
        nrm[:, k] = torch.where(
            active[:, None], discovery.c1_normals[:, k], nrm[:, k]
        )

    # Diffuse vertex at slot d1 (SCATTERING=8).
    row_idx = torch.arange(rows, device=device, dtype=torch.int64)
    prim[row_idx, d1] = vertex_face.to(torch.int32)
    mat[row_idx, d1] = discovery.v_material.to(torch.int32)
    itype[row_idx, d1] = 8
    pos[row_idx, d1] = discovery.v_pos
    nrm[row_idx, d1] = vertex_normal

    # C2 leg: slots d1+1 .. d1+d2.
    for k in range(span):
        active = k < d2
        rows_k = torch.nonzero(active, as_tuple=False).reshape(-1)
        if int(rows_k.numel()) == 0:
            continue
        slot = d1[rows_k] + 1 + k
        prim[rows_k, slot] = discovery.c2_primitive[rows_k, k]
        mat[rows_k, slot] = discovery.c2_material[rows_k, k]
        itype[rows_k, slot] = 1
        pos[rows_k, slot] = discovery.c2_positions[rows_k, k]
        nrm[rows_k, slot] = discovery.c2_normals[rows_k, k]

    return {
        "primitive_sequence": prim,
        "material_sequence": mat,
        "interaction_type": itype,
        "interaction_positions": pos,
        "interaction_normals": nrm,
    }


def _extend_evaluated_paths_chain(
    evaluated: EvaluatedPaths,
    sidecars: EvaluatedPathSidecars,
    discovery: ScatterChainDiscovery,
    physics: dict[str, torch.Tensor],
    vertex_face: torch.Tensor,
    *,
    tx_power: torch.Tensor,
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars]:
    """Append component_id=6 multi-slot scatter-chain rows to typed contracts."""

    topology = evaluated.topology
    geometry = evaluated.geometry
    fields = evaluated.fields
    device = topology.valid.device
    keep = physics["keep"]
    idx = torch.nonzero(keep, as_tuple=False).reshape(-1)
    count = int(idx.numel())
    if count == 0:
        return evaluated, sidecars

    discovery = _index_discovery(discovery, idx)
    vertex_face = vertex_face[idx]
    vertex_normal = physics["n_o"][idx]
    gain = physics["gain"][idx]
    amplitude = physics["amplitude"][idx]
    length = physics["length"][idx]

    depth = (discovery.d1 + 1 + discovery.d2).to(torch.int32)
    chain_width = int(depth.max().item())
    base_width = int(topology.primitive_sequence.shape[1])
    width = max(base_width, chain_width, 1)

    slots = _chain_topology_slots(discovery, vertex_face, vertex_normal, width)
    zero = torch.zeros_like(amplitude)
    tx_power_row = tx_power.index_select(0, discovery.tx_id.to(torch.int64)).clamp_min(
        1.0e-30
    )
    coefficient = torch.complex(amplitude / tx_power_row.sqrt(), zero)
    field_xyz = torch.zeros((count, 3), device=device, dtype=torch.complex64)

    def cat(existing: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        return torch.cat((existing, new.to(existing.dtype))).contiguous()

    def pad_seq(seq: torch.Tensor, fill: int) -> torch.Tensor:
        if int(seq.shape[1]) >= width:
            return seq
        tail = torch.full(
            (seq.shape[0], width - int(seq.shape[1])),
            fill,
            device=device,
            dtype=seq.dtype,
        )
        return torch.cat((seq, tail), dim=1)

    def pad_vec(seq: torch.Tensor) -> torch.Tensor:
        if int(seq.shape[1]) >= width:
            return seq
        tail = torch.zeros(
            (seq.shape[0], width - int(seq.shape[1]), 3),
            device=device,
            dtype=seq.dtype,
        )
        return torch.cat((seq, tail), dim=1)

    new_topology = PathTopology(
        valid=cat(topology.valid, torch.ones((count,), device=device, dtype=torch.bool)),
        tx_id=cat(topology.tx_id, discovery.tx_id),
        rx_id=cat(topology.rx_id, discovery.rx_id),
        depth=cat(topology.depth, depth),
        component_id=cat(
            topology.component_id,
            torch.full((count,), 6, device=device, dtype=torch.int32),
        ),
        primitive_id=cat(topology.primitive_id, vertex_face),
        edge_id=cat(
            topology.edge_id,
            torch.full((count,), -1, device=device, dtype=torch.int32),
        ),
        material_id=cat(topology.material_id, discovery.v_material),
        primitive_sequence=torch.cat(
            (pad_seq(topology.primitive_sequence, -1), slots["primitive_sequence"])
        ).contiguous(),
        material_sequence=torch.cat(
            (pad_seq(topology.material_sequence, -1), slots["material_sequence"])
        ).contiguous(),
        interaction_type=torch.cat(
            (pad_seq(topology.interaction_type, 0), slots["interaction_type"])
        ).contiguous(),
    )
    new_geometry = PathGeometry(
        row_identity=new_topology.row_identity,
        path_length_m=cat(geometry.path_length_m, length),
        delay_s=cat(geometry.delay_s, length / C0),
        field_direction=cat(geometry.field_direction, discovery.d_o),
        interaction_position=cat(geometry.interaction_position, discovery.v_pos),
        interaction_normal=cat(geometry.interaction_normal, vertex_normal),
        interaction_positions=torch.cat(
            (pad_vec(geometry.interaction_positions), slots["interaction_positions"])
        ).contiguous(),
        interaction_normals=torch.cat(
            (pad_vec(geometry.interaction_normals), slots["interaction_normals"])
        ).contiguous(),
    )
    new_fields = PathFields(
        row_identity=new_topology.row_identity,
        path_gain=cat(fields.path_gain, gain),
        path_field=cat(fields.path_field, torch.complex(amplitude, zero)),
        field_xyz=cat(fields.field_xyz, field_xyz),
        coefficient=cat(fields.coefficient, coefficient),
    )
    return (
        EvaluatedPaths(
            topology=new_topology, geometry=new_geometry, fields=new_fields
        ),
        sidecars,
    )


def _index_discovery(
    discovery: ScatterChainDiscovery, idx: torch.Tensor
) -> ScatterChainDiscovery:
    return ScatterChainDiscovery(
        **{
            spec: getattr(discovery, spec)[idx].contiguous()
            for spec in (
                "tx_id",
                "rx_id",
                "sample_index",
                "d1",
                "d2",
                "c1_positions",
                "c1_normals",
                "c1_primitive",
                "c1_material",
                "L1",
                "d_i",
                "c2_positions",
                "c2_normals",
                "c2_primitive",
                "c2_material",
                "L2",
                "d_o",
                "v_pos",
                "v_normal",
                "v_material",
                "weight",
                "cos_i",
                "cos_o",
                "patch_row",
            )
        }
    )


def append_chain_scattering_paths(
    scene: Scene,
    config: TopologyConfig,
    evaluated: EvaluatedPaths,
    sidecars: EvaluatedPathSidecars,
    info: dict[str, Any],
    *,
    ad_mode: str,
) -> tuple[EvaluatedPaths, EvaluatedPathSidecars]:
    """ADR-021 D1: discover and append enumerated scatter-chain rows.

    Default-OFF: ``scattering_chain_max_depth < 1`` returns the inputs unchanged
    (this branch is never entered, so the pipeline is byte-identical).
    """

    device = evaluated.device
    compiled = require_compiled(scene)
    if not compiled.rayd.available:
        raise RuntimeError(
            "deterministic scatter-chain discovery requires RayD native capability"
        )
    screens = realization_phase_screens(compiled.materials, compiled.assignments)
    ensemble_faces = _ensemble_scatter_faces(compiled, screens, device=device)
    samples = build_chain_samples(compiled, config, ensemble_faces, device=device)
    info["chain_sample_count"] = 0 if samples is None else int(samples.position.shape[0])
    if samples is None:
        return evaluated, sidecars

    tx_positions, tx_power = transmitter_tensors(scene, device=device)
    rx_positions, _layout = receiver_positions_and_layout(scene, device=device)
    if tx_positions.numel() == 0 or rx_positions.numel() == 0:
        return evaluated, sidecars
    tx_pol = transmitter_polarizations(scene, device=device)
    rx_pol = receiver_polarizations(scene, device=device)
    records = compiled.rayd.edge_records()
    vertices = records.vertices
    scene_diagonal = (vertices.max(dim=0).values - vertices.min(dim=0).values).norm()

    discovery = discover_scatter_chains(
        compiled,
        config,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        samples=samples,
        scene_diagonal=scene_diagonal,
    )
    info["chain_row_count"] = 0 if discovery is None else discovery.row_count
    if discovery is None:
        return evaluated, sidecars

    physics = _chain_ensemble_evaluate(
        scene,
        compiled,
        config,
        discovery,
        tx_pol=tx_pol,
        rx_pol=rx_pol,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        tx_power=tx_power,
        device=device,
        ad_mode=ad_mode,
    )
    vertex_face = samples.face_id[discovery.sample_index].contiguous()
    evaluated, sidecars = _extend_evaluated_paths_chain(
        evaluated,
        sidecars,
        discovery,
        physics,
        vertex_face,
        tx_power=tx_power,
    )
    info["chain_kept_count"] = int(physics["keep"].sum().item())
    return evaluated, sidecars
