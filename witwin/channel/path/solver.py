from __future__ import annotations

import time
from collections.abc import Mapping
from types import SimpleNamespace

import drjit as dr
import torch

from witwin.channel.core.numerics import arrays
from witwin.channel.core.scene import Receiver, Scene, Transmitter
from witwin.channel.core.numerics.arrays import scalar
from witwin.channel.core.physics.polarization import effective_rx_polarization
from witwin.channel.core.results.ray_mode import DEFAULT_RAY_MODE
from witwin.channel.core.runtime import (
    TraceContext,
    assert_scene_materials_complete,
    point_grad_enabled,
    scene_geometry_grad_enabled,
    scene_material_grad_enabled,
)
from witwin.channel.core.geometry.mesh_buffers import to_point3f
from witwin.channel.core.numerics.tensors import to_torch_view
from witwin.channel.deterministic import types as wt
from witwin.channel.deterministic.diffraction.accumulation import (
    trace_diffraction_raw_collections,
)
from witwin.channel.deterministic.diffraction.state import PATH_EXPORT_REDUCED_STATE_LAYOUT
from witwin.channel.deterministic.trace.path_export import (
    collect_diffraction_state_paths,
    collect_los_paths,
    collect_reflection_paths,
    collect_reflection_paths_for_transmitters,
    empty_raw_paths,
    finalize_raw_paths,
    path_export_state_layout,
    spherical_angles,
)
from witwin.channel.deterministic.trace.path_export_assembly import assemble_result_payload
from .config import Config, PathSolveSpec, derive_wave_params, resolve_solver_controls
from .result import InteractionType, PathResult


def _point_list_to_point3f(points, *, role: str) -> wt.Point3f:
    if not points:
        raise ValueError(f"{role} endpoint list must not be empty.")
    if all(all(hasattr(point, axis) for axis in ("x", "y", "z")) for point in points):
        return wt.Point3f(
            _concat_float_arrays([wt.Float(point.x) for point in points]),
            _concat_float_arrays([wt.Float(point.y) for point in points]),
            _concat_float_arrays([wt.Float(point.z) for point in points]),
        )
    return to_point3f(points, role=role, allow_single=False)


def _gather_tx_position(tx_positions, tx_index: int) -> wt.Point3f:
    index = dr.full(wt.UInt32, int(tx_index), 1)
    return wt.Point3f(
        dr.gather(wt.Float, tx_positions.x, index),
        dr.gather(wt.Float, tx_positions.y, index),
        dr.gather(wt.Float, tx_positions.z, index),
    )


def _resolve_scene_endpoint_list(scene: Scene, endpoint, *, role: str) -> list:
    if isinstance(endpoint, str):
        return [getattr(scene, role)(endpoint)]
    if isinstance(endpoint, (Receiver, Transmitter)):
        return [endpoint]
    if isinstance(endpoint, (list, tuple)):
        items = [
            getattr(scene, role)(item) if isinstance(item, str) else item
            for item in endpoint
        ]
        if not items:
            raise ValueError(f"{role} endpoint list must not be empty.")
        return items
    return [endpoint]


def _shared_polarization(endpoints: list, *, role: str):
    pols = [endpoint.polarization for endpoint in endpoints if endpoint.polarization is not None]
    if pols and any(pol != pols[0] for pol in pols[1:]):
        raise ValueError(f"path {role} endpoints in one solve must share polarization.")
    return pols[0] if pols else None


def _transmitters_from_endpoint(scene: Scene, transmitter):
    items = _resolve_scene_endpoint_list(scene, transmitter, role="transmitter")
    if not all(isinstance(item, Transmitter) for item in items):
        raise TypeError("path transmitter endpoint must be a Transmitter or non-empty sequence of Transmitter instances.")
    positions = _point_list_to_point3f([item.position for item in items], role="transmitter")
    tx_polarization = _shared_polarization(items, role="transmitter") or (1.0, 0.0, 0.0)
    return positions, ",".join(item.name for item in items), tx_polarization, items


def _receivers_from_endpoint(scene: Scene, receiver):
    items = _resolve_scene_endpoint_list(scene, receiver, role="receiver")
    if not all(isinstance(item, Receiver) for item in items):
        raise TypeError("path receiver endpoint must be a Receiver or non-empty sequence of Receiver instances.")
    positions = _point_list_to_point3f([item.position for item in items], role="receiver")
    return positions, ",".join(item.name for item in items), _shared_polarization(items, role="receiver"), items


def _normalize_transmitters(*, scene: Scene, transmitter):
    if transmitter is None:
        raise ValueError("paths.solve requires transmitter.")
    return _transmitters_from_endpoint(scene, transmitter)


def _normalize_receivers(*, scene: Scene, receiver):
    if receiver is None:
        raise ValueError("paths.solve requires receiver.")
    return _receivers_from_endpoint(scene, receiver)


def _stamp_raw(raw: dict[str, object], *, tx_index: int, receiver_index_map=None) -> None:
    n = int(dr.width(raw["rx_index"]))
    if receiver_index_map is not None and n > 0:
        raw["rx_index"] = dr.gather(wt.UInt32, receiver_index_map, raw["rx_index"])
    raw["tx_index"] = dr.full(wt.UInt32, int(tx_index), n)


def _rayd_diffraction_path_backend_metadata() -> dict[str, object]:
    return {
        "implementation": "rayd_trace_dfr_paths_order1",
        "path_export_backend": "rayd_optix_compact_paths",
        "max_order": 1,
    }


def _drjit_diffraction_path_backend_metadata(max_order: int) -> dict[str, object]:
    return {
        "implementation": "drjit_diffraction_state_path_export",
        "path_export_backend": "drjit_state_materialization",
        "max_order": int(max_order),
    }


def _path_has_ad_inputs(*, scene: Scene, tx_positions, rx_positions) -> bool:
    return (
        point_grad_enabled(tx_positions)
        or point_grad_enabled(rx_positions)
        or scene_geometry_grad_enabled(scene)
        or scene_material_grad_enabled(scene)
    )


def _resolve_reflection_path_backend(
    *,
    config: Config,
    scene: Scene,
    tx_positions,
    rx_positions,
) -> dict[str, object]:
    requested = str(config.tuning.reflection_field_backend)
    ad_inputs = _path_has_ad_inputs(
        scene=scene,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
    )
    if requested == "native":
        return {
            "requested": requested,
            "epc_backend": "rayd_optix",
            "prefer_rayd_epc": True,
            "require_rayd_epc": True,
            "rayd_epc_required": True,
            "ad_inputs": bool(ad_inputs),
        }
    return {
        "requested": requested,
        "epc_backend": "drjit",
        "prefer_rayd_epc": False,
        "require_rayd_epc": False,
        "rayd_epc_required": False,
        "ad_inputs": bool(ad_inputs),
    }


def _resolve_diffraction_path_accumulate_primal(
    *,
    config: Config,
    scene: Scene,
    tx_positions,
    rx_positions,
    max_order: int,
) -> str:
    mode = str(getattr(config.tuning.diffraction_execution, "accumulate_primal", "auto"))
    if mode != "auto":
        return mode
    if int(max_order) == 1:
        return "rayd_optix"
    return "drjit"


def _edge_object_indices_for_native_paths(scene: Scene, edge_data, edge_idx):
    count = int(dr.width(edge_idx))
    if edge_data is None or int(edge_data.get("n_edges", 0)) <= 0:
        return dr.full(wt.Int32, -1, count)
    n_edges = int(edge_data["n_edges"])
    valid_edge = (edge_idx >= wt.Int32(0)) & (edge_idx < wt.Int32(n_edges))
    safe_edge = wt.UInt32(dr.select(valid_edge, edge_idx, wt.Int32(0)))
    face0 = dr.gather(wt.Int32, edge_data["adjacent_face0"], safe_edge)
    face1 = dr.gather(wt.Int32, edge_data["adjacent_face1"], safe_edge)
    use_face0 = face0 >= wt.Int32(0)
    face = dr.select(use_face0, face0, face1)
    valid_face = valid_edge & (face >= wt.Int32(0))
    safe_face = wt.UInt32(dr.select(valid_face, face, wt.Int32(0)))
    return scene.gather_structure_indices(safe_face, valid_mask=valid_face)


def collect_native_diffraction_paths(
    *,
    state_arrays,
    edge_data,
    scene: Scene,
    rx_positions,
    tx_pos,
    wavelength: float,
    k: float,
    max_paths: int,
    seed: int,
    return_geometry: bool,
    stats: dict[str, object] | None = None,
) -> dict[str, object]:
    if stats is None:
        stats = {}
    stats.clear()
    stats.update(
        {
            "backend": "rayd_optix_compact_paths",
            "input_states": 0 if state_arrays is None else int(state_arrays.get("n_states", 0)),
            "receiver_count": int(dr.width(rx_positions.x)) if rx_positions is not None else 0,
            "output_paths": 0,
            "materialization_deferred": False,
        }
    )
    backend_metadata = _rayd_diffraction_path_backend_metadata()
    if (
        state_arrays is None
        or int(state_arrays.get("n_states", 0)) == 0
        or rx_positions is None
        or int(dr.width(rx_positions.x)) == 0
    ):
        raw = empty_raw_paths(depth=1, return_geometry=return_geometry)
        raw["metadata"] = {"n_paths": 0, "runtime_backend": backend_metadata}
        return raw

    native = scene.trace_dfr_paths(
        tx_positions=tx_pos,
        rx_positions=rx_positions,
        state_arrays=state_arrays,
        config=SimpleNamespace(wavelength=float(wavelength), k=float(k)),
        max_order=1,
        max_paths=int(max_paths),
        seed=int(seed),
        return_geometry=bool(return_geometry),
        active=True,
    )
    capacity = int(getattr(native, "capacity", int(dr.width(native.rx_id))))
    count = max(0, min(capacity, int(scalar(wt.Int32(native.count)))))
    stats["output_paths"] = count
    if count == 0:
        raw = empty_raw_paths(depth=1, return_geometry=return_geometry)
        raw["metadata"] = {"n_paths": 0, "runtime_backend": backend_metadata}
        return raw

    idx = dr.arange(wt.UInt32, count)
    rx_i32 = dr.gather(wt.Int32, wt.Int32(native.rx_id), idx)
    rx_index = wt.UInt32(dr.maximum(rx_i32, wt.Int32(0)))
    edge_idx = dr.gather(wt.Int32, wt.Int32(native.edge0), idx)
    native_point_0 = wt.Point3f(native.p0)
    point_0 = dr.gather(wt.Point3f, native_point_0, idx)
    rx_point = wt.Point3f(
        dr.gather(wt.Float, rx_positions.x, rx_index),
        dr.gather(wt.Float, rx_positions.y, rx_index),
        dr.gather(wt.Float, rx_positions.z, rx_index),
    )
    departure_dir = point_0 - tx_pos
    arrival_dir = rx_point - point_0
    theta_t, phi_t = spherical_angles(departure_dir)
    theta_r, phi_r = spherical_angles(arrival_dir)
    field_x = wt.Complex2f(native.field_x)
    a = wt.Complex2f(
        dr.gather(wt.Float, wt.Float(field_x.real), idx),
        dr.gather(wt.Float, wt.Float(field_x.imag), idx),
    )
    tau = dr.gather(wt.Float, wt.Float(native.delay), idx)
    type_slots = (dr.full(wt.Int32, wt.InteractionType.DIFFRACTION, count),)
    vertex_slots = normal_slots = object_slots = None
    if return_geometry:
        vertex_slots = (point_0,)
        if edge_data is None or int(edge_data.get("n_edges", 0)) <= 0:
            normal_slots = (arrays.zeros_vector3(count),)
            object_slots = (dr.full(wt.Int32, -1, count),)
        else:
            n_edges = int(edge_data["n_edges"])
            valid_edge = (edge_idx >= wt.Int32(0)) & (edge_idx < wt.Int32(n_edges))
            safe_edge = wt.UInt32(dr.select(valid_edge, edge_idx, wt.Int32(0)))
            normal_slots = (
                dr.select(
                    valid_edge,
                    dr.gather(wt.Vector3f, edge_data["n0"], safe_edge),
                    arrays.zeros_vector3(count),
                ),
            )
            object_slots = (
                _edge_object_indices_for_native_paths(scene, edge_data, edge_idx),
            )

    return finalize_raw_paths(
        rx_index=rx_index,
        a=a,
        tau=tau,
        theta_t=theta_t,
        phi_t=phi_t,
        theta_r=theta_r,
        phi_r=phi_r,
        type_slots=type_slots,
        vertex_slots=vertex_slots,
        normal_slots=normal_slots,
        object_slots=object_slots,
        metadata={"n_paths": count, "runtime_backend": backend_metadata},
    )


def _resolved_endpoint_arrays(scene: Scene, endpoints: list, *, role: str) -> list:
    resolver = scene.transmitter_array if role == "transmitter" else scene.receiver_array
    return [resolver(endpoint) for endpoint in endpoints]


def _uniform_array_ant_count(arrays: list, *, role: str) -> int:
    counts = [int(array.num_ant) for array in arrays]
    if not counts:
        return 1
    if any(count != counts[0] for count in counts[1:]):
        raise ValueError(f"path {role} arrays in one solve must share num_ant for rectangular Result tensors.")
    return counts[0]


def _array_has_single_origin_element(array) -> bool:
    positions = getattr(array, "element_positions", None)
    if positions is None:
        return True
    try:
        if int(dr.width(positions.x)) == 1:
            return all(
                abs(scalar(component)) <= 1.0e-12
                for component in (positions.x, positions.y, positions.z)
            )
        return False
    except Exception:
        pass
    try:
        rows = positions.tolist() if hasattr(positions, "tolist") else positions
        if len(rows) != 1:
            return False
        return all(abs(float(value)) <= 1.0e-12 for value in rows[0])
    except Exception:
        return False


def _array_has_synthetic_response(array) -> bool:
    if int(array.num_ant) != 1:
        return True
    if str(getattr(array, "pattern", "iso")) != "iso":
        return True
    if getattr(array, "element_orientations", None) is not None:
        return True
    return not _array_has_single_origin_element(array)


def _synthetic_array_expansion_required(tx_arrays: list, rx_arrays: list) -> bool:
    return any(_array_has_synthetic_response(array) for array in (*tx_arrays, *rx_arrays))


def _endpoint_orientation_tensor(endpoints: list, *, device, dtype) -> torch.Tensor:
    rows = [
        tuple(float(value) for value in (endpoint.orientation or (0.0, 0.0, 0.0)))
        for endpoint in endpoints
    ]
    return torch.tensor(rows, device=device, dtype=dtype)


def _euler_local_to_world_matrix(angles: torch.Tensor) -> torch.Tensor:
    yaw, pitch, roll = angles.unbind(dim=-1)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    return torch.stack(
        [
            torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], dim=-1),
            torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], dim=-1),
            torch.stack([-sp, cp * sr, cp * cr], dim=-1),
        ],
        dim=-2,
    )


def _rotate_local_to_world_tensor(vectors: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    matrix = _euler_local_to_world_matrix(angles)
    return torch.matmul(matrix, vectors.unsqueeze(-1)).squeeze(-1)


def _rotate_world_to_local_tensor(vectors: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    matrix = _euler_local_to_world_matrix(angles).transpose(-1, -2)
    return torch.matmul(matrix, vectors.unsqueeze(-1)).squeeze(-1)


def _array_slot_positions_tensor(arrays: list, endpoints: list, *, device, dtype) -> torch.Tensor:
    slots = []
    endpoint_orientations = _endpoint_orientation_tensor(endpoints, device=device, dtype=dtype)
    for endpoint_index, array in enumerate(arrays):
        positions = torch.stack(
            [
                to_torch_view(array.element_positions.x, dtype=torch.float32, device=device),
                to_torch_view(array.element_positions.y, dtype=torch.float32, device=device),
                to_torch_view(array.element_positions.z, dtype=torch.float32, device=device),
            ],
            dim=-1,
        ).to(device=device, dtype=dtype)
        repeated = positions.repeat_interleave(int(array.num_polarization_slots), dim=0)
        orientation = endpoint_orientations[endpoint_index].reshape(1, 3)
        slots.append(_rotate_local_to_world_tensor(repeated, orientation))
    return torch.stack(slots, dim=0)


def _array_slot_orientations_tensor(arrays: list, endpoints: list, *, device, dtype) -> torch.Tensor:
    slots = []
    endpoint_orientations = _endpoint_orientation_tensor(endpoints, device=device, dtype=dtype)
    for endpoint_index, array in enumerate(arrays):
        if array.element_orientations is None:
            element_orientations = torch.zeros((int(array.num_elements), 3), device=device, dtype=dtype)
        else:
            element_orientations = torch.stack(
                [
                    to_torch_view(array.element_orientations.x, dtype=torch.float32, device=device),
                    to_torch_view(array.element_orientations.y, dtype=torch.float32, device=device),
                    to_torch_view(array.element_orientations.z, dtype=torch.float32, device=device),
                ],
                dim=-1,
            ).to(device=device, dtype=dtype)
        repeated = element_orientations.repeat_interleave(int(array.num_polarization_slots), dim=0)
        slots.append(repeated + endpoint_orientations[endpoint_index].reshape(1, 3))
    return torch.stack(slots, dim=0)


def _pattern_gain(pattern: str, local_direction: torch.Tensor) -> torch.Tensor:
    if pattern == "iso":
        return torch.ones(local_direction.shape[:-1], device=local_direction.device, dtype=local_direction.dtype)
    x = local_direction[..., 0]
    y = local_direction[..., 1]
    z = torch.clamp(local_direction[..., 2], min=-1.0, max=1.0)
    theta = torch.acos(z)
    if pattern == "dipole":
        return torch.sqrt(torch.full_like(theta, 1.5)) * torch.sin(theta).clamp_min(0.0)
    if pattern == "tr38901":
        phi = torch.atan2(y, x)
        theta_3db = torch.full_like(theta, 65.0 * torch.pi / 180.0)
        phi_3db = torch.full_like(theta, 65.0 * torch.pi / 180.0)
        attenuation_v_db = torch.minimum(
            12.0 * torch.square((theta - 0.5 * torch.pi) / theta_3db),
            torch.full_like(theta, 30.0),
        )
        attenuation_h_db = torch.minimum(
            12.0 * torch.square(phi / phi_3db),
            torch.full_like(theta, 30.0),
        )
        attenuation_db = torch.minimum(
            attenuation_v_db + attenuation_h_db,
            torch.full_like(theta, 30.0),
        )
        gain_db = 8.0 - attenuation_db
        return torch.pow(torch.full_like(gain_db, 10.0), gain_db / 20.0)
    raise ValueError(f"Unsupported antenna pattern {pattern!r}.")


def _pattern_gains(
    *,
    arrays: list,
    endpoints: list,
    directions: torch.Tensor,
    role: str,
) -> torch.Tensor:
    device = directions.device
    dtype = directions.dtype
    orientations = _array_slot_orientations_tensor(arrays, endpoints, device=device, dtype=dtype)
    if role == "transmitter":
        local = _rotate_world_to_local_tensor(directions.unsqueeze(2), orientations.unsqueeze(0).unsqueeze(3))
        parts = [
            _pattern_gain(array.pattern, local[:, index, :, :, :])
            for index, array in enumerate(arrays)
        ]
        return torch.stack(parts, dim=1)
    if role == "receiver":
        rx_directions = -directions
        local = _rotate_world_to_local_tensor(rx_directions.unsqueeze(1), orientations.unsqueeze(2).unsqueeze(3))
        parts = [
            _pattern_gain(array.pattern, local[index, :, :, :, :])
            for index, array in enumerate(arrays)
        ]
        return torch.stack(parts, dim=0)
    raise ValueError(f"Unknown antenna pattern role {role!r}.")


def _pattern_gains_explicit_transmitter(*, arrays: list, endpoints: list, directions: torch.Tensor) -> torch.Tensor:
    device = directions.device
    dtype = directions.dtype
    orientations = _array_slot_orientations_tensor(arrays, endpoints, device=device, dtype=dtype)
    local = _rotate_world_to_local_tensor(
        directions,
        orientations.unsqueeze(0).unsqueeze(0).unsqueeze(4),
    )
    parts = [
        _pattern_gain(array.pattern, local[:, :, index, :, :, :])
        for index, array in enumerate(arrays)
    ]
    return torch.stack(parts, dim=2)


def _pattern_gains_explicit_receiver(*, arrays: list, endpoints: list, directions: torch.Tensor) -> torch.Tensor:
    device = directions.device
    dtype = directions.dtype
    orientations = _array_slot_orientations_tensor(arrays, endpoints, device=device, dtype=dtype)
    local = _rotate_world_to_local_tensor(
        -directions,
        orientations.unsqueeze(2).unsqueeze(3).unsqueeze(4),
    )
    parts = [
        _pattern_gain(array.pattern, local[index, :, :, :, :, :])
        for index, array in enumerate(arrays)
    ]
    return torch.stack(parts, dim=0)


def _rotate_point3f_euler(points: wt.Point3f, orientation: wt.Point3f) -> wt.Point3f:
    yaw, pitch, roll = orientation.x, orientation.y, orientation.z
    cy, sy = dr.cos(yaw), dr.sin(yaw)
    cp, sp = dr.cos(pitch), dr.sin(pitch)
    cr, sr = dr.cos(roll), dr.sin(roll)

    x1 = points.x
    y1 = cr * points.y - sr * points.z
    z1 = sr * points.y + cr * points.z
    x2 = cp * x1 + sp * z1
    y2 = y1
    z2 = -sp * x1 + cp * z1
    return wt.Point3f(cy * x2 - sy * y2, sy * x2 + cy * y2, z2)


def _endpoint_orientation_point3f(endpoint, *, role: str) -> wt.Point3f:
    return to_point3f(endpoint.orientation or (0.0, 0.0, 0.0), role=f"{role}.orientation")


def _array_slot_positions_point3f(array) -> wt.Point3f:
    positions = array.element_positions
    pol_count = int(array.num_polarization_slots)
    if pol_count == 1:
        return positions
    return wt.Point3f(
        dr.repeat(positions.x, pol_count),
        dr.repeat(positions.y, pol_count),
        dr.repeat(positions.z, pol_count),
    )


def _concat_float_arrays(parts: list) -> wt.Float:
    if len(parts) == 1:
        return parts[0]
    return dr.concat(parts)


def _expanded_endpoint_positions(endpoints: list, arrays: list, *, role: str) -> wt.Point3f:
    x_parts, y_parts, z_parts = [], [], []
    for endpoint, array in zip(endpoints, arrays):
        base = to_point3f(endpoint.position, role=f"{role}.position")
        slots = _rotate_point3f_euler(
            _array_slot_positions_point3f(array),
            _endpoint_orientation_point3f(endpoint, role=role),
        )
        width = int(array.num_ant)
        x_parts.append(dr.repeat(base.x, width) + slots.x)
        y_parts.append(dr.repeat(base.y, width) + slots.y)
        z_parts.append(dr.repeat(base.z, width) + slots.z)
    return wt.Point3f(
        _concat_float_arrays(x_parts),
        _concat_float_arrays(y_parts),
        _concat_float_arrays(z_parts),
    )


def _direction_from_angles(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    sin_theta = torch.sin(theta)
    return torch.stack(
        [
            sin_theta * torch.cos(phi),
            sin_theta * torch.sin(phi),
            torch.cos(theta),
        ],
        dim=-1,
    )


def _expand_payload_for_arrays(
    payload: dict[str, object],
    *,
    tx_arrays: list,
    rx_arrays: list,
    tx_endpoints: list,
    rx_endpoints: list,
    wavelength: float,
) -> dict[str, object]:
    num_tx_ant = _uniform_array_ant_count(tx_arrays, role="transmitter")
    num_rx_ant = _uniform_array_ant_count(rx_arrays, role="receiver")
    payload["num_tx_ant"] = num_tx_ant
    payload["num_rx_ant"] = num_rx_ant
    payload["num_time_steps"] = 1
    a = payload["a"]
    tau = payload["tau"]
    theta_t = payload["theta_t"]
    phi_t = payload["phi_t"]
    theta_r = payload["theta_r"]
    phi_r = payload["phi_r"]
    valid = payload["valid"]
    types = payload["types"]
    device = a.device
    real_dtype = tau.dtype
    tx_positions = _array_slot_positions_tensor(tx_arrays, tx_endpoints, device=device, dtype=real_dtype)
    rx_positions = _array_slot_positions_tensor(rx_arrays, rx_endpoints, device=device, dtype=real_dtype)

    departure = _direction_from_angles(theta_t, phi_t)
    arrival = _direction_from_angles(theta_r, phi_r)
    tx_dot = (departure.unsqueeze(2) * tx_positions.unsqueeze(0).unsqueeze(3)).sum(dim=-1)
    rx_dot = (arrival.unsqueeze(1) * rx_positions.unsqueeze(2).unsqueeze(3)).sum(dim=-1)
    phase = torch.exp(
        1j
        * (2.0 * torch.pi / float(wavelength))
        * (tx_dot.unsqueeze(1) - rx_dot.unsqueeze(3))
    )
    tx_gain = _pattern_gains(
        arrays=tx_arrays,
        endpoints=tx_endpoints,
        directions=departure,
        role="transmitter",
    )
    rx_gain = _pattern_gains(
        arrays=rx_arrays,
        endpoints=rx_endpoints,
        directions=arrival,
        role="receiver",
    )
    gain = rx_gain.unsqueeze(3) * tx_gain.unsqueeze(1)

    payload["a"] = (a.unsqueeze(1).unsqueeze(3) * phase * gain).unsqueeze(-1).contiguous()
    payload["tau"] = tau.unsqueeze(1).unsqueeze(3).expand(-1, num_rx_ant, -1, num_tx_ant, -1).contiguous()
    payload["theta_t"] = theta_t.unsqueeze(1).unsqueeze(3).expand(-1, num_rx_ant, -1, num_tx_ant, -1).contiguous()
    payload["phi_t"] = phi_t.unsqueeze(1).unsqueeze(3).expand(-1, num_rx_ant, -1, num_tx_ant, -1).contiguous()
    payload["theta_r"] = theta_r.unsqueeze(1).unsqueeze(3).expand(-1, num_rx_ant, -1, num_tx_ant, -1).contiguous()
    payload["phi_r"] = phi_r.unsqueeze(1).unsqueeze(3).expand(-1, num_rx_ant, -1, num_tx_ant, -1).contiguous()
    payload["valid"] = valid.unsqueeze(1).unsqueeze(3).expand(-1, num_rx_ant, -1, num_tx_ant, -1).contiguous()
    payload["types"] = types.unsqueeze(1).unsqueeze(3).expand(-1, num_rx_ant, -1, num_tx_ant, -1, -1).contiguous()
    payload["num_paths"] = payload["num_paths"].unsqueeze(1).unsqueeze(3).expand(-1, num_rx_ant, -1, num_tx_ant).contiguous()
    if payload.get("vertices") is not None:
        payload["vertices"] = payload["vertices"].unsqueeze(1).unsqueeze(3).expand(-1, num_rx_ant, -1, num_tx_ant, -1, -1, -1).contiguous()
    if payload.get("normals") is not None:
        payload["normals"] = payload["normals"].unsqueeze(1).unsqueeze(3).expand(-1, num_rx_ant, -1, num_tx_ant, -1, -1, -1).contiguous()
    if payload.get("objects") is not None:
        payload["objects"] = payload["objects"].unsqueeze(1).unsqueeze(3).expand(-1, num_rx_ant, -1, num_tx_ant, -1, -1).contiguous()
    return payload


def _reshape_explicit_payload_for_arrays(
    payload: dict[str, object],
    *,
    base_num_rx: int,
    base_num_tx: int,
    num_rx_ant: int,
    num_tx_ant: int,
    tx_arrays: list,
    rx_arrays: list,
    tx_endpoints: list,
    rx_endpoints: list,
    tx_positions,
    rx_positions,
) -> dict[str, object]:
    payload["num_rx"] = int(base_num_rx)
    payload["num_tx"] = int(base_num_tx)
    payload["num_rx_ant"] = int(num_rx_ant)
    payload["num_tx_ant"] = int(num_tx_ant)
    payload["num_time_steps"] = 1
    payload["tx_positions"] = tx_positions
    payload["rx_positions"] = rx_positions
    payload["a"] = payload["a"].reshape(base_num_rx, num_rx_ant, base_num_tx, num_tx_ant, -1).unsqueeze(-1).contiguous()
    payload["tau"] = payload["tau"].reshape(base_num_rx, num_rx_ant, base_num_tx, num_tx_ant, -1).contiguous()
    payload["theta_t"] = payload["theta_t"].reshape(base_num_rx, num_rx_ant, base_num_tx, num_tx_ant, -1).contiguous()
    payload["phi_t"] = payload["phi_t"].reshape(base_num_rx, num_rx_ant, base_num_tx, num_tx_ant, -1).contiguous()
    payload["theta_r"] = payload["theta_r"].reshape(base_num_rx, num_rx_ant, base_num_tx, num_tx_ant, -1).contiguous()
    payload["phi_r"] = payload["phi_r"].reshape(base_num_rx, num_rx_ant, base_num_tx, num_tx_ant, -1).contiguous()
    departure = _direction_from_angles(payload["theta_t"], payload["phi_t"])
    arrival = _direction_from_angles(payload["theta_r"], payload["phi_r"])
    tx_gain = _pattern_gains_explicit_transmitter(
        arrays=tx_arrays,
        endpoints=tx_endpoints,
        directions=departure,
    )
    rx_gain = _pattern_gains_explicit_receiver(
        arrays=rx_arrays,
        endpoints=rx_endpoints,
        directions=arrival,
    )
    payload["a"] = (payload["a"] * rx_gain.unsqueeze(-1) * tx_gain.unsqueeze(-1)).contiguous()
    payload["valid"] = payload["valid"].reshape(base_num_rx, num_rx_ant, base_num_tx, num_tx_ant, -1).contiguous()
    payload["types"] = payload["types"].reshape(base_num_rx, num_rx_ant, base_num_tx, num_tx_ant, payload["types"].shape[-2], payload["types"].shape[-1]).contiguous()
    payload["num_paths"] = payload["num_paths"].reshape(base_num_rx, num_rx_ant, base_num_tx, num_tx_ant).contiguous()
    if payload.get("vertices") is not None:
        payload["vertices"] = payload["vertices"].reshape(base_num_rx, num_rx_ant, base_num_tx, num_tx_ant, payload["vertices"].shape[-3], payload["vertices"].shape[-2], payload["vertices"].shape[-1]).contiguous()
    if payload.get("normals") is not None:
        payload["normals"] = payload["normals"].reshape(base_num_rx, num_rx_ant, base_num_tx, num_tx_ant, payload["normals"].shape[-3], payload["normals"].shape[-2], payload["normals"].shape[-1]).contiguous()
    if payload.get("objects") is not None:
        payload["objects"] = payload["objects"].reshape(base_num_rx, num_rx_ant, base_num_tx, num_tx_ant, payload["objects"].shape[-2], payload["objects"].shape[-1]).contiguous()
    return payload


def solve(
    *,
    scene: Scene,
    transmitter,
    receiver,
    config: Config | None = None,
) -> PathResult:
    """Solve discrete propagation paths for transmitter and receiver sets."""

    config = Config() if config is None else config
    if not isinstance(config, Config):
        raise TypeError("config must be a witwin.channel.path.Config or None.")
    frequency = scene.frequency
    if frequency is None:
        raise ValueError("paths.solve requires Scene.frequency.")
    tx_positions, tx_name, tx_polarization, tx_endpoints = _normalize_transmitters(
        scene=scene, transmitter=transmitter,
    )
    rx_positions, receiver_name, rx_polarization, rx_endpoints = _normalize_receivers(
        scene=scene, receiver=receiver,
    )
    tx_arrays = _resolved_endpoint_arrays(scene, tx_endpoints, role="transmitter")
    rx_arrays = _resolved_endpoint_arrays(scene, rx_endpoints, role="receiver")
    num_tx_ant = _uniform_array_ant_count(tx_arrays, role="transmitter")
    num_rx_ant = _uniform_array_ant_count(rx_arrays, role="receiver")
    trace_tx_positions = tx_positions
    trace_rx_positions = rx_positions
    if not config.synthetic_array:
        trace_tx_positions = _expanded_endpoint_positions(tx_endpoints, tx_arrays, role="transmitter")
        trace_rx_positions = _expanded_endpoint_positions(rx_endpoints, rx_arrays, role="receiver")
    wavelength, k = derive_wave_params(frequency=frequency)
    solver_controls = resolve_solver_controls(
        config, max_diffractions_override=int(config.max_diffraction_order),
    )
    num_rx = int(dr.width(rx_positions.x))
    num_tx = int(dr.width(tx_positions.x))
    trace_num_rx = int(dr.width(trace_rx_positions.x))
    trace_num_tx = int(dr.width(trace_tx_positions.x))
    effective = solver_controls["effective"]
    reflection_backend = _resolve_reflection_path_backend(
        config=config,
        scene=scene,
        tx_positions=trace_tx_positions,
        rx_positions=trace_rx_positions,
    )

    assert_scene_materials_complete(scene)
    scene.diffraction_edge_count(edge_policy=config.edge_policy)

    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    los_raw = collect_los_paths(
        scene=scene,
        rx_positions=trace_rx_positions,
        tx_pos=trace_tx_positions,
        wavelength=wavelength,
        k=k,
        tx_polarization=tx_polarization,
        rx_polarization=rx_polarization,
    )
    timing["los"] = time.perf_counter() - t0

    tx_runtimes = []
    reflection_raw_collections: list[Mapping[str, object]] = []
    reflection_details = []
    t0 = time.perf_counter()
    for tx_index in range(trace_num_tx):
        tx_position = _gather_tx_position(trace_tx_positions, tx_index)
        runtime = TraceContext.from_config(
            tx_pos=tx_position,
            config=SimpleNamespace(
                tx_polarization=tx_polarization, rx_polarization=rx_polarization,
            ),
            rx_positions=trace_rx_positions,
            wavelength=wavelength,
            k=k,
        )
        tx_runtimes.append(runtime)
    reflection_raw_collections, reflection_details = collect_reflection_paths_for_transmitters(
        scene=scene,
        rx_positions=trace_rx_positions,
        tx_positions=trace_tx_positions,
        wavelength=wavelength,
        k=k,
        n_rays=effective["reflection_n_rays"],
        max_reflections=effective["reflection_max_bounces"],
        mode="3d",
        tx_polarization=tx_polarization,
        rx_polarization=rx_polarization,
        min_ray_contribution_threshold=config.tuning.min_ray_contribution_threshold,
        use_scene_materials=True,
        return_geometry=config.return_geometry,
        reflection_details=None,
        prefer_rayd_epc=bool(reflection_backend["prefer_rayd_epc"]),
        require_rayd_epc=bool(reflection_backend["require_rayd_epc"]),
    )
    reflection_raw_collections = list(reflection_raw_collections)
    reflection_details = list(reflection_details)
    timing["reflection"] = time.perf_counter() - t0

    diffraction_raw_collections: list[Mapping[str, object]] = []
    diffraction_group_metadata: list[dict] = []
    diffraction_runtime_backend: dict[str, object] | None = None
    t0 = time.perf_counter()
    if effective["max_diffractions"] > 0:
        diffraction_accumulate_mode = _resolve_diffraction_path_accumulate_primal(
            config=config,
            scene=scene,
            tx_positions=trace_tx_positions,
            rx_positions=trace_rx_positions,
            max_order=int(effective["max_diffractions"]),
        )
        use_native_diffraction_paths = diffraction_accumulate_mode == "rayd_optix"
        if use_native_diffraction_paths and int(effective["max_diffractions"]) != 1:
            raise RuntimeError(
                "path diffraction_execution.accumulate_primal='rayd_optix' currently supports "
                "first-order diffraction path export only."
            )
        if use_native_diffraction_paths:
            diffraction_runtime_backend = _rayd_diffraction_path_backend_metadata()
        else:
            diffraction_runtime_backend = _drjit_diffraction_path_backend_metadata(
                int(effective["max_diffractions"])
            )
        spec = PathSolveSpec(ray_mode=DEFAULT_RAY_MODE, name=receiver_name)
        diffraction_config = SimpleNamespace(
            enable_rd_diffraction=bool(config.tuning.enable_rd_diffraction),
            rx_polarization=rx_polarization,
            diffraction_execution=config.tuning.diffraction_execution,
            shadow_support_cutoff_db=config.tuning.shadow_support_cutoff_db,
        )
        for tx_index, runtime in enumerate(tx_runtimes):
            raw_state_collections = trace_diffraction_raw_collections(
                runtime=runtime,
                scene=scene,
                config=diffraction_config,
                solver_controls=solver_controls,
                spec=spec,
                reflection_detail=reflection_details[tx_index],
                state_layout=PATH_EXPORT_REDUCED_STATE_LAYOUT,
            )
            for state_raw in raw_state_collections:
                path_collection_stats: dict = {}
                if use_native_diffraction_paths:
                    state_arrays_for_limit = state_raw.get("state_arrays")
                    state_count_for_limit = (
                        0
                        if state_arrays_for_limit is None
                        else int(state_arrays_for_limit.get("n_states", 0))
                    )
                    raw = collect_native_diffraction_paths(
                        state_arrays=state_raw.get("state_arrays"),
                        edge_data=state_raw.get("edge_data"),
                        scene=scene,
                        rx_positions=state_raw.get("rx_positions"),
                        tx_pos=runtime.tx.position,
                        wavelength=wavelength,
                        k=k,
                        max_paths=int(config.max_num_paths or max(1, state_count_for_limit)),
                        seed=0,
                        return_geometry=config.return_geometry,
                        stats=path_collection_stats,
                    )
                else:
                    raw = collect_diffraction_state_paths(
                        state_arrays=state_raw.get("state_arrays"),
                        edge_data=state_raw.get("edge_data"),
                        scene=scene,
                        rx_positions=state_raw.get("rx_positions"),
                        tx_pos=runtime.tx.position,
                        wavelength=wavelength,
                        k=k,
                        tx_polarization=tx_polarization,
                        rx_polarization=rx_polarization,
                        material_detail=None,
                        return_geometry=config.return_geometry,
                        stats=path_collection_stats,
                    )
                _stamp_raw(raw, tx_index=tx_index, receiver_index_map=state_raw["receiver_index_map"])
                diffraction_raw_collections.append(raw)
                state_arrays = state_raw.get("state_arrays")
                rx_group = state_raw.get("rx_positions")
                diffraction_group_metadata.append({
                    "tx_index": tx_index,
                    "receiver_count": int(dr.width(rx_group.x)) if rx_group is not None else 0,
                    "grouping_axis": "z",
                    "grouping_rule": "shared_z_height_slice",
                    "state_layout": path_export_state_layout(state_arrays) or "full",
                    "n_edge_states": 0 if state_arrays is None else int(state_arrays["n_states"]),
                    "n_paths": int(raw["metadata"].get("n_paths", 0)),
                    "path_collection": path_collection_stats,
                })
    timing["diffraction"] = time.perf_counter() - t0

    first_tx_position = _gather_tx_position(tx_positions, 0)
    reflection_detail_for_metadata = next((d for d in reflection_details if d is not None), None)
    active_rx_polarization = effective_rx_polarization(rx_polarization, tx_polarization)
    metadata: dict[str, object] = {
        "package": "witwin.channel.path",
        "transmitter_sampling": {
            "name": tx_name,
            "kind": "path",
            "transmitter_count": num_tx,
            "trace_transmitter_count": trace_num_tx,
            "positions_shape": (num_tx, 3),
        },
        "receiver_sampling": {
            "name": receiver_name,
            "kind": "path",
            "receiver_count": num_rx,
            "trace_receiver_count": trace_num_rx,
            "positions_shape": (num_rx, 3),
            "ray_mode": DEFAULT_RAY_MODE,
            "max_num_paths_requested": config.max_num_paths,
            "return_geometry": config.return_geometry,
        },
        "polarization_transport": {
            "enabled": True,
            "tx_polarization": tx_polarization,
            "rx_polarization": active_rx_polarization,
            "rx_polarization_source": (
                "explicit" if rx_polarization is not None else "default_from_tx_polarization"
            ),
            "result_basis": "receiver_polarization_projected_to_arrival_ray",
        },
        "solver_mode": solver_controls,
        "array_model": {
            "synthetic_array": bool(config.synthetic_array),
            "assumption": "synthetic_array=True traces endpoint centers and assumes far-field plane-wave phase across the aperture; use explicit mode for near-field or large apertures.",
            "num_tx_ant": num_tx_ant,
            "num_rx_ant": num_rx_ant,
            "tx_patterns": tuple(array.pattern for array in tx_arrays),
            "rx_patterns": tuple(array.pattern for array in rx_arrays),
        },
        "execution_intent": dict(solver_controls["execution_intent"]),
        "interaction_type_codes": {
            "none": int(InteractionType.NONE),
            "reflection": int(InteractionType.REFLECTION),
            "diffraction": int(InteractionType.DIFFRACTION),
            "transmission_reserved": int(InteractionType.TRANSMISSION),
            "scattering_reserved": int(InteractionType.SCATTERING),
        },
        "angle_convention": {
            "theta_reference": "zenith_from_+z",
            "phi_reference": "azimuth_from_+x_toward_+y",
            "aod_direction": "tx_to_first_interaction_or_rx",
            "aoa_direction": "last_interaction_to_rx",
        },
        "path_counts": {
            "los": int(los_raw["metadata"].get("n_paths", 0)),
            "reflection": int(sum(r["metadata"].get("n_paths", 0) for r in reflection_raw_collections)),
            "diffraction": int(sum(r["metadata"].get("n_paths", 0) for r in diffraction_raw_collections)),
        },
        "diffraction_groups": tuple(diffraction_group_metadata),
    }
    runtime_backends = {
        "reflection": {
            "field_backend_requested": str(reflection_backend["requested"]),
            "epc_backend": str(reflection_backend["epc_backend"]),
            "rayd_epc_preferred": bool(reflection_backend["prefer_rayd_epc"]),
            "rayd_epc_required": bool(reflection_backend["rayd_epc_required"]),
            "ad_inputs": bool(reflection_backend["ad_inputs"]),
        },
    }
    if diffraction_runtime_backend is not None:
        runtime_backends["diffraction"] = dict(diffraction_runtime_backend)
    metadata["runtime_backends"] = runtime_backends
    if reflection_detail_for_metadata is not None:
        metadata["reflection_sampling"] = dict(
            getattr(reflection_detail_for_metadata, "reflection_sampling", {}) or {}
        )
    metadata["timing"] = dict(timing)

    payload = assemble_result_payload(
        name=receiver_name,
        num_tx=trace_num_tx,
        num_rx=trace_num_rx,
        max_num_paths=config.max_num_paths,
        tx_pos=(scalar(first_tx_position.x), scalar(first_tx_position.y), scalar(first_tx_position.z)),
        tx_positions=trace_tx_positions,
        rx_positions=trace_rx_positions,
        frequency=float(frequency),
        wavelength=wavelength,
        raw_collections=[los_raw, *reflection_raw_collections, *diffraction_raw_collections],
        return_geometry=config.return_geometry,
        metadata=metadata,
    )
    if config.synthetic_array:
        payload["num_tx"] = num_tx
        payload["num_rx"] = num_rx
        payload["tx_positions"] = tx_positions
        payload["rx_positions"] = rx_positions
        payload["num_tx_ant"] = num_tx_ant
        payload["num_rx_ant"] = num_rx_ant
        payload["num_time_steps"] = 1
        if _synthetic_array_expansion_required(tx_arrays, rx_arrays):
            payload = _expand_payload_for_arrays(
                payload,
                tx_arrays=tx_arrays,
                rx_arrays=rx_arrays,
                tx_endpoints=tx_endpoints,
                rx_endpoints=rx_endpoints,
                wavelength=wavelength,
            )
    else:
        payload = _reshape_explicit_payload_for_arrays(
            payload,
            base_num_rx=num_rx,
            base_num_tx=num_tx,
            num_rx_ant=num_rx_ant,
            num_tx_ant=num_tx_ant,
            tx_arrays=tx_arrays,
            rx_arrays=rx_arrays,
            tx_endpoints=tx_endpoints,
            rx_endpoints=rx_endpoints,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
        )
    return PathResult._from_payload(payload)


__all__ = ["solve"]
