# Copyright Xingyu Chen.
# Endpoint geometry, antenna response, and the scene-leaf AD seam.

"""Endpoint geometry, antenna response, and the scene-leaf AD seam."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch

from witwin.core import (
    AntennaPattern as _CoreAntennaPattern,
    AntennaState,
    ReceiverGrid as CoreReceiverGrid,
    Scene,
    SceneSnapshot,
    antenna_orientation_matrix,
    quat_from_euler,
    quat_multiply,
)

if TYPE_CHECKING:
    from witwin.channel.scene.compiler import CompiledScene


_C0 = 299_792_458.0


def _vector3(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    value = value.to(dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def orientation_matrix(orientation: torch.Tensor) -> torch.Tensor:
    """Return the local-to-world yaw/pitch/roll rotation matrix.

 The orientation vector is ``(yaw, pitch, roll)`` in radians and uses the
 intrinsic Z-Y-X convention.
 """

    if orientation.shape == (4,):
        from witwin.core.math import quat_to_rotation_matrix

        return quat_to_rotation_matrix(orientation).to(dtype=torch.float32)
    yaw, pitch, roll = _vector3("orientation", orientation).unbind()
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    return torch.stack(
        (
            torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr)),
            torch.stack((sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr)),
            torch.stack((-sp, cp * sr, cp * cr)),
        )
    ).to(dtype=torch.float32)


def pattern_field_response(
    pattern: _CoreAntennaPattern,
    local_direction: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one canonical Core antenna pattern in the endpoint-local frame."""

    if not isinstance(pattern, _CoreAntennaPattern):
        raise TypeError("pattern must be a witwin.core.AntennaPattern")
    if local_direction.shape[-1] != 3:
        raise ValueError("local_direction must have a vec3 tail")
    direction = local_direction.to(dtype=torch.float32)
    direction = direction / torch.linalg.vector_norm(
        direction, dim=-1, keepdim=True
    ).clamp_min(torch.finfo(torch.float32).tiny)
    if pattern.kind == "custom":
        assert pattern.custom is not None
        response = pattern.custom(direction)
        if response.shape != direction.shape[:-1]:
            raise ValueError("custom pattern response must match direction batch shape")
        return response.to(device=direction.device, dtype=torch.complex64)
    if pattern.kind == "vertical":
        response = torch.sqrt(
            torch.clamp(1.0 - direction[..., 2].square(), min=0.0)
        )
    elif pattern.kind == "horizontal":
        response = torch.sqrt(
            torch.clamp(1.0 - direction[..., 0].square(), min=0.0)
        )
    else:
        response = torch.ones(direction.shape[:-1], device=direction.device)
    return response.to(dtype=torch.complex64)


def steering_vector(
    array: object,
    direction: torch.Tensor,
    *,
    frequency_hz: float,
    orientation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``exp(+j k r·u)`` under the package's ``exp(-j k d)`` convention."""

    if isinstance(frequency_hz, torch.Tensor):
        # Synthetic-array steering is evaluated at the primal frequency; its
        # frequency derivative is exactly zero for single-element centre
        # arrays and detached otherwise (material and frequency derivatives fixed-array contract).
        frequency_hz = float(frequency_hz.detach())
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    if direction.shape[-1] != 3:
        raise ValueError("direction must have a vec3 tail")
    device = direction.device
    positions = array.positions.to(device=device)
    if orientation is not None:
        rotation = orientation_matrix(orientation).to(device=device)
        positions = positions @ rotation.T
    unit = direction.to(dtype=torch.float32)
    unit = unit / torch.linalg.vector_norm(unit, dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    phase = (2.0 * math.pi * float(frequency_hz) / _C0) * torch.einsum(
        "...c,ac->...a", unit, positions
    )
    return torch.exp(1.0j * phase).to(torch.complex64)


def apply_precoding_combining(
    coefficients: torch.Tensor,
    *,
    tx_weights: torch.Tensor,
    rx_weights: torch.Tensor,
) -> torch.Tensor:
    """Combine ``(..., rx_ant, tx_ant)`` channel coefficients into one stream."""

    if coefficients.ndim < 2:
        raise ValueError("coefficients must have rx_ant and tx_ant tail dimensions")
    if tx_weights.shape != (coefficients.shape[-1],):
        raise ValueError("tx_weights must match tx_ant")
    if rx_weights.shape != (coefficients.shape[-2],):
        raise ValueError("rx_weights must match rx_ant")
    tx = tx_weights.to(device=coefficients.device, dtype=torch.complex64)
    rx = rx_weights.to(device=coefficients.device, dtype=torch.complex64)
    return torch.einsum("...rt,t,r->...", coefficients, tx, rx.conj())


def apply_endpoint_weights(
    coefficients: torch.Tensor,
    *,
    tx_weights: torch.Tensor,
    rx_weights: torch.Tensor,
) -> torch.Tensor:
    """Combine ``(rx, rx_ant, tx, tx_ant, ...)`` endpoint channels.

 The leading endpoint and antenna dimensions match:class:`PathResult`.
 Any trailing signal dimensions (path, time, frequency, or tap) are
 preserved. Receiver weights follow the usual conjugating convention.
 """

    if coefficients.ndim < 4:
        raise ValueError(
            "coefficients must have (rx, rx_ant, tx, tx_ant, ...) dimensions"
        )
    expected_tx = (coefficients.shape[2], coefficients.shape[3])
    expected_rx = (coefficients.shape[0], coefficients.shape[1])
    if tx_weights.shape != expected_tx:
        raise ValueError(f"tx_weights must have shape {expected_tx}")
    if rx_weights.shape != expected_rx:
        raise ValueError(f"rx_weights must have shape {expected_rx}")
    tx = tx_weights.to(device=coefficients.device, dtype=torch.complex64)
    rx = rx_weights.to(device=coefficients.device, dtype=torch.complex64)
    tail = (1,) * (coefficients.ndim - 4)
    weighted = coefficients * tx.reshape(1, 1, *expected_tx, *tail)
    weighted = weighted * rx.conj().reshape(*expected_rx, 1, 1, *tail)
    return weighted.sum(dim=3).sum(dim=1)


def validate_scalar_endpoint_features(
    transmitters: Sequence[object],
    receivers: Sequence[object],
    *,
    solver: str,
) -> None:
    """Reject endpoint features that a scalar/power solver cannot consume."""

    for endpoint in (*tuple(transmitters), *tuple(receivers)):
        if endpoint.array.num_antennas != 1:
            raise ValueError(f"{solver} does not support antenna arrays")
        if endpoint.pattern.kind != "isotropic":
            raise ValueError(
                f"{solver} does not support directional antenna patterns"
            )
        weights = (
            endpoint.precoding
            if hasattr(endpoint, "precoding")
            else endpoint.combining
        )
        if weights is not None:
            raise ValueError(f"{solver} does not support precoding or combining")


@dataclass(frozen=True, slots=True)
class _AntennaArrayView:
    positions: torch.Tensor

    @property
    def num_antennas(self) -> int:
        return int(self.positions.shape[0])

    def world_positions(
        self, origin: torch.Tensor, orientation: torch.Tensor
    ) -> torch.Tensor:
        rotation = antenna_orientation_matrix(
            orientation, reference=self.positions
        )
        return origin.to(device=self.positions.device) + self.positions @ rotation.T


@dataclass(frozen=True, slots=True)
class _EndpointView:
    source: AntennaState
    position_override: torch.Tensor | None = None
    orientation_override: torch.Tensor | None = None
    single_element: bool = False

    @property
    def position(self) -> torch.Tensor:
        return (
            self.source.position
            if self.position_override is None
            else self.position_override
        )

    @property
    def orientation(self) -> torch.Tensor:
        orientation = (
            self.source.orientation
            if self.orientation_override is None
            else self.orientation_override
        )
        if orientation is None:
            return self.position.new_zeros((3,))
        return orientation

    @property
    def polarization(self) -> torch.Tensor:
        polarization = self.source.polarization
        if polarization is None:
            default = (
                (1.0, 0.0, 0.0)
                if self.source.pattern.kind == "horizontal"
                else (0.0, 0.0, 1.0)
            )
            polarization = self.position.new_tensor(default)
        norm = torch.linalg.vector_norm(polarization)
        if polarization.device.type == "cpu" and (
            not bool(torch.isfinite(norm)) or float(norm) <= 0.0
        ):
            raise ValueError("polarization must be finite and non-zero")
        rotation = antenna_orientation_matrix(
            self.orientation, reference=polarization
        )
        return (rotation @ (polarization / norm)).contiguous()

    @property
    def pattern(self):
        return self.source.pattern

    @property
    def array(self) -> _AntennaArrayView:
        positions = self.source.element_positions
        if positions is None or self.single_element:
            positions = self.position.new_zeros((1, 3))
        return _AntennaArrayView(positions)

    @property
    def synthetic_array(self) -> bool:
        return self.source.synthetic_array


@dataclass(frozen=True, slots=True)
class _TransmitterView(_EndpointView):
    @property
    def power_w(self):
        return 1.0 if self.source.power_w is None else self.source.power_w

    @property
    def precoding(self) -> torch.Tensor | None:
        return self.source.weights


@dataclass(frozen=True, slots=True)
class _ReceiverPointView(_EndpointView):
    @property
    def combining(self) -> torch.Tensor | None:
        return self.source.weights


@dataclass(frozen=True, slots=True)
class _ReceiverGridView(_EndpointView):
    source: CoreReceiverGrid
    x_axis_override: torch.Tensor | None = None
    y_axis_override: torch.Tensor | None = None

    @property
    def origin(self) -> torch.Tensor:
        return self.position

    @property
    def x_axis(self) -> torch.Tensor:
        return (
            self.source.x_axis
            if self.x_axis_override is None
            else self.x_axis_override
        )

    @property
    def y_axis(self) -> torch.Tensor:
        return (
            self.source.y_axis
            if self.y_axis_override is None
            else self.y_axis_override
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.source.shape

    @property
    def spacing(self) -> tuple[float, float]:
        spacing = self.source.spacing
        return (float(spacing[0]), float(spacing[1]))

    @property
    def combining(self) -> torch.Tensor | None:
        return self.source.weights


@dataclass(frozen=True, slots=True)
class SolverScene:
    """Internal solver binding over one immutable ``CompiledScene``."""

    compiled: CompiledScene
    structures: tuple[object, ...]
    transmitters: tuple[_TransmitterView, ...]
    receivers: tuple[_ReceiverPointView | _ReceiverGridView, ...]
    frequency: float | torch.Tensor
    metadata: object


def _as_quaternion(orientation: torch.Tensor) -> torch.Tensor:
    if orientation.shape == (4,):
        return orientation
    yaw, pitch, roll = orientation.unbind()
    return quat_from_euler(
        roll,
        pitch,
        yaw,
        device=orientation.device,
        dtype=orientation.dtype,
    )


def _endpoint_views(
    scene_or_snapshot: Scene | SceneSnapshot,
) -> tuple[_EndpointView, ...]:
    if isinstance(scene_or_snapshot, Scene):
        states = tuple((endpoint, None) for endpoint in scene_or_snapshot.endpoints)
    else:
        states = tuple(
            (state.antenna, state.rigid_motion)
            for state in scene_or_snapshot.endpoints
        )
    views: list[_EndpointView] = []
    for endpoint, motion in states:
        position = endpoint.position
        orientation = endpoint.orientation
        grid_x_axis = endpoint.x_axis if isinstance(endpoint, CoreReceiverGrid) else None
        grid_y_axis = endpoint.y_axis if isinstance(endpoint, CoreReceiverGrid) else None
        if motion is not None:
            if motion.translation is not None:
                position = position + motion.translation
            if motion.rotation is not None:
                motion_rotation = _as_quaternion(motion.rotation)
                motion_matrix = antenna_orientation_matrix(
                    motion.rotation,
                    reference=position,
                )
                if orientation is None:
                    orientation = motion_rotation
                else:
                    orientation = quat_multiply(
                        motion_rotation, _as_quaternion(orientation)
                    )
                if grid_x_axis is not None:
                    grid_x_axis = motion_matrix @ grid_x_axis
                    grid_y_axis = motion_matrix @ grid_y_axis
        if endpoint.role == "tx":
            views.append(
                _TransmitterView(
                    endpoint,
                    position_override=position,
                    orientation_override=orientation,
                )
            )
        elif isinstance(endpoint, CoreReceiverGrid):
            views.append(
                _ReceiverGridView(
                    endpoint,
                    position_override=position,
                    orientation_override=orientation,
                    x_axis_override=grid_x_axis,
                    y_axis_override=grid_y_axis,
                )
            )
        else:
            views.append(
                _ReceiverPointView(
                    endpoint,
                    position_override=position,
                    orientation_override=orientation,
                )
            )
    return tuple(views)


def _require_host_constant(name: str, value: object) -> None:
    if not isinstance(value, torch.Tensor):
        return
    if value.device.type != "cpu" or value.requires_grad:
        raise RuntimeError(
            f"{name} requires the tensor-native endpoint ABI; the current "
            "native scalar boundary accepts only constant CPU tensors"
        )


def _validate_scalar_endpoint_boundary(views: tuple[_EndpointView, ...]) -> None:
    """Reject only grid leaves still consumed by the scalar grid ABI."""

    for index, view in enumerate(views):
        if isinstance(view, _ReceiverGridView):
            prefix = f"endpoint[{index}]"
            _require_host_constant(f"{prefix}.position", view.position)
            _require_host_constant(f"{prefix}.x_axis", view.x_axis)
            _require_host_constant(f"{prefix}.y_axis", view.y_axis)
            _require_host_constant(f"{prefix}.spacing", view.source.spacing)


def bind_solver_scene(compiled: CompiledScene) -> SolverScene:
    views = _endpoint_views(compiled.source)
    _validate_scalar_endpoint_boundary(views)
    return SolverScene(
        compiled=compiled,
        structures=compiled.structures,
        transmitters=tuple(
            view for view in views if isinstance(view, _TransmitterView)
        ),
        receivers=tuple(
            view for view in views if not isinstance(view, _TransmitterView)
        ),
        frequency=compiled.reference_frequency_hz,
        metadata=compiled.source.metadata,
    )


def require_compiled(scene: SolverScene | CompiledScene) -> CompiledScene:
    # ``compiler`` imports this module for its endpoint tensor exports, so the
    # compiled-scene type is resolved here rather than at module scope. That
    # keeps the scene lifetime one-way at import time: an importer of the
    # endpoint views does not drag in the compiler and everything it compiles
    # against.
    from witwin.channel.scene.compiler import CompiledScene

    if isinstance(scene, SolverScene):
        return scene.compiled
    if isinstance(scene, CompiledScene):
        return scene
    raise TypeError("expected a Channel SolverScene or CompiledScene")


# Internal names used by solver-domain modules. They are intentionally absent
# from the package root.
ReceiverGrid = _ReceiverGridView
ReceiverPoint = _ReceiverPointView
Transmitter = _TransmitterView


@dataclass(frozen=True, slots=True)
class AxisAlignedGridSpec:
    grid: ReceiverGrid
    axis: int
    position: float
    coord0_min: float
    coord0_max: float
    coord1_min: float
    coord1_max: float
    resolution0: int
    resolution1: int
    cell_area: float


def vector3_tuple(value: torch.Tensor) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def first_receiver_grid(scene: object) -> ReceiverGrid | None:
    for receiver in scene.receivers:
        if isinstance(receiver, ReceiverGrid):
            return receiver
    return None


def component_grid_shape(grid: ReceiverGrid) -> tuple[int, int]:
    return (grid.shape[1], grid.shape[0])


def _axis_index(
    values: tuple[float, float, float], *, name: str
) -> tuple[int, float]:
    nonzero = [idx for idx, value in enumerate(values) if abs(value) > 1.0e-6]
    if len(nonzero) != 1:
        raise ValueError(f"{name} must be axis-aligned")
    index = nonzero[0]
    value = values[index]
    sign = 1.0 if value > 0.0 else -1.0
    if abs(abs(value) - 1.0) > 1.0e-5:
        raise ValueError(f"{name} must be a unit axis vector")
    return index, sign


def axis_aligned_grid_spec(grid: ReceiverGrid) -> AxisAlignedGridSpec:
    rows, cols = grid.shape
    origin = vector3_tuple(grid.origin)
    axis0, sign0 = _axis_index(
        vector3_tuple(grid.x_axis), name="ReceiverGrid.x_axis"
    )
    axis1, sign1 = _axis_index(
        vector3_tuple(grid.y_axis), name="ReceiverGrid.y_axis"
    )
    if axis0 == axis1:
        raise ValueError("ReceiverGrid axes must be orthogonal")
    axis = ({0, 1, 2} - {axis0, axis1}).pop()
    expected = (1, 2) if axis == 0 else (0, 2) if axis == 1 else (0, 1)
    if (axis0, axis1) != expected:
        raise ValueError("ReceiverGrid axes must match RayD grid coordinate order")

    step0 = float(grid.spacing[0]) * sign0
    step1 = float(grid.spacing[1]) * sign1
    first0 = origin[axis0]
    first1 = origin[axis1]
    last0 = first0 + step0 * float(rows - 1)
    last1 = first1 + step1 * float(cols - 1)
    half0 = abs(float(grid.spacing[0])) * 0.5
    half1 = abs(float(grid.spacing[1])) * 0.5
    coord0_min = min(first0, last0) - half0
    coord0_max = max(first0, last0) + half0
    coord1_min = min(first1, last1) - half1
    coord1_max = max(first1, last1) + half1
    return AxisAlignedGridSpec(
        grid=grid,
        axis=axis,
        position=origin[axis],
        coord0_min=coord0_min,
        coord0_max=coord0_max,
        coord1_min=coord1_min,
        coord1_max=coord1_max,
        resolution0=rows,
        resolution1=cols,
        cell_area=abs((coord0_max - coord0_min) * (coord1_max - coord1_min))
        / float(rows * cols),
    )


def scene_vertex_table(scene: object, compiled: object) -> torch.Tensor:
    """Live global vertex table matching ``compiled.geometry.vertices``.

 RayD concatenates structure meshes in scene order, so the live table is
 the concatenation of the structure vertex tensors. Returning the live
 tensors (rather than the native export) is what lets mesh-vertex
 gradients exist at all.
 """

    native = compiled.geometry.vertices
    if not scene.structures:
        return native
    vertices = torch.cat(
        [
            structure.vertices.to(device=native.device, dtype=torch.float32)
            for structure in scene.structures
        ],
        dim=0,
    )
    if vertices.shape != native.shape:
        raise RuntimeError(
            "differentiable vertex table does not match the native scene "
            f"table: {tuple(vertices.shape)} vs {tuple(native.shape)}"
        )
    return vertices


def transmitter_positions_ad(
    scene: object, native: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    """Live transmitter positions (the native builder flattens to host floats)."""

    if not scene.transmitters:
        return native
    return torch.stack(
        [
            transmitter.position.to(device=device, dtype=torch.float32)
            for transmitter in scene.transmitters
        ],
        dim=0,
    )


def receiver_positions_ad(
    scene: object, native: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    """Live receiver positions for point receivers.

 Grid receivers are generated natively from origin/axes/spacing and stay
 detached: a grid exposes no per-receiver position tensor for a user to
 mark requires_grad, so nothing is silently zeroed here.
 """

    if not scene.receivers or not all(
        isinstance(receiver, ReceiverPoint) for receiver in scene.receivers
    ):
        return native
    return torch.stack(
        [
            receiver.position.to(device=device, dtype=torch.float32)
            for receiver in scene.receivers
        ],
        dim=0,
    )


def transmitter_polarizations_f32(
    scene: object, *, device: torch.device
) -> torch.Tensor:
    """Transmitter polarizations as a contiguous float32 ``(N, 3)`` tensor.

 Row order matches the transmitter order of the logical scene. The vectors
 are already unit and oriented by the Core transmitter model, so this only
 stacks them, casts to float32, and makes the result contiguous.

 This is NOT:func:`witwin.channel.scene.compiler.transmitter_polarizations_as_stored`,
 which is a straight device upload: it keeps whatever dtype and layout the
 scene stored, and its empty case comes from the native transmitter builder
 rather than ``device``. Both are live and each has its own callers; the two
 names record the difference instead of hiding it behind one spelling.
 """

    values = [tx.polarization for tx in scene.transmitters]
    if not values:
        return torch.empty((0, 3), device=device, dtype=torch.float32)
    return torch.stack(values).to(device=device, dtype=torch.float32).contiguous()


def receiver_polarizations_f32(
    scene: object,
    *,
    device: torch.device,
    grid: ReceiverGrid | None = None,
) -> torch.Tensor:
    """Receiver polarizations as a contiguous float32 ``(N, 3)`` tensor.

 With ``grid``, one grid polarization is broadcast over that grid's points.
 Without it, the scene's receivers are expanded in order: a grid contributes
 one row per point, a point receiver one row.
 """

    if grid is not None:
        return (
            grid.polarization.to(device=device, dtype=torch.float32)
            .expand(grid.shape[0] * grid.shape[1], 3)
            .contiguous()
        )
    values: list[torch.Tensor] = []
    for receiver in scene.receivers:
        if isinstance(receiver, ReceiverGrid):
            values.extend([receiver.polarization] * (receiver.shape[0] * receiver.shape[1]))
        elif isinstance(receiver, ReceiverPoint):
            values.append(receiver.polarization)
        else:
            raise TypeError(f"receiver type is not accepted: {type(receiver)!r}")
    if not values:
        return torch.empty((0, 3), device=device, dtype=torch.float32)
    return torch.stack(values).to(device=device, dtype=torch.float32).contiguous()


__all__: list[str] = []