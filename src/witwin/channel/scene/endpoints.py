from __future__ import annotations

from dataclasses import dataclass

import torch

from witwin.core import (
    AntennaState,
    ReceiverGrid as CoreReceiverGrid,
    Scene,
    SceneSnapshot,
    antenna_orientation_matrix,
    quat_from_euler,
    quat_multiply,
)

from witwin.channel.scene.compiled import CompiledScene


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

__all__: list[str] = []
