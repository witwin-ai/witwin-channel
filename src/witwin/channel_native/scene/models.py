from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol

import torch

from witwin.channel_native.core.antenna import AntennaArray, AntennaPattern, orientation_matrix

if TYPE_CHECKING:
    from witwin.channel_native.core.edge_policy import EdgePolicy
    from witwin.channel_native.scene.compiled import CompiledScene
    from witwin.channel_native.scene.kernels.rayd_scene import RayDSceneResource


class Material(Protocol):
    def parameters(self, frequency_hz: float | None = None) -> dict[str, float | int | str]:
        ...


def _as_vector3(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return tensor.to(dtype=torch.float32).contiguous()


def _as_polarization(
    value: torch.Tensor | None,
    *,
    pattern: AntennaPattern,
    orientation: torch.Tensor,
) -> torch.Tensor:
    default = [1.0, 0.0, 0.0] if pattern.kind == "horizontal" else [0.0, 0.0, 1.0]
    polarization = (
        torch.tensor(default, dtype=torch.float32)
        if value is None
        else _as_vector3("polarization", value)
    )
    norm = torch.linalg.vector_norm(polarization)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        raise ValueError("polarization must be finite and non-zero")
    rotation = orientation_matrix(orientation).to(device=polarization.device)
    return (rotation @ (polarization / norm)).contiguous()


def _as_orientation(value: torch.Tensor | None) -> torch.Tensor:
    return (
        torch.zeros(3, dtype=torch.float32)
        if value is None
        else _as_vector3("orientation", value)
    )


def _as_pattern(value: AntennaPattern | str) -> AntennaPattern:
    return AntennaPattern(value) if isinstance(value, str) else value


def _as_array(value: AntennaArray | None) -> AntennaArray:
    return AntennaArray.single() if value is None else value


def _as_weights(
    name: str, value: torch.Tensor | None, *, num_antennas: int
) -> torch.Tensor | None:
    if value is None:
        return None
    if value.shape != (num_antennas,):
        raise ValueError(f"{name} must have shape ({num_antennas},)")
    value = value.to(dtype=torch.complex64).contiguous()
    if not bool(torch.isfinite(value.real).all() & torch.isfinite(value.imag).all()):
        raise ValueError(f"{name} must be finite")
    return value


def planar_uv(
    vertices: torch.Tensor,
    axis_u: torch.Tensor,
    axis_v: torch.Tensor,
    origin: torch.Tensor | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Planar UV generation: project vertices onto two in-plane axes.

    ``uv[i] = scale * ((vertices[i] - origin) . axis_u,
    (vertices[i] - origin) . axis_v)``. The axes are used as given (not
    normalized) so callers control the metric-to-UV mapping; ``origin``
    defaults to the world origin. Returns float32 ``(N, 2)``. Intended for
    rectangle/box test structures whose faces share one plane per axis pair.
    """

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    axis_u = _as_vector3("axis_u", axis_u)
    axis_v = _as_vector3("axis_v", axis_v)
    base = vertices.to(dtype=torch.float32)
    if origin is not None:
        base = base - _as_vector3("origin", origin).to(device=base.device)
    uv = torch.stack(
        (
            base @ axis_u.to(device=base.device),
            base @ axis_v.to(device=base.device),
        ),
        dim=1,
    )
    return (float(scale) * uv).contiguous()


@dataclass(frozen=True, slots=True)
class Structure:
    vertices: torch.Tensor
    faces: torch.Tensor
    material: Material
    name: str = ""
    surface_id: int = 0
    metadata: dict[str, object] | None = None
    # Optional UV parametrization for phase-screen height sampling. UV
    # vertices are indexed by face_uv (RayD mesh layout), so their count is
    # independent of the position vertex count. Both must be given together.
    uv: torch.Tensor | None = None
    face_uv: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        object.__setattr__(self, "vertices", self.vertices.to(dtype=torch.float32).contiguous())
        object.__setattr__(self, "faces", self.faces.to(dtype=torch.int32).contiguous())
        if (self.uv is None) != (self.face_uv is None):
            raise ValueError("uv and face_uv must be provided together")
        if self.uv is not None:
            if self.uv.ndim != 2 or self.uv.shape[1] != 2:
                raise ValueError("uv must have shape (V, 2)")
            if self.face_uv.ndim != 2 or self.face_uv.shape[1] != 3:
                raise ValueError("face_uv must have shape (F, 3)")
            if self.face_uv.shape[0] != self.faces.shape[0]:
                raise ValueError("face_uv must have one row per face")
            uv = self.uv.to(dtype=torch.float32).contiguous()
            face_uv = self.face_uv.to(dtype=torch.int32).contiguous()
            if face_uv.numel() and (
                int(face_uv.min()) < 0 or int(face_uv.max()) >= uv.shape[0]
            ):
                raise ValueError("face_uv indices must be in [0, uv rows)")
            object.__setattr__(self, "uv", uv)
            object.__setattr__(self, "face_uv", face_uv)

    def with_vertices(self, vertices: torch.Tensor) -> Structure:
        return replace(self, vertices=vertices)

    def with_material(self, material: Material) -> Structure:
        return replace(self, material=material)


@dataclass(frozen=True, slots=True)
class Transmitter:
    position: torch.Tensor
    power_w: float = 1.0
    polarization: torch.Tensor | None = None
    orientation: torch.Tensor | None = None
    pattern: AntennaPattern | str = "isotropic"
    array: AntennaArray | None = None
    synthetic_array: bool = True
    precoding: torch.Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _as_vector3("position", self.position))
        orientation = _as_orientation(self.orientation)
        pattern = _as_pattern(self.pattern)
        array = _as_array(self.array)
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "pattern", pattern)
        object.__setattr__(self, "array", array)
        object.__setattr__(
            self,
            "polarization",
            _as_polarization(
                self.polarization, pattern=pattern, orientation=orientation
            ),
        )
        object.__setattr__(
            self,
            "precoding",
            _as_weights("precoding", self.precoding, num_antennas=array.num_antennas),
        )


@dataclass(frozen=True, slots=True)
class ReceiverPoint:
    position: torch.Tensor
    polarization: torch.Tensor | None = None
    orientation: torch.Tensor | None = None
    pattern: AntennaPattern | str = "isotropic"
    array: AntennaArray | None = None
    synthetic_array: bool = True
    combining: torch.Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _as_vector3("position", self.position))
        orientation = _as_orientation(self.orientation)
        pattern = _as_pattern(self.pattern)
        array = _as_array(self.array)
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "pattern", pattern)
        object.__setattr__(self, "array", array)
        object.__setattr__(
            self,
            "polarization",
            _as_polarization(
                self.polarization, pattern=pattern, orientation=orientation
            ),
        )
        object.__setattr__(
            self,
            "combining",
            _as_weights("combining", self.combining, num_antennas=array.num_antennas),
        )


@dataclass(frozen=True, slots=True)
class ReceiverGrid:
    origin: torch.Tensor
    x_axis: torch.Tensor
    y_axis: torch.Tensor
    shape: tuple[int, int]
    spacing: tuple[float, float]
    polarization: torch.Tensor | None = None
    orientation: torch.Tensor | None = None
    pattern: AntennaPattern | str = "isotropic"
    array: AntennaArray | None = None
    synthetic_array: bool = True
    combining: torch.Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", _as_vector3("origin", self.origin))
        object.__setattr__(self, "x_axis", _as_vector3("x_axis", self.x_axis))
        object.__setattr__(self, "y_axis", _as_vector3("y_axis", self.y_axis))
        orientation = _as_orientation(self.orientation)
        pattern = _as_pattern(self.pattern)
        array = _as_array(self.array)
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "pattern", pattern)
        object.__setattr__(self, "array", array)
        object.__setattr__(
            self,
            "polarization",
            _as_polarization(
                self.polarization, pattern=pattern, orientation=orientation
            ),
        )
        object.__setattr__(
            self,
            "combining",
            _as_weights("combining", self.combining, num_antennas=array.num_antennas),
        )
        if len(self.shape) != 2 or self.shape[0] <= 0 or self.shape[1] <= 0:
            raise ValueError("shape must be two positive integers")
        if len(self.spacing) != 2 or self.spacing[0] <= 0.0 or self.spacing[1] <= 0.0:
            raise ValueError("spacing must be two positive values")

    def points(self) -> torch.Tensor:
        rows, cols = self.shape
        origin = (float(self.origin[0]), float(self.origin[1]), float(self.origin[2]))
        x_axis = (float(self.x_axis[0]), float(self.x_axis[1]), float(self.x_axis[2]))
        y_axis = (float(self.y_axis[0]), float(self.y_axis[1]), float(self.y_axis[2]))
        points: list[tuple[float, float, float]] = []
        for i in range(rows):
            for j in range(cols):
                x_weight = i * self.spacing[0]
                y_weight = j * self.spacing[1]
                points.append(
                    (
                        origin[0] + x_axis[0] * x_weight + y_axis[0] * y_weight,
                        origin[1] + x_axis[1] * x_weight + y_axis[1] * y_weight,
                        origin[2] + x_axis[2] * x_weight + y_axis[2] * y_weight,
                    )
                )
        return torch.tensor(points, dtype=torch.float32)


Receiver = ReceiverPoint | ReceiverGrid

_RAYD_EDGE_INFO_PLANE_TOL = 1.34e-5


@dataclass(frozen=True, slots=True)
class Scene:
    structures: tuple[Structure, ...]
    transmitters: tuple[Transmitter, ...]
    receivers: tuple[Receiver, ...]
    frequency: float | torch.Tensor
    metadata: dict[str, object]
    _geometry_version: int = 0
    _material_version: int = 0
    _assignment_version: int = 0
    _compiled_cache: CompiledScene | None = field(
        default=None, init=False, compare=False, repr=False
    )
    _rayd_cache: RayDSceneResource | None = field(
        default=None, init=False, compare=False, repr=False
    )

    def __init__(
        self,
        *,
        structures: list[Structure] | tuple[Structure, ...],
        transmitters: list[Transmitter] | tuple[Transmitter, ...],
        receivers: list[Receiver] | tuple[Receiver, ...],
        frequency: float | torch.Tensor,
        metadata: dict[str, object] | None = None,
        _geometry_version: int = 0,
        _material_version: int = 0,
        _assignment_version: int = 0,
    ) -> None:
        # The carrier frequency may be a 0-d torch tensor so it can carry
        # requires_grad / forward-mode tangents into the differentiable field
        # kernels (plan 07 AD-1). Every non-AD consumer reads it through
        # float(scene.frequency), which detaches by contract.
        if isinstance(frequency, torch.Tensor):
            if frequency.ndim != 0:
                raise ValueError("tensor frequency must be a 0-d tensor")
            if float(frequency.detach()) <= 0.0:
                raise ValueError("frequency must be positive")
        else:
            if frequency <= 0.0:
                raise ValueError("frequency must be positive")
            frequency = float(frequency)
        object.__setattr__(self, "structures", tuple(structures))
        object.__setattr__(self, "transmitters", tuple(transmitters))
        object.__setattr__(self, "receivers", tuple(receivers))
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "metadata", dict(metadata or {}))
        object.__setattr__(self, "_geometry_version", _geometry_version)
        object.__setattr__(self, "_material_version", _material_version)
        object.__setattr__(self, "_assignment_version", _assignment_version)
        object.__setattr__(self, "_compiled_cache", None)
        object.__setattr__(self, "_rayd_cache", None)

    @classmethod
    def load_mitsuba(cls, filename: str, **kwargs) -> Scene:
        from witwin.channel_native.scene.loader import load_mitsuba

        return load_mitsuba(filename, scene_cls=cls, **kwargs)

    def add(self, obj: Transmitter | Receiver) -> Scene:
        if isinstance(obj, Transmitter):
            object.__setattr__(self, "transmitters", self.transmitters + (obj,))
            return self
        if isinstance(obj, (ReceiverPoint, ReceiverGrid)):
            object.__setattr__(self, "receivers", self.receivers + (obj,))
            return self
        raise TypeError(f"scene object type is not accepted: {type(obj).__name__}")

    def with_structure_vertices(self, index: int, vertices: torch.Tensor) -> Scene:
        structures = list(self.structures)
        structures[index] = structures[index].with_vertices(vertices)
        return replace(
            self,
            structures=tuple(structures),
            _geometry_version=self._geometry_version + 1,
        )

    def with_structure_material(self, index: int, material: object) -> Scene:
        structures = list(self.structures)
        structures[index] = structures[index].with_material(material)  # type: ignore[arg-type]
        return replace(
            self,
            structures=tuple(structures),
            _material_version=self._material_version + 1,
        )

    def with_frequency(self, frequency: float) -> Scene:
        return replace(
            self, frequency=frequency, _material_version=self._material_version + 1
        )

    def diffraction_edge_count(self, edge_policy: EdgePolicy | None = None) -> int:
        from witwin.channel_native.core.edge_policy import DEFAULT_EDGE_POLICY

        policy = DEFAULT_EDGE_POLICY if edge_policy is None else edge_policy
        if not policy.edge_diffraction:
            return 0
        if len(self.structures) == 0:
            return 0
        rayd_scene = self.rayd_scene()
        if not rayd_scene.available:
            raise RuntimeError(
                "diffraction edge counting requires RayD native scene capability"
            )
        return _diffraction_edge_count_from_rayd_scene(rayd_scene, policy)

    @property
    def n_diffraction_edges(self) -> int:
        return self.diffraction_edge_count()

    def rayd_scene(self) -> RayDSceneResource:
        from witwin.channel_native.core.edge_selection import resolve_scene_edge_policy
        from witwin.channel_native.scene.kernels.rayd_scene import (
            build_scene_from_structures,
        )

        cached = self._rayd_cache
        if cached is not None:
            return cached
        rayd = build_scene_from_structures(self.structures)
        # Expose the scene's edge policy to the diffraction edge-geometry
        # builders so path generation honors it (audit DF-4).
        rayd.runtime_cache["edge_policy"] = resolve_scene_edge_policy(self)
        object.__setattr__(self, "_rayd_cache", rayd)
        return rayd

    def compile(self) -> CompiledScene:
        from witwin.channel_native.scene.compile import compile_scene

        return compile_scene(self)


def _diffraction_edge_count_from_rayd_scene(
    rayd_scene: RayDSceneResource, edge_policy: EdgePolicy
) -> int:
    from witwin.channel_native.propagation.geometry.kernels import (
        primitives as geometry_primitives,
    )

    records = rayd_scene.edge_records()
    return geometry_primitives.core_diffraction_edge_count(
        vertices=records.vertices,
        faces=records.faces,
        face_normals=records.face_normals,
        edge_v0=records.edge_v0,
        edge_v1=records.edge_v1,
        face0=records.face0,
        face1=records.face1,
        vertical_only=edge_policy.vertical_only,
        vertical_ratio=float(edge_policy.vertical_ratio),
        boundary_half_plane=edge_policy.boundary_edge_policy == "half_plane",
        plane_tol=_RAYD_EDGE_INFO_PLANE_TOL,
    )


# Keep the long-standing public import and pickle paths stable while this
# module becomes the canonical implementation owner.
Material.__module__ = "witwin.channel_native.core.objects"
planar_uv.__module__ = "witwin.channel_native.core.objects"
Structure.__module__ = "witwin.channel_native.core.objects"
Transmitter.__module__ = "witwin.channel_native.core.objects"
ReceiverPoint.__module__ = "witwin.channel_native.core.objects"
ReceiverGrid.__module__ = "witwin.channel_native.core.objects"
Scene.__module__ = "witwin.channel_native.core.scene"
