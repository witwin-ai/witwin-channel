"""Solver-shared runtime bundles and scene helpers with duck-typed contracts.

These helpers reach through the Scene's private runtime caches
(``_merged_vertices()``, ``_triangle_runtime()``) via duck typing so this
package never imports ``witwin.channel.core.scene``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import drjit as dr
import torch
from witwin.channel import types as wt

from witwin.channel.core.numerics.arrays import scalar
from witwin.channel.core.numerics.constants import SPEED_OF_LIGHT
from witwin.channel.core.physics.materials import resolve_surface_material, scene_has_material_table
from witwin.channel.core.physics.wave_math import material_angular_frequency


_XYZ = ("x", "y", "z")


def to_point3f(value, *, role: str = "point", allow_single: bool = True) -> wt.Point3f:
    """Coerce ``value`` into ``wt.Point3f``.

    Accepts a ``wt.Point3f``, an object with ``.x/.y/.z`` attributes, a
    length-3 scalar sequence (when ``allow_single``), or a ``torch.Tensor``
    of shape ``(3,)`` (when ``allow_single``) or ``(N, 3)``.
    """
    if isinstance(value, wt.Point3f):
        if dr.width(value.x) == 0:
            raise ValueError(f"{role} must contain at least one point.")
        return value
    if all(hasattr(value, axis) for axis in _XYZ):
        return wt.Point3f(value.x, value.y, value.z)
    if isinstance(value, torch.Tensor):
        return _tensor_to_point3f(value, role=role, allow_single=allow_single)
    if isinstance(value, Sequence):
        if allow_single and len(value) == 3 and all(not hasattr(v, "__len__") for v in value):
            return wt.Point3f(value[0], value[1], value[2])
        if not value:
            raise ValueError(f"{role} sequence must not be empty.")
        return _tensor_to_point3f(
            torch.as_tensor(list(value), dtype=torch.float32),
            role=role,
            allow_single=allow_single,
        )
    raise TypeError(
        f"{role} must be a wt.Point3f, length-3 sequence, or torch tensor of shape (3,)/(N, 3)."
    )


def _tensor_to_point3f(tensor: torch.Tensor, *, role: str, allow_single: bool) -> wt.Point3f:
    if allow_single and tensor.ndim == 1 and tensor.shape[0] == 3:
        tensor = tensor.reshape(1, 3)
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        expected = "(3,) or (N, 3)" if allow_single else "(N, 3)"
        raise ValueError(f"{role} tensor must have shape {expected}.")
    if tensor.shape[0] == 0:
        raise ValueError(f"{role} must contain at least one point.")
    tensor = tensor.to(dtype=torch.float32).contiguous()
    return wt.Point3f(wt.Float(tensor[:, 0]), wt.Float(tensor[:, 1]), wt.Float(tensor[:, 2]))


def to_vector3f(value) -> wt.Vector3f:
    """Coerce a single 3-vector (Vector3f/duck-typed/length-3 sequence) into wt.Vector3f."""
    if all(hasattr(value, axis) for axis in _XYZ):
        return wt.Vector3f(value.x, value.y, value.z)
    if isinstance(value, Sequence) and len(value) == 3:
        return wt.Vector3f(value[0], value[1], value[2])
    raise TypeError("Vector must be a DrJit 3-vector or length-3 sequence.")


@dataclass(slots=True)
class Wave:
    """GPU-resident wavelength and wavenumber."""

    wavelength: object
    k: object | None = None

    def __post_init__(self) -> None:
        self.wavelength = wt.Float(self.wavelength)
        self.k = wt.Float(self.k) if self.k is not None else wt.Float(2.0 * math.pi) / self.wavelength

    @classmethod
    def from_frequency(cls, frequency: float) -> "Wave":
        wavelength = SPEED_OF_LIGHT / float(frequency)
        return cls(wavelength=wavelength, k=2.0 * math.pi / wavelength)

    @property
    def wavelength_scalar(self) -> float:
        return scalar(self.wavelength)

    @property
    def k_scalar(self) -> float:
        return scalar(self.k)


@dataclass(slots=True)
class Tx:
    """GPU-resident transmitter state derived from scene endpoints."""

    position: object
    polarization: object = (1.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        self.position = to_point3f(self.position, role="transmitter")
        self.polarization = to_vector3f(self.polarization)

    @property
    def polarization_tuple(self) -> tuple[float, float, float]:
        return (scalar(self.polarization.x), scalar(self.polarization.y), scalar(self.polarization.z))


@dataclass(slots=True)
class Rx:
    """GPU-resident receiver state derived from scene endpoints."""

    positions: object
    polarization: object | None = None

    def __post_init__(self) -> None:
        self.positions = to_point3f(self.positions, role="receiver")
        if self.polarization is not None:
            self.polarization = to_vector3f(self.polarization)

    def effective_polarization(self, tx: Tx):
        return tx.polarization if self.polarization is None else self.polarization


@dataclass(slots=True)
class Material:
    """GPU-resident scalar gain used around scene-owned material lookups."""

    reflection_coef: object = 1.0

    def __post_init__(self) -> None:
        self.reflection_coef = wt.Float(self.reflection_coef)

    @property
    def gain(self):
        return self.reflection_coef

    @property
    def gain_scalar(self) -> float:
        return scalar(self.reflection_coef)


@dataclass(frozen=True, slots=True)
class TraceCtx:
    """Per-sample bundle handed to LoS/reflection/diffraction trace stages."""

    scene: object
    runtime: "TraceContext"
    sample_grid: object
    config: object
    spec: object
    solver_controls: dict
    grad_preserving: bool
    n_rx: int


@dataclass(slots=True)
class TraceContext:
    """Runtime bundles that move through LoS/reflection/diffraction stages."""

    tx: Tx
    wave: Wave
    reflection: Material
    diffraction: Material
    rx: Rx | None = None

    @classmethod
    def from_config(
        cls,
        *,
        tx_pos,
        config,
        rx_positions=None,
        wavelength: float | None = None,
        k: float | None = None,
    ) -> "TraceContext":
        tx = Tx(position=tx_pos, polarization=getattr(config, "tx_polarization", (1.0, 0.0, 0.0)))
        resolved_wavelength = float(wavelength if wavelength is not None else getattr(config, "wavelength"))
        resolved_k = float(k if k is not None else getattr(config, "k"))
        wave = Wave(wavelength=resolved_wavelength, k=resolved_k)
        rx = None if rx_positions is None else Rx(
            positions=rx_positions,
            polarization=getattr(config, "rx_polarization", None),
        )
        return cls(tx=tx, wave=wave, reflection=Material(1.0), diffraction=Material(1.0), rx=rx)

    def with_rx(self, positions, polarization=None) -> "TraceContext":
        if polarization is not None:
            rx_polarization = polarization
        elif self.rx is None:
            rx_polarization = None
        else:
            rx_polarization = self.rx.polarization
        return TraceContext(
            tx=self.tx,
            wave=self.wave,
            reflection=self.reflection,
            diffraction=self.diffraction,
            rx=Rx(positions=positions, polarization=rx_polarization),
        )


def assert_scene_materials_complete(scene) -> None:
    """One-shot check that every triangle in the material table has an assigned material."""
    if scene is None:
        return
    tri_data = scene._triangle_runtime()
    if tri_data is None:
        return
    specified = tri_data.get("material_specified")
    if specified is None:
        raise RuntimeError(
            "Scene has a triangle material table but no per-triangle "
            "'material_specified' flag. Rebuild the scene."
        )
    if not bool(dr.all(specified)):
        raise RuntimeError(
            "Scene material table requires every triangle to have an "
            "assigned material (witwin.core.Material). Found triangles with no material."
        )


def point_grad_enabled(point) -> bool:
    """True if any axis of a Point3f/Vector3f carries gradient tracking."""
    if point is None:
        return False
    return any(dr.grad_enabled(getattr(point, axis)) for axis in _XYZ)


def scene_geometry_grad_enabled(scene) -> bool:
    if scene is None:
        return False
    tri_data = scene._triangle_runtime()
    merged_vertices = getattr(scene, "_merged_vertices", None)
    vertices = None if merged_vertices is None else merged_vertices()
    if point_grad_enabled(vertices):
        return True
    if tri_data is None:
        return False
    return any(
        point_grad_enabled(tri_data[key])
        for key in ("v0", "v1", "v2")
        if key in tri_data
    )


def scene_material_grad_enabled(scene) -> bool:
    if scene is None:
        return False
    tri_data = scene._triangle_runtime()
    if tri_data is None:
        return False
    return any(
        dr.grad_enabled(tri_data[key])
        for key in ("material_eps_r", "material_sigma_e")
        if tri_data.get(key) is not None
    )


__all__ = [
    "Material",
    "Rx",
    "TraceContext",
    "TraceCtx",
    "Tx",
    "Wave",
    "assert_scene_materials_complete",
    "material_angular_frequency",
    "point_grad_enabled",
    "resolve_surface_material",
    "scene_geometry_grad_enabled",
    "scene_has_material_table",
    "scene_material_grad_enabled",
    "to_point3f",
]
