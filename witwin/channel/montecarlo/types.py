from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import drjit.cuda.ad as cuda_ad


Float = cuda_ad.Float
UInt32 = cuda_ad.UInt32
Int32 = cuda_ad.Int32
Bool = cuda_ad.Bool
Point2f = cuda_ad.Array2f
Point3f = cuda_ad.Array3f
Vector2f = cuda_ad.Array2f
Vector3f = cuda_ad.Array3f
Vector3u = cuda_ad.Array3u
Complex2f = cuda_ad.Complex2f
Matrix4f = cuda_ad.Matrix4f


class InteractionType(enum.IntFlag):
    NONE = 0
    REFLECTION = 1
    DIFFRACTION = 2
    TRANSMISSION = 4
    SCATTERING = 8


class Integrator(ABC):
    """Base class for transport-family integrators."""

    mode: str

    @abstractmethod
    def integrate(self, *args, **kwargs):
        """Run the transport family and return a package result payload."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """Resolved batch sizes for reflection and diffraction phases."""
    ray_batch_size: int
    ray_batch_count: int
    ray_policy: str
    diffraction_batch_size: int
    diffraction_batch_count: int
    diffraction_policy: str
    free_cuda_bytes: int
    scatter_safe_batch_cap: int


@dataclass(slots=True)
class TraceTiming:
    """Accumulated timing for each solver phase."""
    los_seconds: float = 0.0
    reflection_seconds: float = 0.0
    diffraction_seconds: float = 0.0
    state_preparation_seconds: float = 0.0
    scatter_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass(slots=True)
class PathCounts:
    """Accepted hit counts per propagation component."""
    los: UInt32 = field(default_factory=lambda: UInt32(0))
    reflection: UInt32 = field(default_factory=lambda: UInt32(0))
    diffraction: UInt32 = field(default_factory=lambda: UInt32(0))


@dataclass(slots=True)
class ReflectionPhaseResult:
    """Store the reflection-phase outputs used by later solver stages."""
    path_counts: PathCounts
    path_tape_store: object | None
    diff_state_store: object | None


@dataclass(slots=True)
class DiffractionPhaseResult:
    """Store the diffraction-phase outputs used by metadata and AD replay."""
    runtime_reuse: dict[str, object]
    state_pool: dict[str, int]
    runtime_backend: dict[str, object]
    edge_indices: object | None
    diff_length_weight: float
    diffraction_tape_store: object | None
    batch_plan: BatchPlan


__all__ = [
    "BatchPlan",
    "Bool",
    "Complex2f",
    "DiffractionPhaseResult",
    "Float",
    "Int32",
    "Integrator",
    "InteractionType",
    "Matrix4f",
    "PathCounts",
    "Point2f",
    "Point3f",
    "ReflectionPhaseResult",
    "TraceTiming",
    "UInt32",
    "Vector2f",
    "Vector3f",
    "Vector3u",
]
