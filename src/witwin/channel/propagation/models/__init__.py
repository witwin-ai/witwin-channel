"""Internal row-aligned propagation data contracts."""

from .capacity import (
    CanonicalEvaluatedPaths,
    CanonicalPathSelection,
    CapacityEvaluatedPaths,
    CapacityPathLayout,
    CapacityPathSelection,
)
from .coupled import CoupledCandidateCapacity
from .contracts import TopologyConfig
from .evaluated import EvaluatedPaths
from .fields import PathFields
from .geometry import PathGeometry
from .penetration import (
    SegmentPenetrationBackwardResult,
    SegmentPenetrationJvpResult,
    SegmentPenetrationPolicy,
    SegmentPenetrationResult,
    SegmentPenetrationTapeResult,
)
from .reflection import ReflectionCandidateCapacity
from .topology import PathTopology
from .transmission import TransmissionTopologyCapacity

__all__ = [
    "CanonicalEvaluatedPaths",
    "CanonicalPathSelection",
    "CapacityEvaluatedPaths",
    "CapacityPathLayout",
    "CapacityPathSelection",
    "CoupledCandidateCapacity",
    "EvaluatedPaths",
    "PathFields",
    "PathGeometry",
    "PathTopology",
    "ReflectionCandidateCapacity",
    "SegmentPenetrationBackwardResult",
    "SegmentPenetrationJvpResult",
    "SegmentPenetrationPolicy",
    "SegmentPenetrationResult",
    "SegmentPenetrationTapeResult",
    "TopologyConfig",
    "TransmissionTopologyCapacity",
]
