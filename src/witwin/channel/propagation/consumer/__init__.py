"""Stable solver-neutral propagation consumer façade."""

from .contracts import (
    CONTRACT_VERSION,
    Complex3Transport,
    EndpointBatch,
    FixedTopologyEvaluation,
    FixedTopologyRequest,
    JonesTransport,
    PropagationCapabilities,
    PropagationConvention,
    PropagationDiagnostics,
    PropagationEvaluation,
    PropagationGeometry,
    PropagationPathBatch,
    PropagationRequest,
    PropagationTopology,
    ScalarTransport,
)
from .service import evaluate, reevaluate

__all__ = [
    "CONTRACT_VERSION",
    "Complex3Transport",
    "EndpointBatch",
    "FixedTopologyEvaluation",
    "FixedTopologyRequest",
    "JonesTransport",
    "PropagationCapabilities",
    "PropagationConvention",
    "PropagationDiagnostics",
    "PropagationEvaluation",
    "PropagationGeometry",
    "PropagationPathBatch",
    "PropagationRequest",
    "PropagationTopology",
    "ScalarTransport",
    "evaluate",
    "reevaluate",
]
