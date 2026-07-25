"""Stable solver-neutral propagation consumer facade.

Call :func:`capabilities` to discover the supported components, responses,
topology modes, and AD modes before building a request. Build a
:class:`PropagationRequest` for discovery, or a :class:`FixedTopologyRequest`
to reevaluate an already-discovered topology, then pass it to :func:`evaluate`
or :func:`reevaluate` together with a compiled scene.
"""

from .contracts import (
    AD_MODES,
    COMPONENTS,
    CONTRACT_VERSION,
    MAX_DEPTH,
    RESPONSES,
    TOPOLOGY_MODES,
    Complex3Transport,
    EndpointBatch,
    FixedTopologyBucket,
    FixedTopologyEvaluation,
    FixedTopologyRequest,
    JonesTransport,
    PreparedFixedTopology,
    PropagationAdMode,
    PropagationCapabilities,
    PropagationComponent,
    PropagationConvention,
    PropagationDiagnostics,
    PropagationEvaluation,
    PropagationGeometry,
    PropagationPathBatch,
    PropagationRequest,
    PropagationResponse,
    PropagationTopology,
    PropagationTopologyMode,
    ScalarTransport,
    capabilities,
    prepare_fixed_topology,
)
from .service import evaluate, reevaluate

__all__ = [
    "AD_MODES",
    "COMPONENTS",
    "CONTRACT_VERSION",
    "Complex3Transport",
    "EndpointBatch",
    "FixedTopologyBucket",
    "FixedTopologyEvaluation",
    "FixedTopologyRequest",
    "JonesTransport",
    "MAX_DEPTH",
    "PreparedFixedTopology",
    "PropagationAdMode",
    "PropagationCapabilities",
    "PropagationComponent",
    "PropagationConvention",
    "PropagationDiagnostics",
    "PropagationEvaluation",
    "PropagationGeometry",
    "PropagationPathBatch",
    "PropagationRequest",
    "PropagationResponse",
    "PropagationTopology",
    "PropagationTopologyMode",
    "RESPONSES",
    "ScalarTransport",
    "TOPOLOGY_MODES",
    "capabilities",
    "evaluate",
    "prepare_fixed_topology",
    "reevaluate",
]
