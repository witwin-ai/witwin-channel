"""Stable solver-neutral propagation consumer facade.

Call :func:`capabilities` to discover the supported components, responses,
topology modes, and AD modes before building a request. Build a
:class:`PropagationRequest` for discovery, or a :class:`FixedTopologyRequest`
to reevaluate an already-discovered topology, then pass it to :func:`evaluate`
or :func:`reevaluate` together with a compiled scene.

A whole frame, pulse train, or symbol block is one call rather than one call
per instant: declare ``slot_count`` on a :class:`FixedTopologyRequest` built
over :func:`replicate_over_slots`, or use
:func:`evaluate_time_varying`, which publishes the same rows at ``T`` instants
as a ``[T, K]`` time-varying impulse response.
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
    WorldProvenance,
    capabilities,
    prepare_fixed_topology,
    replicate_over_slots,
)
from .service import evaluate, rediscovery_required, reevaluate
from .time_varying import (
    TimeVaryingEvaluation,
    TimeVaryingRequest,
    TimeVaryingTransport,
    evaluate_time_varying,
)

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
    "TimeVaryingEvaluation",
    "TimeVaryingRequest",
    "TimeVaryingTransport",
    "WorldProvenance",
    "capabilities",
    "evaluate",
    "evaluate_time_varying",
    "prepare_fixed_topology",
    "rediscovery_required",
    "reevaluate",
    "replicate_over_slots",
]
