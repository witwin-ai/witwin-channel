"""Internal row-aligned propagation data contracts."""

from .contracts import EvaluatedRowsSource, TopologyConfig
from .evaluated import EvaluatedPaths
from .fields import PathFields
from .geometry import PathGeometry
from .topology import PathTopology

__all__ = [
    "EvaluatedPaths",
    "EvaluatedRowsSource",
    "PathFields",
    "PathGeometry",
    "PathTopology",
    "TopologyConfig",
]
