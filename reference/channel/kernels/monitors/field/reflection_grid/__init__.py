"""Reflection grid accumulation backends."""

from .drjit_impl import (
    extract_plane_components,
    intersect_and_scatter,
    prepare_plane_intersections,
    run_dda_traversal,
)
from .native_impl import accumulate_reflection_grid

__all__ = [
    "accumulate_reflection_grid",
    "extract_plane_components",
    "intersect_and_scatter",
    "prepare_plane_intersections",
    "run_dda_traversal",
]
