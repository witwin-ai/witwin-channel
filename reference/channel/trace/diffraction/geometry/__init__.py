"""Geometric helpers for diffraction state construction and evaluation.

This module re-exports all public symbols from the sub-modules:
- angles: Edge angle computation and geometry setup
- visibility: Visibility and validity masks
- fields: Field computation, reflection coefficients, and material helpers
"""

from ....utils.constants import DIFFRACTION_MIN_DISTANCE  # noqa: F401

from .angles import *  # noqa: F401, F403
from .visibility import *  # noqa: F401, F403
from .fields import *  # noqa: F401, F403

__all__ = [
    "DIFFRACTION_MIN_DISTANCE",
    "_coerce_material_override",
    "_compute_edge_angles",
    "_compute_edge_geometry",
    "_compute_incident_edge_geometry",
    "_edge_owner_structure_idx",
    "_edge_face_material_inputs",
    "_edge_face_reflection_coefficients",
    "_edge_face_reflection_operators",
    "_evaluate_reflection_prefix_chain",
    "_fused_diffraction_visibility_masks",
    "_intersect_rays_ad",
    "_intersect_rays_ad_with_prim",
    "_normalize_in_wedge_plane",
    "_point_inside_closed_mesh_mask",
    "_point_inside_closed_mesh_mask_single",
    "point_in_triangle_3d",
    "_point_source_field",
    "_point_source_field_normal_derivative",
    "_project_to_wedge_plane",
    "reflect_point_across_plane",
    "_reflected_path_support_mask",
    "_reflection_material_params",
    "_reflection_uses_fresnel",
    "_segment_visibility_mask",
    "_segment_visibility_masks_batched",
    "_slope_derivative_safe_mask",
    "_triangle_surface_intersection",
    "_wedge_exterior_region_mask",
]
