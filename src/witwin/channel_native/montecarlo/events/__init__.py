"""Shared Monte Carlo event contracts for the basic and BDPT solvers.

Owns the two-way specular transmission helpers and the three-way Kirchhoff
scattering event machinery both stochastic solvers share. Re-exports the
public names the MC solvers import.
"""

from __future__ import annotations

from .scattering import (
    RoughMaterialRuntime,
    local_frames,
    rough_material_runtimes,
    sample_scatter_directions,
    scatter_direction_uniforms,
    scattered_subpath_state,
    scattering_map_matrix,
    scattering_nee_connection_samples,
    solid_angle_to_area_jacobian,
    te_tm_incident_power,
    three_way_rough_probabilities,
    world_to_local,
)
from .transmission import (
    event_uniforms,
    incident_te_tm_fractions,
    layer_csr_view,
    scene_diagonal_m,
    straight_transmission_chains,
    transmission_event_probability,
    unpolarized_power_budgets,
)

__all__ = [
    "RoughMaterialRuntime",
    "event_uniforms",
    "incident_te_tm_fractions",
    "layer_csr_view",
    "local_frames",
    "rough_material_runtimes",
    "sample_scatter_directions",
    "scatter_direction_uniforms",
    "scattered_subpath_state",
    "scattering_map_matrix",
    "scattering_nee_connection_samples",
    "scene_diagonal_m",
    "solid_angle_to_area_jacobian",
    "straight_transmission_chains",
    "te_tm_incident_power",
    "three_way_rough_probabilities",
    "transmission_event_probability",
    "unpolarized_power_budgets",
    "world_to_local",
]
