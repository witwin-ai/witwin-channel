"""CPU complex128 electromagnetic oracle (numpy-only reference math).

Public re-exports of the canonical
:mod:`witwin.channel.physics.reference.oracle` implementation.
"""

from witwin.channel.physics.reference.oracle import (
    C0,
    EPS0,
    ETA0,
    MU0,
    Medium,
    RTCoefficients,
    coherent_attenuation,
    complex_sqrt_passive,
    fresnel_interface,
    hemisphere_integral,
    kirchhoff_diffuse_lobe_quadrature,
    kirchhoff_diffuse_lobe_series,
    layer_stack_rt,
    medium_params,
    phase_screen_patch_integral,
    refraction_direction,
    vacuum_medium,
)

__all__ = [
    "C0",
    "EPS0",
    "ETA0",
    "MU0",
    "Medium",
    "RTCoefficients",
    "coherent_attenuation",
    "complex_sqrt_passive",
    "fresnel_interface",
    "hemisphere_integral",
    "kirchhoff_diffuse_lobe_quadrature",
    "kirchhoff_diffuse_lobe_series",
    "layer_stack_rt",
    "medium_params",
    "phase_screen_patch_integral",
    "refraction_direction",
    "vacuum_medium",
]
