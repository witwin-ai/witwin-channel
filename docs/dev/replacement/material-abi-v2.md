# Material ABI v2

Phase 7A replaces the anonymous `model_params` matrix with named, unit-bearing
material tensors. `MaterialStore` ABI version 2 contains stable `material_id`,
explicit `model_id`, `eps_r`, `mu_r`, `sigma_e` (S/m), `gain`, `thickness_m`
(m), `scattering_coefficient`, and `xpd_coefficient`. `material_keys` provide a
stable structure/name trace, and `cache_token` includes the ABI version,
frequency, evaluated parameters, and keys.

Dispersive materials are evaluated at `Scene.frequency` during compilation.
Changing frequency or any material parameters therefore produces a different
cache token. Mitsuba/Sionna XML thickness, scattering coefficient, and XPD
coefficient are retained through import and material compilation. Perfect
conductors use explicit `model_id=2`; the finite effective conductivity is only
an adapter at legacy Fresnel-kernel boundaries and is not the PEC definition.

Phase 7B remains unsupported. Transmission/refraction, absorption, layered
media, rough scattering, tabulated polarimetric BSDFs, medium stacks, and
energy-accounting events are all reported as `false` in the capability
manifest, and no solver claims those event types.

The capability manifest distinguishes the two boundaries explicitly:
`runtime_material_abi_integration` reports the Phase 7A ABI path used by every
solver, while `event_solver_integration` remains false for all Phase 7B events.
