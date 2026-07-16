# Physics reference domain

This package provides the independent electromagnetic reference oracle used to
validate production Torch/CUDA implementations. It is intentionally slow,
CPU-only, NumPy `float64`/`complex128` code and is not a production compute
backend.

## Ownership

- `physics.reference.oracle` is the canonical implementation of the reference
  equations and reference dataclasses.
- `physics.__init__` and `physics.oracle` are compatibility facades that
  re-export the canonical objects; they do not own duplicate implementations.
- `physics.conventions` owns the small stdlib-only set of electromagnetic
  constants that production code may import without depending on the oracle.
- Production material, scattering, propagation, and solver implementations
  remain in their own domains. Golden comparisons in `tests/physics/` own the
  cross-check between those implementations and this oracle.

## Public entry points

`witwin.channel_native.physics` and its `oracle` compatibility facade expose:

- Constants: `C0`, `EPS0`, `ETA0`, and `MU0`.
- Types: `Medium` and `RTCoefficients`.
- Medium/interface helpers: `complex_sqrt_passive`, `medium_params`,
  `vacuum_medium`, `fresnel_interface`, `refraction_direction`, and
  `coherent_attenuation`.
- Layer, roughness, and integration references: `layer_stack_rt`,
  `kirchhoff_diffuse_lobe_series`,
  `kirchhoff_diffuse_lobe_quadrature`, `phase_screen_patch_integral`, and
  `hemisphere_integral`.

Underscore-prefixed admittance, interface, power, and stack helpers in the
canonical module are internal. `physics.reference` is a direct canonical
reference import path, not an additional implementation.

## Dependency rules

- The canonical oracle depends only on the Python standard library and NumPy.
  It must not import Torch, native/runtime loaders, scenes, materials,
  scattering, propagation, or solvers.
- Production code may import constants from `physics.conventions`; it must not
  call the reference oracle to implement a production result.
- The oracle must remain mathematically independent of production kernels so
  comparison tests can detect shared implementation errors.
- New cross-domain imports must satisfy `ci/check_import_graph.py`. In
  particular, the `oracle_production_dependency` rule is a hard failure, not
  allowlist debt.

## Numerical and AD contract

- Time dependence is `exp(+j*w*t)`; propagation is `exp(-j*k*r)`.
- Conductivity is folded into relative permittivity as
  `eps_r - j*sigma_e/(w*EPS0)`. `Medium.eps` and `Medium.mu` are absolute
  quantities; frequency is hertz and SI units are used throughout.
- The passive square-root branch has `Re >= 0` and `Im <= 0`, so
  `exp(-j*k*z)` decays in passive media.
- Admittances are `Y_TE = k_z/(w*mu)` and `Y_TM = w*eps/k_z`. Tangential
  electric-field amplitudes use
  `r = (Y1-Y2)/(Y1+Y2)` and `t = 2*Y1/(Y1+Y2)` for both polarizations.
- Power coefficients are `R=|r|^2`,
  `T=Re(Y2)/Re(Y1)*|t|^2`, and `A=1-R-T`. Values are not clamped.
- Array-like inputs are normalized to NumPy `float64`/`complex128` as
  appropriate. Reference functions preserve explicit quadrature/series
  semantics rather than adopting production approximations.

### AD contract

The reference oracle has no Torch autograd, JVP, or VJP contract. It returns
NumPy values and dataclasses only. Finite differences around it may be used by
tests as independent evidence, but are not a production derivative path.
Production AD ownership remains with native companion kernels and their domain
autograd facades.

## Forbidden fallback

Do not invoke the oracle when CUDA, a native symbol, a scene capability, or a
production AD companion is unavailable. It must never serve as a CPU,
finite-difference, reduced-precision, or silent-degradation fallback. Invalid
inputs or numerical errors must not be caught and replaced with plausible
values; reference results are not clamped to make a comparison pass.

## Maintenance

- Keep `physics.__init__`, `physics.oracle`, and `physics.reference`
  re-exports synchronized when the intended public reference surface changes.
  If this surface is added to or changed within the curated public API
  snapshot, update `ci/public-api-snapshot.json` and include a migration note.
- A production native wrapper or canonical Python owner move must update the
  applicable binding/audit manifest. The completed historical migration ledger
  is archived at `docs/dev/audit/phase12-ops-migration-ledger.json`.
- Any dependency change must pass the import-graph checker, and numerical
  convention changes require corresponding independent golden tests and an
  explicit migration explanation.
