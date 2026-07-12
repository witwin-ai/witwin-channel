# witwin.channel_native Feature List

User-visible features of the native channel solver package. Solver-internal
mechanics live in `docs/dev/plans/`.

## Components

All solvers (deterministic, MC basic, BDPT, path) accept a `components` set
drawn from `{los, reflection, diffraction, transmission, scattering}`.
Defaults remain `{los, reflection, diffraction}`; transmission and scattering
are opt-in. Component power reporting uses an exclusive path class with
priority `scattering > diffraction > transmission > reflection > los`.

- **LoS, specular reflection (depth <= 5), first-order UTD diffraction,
  reflection-diffraction coupling** - pre-existing.
- **Specular transmission (thin_sheet)**: finite-thickness multilayer walls
  via a stable scattering-matrix layer stack (complex Fresnel r/t per
  polarization, Fabry-Perot, absorption). Deterministic/path evaluate
  straight Tx->Rx penetration chains (exact for vacuum/index-matched walls);
  MC basic and BDPT shoot through walls with the exact lateral exit offset.
  BDPT combines exact straight chains with event-selected mixed
  reflection+transmission chains.
- **Rough-surface Kirchhoff scattering**: driven by metric surface
  statistics (`Roughness`: RMS height, anisotropic correlation lengths,
  principal axis). Ensemble mode precomputes polarized Kirchhoff BSDF tables
  (Beckmann series, energy-normalized, reciprocal) at scene compile;
  deterministic/path use visible-patch quadrature, MC basic an area-sampled
  radiomap, BDPT a three-way (reflect/scatter/transmit) event sampler with
  NEE. Specular reflection from rough faces is coherently attenuated by
  C_r = exp(-2 (k0 cos(theta) sigma_h)^2) regardless of requested
  components. `realization_coherent` mode (deterministic) integrates a
  reproducible height-map phase screen per `(scene_seed, surface_id,
  realization_id)`.

## Materials (ABI v3)

- `Dielectric`, `LossyDielectric`, `DispersiveMaterial`, `ITUMaterial`,
  `PerfectConductor` - pre-existing, unchanged behavior (compile as 1-layer
  stacks).
- `PhysicalSurface(layers=(Layer, ...), geometry_mode, roughness_front)`:
  multilayer walls with per-layer thickness/permittivity/conductivity/
  permeability; CSR-packed in `MaterialStore`.
- Dispersion models: constant, power-law (ITU), `DebyeModel`,
  `TabulatedPermittivity`.
- `Roughness` statistics and `PhaseScreen` height-map assignments
  (`SurfaceAssignment`) per structure.
- `geometry_mode="closed_volume"` is declared but rejected in v1 (thin_sheet
  only).

## Geometry

- `Structure.uv` / `face_uv` texture coordinates threaded to the ray tracer
  (`planar_uv` helper for planar surfaces); required for phase-screen
  realizations.
- Path topology supports specular reflection depths 1 through 5 and bounded
  reflection-diffraction coupling with exactly one reflection and one
  diffraction, in both R-D and D-R orders.
- Canonical path selection orders by endpoint, depth, component, and event
  identity; it deduplicates before applying global or per-pair path caps.

## Results

- `PathResult` per-event `InteractionType` includes `TRANSMISSION` and
  `SCATTERING`; scattering paths are exported as incoherent power paths
  (`scattering_paths_incoherent: true` metadata).
- Component power maps / grids include `transmission` and `scattering`
  when requested.
- Solver metadata reports applicability guards (`kirchhoff_domain_exceeded`,
  `phase_screen_geometry_limit_exceeded`), approximation flags
  (`thin_sheet_straight_path_approximation`, geometric group delay), and
  event/sample diagnostics.

## Reference implementation

- `witwin.channel_native.physics.oracle`: CPU complex128 electromagnetic
  oracle (Fresnel, multilayer transfer matrix, Kirchhoff lobes, phase-screen
  patch integrals) backing the golden test suite in `tests/physics/`.
