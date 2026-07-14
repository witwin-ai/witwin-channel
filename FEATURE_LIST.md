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

## BDPT estimator contract

- Coherent endpoint, reflection, transmission, and connection events use an
  authoritative Complex3/Jones field. Local s/p frames are reconstructed at
  every interaction. Scalar throughput is only a sampling-probability proxy
  and never contributes to the received field.
- Ensemble Kirchhoff scattering is explicitly incoherent power-only. It does
  not manufacture a zero-phase Complex3 carrier or claim cross-polarized phase.
- Receiver endpoints have `sensor_depth == 0`; sensor subpaths are not a
  declared capability. `max_light_depth` is the only subpath depth control.
- Reported proposal PDFs exclude free-space and solid-angle-to-area geometry
  Jacobians. Canonically enumerated delta paths have unit discrete mass;
  sampled reflection/transmission events retain their event-selection mass.
- Endpoint connections are single-strategy. Diffraction MIS supports the
  native direct and Keller proposals, with the declared strategy count checked
  against the proposals actually enabled.
- `benchmarks/bench_phase_c_statistics.py` provides the versioned multi-seed
  statistical gate. Full acceptance always runs the 16 seeds fixed in
  `benchmarks/gates/phase_c_statistics.v1.json`.

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

## Differentiable solving (fixed topology, plan 07 AD-1)

- `deterministic` and `path` accept `ad_mode="jvp" | "vjp"` (default
  `"none"`); `montecarlo.basic` and `montecarlo.bdpt` still reject any
  non-`"none"` mode.
- Differentiable parameters this phase: material `eps_r` / `sigma_e` /
  `gain` / `thickness` (per bounce for reflection, per CSR layer for
  transmission) and the carrier frequency. Hit geometry (endpoints,
  interaction positions/normals, polarizations, tx_power) and `mu_r` are
  detached under the fixed-winner contract; requesting their gradient fails
  with `NotImplementedError` instead of returning silent zeros.
- Reverse mode: set `requires_grad_(True)` on the compiled material store
  tensors (`scene.compile().materials.eps_r` etc.) and call `.backward()` on
  a loss built from the result's complex coefficients / `path_gain`.
  Forward mode: `torch.func.jvp` through the field Functions, or
  `torch.autograd.forward_ad.dual_level` through a full solve.
- Frequency as a first-class differentiable input: `Scene(...,
  frequency=torch.tensor(f0, device="cuda", requires_grad=True))` accepts a
  0-d tensor; the scalar is read once per solve (one host sync), dispersive
  material records stay frozen at the primal frequency.
- Implementation: native CUDA backward/jvp companion kernels for
  `field_free_space` / `field_reflection_sequence` /
  `field_transmission_sequence`, wrapped in thin `torch.autograd.Function`s
  and wired into the shared deterministic/path field seam. `ad_mode="none"`
  keeps the exact primal behavior (no graph, no extra launches, zero tape).
- Explicit-failure policy: topologies containing diffraction or coupled
  reflection-diffraction paths, and the scattering component, raise a
  `RuntimeError` naming the interaction before any launch when
  `ad_mode != "none"` (differentiable versions arrive with plan 07 AD-4).
- The reserved `psdr` solver stub has been removed; AD lives in the existing
  solvers' `ad_mode`.

## Reference implementation

- `witwin.channel_native.physics.oracle`: CPU complex128 electromagnetic
  oracle (Fresnel, multilayer transfer matrix, Kirchhoff lobes, phase-screen
  patch integrals) backing the golden test suite in `tests/physics/`.
- `tests/ad/_reference_fields.py`: pure-torch complex128 mirrors of the
  free-space carrier, finite-slab Fresnel reflection chain, and Rouard
  transmission stack used as forward-parity and gradient oracles for the
  native AD companion kernels.
