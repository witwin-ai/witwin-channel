# Plan 05 Implementation Progress

Status log for `05-physical-scattering-transmission-plan.md` /
`05-implementation-contract.md`. Dates are 2026-07-11 unless noted.

## Landed

### Wave 0/1 - contracts, oracle, ABI, component plumbing
- `05-implementation-contract.md` fixes all binding conventions (component
  ids, ABI v3 tensor layout, EM sign/branch rules, thin_sheet geometry in
  both evaluation contexts, BDPT throughput semantics, Kirchhoff v1 scope).
- CPU complex128 oracle: `src/witwin/channel_native/physics/oracle.py`
  (Fresnel admittance form, transfer-matrix layer stack with overflow-safe
  scaling, vector Snell, coherent roughness attenuation, Beckmann-series and
  quadrature Kirchhoff lobes, phase-screen patch integral). Golden tests in
  `tests/physics/` cover plan section 11.1 items 1-8.
- Material ABI v3 (`MATERIAL_ABI_VERSION = 3`): CSR layer tensors, roughness
  fields, `geometry_mode_id`, `scatter_model_id` on `MaterialStore`; Python
  API `DebyeModel`, `TabulatedPermittivity`, `Layer`, `Roughness`,
  `PhysicalSurface`, `PhaseScreen`, `SurfaceAssignment`; legacy materials
  compile as 1-layer CSR unchanged.
- Component set extended to `{los, reflection, diffraction, transmission,
  scattering}` across configs, capabilities, topology enums, result adapters,
  and metadata; defaults unchanged (new components opt-in); exclusive
  `path_class` priority `scattering > diffraction > transmission >
  reflection > los`.

### Wave 2 - native EM core + specular transmission (all four solvers)
- `native/channel_native/em/{complex,medium,fresnel,layer_stack}.cuh`:
  passive-branch complex sqrt, admittance Fresnel, stable backward Airy
  recursion (decaying exponentials only) returning complex r/t for CSR layer
  stacks; parity op `em_layer_stack_eval`.
- BDPT throughput contract fix: scalar throughput now carries amplitude
  `sqrt(R_eff)` / `sqrt(T_eff)` (was power reflectance multiplied onto a
  complex amplitude); verified no contribution kernel reads it.
- `field_transmission_sequence` (endpoint-connection context) and
  `bdpt_transmitted_light_subpath_state` (shooting context with exact lateral
  exit offset and interior-phase compensation); `build_info` reports
  `material_abi_version: 3`.
- Deterministic + path solvers: transmission topology by batched visibility
  marching (component_id 5), field evaluation through the stack op,
  PathResultV2 export with per-event `InteractionType.TRANSMISSION`,
  `thin_sheet_straight_path_approximation` metadata.
- MC basic: straight penetration-chain transmission radiomap.
- BDPT: hybrid estimator - exact straight endpoint chains (discrete measure)
  plus event-selected mixed reflection+transmission chains (seeded
  `p_r/p_t` selection, `1/sqrt(p)` field scaling); native component
  classification and a real `transmission` map slot.
- Decisive vacuum-wall identity tests pass in every solver (a 1-layer vacuum
  thin_sheet wall reproduces the empty-scene LoS complex field).

### Wave 3 core - UV, phase screen, Kirchhoff tables
- `Structure.uv/face_uv` threaded to RayD (previously empty); `planar_uv`
  helper.
- `src/witwin/channel_native/scattering/`: KirchhoffTable precompute
  (float64 oracle-backed Beckmann series, energy-exact per-bin normalization
  with a [0.25, 4] shape-sanity band, reciprocity-symmetrized), torch GPU
  eval/sample/pdf (forward and reverse), `PhaseScreenRuntime` (GPU height
  texture, phasor-domain footprint handling, seeded Gaussian realizations),
  `event_budget` energy split (R_coh / R_diff / T_bar / A with passivity
  checks); lazy `CompiledScene.kirchhoff_tables` / `phase_screen_runtimes`.

### Wave 3 solvers - scattering integration
(in progress; deterministic/path and MC basic/BDPT agents running)

## Known deviations from the plan narrative (documented choices)
- Kirchhoff eval/sample runs as PyTorch GPU tensor code, not dedicated CUDA
  kernels (`em/kirchhoff_bsdf.cuh` etc. deferred until profiling justifies
  them); PyTorch-native is a repo hard requirement and the tables are
  gather+FMA workloads.
- Legacy smooth-reflection kernels (`deterministic_field.cu`,
  `reflection.cu`, `field_transport.cuh::slab_fresnel`) keep their verified
  implementations; the shared `em/` core is authoritative for all NEW code
  and is parity-tested against the oracle. Full unification is follow-up
  work.
- Deterministic/path transmission uses the straight Tx->Rx segment
  approximation (exact for vacuum/index-matched walls), per contract
  section 4; MC/BDPT shooting applies the exact lateral exit offset.
- `path/deterministic` group delay is geometric in v1 (narrowband carrier
  transfer function only).
- BDPT scattering contributions are ensemble power (zero-phase placeholder
  fields); never coherently combined with specular components.

## Environment notes
- RayDi HEAD `b271b23` is broken (unrelated session); native builds pin a
  `git archive` snapshot of `0523e06` (see memory/build docs). Re-point
  `RAYD_SOURCE_DIR` when RayDi HEAD is fixed.
- Two environment-only test failures on this box: benchmark cold-import
  (package not pip-installed into witwin2) and the bdpt perf-gate sibling
  checkout when run from a worktree.
