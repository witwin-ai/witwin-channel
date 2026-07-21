# witwin.channel_native Feature List

User-visible features of the native channel solver package. Solver-internal
mechanics live in `docs/dev/plans/`.

## Components

All solvers (deterministic, MC basic, BDPT, path) accept a `components` set
drawn from `{los, reflection, diffraction, transmission, scattering}`.
Defaults remain `{los, reflection, diffraction}`; transmission and scattering
are opt-in. Component power reporting uses an exclusive path class with
priority `scattering > diffraction > transmission > reflection > los`.

## Native runtime boundary

- `_channel_native` is the single production extension. It source-links RayD
  `474c122aa3cd6b6d098675e076a73e6f485bd6be` and calls the typed
  `rayd::torch` C++ API directly; no RayD Python module, second dispatcher,
  copied C ABI, getter table, or dynamic symbol lookup participates. RayD's
  legacy `extern "C"` Torch integration entry points are retired.
- Scene ownership crosses Python/C++ as `RayDSceneResource`; integer scene
  handles and the former bridge/common indirection are removed. RayD-owned ABI
  names use `rayd_*`; Channel-composed R-D/D-D geometry uses `coupled_*`.
- The locked integration header is
  `backends/torch/include/rayd/torch/integration.h` with SHA-256
  `57f83ea460e376166fd5ee22a8243a7c1576a290e1de99c0cbe8e86e93392e14`
  and identity
  `rayd.torch.integration`. The numeric API version is 6 and is validated
  independently from this stable, capability-neutral source name.
- ADR-027 Phase P exposes the complete RayD fixed-capacity straight-segment
  penetration primal/tape/VJP/JVP family, Channel named facades, shared
  capacity-failure wiring, compile-frozen policy diagonals, component-5
  topology pack, and MC wall-product prefix semantics. Path, Deterministic,
  and the ADR-008 BDPT oracle now share one live pair-major
  `EnumeratedFullDistance` batch through the enumerated engine. The old
  per-depth active-row/closest-hit march is deleted without a compatibility
  alias. Monte Carlo Basic retains its existing route until its separate
  `MonteCarloTargetInset` atomic switch.
- RayD is the unique numerical source owner of the shared complex, medium,
  Fresnel, layer-stack, Jones/field-transport primal/dual headers and of
  `em_layer_stack_eval/backward/jvp` and
  `field_transmission_sequence/backward/jvp`. Channel retains the material
  ABI/CSR, field-row schemas, validation, cache, `_channel_native` bindings,
  Python facades, and the fused BDPT transmitted-state family; there is no
  Channel-private transmission-sequence numerical copy or compatibility
  forwarding header.
- ADR-025 assigns diffraction ownership by complete operation family. Phase 8A
  atomically moved the pure-wedge fixed-winner primal/backward/JVP numerical
  owner to RayD while preserving Channel ABI and typed field/autograd facades.
  The former Channel numerical TU is deleted; there is no forwarding source,
  fallback, second launch, or second compiled owner. MC Sionna fixed-tape and
  coupled R-D/D-D primal/backward/JVP families remain complete Channel owners.
  Pure wedge keeps exporter-locked `--use_fast_math`; MC and coupled families
  remain precise. Phase 8B still owns the sample-tape rename and native
  transmitter-edge visibility planning/selection operation.
- ADR-026 assigns 17 solver-neutral resident scattering runtime contracts to
  RayD. Phase 10A atomically moved the eleven table, single-bounce ensemble,
  and patch-integral contracts plus all seven table-interpolation helpers;
  Phase 10B moved the six fused ensemble/realization chain contracts. Channel
  retains the stable ABI and typed domain/autograd facades for all 17.
  Table and phase-screen construction/cache/seed lifecycle, event
  probabilities, topology/packing, RNG/MIS, accumulation, and result policy
  remain Channel owners. Per-TU default/`--fmad=false` modes and the
  family-specific chain geometry AD behavior are frozen.

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
- Coupled double diffraction (D->D, component id 7; ADR-013): `coupled_paths`
  enables the uniform order-2 compensator family {R->D, D->R, D->D}, enumerating
  ordered edge pairs (TX -> e1 -> e2 -> RX) under the shared coupled candidate
  budget and accumulating into the coupled component slot; coupled-off solves
  stay byte-identical.
- Canonical path selection orders by endpoint, depth, component, and event
  identity; it deduplicates before applying global or per-pair path caps.

## Results

- ADR-029 adds explicit `path_capacity_per_pair` and
  `diffraction_state_capacity` fields to Path and Deterministic configuration.
  This configuration-only step requires exact non-negative integers (Boolean,
  float, and NaN values are rejected) and validates the per-pair `max_paths`
  bound; solver enforcement and capacity-backed result activation remain
  staged Phase 12 work.
- `PathResult` per-event `InteractionType` includes `TRANSMISSION` and
  `SCATTERING`; scattering paths are exported as incoherent power paths
  (`scattering_paths_incoherent: true` metadata).
- Component power maps / grids include `transmission` and `scattering`
  when requested.
- Solver metadata reports applicability guards (`kirchhoff_domain_exceeded`,
  `phase_screen_geometry_limit_exceeded`), approximation flags
  (`thin_sheet_straight_path_approximation`, geometric group delay), and
  event/sample diagnostics.

## Full-wave validation workflow

- `python -m benchmarks.fullwave_validation` provides versioned single-cube
  and original-channel-layout three-cube cases for PEC and transmissive
  dielectric materials. It writes deterministic and Tidy3D complex-field
  references in one NPZ schema, supports same-observable complex calibration
  and independent empty-scene magnitude calibration, and reports NMSE,
  magnitude correlation, dB error, and ISB/RSB support-edge jump statistics.
  Tidy3D cloud submission is never implicit: `prepare` only writes a simulation
  and `solve-tidy3d` requires either downloaded simulation data or an explicit
  `--submit` flag.

## Field continuity and polarization consistency (2026-07 UTD fixes)

- The deterministic coherent field is polarization-consistent across all
  components: LoS/reflection/transmission/diffraction transport the true
  transmitter polarization as an unnormalized transverse projection
  (short-dipole sin θ pattern), and the exported scalar is the
  receiver-polarization projection `p̂_rx·E⃗` — the same physical observable
  as a full-wave `Ez` monitor. The MC-basic estimator uses the same
  transmitter-polarization conventions.
- Finite-edge UTD diffraction is continuous through shadow boundaries,
  edge-endpoint/vertex regions, and the edges' extension planes: the
  Kouyoumjian–Pathak coefficient's GO-compensating (odd) component enters
  exactly where GO toggles and is suppressed beyond the edge ends, while the
  smooth background carries a normalized Fresnel endpoint truncation
  (`docs/dev/audit/utd-continuity-fix-design.md`, F1–F6). The legacy 5 cm
  UTD distance gate, the endpoint-continuation branch, and the `exp(-u²)`
  completion taper are removed. Continuity is pinned by
  `tests/deterministic/test_field_continuity.py`; against the recorded
  Maxwell reference the single-cube ISB p95 excess jump moved from +6.96 dB
  to -0.10 dB and envelope NMSE from 0.217 to 0.044.

## Differentiable solving (fixed topology, plan 07 AD-1 through AD-4)

- `deterministic`, `path` and `montecarlo.basic` accept
  `ad_mode="jvp" | "vjp"` (default `"none"`); `montecarlo.bdpt` still
  rejects any non-`"none"` mode. The capability manifest advertises this:
  `supports_ad=True` with `ad_modes=["none", "jvp", "vjp"]` for the three
  fixed-topology solvers, and the global `ad_contract` states what is and
  is not delivered (fixed-topology JVP/VJP yes; visibility/topology
  discontinuity estimator no; bdpt no; per-solver exclusions listed).
- Differentiable parameters: material `eps_r` / `sigma_e` / `gain` /
  `thickness` (per bounce for reflection, per CSR layer for transmission),
  the carrier frequency, and since AD-2 the continuous hit geometry:
  TX/RX positions and mesh vertices flow through the native field kernels'
  geometry adjoints (`source`, `target`, `interaction_positions`,
  `interaction_normals`). `path_length_m` / `delay_s` are differentiable
  outputs of the geometry (time-of-arrival losses), with an exactly zero
  cotangent into materials and frequency. The discrete winner
  (polarizations, tx_power, `mu_r`, material ids, valid masks) stays fixed;
  requesting its gradient fails with `NotImplementedError` instead of
  returning silent zeros, and path birth/death discontinuities remain out
  of contract (fixed-winner gradients only).
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
  `field_free_space` / `field_reflection_sequence`, plus the typed RayD native
  `field_transmission_sequence` family, wrapped in thin
  `torch.autograd.Function`s and wired into the shared deterministic/path field
  seam. `ad_mode="none"`
  keeps the exact primal behavior (no graph, no extra launches, zero tape).
- UTD diffraction and coupled reflection-diffraction AD (plan 07 AD-4):
  under `ad_mode != "none"` the shared field seam re-evaluates RayD's
  order-1 wedge export from the frozen topology (edge id, edge geometry,
  wedge-face materials, half-space Fresnel with the export's `+z` tx
  polarization convention) with a native kernel whose forward is RayD's own
  scalar-templated UTD implementation; instantiating the same code with
  dual scalars yields the exact backward/jvp (no finite differences, no
  duplicated physics). The receiver projection (`field_project_complex3`)
  and the coupled R-D transport (`field_coupled_rd`, 12 material scalars +
  frequency + endpoints) get native companions the same way, and under
  geometry AD the coupled interaction points are re-solved differentiably
  from the frozen winner (image source + stationary edge point + wall
  crossing). The coupled pseudo-infinite edge truncation factor is a frozen
  regularizer of the differentiation (its float32 endpoint ripple is
  measurement noise amplified by the 1e5 lever arm; the true infinite-edge
  derivative is zero).
- Mesh-vertex gradients cover every wired interaction (plan 07 AD-4b):
  reflection (any depth) through RayD's fixed-winner EPC chain,
  transmission through the differentiable face-normal table, and wedge
  diffraction through edge tables rebuilt from the winner vertices inside
  the wedge kernel (edge anchor/direction/bounds, sign-aligned face
  normals, exterior angle; the frozen discovery tables pin the plane
  assignment so RayD's ordering conventions cannot drift the primal).
  A LoS path touches no face, so its vertex gradient is structurally zero
  (pinned by test). Coupled R-D paths do not support vertex gradients: the
  coupled adjoints take the wall plane and edge tables as frozen winners,
  and the solver fails loudly (`NotImplementedError`) instead of returning
  a silently incomplete gradient (registered as `xfail(strict=True)`).
- Deterministic scattering AD (ADR-014): the two native scattering ops gain
  registered JVP/VJP companions (`scattering_ensemble_eval_{backward,jvp}`,
  `scattering_patch_integral_eval_{backward,jvp}`). Under `ad_mode != "none"`
  the enumerated scattering seam builds the radiometric scale `coef` (ensemble
  rows) and the wavenumber `k0` plus the outer amplitude scale (realization
  rows) as Torch scalars and dispatches the `_ad` wrappers, so gradients reach
  the resident Kirchhoff BSDF tables, the phase-screen heights, the
  fixed-topology row geometry, and the carrier frequency; the fixed
  winner/visibility structure, polarizations, material ids, table metadata and
  quadrature nodes stay frozen and reject a requested gradient/tangent loudly.
  `ad_mode == "none"` keeps the exact bitwise primal path (no tape, no
  `autograd.Function`). The AD kernels recompute their primals under
  `--fmad=false` in lockstep with the forwards. The former path/deterministic
  pipeline guard rejecting `scattering` under `ad_mode != "none"` is lifted:
  both solvers now solve scattering scenes end-to-end in `vjp`/`jvp` mode
  (locked by `tests/ad/test_solver_scattering_ad.py`).
- Scattering AD completion (ADR-015): the remaining forward-only scattering
  paths gain native derivative companions, so scattering is differentiable
  across every non-BDPT solver. (A) Monte Carlo basic scattering: the resident
  Kirchhoff BSDF table lookup gains `scattering_table_eval_{backward,jvp}`; the
  MC-basic scattering map (`scattering_map_matrix`) dispatches the `_ad` wrapper
  under `ad_mode != "none"`, carrying gradients to the table values, the carrier
  frequency (through the `(lambda/4pi)^2` amplitude factor) and `tx_power`,
  while the multinomial sample set, both visibility masks and the active gates
  stay frozen (detached). (B) Realization material chain: the enumerated
  realization rows dispatch `em_layer_stack_ad`, completing
  `total -> r_te/r_tm -> layer eps_r / sigma_e / thickness / frequency /
  cos_theta`. (C) Differentiable Kirchhoff table construction: the float64
  numpy build stays the bit-for-bit compile-time island, wrapped by an
  `autograd.Function` whose native `kirchhoff_table_build_{backward,jvp}`
  companions differentiate the resident `f_te`/`f_tm` values w.r.t. the
  roughness statistics (`sigma_h`, `corr_x`, `corr_y`), the CSR layer
  parameters and the carrier frequency (implicit-Sinkhorn adjoint solved with
  a cuSOLVER dense LU). All primal paths stay bitwise identical under
  `ad_mode == "none"`; BDPT keeps rejecting every `ad_mode`. Bindings re-frozen
  183 -> 187 (locked by `tests/ad/test_mc_basic_scattering_ad.py`,
  `tests/scattering/test_table_build_ad.py`).
- Multi-bounce chain scattering (ADR-021 D1/D2, default OFF): Deterministic and
  Path gain an enumerated scatter-chain path class with exactly one diffuse
  vertex and specular reflection chains on either side
  (`TX --C1--> v_s --C2--> RX`, `1 <= d1 + d2 <= scattering_chain_max_depth`).
  Two new fused native ops evaluate complete chain rows in one launch: Op A
  `scattering_chain_ensemble_eval` (power domain, generalizes the ensemble op 1)
  and Op B `scattering_chain_realization_eval` (coherent phase-screen, generalizes
  the patch-integral op 2), each reusing the shared `field_transport.cuh` /
  `scattering_table.cuh` device primitives with no host physics. Both are
  born-differentiable: `scattering_chain_ensemble_eval_{backward,jvp}` and
  `scattering_chain_realization_eval_{backward,jvp}`. New config on the
  Deterministic and Path `Config`: `scattering_chain_max_depth: int = 0`
  (0 disables discovery, bitwise no-op), `scattering_chain_samples_per_m2: float
  = 2.0`, `scattering_chain_max_rows: int = 256`. The `d1 = d2 = 0` degenerate
  row collapses symbol-for-symbol to op 1 / op 2 (lockstep-pinned, not
  production-dispatched).
- Coherent scattering combine (ADR-021 D3, default OFF): new Deterministic/Path
  config `scattering_coherent: bool = False`. OFF is bit-identical to today (the
  scattering slot stays the incoherent `SUM |field|^2` power term). ON (requires
  realization / Op B rows, refuses ensemble-only solves loudly) sums the complex
  `path_field` of scattering rows per (tx, rx) and finalizes `|sum|^2`, following
  the ADR-019 per-component phasor precedent. It is implemented as a defaulted
  `scattering_combine_domain` argument on the existing `deterministic_accumulate_flat`
  op (no new primal ABI symbol); `combine == 0` never enters the branch and the
  kernels stay byte-identical.
- BDPT multi-order scattering (ADR-021 D4, default OFF): `montecarlo.bdpt.Config`
  gains `max_scattering_order: int = 1` (default 1 keeps today's terminal
  single-scatter behaviour, bitwise: seed consumption, event partition and NEE
  rows untouched). For `> 1` a scatter-selected hit emits NEE rows as today and
  continues the subpath via the resident reciprocal table CDF, in the power
  domain (documented v1 carrier contract). MC-basic keeps its single-scatter
  analytic deposit (`scattering max_depth = 1` metadata). Native binding count
  re-frozen 193 -> 199 (the 6 ADR-021 chain symbols).
- Explicit-failure policy: a montecarlo.basic reflection solve whose
  `max_depth` exceeds the native reflection AD depth cap
  (`ops.mc_reflection_ad_max_depth()`, mirrored from the kernel constant) is
  rejected at `solve()` instead of failing mid-backward.
- AD metadata (plan 07 AD-4): `result.metadata["kernel"]` reports the real
  `ad_status`, `tape_bytes` (bytes retained via `save_for_backward`; zero
  for `none`/`jvp`), `backward_launch_count` / `jvp_launch_count`
  (registered companion launches, one `AdLaunchLedger` shape across
  montecarlo.basic / deterministic / path), plus `forward_time_ms`
  (CUDA-synchronized solve wall time; a jvp solve carries its dual pass
  here) and `peak_memory_bytes` (how far the solve raised the process CUDA
  high-water mark). A vjp solve cannot observe its future backward, so
  reverse-pass time/memory budgets are pinned by the CI gates in
  `tests/ad/test_ad_budgets.py` (tolerance freeze + forward/backward
  time, tape and peak-memory overhead budgets) rather than a metadata
  field.
- The reserved `psdr` solver stub has been removed; AD lives in the existing
  solvers' `ad_mode`.
- Monte Carlo basic power map AD (plan 07 AD-3): the incoherent
  `Result.path_gain` radiomap/matrix differentiates with respect to the
  compiled material store leaves (per-face `eps_r` / `sigma_e` / `gain` /
  `thickness` for the reflection map, per-CSR-layer parameters for the
  transmission map), the carrier frequency (including the LoS
  `(lambda/4pi)^2` aperture and the radiomap deposit weight), and the LoS
  TX/RX point positions. Materials come from the compiled store in BOTH
  `ad_mode="none"` and the AD modes (one source, same values; the old
  host-float flattening is gone). The RayD trace/sampling tapes, sampled
  directions, deposit binning and visibility masks are frozen winners; the
  reflection deposit weight is analytically independent of the ray origin,
  so the transmitter-position gradient of the reflection map is an exact
  zero (delivered through a live graph, not a missing gradient). The
  transmission map's transmitter gradient is genuinely nonzero (the
  straight-line incidence cosine, and with it every per-wall transmittance,
  moves with the live march origin) and the layer-stack dual carries the
  squared transverse wave number so exactly-normal rays keep a finite
  cos_theta derivative. Grid receivers expose no per-receiver position leaf
  (the grid is the output).
- Monte Carlo basic diffraction map AD (plan 07 AD-4b): the Sionna-style
  Keller-cone diffraction radiomap differentiates with respect to the
  wedge-face slab materials (`eps_r` / `sigma_e` / `gain` / `thickness`),
  the carrier frequency and the transmitter position. The per-lane row of
  the tape accumulator is templated over the scalar type: the float
  instantiation IS the primal deposit and the dual instantiation carries
  the exact derivative through the recomputed cone geometry, the incident
  spherical wave, the stored slab face operators and the UTD pair
  (fixed-point + stored-ops convention). The RayD sampling tape (active /
  state / cell / u), the per-lane azimuth and the deposit binning stay
  frozen winners. Unlike the reflection map, the transmitter gradient is
  genuinely nonzero here (the deposit carries the incident 1/s wave and
  the source-dependent cone orientation). The scattering map keeps
  rejecting AD loudly.

## Reference implementation

- `witwin.channel_native.physics.oracle`: CPU complex128 electromagnetic
  oracle (Fresnel, multilayer transfer matrix, Kirchhoff lobes, phase-screen
  patch integrals) backing the golden test suite in `tests/physics/`.
- `tests/ad/_reference_fields.py`: pure-torch complex128 mirrors of the
  free-space carrier, finite-slab Fresnel reflection chain, and Rouard
  transmission stack used as forward-parity and gradient oracles for the
  native AD companion kernels.
