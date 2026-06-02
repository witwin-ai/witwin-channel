# Coherent Polarimetric Scattering Model Plan

Status: Active
Category: Plan
Last reviewed: 2026-05-22

This plan adds a **coherent, complex-valued, full-polarimetric (2×2) surface
scattering model** to `witwin.channel`, shared by the Monte Carlo and
deterministic radiomap solvers. It is the L1-moat "differentiable rough-surface
scattering" feature from `plans/01-platform-strategy-and-research-directions.md`,
staged so the first increment is a parametric model and the eventual full-wave
**T-matrix** (`plans/00` / the T-matrix scattering direction) drops into the same
interface with no integrator changes.

This plan is in the 20–29 physics/feature family (that band is full, so the file
takes prefix 30); it is a physics feature, not an operational rollout.

## Motivation and Current State

The interaction taxonomy today is reflection + diffraction. There is **no
scattering**: `scattering_coefficient` / `xpd_coefficient` are imported from the
Sionna adaptor (`core/scene/sionna_adaptor.py`) and stored as dead metadata,
never read by a solver.

What already exists and must be reused, not rebuilt:

- **Complex, phase-tracked, polarimetric reflection.** Fresnel TE/TM
  coefficients are complex (`fresnel_reflection`, `wave_math.py:57`), complex
  permittivity is `ε_r − jσ/ωε₀` (`complex_relative_permittivity`,
  `wave_math.py:53`), and the reflected field is transported as a complex vector
  field (`reflect_field_vector`, `polarization.py:341`).
- **A complete 2×2 Jones-operator toolkit** in `core/physics/polarization.py`:
  `jones_operator_diagonal`, `apply_jones_operator`, `jones_operator_matmul`,
  `jones_operator_rotator`, `jones_operator_in_basis`, `fresnel_diagonal_operator`,
  and vector⇄Jones transforms. This is exactly the algebra a 2×2 scattering
  matrix needs.
- **Coherent field accumulation with `exp(−jk r)` phase** in both solvers
  (`montecarlo/grid_ops.py` `point_source`; deterministic `_rayd_path_unit_field`,
  `epc.py`). Radiomap diagnostics already carry `coherent` / `incoherent` /
  `coherent_power` channels (`montecarlo/integrators/basic.py:83`).

The requirement is therefore narrow and well-supported: a **scattered field that
is added coherently** (carries `exp(−jk(r_in+r_out))` phase, accumulated into the
`coherent` channel), as opposed to the classical ITU power-only incoherent
diffuse term. The scattered field is produced by a complex 2×2 matrix per
direction pair.

## Design Keystone: the scattering matrix is the T-matrix slot

At a surface hit point `x` with outward normal `n̂`, incident direction `k̂_in`
(toward the surface) and outgoing direction `k̂_out`, the scattered field is

```
E_out(k̂_out) = S(k̂_in, k̂_out, n̂; material, λ) · E_in
```

where `S` is a **2×2 complex Jones matrix** in the (TE, TM) bases of the incident
and outgoing rays. This object is the per-direction-pair scattering matrix. It is
the parametric stand-in for a full-wave T-matrix: the future
`TMatrixScatteringModel` returns the same 2×2 complex operator from an
interpolated full-wave table, so swapping it requires **zero integrator changes**.

Single interface (new module `core/physics/scattering.py`):

```
def scattering_jones_operator(
    *, k_in, k_out, normal, material, wavelength,
) -> jones_operator   # 2x2 complex, in the standard incident/outgoing TE/TM bases
```

`material` carries the new scattering parameters (below). Callers (MC, det)
already own basis construction and `exp(−jk r)` phase; the operator returns only
the complex amplitude matrix.

### Parametric model (first implementation: coherent Degli-Esposti + Fresnel anchor)

- **Energy split.** Specular reflection amplitude is scaled by `√(1 − S²)`; the
  scattered lobe carries amplitude `S` of the field, so reflected power
  `|R|²` splits into `|R|²(1−S²)` specular + `|R|² S²` scattered (energy
  conserved). `S` is the per-material scattering coefficient.
- **Directive lobe.** Co-pol amplitude shape `√(F_lobe(ψ))` with the
  Degli-Esposti single lobe `F_lobe(ψ) = F0 · ((1 + cos ψ)/2)^{α_R}`, `ψ` the
  angle between `k̂_out` and the specular direction. `α_R` is the new lobe-width
  material parameter; `F0` normalizes the lobe to unit power over the hemisphere
  (needed for the energy budget).
- **Material anchor.** The co-pol amplitude is anchored to the local Fresnel
  coefficient `R_eff(θ_in)` so the scattered field stays physically tied to the
  material (reuse `fresnel_reflection`).
- **Polarimetric cross terms.** Off-diagonal entries set by a cross-pol ratio
  (Degli-Esposti `K_xpol` / XPD) so `S` is a full 2×2 matrix, not diagonal. This
  is the polarimetric content the diagonal Fresnel reflection cannot express.
- **Coherent phase.** The scattered contribution is accumulated with
  `exp(−jk(r_in + r_out))`, supplied by the existing path-length→phase machinery.
  This is the explicit "complex with phase" requirement and the difference from
  an incoherent diffuse model.

### Invariants (test contract, also the contract the future T-matrix must satisfy)

- **Reciprocity:** `S(k_in, k_out) = S(k_out, k_in)^T` up to basis convention.
- **Energy budget:** specular + ∫_hemisphere scattered ≤ incident, with equality
  in the lossless limit.
- **S = 0 ⇒ identity to today:** with the default scattering coefficient zero,
  every solver output is numerically unchanged.

## Material parameters

Extend `witwin.core.Material` (`core/witwin/core/material.py`) with:

- `scattering_coefficient: float = 0.0`  (S, clamped to [0, 1])
- `alpha_r: float = 4.0`                 (Degli-Esposti lobe exponent)
- `xpd_ratio: float = 0.0`               (cross-pol coupling, 0 = co-pol only)

Defaults give `S = 0` ⇒ no scattering ⇒ bit-identical to current behavior, so
the plumbing stage is safe to land before any integration.

Plumb end to end (mirror the existing `eps_r`/`sigma_e`/`mu_r` path):

1. `StaticMaterialSample` / `FrequencyMaterialSample` carry the new fields
   (`material.py`).
2. `SceneBuilder._extract_structure_meshes` material_entries +
   `_build_triangle_material_data` concat + `_attach_material_data` writes
   `material_scattering_s`, `material_alpha_r`, `material_xpd`
   (`core/scene/builder.py:140,210,226`).
3. `Scene.triangle_material()` gathers and returns the three fields with defaults
   (`core/scene/scene.py:560`).
4. `FaceMaterial` + `resolve_surface_material` expose them to solvers
   (`core/physics/materials.py:19,34`).

## Stages

Each stage is independently landable. Stages A–B are inert (S = 0 preserves all
outputs). Update `FEATURE_LIST.md` in the stage that makes scattering
user-visible (Stage C).

### Stage A — Scattering operator + tests (no integration)

- New `core/physics/scattering.py`: `scattering_jones_operator(...)`, lobe
  normalization `F0`, hemisphere integration helper for the energy check, built
  entirely from existing `wave_math` / `polarization` primitives (DrJit-native,
  no NumPy/Torch in the hot path).
- Tests under `tests/` (a new `tests/scattering/`): reciprocity, energy budget,
  S = 0 ⇒ zero scattered operator, α_R ⇒ lobe-width monotonicity, XPD ⇒
  off-diagonal magnitude.

### Stage B — Material parameters end to end

- The plumbing in "Material parameters" above.
- Regression: existing radiomap tests unchanged (S = 0 default). Add a small test
  that a scene with `scattering_coefficient > 0` round-trips through
  `Scene.triangle_material()`.

### Stage C — Monte Carlo scattering (first order)

- Add a scattering phase to `montecarlo/integrators/basic.py` (`primal`, after the
  reflection phase). Receivers are dense fixed grid cells, so use **next-event
  estimation**: at each first-order reflection hit (the hit data is already
  materialized for the reflection/wedge flow), connect deterministically to each
  cell, weight by `scattering_jones_operator(k_in, hit→cell)` and the coherent
  `exp(−jk(r_in+r_out))` phase. Reuse `GridContributionStore` + the existing
  per-hit→cell point evaluation; this is the same machinery reflection/diffraction
  already use.
- Apply the `√(1−S²)` specular reduction inside the reflection field op
  (`reflect_field_vector` / `fresnel_diagonal_operator`) so the two components
  conserve energy.
- New `scattering` component key in `_empty_radio_map`, summed in
  `_finalize_component_totals`, registered in `metadata.COMPONENT_NAMES` and the
  AD component list. Accumulate into `coherent`, `incoherent`, and
  `coherent_power` consistently.
- Tests: energy conservation (sweep S∈[0,1]: specular(S)+scattered(S) ≈
  reflected(0) per cell within MC noise); coherent interference (a flat plate near
  specular shows interference between specular and scattered, proving phase is
  carried, not power-summed); S = 0 regression.
- Update `FEATURE_LIST.md`.

### Stage D — Deterministic scattering (first order)

- In the EPC reflection accumulation (`deterministic/reflection/epc.py`), at each
  first-order reflection hit add a scattered contribution toward each cell using
  the same `scattering_jones_operator` and the existing `_rayd_path_unit_field`
  phase; apply the same `√(1−S²)` specular reduction.
- Tests: deterministic-vs-MC scattered-power agreement on a flat plate; S = 0
  regression; energy conservation.

### Stage E — Differentiability

- `S`, `α_R`, `xpd` are smooth scalar parameters, so forward (JVP) and reverse
  (VJP) AD should finite-difference-validate where geometry/reflection
  discontinuities do not. This makes scattering an early differentiability win.
- Wire the three parameters into the parity matrix from
  `plans/29-radiomap-differentiability-parity-plan.md` (material × {scattering} ×
  {forward, reverse}). Confirm gradients flow through the lobe, the energy split,
  and the cross-pol terms.

### Stage F — Higher-order scattering (next increment, not first)

- Scattering after a reflection chain (TX→R…→surface→cell), and reflection after
  scattering. Bounded by the existing `max_bounces` budget. Deferred until
  first-order is validated; flagged here so the operator interface is designed for
  chained composition (`jones_operator_matmul`).

### Stage G — Native / RayD + CUDA mirror (performance, deferred)

- Mirror the operator in the `rayd_complex_polarized_fresnel` reflection
  accumulation backend and its CUDA kernel
  (`deterministic/kernels/reflection/`) so the fast non-AD path matches the
  reference. Follow `standards/31-cuda-kernel-migration-workflow.md`. Deferred:
  not required for correctness, only throughput.

### Stage H — T-matrix model (the moat; out of scope here, interface only)

- A future `TMatrixScatteringModel` implementing `scattering_jones_operator(...)`
  from a full-wave (witwin/maxwell FDFD/FDTD) table, interpolated over
  `(θ_in, φ_in, θ_out, φ_out)` per material, autodiff through the interpolation.
  This plan deliberately builds the parametric model behind the **same interface**
  so Stage H is a model swap, not an integrator change. See the T-matrix
  scattering direction in `plans/01`.

## Non-Goals

- No incoherent power-only ITU diffuse term; the scattered field is coherent
  (complex, phase-carrying) by requirement.
- No heuristic smoothing or ad hoc gradient hacks (repo rule).
- No T-matrix / full-wave coupling in this plan (Stage H is interface-only).
- No new CPU fallback paths; DrJit-native internals.
- Higher-order scattering and the native/CUDA mirror are explicitly later stages.

## Validation Summary

- S = 0 numerical identity to current outputs across existing radiomap tests.
- Energy conservation under S sweep (MC and deterministic).
- Reciprocity and energy-budget unit tests on the operator.
- Coherent-interference test proving phase is carried.
- FD-vs-AD parity for `dS`, `dα_R`, `dxpd` (forward and reverse), folded into the
  plan-29 parity gate.
- Cross-check the diffuse lobe magnitude/shape against the in-tree Sionna RT
  reference used by `plans/26-mc-sionna-parity-acceleration-plan.md`.

## Key Files

| Concern | File |
| --- | --- |
| New scattering operator | `witwin/channel/core/physics/scattering.py` (new) |
| Jones / Fresnel reuse | `core/physics/polarization.py`, `core/physics/wave_math.py` |
| Material schema | `core/witwin/core/material.py` |
| Material plumbing | `channel/core/scene/builder.py`, `core/scene/scene.py`, `core/physics/materials.py` |
| MC integration | `channel/montecarlo/integrators/basic.py`, `montecarlo/trace/reflection.py`, `montecarlo/grid_ops.py` |
| Deterministic integration | `channel/deterministic/reflection/epc.py` |
| Differentiability gate | `plans/29-radiomap-differentiability-parity-plan.md` |
