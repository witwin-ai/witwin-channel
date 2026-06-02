# Jones / Vector Material Model Migration Plan

## Goal

Upgrade the material-aware forward model from the current scalar shortcut to a
physically consistent vector/Jones formulation for reflection and diffraction.

Differentiability is a hard requirement for this migration. The new model must
remain differentiable with respect to:

- geometry
- transmitter position/orientation
- per-object material parameters such as `eps_r` and `sigma_e`

The target is not "make Fresnel look closer to legacy." The target is:

- keep per-object material control
- keep differentiability
- remove scalar/vector inconsistencies in the forward model
- make scalar monitor fields a projection of a physically consistent Jones model

## Why This Change Is Needed

## Current mismatch

Today, the material-aware path still mixes two different representations:

- a scalar complex field used as the primary forward quantity
- an auxiliary complex 3D polarization field used for diagnostics and some path transport

This causes a structural inconsistency:

- reflection can transport a vector field through TE/TM Fresnel basis rotation
- but the scalar field still depends on a scalarized coefficient
- diffraction still uses scalar wedge-face coefficients `R0/Rn`

The old scalar Fresnel shortcut `0.5 * (r_te + r_tm)` was especially problematic
because it can cancel valid reflected energy at normal incidence.

## Desired end state

The forward model should be:

- path transport in a vector/Jones representation
- material interaction represented by Jones operators, not scalar averages
- scalar output derived only at the end through an explicit projection rule

That is the same high-level structure used by Sionna RT.

## Sionna RT Reference Model

We should borrow the physics structure from Sionna RT, not its public API and
not its full implementation stack.

This is a partial-reference strategy:

- copy the physically important ideas
- do not copy Sionna RT blindly
- keep the current `Scene + Tracer + Result` architecture
- keep the current GPU-first differentiable workflow based on Dr.Jit

## Primary references

- Sionna RT developer guide: [Understanding Radio Materials](https://nvlabs.github.io/sionna/rt/developer/dev_custom_radio_materials.html)
  - Shows that radio materials are defined through Jones-matrix wave transformations rather than scalar coefficients.
- Sionna RT EM primer: [Reflection and Refraction](https://nvlabs.github.io/sionna/rt/em_primer.html)
  - Defines the TE/TM basis change and the channel coefficient as a receiver projection of a path transfer matrix.
- Sionna RT source doc: [jones_matrix_to_world_implicit](https://nvlabs.github.io/sionna/_modules/sionna/rt/utils/jones.html)
  - Encodes the key basis-rotation idea `J = R_out * diag(c1, c2) * R_in^T`.
- Sionna RT source doc: [RadioMaterial._specular_reflection_transmission_matrix](https://nvlabs.github.io/sionna/_modules/sionna/rt/radio_materials/radio_material.html)
  - Uses TE/TM coefficients and basis rotators to build the specular Jones matrix.
- Sionna RT source doc: [RadioMaterial._diffraction_matrix](https://nvlabs.github.io/sionna/_modules/sionna/rt/radio_materials/radio_material.html)
  - Builds a diffraction Jones matrix with wedge-face Fresnel responses for both faces.

## What to copy from Sionna RT

- Keep TE/TM coefficients separate as long as possible.
- Build reflection and diffraction as matrix-valued operators.
- Use explicit basis-rotation matrices at every interaction.
- Project to a scalar coefficient only at the receiver/output boundary.

## What to treat as reference only

- Sionna RT's Mitsuba BSDF integration
- Sionna RT's exact implicit/world basis conventions
- Sionna RT's storage format for Jones matrices
- Sionna RT's public scene/material APIs

Our implementation only needs to reproduce the physical structure that matters
for correctness.

## What not to copy directly

- We do not need Mitsuba BSDF plumbing or Sionna's exact implicit basis API.
- We do not need to adopt Sionna's `4x4` real matrix storage.
- We should stay compatible with the current `Scene + Tracer + Result` architecture.

## Proposed Architecture

## Differentiability constraints

The migration must preserve end-to-end differentiability in the material-aware
branch.

Required properties:

- Jones operators must be built from Dr.Jit differentiable quantities.
- Material gathers from per-triangle/per-face tables must stay inside the Dr.Jit AD graph.
- Basis rotations must be differentiable with respect to geometry and ray directions.
- Final scalar projection from Jones field to `result.field.*` must also remain differentiable.

Non-goals for the implementation:

- no heuristic smoothing of discontinuities
- no detached fallback path inside the material-aware forward model except where
  already required by the existing UTD pole/singularity guards

## 1. Canonical internal representation

Use a canonical 2-component complex Jones representation per path segment.

Each propagating segment carries:

- propagation direction `k_hat`
- Jones vector `e = [e_1, e_2]`
- optional Jones derivative vector for slope terms

We should stop treating the scalar field as the primary transport state inside
reflection and diffraction.

## 2. Canonical basis convention

Adopt one internal transverse basis convention for all transport.

Recommended choice:

- internal path basis = local transverse pair derived from ray direction
- monitor/result basis = current global XY projection for backward-compatible outputs

This gives us:

- physical transport internally
- minimal public API churn initially

Later, we can add a true `rx_polarization` projection API without redoing the internals.

## 3. Reflection model

For every specular reflection:

1. build local TE/TM basis from incident direction and face normal
2. compute `r_te`, `r_tm`
3. build Jones reflection operator in the local interaction basis
4. rotate back into the propagated basis of the reflected ray
5. apply the operator to the incoming Jones vector

Scalar output should not be updated with an independent scalar coefficient.
Instead:

- the scalar field is derived from the updated Jones vector by projection
- this removes the current scalar/vector divergence

## 4. Diffraction model

For every diffraction event:

1. keep the existing UTD scalar geometry kernel
2. replace scalar face coefficients `R0/Rn` with Jones face operators
3. assemble a diffraction Jones operator in the edge-local basis
4. rotate from the incident basis to the edge-local basis and back to the outgoing basis

This is the most important change for physical correctness. Reflection-only Jones
transport is not enough if diffraction still collapses materials to scalar `R0/Rn`.

## 5. Scalar output semantics

Short term:

- keep `result.field.*` as a scalar projection derived from the final Jones field
- keep `result.jones.*` exposed

Recommended projection rule for the first migration:

- project onto the current transmit polarization transported to the outgoing ray

Longer term:

- add explicit `rx_polarization` support and define scalar field as
  `c_rx^H * T_path * c_tx`

That matches the Sionna RT channel formulation more closely.

## Concrete Code Changes

## A. Polarization utilities

Primary file:

- `witwin/channel/polarization.py`

Add or refactor:

- basis-rotation helpers for 2-component Jones vectors
- conversion between current complex 3D vectors and Jones bases
- a canonical "path basis" definition for any ray direction
- reflection Jones operator builder
- diffraction edge-local Jones basis builder

Expected outcome:

- one source of truth for basis rotation and operator application

## B. Material/Fresnel utilities

Primary file:

- `witwin/channel/material.py`

Keep:

- complex permittivity
- TE/TM Fresnel coefficient computation

Add:

- differentiable Jones operator builders for reflection
- helper routines that map per-face material parameters to Jones-form interaction operators

Remove as a forward-model dependency:

- scalar averaging as a first-class material model

If retained at all, it should remain only as a diagnostic helper for regression plots.

## C. Reflection solver

Primary files:

- `witwin/channel/trace/materials.py`
- `witwin/channel/trace/reflection/field.py`

Required changes:

- stop using an independent scalar bounce coefficient for material-aware reflection
- use a Jones operator or equivalent vector operator as the only material interaction
- derive scalar path amplitude from the transported Jones state
- keep legacy scalar mode only for the explicit legacy branch

Expected cleanup:

- no more separate "scalar is one thing, vector is another" logic in the material-aware branch

## D. Diffraction state construction

Primary files:

- `witwin/channel/trace/diffraction/geometry.py`
- `witwin/channel/trace/diffraction/builders.py`
- `witwin/channel/trace/diffraction/field.py`
- `witwin/channel/trace/diffraction/suffix.py`

Required changes:

- replace scalar `r0/rn` storage with per-face TE/TM data or Jones operators
- make prefix reflections and inserted/suffix reflections use the same Jones transport path as direct reflection
- update state arrays so diffraction receives Jones incident fields, not only scalar incident fields plus a sidecar vector
- evaluate diffraction output by applying a diffraction Jones operator to the incoming Jones field

Important note:

The current scalar UTD coefficient itself can stay. The migration target is not
rewriting UTD from scratch; it is upgrading material and polarization handling
around the coefficient into a Jones-consistent operator.

Important differentiability note:

- if per-face Jones operators are precomputed before accumulation, they must be
  precomputed in differentiable Dr.Jit code
- if custom-op/Slang interfaces are touched later, backward behavior for
  material parameters must remain covered by tests before any optimization pass

## E. Tracer/result semantics

Primary files:

- `witwin/channel/trace/tracer.py`
- `witwin/channel/result.py`
- `witwin/channel/config.py`

Required changes:

- define one explicit scalar projection rule for `result.field.*`
- expose metadata describing:
  - transport basis
  - scalar projection rule
  - reflection material model
  - diffraction material model

Recommended follow-up API:

- add `rx_polarization` to `TraceConfig` and `Tracer`

Not required for phase 1, but strongly recommended for phase 2.

## Migration Phases

## Phase 0: Freeze the current baseline

Before refactoring:

- keep existing diagnostic plots
- preserve legacy path behavior
- preserve current opt-in material routing

Deliverables:

- baseline comparison figures already in `tests/main/`
- current scalarization comparison figure retained as a regression artifact

## Phase 1: Reflection Jones unification

Scope:

- reflection only
- remove material-aware scalar/vector divergence
- keep diffraction unchanged for the moment

Acceptance goal:

- reflection scalar field equals a projection of the reflected Jones field by construction

## Phase 2: Diffraction face material upgrade

Scope:

- direct diffraction
- reflection-prefix diffraction
- inserted reflections
- reflected diffraction suffix

Acceptance goal:

- wedge-face material response comes from Jones-consistent face operators
- no scalar `R0/Rn` shortcut remains in the material-aware path

## Phase 3: Output projection cleanup

Scope:

- formalize scalar field semantics
- optionally add `rx_polarization`
- document projection rule in metadata and dev docs

Acceptance goal:

- scalar output meaning is explicit and stable

## Phase 4: Performance recovery

Scope:

- remove avoidable basis recomputation
- coalesce Jones storage
- revisit Slang/custom-op interfaces if necessary

Acceptance goal:

- no major regression in the main GPU workloads relative to the current material-aware branch

## Validation And Acceptance Tests

The acceptance plan should be complete enough to block partial or misleading implementations.

## 1. Unit tests: basis and operator math

Add tests covering:

- TE/TM basis construction is orthonormal and transverse
- reflected basis is correctly rotated after specular reflection
- normal incidence does not spuriously cancel the reflected field
- a Jones reflection operator reproduces direct vector transport for known cases

Recommended file:

- `tests/diffraction/test_polarization_and_finite_edge.py`
- or a new `tests/polarization/test_jones_operators.py`

## 2. Reflection regression tests

Add tests covering:

- lossless dielectric single-bounce reflection:
  - scalar field equals projected Jones field
- uniform per-object material scene:
  - scene material and explicit override remain numerically equivalent
- legacy branch:
  - remains unchanged when `use_scene_materials_* = False`

Recommended file:

- `tests/reflection/test_reflection_material_response.py`

## 3. Diffraction regression tests

Add tests covering:

- direct diffraction material response remains finite and differentiable
- material-aware diffraction uses Jones-consistent face transport
- PEC/symmetric cases reduce to the expected scalar baseline
- heterogeneous-face wedges respond differently on the two faces

Recommended file:

- `tests/diffraction/test_fresnel_gradient_regression.py`
- plus a new wedge-focused material test if needed

## 4. AD/FD gradient agreement tests

These are required for:

- reflection material gradients
- diffraction material gradients
- mixed total-field gradients
- projected Jones-to-scalar gradients

Acceptance thresholds should be checked for:

- total field
- reflection component
- diffraction component

And for parameters:

- `eps_r`
- `center_x`
- `rotation_z`
- `tx_x`

These tests are not optional. A Jones/vector migration that improves forward
physics but breaks AD is not acceptable for this repository.

## 5. Jones consistency tests

Add direct consistency tests:

- `result.field.reflection` vs projected `result.jones.reflection`
- `result.field.diffraction` vs projected `result.jones.diffraction`
- `result.field.total` vs projected `result.jones.total`

The material-aware path should satisfy this by design.

Legacy path may continue to differ if we intentionally keep it scalar-only.

## 6. Visual acceptance tests

Update and preserve:

- `tests/main/test_position_rotation_tx.py`
  - keep legacy vs material-aware figure
  - replace scalarization diagnostic emphasis with Jones-consistency diagnostics once the migration lands
- `tests/main/test_material.py`
  - keep total field
  - keep diffraction forward field
  - keep AD/FD gradients
  - keep diffraction zero-crossing diagnostics
  - add field-vs-Jones projection mismatch summary for reflection/diffraction/total

## 7. Canonical physics acceptance tests

Add small, explicit physics cases:

- single reflector, normal incidence:
  - no artificial TE/TM cancellation
- Brewster-angle reflection for TM:
  - reflected TM component approaches zero
- PEC limit:
  - reflection approaches the legacy conductor behavior
- symmetric wedge with PEC faces:
  - diffraction matches the old scalar PEC reference

## 8. Heterogeneous material acceptance tests

Add scene-level tests for:

- one dielectric object + one conductor object
- two wedge faces with different materials
- multi-object reflection-prefix diffraction with distinct face materials

Acceptance goal:

- material change is localized to the relevant interaction surfaces

## 8a. Differentiability acceptance tests

Add explicit checks that:

- `d field / d eps_r` is finite for reflection, diffraction, and total field
- `d jones / d eps_r` is finite for reflection, diffraction, and total field
- projected scalar gradients match FD after Jones-to-scalar projection
- per-object material changes do not break gradient locality

These tests should include both:

- scene-material tables
- explicit material overrides where still supported

## 9. Performance acceptance tests

Track:

- peak memory
- runtime for `tests/main/test_material.py`
- runtime for `tests/main/test_position_rotation_tx.py`
- runtime for at least one mixed diffraction case

The first correctness-first phase can tolerate some slowdown, but we should
record the regression explicitly before optimizing.

## Acceptance Matrix

| Category | Test | Pass condition |
|---|---|---|
| Basis math | Reflection basis/unitarity tests | All orthogonality and projection checks pass |
| Reflection correctness | Projected scalar equals Jones reflection | Relative error below strict numeric tolerance |
| Diffraction correctness | Direct/mixed diffraction Jones consistency | Relative error below strict numeric tolerance |
| Gradient correctness | AD vs FD for total/reflection/diffraction | Existing or tighter material-gradient thresholds pass |
| Legacy stability | Legacy branch regression | No output change for legacy-default tests |
| Scene-material parity | Uniform scene material vs explicit override | Machine-precision or near-machine-precision agreement |
| Canonical EM cases | Normal incidence, Brewster, PEC, wedge | Expected physical signatures observed |
| Performance | Main GPU tests | No undocumented large regression |

## Recommended Initial File Ownership

Worker slices should be kept disjoint if parallelized later.

### Slice 1: Jones math core

- `witwin/channel/polarization.py`
- `witwin/channel/material.py`

### Slice 2: Reflection migration

- `witwin/channel/trace/materials.py`
- `witwin/channel/trace/reflection/field.py`

### Slice 3: Diffraction migration

- `witwin/channel/trace/diffraction/geometry.py`
- `witwin/channel/trace/diffraction/builders.py`
- `witwin/channel/trace/diffraction/field.py`
- `witwin/channel/trace/diffraction/suffix.py`

### Slice 4: Result/config/tests

- `witwin/channel/trace/tracer.py`
- `witwin/channel/config.py`
- `witwin/channel/result.py`
- `tests/...`

## Non-Goals

This plan does not require:

- transmission/refraction in the first phase
- arbitrary diffuse-scattering Jones models
- matching the legacy scalar field when material-aware Fresnel is enabled
- cloning Sionna RT's code structure or API surface

Those are separate product decisions.

## Recommended Execution Order

1. Freeze current diagnostics and metadata.
2. Refactor reflection so the material-aware scalar field is derived from Jones transport.
3. Refactor diffraction face material handling into Jones-consistent operators.
4. Add explicit scalar projection semantics and metadata.
5. Expand unit/acceptance tests.
6. Re-run visual regressions and performance checks.

## Recommended Commands

```bash
conda activate witwin2
cd channel
python -m pytest tests/diffraction/test_fresnel_gradient_regression.py -q
python -m pytest tests/diffraction/test_polarization_and_finite_edge.py -q
python -m pytest tests/reflection/test_reflection_material_response.py -q
python -m pytest tests/main/test_position_rotation_tx.py --gpu -q
python -m pytest tests/main/test_material.py --gpu -q
```

## Exit Criteria

The migration is complete only when all of the following are true:

- no material-aware reflection path uses an independent scalar coefficient
- no material-aware diffraction face response is based on scalar `R0/Rn` shortcuts
- scalar output is documented as an explicit projection of the Jones result
- AD/FD material gradients pass on reflection, diffraction, and total fields
- AD/FD material gradients also pass after Jones-to-scalar projection
- legacy-default tests remain stable
- visual diagnostics show field/gradient behavior explained by the forward physics rather than scalarization artifacts
- the implementation clearly follows Sionna RT only as a physics reference, not as a copied API/design
