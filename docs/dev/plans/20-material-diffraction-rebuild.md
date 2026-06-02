# Sionna RT Material Diffraction Rebuild Plan

## Status

This document defines the target rebuild of the material-aware diffraction system.

It is not a proposal to keep the current mixed scalar/Jones implementation alive.
It is a replacement plan.

This plan supersedes the temporary transition ideas recorded in:

- `docs/dev/archive/superseded/jones-vector-material-model-plan.md`
- `docs/dev/archive/superseded/los-shadow-boundary-jump-analysis-2026-03-29.md`

Those documents remain useful as investigation history, but they are not the
target architecture.

## Goal

Rebuild the material-aware propagation path so that it follows the same physics
structure as Sionna RT:

- no material-aware scalar diffraction shortcut
- no `r0/rn = 0.5 * (r_te + r_tm)` path in the material-aware diffraction model
- no mixed semantics where `result.field.*` comes from one forward model and
  `result.jones.*` comes from another
- LoS, reflection, and diffraction transported in one consistent vector/Jones
  system
- scalar monitor fields produced only once, at the output boundary, from the
  final vector result
- reflection-prefix diffraction generated from exact path descriptors, not from
  receiver-conditioned image-source reconstruction to the edge
- existing higher-order diffraction families must remain valid, including
  multi-diffraction chains and reflection-diffraction chains
- the slope diffraction coefficient used for normal-derivative transport must
  remain a first-class truth in the rebuilt system

Physical correctness and internal consistency take priority over backward
compatibility.

## Non-Goals

- Preserve the current material-aware scalar shortcut.
- Preserve the current metadata strings if they encode obsolete behavior.
- Keep temporary compatibility branches such as
  `material_scalar_utd_face_coefficients` or
  `mixed_implicit_tx_copolar_jones_and_material_scalar`.
- Add new smoothing heuristics, continuity hacks, or branch-specific fixes.
- Approximate Sionna RT visually while keeping a different internal model.
- Narrow the solver to first-order diffraction only.
- Replace the slope diffraction coefficient with a heuristic or finite-difference
  approximation.

## Design Rules

- The stable public architecture remains `Scene + Tracer + Result`.
- The internal forward truth for the material-aware path is vector/Jones only.
- A path-dependent scalar shortcut is not allowed inside the material-aware
  diffraction path.
- Receiver scalarization must be applied after vector accumulation, not before.
- The material-aware system must not maintain two parallel truths.
- The wedge diffraction operator and the face reflection model must be separate
  layers so the material model can evolve without rewriting the operator.
- The rebuilt first-order material operator must preserve the higher-order
  diffraction pipeline rather than bypass it.
- The slope diffraction coefficient is mandatory because it drives
  normal-derivative transport for higher-order diffraction.
- If a behavior change is user-visible, update `FEATURE_LIST.md` only after the
  rebuilt system is the active implementation.

## Why The Current Architecture Must Be Replaced

### 1. The scalar material-aware diffraction coefficient is not physically invariant

The current scalar shortcut uses:

```text
0.5 * (r_te + r_tm)
```

This is not a basis-invariant field quantity. `r_te` and `r_tm` live in a local
interaction basis. Averaging them before basis transport destroys the canonical
soft/hard channel structure required by wedge diffraction.

At high permittivity the problem becomes severe:

- `r_te` tends toward approximately `-1`
- `r_tm` tends toward approximately `+1`
- the average tends toward approximately `0`

This can suppress a physically dominant face term and force the diffraction
field onto the wrong branch.

### 2. The current material-aware path has two incompatible truths

Today the code can produce:

- a scalar field from a scalar shortcut
- a Jones/vector field from a dyadic operator

These are not the same forward model. Once they diverge, boundary continuity
cannot be reasoned about cleanly.

### 3. Reflection-prefix diffraction is derived from the wrong geometric object

The current reflection-prefix diffraction flow reconstructs edge support from
receiver-conditioned reflection image sources. This is structurally wrong for
prefix diffraction.

The reflection path descriptor used for receiver reconstruction is not the same
thing as a physically valid edge-source descriptor. As a result, valid
reflection-prefix diffraction states can be pruned before any material operator
is evaluated.

## Sionna RT Reference Model

The reference is Sionna RT's wedge diffraction flow:

- `radio_material.py::_diffraction_matrix`
- `field_calculator.py` diffraction interaction setup

The key ideas to copy are:

- keep `TE/TM` coefficients separate
- build a canonical wedge diffraction operator first
- use explicit basis rotators for each face term
- convert to a world/receiver basis only after the operator is assembled
- perform scalarization only at the final observation boundary

This repository has one requirement beyond the Sionna first-interaction model:

- the rebuilt material operator must also expose the slope / normal-derivative
  transport quantities needed by the existing higher-order diffraction solver

The operator structure to mirror is:

```text
phi_hat_prime = normalize(cross(k_i, e_hat))
phi_hat       = -normalize(cross(k_o, e_hat))

D = d12 I
  + W0_out * diag(r_te_0 * d4, r_tm_0 * d4) * W0_in
  + Wn_out * diag(r_te_n * d3, r_tm_n * d3) * Wn_in
```

where:

- `e_hat` is the edge direction
- `n0` and `nn` are the wedge face normals
- `d12 = -(d1 + d2)`
- `d3` and `d4` are the face-linked UTD terms

## Material Model Parity Note

Sionna RT does not stop at a half-space Fresnel interface. Its current material
implementation uses a single-layer slab response model derived from ITU-style
material parameters, including layer thickness.

The current repository does not expose a slab-thickness field in the material
model used by the channel solver. Today the accessible face response is the
half-space Fresnel coefficient family implemented in
`witwin/channel/material.py`.

The rebuild must therefore separate two concerns:

- the canonical wedge diffraction operator
- the face reflection response model that provides `r_te` and `r_tm`

The first operator-integration phase may use the existing half-space Fresnel
response as the face model so the dyadic architecture can be rebuilt first.
That stage is not full Sionna material parity. It is only Sionna-style operator
parity.

If full Sionna parity is required, a later material-model upgrade must:

- extend the public material model with layer thickness or an equivalent slab
  descriptor
- provide a slab-based face response for reflection and diffraction
- plug that response into the same operator interface without changing the
  canonical wedge assembly

This difference must be documented explicitly in tests and in developer notes
until the slab model exists.

## Normalization Contract

The rebuild must not assume that matching the symbolic operator structure is
sufficient. The end-to-end amplitude convention must also be aligned.

The following pieces must be treated as one contract:

- incident-field normalization
- spreading factor definition
- phase convention
- wavelength scaling
- the location where any `lambda / (4 pi)`-style factor is applied

Operator validation must check the complete product of these terms. It is not
enough to compare only `d1..d4` or only the local Jones matrix entries if the
overall field normalization still differs.

This document does not assume that the current spreading factor is wrong by
inspection alone. Instead, the rebuild requires an explicit normalization test
that compares the complete operator-to-field mapping against the chosen
reference convention.

## Target End State

### Forward model

The material-aware forward path has exactly one truth:

- LoS produces a vector/Jones field
- reflection produces a vector/Jones field
- diffraction produces a vector/Jones field
- totals are formed by vector accumulation in a common world basis

`result.field.*` is then derived from the final vector totals by one explicit
scalar projector.

### Output model

- `result.vector.*` or the existing internal equivalent becomes the true field
  quantity for all path families
- `result.jones.*` remains a compatibility view, derived from the final vector
  field in global `XY`
- `result.field.*` is derived from the final vector field through a single
  scalarization rule

There is no separate scalar material-aware diffraction solver.

## Required Architectural Changes

## 1. Unified internal field representation

The primary internal quantity should be:

- a `2 x 1` Jones field carried in a local propagation basis
- plus the propagation basis itself

The preferred internal pattern is:

- propagate per-path interactions in local Jones bases
- convert each completed path contribution to a world vector at the receiver
- accumulate only world vectors across path families

This avoids:

- repeated global `XY` projections in the middle of transport
- path-family-specific scalar shortcuts
- accidental mixing of local and world semantics

Direct accumulation in a 3D complex vector everywhere is not recommended as the
primary transport representation. Jones transport is the right interaction-space
representation. World-vector accumulation is the right cross-path sum space.

## 2. Material diffraction operator rewrite

Create a dedicated module for the rebuilt operator, for example:

- `witwin/channel/trace/diffraction/operator.py`

or:

- `witwin/channel/trace/diffraction/material_operator.py`

Do not continue stacking patches inside the current `field.py` material path.

The new operator module should own:

- canonical wedge basis construction
- canonical `d1..d4` coefficient grouping
- face-basis rotator construction
- dyadic operator assembly
- slope diffraction coefficient assembly for normal-derivative transport
- any derivative terms required by the existing higher-order diffraction chain

The operator must consume geometry and material inputs and produce one operator
truth. It must not emit scalar face coefficients.

The operator API should be designed around a pair of outputs:

- the primary field diffraction operator
- the slope / normal-derivative diffraction operator used to excite the next
  diffraction order

These two outputs must come from the same canonical geometry and material
inputs. They must not be computed from unrelated scalar shortcuts.

## 3. Geometry contract for material diffraction

The geometry layer must expose the quantities needed by the canonical dyadic
operator directly:

- `e_hat`
- `n0`
- `nn`
- `phi`
- `phi_prime`
- `beta0` or the equivalent `sin(beta0)`
- `s`
- `s_prime`
- `d1`
- `d2`
- `d3`
- `d4`

The material-aware path should stop calling a scalar coefficient builder that
expects `R0` and `Rn`.

`utd.py` should be refactored so the material-aware path uses canonical grouped
terms rather than scalar coefficient wrappers. The old scalar wrappers may
remain only if they are still needed by a separate non-material legacy path, but
they must not influence the rebuilt material-aware system.

## 4. State payload redesign

`state.py` for material-aware diffraction must stop storing these as primary
fields:

- `r0`
- `rn`
- pre-baked `face0_operator_m**`
- pre-baked `face1_operator_m**`

The material-aware diffraction state should instead store:

- incident Jones field
- incident Jones basis
- edge geometry identifiers
- adjacent face identifiers
- source/path descriptor identifiers
- source type and order bookkeeping

For higher-order transport the state must also carry, or be able to reconstruct
without ambiguity:

- the incident direction and basis needed for slope transport
- the operator inputs needed to evaluate the slope diffraction coefficient
- the diffraction order and parent interaction descriptor

The state should not cache an outgoing-operator result that depends on the
receiver direction `k_o`.

However, the state is allowed to cache face-local data that does not depend on
the outgoing receiver direction. In particular, the rebuilt state may cache:

- `r_te_0`, `r_tm_0`, `r_te_n`, `r_tm_n`
- face normals or other face-local basis inputs
- incident-angle-dependent face response terms

This is the preferred caching boundary for performance. It avoids repeated
material lookups and repeated Fresnel evaluation while still forbidding cached
operators that already include the outgoing rotation.

Material parameters may be recovered from adjacent faces or from an exact
prefix-path descriptor at evaluation time, but the rebuilt design does not
require recomputing every face response from face IDs on every accumulation.
What is forbidden is caching scalarized truth fields or receiver-dependent
rotated operators.

## 5. Reflection path descriptor redesign

The current reflection path storage in `reflection/paths.py` is not sufficient
for prefix diffraction, because it clusters paths by averaged image source and
then reuses that value as though it were an exact edge-target descriptor.

The rebuilt reflection path descriptor must store exact geometric information
for replay:

- primitive chain
- per-face plane point
- per-face plane normal
- exact recursively constructed image source from `tx_pos`
- optional representative hit points
- optional path multiplicity count for diagnostics only

The critical rule is:

- do not average image sources and then use that averaged point as the exact
  source for edge diffraction geometry

If path merging is retained for performance, it must preserve EPC
semantics. If that cannot be guaranteed, do not merge.

## 6. Reflection-prefix diffraction generation rewrite

The current failure point is:

- `geometry.py::_triangle_surface_intersection(image_source, edge_pos, prim_idx, scene)`

This path assumes that a receiver-conditioned image source can be replayed to an
arbitrary edge position through an exact specular hit. That is too strict and
not the right abstraction for prefix diffraction.

The rebuilt system should replace `_evaluate_reflection_prefix_chain(...)` with
a shared exact path-replay utility, conceptually:

```text
epc_reflection_chain_to_target(path_descriptor, target_pos)
```

This utility should:

- start from the exact prefix descriptor
- replay the reflection chain toward the target of interest
- return exact interaction points when they exist
- return the incident Jones field and basis at the target
- avoid any scalar `chain_weight + chain_vector` split

This same utility should be usable for:

- exact reflection accumulation to receivers
- reflection-prefix diffraction excitation at edges
- reflection-prefix excitation of higher-order diffraction states when a
  reflected path feeds a later diffraction event

That unifies reflection and prefix-diffraction physics instead of duplicating
two similar but inconsistent pipelines.

## 7. LoS / reflection / diffraction accumulation rewrite

`tracer.py` should stop treating scalar fields as the primary total.

The new accumulation order should be:

1. compute LoS vector/Jones contribution
2. compute reflection vector/Jones contribution
3. compute diffraction vector/Jones contribution
4. sum them in the world vector basis
5. derive:
   - scalar fields
   - global `XY` Jones views
   - metadata about the final projector

There must be no path-family-specific scalar accumulation before step 4.

The same accumulation architecture must continue to support:

- direct higher-order diffraction
- reflection-diffraction chains
- any existing mixed family that depends on slope-based normal-derivative
  transport

## 8. Final scalarization rule

The rebuilt system needs one explicit scalarization rule at the output layer.

### Explicit receiver polarization

If `rx_polarization is not None`:

- project the final world vector onto that receiver polarization

### Implicit default receiver

If `rx_polarization is None`:

- define exactly one fixed default receiver basis
- do not use a path-dependent receiver projection
- do not project each path in its own ray basis and then pretend the result is a
  final scalar observable

The recommended rule is:

- use a fixed world-space receive polarization derived from the normalized
  transmitter polarization, or another single documented receiver convention

The important requirement is not the specific basis choice. The important
requirement is that it be unique, documented, and path-independent so that
final-after-sum scalarization is physically and algebraically well-defined.

## File-Level Change Plan

## Files to replace or heavily refactor

- `witwin/channel/trace/diffraction/field.py`
- `witwin/channel/trace/diffraction/geometry.py`
- `witwin/channel/trace/diffraction/state.py`
- `witwin/channel/trace/diffraction/builders.py`
- `witwin/channel/trace/reflection/paths.py`
- `witwin/channel/trace/reflection/field.py`
- `witwin/channel/trace/tracer.py`

## Files to split

- move material diffraction operator assembly into a dedicated module
- move shared reflection-chain replay logic into a dedicated reusable utility

## Files that should lose authority over the material-aware path

- `witwin/channel/material.py::scalar_fresnel_reflection`
- scalar `R0/Rn` helpers as the material-aware diffraction truth

These may still exist if another unrelated path still needs them, but they must
not drive the rebuilt material-aware system.

## Files that may need material-model extension later

- `witwin.core.Material` or its effective scene-material storage
- `witwin/channel/material.py`
- reflection material helpers that currently assume half-space Fresnel only

These are not required to rebuild the dyadic architecture with the current
half-space face model, but they will be required if the project later adopts
slab-response parity with Sionna RT.

## Migration Sequence

## Phase 0: Cleanup and freeze

- keep `tests/support/bin/isb_crosssection.py` as a manual diagnostic script
- remove temporary mixed scalar/Jones material-aware patches
- avoid committing transitional metadata or threshold changes

## Phase 1: Basis, sign, and normalization contract

- keep `jones_rotator`
- keep `jones_operator_multiply`
- add any missing canonical wedge basis helpers
- add explicit convention tests for `diffraction_edge_basis()` versus the
  Sionna-style `phi_hat_prime` / `phi_hat` definitions
- define the end-to-end normalization contract before mainline integration
- ensure all helpers stay Dr.Jit-native and differentiable

## Phase 2: Diffraction operator module

- expose canonical grouped UTD terms
- implement Sionna-style dyadic operator assembly
- introduce a face-response abstraction so the operator can consume either the
  current half-space Fresnel response or a future slab response
- expose the slope diffraction coefficient from the same canonical model
- validate the operator numerically on canonical wedge cases before integrating
  it into the main tracer

## Phase 3: Direct diffraction migration

- migrate direct TX diffraction states to the new incident-Jones payload
- remove scalar `r0/rn` dependence from the material-aware direct path
- confirm ISB behavior in world vector space before touching final scalarization

## Phase 4: Tracer/output migration

- make vector/Jones totals the primary internal result
- derive `result.field.*` only after total vector accumulation
- derive `result.jones.*` from the final vector totals

## Phase 5: Reflection exact-path descriptor

- redesign `reflection/paths.py` to store EPC descriptors
- stop averaging image sources for prefix-diffraction use
- define a replay utility that returns exact Jones transport to an arbitrary
  target
- keep the descriptor interface aligned with higher-order diffraction needs

## Phase 6: Reflection-prefix and higher-order diffraction migration

- rebuild prefix state generation on EPC descriptors
- confirm nonzero reflected-prefix states on the current box scene
- validate RSB continuity in world vector space
- confirm the rebuilt prefix flow still feeds higher-order diffraction when the
  reflected prefix is not the terminal interaction

## Phase 7: Validation and cleanup

- remove dead scalar material-aware code paths
- simplify metadata to reflect the single rebuilt truth
- update tests and `FEATURE_LIST.md`

## Validation Strategy

## 1. Basis and convention validation

Add deterministic tests that verify the local basis conventions explicitly:

- incoming `diffraction_edge_basis()` matches
  `phi_hat_prime = normalize(cross(k_i, e_hat))`
- outgoing `diffraction_edge_basis()` matches
  `phi_hat = -normalize(cross(k_o, e_hat))`
- the corresponding `v` axis preserves the intended handedness
- face-local soft/hard basis construction matches the canonical rotator
  convention used by the operator

These tests should fail on any sign flip or hidden basis swap before operator
validation even begins.

## 2. Operator-level validation

Add deterministic canonical wedge tests that compare our assembled operator to a
direct local implementation of the Sionna formula:

- same geometry
- same material inputs
- same `d1..d4`
- same basis rotators
- same output matrix

This test should validate operator entries directly, not just final field plots.

Add a paired operator test for the slope coefficient:

- same canonical geometry
- same material inputs
- same differentiation or closed-form slope terms
- same output slope operator/coefficient entries

Add a normalization-contract test that compares the complete field factor:

- incident normalization
- spreading factor
- phase factor
- wavelength scaling
- final operator-to-field amplitude

This test should document which face-response model is being used:

- half-space Fresnel parity in the first implementation phase
- slab-response parity only after the material model supports thickness

## 3. ISB validation

Validate in two stages:

- world-vector continuity first
- scalarized continuity second

Required checks:

- the lit-side ideal diffraction vector should be close to
  `shadow total - lit GO`
- the active canonical channel should not flip to the wrong branch
- final scalar output should be a pure projection of the final vector result

## 4. RSB validation

Add a dedicated regression that first checks:

- reflection-prefix diffraction states are nonzero on the current box scene

Only after that should it check:

- reflection-boundary continuity
- reflection-diffraction continuation into the next diffraction order when the
  scene configuration supports it

This prevents a silent regression where the states vanish and the total field is
then debugged at the wrong layer.

## 5. Accumulation identity validation

For LoS, reflection, diffraction, and total:

```text
field_scalar == scalarize(final_world_vector)
```

This must hold numerically for every path family and for the total field.

There should be no special-case scalar path that can disagree with the vector
truth.

## 6. Higher-order diffraction validation

Add dedicated regressions for the existing multi-order solver families:

- direct multi-diffraction states remain populated
- reflection-diffraction states remain populated
- slope-based normal-derivative transport remains numerically finite and
  directionally consistent
- higher-order totals remain derived from the same vector/Jones truth as
  first-order totals

These tests must exist before the legacy material shortcut is removed, so the
rebuild cannot silently regress higher-order support while fixing first-order
ISB or RSB behavior.

## 7. Manual diagnostics

Keep the following as manual support tools during migration:

- `tests/support/bin/isb_crosssection.py`

Upgrade it when needed so it can plot:

- scalar field cross-sections
- vector norm or selected vector components
- global `XY` Jones power
- higher-order path-family slices when debugging multi-diffraction continuity

Manual plots should support the tests. They should not be the only evidence that
the rebuild is correct.

## Acceptance Criteria

The rebuild is complete only when all of the following are true:

- material-aware diffraction no longer uses scalar `R0/Rn` as its truth
- direct and prefix diffraction both use the dyadic material operator
- reflection-prefix states exist and contribute in the current RSB scene
- higher-order diffraction and reflection-diffraction families still work after
  the rebuild
- the slope diffraction coefficient remains the source of truth for
  normal-derivative transport
- the face reflection model is a separate layer from the wedge operator
- the active implementation documents whether it is using half-space Fresnel or
  slab-response parity
- the end-to-end normalization contract is validated rather than assumed
- ISB continuity is explained by one vector truth, not by mixed scalar/Jones
  paths
- `result.field.*` is always derived from final vector accumulation
- `result.jones.*` is a derived compatibility view, not a parallel solver truth
- no temporary mixed scalar/Jones metadata remains
- no backward-compatibility shim is kept solely to preserve the old material
  diffraction behavior

## Explicit No-Legacy Policy For This Rebuild

This rebuild should not preserve the current material-aware implementation as a
fallback.

Do not keep:

- a legacy material-aware scalar diffraction branch
- a mixed scalar/Jones output mode
- compatibility flags that select between old and new material-aware diffraction
- transition-only metadata strings after the new path is active

The rebuilt material-aware diffraction path should replace the current one.

If a compatibility concern appears, the default answer is to delete the obsolete
branch and update the tests, unless there is a strong externally visible public
API reason to keep it.
