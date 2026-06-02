# Multi-Order Diffraction Completeness Checklist and Task Plan

Date: 2026-03-16

## Scope

This document tracks the remaining work for multi-order diffraction, mixed reflection/diffraction paths, and physical-fidelity improvements in the current solver.

It is based on the current implementations in:

- `witwin/channel/trace/tracer.py`
- `witwin/channel/trace/reflection/field.py`
- `witwin/channel/trace/diffraction/`
- `witwin/channel/trace/diffraction/utd.py`
- `witwin/channel/scene/`
- `witwin/channel/validation.py`
- `tests/*`
- `sionna-rt-reference/*`

This document focuses on:

1. Path-family completeness
2. Physical accuracy
3. Validation completeness
4. Engineering and implementation order

Important constraint:

- Do not introduce soft smoothing or heuristic smoothing for gradients or field continuity.
- If boundary treatment is needed, prefer exact geometric/boundary formulations or mathematically justified uniform diffraction formulations.

## Current Status Snapshot

| Topic | Current status | Assessment |
|---|---|---|
| `S -> D` direct diffraction | Implemented | Usable baseline |
| `R^n -> D` reflection-prefix diffraction | Implemented through sampled path-faithful reflected image-source chains with chain backtracking | Approximate, but materially and phase consistent |
| `D -> D`, `D -> D -> D` | Implemented through recursive edge-state propagation | Engineering approximation, not closed-form multi-wedge theory |
| `... -> D -> R^n` | Implemented through reflected suffix tracing | Approximate Monte Carlo suffix, with Fresnel/material response carried in both scalar and Jones/vector fields |
| `R -> D -> R` | Can appear from current prefix + suffix combination | Approximate |
| `D -> R -> D` | Implemented through sampled inserted-reflection diffraction states | Approximate, but now present and auditable |
| Mixed-path ownership / duplicate-count control | Implemented through explicit field-component ownership rules and state-audit ownership labels | `a_ref`, `a_dif_direct`, and `a_dif_mixed` are now disjoint by construction |
| Mixed-path budget / pruning policy | Implemented as explicit deterministic state-budget caps with metadata-visible reports | Default-off, contribution-ranked pruning |
| Arbitrary alternating mixed chains | Implemented through per-diffraction reflection-depth histories and configurable inserted-reflection depth budgets | Approximate alternating-chain coverage with one sampled reflection allowed between consecutive diffraction events |
| Explicit solver modes | Implemented through `accuracy` and `fast_approximate` tracer modes | Requested vs effective controls are now visible in `solver_metadata.solver_mode` |
| State-explosion guardrails | Implemented through fast-mode caps plus profiling metadata | Per-order state counts, estimated peak memory, and applied guardrails are visible in `solver_metadata.performance_guardrails` |
| Boundary/open-edge policy | Implemented through explicit `boundary_edge_policy` selection on `Scene` and tracer metadata | Default `exclude`; optional `half_plane` approximation is documented and audited |
| Shadow-boundary treatment | Implemented through exact wedge-face half-space classification in the plane perpendicular to the edge | Diffraction validity now uses geometric source/target exterior tests instead of hard `phi/phi_prime` angular clipping |
| Diffraction-angle derivative treatment | Implemented through analytic chain-rule derivatives of the scalar UTD approximation | Production solver no longer uses finite-difference angle derivatives in `diffraction/utd.py` |
| Full 3D non-vertical edge diffraction | Implemented approximately through `edge_selection_mode="all_edges"` and wedge-plane distance/angle evaluation | Generic non-vertical interior edges now participate in diffraction, but later physics items still remain |
| Finite-edge endpoint policy | Implemented through explicit `finite_edge_policy` control and metadata-visible treatment notes | Default `infinite_wedge`; optional `exclude_clamped_anchors` suppresses generic-edge anchors that land on geometric endpoints |
| Reflection material/angle dependence | Implemented with Fresnel reflection and Jones-vector transport | Material- and angle-aware; reflection coefficients use TE/TM Fresnel response |
| Diffraction material dependence (`R0`, `Rn`) | Implemented with scalar face-response terms | Material-aware, but both wedge faces still share one material model |
| Polarization/Jones transport | Implemented across LoS, reflection, and diffraction, including mixed suffix paths | Result surfaces now expose global-XY Jones projections, while diffraction still uses the current scalar UTD coefficient lifted into an edge-local vector transport |
| Closed-form higher-order validation | Explicit double- and triple-diffraction expansion references plus first-order Sionna overlap are implemented | Current plan coverage is complete; broader literature-backed validation can still be extended later |

## Current Verification Snapshot

Observed targeted local regression status after the current completed Phase 0 / Phase 1 / Phase 2 / Phase 3 / Phase 4 / Phase 5 work:

- Passing:
  - `tests/test_polarization_and_finite_edge.py`
  - `tests/test_diffraction_conventions.py`
  - `tests/test_reflection_material_response.py`
  - `tests/test_reflection_prefix_diffraction.py`
  - `tests/test_mixed_diffraction_toggle.py`
  - `tests/test_rd_multipath_consistency.py`
  - `tests/test_diffraction_order_control.py`
  - `tests/test_diffraction_order_breakdown.py`
  - `tests/test_drd_inserted_reflection.py`
  - `tests/test_mixed_path_budget_ownership.py`
  - `tests/test_mixed_path_regression_scenes.py`
  - `tests/test_alternating_mixed_chain_generalization.py`
  - `tests/test_slope_diffraction_propagation.py`
  - `tests/test_validation_state_audit.py`
  - `tests/test_validation_references.py`
  - `tests/test_solver_modes_and_guardrails.py`
  - `tests/test_boundary_edge_policy.py`
  - `tests/test_shadow_boundary_treatment.py`
  - `tests/test_utd_angle_derivatives.py`
  - `tests/test_generic_edge_diffraction.py`
  - Targeted regression slice status: `42 passed`

Current verification meaning:

- The previously stale slope test has been repaired and is green again.
- The suite now protects state history, sampled reflection-prefix bookkeeping, generalized alternating mixed states, ownership labeling, deterministic pruning reports, dedicated mixed-path family reconstructions, explicit double/triple-diffraction references, first-order Sionna overlap checks, geometric shadow-boundary classification, analytic diffraction-angle derivatives, generic non-vertical edge selection, finite-edge policy selection, Jones/polarization transport including suffix reflection coupling, material-aware reflection/diffraction response, boundary-edge policy selection, order breakdown reconstruction, and convention invariants.
- Full-suite coverage is still incomplete. In particular, there is not yet literature-backed validation for arbitrary higher-order wedge configurations or a full polarized-diffraction reference benchmark.

## Reference Boundary

The local Sionna RT reference is useful for first-order overlap cases, but it is not a higher-order mixed-path reference for this project. In particular, `sionna-rt-reference/src/sionna/rt/path_solvers/paths_buffer.py` states that diffraction is only supported for first order.

Implication:

- Matching Sionna RT is useful for first-order diffraction and shared conventions.
- It is not sufficient to validate this repository's custom higher-order or mixed-path extensions.

## Rating Scale

### Importance

- `Critical`: Directly blocks physical credibility or causes major missing path families
- `High`: Strong impact on accuracy or completeness, but solver still runs without it
- `Medium`: Improves realism, robustness, or maintainability, but is not the first blocker
- `Low`: Nice-to-have, performance-only, or late-stage refinement

### Difficulty

- `Small`: 1-2 focused coding sessions
- `Medium`: Several coordinated edits plus tests
- `Large`: Multi-file solver changes with nontrivial validation work
- `Very Large`: Architectural changes or research-level implementation/validation

## Comprehensive Checklist

### A. Foundations and Bookkeeping

#### [x] C01. Formalize the interaction-path taxonomy and naming

- Importance: `Critical`
- Difficulty: `Medium`
- Why it matters:
  - The current code supports some path families implicitly, but the solver does not expose a complete path taxonomy.
  - Without a formal matrix, mixed-path implementation will likely omit or double-count families.
- Deliverable:
  - A path-family table covering `S`, `R`, `D`, and all allowed alternations up to a chosen depth budget
  - A naming convention for "implemented exactly", "implemented approximately", and "not implemented"
- Notes:
  - This is a prerequisite for all later completeness work.

#### [x] C02. Split solver claims into exact vs approximate coverage

- Importance: `High`
- Difficulty: `Small`
- Why it matters:
  - The current solver already goes beyond first-order diffraction, but several families are engineering approximations rather than exact path solutions.
  - The documentation and outputs should reflect that distinction explicitly.
- Deliverable:
  - Documentation and result metadata that state whether each family is exact, approximate, or absent

#### [x] C03. Extend `state_audit` to record full mixed-path metadata

- Importance: `High`
- Difficulty: `Medium`
- Why it matters:
  - Current audit mainly records edge history and order.
  - It does not fully expose reflection-prefix depth, reflection-suffix depth, or full interaction sequence.
- Deliverable:
  - Additional audit fields such as `path_sequence`, `prefix_reflection_depth`, `suffix_reflection_depth`, `source_type`, and `approximation_mode`

### B. Reflection-Prefix Accuracy

#### [x] R01. Replace per-triangle reflected-source clustering with path-faithful reflected sources

- Importance: `Critical`
- Difficulty: `Large`
- Why it matters:
  - Current `R^n -> D` is built from per-triangle source clusters, not from exact specular chains.
  - Averaging reflected sources on the same triangle smears phase and geometry.
- Deliverable:
  - Either exact per-path reflected sources
  - Or chain-aware clustering that preserves phase, chain identity, and geometry much better than the current mean-like cluster
- Main risk:
  - State count can explode if every reflected path is preserved naively.

#### [x] R02. Support exact cumulative image sources for multi-bounce reflection prefixes

- Importance: `Critical`
- Difficulty: `Large`
- Why it matters:
  - Current reflection detail does not store exact cumulative image-source chains for `R^2`, `R^3`, ...
  - This makes higher-bounce reflection-prefix diffraction physically weaker than it should be.
- Deliverable:
  - A representation of exact or chain-faithful cumulative image sources for each reflection prefix
- Dependency:
  - Strongly related to `R01`

#### [x] R03. Preserve complex amplitude and phase in reflection detail

- Importance: `Critical`
- Difficulty: `Medium`
- Why it matters:
  - Current reflection detail reduces reflected-source strength to scalar amplitude-like weights.
  - Prefix diffraction should inherit complex field state, not only a scalar magnitude proxy.
- Deliverable:
  - Reflection detail carries complex weights or equivalent phase-consistent state
- Dependency:
  - Best implemented together with `R01` and `R02`

#### [x] R04. Replace single-plane reflected-path support checks with exact chain backtracking

- Importance: `High`
- Difficulty: `Large`
- Why it matters:
  - Current support checks for reflected prefixes rely on local geometry tests around the last reflecting primitive.
  - For multi-bounce prefixes, exact validity requires backtracking the whole specular chain.
- Deliverable:
  - Exact or chain-faithful validation for `R^n -> D`

### C. Missing Mixed-Path Families

#### [x] M01. Add `D -> R -> D`

- Importance: `Critical`
- Difficulty: `Large`
- Why it matters:
  - This is the most obvious missing family after the current prefix/suffix implementation.
  - Without it, the solver still misses important mixed interactions in cluttered scenes.
- Deliverable:
  - Support for a reflection event inserted between two diffraction events
- Dependency:
  - `C01`, `C03`

#### [x] M02. Generalize to arbitrary alternating mixed chains up to a depth budget

- Importance: `Critical`
- Difficulty: `Very Large`
- Why it matters:
  - Long-term completeness requires a unified solver model for chains like `R-D-R-D`, `D-R-D-R`, and similar sequences.
  - This is the real architectural step beyond ad hoc prefix/suffix composition.
- Deliverable:
  - A unified path-state machine or path-expansion engine with configurable depth budget
- Notes:
  - This is likely the largest solver change in the roadmap.

#### [x] M03. Add duplicate-counting controls across `a_ref`, direct diffraction, and mixed diffraction

- Importance: `High`
- Difficulty: `Medium`
- Why it matters:
  - Once more mixed families are added, the risk of counting the same physical path through multiple code paths increases.
  - This is especially important if exact mixed-chain reconstruction and suffix sampling coexist.
- Deliverable:
  - Path identity rules
  - Duplicate suppression or ownership rules for each family

#### [x] M04. Add path-budget and pruning strategy for mixed-path combinatorics

- Importance: `High`
- Difficulty: `Large`
- Why it matters:
  - Exact mixed chains can grow combinatorially.
  - Without principled pruning, the accurate solver mode will become unusable.
- Deliverable:
  - A path-budget policy based on depth, contribution estimate, or deterministic culling rules
- Constraint:
  - Pruning must not silently change solver semantics without audit visibility.

### D. Physics Fidelity

#### [x] P01. Replace global scalar reflection coefficient with angle- and material-dependent Fresnel reflection

- Importance: `Critical`
- Difficulty: `Medium`
- Why it matters:
  - Current reflection uses a single global scalar `reflection_coef`.
  - This omits incidence angle, material, and polarization behavior.
- Deliverable:
  - Per-interaction reflection coefficient based on local incidence angle and material properties

#### [x] P02. Feed wedge-face material response into diffraction through `R0` and `Rn`

- Importance: `High`
- Difficulty: `Medium`
- Why it matters:
  - The diffraction coefficient currently behaves like a PEC-style wedge by default.
  - Non-metallic wedge faces should alter the diffraction coefficient.
- Deliverable:
  - Material-aware `R0` and `Rn` handling in diffraction evaluation
- Dependency:
  - Material model definition compatible with `P01`

#### [x] P03. Add polarization/Jones-vector transport across LoS, reflection, and diffraction

- Importance: `High`
- Difficulty: `Very Large`
- Why it matters:
  - Scalar fields are not enough for physically faithful multipath in general scenes.
  - Reflection and diffraction both depend on polarization.
- Deliverable:
  - Vector/Jones-field transport through the whole solver
- Notes:
  - This is a major architecture upgrade.
- Current implementation note:
  - `Tracer.trace()` now exposes `jones_los`, `jones_ref`, `jones_dif_direct`, `jones_dif_mixed`, `jones_dif`, and `jones_tot`.
  - LoS transport projects the transmit polarization into the plane transverse to each propagation ray.
  - Reflection transport uses TE/TM Fresnel response with basis rotation in the reflection plane.
  - Diffraction transport uses the current scalar UTD coefficient lifted into an edge-local vector transport, and mixed suffix reflection now propagates Jones/vector fields together with the scalar suffix field.

#### [x] P04. Replace hard shadow-boundary truncation with exact boundary-consistent treatment

- Importance: `High`
- Difficulty: `Very Large`
- Why it matters:
  - The current validity gating still uses hard angular masks around wedge regions.
  - This is a known weak point near RSB/ISB transitions.
- Deliverable:
  - Exact boundary handling or mathematically rigorous uniform treatment
- Constraint:
  - No soft smoothing
  - No heuristic smoothing
- Current implementation note:
  - Diffraction source/target validity is now classified by exact wedge-face half-space tests in the plane perpendicular to the edge.
  - The production solver no longer uses hard `phi/phi_prime` range clipping to decide whether a diffracted contribution is allowed.
  - Boundary cases remain unsmoothed: targets on the wedge boundary are retained by the geometric classification, while solid-interior targets are rejected exactly.

#### [x] P05. Replace finite-difference angle derivatives with analytic or otherwise rigorous derivatives

- Importance: `Medium`
- Difficulty: `Large`
- Why it matters:
  - Slope-channel propagation must not rely on fragile angle-step tuning in `diffraction/utd.py`.
  - A direct derivative implementation is more robust and easier to audit near higher-order recursion.
- Deliverable:
  - Analytic derivatives if available
  - Or a mathematically cleaner derivative implementation that avoids fragile finite-difference behavior

#### [x] P06. Support non-vertical edges and full 3D diffraction geometry

- Importance: `High`
- Difficulty: `Very Large`
- Why it matters:
  - The current scene pipeline filters for near-vertical edges and therefore behaves like a 2.5D solver in many cases.
  - This is a major completeness limitation in true 3D scenes.
- Deliverable:
  - Optional generic-edge mode
  - Correct 3D edge selection and angle handling
- Current implementation note:
  - `edge_selection_mode="all_edges"` now enables generic non-vertical diffraction candidates.
  - The solver evaluates source/receiver geometry in the plane perpendicular to the edge direction and anchors each selected edge at the calculation-height clamp/intersection point.
  - This completes the generic-edge enablement milestone for the current scalar UTD solver; later fidelity extensions can still refine the polarized diffraction model or endpoint physics beyond the current scope.

#### [x] P07. Assess finite edge-length and endpoint effects

- Importance: `Medium`
- Difficulty: `Large`
- Why it matters:
  - Current UTD treatment behaves like an infinite straight wedge-edge approximation.
  - Finite edges and endpoints can matter in realistic meshes.
- Deliverable:
  - A documented decision:
    - either explicitly declare endpoint diffraction out of scope,
    - or implement/approximate it in a controlled way
- Current implementation note:
  - The solver now exposes `finite_edge_policy="infinite_wedge" | "exclude_clamped_anchors"`.
  - The default remains the locally infinite straight-edge UTD approximation.
  - In generic-edge mode, `exclude_clamped_anchors` removes anchors that land on geometric endpoints, making the current endpoint-treatment decision explicit and metadata-visible without introducing a heuristic endpoint diffraction model.

### E. Geometry and Convention Robustness

#### [x] G01. Validate wedge ordering, `phi`, `phi_prime`, reciprocity, and path orientation conventions

- Importance: `Critical`
- Difficulty: `Medium`
- Why it matters:
  - A large fraction of diffraction bugs come from face ordering and angle convention mismatches.
  - This must be locked down before closed-form comparison is meaningful.
- Deliverable:
  - Canonical convention tests and documentation

#### [x] G02. Clarify policy for boundary/open edges

- Importance: `Medium`
- Difficulty: `Medium`
- Why it matters:
  - Boundary edges currently exist in geometry preprocessing, but their actual diffraction participation is not a fully explicit policy.
  - This can create silent mismatches between geometry expectation and solver behavior.
- Deliverable:
  - A documented and tested policy for open edges and single-face edges

#### [x] G03. Make vertical-edge filtering an explicit solver mode, not a hidden assumption

- Importance: `High`
- Difficulty: `Small`
- Why it matters:
  - Right now the filtering behavior is architectural, not merely a tuning option.
  - Users should know whether they are running a vertical-edge-only approximation.
- Deliverable:
  - Clear configuration surface and documentation

### F. Validation and Ground Truth

#### [x] V01. Add a closed-form double-diffraction reference evaluator

- Importance: `Critical`
- Difficulty: `Very Large`
- Why it matters:
  - Without a closed-form reference, second-order diffraction accuracy remains largely unproven.
- Deliverable:
  - Complex-field reference for canonical double-wedge scenes
- Dependency:
  - `G01`

#### [x] V02. Add a closed-form triple-diffraction reference evaluator

- Importance: `High`
- Difficulty: `Very Large`
- Why it matters:
  - Third-order support is currently only solver-side.
  - This is needed to establish credibility beyond second order.
- Deliverable:
  - Complex-field reference for canonical triple-wedge scenes
- Dependency:
  - `V01`

#### [x] V03. Build mixed-path regression scenes with complex-field expectations

- Importance: `Critical`
- Difficulty: `Large`
- Why it matters:
  - The current suite checks mainly for non-zero or structurally valid output.
  - Mixed-path families need explicit scenes where the expected path family is known.
- Deliverable:
  - Scene set for `R->D`, `D->R`, `R->D->R`, `D->R->D`
  - Complex-field or phase-consistency assertions

#### [x] V04. Compare first-order overlap cases against local Sionna RT

- Importance: `High`
- Difficulty: `Medium`
- Why it matters:
  - Sionna RT is still useful as a first-order external baseline for shared path families.
  - This helps validate conventions before higher-order comparisons.
- Deliverable:
  - Automated comparison harness for overlap cases only

#### [x] V05. Repair and strengthen stale multi-order tests

- Importance: `Critical`
- Difficulty: `Small`
- Why it matters:
  - The current failing slope test means part of the higher-order behavior is no longer guarded.
  - This should be fixed before larger solver surgery.
- Deliverable:
  - Green regression suite
  - Additional assertions that detect phase-loss and path-history regressions

### G. Engineering and Solver Modes

#### [x] E01. Separate `accuracy` mode from `fast approximate` mode

- Importance: `High`
- Difficulty: `Medium`
- Why it matters:
  - Some current approximations are useful for speed.
  - Future exact mixed-path work should not silently replace all fast paths.
- Deliverable:
  - Explicit solver modes with documented guarantees

#### [x] E02. Profile and control state explosion after mixed-path generalization

- Importance: `High`
- Difficulty: `Large`
- Why it matters:
  - Exact mixed-path support can quickly dominate memory and runtime.
- Deliverable:
  - Profiling report
  - Memory/runtime budgets
  - Guardrails around `max_reflections`, `max_diffractions`, and mixed depth

## Recommended Priority Order

### Priority P0

These items should be done first because they either fix immediate physical credibility issues or unblock the roadmap:

- `C01` Formalize path taxonomy
- `C03` Extend state audit
- `R01` Path-faithful reflected sources
- `R02` Exact cumulative image sources for reflection prefixes
- `R03` Preserve complex amplitude/phase in reflection detail
- `P01` Fresnel reflection model
- `V01` Closed-form double-diffraction reference
- `V05` Repair stale tests

### Priority P1

These items are next because they close major completeness gaps:

- `R04` Exact prefix validity backtracking
- `M01` Add `D -> R -> D`
- `M03` Duplicate-count control
- `M04` Path-budget strategy
- `P02` Material-aware wedge diffraction
- `G01` Convention validation
- `V03` Mixed-path regression scenes
- `V04` Sionna overlap comparison

### Priority P2

These items are important but should follow the core solver corrections:

- `C02` Exact vs approximate labeling
- `[x]` `P05` Better derivative treatment
- `[x]` `P06` Full 3D generic-edge support
- `G02` Boundary-edge policy
- `G03` Vertical-edge mode exposure
- `E01` Accuracy vs fast mode
- `E02` Performance guardrails

### Priority P3

These items are valuable late-stage extensions:

- `M02` Full arbitrary mixed-chain generalization
- `[x]` `P03` Polarization/Jones transport
- `[x]` `P04` Exact shadow-boundary treatment
- `[x]` `P07` Finite-edge endpoint effects
- `[x]` `V02` Closed-form triple-diffraction reference

## Phased Task Plan

### Phase 0. Baseline Repair and Audit Foundation [Completed]

Goal:

- Make the current solver observable and testable before changing physics.

Included items:

- `C01`
- `C02`
- `C03`
- `V05`
- `G03`

Why this phase is first:

- It is the lowest-risk phase.
- It clarifies exactly what the solver currently claims to compute.
- It prevents hidden regressions during later solver refactors.

Exit criteria:

- Path-family matrix is written down
- State audit exposes enough metadata to classify every current path
- Current stale tests are repaired and green

### Phase 1. Highest-Value Prefix Accuracy Fixes [Completed]

Goal:

- Make `R^n -> D` physically credible before adding more missing path families.

Included items:

- `R01`
- `R02`
- `R03`
- `R04`
- `P01`
- `P02`
- `G01`

Why this phase is high value:

- Current mixed-path accuracy is dominated by reflection-prefix approximation.
- Fixing this improves both existing `R->D` and future mixed-path extensions.

Exit criteria:

- Reflection-prefix diffraction is driven by path-faithful or chain-faithful reflected sources
- Complex phase is preserved end-to-end in prefix state
- First-order and prefix conventions are validated

### Phase 2. Fill the First Missing Mixed-Path Gap [Completed]

Goal:

- Add the first truly missing mixed family in a controlled way.

Current state:

- `M01` is now implemented approximately through sampled inserted-reflection diffraction states.
- `M03` is now implemented through explicit field-component ownership rules and audit-visible ownership labels.
- `M04` is now implemented through explicit deterministic state-budget caps with metadata-visible pruning reports.
- `V03` is now implemented through dedicated mixed-path regression scenes with family-level complex-field reconstruction checks.

Included items:

- `[x]` `M01`
- `[x]` `M03`
- `[x]` `M04`
- `[x]` `V03`

Why this phase comes before full generalization:

- `D -> R -> D` is the most meaningful missing family that can still be implemented without a full path-grammar rewrite.
- It gives a concrete test bed for mixed-path ownership and pruning.

Exit criteria:

- `D -> R -> D` exists
- Double counting is controlled
- At least one dedicated mixed-path regression scene validates it

### Phase 3. Validation Against Theory and External Baselines [Completed]

Goal:

- Move from solver self-consistency to actual physical validation.

Included items:

- `[x]` `V01`
- `[x]` `V04`
- `[x]` part of `V03`

Why this phase is essential:

- Without it, the solver may remain internally consistent but physically wrong.

Exit criteria:

- Closed-form double-diffraction comparison is implemented
- First-order overlap cases are compared against Sionna RT
- Error is evaluated at complex-field level, not only power

### Phase 4. Architectural Upgrade for General Mixed Chains [Completed]

Goal:

- Move from ad hoc prefix/suffix composition to a unified mixed-path solver.

Included items:

- `[x]` `M02`
- `[x]` `E01`
- `[x]` `E02`

Why this phase is late:

- It is expensive and risky.
- It should only be started after prefix accuracy and validation infrastructure are in place.

Exit criteria:

- A unified interaction-path engine exists
- Mixed-depth control is explicit
- Exact vs approximate solver modes are clearly separated

### Phase 5. Extended Physics and 3D Generalization [Completed]

Goal:

- Push the solver beyond scalar 2.5D-like behavior into more complete electromagnetic modeling.

Included items:

- `[x]` `P03`
- `[x]` `P04`
- `[x]` `P05`
- `[x]` `P06`
- `[x]` `P07`
- `[x]` `V02`
- `[x]` `G02`

Why this phase is last:

- These tasks are high cost and have strong dependencies on the earlier phases.
- They make sense only after the mixed-path baseline is physically trustworthy.

Exit criteria:

- Full 3D generic-edge mode exists or is explicitly ruled out
- Boundary treatment remains exact, without smoothing shortcuts
- Higher-order reference validation goes beyond second order

## Suggested Extension Sequence Beyond This Plan

All checklist items in this plan are now completed. If the solver is extended beyond the current plan, the most reasonable next steps are:

1. Replace the current scalar-lifted diffraction vector transport with a more fully polarized diffraction formulation if a stronger Jones-level physical model is required
2. Add per-face wedge material assignment so `R0` and `Rn` are no longer forced to share one material model
3. Expand theory-backed validation for more general higher-order wedge configurations and finite-edge cases

## Summary Judgment

The scoped multi-order diffraction plan tracked in this document is now complete.

The current solver is well beyond a simple first-order diffraction implementation: it now covers audited mixed-path families, explicit ownership and pruning rules, Jones/vector transport, exact geometric shadow-boundary classification, generic non-vertical edges, finite-edge policy selection, and theory/external-baseline validation for the cases included in this plan.

The main remaining work, if pursued beyond this checklist, is no longer about missing plan items. It is about raising physical fidelity and validation depth beyond the present scope.

The most expensive long-term work is:

- replacing the current scalar-lifted diffraction vector transport with a stronger polarized diffraction model
- extending finite-edge handling from an explicit policy to a richer endpoint diffraction model if required
- broadening higher-order reference validation beyond the canonical benchmark scenes in this repository
- exact boundary treatment without smoothing
