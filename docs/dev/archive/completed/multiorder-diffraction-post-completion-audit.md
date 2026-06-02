# Post-Completion Audit of the Multi-Order Diffraction Implementation

Date: 2026-03-16

## Purpose

This document records a post-completion engineering audit of the current
multi-order diffraction implementation after all items in
`docs/dev/archive/completed/multiorder-diffraction-completeness-checklist.md` were marked complete.

The goal of this audit is not to redefine the checklist. It is to answer four
practical questions:

1. Is the checklist actually completed in code?
2. Is the current solver also the final physically complete solver?
3. Are there still compatibility shims, silent fallbacks, or defensive
   behaviors that should be removed?
4. Is there duplicated or effectively dead code that should be cleaned up?

## Cleanup Status

The findings below were audited before the immediate cleanup pass that removed:

- the silent Jones/vector fallback in diffraction state evaluation
- the `a_rd` compatibility alias from `Tracer.trace()`
- the reflection-solver compatibility return placeholder and reserved
  compatibility flag
- the unused scene compatibility wrapper/property and several unused helper
  functions

Status after that cleanup pass:

- `F1`: addressed
- `F3`: addressed
- `F4`: addressed for the items listed in this audit
- `F2`: still open

## Executive Judgment

### Checklist status

The scoped checklist is completed.

The implementation now includes:

- audited mixed-path state metadata
- reflection-prefix and mixed-path bookkeeping
- deterministic ownership and pruning policy
- closed-form double- and triple-diffraction validation harnesses
- explicit boundary/open-edge policy
- explicit finite-edge policy
- generic-edge diffraction mode
- Jones/vector transport across LoS, reflection, and diffraction
- suffix reflection propagation in both scalar and Jones/vector channels

### Physical completeness status

The current solver is not yet the final physically complete full-3D
electromagnetic solver.

This is not hidden in the codebase. The current implementation explicitly
states that:

- diffraction still uses the current scalar UTD coefficient model
- Jones/vector transport is layered on top of that scalar diffraction model
- finite-edge handling is still based on a locally infinite straight-edge UTD
  approximation plus an explicit endpoint policy

So the correct summary is:

- checklist complete: yes
- physically final model: no

## What Is Solid

The following claims are supported by the current code and regression suite:

- mixed-path families are surfaced and labeled as exact, approximate, or absent
- ownership between `a_ref`, `a_dif_direct`, `a_dif_mixed`, and `a_dif` is
  explicit
- reflection-prefix and inserted-reflection chains are auditable
- `... -> D -> R^n` suffix reflection is included in both scalar and
  Jones/vector diffraction outputs
- boundary/open-edge policy is explicit instead of implicit
- finite-edge policy is explicit instead of implicit
- no soft smoothing or heuristic smoothing is used for gradient handling
- diffraction-angle derivatives use analytic chain-rule expressions in the
  current scalar UTD model

## Findings

### F1. Silent zero-vector fallback still exists in diffraction state evaluation

Severity: High

Status: Addressed in the immediate cleanup pass after this audit.

In the modular `witwin/channel/trace/diffraction/` implementation, `_edge_state_field_to_targets()` still accepts
states that do not contain the new polarization fields. When those vector
entries are missing, it silently substitutes `vector_zero(width)` for both the
incident vector field and the incident derivative vector field.

Effect:

- scalar diffraction still evaluates
- Jones/vector diffraction silently collapses to zero for those paths
- a malformed or stale state producer can therefore pass through without a hard
  failure

Why this matters:

- this weakens the guarantee behind `P03`
- it behaves like a compatibility fallback rather than a strict post-upgrade
  contract

Recommended action:

- remove the silent fallback
- require vector fields to exist on all production diffraction states
- raise an error when the state schema is incomplete

### F2. Triangle-pair path validation still contains an explicit heuristic

Severity: High

Status: Still open.

Reflection and suffix-reflection path validation still use the assumption that
adjacent triangles belonging to the same surface may be obtained via
`prim_idx ^ 1`.

This appears in:

- `witwin/channel/trace/reflection/field.py`
- `witwin/channel/trace/diffraction/`

Effect:

- the solver is still using a mesh-layout heuristic in geometric validation
- this is not smoothing, but it is still an approximation that depends on mesh
  triangle ordering

Why this matters:

- it is weaker than a geometry-derived adjacency relation
- it can fail on arbitrary meshes whose triangle ordering does not match the
  assumed pairing rule

Recommended action:

- replace triangle-pair guessing with explicit adjacency or same-face-group
  lookup

### F3. Several compatibility shims remain in the codebase

Severity: Medium

Status: Addressed in the immediate cleanup pass after this audit.

The repository guidance says not to preserve legacy or compatibility paths
unless explicitly needed, but several such layers still remain.

Examples:

- `Tracer.trace()` still exposes `a_rd` as a legacy alias of mixed diffraction
- `Scene.n_vertical_edges` is kept as a backward-compatible property
- `filter_vertical_edges()` is kept as a backward-compatible wrapper
- `compute_reflection_field()` still accepts the reserved compatibility flag
  `enable_rd_diffraction`
- the reflection API still returns a placeholder `zero_field` for compatibility

Effect:

- the external API still carries some old naming and compatibility baggage
- new ownership and result surfaces already exist, but the old alias layer was
  not fully removed

Recommended action:

- remove compatibility aliases and wrappers that are no longer needed by the
  current repository
- keep only the explicit modern result surfaces and scene selectors

### F4. There is still duplicated or low-value helper code

Severity: Medium

Status: Addressed for the items listed in this audit.

Some code is redundant or currently unused.

Examples:

- `witwin/channel/polarization.py`
  - `vector_power()`
  - `jones_power()`
- `witwin/channel/trace/reflection/field.py`
  - `polarization_vec` is initialized twice before being overwritten with the
    final value

Effect:

- extra code paths remain to be maintained
- the implementation looks less intentional than it should

Recommended action:

- remove unused helpers
- collapse repeated initialization into the final direct construction

## Non-Findings

The following issues were specifically checked and were not found as current
problems.

### NF1. No soft smoothing or heuristic smoothing for gradients

No soft smoothing or heuristic smoothing was found in the current diffraction
gradient path.

The current implementation uses:

- exact geometric validity classification for wedge source/target regions
- analytic angle-derivative expressions

This matches the project constraint against smoothing-based gradient handling.

### NF2. Numerical guards exist, but they are not the main problem

The code does contain defensive numerical guards, including:

- `_safe_normalize()`
- `clip()` for grazing-angle stability
- `isfinite()` cleanup after Fresnel evaluation

These are numerically reasonable protections against NaNs and singular cases.
They should not be confused with compatibility fallbacks or physics-level
heuristics.

## Recommended Cleanup Order

If a cleanup pass is scheduled, the highest-value sequence is:

1. Remove the silent zero-vector fallback in diffraction state evaluation
2. Replace sibling-triangle heuristics with explicit geometric adjacency
3. Remove legacy compatibility aliases and wrappers
4. Delete unused helpers and redundant initialization

## Final Assessment

The current implementation is not a fake completion. The checklist was genuinely
implemented and the important approximations are explicitly labeled in both code
and documentation.

However, the codebase is not yet fully cleaned up after that implementation
work. The main remaining issues are:

- one silent fallback that weakens the polarization contract
- one explicit mesh-order heuristic in path validation
- several compatibility shims that should be removed
- a small amount of redundant or unused helper code

That means the correct engineering conclusion is:

- feature plan completion: achieved
- cleanup and hardening pass: still recommended
- final physically complete solver: not yet achieved
