# Current Diffraction Path Taxonomy

Status: Active
Category: Standard
Last reviewed: 2026-05-20

This document captures the current solver-side interaction taxonomy that is now
surfaced through `result["solver_metadata"]` and diffraction `state_audit`
metadata.

It complements `docs/dev/archive/completed/multiorder-diffraction-completeness-checklist.md` by
describing the path families that are implemented today, how they are labeled,
and whether the current implementation is exact or approximate.

## Solver Mode

- Diffraction edge selection mode: `vertical_only` or `all_edges`
- Selection control lives on solver config through `Config(edge_policy=EdgePolicy(edge_selection_mode="vertical_only" | "all_edges", vertical_ratio=0.7))`, with `all_edges` as the default.
- Standalone edge-diffraction control lives on solver config through `Config(edge_policy=EdgePolicy(edge_diffraction=True | False))`, defaulting to `True`
- Boundary/open-edge policy lives on solver config through `Config(edge_policy=EdgePolicy(boundary_edge_policy="exclude" | "half_plane" | None))`
- Finite-edge model: fixed `finite_wedge`
- Default boundary policy: `half_plane`, via `edge_diffraction=True`
- Sionna/Mitsuba XML import stores scene geometry only; callers select import edge policy through the solver config used for tracing.
- Closed-wedge-only policy: `Config(edge_policy=EdgePolicy(edge_diffraction=False))` or `Config(edge_policy=EdgePolicy(boundary_edge_policy="exclude"))`
- Generic-edge mode status: implemented approximately in `all_edges`
- Current limitation: the diffraction coefficient still uses the current scalar UTD formulation, while Jones/vector transport and finite-edge handling are layered on top as the scoped physical-fidelity upgrade for this plan

The current solver therefore now supports both the explicit legacy 2.5D-style
`vertical_only` mode and the default approximate generic-edge mode, with explicit
polarization transport and a fixed finite-wedge treatment now exposed in solver
results.

## Convention Checks

The current regression suite explicitly checks the following diffraction
conventions:

- wedge face ordering uses the smaller positive rotation from `cross(n0, e_hat)`
  to `cross(nn, e_hat)`, equal to `(wedge_n - 1) * pi`
- the incident-angle helper used by `state_audit` matches the general
  propagation-angle helper for `phi_prime`
- the scalar UTD coefficient remains reciprocal under source/receiver swap when
  the same wedge and face responses are used

The current validation helpers also expose:

- an explicit order-2 double-diffraction pair-expansion reference on the
  canonical double-wedge case
- an explicit order-3 triple-diffraction triplet-expansion reference on the
  canonical triple-wedge case
- a first-order overlap comparison against the local Sionna RT utility
  implementation on the canonical single-wedge case

Derivative treatment:

- diffraction-angle derivatives in the production solver now use analytic
  chain-rule expressions for the current scalar UTD approximation
- the production path no longer uses finite-difference angle steps from
  `diffraction/utd.py` for slope-channel propagation

Shadow-boundary treatment:

- diffraction validity is classified with exact wedge-face half-space tests in
  the plane perpendicular to the edge
- the production path no longer uses hard `phi/phi_prime` range clipping to
  accept or reject diffracted contributions
- wedge-boundary receivers remain valid without any soft or heuristic
  smoothing, while solid-interior receivers are rejected geometrically
- coherent matched-ISB radio-map completion keeps true visibility and
  solid-interior rejection as hard geometry, but its scene-edge incident
  statistics now apply the deterministic angular shadow-completion decay curve
  as a receiver-side narrow-band continuous target-support weight instead of a
  hard wedge-exterior cutoff
- reflection secondary-visibility F-weighting is an in-line reflection-chain
  segment attenuation, not diffraction post-processing; it must not be enabled
  together with matched-ISB shadow-boundary correction on the same reflection
  pass

Generic-edge treatment:

- `vertical_only`
  - keeps the legacy near-vertical diffraction-edge subset for 2.5D-like runs
- `all_edges`
  - allows non-vertical interior edges to participate in diffraction
  - builds one diffraction state anchor per selected edge at the calculation
    height clamp/intersection point
  - evaluates source/receiver distances in the plane perpendicular to the edge
    direction so non-vertical edges are no longer silently excluded
  - still uses the current scalar wedge model, so this is generic-edge support
    within the existing scalar UTD formulation rather than a full polarized
    diffraction-matrix model

Finite-edge treatment:

- `finite_wedge`
  - carries each selected diffraction edge's finite axial segment bounds into the
    Dr.Jit diffraction evaluator
  - replaces the old infinite-line continuation with a smooth finite-segment
    truncation factor derived from the stationary-phase point and Fresnel
    endpoint integrals
  - requires explicit `edge_line_min` / `edge_line_max` bounds during scene
    compilation, state construction, field evaluation, and radio-map replay
  - keeps public AD-capable native UTD paths on the validated Dr.Jit finite-wedge
    replay until a native finite-wedge backward path is implemented

Polarization transport:

- LoS
  - projects the transmit polarization onto the plane transverse to each ray
- Reflection
  - uses TE/TM Fresnel response with basis rotation in the reflection plane
- Diffraction
  - transports a Jones/vector field by lifting the current scalar UTD
    coefficient into an edge-local vector basis, including mixed suffix
    reflection after diffraction

## Path-Family Matrix

| Path family | Current status | Coverage label in metadata | Notes |
|---|---|---|---|
| `S -> D` | Implemented exactly in the current solver mode | `exact` | Direct transmitter to diffraction edge |
| `R^n -> D` | Implemented approximately | `approximate` | Reflection prefixes come from sampled path-faithful reflected image-source chains with Fresnel-weighted complex reflection factors |
| `S -> D -> ... -> D` | Implemented approximately | `approximate` | Higher-order diffraction uses recursive edge-state propagation |
| `R^n -> D -> ... -> D` | Implemented approximately | `approximate` | Inherits sampled reflection-prefix coverage and recursive diffraction approximations |
| `... -> D -> R^n` | Implemented approximately | `approximate` | Reflection suffix uses Monte Carlo reflection tracing from the last diffraction state |
| `D -> R -> D` | Implemented approximately | `approximate` | One inserted reflection is sampled from the current diffraction state and re-injected as the source for the next diffraction state |
| Arbitrary alternating mixed chains | Implemented approximately | `approximate` | Alternating chains are expanded recursively with one sampled reflection allowed between consecutive diffraction events, up to the resolved `max_diffraction_order` and `Tuning.max_inserted_reflections_per_path` |

## State Audit Fields

The diffraction `state_audit` now records the following path metadata per state:

- `source_type_code`
- `source_type`
- `ownership_code`
- `ownership`
- `prefix_reflection_depth`
- `intermediate_reflection_depth`
- `suffix_reflection_depth`
- `path_reflection_depth_0`, `path_reflection_depth_1`, ...
- `approximation_mode_code`
- `approximation_mode`
- `path_sequence`

The human-readable fields use the following labels:

- `source_type`
  - `direct_tx`
  - `reflection_prefix`
- `ownership`
  - `direct_diffraction`
  - `mixed_diffraction`
- `approximation_mode`
  - `exact_direct_first_order`
  - `approx_recursive_diffraction`
  - `approx_sampled_reflection_prefix`
  - `approx_sampled_reflection_prefix_chain`
  - `approx_sampled_inserted_reflection`
  - `approx_sampled_inserted_reflection_chain`

Examples:

- `S -> D`
  - `source_type = direct_tx`
  - `prefix_reflection_depth = 0`
  - `intermediate_reflection_depth = 0`
  - `suffix_reflection_depth = 0`
  - `approximation_mode = exact_direct_first_order`
- `S -> D -> D`
  - `source_type = direct_tx`
  - `prefix_reflection_depth = 0`
  - `intermediate_reflection_depth = 0`
  - `suffix_reflection_depth = 0`
  - `approximation_mode = approx_recursive_diffraction`
- `S -> R -> D`
  - `source_type = reflection_prefix`
  - `prefix_reflection_depth = 1`
  - `intermediate_reflection_depth = 0`
  - `suffix_reflection_depth = 0`
  - `approximation_mode = approx_sampled_reflection_prefix`
- `S -> D -> R -> D`
  - `source_type = direct_tx`
  - `prefix_reflection_depth = 0`
  - `intermediate_reflection_depth = 1`
  - `suffix_reflection_depth = 0`
  - `approximation_mode = approx_sampled_inserted_reflection`
- `S -> D -> R -> D -> R -> D`
  - `source_type = direct_tx`
  - `prefix_reflection_depth = 0`
  - `intermediate_reflection_depth = 2`
  - `path_reflection_depth_0 = 0`
  - `path_reflection_depth_1 = 1`
  - `path_reflection_depth_2 = 1`
  - `suffix_reflection_depth = 0`
  - `approximation_mode = approx_sampled_inserted_reflection`

## Result Metadata Surface

`Tracer.trace()` now returns `solver_metadata` with:

- `solver_mode`
- `performance_guardrails`
- `edge_selection_mode`
- `boundary_edge_policy`
- `shadow_boundary_treatment`
- `edge_selection_summary`
- `vertical_ratio`
- `max_diffraction_order`
- `reflection_prefix_path_count`
- `reflection_suffix_enabled`
- `reflection_suffix_budget`
- `mixed_chain_budget`
- `path_budget_policy`
- `field_component_ownership`
- `polarization_transport`
- `finite_edge_treatment`
- `audit_value_meanings`
- `path_families`

Solver mode conventions:

- `accuracy`
  - preserves the requested mixed-path depth and does not apply additional
    automatic pruning beyond explicit user budgets
- `fast_approximate`
  - caps sampled reflection rays, reflection bounces, mixed depth, and
    per-order state budgets to keep runtime bounded

Performance guardrail metadata now includes:

- requested vs effective solver controls
- guardrail changes applied by fast mode
- per-order state-count profiling
- estimated peak state memory before pruning
- a coarse risk label for state explosion

Boundary-edge policy conventions:

- `exclude`
  - single-face/open diffraction edges do not participate in diffraction and are
    absent from edge mapping, projected diffraction points, and solver audits
- `half_plane`
  - single-face/open diffraction edges are kept as an approximate half-plane wedge
    with `wedge_n = 2`, and this approximation is surfaced in metadata

`Tracer.trace()` now also exposes:

- `a_dif_direct`
- `a_dif_mixed`
- `jones_los`
- `jones_ref`
- `jones_dif_direct`
- `jones_dif_mixed`
- `jones_dif`
- `jones_tot`

Ownership convention:

- `a_ref` owns pure reflection-only paths
- `a_dif_direct` owns diffraction states with zero reflection depth
- `a_dif_mixed` owns diffraction states with non-zero prefix, inserted, or suffix reflection depth
- `a_dif = a_dif_direct + a_dif_mixed`
- `a_tot = a_los + a_ref + a_dif`

This metadata is intended to make current solver claims explicit before later
physics and mixed-path upgrades.

## Reflection Material Parameters

The current tracer exposes:

- `reflection_coef`
- `reflection_relative_permittivity`
- `reflection_conductivity`

These parameters now affect:

- direct and higher-order diffraction through scalar face-response terms `R0`
  and `Rn`
- main reflection field `a_ref`
- reflection-prefix diffraction states
- reflection suffix after diffraction (`... -> D -> R^n`)

The current solver now transports Jones/vector fields across LoS, reflection,
and diffraction. Reflection uses TE/TM Fresnel transport, while diffraction
uses the current scalar UTD coefficient lifted into an edge-local vector
transport.

Current material limitation:

- diffraction still uses one shared material model for both wedge faces
- per-face material assignment is not implemented yet
