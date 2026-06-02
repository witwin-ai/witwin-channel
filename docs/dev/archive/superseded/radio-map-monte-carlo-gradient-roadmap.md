Status: Draft
Category: Plan
Last reviewed: 2026-04-10

# Radio-Map Monte Carlo Gradient Roadmap

## Purpose

This document records the current differentiable Monte Carlo radio-map gradient design, the remaining local finite-difference patches in the runtime path, and the recommended migration path toward a stricter and more general estimator.

## Current Runtime Design

The current Monte Carlo radio-map AD path is a mixed estimator:

- Continuous pathwise terms are replayed from recorded tapes and differentiated through Dr.Jit/native sparse-coefficient JVP/VJP extraction.
- TX-to-measurement accumulation transport is not fully reparameterized. It is currently patched with fixed-tape replay plus central differences.
- The active metadata contract reports this explicitly as `fixed_tape_full_replay_central_difference` with tape layout `single_solver_native_sparse_coeff_tape_v3`.

Relevant implementation entrypoints:

- `witwin/channel/monitors/radio_map/monte_carlo/trace.py`
- `witwin/channel/monitors/radio_map/monte_carlo/custom_op.py`
- `witwin/channel/monitors/radio_map/monte_carlo/ad_support.py`

## Current Local Finite-Difference Patches

The current runtime-local FD patches are limited to the TX accumulation transport correction in Monte Carlo radio-map AD mode.

### 1. LOS TX Accumulation Transport Basis Maps

- Function: `los_tx_transport_basis_maps()`
- File: `witwin/channel/monitors/radio_map/monte_carlo/ad_support.py`
- Mechanism:
  - Replays the recorded LOS transport tape with the blocker primitive identity fixed.
  - Shifts TX by `(+/- step, 0, 0)`, `(0, +/- step, 0)`, and `(0, 0, +/- step)`.
  - Re-splats shifted hits into the grid.
  - Estimates the transport Jacobian by central difference.
- Step size: `_MC_TX_TRANSPORT_FD_STEP = 1.0e-3`

This patch exists because the radio-map cell accumulation depends on the hit location on the measurement plane, which is still treated as a discrete accumulation target in the primal solver path.

### 2. Reflection TX Accumulation Transport Basis Maps

- Function: `reflection_tx_transport_basis_maps()`
- File: `witwin/channel/monitors/radio_map/monte_carlo/ad_support.py`
- Mechanism:
  - Replays the recorded reflection transport tape with fixed reflection primitive history and fixed blocker primitive identity.
  - Applies the same `x/y/z` central-difference TX shifts.
  - Re-splats shifted reflected hits into the grid.
  - Uses the resulting maps as the reflection transport correction.
- Step size: `_MC_TX_TRANSPORT_FD_STEP = 1.0e-3`

### 3. Runtime Metadata Contract

The runtime metadata reports the presence of these FD patches through:

- `tx_accumulation_transport_mode = "fixed_tape_full_replay_central_difference"`
- `tx_accumulation_transport_step = 1.0e-3`

This is currently emitted only when Monte Carlo radio-map AD mode is enabled.

## Current Heuristic / Approximation Inventory

Not every approximation in the Monte Carlo implementation is the same kind of risk. The items below are grouped by their role in the estimator.

### 1. Estimator-Side Approximations

These directly affect what gradient is being estimated.

- Fixed blocker identity in LOS TX transport replay
  - Files:
    - `witwin/channel/monitors/radio_map/monte_carlo/ad_support.py`
  - Relevant implementation:
    - `los_tx_transport_basis_maps()`
    - `blocker_prim_idx = tape.transport_blocker_prim_idx`
  - Effect:
    - The TX-shifted transport replay keeps the blocker primitive identity fixed from the primal tape.
    - This does not model blocker identity switches as an explicit boundary term.

- Fixed reflection primitive history in reflection TX transport replay
  - Files:
    - `witwin/channel/monitors/radio_map/monte_carlo/ad_support.py`
  - Relevant implementation:
    - `reflection_tx_transport_basis_maps()`
    - `prim_idx = tape.transport_prim_index_by_bounce[bounce_slot]`
    - `prim_idx=tape.transport_blocker_prim_idx`
  - Effect:
    - Shifted reflection replays keep both the reflection primitive sequence and the blocker primitive identity fixed from the primal tape.
    - This excludes path topology changes from the transport estimator.

- Tent-splat transport deposition
  - Files:
    - `witwin/channel/monitors/radio_map/monte_carlo/ad_support.py`
  - Relevant implementation:
    - `_tent_splat_to_grid()`
  - Effect:
    - Transport maps are deposited with a tent basis over neighboring cells rather than the primal hard-cell deposition rule.
    - This is smoother than hard binning and helps transport differentiation, but it is still an engineering replacement for a formal continuous measurement operator.

- Frozen diffraction support and branch integers during replay
  - Files:
    - `witwin/channel/monitors/radio_map/monte_carlo/ad_support.py`
    - `witwin/channel/monitors/radio_map/monte_carlo/field.py`
  - Relevant implementation:
    - `support_override={...}`
    - `_override_mask(...)`
    - `_override_value(...)`
  - Effect:
    - Diffraction replay reuses primal-tape values for `field_valid`, `pole_safe`, `dif_n_p`, `dif_n_m`, `sum_n_p`, and `sum_n_m`.
    - This stabilizes replay across local perturbations, but it freezes discrete support and branch decisions instead of treating them as separate event terms.

- Hard cell deposition in the primal monitor path
  - Files:
    - `witwin/channel/monitors/radio_map/monte_carlo/common.py`
  - Relevant implementation:
    - `_axis_aligned_cell_index()`
    - `_scatter_component()`
  - Effect:
    - Primal accumulation still uses clipped axis-aligned hard cell indexing.
    - This keeps the measurement operator piecewise constant at cell boundaries, which is one reason the transport correction currently exists.

### 2. Sampling and Discovery Heuristics

These are not local FD patches, but they do define which Monte Carlo states are sampled or discovered.

- Length-proportional diffraction state sampling
  - Files:
    - `witwin/channel/monitors/radio_map/monte_carlo/diffraction.py`
  - Relevant implementation:
    - `LengthProportionalStateSampler`
    - `LengthProportionalStateSampler.from_line_length(...)`
  - Effect:
    - Diffraction states are sampled proportional to discovered edge-line length.
    - This is an importance-sampling choice, not a full enumeration of all available states.

- Best-edge wedge discovery from local hit data
  - Files:
    - `witwin/channel/monitors/radio_map/monte_carlo/diffraction.py`
  - Relevant implementation:
    - `DirectTxWedgeDiscovery.best_edge_indices_from_hit_data(...)`
  - Effect:
    - Candidate edges are scanned and the closest silhouette/exterior-compatible edge is selected.
    - This is a local discovery heuristic, not a general event-space construction.

- Vertical-edge-scoped diffraction state storage
  - Files:
    - `witwin/channel/monitors/radio_map/monte_carlo/diffraction.py`
  - Relevant implementation:
    - `capacity = max(0, len(scene.vertical_edges))`
  - Effect:
    - Current state storage is sized from `scene.vertical_edges`.
    - This reflects a deliberately narrow implementation contract rather than a general diffraction representation.

### 3. Numerical Robustness Heuristics

These are engineering offsets intended to avoid self-intersection or degenerate configurations.

- Detached ray-origin offset helper
  - Files:
    - `witwin/channel/monitors/radio_map/monte_carlo/common.py`
  - Relevant implementation:
    - `_spawn_offset_ray_origin()`
    - `offset_scale = 1.0e-5 * (1 + max(abs(point_pos)))`
    - `signed_offset = dr.detach(...)`
  - Effect:
    - Ray origins are nudged away from surfaces with a detached scale factor.
    - This is a numerical stabilization rule, not a physically meaningful estimator term.

- Reflection self-hit bias
  - Files:
    - `witwin/channel/monitors/radio_map/monte_carlo/reflection.py`
  - Relevant implementation:
    - `si.p + reflected_dir * RAY_ORIGIN_BIAS`
  - Effect:
    - Reflected rays are offset after a bounce to avoid immediate self-intersection.

- Diffraction launch-point offset
  - Files:
    - `witwin/channel/monitors/radio_map/monte_carlo/common.py`
    - `witwin/channel/monitors/radio_map/monte_carlo/diffraction.py`
  - Relevant implementation:
    - `_MC_DIFFRACTION_OFFSET = 5.0e-2`
    - `diff_point_offset = diff_point + _MC_DIFFRACTION_OFFSET * offset_normal`
  - Effect:
    - Diffraction launches are shifted away from the wedge support geometry.
    - This improves robustness but is still a hand-tuned geometric offset.

### 4. Work-Control and Performance Heuristics

These do not redefine the gradient mathematically, but they do change runtime work allocation and pruning.

- Russian roulette continuation control
  - Files:
    - `witwin/channel/monitors/radio_map/monte_carlo/reflection.py`
    - `witwin/channel/monitors/radio_map/monte_carlo/common.py`
  - Relevant implementation:
    - `continue_prob = min(gain_no_spread, rr_prob)`
    - lower-bounded by `_MC_MIN_RR_PROBABILITY = 1.0e-8`
  - Effect:
    - This is a standard Monte Carlo path-pruning mechanism, but it is still a runtime control heuristic.

- Stop-threshold pruning
  - Files:
    - `witwin/channel/monitors/radio_map/monte_carlo/reflection.py`
    - `witwin/channel/monitors/radio_map/monte_carlo/common.py`
  - Relevant implementation:
    - `_stop_threshold_linear(...)`
    - `active = active & (gain_no_spread * fspl * fspl > stop_threshold_linear)`
  - Effect:
    - Low-contribution paths can be pruned by a user-configured threshold.
    - This is a practical runtime filter, not part of a strict unbiased estimator contract.

- Batch-size memory planning
  - Files:
    - `witwin/channel/monitors/radio_map/monte_carlo/common.py`
  - Relevant implementation:
    - `_MC_RAY_BATCH_BUDGET_RATIO = 0.5`
    - `_MC_DIFFRACTION_BATCH_BUDGET_RATIO = 0.5`
    - `_MC_ESTIMATED_RAY_WORKING_SET_BYTES = 768`
    - `_MC_ESTIMATED_DIFFRACTION_SAMPLE_BYTES = 3072`
  - Effect:
    - Runtime batch sizes are derived from fixed estimated working-set sizes and memory budget ratios.
    - These are engineering heuristics for GPU execution planning.

## What Is Not Using Local FD

The following parts of the current runtime are not using local finite differences:

- LOS sparse coefficient extraction for continuous pathwise terms
- Reflection sparse coefficient extraction for continuous pathwise terms
- Diffraction sparse coefficient extraction for continuous pathwise terms
- Vertex and material gradients collected through the replayed sparse-coefficient pipeline
- Native sparse-coefficient JVP/VJP aggregation inside the Monte Carlo custom op

Those paths are still approximate in the broader estimator sense because topology and visibility events are not yet handled by a separate boundary estimator, but they are not implemented through local FD stencils.

## Current Limitations

The current mixed design is useful and numerically stable on validated scenes, but it is not the final estimator architecture.

Main limitations:

- The transport term is a local FD patch, not a full sample-space reparameterization.
- Visibility and path birth/death events are not modeled as explicit boundary terms.
- Measurement accumulation is still fundamentally tied to discrete cell deposition instead of a cleaner continuous measurement operator.
- The implementation contract is intentionally narrow: axis-aligned planar radio maps, incoherent combine mode, matched-isotropic receiver model, and current direct first-order diffraction support.

## Recommended Migration Plan

### Phase A. Write the Estimator Contract Down

- Formalize the radio-map gradient as the sum of:
  - continuous pathwise terms
  - visibility-boundary terms
  - measurement-accumulation transport terms
- For each term, declare:
  - estimator definition
  - approximation assumptions
  - expected failure modes
  - validation method

### Phase B. Replace TX Transport FD With Sample-Space Replay

- Promote ray sampling, Russian roulette, edge-position sampling, and Keller-cone sampling to explicit replay variables.
- Differentiate the replayed measurement hit location with respect to scene and TX parameters directly in sample space.
- Remove the fixed-tape central-difference basis-map path once the replayed transport Jacobian reaches parity.

### Phase C. Separate Boundary Estimation From Pathwise Replay

- Add explicit estimators for:
  - blocker identity changes
  - path birth/death events
  - measurement-cell boundary crossings
- Keep these terms separate from the continuous sparse-coefficient path so correctness arguments remain local and auditable.

### Phase D. Replace Hard Cell Deposition With a Better Measurement Operator

- Introduce a continuous measurement-space representation, such as a piecewise-linear basis over the measurement surface.
- Treat box-cell outputs as a linear projection of that representation instead of the primary differentiable object.
- Use this as the long-term replacement for ad hoc transport fixes around discrete binning.

### Phase E. Unify Reflection and Diffraction AD Interfaces

- Give reflection and diffraction the same internal interface shape:
  - sample generation
  - primal replay
  - pathwise derivative extraction
  - boundary derivative extraction
  - measurement pushforward
- This should be done before extending to higher-order diffraction or more general measurement surfaces.

### Phase F. Extend the Contract Scope

- Extend from axis-aligned planar radio maps to:
  - general planar surfaces
  - mesh measurement surfaces
- Only expand path families after the estimator decomposition is clean.

### Phase G. Move Performance Work After Estimator Cleanup

- Keep Python/Dr.Jit reference implementations until the estimator contract is stable.
- Then fuse stable pieces into native kernels.
- Avoid adding new local FD paths or backend-specific patches before the estimator architecture converges.

## Validation Gates

Each migration phase should preserve or improve the following checks:

- JVP vs VJP consistency
- AD vs finite-difference agreement on stable scenes
- convergence with increasing sample count
- robustness near:
  - cell crossings
  - occluder identity switches
  - grazing incidence
  - path appearance and disappearance events

## Immediate Engineering Rule

Do not add new runtime-local finite-difference patches to the Monte Carlo radio-map gradient path unless they are explicitly documented in this file with:

- scope
- rationale
- exact stencil
- step size
- replacement plan
