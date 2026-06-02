# Matched ISB Completion Gradient Noise Plan

Status: Active
Category: Plan
Last reviewed: 2026-04-11

## Purpose

This document captures the current debugging conclusions for the deterministic
three-cube radiomap gradient mismatch in
`shadow_boundary_mode="matched_isb_completion"`, records the reproducible GPU
workflow, and defines the recommended remediation path.

The goal is to separate three classes of behavior that were previously mixed
together in one diffraction plot:

- true discrete-event finite-difference terms;
- raw diffraction AD omissions or topology mismatch;
- noise introduced by the matched-ISB completion surrogate itself.

## Scope

This note covers:

- `RadioMapMonitor(..., combine_mode="coherent", receiver_model="matched_isotropic")`
- deterministic three-cube radiomap diagnostics
- `tx_x` and `cube1_x` gradient comparisons
- the lower-right shadow-side region where `matched_isb_completion_only` finite
  differences appeared noise-like

This note does not propose any public API change. The stable architecture
remains `Scene + Tracer + Result`.

## Reproduced Symptom

The maintained three-cube radiomap benchmark showed that:

- the full folded diffraction gradient contained a large AD-vs-FD mismatch;
- the mismatch became visually concentrated in the shadow-side lower-right
  region when the diffraction diagnostics were split into:
  - `raw_diffraction`
  - `matched_isb_completion_only`
  - `folded_diffraction`
- after removing matched-ISB completion from the plotted diffraction field,
  the same region became much cleaner, especially for `tx_x`

The practical observation is:

- `raw_diffraction` still has some AD-vs-FD mismatch, especially for
  `cube1_x`, but it does not explain the full "sandpaper" texture;
- `matched_isb_completion_only` is the main source of the fine-grained FD
  speckle in that region.

## Verified GPU Workflow

Use the `witwin2` conda environment for every command below.

### Regression Commands

```bash
python -m pytest tests/scene/test_radio_map_monitors.py --gpu -k "three_cube_forward_baseline_matches_cell_accumulation_for_diffraction or three_cube_baseline_reports_diffraction_diagnostic_counts" -q
python -m pytest tests/main/test_radiomap_gradients_three_cubes_main.py --gpu -q
```

These checks validate:

- forward `baseline` vs `cell_accumulation` diffraction parity on the three-cube
  scene;
- availability of the new diffraction diagnostic counts and AD/FD backend
  metadata;
- end-to-end generation of the maintained three-cube gradient outputs.

### Main Diagnostic Command

```bash
python -m tests.main.plot_radiomap_gradients_three_cubes --output-prefix tests/output/radiomap_three_cubes_gradients
```

Primary outputs:

- `tests/output/radiomap_three_cubes_gradients.png`
- `tests/output/radiomap_three_cubes_gradients_components.png`
- `tests/output/radiomap_three_cubes_gradients.json`
- `tests/output/radiomap_three_cubes_gradients.npz`

The current debugging workflow also relies on additional derivative views built
from the saved arrays:

- `tests/output/radiomap_three_cubes_raw_diffraction_full.png`
- `tests/output/radiomap_three_cubes_raw_diffraction_region_zoom.png`
- `tests/output/radiomap_three_cubes_matched_isb_region_zoom.png`

## Current Findings

### 1. Forward backend mismatch is not the main cause

The benchmark now forces AD and FD gradient comparisons onto the same
`baseline` backend, while separately reporting forward
`baseline` vs `cell_accumulation` parity.

For the maintained three-cube diffraction forward case, the two forward
backends stayed effectively identical at the map level. The large gradient
delta is therefore not explained by "AD used baseline while FD used
cell_accumulation".

### 2. The lower-right completion artifact is not a broken coherent forward field

In the shadow-side region `x in [0, 10], y in [-10, -5]`, the
`matched_isb_completion_only` forward term remained smooth and stable between
`+h` and `-h`.

Observed behavior:

- region-wise `+h/-h` completion maps stayed highly correlated;
- the completion forward magnitude remained very small;
- the FD derivative, however, showed strong high-frequency sign flipping.

Conclusion:

- the primal completion field does not look like random-phase accumulation;
- the noise is introduced at the derivative level, not because the forward
  completion itself lost coherence.

### 3. Removing completion removes most of the noise texture

When the same full-scene gradient plot is rebuilt with `raw_diffraction`
instead of folded diffraction:

- `tx_x` becomes much cleaner and AD/FD correlation rises substantially;
- `cube1_x` still shows mismatch, but the result is structured and sparse
  rather than dense speckle.

Conclusion:

- the right-lower "noise-like" texture is primarily a matched-ISB completion
  issue;
- raw diffraction still contains real discrete-event mismatch, but it is not
  the dominant source of the completion-only speckle.

## Root Cause

The matched-ISB completion issue is not a native-vs-reference inconsistency.
The native CUDA path and the reference path intentionally implement the same
surrogate contract.

The root cause is structural: the current completion module mixes a continuous
average-response estimate with a discontinuous maximum-weight selector, and AD
tracks only the currently selected branch.

### A. Completion is a residual correction, not an independent field solve

The matched-ISB completion algebra is:

- `smooth_direct_mode`
- minus `hard_direct_mode`
- minus `incident_weight * direct_mode_excess`

Reference entry points:

- `witwin/channel/kernels/monitors/field/radio_map_accumulate/native_impl.py:158`
- `witwin/channel/kernels/monitors/field/radio_map_accumulate/radio_map_accumulate.cu:127`

This makes completion a small cancellation residual. Such a residual is
inherently sensitive to any discontinuity in its driving terms.

### B. The module mixes "average response" with "maximum edge weight"

Scene-wide shadow-boundary incident statistics accumulate:

- `sum_incident_weight`
- `max_incident_weight`
- weighted average incident response numerators

See:

- `witwin/channel/monitors/radio_map/deterministic/cell_accumulation.py:354`

The final matched-ISB completion then uses:

- `aggregated_incident_response = scene_incident_response`
- `aggregate_incident_weight = scene_max_incident_weight`

See:

- `witwin/channel/monitors/radio_map/deterministic/cell_accumulation.py:759`

This is the main design flaw. The completion phase term and completion weight
do not come from the same aggregate object:

- phase and complex response come from an all-edge weighted average;
- the scalar weight comes from the single strongest edge.

That mixed surrogate is not self-consistent under perturbation.

### C. AD for `max_incident_weight` is explicitly an argmax-route derivative

The CUDA incident-statistics implementation:

1. computes `max_incident_weight` with an atomic max;
2. finds `argmax_edge_idx`;
3. routes JVP/VJP for `max_incident_weight` only through that winning edge.

Key implementation points:

- `atomic_max_nonnegative(&max_incident_weight[rx_idx], primal.weight);`
  in `radio_map_accumulate.cu:1406`
- argmax selection in `radio_map_accumulate.cu:1468`
- JVP routing in `radio_map_accumulate.cu:1572`
- backward routing in `radio_map_accumulate.cu:1634`

So the current AD semantics are:

- differentiate the current winner edge only;
- ignore winner-switch events.

Finite differences do not ignore those switches. As soon as the winning edge
changes between `+h` and `-h`, FD includes that jump and the map becomes
speckled.

### D. Completion also depends on fixed hard branches

The matched-ISB completion algebra takes the following as discrete side inputs:

- `hard_visibility`
- `interior_mask`

The completion JVP and backward kernels use those values as fixed branch
selectors:

- `side_sign = local_hard_visibility > 0.0f ? 1.0f : -1.0f`
- early zeroing for `local_interior_mask`

See:

- `radio_map_accumulate.cu:1045`
- `radio_map_accumulate.cu:1106`
- `radio_map_accumulate.cu:1216`
- `radio_map_accumulate.cu:1270`

Those masks are not differentiated. This is acceptable as long as the only
remaining mismatch is a true discrete boundary event. The current problem is
that completion adds a second, avoidable source of non-smoothness through the
`max_incident_weight` surrogate.

### E. Additional hard cutoffs still contribute noise

The completion weight and response also inherit other non-smooth components:

- `_utd_transition_weight(...)` uses `max`/`min` clipping
  in `cell_accumulation.py:125`
- `support_mask` hard-culls unsupported edges
  in `cell_accumulation.py:318`
- `hard_visibility` is thresholded from LoS coherent power
  in `cell_accumulation.py:723`

These remain real branch events, but the debugging evidence indicates that the
dominant completion-only speckle amplifier is still the
`average-response + max-weight + argmax AD` design.

## Debugging Checklist

When this issue is revisited, use the following narrowing order.

1. Confirm that the run uses the same gradient backend for AD and FD.
   Check `ad_backend` and `fd_backend` in the JSON output.
2. Compare `raw_diffraction`, `matched_isb_completion_only`, and
   `folded_diffraction` separately.
3. If noise is visible, first check whether the primal completion field changed
   smoothly between `+h` and `-h`.
4. Check `plus_diffraction_diagnostics` and `minus_diffraction_diagnostics` for:
   - `prepared_state_count`
   - `visible_pair_count`
   - `support_pair_count`
   - `pair_valid_count`
   - `shadow_completion_count`
   - `interior_count`
   - `hard_visibility_zero_count`
5. If the noise is isolated to `matched_isb_completion_only` while
   `raw_diffraction` stays structured, inspect the completion surrogate before
   suspecting the raw diffraction field.
6. Treat `argmax_edge_idx` changes or small `argmax_margin` as the primary
   indicator that max-weight routing is contaminating FD.

## Recommended Remediation

The recommended fix is not to soften geometry visibility or insert heuristic
gradient hacks. The fix should target the completion surrogate itself.

### Phase 1: Add diagnostic observability

Before changing the algorithm, add the following completion-side diagnostics to
the benchmark metadata:

- `sum_incident_weight`
- `max_incident_weight`
- `argmax_edge_idx`
- `support_edge_count`
- `argmax_margin = max_weight - second_max_weight`

This makes it possible to prove that a noisy region aligns with winner-edge
switching instead of only inferring it from the final field.

### Phase 2: Stop using `max_incident_weight` in the production completion path

`max_incident_weight` may remain as a diagnostic output, but it should not drive
the production matched-ISB completion surrogate.

The completion path should use a continuous aggregate weight that is built from
the same edge ensemble that defines the aggregate response.

Minimum acceptable experiment:

- replace `aggregate_incident_weight = scene_max_incident_weight`
  with a bounded continuous aggregate such as `clamp(sum_incident_weight, 0, 1)`

This experiment is not the final preferred formula, but it is the fastest way
to verify that the `argmax` route is the dominant noise source.

### Phase 3: Use a self-consistent aggregate transition model

Preferred production direction:

1. keep per-edge `w_e` and `t_e`;
2. compute
   - `sum_w = sum_e w_e`
   - `sum_wt = sum_e w_e * t_e`
3. define
   - `incident_response = sum_wt / max(sum_w, eps)`
4. replace the scalar weight with a continuous aggregate built from the same
   `w_e` set

Recommended aggregate weight candidates:

- low-risk candidate:
  - `aggregate_weight = clamp(sum_w, 0, 1)`
- preferred candidate:
  - `aggregate_weight = 1 - product_e (1 - clamp(w_e, 0, 1))`

Why these are better than `max_incident_weight`:

- they use all supporting edges, not only the winner;
- they are continuous when edge influence is redistributed;
- they remove the need for `argmax_edge_idx`-based AD routing;
- they keep the weight bounded in `[0, 1]`.

### Phase 4: Keep real discrete events discrete

Do not try to "fix" the remaining AD-vs-FD gap by softening:

- `hard_visibility`
- `interior_mask`
- support-region classification

Those are genuine branch events. The goal is not exact AD-FD identity at every
boundary. The goal is to eliminate the avoidable speckle introduced by the
completion surrogate so that the remaining mismatch is only due to true
topology or visibility discontinuities.

### Phase 5: Regression and acceptance gates

After the surrogate rewrite, rerun the maintained GPU workflow and require all
of the following:

- `raw_diffraction` parity must remain within current tolerance;
- `matched_isb_completion_only` must no longer show dense high-frequency speckle
  in the right-lower shadow region;
- regional `AD/FD` correlation for `matched_isb_completion_only` must improve
  materially for both `tx_x` and `cube1_x`;
- the region-wide high-frequency power ratio of completion-only FD should drop
  sharply relative to the current baseline;
- no public `Scene + Tracer + Result` contract may change;
- no heuristic smoothing or geometry softening may be introduced.

## Suggested Implementation Order

1. Add completion aggregation diagnostics.
2. Introduce an internal diagnostic-only aggregation mode that swaps
   `max_incident_weight` for a continuous aggregate.
3. Re-run the three-cube gradient benchmark and compare:
   - `matched_isb_completion_only`
   - `raw_diffraction`
   - `folded_diffraction`
4. If the speckle drops as expected, replace the production completion
   surrogate with a self-consistent aggregate transition model.
5. Extend native/reference consistency tests to cover the new aggregate
   contract and its JVP/VJP behavior.

## Status Summary

Current decision:

- the lower-right noise is primarily a matched-ISB completion issue;
- the issue is structural and design-level, not a native/reference mismatch;
- the first concrete fix target should be the completion aggregate-weight
  definition, not raw diffraction and not visibility smoothing.
