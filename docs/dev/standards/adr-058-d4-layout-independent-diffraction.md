# ADR-058: Diffraction accumulation is independent of the receiver layout

- **Status:** Accepted (owner decision D4, analysis report 10.13).
- **Date:** 2026-11-10.
- **Decision ID:** `d4-layout-independent-diffraction`.
- **Scope:** `witwin.channel.deterministic` accumulation of the `diffraction`
  component. Numerical change for scenes with exactly one `ReceiverGrid`.

## Context

`deterministic` accumulated every component, for every receiver layout, through
the native flat accumulator: per receiver, coherent mode squares the summed scalar
row field and incoherent mode sums the row powers. For one layout only, a scene
with exactly one `ReceiverGrid`, the solver then replaced the diffraction
component power with `sum_xyz |sum_rows E_xyz|^2`, a per-cell vector sum that the
diffraction stage builds with a Torch `index_add_`, and in incoherent mode also
swapped it into `path_gain`.

Consequences:

- The same location gave a different diffraction power as a grid cell and as a
  point receiver (the row-equivalence gate measured up to 2.47x relative
  difference per cell in coherent mode and 1.46x in incoherent mode on the wedge
  scene).
- In incoherent mode the grid's diffraction rows were combined coherently, unlike
  every other component and unlike the documented meaning of `coherent=False`.
- In coherent mode the replaced component power no longer matched the total, which
  keeps the scalar row fields.
- The replacement was Torch arithmetic on the hot path (`abs`, `square`, `sum`,
  a layout copy, and in incoherent mode a subtract and an add) and was not
  reproducible: two runs of the unchanged solver differed in the last bits of the
  grid diffraction power because of the atomic `index_add_`.

## Decision

The replacement is removed. The diffraction component accumulates like every
other component for every layout: coherent mode squares the summed scalar row
field, incoherent mode sums row powers, and `path_gain` is the accumulator's total.
`diagnostics["diffraction_accumulation"]` is always `scalar_path_accumulation`.

The alternative, applying the vector sum to every layout, was rejected: it would
keep incoherent mode coherent for one component, keep the component power
inconsistent with the coherent total, and move Torch arithmetic onto the point
path as well. A vector-coherent receiver model belongs in the field stage, where it
would change the totals consistently for all components.

## Evidence

- `tests/propagation/test_entry_row_equivalence.py::test_diffraction_accumulation_is_independent_of_receiver_layout`
  (a 3 x 2 grid against the same six point receivers, coherent and incoherent;
  `component_power["diffraction"]` and `path_gain` agree to 1e-5) was a strict
  xfail before this change and passes after it.
- `tests/deterministic`, `tests/path`, `tests/accuracy` (reference oracles and the
  MC consistency gate): 346 passed, 11 xfailed, unchanged otherwise.
- Launches: the grid solve drops the Torch launches of the replacement (four in
  coherent mode, six in incoherent mode); see the budget update in the same
  release.

## Not changed

The diffraction stage still built the `diffraction_vector_field` sidecar (and its
capacity sanitizer) after this change, with no reader. The M3-7 fused field kernel
change removed it: the Torch `index_add_` in `interactions.diffraction`, the engine
plumbing, the `EvaluatedPathSidecars` field, and the native
`enumerated_capacity_failure_vector_sanitize` binding and kernel.
