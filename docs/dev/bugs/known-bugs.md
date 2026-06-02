# Known Bugs

Status: Active
Category: Bug
Last reviewed: 2026-04-10

This file is the single active bug inventory for `docs/dev/`. Legacy bug notes that are already historical or superseded belong in `docs/dev/archive/superseded/`.

## Native UTD Full-Cartesian Into-Buffer Zero-Output Regression

### Symptoms

- The native full-cartesian UTD primal path could return all-zero coherent and
  vector outputs even when the underlying pair contribution was physically
  non-zero.
- The reproduced failure affected the native full-cartesian radiomap
  accumulation route and also the native scalar-power pair wrapper used by the
  matched-isotropic radiomap path.
- The zero result was silent: the CUDA pair math itself remained finite and
  non-zero, but the final Dr.Jit-visible output arrays stayed zero.

### Root Cause

The failure was not in diffraction physics and not in packed-state loading.

Current narrowing established all of the following:

1. `compute_pair_field_terms(...)`, `compute_pair_vector_contribution(...)`,
   and `compute_pair_contribution(...)` all produced non-zero CUDA results for a
   known failing pair.
2. Loading the same pair through the packed 81-slot SoA path also produced the
   same non-zero CUDA result.
3. Receiver target materialization was correct.

The actual defect was the output contract:

- raw native `..._into(...)` launchers wrote directly into Python-owned Dr.Jit
  CUDA arrays through mutable data pointers;
- those pointer writes did not reliably become a new Dr.Jit-visible value;
- as a result, the kernel executed real work but the Python-side arrays stayed
  zero.

In short: the old UTD `into-buffer` API shape was invalid for this Dr.Jit
ownership pattern.

### Current Handling

- The native scalar-power wrapper now uses a return-valued launcher instead of
  mutating preallocated Dr.Jit arrays in place.
- The native full-cartesian primal forward wrapper now uses a return-valued
  tiled-array launcher and accumulates the returned arrays on the Python side.
- Regression coverage now includes the restored full-cartesian native forward
  parity and JVP parity paths.

### Remaining Risk

- Any remaining native wrapper that still depends on raw
  `..._into(preallocated_drjit_arrays)` mutation should be treated as suspect
  until it is explicitly audited.
- This bug class is broader than UTD. It is an output-ownership contract risk
  whenever native code assumes that mutating a Dr.Jit array through a raw data
  pointer is equivalent to producing a new Dr.Jit value.

### Manual Validation

Useful commands for this issue:

```bash
python -m pytest tests/backend/test_native_kernel_consistency.py -k "test_native_utd_forward_matches_drjit_reference or test_native_utd_jvp_geometry_matches_drjit_reference" --gpu
python -m pytest tests/backend/test_native_kernel_consistency.py -k "test_native_utd_forward_uses_native_primal_for_no_grad_inputs or test_native_utd_forward_falls_back_to_drjit_for_ad_sensitive_inputs" -q
```

## RadioMap Native Coherent Diffraction Cell-Center Parity Failure

### Symptoms

- `RadioMapMonitor(combine_mode="coherent")` can produce large diffraction
  spikes when the native coherent accumulation path is used on cell-centered
  multipath scenes.
- The reproduced failure is scene-dependent. The simpler wall parity case still
  passes, but the richer multipath scene used by the radio-map main plots does
  not.
- On a reproduced `128 x 128` multipath scene with `reflection_n_rays=256`,
  `enable_rd_diffraction=True`, and `max_diffractions=2`, the following maxima
  were observed:
  - `FieldMonitor` total power max: about `6.60e-2`
  - `RadioMapMonitor` coherent baseline `path_gain` max: about `2.32e-1`
  - `RadioMapMonitor` coherent native `path_gain` max: about `3.54e5`
- The inflated native value is dominated by diffraction. A reproduced peak was:
  - diffraction amplitude max about `5.95e2`
  - peak location near `x=3.140625`, `y=-0.609375`

### Root Cause

The issue is not limited to `trace_radio_map.py` orchestration.

Current narrowing shows:

1. The failure reproduces when `compute_diffraction_field(...)` is called
   directly on the cell-centered radio-map receiver positions.
2. The same receiver positions, when evaluated through the baseline
   `collect_diffraction_state_paths(...)` path-export route, do not produce the
   inflated coherent result.
3. This means there is a real parity gap between the direct diffraction
   accumulation path and the path-export reference on certain cell-centered
   receiver layouts.

There is also a broader semantic mismatch behind the severity of the divergence:

- the baseline coherent reducer sums per-path scalar coefficients that were
  already projected in a path-local receive basis;
- the native coherent path accumulates vector fields and projects them in the
  monitor-plane tangential basis.

Those two coherent definitions are not equivalent once multiple arrival
directions are involved.

### Current Handling

- The issue is documented and reproducible through
  `tests/support/bin/analyze_radiomap_monitor.py`.
- The simpler wall parity regression remains in place because it still catches
  basic wiring errors for the native coherent path.
- The broader multipath parity failure is now treated as an active known defect
  and a blocker for treating the current native coherent radiomap path as fully
  general-purpose.

### Manual Validation

Useful commands for this issue:

```bash
python -m tests.support.bin.analyze_radiomap_monitor --grid-size 128 --n-rays 256
python -m pytest tests/scene/test_radio_map_monitors.py -q --gpu
python -m pytest tests/main/test_radiomap_native_wall_main.py -q --gpu
```

## Reflection Prefix Canonicalization Regression

### Symptoms

- Reflection-field gradients can disappear entirely in optimization workflows.
- In rotated-scene cases such as `tests.grad.grad_rotation`, the reflected field can become spuriously stronger after rotating the cube, even though the unrotated case looks normal.
- In `tests.support.bin.isb_crosssection` with the rotated high-permittivity
  box scene, the reflection-support boundary can look broken even when the ISB
  side is improved. A reproduced case at `y=-4` showed:
  - reflection boundary near `x≈-3.585`
  - `|E_ref,lit|≈2.01e-3`
  - `|E_dif,lit|≈8.31e-4`
  - `|E_tot,lit|≈1.63e-3`, `|E_tot,shd|≈2.09e-3`
  - relative total jump `≈21.9%`
- In that RSB case, the dominant symptom is not a diffraction shadow-boundary
  branch flip. It is that the forward reflection field itself appears too
  strong, so the reflected field drop is not compensated correctly by the
  available diffraction terms.

### Root Cause

This regression came from two coupled reflection-path changes:

1. Reflection geometry was detached from AD in the reflection field pipeline, which cut gradient flow from the reflected field back to scene geometry.
2. Reflection-prefix path merging was rewritten with a quantize-then-merge Torch path that was not equivalent to commit `82c0caf` (`Fix reflection strength bug`).

The second change is the forward-strength bug. In rotated scenes, the same physical reflection face can be split into multiple near-identical canonical paths. Those duplicate paths are then accumulated as separate contributions, which inflates reflection strength.

The same inflated-reflection symptom can therefore show up in diffraction
diagnostics that happen to cross a reflection-support boundary. In that case the
plot may look like an RSB continuity failure, but the first thing to check is
whether the reflected field itself is already too large before blaming the
diffraction term.

### Current Handling

The current fix keeps the source-level physical path semantics instead of patching the final field:

- Reflection geometry stays on the differentiable path so reflected-field gradients are preserved.
- Canonical path image sources are still averaged with DrJit `scatter_reduce` so the merged path remains differentiable.
- Reflection-prefix canonicalization now uses a greedy clustering scheme per canonical face chain:
  - first canonicalize the reflected primitive chain;
  - preserve first-seen path order;
  - within each chain, compare every candidate against the first-seen representative of the current cluster;
  - merge members whose image-source coordinates are within `REFLECTION_PATH_IMAGE_SOURCE_TOL = 1e-5`.

This greedy clustering is intentional. It matches the old `82c0caf` behavior more closely than the later quantized centroid-based merge, and it avoids creating extra canonical paths after rotation.

### Why Greedy Clustering

The failed implementation effectively merged by quantized buckets and then merged bucket averages again. That allowed tiny rotated-scene perturbations to survive the first grouping step and reappear as duplicated physical paths.

The current greedy clustering instead answers the physically relevant question directly:

- do these rays belong to the same canonical reflection chain?
- are their image sources within the canonical merge tolerance of the same representative path?

If both are true, they are treated as one physical reflection path.

### Regression Coverage

The repository now includes regression coverage for both the generic canonicalization case and the rotated single-cube case in:

- `tests/test_reflection_prefix_path_canonicalization.py`

Relevant manual validation commands:

```bash
python -m pytest tests/test_reflection_prefix_path_canonicalization.py -q --gpu
python -m tests.grad.grad_rotation
python -m tests.grad.grad_reflection
python -m tests.grad.run_all --pytest-defaults
```

## K1 First-Launch Forward Instability

### Symptoms

- With the fused diffraction totals bridge enabled through `DiffractionExecutionConfig(accumulate_primal="custom_op_partitioned", ...)`, the first CUDA forward trace can differ from the second forward trace even when the process, tracer, scene, monitor, and transmitter inputs are identical.
- In multipath visualization workflows such as `tests/main/test_multipath_main.py`, this shows up as horizontal or banded field artifacts and inflated AD/FD mismatch plots.
- The issue becomes easier to see at larger monitor resolutions. Reducing grid size can make the artifact less visible, but does not fix the underlying problem.
- The instability affects the forward field itself, not only gradients.

### Root Cause

Current debugging points to the K1 fused diffraction accumulation path, not the reflection symbolic/evaluated DDA switch:

1. Re-running the same trace with the default K1-enabled runtime produced different forward fields across the first and second launch in the same process.
2. Forcing the strict Dr.Jit accumulation path removed that repeatability failure in the same scenarios.
3. Toggling suffix Slang DDA and suffix symbolic/evaluated DDA did not remove the first-launch drift while K1 stayed enabled.

The current conclusion is that the opt-in K1 CUDA path is not yet repeatable on its first real launch for these workloads.

### Current Handling

- The repository now keeps the K1 fused bridge disabled by default.
- Users must explicitly opt in through `DiffractionExecutionConfig(accumulate_primal="custom_op_partitioned", ...)` to use that path.
- The default strict runtime uses the Dr.Jit accumulation path, which is slower but repeatable in the reproduced multipath cases.

### Temporary Workarounds

- Preferred: keep `accumulate_primal="drjit"` for correctness-sensitive forward-field and AD/FD comparison workflows.
- In local experiments, a small warm-up trace before the main trace often stabilizes the subsequent forward field, but this is not treated as a real fix and should not be relied on for validation or optimization workflows.

### Regression Coverage

The repository now includes a default-path repeatability regression test in:

- `tests/test_default_trace_repeatability.py`

Relevant manual validation commands:

```bash
python -m pytest tests/test_default_trace_repeatability.py -q --gpu
python -m pytest tests/main/test_multipath_main.py -q --gpu
```

## Multipath AD Panel Lazy-Buffer Contamination

### Symptoms

- `tests/output/multipath.png` could show a different `tx_x` AD gradient panel than `tests/output/multipath_tx_x_power_components.png`, even when both panels were supposed to represent the same total-power AD quantity.
- The mismatch appeared directly in the panel title statistics. A reproduced bad case showed:
  - `multipath.png` first-row AD panel: `mean=-112.49 med=-89.34 std=86.09`
  - `multipath_tx_x_power_components.png` total-power AD panel: `mean=-119.19 med=-96.95 std=86.33`
- The issue was specific to the plotting path. Computing the AD panel in isolation produced the same values as the component figure.

### Root Cause

The multipath plotting path kept the DrJit AD result lazy for too long.

In both:

- `tests/main/plot_multipath_components.py`
- `tests/main/test_multipath_main.py`

the code previously did this in the wrong order:

1. compute `ad_grad`
2. launch the FD trace
3. convert `ad_grad` to NumPy

Because `ad_grad` had not yet been materialized, the later FD trace could perturb the backing DrJit buffer. That changed the already-computed AD image before it was converted to NumPy, which made the final rendered AD panel depend on evaluation order.

### Current Handling

The plotting code now materializes the AD image immediately after `ad_gradient_field(...)` returns and before any FD work begins.

The fixed order is:

1. compute `ad_grad`
2. convert `ad_grad` to NumPy
3. launch the FD trace
4. convert the FD result to NumPy

This keeps the rendered AD panel stable and aligns the multipath figure with the per-component figure for the shared `tx_x` total-power panel.

### Regression Coverage

The fix is covered by the existing multipath main figure test:

- `tests/main/test_multipath_main.py`

Relevant validation command:

```bash
python -m pytest tests/main/test_multipath_main.py -q --gpu
```

## Multipath Cleanup Gradient Divergence Behind EPC-Only Forward Recovery

### Symptoms

- On `cleanup`, the multipath diagnostic forward field can become visibly wrong
  if the diagnostic trace stops forcing reflection EPC in
  `tests/main/plot_multipath_components.py`.
- Restoring the forced-EPC diagnostic path brings the forward result back to the
  expected shape and keeps `tests/main/test_multipath_main.py` passing again.
- Even after forward recovery, the AD and FD total-power gradients in the same
  multipath diagnostic remain inconsistent, especially for `tx_x` and
  `cube1_x`.
- The failure is therefore not a pure forward bug. The current reproduced state
  is:
  - forward: correct with the EPC diagnostic override enabled;
  - gradient comparison: still wrong in the `cleanup` multipath workflow.

### Root Cause

The forward regression and the remaining gradient regression are coupled but not
identical.

Current narrowing shows:

1. Removing the diagnostic EPC override changes the reflection forward model and
   breaks the expected multipath forward image on `cleanup`.
2. Restoring the override re-aligns the forward result, so EPC is currently a
   required workaround for the multipath diagnostic on this branch.
3. The AD/FD mismatch still persists after that forward recovery, which means
   the remaining defect sits in the gradient-sensitive reflection/diffraction
   path rather than in the plain forward replay alone.

The exact faulty function has not been isolated yet. Current evidence points to
the AD-sensitive replay/routing chain used by the multipath diagnostic after
EPC-based forward recovery, not to the absence of EPC itself.

### Current Handling

- Keep the forced-EPC diagnostic override in
  `tests/main/plot_multipath_components.py` so the multipath forward result
  stays aligned with the current reference behavior.
- Treat the `cleanup` multipath AD/FD comparison as a known-bad diagnostic until
  the remaining gradient routing issue is fixed.
- Use the current `radio-map-monitor` behavior as the practical reference for
  the expected multipath result while this bug remains open.

### Manual Validation

Useful commands for this issue:

```bash
python -m pytest tests/main/test_multipath_main.py -q --gpu
python -m tests.main.plot_multipath_components --parameter tx_x --grid-size 64 --n-rays 640
python -m tests.main.plot_multipath_components --parameter cube1_x --grid-size 64 --n-rays 640
```

## Direct-Recursive Diffraction Variable-Support FD False Positive

### Symptoms

- In multipath component figures, the `Dif Direct Rec` column could show a nearly full-panel AD/FD mismatch even when most direct-recursive pairs were locally smooth.
- A reproduced `cube1_x` case produced a `Dif Direct Rec` relative L2 error close to `1` and visually looked like the whole recursive field gradient was missing.
- Pair-wise breakdown showed the mismatch was concentrated in a few recursive pairs, especially `8->1`, instead of being a uniform failure across all direct-recursive diffraction terms.

### Root Cause

The main reproduced failure was not a brute-force selection issue and not the scene-level segment visibility mask.

For pair `8->1`, the source-side recursive diffraction geometry sat exactly on a wedge exterior-region boundary. At the base point:

- the wedge-side signed distance was `0`
- `-h` moved the pair to the valid side
- `+h` moved the pair to the invalid side

That boundary flip came from the wedge exterior classification in:

- `witwin/channel/trace/diffraction/geometry/visibility.py`

and then propagated into `geometry_valid` in:

- `witwin/channel/trace/diffraction/field.py`

The result was a variable-support central FD estimate: the `+h` trace could drop the entire `8->1` contribution over a large receiver region while the base and `-h` traces still kept it. AD only differentiated the continuous branch that existed at the base point, so AD and FD were no longer estimating the same mathematical object.

### Current Handling

The diagnostic plotting path now supports a fixed-support FD mode for the direct-recursive diffraction component.

The implemented changes are:

- `samples/save_multipath_main_component_gradient_figure.py` now accepts `--direct-recursive-fd-support {variable_support,fixed_support}` and defaults to `fixed_support`.
- The fixed-support path records the base direct-recursive pair set and replays the same pair/support masks for the `+h` and `-h` traces before forming the FD estimate.
- `witwin/channel/trace/diffraction/field.py` now exposes diagnostic support capture/override hooks so the plotting workflow can freeze `field_valid`, `pole_safe`, `slope_safe`, and `has_slope` during that replay.
- `tests/support/bin/save_multipath_direct_recursive_pair_figure.py` was added to inspect AD / FD / AD-FD maps for each direct-recursive pair separately.

This change is a diagnostic fix. It does not change the physical solver path selection used by the normal trace. It only makes the AD/FD comparison fairer by holding the recursive support set fixed during the finite-difference estimate.

### Known Limitation

- The fixed-support comparison only removes the support-jump false positive in the diagnostic workflow. It is not a general cure for all non-smooth diffraction boundary cases.
- If a comparison intentionally wants the true finite perturbation of the full solver, `variable_support` remains the physically faithful mode because it includes the real support change.

### Manual Validation

Useful commands and outputs for this issue:

```bash
python -m samples.save_multipath_main_component_gradient_figure --parameter cube1_x --direct-recursive-fd-support fixed_support
python -m tests.support.bin.save_multipath_direct_recursive_pair_figure --parameter cube1_x
```

Relevant generated figures:

- `tests/output/multipath_cube1_x_direct_recursive_pairs_all.png`
- `tests/output/multipath_cube1_x_direct_recursive_pairs.png`

## Archived Legacy Bug Note

The old root-level `docs/BUGs.md` note was archived as `docs/dev/archive/superseded/legacy-bugs-note.md`.

That note described a historical reflection-field phase mismatch caused by using cell-center coordinates while the grid itself used edge-point coordinates. The note was already phrased as past behavior and did not include modern reproduction or validation context, so it is not kept in the active bug set.
