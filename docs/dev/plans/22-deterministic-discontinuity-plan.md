# Deterministic Discontinuity Plan

Status: Active
Category: Plan
Last reviewed: 2026-05-19

## Purpose

Track deterministic radiomap discontinuity diagnostics and follow-up work for
single-object shadow-boundary cases. The immediate maintained repro is the
single-cube notebook with horizontal power-line checks at `y = -4` and `y = 4`.

## Context

The current discontinuity inspection need is not a new public feature. It is a
diagnostic workflow for understanding how total power and the LoS, reflection,
and diffraction components mix across lines that cut through lit, shadow, and
transition regions.

The reference pattern comes from older field-monitor tests that extracted a
fixed `y = -4` row and compared LoS, diffraction, and total field behavior near
a shadow-boundary transition. The current standalone deterministic radiomap
example uses component power maps instead of field monitor payloads, so the
notebook diagnostic should report power and component fractions from
`result.components`.

## Current Single-Cube Diagnostic

Maintained files:

1. `examples/deterministic_radiomap_single_cube.py`
2. `examples/deterministic_radiomap_single_cube.ipynb`
3. `tests/deterministic/test_example_deterministic_radiomap_single_cube.py`

Current notebook contract:

1. The transmitter is intentionally shifted left of the y-axis to make the
   single-cube shadow and transition regions asymmetric.
2. The notebook extracts the nearest grid rows to `y = -4` and `y = 4`.
3. For each line, it reports total `path_gain`, component powers for `los`,
   `reflection`, and `diffraction`, and normalized component fractions.
4. The line plots are diagnostic outputs only; they do not change solver
   physics, smoothing, shadow-boundary correction, or material behavior.

## Investigation Notes

2026-05-18:

1. `outputs/single_cube_current_2d_path_gain.png` was reproduced pixel-for-pixel
   from commit `4a96da9` with the single-cube notebook path:
   `SingleCubeExperiment(grid_shape=(256, 256), forward_reflection_n_rays=192,
   reflection_max_bounces=2, max_diffractions=1,
   shadow_boundary_correction=True)`, followed by
   `plot_forward(...).figure.savefig(..., dpi=180, bbox_inches="tight")`.
2. The same plot is not evidence that the raw UTD field is continuous at the
   LoS incident shadow boundary. Its apparent ISB smoothing comes from the
   deterministic shadow-boundary correction path being enabled.
3. The correction-enabled field appears to smooth the ISB visually, but it may
   introduce bias relative to the physically correct coherent field. It must be
   treated as a diagnostic/post-processing term until its physical equivalence
   to the missing UTD boundary contribution is proven.
4. Follow-up work must distinguish three quantities: raw coherent UTD
   diffraction, the LoS/reflection support fields it should cancel at physical
   boundaries, and any post-processing correction term. The target fix remains
   native raw UTD field continuity, not heuristic post-processing.
5. The native root cause is not only the beta branch sign. Direct first-order
   3D UTD accumulation was evaluating finite-edge states at the stored edge
   anchor rather than the receiver-dependent stationary diffraction point. It
   also multiplied the direct first-order shadow-boundary transition by a
   finite-edge truncation factor; without endpoint diffraction terms this biases
   the canonical GO/UTD cancellation. The current fix keeps finite-edge support
   and visibility, but evaluates the direct first-order transition at the
   stationary point with the incident Tx field recomputed there.
6. Low-level UTD pair contributions are allowed to carry the physical ISB/RSB
   jump needed to cancel hard LoS/reflection support changes. The acceptance
   target is continuity of the raw coherent total field, without relying on
   `shadow_boundary_correction`.
7. A later raw-field plot exposed a separate finite-edge endpoint artifact:
   direct first-order states were treated as invalid when the receiver-dependent
   stationary point crossed a finite edge endpoint. This created a hard
   endpoint visibility/support cutoff, visible as circular or arc-like jumps in
   the diffraction component. The current native path keeps the finite-edge
   endpoint support as a real visibility term, but softens the former hard
   cutoff with a local wavelength-scaled Gaussian endpoint visibility. It evaluates the local
   UTD state at the receiver-dependent stationary point, uses the clamped
   physical endpoint only for scene visibility, preserves the canonical
   interior UTD strength needed for LoS/reflection shadow-boundary
   cancellation, and rapidly decays endpoint-outside contributions instead of
   allowing the infinite edge extension to remain globally visible.
8. A later shadow-region jump near the top/vertical wedge projections was not a
   finite-edge visibility cutoff. The selected stationary point was inside the
   physical edge, but the projected 2D wedge angle still crossed the direct ISB
   branch at `phi - phi_prime = pi` while the target remained in the local 3D
   wedge shadow (`target_exterior == false`). In that case the 2D branch jump is
   unpaired by any 3D GO support jump. The native scalar/vector UTD path now
   keeps the standard branch for local exterior/physical GO-boundary cases, and
   uses direct-ISB continuation for selected direct first-order states in the
   local 3D wedge shadow.
9. `shadow_boundary_correction=True` is not a physically complete fix for the
   current finite-wedge 3D UTD path. The correction path still matches a direct
   LoS/ISB transition against `los + diffraction`, uses Tx-to-edge visibility
   and the old truncation response, and does not include reflection-prefix RSB
   states or the current selected-stationary native UTD geometry. It can be
   useful as a diagnostic visualizer, but it can also bias the coherent field
   and must not be used as the acceptance mechanism for raw UTD continuity.
10. The apparent empty reflection field for the cube left face was traced to
    first-bounce reflection path discovery, not to reflection EPC or UTD
    physics. With `forward_reflection_n_rays=192`, sampled RayD discovery found
    only the `y=-1` face image source `(-2, 3, 4)` and missed the physically
    valid `x=-1` left-face image source `(0, -5, 4)`. Increasing the ray count
    could discover that face by chance, confirming this was a sampling coverage
    bug. First-bounce reflection discovery now enumerates one image source per
    coplanar scene surface group and leaves per-receiver specular hit, surface
    containment, and occlusion checks to the existing EPC path. The 192-ray
    single-cube case now includes all six cube face image sources and the
    left-face projection has nonzero reflection power without correction.
11. Remaining raw-field discontinuities are concentrated in local 3D wedge
    projection corners. Visual inspection suggests quadrant-dependent behavior:
    some adjacent quadrants remain continuous while transitions involving the
    opposite exterior pole of the projected wedge still jump. The next debugging
    step is to instrument the dominant pair/state near such a corner and compare
    local wedge angles, `source_exterior`/`target_exterior`, endpoint/stationary
    support, and selected ISB/RSB branch handling against the 2D reference.
12. The deterministic shadow-boundary correction path now has an explicit RSB
    term in addition to the existing matched ISB term. Dense correction
    statistics evaluate the same receiver-dependent finite-edge stationary point
    as raw UTD, compute incident and reflection transition responses separately,
    and use max transition weight with weighted response averaging. This makes
    correction useful again as a diagnostic for the reflection-boundary region,
    but it is still not the acceptance mechanism for raw UTD continuity.
13. The dense native radio-map correction kernel has been partially aligned for
    selected-stationary finite-edge geometry, but it still only exposes incident
    statistics. Until native RSB outputs are added, the ISB/RSB correction uses
    the Dr.Jit dense reference statistics for complete transition terms. The
    remaining raw UTD work is to move the same ISB/RSB handling into the native
    diffraction field itself rather than relying on post-processing correction.

## Acceptance Gates

The single-cube discontinuity diagnostic is acceptable when:

1. The selected `y = -4` and `y = 4` rows are reported with their sampled grid
   coordinates, not assumed to land exactly on the requested y values.
2. Total power and all component powers are finite along both rows.
3. Component fractions are normalized from the component-power sum where that
   sum is positive, and are zero where no component contributes.
4. The test suite covers the row extraction and component-mix normalization on
   a small grid.
5. The notebook remains a diagnostic example and does not introduce runtime
   helper paths under `witwin/`.

## Follow-Up Questions

1. Should the discontinuity acceptance check include an explicit jump metric
   near detected LoS or reflection support boundaries, mirroring the old
   `relative_jump` reference test?
2. Should the diagnostic split direct diffraction and mixed
   reflection-diffraction families once deterministic result metadata exposes
   stable per-family component slices?
3. Should the same y-line diagnostic be added to the three-cube notebook for
   comparison against multi-obstacle discontinuities?
4. Should 3D reflected-prefix diffraction states get the same selected
   stationary-point ISB/RSB continuation as direct first-order states, using the
   actual reflected-chain incident direction rather than the direct Tx vector?
5. Should the deterministic dense native shadow-boundary statistics kernel be
   extended to output RSB weights/responses, or should correction remain a
   Dr.Jit-only diagnostic path while raw native UTD continuity is repaired?

## Verification Commands

Use the `witwin2` environment:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest -q tests\deterministic\test_example_deterministic_radiomap_single_cube.py
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\deterministic\test_example_deterministic_radiomap_single_cube.py tests\deterministic\test_deterministic_diffraction_accumulation.py --gpu -q
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\deterministic\test_deterministic_shadow_boundary_correction_guard.py --gpu -q
```

Do not use `ruff` as part of this plan unless a future task explicitly asks for
formatting or lint verification.
