# LoS Shadow-Boundary Jump Analysis (2026-03-29)

## Scope

This note summarizes the investigation around the remaining jump near the LoS shadow boundary in the material-aware path, mainly observed through:

- `tests/main/test_position_rotation_tx_use_scene_materials.py`
- `tests/main/test_material.py`

The goal here is not to present a final theory as fact. It records what was observed, what was tested, what was fixed, and which directions turned out to be wrong or incomplete.

## Problem Statement

Observed behavior:

- Reflection-edge jumps became much smaller when `eps_r` was raised toward a metal-like regime.
- A noticeable jump still remained near the LoS shadow boundary.
- The jump depended strongly on which observable was plotted:
  - scalar total field
  - Jones XY total power
  - per-component fields

This made it easy to mix together several different issues.

## Main Findings

### 1. The old average-TE/TM scalar shortcut was physically wrong

The previous material-aware scalar reflection/diffraction shortcut used:

```text
0.5 * (r_te + r_tm)
```

This is not a physically valid scalarization of Fresnel/Jones transport, because:

- `r_te` and `r_tm` live in a local TE/TM basis
- they are not basis-invariant scalar amplitudes
- directly averaging them can create artificial cancellation

At high `eps_r`, a common limiting behavior is approximately:

- `r_te -> -1`
- `r_tm -> +1`

Then the average tends toward zero even though the reflected field should not disappear.

This explained the earlier wrong trend where the scalar reflection field looked weaker for larger `eps_r`.

### 2. Observable mismatch caused several false alarms

Different figures were not plotting the same quantity:

- `position_rotation_tx_use_scene_materials.png` used `result.primary.field.total`
- `material.png` also included Jones-based power views such as `|Ex|^2 + |Ey|^2`

These are not equivalent.

Important consequence:

- A scalar field can look smooth while Jones/vector power still shows a visible boundary feature.
- A Jones total-power plot is not evidence that scalar total-field computation is wrong.

This was one major source of confusion during the investigation.

### 3. The remaining LoS boundary issue is not explained by a simple global 180-degree phase flip

Because of prior experience in another repo, a sign error was a reasonable hypothesis.

Tests performed:

- flip the whole diffraction field sign
- compare `LoS + reflection + diffraction` versus `LoS + reflection - diffraction`
- inspect whether the LoS shadow-boundary jump disappears while reflection boundaries worsen

Result:

- a global diffraction sign flip did **not** remove the LoS boundary jump in a clean way
- it mainly perturbed other regions and could make reflection-boundary behavior worse

Conclusion:

- a naive global 180-degree sign correction is **not** supported by the current evidence
- if there is still a sign/convention bug, it is more likely local to a specific UTD term, boundary classification, or basis convention

### 4. The `safe_*` masking was not the original cause of the LoS jump

During debugging, a very large spike was found near one boundary in the material-aware vector/Jones diffraction path.

That spike came from unmasked slope/operator terms such as:

- `*_dphi_prime`
- `*_d2phi_phi_prime`

These large terms were entering the forward Jones/vector operator even where the slope state should have been treated as invalid or pole-adjacent.

Masking those terms with the existing `has_slope` / `slope_safe` logic removed that spike.

However:

- this fixed a **secondary blow-up**
- it did **not** explain the original LoS shadow-boundary jump

Current judgment:

- the `safe_*` changes addressed a real bug
- but they were not the root cause of the original boundary discontinuity

### 5. Removing target-side hard half-space clipping improved one real discontinuity

The diffraction path previously hard-clipped target evaluation with an infinite-wedge half-space mask.

That created a hard zero/nonzero discontinuity at one shadow-boundary location.

After removing target-side hard clipping from the forward evaluation:

- the hard zero jump was reduced
- the field became continuous there
- but a remaining LoS-boundary kink still stayed

So there were at least two layers of issues:

1. a hard discontinuity from target-side clipping
2. a remaining mismatch between binary LoS occlusion and diffraction compensation

## What Was Fixed

### A. Material-aware public scalar field no longer uses average TE/TM

The material-aware default public scalar field was changed to use Jones/vector transport first, then scalarize at the receiver stage:

- if `rx_polarization` is explicit: use that receiver projection
- if `rx_polarization is None`: use implicit Tx-co-polar projection

This removed the non-physical average-TE/TM shortcut from the material-aware default path.

### B. Vector/Jones diffraction spike near one boundary was removed

The vector/Jones diffraction path had a real operator bug:

- slope-related forward operators were using unsafe derivative terms near the pole
- this produced a large artificial spike

That spike was removed by masking the unsafe operator terms before building the Jones/vector operator.

### C. Target-side forward diffraction hard clipping was removed

The forward diffraction evaluation no longer hard-zeros targets using the old infinite-wedge target half-space mask.

This improved one class of shadow-boundary discontinuity.

## What Still Looks Unresolved

The remaining LoS-boundary jump does **not** currently look like a solved problem.

Current best interpretation:

- LoS remains a binary visibility field
- diffraction now behaves better, but it still does not fully compensate the binary LoS cutoff at that boundary in the current scalar observable
- the mismatch may still involve:
  - a local sign/convention issue in one UTD family
  - a basis/projection mismatch in the scalarized co-polar observable
  - incomplete shadow-boundary cancellation in the current field decomposition

In other words:

- the old average-TE/TM shortcut was definitely wrong
- the vector spike was definitely wrong
- but the remaining LoS shadow-boundary kink is still an open modeling/debugging problem

## Evidence Collected

### Reflection-edge behavior versus `eps_r`

Raising `eps_r` toward a metal-like regime made the reflection-edge behavior look much more correct.

This supports the idea that:

- the old scalar material shortcut was distorting the field
- high-contrast reflection is being represented more consistently after moving to Jones-based scalarization

### Scalar total field versus Jones total power

Comparing these directly was misleading.

Key lesson:

- always compare like-for-like observables before concluding a computation is wrong

### First-order phase-flip hypothesis

The historical clue from the other repo was useful and worth testing, but current evidence does not support a simple global sign inversion as the primary fix here.

## Wrong or Incomplete Directions

These directions were explored and should not be treated as the main fix:

### 1. Assuming the remaining jump was mainly due to missing transmission

For high-`eps_r` metal-like cases, transmission should be weak anyway.

So transmission is not the right first explanation for the near-PEC LoS-boundary kink.

### 2. Comparing scalar total field directly to Jones power and calling it a bug

Those are different observables.

The mismatch itself is not proof of a solver error.

### 3. Using a global 180-degree diffraction sign flip as the fix

This did not cleanly remove the LoS-boundary jump and often harmed behavior elsewhere.

### 4. Treating `safe_*` masking as the original root cause

The masking fixed a genuine vector/Jones spike, but it was not the original LoS shadow-boundary problem.

## Performance Side Note

The recent slowdown investigation showed:

- previous committed states of the test ran in about 10 to 11 seconds
- the current heavier `test_material.py` by itself could still run in a reasonable time in a clean worktree
- the long stalls were strongly affected by cold runs after changing diffraction kernels and by the enlarged visual test workload

So the timeout problem should not be read as proof that the `safe_*` mask was the physical cause of the LoS jump.

## Current Judgment

The most defensible summary is:

1. The old material-aware scalar averaging was wrong and had to be removed.
2. A separate vector/Jones diffraction spike existed and was fixed by safe operator masking.
3. A hard target-side diffraction clipping discontinuity also existed and was reduced.
4. The remaining LoS shadow-boundary jump is still not fully explained or fixed.
5. The current evidence does **not** support a simple global 180-degree diffraction sign flip.
6. The current evidence also suggests the `safe_*` fix was not the original root cause, only a secondary correction.

## Recommended Next Debugging Steps

If this investigation is continued, the next useful checks are:

1. Compare individual UTD beta groups and face operators across the boundary, not just the final summed diffraction field.
2. Compare scalar co-polar projection against full Jones/vector behavior at the exact same cut line.
3. Inspect whether one specific boundary-family term changes sign relative to the reference implementation, instead of testing a global sign flip.
4. Trace the exact decomposition at the remaining LoS boundary:
   - LoS
   - reflection
   - each diffraction family
   - final scalar projection
5. Keep the current Jones-based scalarization; do not reintroduce average-TE/TM scalar shortcuts.

