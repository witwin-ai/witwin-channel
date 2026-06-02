# Deterministic Endpoint Diffraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic-only endpoint diffraction model for finite wedge edges, and remove the remaining hard finite-edge support artifacts from deterministic raw UTD, path export, and shadow-boundary correction paths.

**Architecture:** Keep the existing locally infinite wedge UTD as the primary edge contribution. Add a separate endpoint contribution family for finite-edge endpoint regions, with explicit support metadata and parity tests between the Dr.Jit reference path and native deterministic kernels. Do not add Monte Carlo support in this plan.

**Tech Stack:** Python 3 in the `witwin2` conda environment, Dr.Jit, deterministic CUDA extension code under `witwin/channel/deterministic/kernels/`, pytest GPU tests.

---

## Current Deterministic Findings

The current deterministic finite-edge handling is a finite-segment correction, not a physical endpoint diffraction field.

1. `witwin/channel/deterministic/path/diffraction_impl/math.py::Geo.finite_edge_diffraction_point` computes a receiver-dependent stationary point on the infinite edge line. It marks whether that point lies inside the finite segment, but validity only requires finite edge length and a finite parameter.
2. `witwin/channel/deterministic/path/diffraction_impl/forward.py::ForwardEval.target_support` moves direct first-order states to that stationary point, uses the clamped endpoint only for segment visibility, and keeps outside-segment stationary points alive.
3. `ForwardEval.stationary_completion_factor` prevents a hard cutoff by normalizing a finite Fresnel endpoint integral and applying a wavelength-scaled Gaussian decay outside the segment.
4. `witwin/channel/deterministic/kernels/utd/native_impl.py::_utd_accumulate_forward_native_primal` repeats the stationary-point prefilter logic before the native kernel, while `witwin/channel/deterministic/kernels/utd/utd_math.h` repeats the selection and finite completion inside CUDA. These duplicate implementations need parity tests because mismatched endpoint classification can create deterministic discontinuities.
5. `witwin/channel/deterministic/kernels/utd/native_impl.py::utd_pair_vectors` packs states without `select_diffraction_point`; suffix evaluation through `ForwardEval.trace_suffix` therefore evaluates pair fields at the stored state anchor, not at a target-dependent stationary point. This is a deterministic suffix-specific endpoint risk.
6. `witwin/channel/deterministic/kernels/radio_map_accumulate/native_impl.py::_reference_shadow_boundary_incident_statistics` uses a finite Fresnel scale but still applies support at the stored edge anchor with hard `wedge_exterior_mask` tests. Shadow-boundary correction can therefore disagree with raw UTD near finite endpoints.
7. Existing test coverage includes a two-cell endpoint continuity regression in `tests/deterministic/test_example_deterministic_radiomap_single_cube.py::test_deterministic_single_cube_raw_utd_finite_endpoint_is_not_hard_cutoff`, but it does not isolate endpoint support, native/Dr.Jit parity, suffix pair-vector behavior, or shadow-boundary correction parity.

Reference model target: Albani, Carluccio, and Pathak describe a UTD treatment for vertices formed by truncated wedges in IEEE TAP 63(7), 3136-3143, DOI `10.1109/TAP.2015.2427877`. This plan implements a scoped deterministic approximation with explicit tests before extending to a full truncated-curved-wedge vertex theory.

## Scope

In scope:

- Deterministic first-order direct endpoint diffraction for finite straight wedge edges.
- Deterministic raw radio-map UTD accumulation.
- Deterministic path export pair evaluation.
- Deterministic reflected suffix pair-vector evaluation where the target is known.
- Deterministic shadow-boundary incident statistics.
- Dr.Jit reference implementation first, native CUDA parity second.

Out of scope:

- Monte Carlo and BDPT endpoint diffraction.
- New public scene APIs.
- New CPU fallback paths.
- A full general vertex theory for arbitrary truncated curved wedges.

## Design

Represent each finite edge as three deterministic contributions:

1. Interior stationary edge contribution: existing UTD edge field when the stationary point is inside the finite segment.
2. Lower endpoint contribution: active when the stationary point exits below `edge_line_min`.
3. Upper endpoint contribution: active when the stationary point exits above `edge_line_max`.

The endpoint contribution is evaluated at the physical endpoint, uses the same incident field and Jones transport machinery as the edge contribution, and has its own endpoint transition weight. The first implementation uses the existing finite Fresnel endpoint residual as the amplitude envelope:

```text
lower_endpoint_residual = finite_wedge_truncation_factor_bounds(line_min, line_max, stationary_u=0)
                          - finite_wedge_truncation_factor_bounds(0, line_max - line_min, stationary_u=0)

upper_endpoint_residual = finite_wedge_truncation_factor_bounds(line_min, line_max, stationary_u=0)
                          - finite_wedge_truncation_factor_bounds(line_min - line_max, 0, stationary_u=0)
```

Use this residual as the endpoint source strength, not as a replacement for the interior edge field. Gate it with geometric visibility from source to endpoint and endpoint to target, adjacent-face ignore rules, source/target exterior classification at the endpoint, and a finite wavelength-scaled transition region. The endpoint contribution must be metadata-visible as `endpoint_lower` and `endpoint_upper`.

---

### Task 1: Deterministic Endpoint Diagnostics

**Files:**
- Create: `tests/deterministic/test_endpoint_diffraction_support.py`
- Modify: none

- [ ] **Step 1: Add a direct geometry unit test for stationary-point side classification**

Create `tests/deterministic/test_endpoint_diffraction_support.py` with:

```python
import pytest

import witwin.channel as wt
from witwin.channel.deterministic.path.diffraction_impl.math import Geo


def test_finite_edge_stationary_point_reports_inside_and_sides():
    state = {
        "edge_pos": wt.Point3f(0.0, 0.0, 0.0),
        "edge_dir": wt.Vector3f(0.0, 0.0, 1.0),
        "source_pos": wt.Point3f(-2.0, -1.0, 0.0),
        "edge_line_min": wt.Float(-1.0),
        "edge_line_max": wt.Float(1.0),
    }

    inside = Geo.finite_edge_diffraction_point(state, wt.Point3f(2.0, 1.0, 0.0))
    below = Geo.finite_edge_diffraction_point(state, wt.Point3f(2.0, 1.0, -4.0))
    above = Geo.finite_edge_diffraction_point(state, wt.Point3f(2.0, 1.0, 4.0))

    assert bool(inside["valid"][0])
    assert bool(inside["inside"][0])
    assert bool(below["valid"][0])
    assert not bool(below["inside"][0])
    assert float(below["parameter"][0]) < 0.0
    assert bool(above["valid"][0])
    assert not bool(above["inside"][0])
    assert float(above["parameter"][0]) > 2.0
```

- [ ] **Step 2: Run the new test and record current baseline**

Run:

```powershell
conda run -n witwin2 python -m pytest tests/deterministic/test_endpoint_diffraction_support.py -q
```

Expected: pass. This test documents the current stationary-point side classification before endpoint diffraction is added.

- [ ] **Step 3: Add a raw UTD continuity test with three samples across one endpoint**

Append to the same file:

```python
import numpy as np

from examples.field_solver_single_cube_main import SingleCubeExperiment


@pytest.mark.gpu
def test_raw_utd_endpoint_crossing_has_no_hard_support_drop():
    experiment = SingleCubeExperiment(
        bounds=((-0.09, 0.09), (7.033, 7.047)),
        grid_shape=(3, 1),
        forward_reflection_n_rays=64,
        reflection_max_bounces=0,
        max_diffractions=1,
        shadow_boundary_correction=False,
    )

    result = experiment.solve()
    coherent = result.field.vector_coherent
    vectors = np.stack(
        [np.asarray(coherent["diffraction"][axis], dtype=np.complex64) for axis in ("x", "y", "z")],
        axis=1,
    )
    magnitudes = np.linalg.norm(vectors, axis=1)

    assert float(np.min(magnitudes)) > 0.1 * float(np.max(magnitudes))
    assert float(np.max(np.linalg.norm(np.diff(vectors, axis=0), axis=1))) < 0.9 * float(np.max(magnitudes))
```

- [ ] **Step 4: Run the endpoint crossing test**

Run:

```powershell
conda run -n witwin2 python -m pytest tests/deterministic/test_endpoint_diffraction_support.py::test_raw_utd_endpoint_crossing_has_no_hard_support_drop -q --gpu
```

Expected before endpoint implementation: pass if the current Gaussian completion is active; keep the numeric result in the task notes because endpoint diffraction will change the expected envelope.

---

### Task 2: Shared Deterministic Endpoint State

**Files:**
- Modify: `witwin/channel/deterministic/path/diffraction_impl/math.py`
- Modify: `witwin/channel/deterministic/kernels/utd/utd_math.h`
- Test: `tests/deterministic/test_endpoint_diffraction_support.py`

- [ ] **Step 1: Add endpoint side metadata to the Dr.Jit finite-edge selection**

In `Geo.finite_edge_diffraction_point`, extend the returned dictionary:

```python
below = finite_valid & (diffraction_parameter <= wt.Float(0.0))
above = finite_valid & (diffraction_parameter >= edge_length)
endpoint_parameter = dr.select(below, wt.Float(0.0), edge_length)
endpoint_point = edge_origin + edge_hat * endpoint_parameter

return {
    "point": diffraction_point,
    "visibility_point": visibility_point,
    "endpoint_point": endpoint_point,
    "edge_line_min": -diffraction_parameter,
    "edge_line_max": edge_length - diffraction_parameter,
    "edge_origin": edge_origin,
    "edge_length": edge_length,
    "parameter": diffraction_parameter,
    "valid": finite_valid,
    "inside": finite_inside,
    "below": below,
    "above": above,
}
```

- [ ] **Step 2: Mirror the side metadata in CUDA**

In `witwin/channel/deterministic/kernels/utd/utd_math.h`, extend `FiniteEdgePointSelection` with:

```cpp
float3a visibilityPoint;
float3a endpointPoint;
bool below;
bool above;
```

Update `finite_edge_diffraction_point` to compute the clamped parameter, visibility point, endpoint point, `below`, and `above` with the same boundary convention as Dr.Jit:

```cpp
float clamped = fminf(fmaxf(parameter, 0.f), edgeLength);
bool below = valid && parameter <= 0.f;
bool above = valid && parameter >= edgeLength;
float endpointParameter = below ? 0.f : edgeLength;
```

- [ ] **Step 3: Extend the unit test to assert side metadata**

Add assertions to `test_finite_edge_stationary_point_reports_inside_and_sides`:

```python
assert bool(below["below"][0])
assert not bool(below["above"][0])
assert bool(above["above"][0])
assert not bool(above["below"][0])
```

- [ ] **Step 4: Run the deterministic endpoint support unit test**

Run:

```powershell
conda run -n witwin2 python -m pytest tests/deterministic/test_endpoint_diffraction_support.py::test_finite_edge_stationary_point_reports_inside_and_sides -q
```

Expected: pass.

---

### Task 3: Dr.Jit Endpoint Contribution Reference

**Files:**
- Modify: `witwin/channel/deterministic/path/diffraction_impl/forward.py`
- Test: `tests/deterministic/test_endpoint_diffraction_support.py`

- [ ] **Step 1: Add endpoint residual helpers**

Add helpers near `ForwardEval.stationary_completion_factor`:

```python
    def endpoint_residual_factors(edge_state, edge_geometry, target_pos, wave: Wave, *, width, inside):
        edge_line_min, edge_line_max = Geo.state_line_bounds(edge_state, context="_finite_wedge_endpoint_residual_factors")
        line_min = repeat_float(edge_line_min, width)
        line_max = repeat_float(edge_line_max, width)
        edge_length = line_max - line_min
        raw = ForwardEval.truncation_factor_with_bounds(
            edge_state, edge_geometry, target_pos, wave,
            width=width, line_min=line_min, line_max=line_max, stationary_u=wt.Float(0.0),
        )
        lower_reference = ForwardEval.truncation_factor_with_bounds(
            edge_state, edge_geometry, target_pos, wave,
            width=width, line_min=dr.zeros(wt.Float, width), line_max=edge_length, stationary_u=wt.Float(0.0),
        )
        upper_reference = ForwardEval.truncation_factor_with_bounds(
            edge_state, edge_geometry, target_pos, wave,
            width=width, line_min=-edge_length, line_max=dr.zeros(wt.Float, width), stationary_u=wt.Float(0.0),
        )
        lower = dr.select(inside, complex_zero(width), raw - lower_reference)
        upper = dr.select(inside, complex_zero(width), raw - upper_reference)
        return lower, upper
```

- [ ] **Step 2: Add endpoint visibility masks in `target_support`**

Inside `target_support`, after `diffraction_point` is built, compute endpoint masks:

```python
endpoint_lower_mask = diffraction_point.get("below", dr.zeros(wt.Bool, width))
endpoint_upper_mask = diffraction_point.get("above", dr.zeros(wt.Bool, width))
endpoint_point = diffraction_point.get("endpoint_point", diffraction_point["visibility_point"])
endpoint_visible = dr.full(wt.Bool, True, width)
if scene is not None and enable_segment_visibility:
    endpoint_target_visible = scene.segment_visible(
        endpoint_point,
        target_pos,
        ignore_prim_idx=ignore_prim_idx,
        ignore_structure_idx=owner_structure_idx,
    )
    endpoint_source_visible = scene.segment_visible(
        source_pos_b,
        endpoint_point,
        ignore_prim_idx=ignore_prim_idx,
    )
    endpoint_visible = endpoint_target_visible & (~direct_mask | endpoint_source_visible)
```

Add these values to the returned support context:

```python
"endpoint_point": endpoint_point,
"endpoint_lower_mask": endpoint_lower_mask & endpoint_visible,
"endpoint_upper_mask": endpoint_upper_mask & endpoint_visible,
```

- [ ] **Step 3: Add endpoint field terms in `to_targets`**

After `scaled_field_jones` is computed, evaluate endpoint residual factors:

```python
endpoint_lower_factor, endpoint_upper_factor = ForwardEval.endpoint_residual_factors(
    edge_state,
    edge_geometry,
    target_pos,
    wave,
    width=width,
    inside=support_context["diffraction_point"]["inside"],
)
endpoint_lower_mask = support_context["endpoint_lower_mask"]
endpoint_upper_mask = support_context["endpoint_upper_mask"]
endpoint_factor = dr.select(
    endpoint_lower_mask,
    endpoint_lower_factor,
    dr.select(endpoint_upper_mask, endpoint_upper_factor, complex_zero(width)),
)
endpoint_jones = jones_scale(field_jones, endpoint_factor)
endpoint_scaled_jones = jones_scale(endpoint_jones, local_scale * phase)
endpoint_vector = vector_from_jones(endpoint_scaled_jones, outgoing_edge_basis)
```

Before returning `field` and `vector_field`, add the endpoint field to the existing edge field on active endpoint masks:

```python
endpoint_mask = endpoint_lower_mask | endpoint_upper_mask
field = dr.select(endpoint_mask, field + endpoint_scaled_jones["u"], field)
vector_field = vector_select(endpoint_mask, vector_add(vector_field, endpoint_vector), vector_field)
```

- [ ] **Step 4: Run deterministic raw endpoint tests**

Run:

```powershell
conda run -n witwin2 python -m pytest tests/deterministic/test_endpoint_diffraction_support.py -q --gpu
```

Expected: pass. If the continuity threshold needs adjustment, record the old and new magnitudes in the test comment before changing the numeric threshold.

---

### Task 4: Native UTD Endpoint Parity

**Files:**
- Modify: `witwin/channel/deterministic/kernels/utd/utd_math.h`
- Modify: `witwin/channel/deterministic/kernels/utd/native_impl.py`
- Modify: `tests/deterministic/test_deterministic_diffraction_accumulation.py`
- Test: `tests/deterministic/test_deterministic_diffraction_accumulation.py`

- [ ] **Step 1: Add CUDA endpoint residual helpers**

Add CUDA equivalents of the Dr.Jit residual calculation beside `finite_wedge_stationary_completion_factor`:

```cpp
UTD_DINLINE void finite_wedge_endpoint_residual_factors(
    PairInputs state,
    float3a tgtPos,
    float k,
    bool inside,
    Complex& lower,
    Complex& upper)
{
    if (inside) {
        lower = cplx_zero();
        upper = cplx_zero();
        return;
    }
    float edgeLength = state.edgeLineMax - state.edgeLineMin;
    Complex raw = finite_wedge_truncation_factor_bounds(
        state, tgtPos, k, state.edgeLineMin, state.edgeLineMax, true);
    Complex lowerReference = finite_wedge_truncation_factor_bounds(
        state, tgtPos, k, 0.f, edgeLength, true);
    Complex upperReference = finite_wedge_truncation_factor_bounds(
        state, tgtPos, k, -edgeLength, 0.f, true);
    lower = cplx_sub(raw, lowerReference);
    upper = cplx_sub(raw, upperReference);
}
```

- [ ] **Step 2: Add endpoint contribution to CUDA pair field**

In `compute_pair_field_terms`, after `field` is computed from incident field and derivative field, add:

```cpp
if (selectedStationary && !selectedInside) {
    FiniteEdgePointSelection endpointPoint = finite_edge_diffraction_point(state, tgtPos);
    Complex lowerResidual, upperResidual;
    finite_wedge_endpoint_residual_factors(state, tgtPos, k, selectedInside, lowerResidual, upperResidual);
    Complex endpointFactor = endpointPoint.below ? lowerResidual : (endpointPoint.above ? upperResidual : cplx_zero());
    Complex endpointGain = cplx_mul(directGain, endpointFactor);
    field = cplx_add(field, cplx_mul(incidentField, endpointGain));
}
```

Use the same pattern in the vector/Jones path if `utd_math.h` has a separate vector pair evaluator.

- [ ] **Step 3: Add native/Dr.Jit parity coverage**

In `tests/deterministic/test_deterministic_diffraction_accumulation.py`, add a GPU test that evaluates a hand-built finite-edge state at one below-endpoint and one above-endpoint receiver with the Dr.Jit path and native path. Use the existing helper patterns in that file for state construction, then assert:

```python
np.testing.assert_allclose(native_values, drjit_values, rtol=2e-3, atol=2e-5)
```

- [ ] **Step 4: Run native parity tests**

Run:

```powershell
conda run -n witwin2 python -m pytest tests/deterministic/test_deterministic_diffraction_accumulation.py -k "endpoint or native_utd" -q --gpu
```

Expected: pass.

---

### Task 5: Deterministic Suffix Pair-Vector Endpoint Handling

**Files:**
- Modify: `witwin/channel/deterministic/kernels/utd/native_impl.py`
- Modify: `witwin/channel/deterministic/path/diffraction_impl/forward.py`
- Test: `tests/deterministic/test_endpoint_diffraction_support.py`

- [ ] **Step 1: Allow native pair-vector evaluation to select stationary points**

Change `utd_pair_vectors` signature:

```python
def utd_pair_vectors(
    state_arrays: dict,
    target_pos,
    *,
    wave: Wave,
    material: Material | None = None,
    select_diffraction_point: bool = True,
):
```

Pack state SoA with:

```python
_pack_state_soa(state_arrays, select_diffraction_point=bool(select_diffraction_point))
```

- [ ] **Step 2: Pass the flag from suffix tracing**

In `ForwardEval.trace_suffix`, call:

```python
field_at_hit, field_at_hit_vector = utd_pair_vectors(
    batch_states,
    hit_p,
    wave=wave,
    material=reflection_material,
    select_diffraction_point=True,
)
```

- [ ] **Step 3: Add a suffix endpoint regression**

Add a GPU test that enables `reflection_max_bounces=1`, `max_diffractions=1`, and a grid strip crossing the finite edge endpoint. Assert the reflected diffraction component has no one-cell hard zero next to nonzero neighbors:

```python
assert float(np.min(diffraction_magnitudes)) > 0.05 * float(np.max(diffraction_magnitudes))
```

- [ ] **Step 4: Run suffix endpoint regression**

Run:

```powershell
conda run -n witwin2 python -m pytest tests/deterministic/test_endpoint_diffraction_support.py -k suffix -q --gpu
```

Expected: pass.

---

### Task 6: Shadow-Boundary Correction Alignment

**Files:**
- Modify: `witwin/channel/deterministic/kernels/radio_map_accumulate/native_impl.py`
- Modify: `witwin/channel/deterministic/kernels/radio_map_accumulate/radio_map_accumulate.cu`
- Test: `tests/deterministic/test_deterministic_shadow_boundary_correction_guard.py`

- [ ] **Step 1: Add stationary endpoint selection to shadow-boundary incident statistics**

In `_reference_shadow_boundary_incident_statistics`, compute the same finite-edge stationary selection used by raw UTD before `support_mask`:

```python
selection_state = {
    "edge_pos": pair_edge_pos,
    "edge_dir": pair_edge_dir,
    "source_pos": pair_tx,
    "edge_line_min": pair_line_min,
    "edge_line_max": pair_line_max,
}
selected = Geo.finite_edge_diffraction_point(selection_state, rx_pos)
support_edge_pos = selected["point"]
visibility_edge_pos = selected["visibility_point"]
```

Use `support_edge_pos` for `wedge_exterior_mask` and finite factor evaluation, and use `visibility_edge_pos` for any segment visibility added in this path.

- [ ] **Step 2: Mirror the same selection in native radio-map accumulate CUDA**

In `radio_map_accumulate.cu`, update the incident statistics kernel to use the same finite-edge stationary point and clamped visibility point as `utd_math.h`. Keep the existing raw inputs and output buffers unchanged.

- [ ] **Step 3: Add guard test for shadow-boundary endpoint parity**

In `tests/deterministic/test_deterministic_shadow_boundary_correction_guard.py`, add a GPU test comparing `shadow_boundary_correction=False` and `True` for a three-cell endpoint crossing. Assert correction does not introduce a larger discontinuity than raw UTD:

```python
raw_step = np.max(np.linalg.norm(np.diff(raw_vectors, axis=0), axis=1))
corrected_step = np.max(np.linalg.norm(np.diff(corrected_vectors, axis=0), axis=1))
assert corrected_step <= 1.25 * raw_step
```

- [ ] **Step 4: Run shadow-boundary correction guard tests**

Run:

```powershell
conda run -n witwin2 python -m pytest tests/deterministic/test_deterministic_shadow_boundary_correction_guard.py -q --gpu
```

Expected: pass.

---

### Task 7: Metadata And Documentation

**Files:**
- Modify: `docs/dev/standards/40-diffraction-path-taxonomy.md`
- Modify: `docs/dev/plans/22-deterministic-discontinuity-plan.md`
- Modify: `FEATURE_LIST.md`

- [ ] **Step 1: Document deterministic endpoint diffraction semantics**

In `docs/dev/standards/40-diffraction-path-taxonomy.md`, update the finite-edge section to include:

```markdown
- `endpoint_diffraction`
  - deterministic-only finite-edge endpoint contribution
  - emits separate lower/upper endpoint terms when the stationary point exits
    the physical edge segment
  - does not replace the interior finite-edge UTD contribution
  - not implemented for Monte Carlo paths
```

- [ ] **Step 2: Update discontinuity notes**

In `docs/dev/plans/22-deterministic-discontinuity-plan.md`, add a dated note that the endpoint hard-cutoff mitigation has been replaced or supplemented by deterministic endpoint diffraction. Include the test commands from Tasks 3, 4, and 6.

- [ ] **Step 3: Update feature list**

Add a concise deterministic solver entry to `FEATURE_LIST.md`:

```markdown
- Deterministic finite-edge endpoint diffraction now adds explicit lower/upper
  endpoint contributions near finite wedge endpoints, reducing endpoint support
  discontinuities without changing Monte Carlo behavior.
```

- [ ] **Step 4: Run doc-sensitive tests**

Run:

```powershell
conda run -n witwin2 python -m pytest tests/test_example_workload_configs.py tests/deterministic/test_endpoint_diffraction_support.py -q --gpu
```

Expected: pass.

---

## Acceptance Commands

Run these before marking the implementation complete:

```powershell
conda run -n witwin2 python -m pytest tests/deterministic/test_endpoint_diffraction_support.py -q --gpu
conda run -n witwin2 python -m pytest tests/deterministic/test_deterministic_diffraction_accumulation.py -k "endpoint or native_utd" -q --gpu
conda run -n witwin2 python -m pytest tests/deterministic/test_deterministic_shadow_boundary_correction_guard.py -q --gpu
```

Expected result: all pass, with no hard endpoint support drop in deterministic raw UTD, suffix pair-vector evaluation, native accumulation, or shadow-boundary correction.
