# RayD Deterministic Exact Coherent Accumulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a RayD exact coherent diffraction accumulator for `witwin.channel.deterministic` so deterministic radiomap diffraction can use a fused RayD/OptiX backend without changing the solver's complex-vector coherent semantics.

**Architecture:** Keep the public Channel contract unchanged: users still call `witwin.channel.deterministic.solve(scene, transmitter, receiver, config)`. RayD owns the state-to-grid coherent accumulation launch and OptiX visibility; Channel owns deterministic state preparation, result assembly, metadata, parity gates, and performance benchmarks. Deterministic parity requires a full Channel UTD state payload, not the reduced Monte-Carlo `DfrStates` payload, because the vector formula depends on incident Jones vectors, derivative Jones vectors, face operators, face material parameters, finite-edge bounds, stationary-point flags, and direct/mixed ownership.

**Tech Stack:** Python 3.11 in `witwin2`, DrJit CUDA arrays, RayD C++17/CUDA/OptiX, nanobind, CMake/MSVC on Windows, Channel deterministic solver tests, RayD DrJit unittest tests.

---

## Scope

In scope:

- A RayD exact coherent grid accumulator for deterministic first-, second-, and third-order diffraction state collections.
- Channel integration behind an explicit deterministic diffraction execution mode.
- First-order primal parity against the existing Channel deterministic accumulator.
- Performance stress tests on Munich-style workloads.
- AD routing discipline: explicit unsupported AD rejection first, followed by RayD AD only after finite-difference parity is available.

Out of scope for the first implementation:

- Replacing Path Solver `trace_dfr_paths`.
- Replacing Monte Carlo `accum_dfr_direct` / `accum_dfr`.
- Adding CPU fallback paths.
- Adding heuristic smoothing or soft visibility approximations.

## Current Findings

The current deterministic first-order Munich smoke workload (`16 x 16` receiver grid, `4164` diffraction states) spends about `76.95 s` in `diffraction_accumulation_seconds` on the existing Channel path. The first RayD reduced-state prototype finishes in about `0.034 s`, but it is not numerically equivalent: it uses a scalar sampled diffraction weight and writes only the x vector component. The next implementation must replace that approximation with Channel's native UTD vector formula.

Channel's current native UTD vector formula is defined by:

- `witwin/channel/deterministic/kernels/utd/utd_types.h`
- `witwin/channel/deterministic/kernels/utd/utd_math.h`
- `witwin/channel/deterministic/kernels/utd/utd_accumulate.cu`
- `witwin/channel/deterministic/kernels/utd/native_impl.py::_pack_state_soa`

The parity target is `compute_pair_contribution(PairInputs state, float3a target, float k, MaterialParams mat)` plus the same direct/mixed ownership classification used by `Geo.ownership_code(...)`.

## New RayD Interface

### `DfrCoherentOptions`

```cpp
struct DfrCoherentOptions {
    float wavelength;
    float k;
    int max_order;
    int receiver_model;
    int select_diffraction_point;
    int prefilter_visibility;
    int collect_debug_counts;
};
```

For the first landing, `max_order` must be `1` and `receiver_model` must be `RAYD_DFR_MATCHED_ISO`.

### `DfrCoherentAccum`

```cpp
template <typename Float_>
struct DfrCoherentAccumData {
    int grid_cell_count;
    ComplexT<Float_> direct_field_x;
    ComplexT<Float_> direct_field_y;
    ComplexT<Float_> direct_field_z;
    ComplexT<Float_> multi_field_x;
    ComplexT<Float_> multi_field_y;
    ComplexT<Float_> multi_field_z;
    IntT<Float_> direct_count;
    IntT<Float_> multi_count;
    IntT<Float_> visibility_reject_count;
    IntT<Float_> utd_reject_count;
};
```

The result must preserve coherent complex fields. Channel computes power after summing LoS, reflection, and diffraction.

### `DfrCoherentUtdStates`

The deterministic RayD interface needs a full UTD state payload with fields matching Channel's 84-slot native SoA pack:

- edge geometry: `edge_pos`, `edge_dir`, `n0`, `n_face_n`, `wedge_n`, `edge_line_min`, `edge_line_max`, `source_pos`
- scalar transport: `incident_field`, `incident_normal_derivative`, `r_face0`, `r_face_n`
- vector/Jones transport: `incident_vector_*`, `incident_normal_derivative_vector_*`, `incident_jones_*`, `incident_derivative_jones_*`, `incident_basis_*`
- face operators: `face0_operator_m**`, `face1_operator_m**`
- face materials: `face0_eta_r`, `face0_mu_r`, `face0_sigma`, `face0_gain`, `face0_use_fresnel`, `face1_eta_r`, `face1_mu_r`, `face1_sigma`, `face1_gain`, `face1_use_fresnel`
- dispatch support: `owner_code`, `adjacent_face0`, `adjacent_face1`, `select_stationary_point`

The existing `DfrStates` remains the reduced Monte Carlo/path state table. It is not sufficient for deterministic UTD vector parity.

### `Scene.accum_dfr_coherent_direct(...)`

RayD Python shape:

```python
result = scene.accum_dfr_coherent_direct(
    full_utd_states,
    grid,
    material,
    options,
    active=True,
)
```

Channel wrapper shape:

```python
result = channel_scene.accum_dfr_coherent_direct(
    diffraction_states=state_arrays,
    grid=grid,
    config=resolved_config,
    active=True,
)
```

Behavior:

- Exact lanes are flattened `(state_index, receiver_cell_index)`.
- The kernel computes receiver cell centers from `DfrGrid`.
- The kernel evaluates visibility through OptiX.
- The kernel evaluates the same first-order finite-edge stationary-point policy used by Channel's current deterministic native UTD path.
- The kernel computes `compute_pair_contribution(...)` from the full Channel UTD vector formula.
- The kernel atomically adds complex vector fields into direct or mixed output buffers.
- The first full-parity version rejects AD inputs until the RayD custom-op derivative is implemented and finite-difference tested. Discrete visibility, stationary-point selection, and receiver-cell ownership remain detached in the AD contract.

## Revised Acceptance Gates

- RayD full-state ABI exists and is not confused with reduced `DfrStates`.
- A small first-order deterministic scene matches Channel native vector outputs within `rtol=1e-3, atol=1e-10`.
- Munich `16 x 16` explicit RayD mode is finite and at least `10x` faster than the current Channel backend.
- `auto` is switched only if parity and performance gates pass for non-AD first-order deterministic workloads.
- AD inputs continue to raise a clear error until a fixed-topology RayD AD path is implemented and tested.

## 2026-05-23 Execution Status

- Completed: RayD full-state deterministic UTD ABI (`DfrCoherentUtdStates`) and Channel wrapper routing.
- Completed: RayD OptiX coherent accumulator now evaluates the copied Channel native UTD vector formula instead of the reduced scalar sampled weight.
- Completed: Channel now passes the same material angular frequency and transmitter polarization used by the native UTD path.
- Completed: RayD finite-edge stationary-point validity now matches Channel's finite-edge selection (`0 < parameter < edge_length`).
- Completed: AD-sensitive inputs avoid RayD and stay on the existing DrJit/native path.
- Completed: deterministic `auto` now resolves to RayD exact coherent accumulation for supported primal first-, second-, and third-order axis-aligned grids.
- Completed: Munich `16 x 16` smoke remains finite and shows large performance improvement (`0.0381 s` RayD diffraction accumulation vs `75.069 s` legacy Channel accumulation).
- Parity status: Munich diffraction sums now match within about `0.0065%` (`1.4506975e-08` RayD auto vs `1.4506045e-08` legacy Channel), with remaining differences limited to tiny boundary-tail cells.
- Completed: Munich `256 x 256` memory-safe smoke now runs for one, two, and three diffraction orders with RayD accumulation times `0.193 s`, `0.267 s`, and `0.327 s`; total solve times are `5.915 s`, `9.053 s`, and `10.831 s`.
- Decision: keep RayD enabled only behind the supported-workload `auto` gate; AD and unsupported workloads continue to use the previous backend.

## 2026-05-23 Native State Preparation Status

- Completed: RayD `DfrCoherentEdge` ABI and `Scene.build_dfr_coherent_tx_states(...)` now build compact deterministic first-order Tx states in RayD for non-AD workloads.
- Completed: Channel `tx_first(...)` routes supported non-AD state preparation to RayD and keeps AD-sensitive transmitter, scene-geometry, material, polarization, or active-mask inputs on the existing DrJit path.
- Completed: RayD first-order state preparation now uses the same surface-group-expanded segment visibility payload as `Scene.segment_visible(...)`, source exterior masking, incident scalar/vector/Jones payload construction, face material payload gathering, and compact emission.
- Verified: Munich `all_edges + half_plane` Tx-first state parity is exact for the local edge set (`4164` old states, `4164` RayD states, `0` extra, `0` missing). The isolated Tx-first timing changed from about `4.43 s` to about `0.64 s` on the test machine.
- Completed: RayD now exposes `Scene.build_dfr_coherent_higher_candidates(...)` plus `DfrCoherentCandidatePairs`, a narrow higher-order candidate primitive that emits compact `(prev_state_index, edge_index)` pairs from outgoing state bases using RayD edge-BVH probes. Channel routes supported non-AD `bvh_pairs(...)` calls through this primitive, keeps dedupe, UTD field evaluation, lineage, and AD fallback in Channel, and can request RayD inter-edge visibility filtering when topology preservation is not required.
- Completed: Channel higher-order builder reports now record per-order `stage_seconds` for pre-expansion, higher-order state construction, inserted reflection expansion, post-budget pruning, lineage finalization, and total order time. Use this metadata to decide whether the next native lowering should target UTD field/vector transport, face response, or compact state emission.
- Remaining: full higher-order native state expansion still needs a RayD CUDA state-emission kernel. The existing exact coherent formula is currently device-side accumulator logic, so full lowering must emit compact next-order state payloads from a native visibility/UTD-evaluation launch without moving solver policy or lineage into RayD.

## Task 1: RayD Failing ABI Tests

**Files:**

- Modify: `E:\Code\RayDi\tests\drjit\test_diffraction_accumulation.py`

- [ ] Add a subprocess-isolated test that imports `rayd`, constructs `rayd.DfrCoherentOptions()` and `rayd.DfrCoherentAccum()`, and asserts the expected public attributes exist.
- [ ] Add a test that creates a one-triangle scene and calls `scene.accum_dfr_coherent_direct(...)` with no state data to verify the method exists and raises a clear state-count error.
- [ ] Run:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.drjit.test_diffraction_accumulation -v
```

Expected red result: failure because `DfrCoherentOptions`, `DfrCoherentAccum`, or `Scene.accum_dfr_coherent_direct` does not exist.

## Task 2: RayD ABI And Bindings

**Files:**

- Modify: `E:\Code\RayDi\include\rayd\multipath\diffraction_accumulation.h`
- Modify: `E:\Code\RayDi\include\rayd\rayd.h`
- Modify: `E:\Code\RayDi\src\rayd.cpp`
- Modify: `E:\Code\RayDi\src\scene\scene_multipath.cpp`

- [ ] Add `DfrCoherentOptions`.
- [ ] Add `DfrCoherentAccum` and `DfrCoherentAccumAD`.
- [ ] Add `Scene::accum_dfr_coherent_direct(...)` declarations and nanobind bindings.
- [ ] Implement a minimal placeholder that validates non-empty states and returns zero-filled outputs with the correct grid width.
- [ ] Run the Task 1 RayD tests and confirm the ABI tests pass while numerical tests are still absent.

## Task 3: RayD Primal Kernel Tests

**Files:**

- Modify: `E:\Code\RayDi\tests\drjit\test_diffraction_accumulation.py`

- [ ] Add a one-wall / one-edge coherent accumulation test that asserts `direct_field_x` has receiver-grid width and at least one nonzero finite value.
- [ ] Add a blocked receiver test that places an occluder between the edge and the receiver and asserts the accepted count or field magnitude decreases.
- [ ] Add a debug-counter width test for `direct_count`, `multi_count`, `visibility_reject_count`, and `utd_reject_count`.
- [ ] Run the RayD diffraction tests.

Expected red result: tests fail because the placeholder implementation returns zeros.

## Task 4: RayD Exact Coherent Primal Kernel

**Files:**

- Modify: `E:\Code\RayDi\include\rayd\multipath\diffraction_accumulation_params.h`
- Modify: `E:\Code\RayDi\src\multipath\diffraction_accumulation.cu`
- Modify: `E:\Code\RayDi\src\multipath\pipelines.cpp`
- Modify: `E:\Code\RayDi\src\scene\scene_multipath.cpp`
- Modify: `E:\Code\RayDi\CMakeLists.txt` only if a new translation unit is required.

- [ ] Add OptiX launch parameters for exact coherent accumulation.
- [ ] Add output buffers for six complex vector fields and debug counters.
- [ ] Launch one lane per `(state, receiver cell)`.
- [ ] Reuse existing RayD diffraction UTD math helpers where possible; do not duplicate large formulas unless the existing helpers are device-only and already local to `diffraction_accumulation.cu`.
- [ ] Preserve finite-edge selected-point behavior for direct first-order states.
- [ ] Preserve adjacent primitive ignore behavior for visibility.
- [ ] Route source-type / reflection-depth ownership into direct versus multi outputs in the same way Channel classifies deterministic diffraction components.
- [ ] Regenerate or update the committed PTX header if the build requires it.
- [ ] Run RayD diffraction tests.

## Task 5: RayD Build And Install

**Files:**

- RayD build tree under `E:\Code\RayDi\build`
- Installed `rayd` extension in `witwin2`

- [ ] Build RayD with the existing CMake/Ninja/MSVC flow.
- [ ] Install the built extension into the `witwin2` environment used by Channel.
- [ ] Run:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.drjit.test_diffraction_accumulation -v
```

Expected result: RayD diffraction tests pass.

## Task 6: Channel Failing Routing Tests

**Files:**

- Modify: `tests\deterministic\test_deterministic_diffraction_accumulation.py`
- Modify: `tests\deterministic\test_deterministic_munich_profiler.py`

- [ ] Add a test that deterministic `Tuning(diffraction_execution={"accumulate_primal": "rayd_exact_coherent"})` records metadata `implementation == "rayd_accum_dfr_coherent_direct_exact"` on a small first-order scene.
- [ ] Add a test that explicit `rayd_exact_coherent` raises for `max_diffraction_order > 1`.
- [ ] Add a test that explicit `rayd_exact_coherent` raises for AD-enabled state/material/geometry inputs until RayD AD is implemented.
- [ ] Add profiler CLI parsing for `--diffraction-accumulate-primal rayd_exact_coherent`.
- [ ] Run the focused deterministic tests.

Expected red result: config validation rejects the new mode or the metadata is missing.

## Task 7: Channel Config And Scene Wrapper

**Files:**

- Modify: `witwin\channel\deterministic\config.py`
- Modify: `witwin\channel\core\scene\scene.py`

- [ ] Extend deterministic `AccumulatePrimalMode` to include `"rayd_exact_coherent"`.
- [ ] Add `Scene.accum_dfr_coherent_direct(...)` wrapper that builds RayD `DfrCoherentOptions`, material payload, grid descriptor, active mask, and state table.
- [ ] Reject symbolic recording for this wrapper.
- [ ] Reject AD inputs in this wrapper until Task 12 lands.
- [ ] Preserve DrJit-native arrays at the boundary; do not use NumPy, Torch, or DLPack.
- [ ] Run Channel config and deterministic focused tests.

## Task 8: Channel Deterministic Accumulation Dispatch

**Files:**

- Modify: `witwin\channel\deterministic\diffraction\accumulation.py`
- Modify: `witwin\channel\deterministic\trace\diffraction.py`
- Modify: `witwin\channel\deterministic\solver.py` only for metadata aggregation if needed.

- [ ] Add backend resolver for deterministic diffraction accumulation.
- [ ] Dispatch explicit `"rayd_exact_coherent"` to the RayD wrapper when the workload is supported.
- [ ] Convert RayD `direct_field_*` and `multi_field_*` outputs into Channel vector dictionaries.
- [ ] Preserve the existing UTD native path for `"auto"` and `"drjit"`.
- [ ] Attach metadata with `implementation="rayd_accum_dfr_coherent_direct_exact"`, `coherence="complex_vector_sum"`, `estimator="exact_state_receiver_sum"`, and `ad_contract="primal_non_ad_only"`.
- [ ] Run the Task 6 deterministic tests and confirm they pass.

## Task 9: Channel Numerical Parity Tests

**Files:**

- Modify: `tests\deterministic\test_deterministic_diffraction_accumulation.py`

- [ ] Add a small-scene test that runs the current backend and `rayd_exact_coherent` backend on the same scene and compares diffraction component power sums.
- [ ] Add a complex-vector parity helper for `result.field.vector_coherent["diffraction"]`.
- [ ] Use tolerances no looser than `rtol=1e-3, atol=1e-10` initially; tighten if RayD and Channel math match better.
- [ ] Run the focused deterministic parity test.

Expected result: current backend and RayD exact coherent backend agree within the recorded tolerance.

## Task 10: Deterministic Benchmark Integration

**Files:**

- Modify: `tests\support\bin\profile_deterministic_munich.py`
- Modify: `tests\deterministic\test_deterministic_munich_profiler.py`
- Create or update: `docs\dev\optimization\rayd-deterministic-exact-coherent-accumulator-2026-05-23.md`

- [ ] Add profiler CLI option `--diffraction-accumulate-primal` with choices `auto`, `drjit`, `rayd_optix`, and `rayd_exact_coherent`.
- [ ] Include the selected mode in scenario metadata.
- [ ] Include runtime backend metadata in profiler output.
- [ ] Run a smoke benchmark at `16 x 16` and `64 x 64` if GPU memory permits.
- [ ] Record old backend timing, RayD exact timing, speedup ratio, memory peak, state count, and result finite stats in the optimization note.

## Task 11: Channel Performance Gate

**Files:**

- Modify or create: `tests\performance\test_deterministic_rayd_exact_smoke.py`
- Modify: `tests\support\bin\profile_deterministic_munich.py`

- [ ] Add opt-in pytest smoke coverage behind `--run-optimize` or the existing performance flag pattern.
- [ ] Gate only setup, metadata, and finite output in pytest smoke.
- [ ] Keep speedup comparisons in manual benchmark JSON, not normal CI, unless a baseline file is provided.
- [ ] Run the new smoke test with the opt-in flag.

## Task 12: RayD AD Failing Tests

**Files:**

- Modify: `E:\Code\RayDi\tests\drjit\test_diffraction_accumulation.py`

- [ ] Add a JVP test for source position perturbation through `accum_dfr_coherent_direct`.
- [ ] Add a VJP or backward test for material parameter perturbation if the existing RayD diffraction AD custom-op pattern supports VJP.
- [ ] Add an FD parity helper on a tiny fixed topology scene.
- [ ] Run the RayD AD tests and confirm they fail because AD is not implemented.

## Task 13: RayD AD Implementation

**Files:**

- Modify: `E:\Code\RayDi\include\rayd\multipath\diffraction_accumulation_ad.h`
- Modify: `E:\Code\RayDi\src\multipath\diffraction_accumulation_ad.cu`
- Modify: `E:\Code\RayDi\include\rayd\multipath\diffraction_accumulation.h`
- Modify: `E:\Code\RayDi\src\scene\scene_multipath.cpp`

- [ ] Implement fixed-topology JVP for continuous source/state/receiver/material inputs.
- [ ] Implement VJP where RayD's existing diffraction accumulation AD custom-op already provides a matching pattern.
- [ ] Keep discrete visibility, edge selection, receiver cell ownership, and reject masks detached.
- [ ] Add metadata or documentation in test names that the derivative contract is fixed-topology.
- [ ] Run RayD diffraction AD tests.

## Task 14: Channel AD Routing

**Files:**

- Modify: `witwin\channel\core\scene\scene.py`
- Modify: `witwin\channel\deterministic\diffraction\accumulation.py`
- Modify: `tests\deterministic\test_deterministic_material_gradients.py`
- Modify: `tests\deterministic\test_deterministic_diffraction_accumulation.py`

- [ ] Remove the Phase-1 AD rejection only after RayD AD tests pass.
- [ ] Add Channel FD-vs-AD parity for a tiny deterministic scene using `rayd_exact_coherent`.
- [ ] Keep `auto` on the current backend for AD until this parity test is stable.
- [ ] Run deterministic AD tests.

## Task 15: Documentation And Default Policy

**Files:**

- Update: `FEATURE_LIST.md`
- Update: `docs\dev\optimization\rayd-deterministic-exact-coherent-accumulator-2026-05-23.md`
- Update: `docs\dev\README.md` if this plan is completed or superseded.

- [ ] Add a feature-list entry for explicit deterministic RayD exact coherent diffraction acceleration.
- [ ] Document supported workloads and unsupported workloads.
- [ ] Document AD contract and fixed-topology derivative limits.
- [ ] Keep `auto` unchanged unless benchmark and parity records justify a separate default-switch plan.

## Verification Matrix

RayD:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.drjit.test_diffraction_accumulation -v
```

Channel focused:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\deterministic\test_deterministic_diffraction_accumulation.py tests\deterministic\test_deterministic_munich_profiler.py -q
```

Channel AD:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest tests\deterministic\test_deterministic_material_gradients.py -q --gpu
```

Benchmark smoke:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.support.bin.profile_deterministic_munich --munich-xml E:\Code\witwin-platform\channel\reference\sionna-rt-reference-2.0.1\src\sionna\rt\scenes\munich\munich.xml --grid-size 16 --max-diffractions 1 --reflection-n-rays 1 --reflection-max-bounces 0 --edge-selection-mode all_edges --boundary-edge-policy half_plane --solver-mode fast_approximate --memory-profile memory_safe --diffraction-accumulate-primal rayd_exact_coherent --assert-finite --json
```

## Self-Review

- Spec coverage: The plan covers RayD ABI, RayD primal kernel, Channel dispatch, parity, performance, AD, and docs.
- Placeholder scan: No step uses vague placeholder language.
- Type consistency: The public names are consistently `DfrCoherentOptions`, `DfrCoherentAccum`, and `accum_dfr_coherent_direct`.
- Scope check: Higher-order chain exact accumulation is explicitly deferred; first-order deterministic exact coherent acceleration is independently testable.
