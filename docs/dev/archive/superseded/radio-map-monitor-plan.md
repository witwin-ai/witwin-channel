# RadioMapMonitor Implementation Plan

Status: Active
Category: Plan
Last reviewed: 2026-04-04

## Objective

Implement a `RadioMapMonitor` that is a practical replacement for Sionna RT 2.0.0 radio maps while staying aligned with this repository's architecture and standards:

- public architecture stays `Scene + Tracer + Result`
- reflection and diffraction contributions remain differentiable
- runtime hot paths stay Dr.Jit-native at the boundary and CUDA-native inside
- new dense-field work follows the CUDA kernel development and migration standards
- no new Torch, NumPy, or DLPack transport is introduced inside hot kernel paths

This document is a phased implementation plan only. It does not authorize partial shortcuts such as a Torch-side prototype becoming the long-term runtime path.

## Implementation Status

Current rollout status as of 2026-04-04:

Completed baseline work:

1. Phase 1 is implemented:
   - `RadioMapMonitor` is a first-class monitor kind.
   - `RadioMapResult` is integrated into the stable `Scene + Tracer + Result` architecture.
   - `Scene(...)`, `Tracer.trace(...)`, `Tracer.trace_many(...)`, and `Result.monitor(...)` now route radio-map payloads directly.
   - `RadioMapMonitor` now defaults to first-order diffraction unless `max_diffractions` is overridden explicitly.
2. Phase 2 is implemented:
   - axis-aligned cell-centered planar radio maps run on the existing differentiable LoS, reflection, and diffraction solver stack,
   - the baseline metric contract is frozen as incoherent `sum |a_path|^2`,
   - optional coherent diagnostics remain available for parity and debugging,
   - tracer-level reflection discovery is now shared across compatible radio-map and path-monitor workloads instead of rediscovering the same specular families per monitor.
3. Phase 5 baseline-quality quadrature support is implemented:
   - fixed `center` and `stratified_fixed_n` sampling modes now exist,
   - weighted per-cell averaging is part of the public metric contract.
4. Phase 6 baseline metric utilities are implemented:
   - `RSS`, single-trace `SINR`, and `trace_many(...)` multi-TX `SINR` plus transmitter-association maps are available,
   - result metadata records the reducer assumptions and noise-power source,
   - `RadioMapResult.sample_metric_positions(...)` provides world-space sampling on the computed radio map with metric and transmitter-association filters,
   - AD-safe repeated radio-map traces now reuse prepared direct-diffraction state groups through the tracer-level persistent cache, matching the established PathMonitor cache policy,
   - radio-map metrics now support explicit `combine_mode="incoherent"|"coherent"` selection while keeping incoherent accumulation as the default contract.
5. Phase 7 baseline oriented planar support is implemented:
   - rotated planar radio-map surfaces are supported through `center`, `orientation`, and `size`,
   - axis-aligned surfaces remain the explicit fast-path shape and default user entrypoint.
6. Phase 3 native reflection radio-map accumulation is implemented for the production axis-aligned coherent path:
   - `native_grid.py` now feeds the existing native reflection grid accumulator directly,
   - coherent axis-aligned radio maps auto-select this native path through `trace_radio_map_monitor(..., radio_map_accumulation_backend="auto")`,
   - public result metadata records the resolved radio-map accumulation backend and the reflection backend that actually ran.
7. Phase 4 native diffraction radio-map accumulation is implemented for the production axis-aligned coherent path:
   - radio-map tracing now reuses receiver tiling, reflected-suffix replay, and native UTD accumulation through `_accumulate_state_subset_field(...)`,
   - the native coherent path keeps its own full diffraction-state layout and cache key so it does not alias the reduced path-export layout used by the baseline reducer,
   - regression coverage now includes baseline-vs-native parity on a scene that exercises both reflection and diffraction contributions, plus a public `Tracer.trace(...)` test that verifies automatic native coherent dispatch.
8. The native incoherent follow-up phase is partially implemented as an explicit backend:
   - `RadioMapMonitor(..., accumulation_backend="cell_accumulation")` now routes axis-aligned gradient-disabled incoherent radio maps through direct reflection-family replay plus direct diffraction-state scalar-power accumulation,
   - the path reuses receiver tiling, reflection-family tile plans, and the reduced diffraction state layout instead of materializing path-export payloads,
   - diffraction scalar-power accumulation currently uses a reliable native UTD pair-EPC route that evaluates per-pair vector contributions in CUDA and performs the final scalar-power scatter reduction in Python/Dr.Jit,
   - radio-map scheduling is now owned by `monitors/radio_map/scheduler.py` for both reflection and diffraction so the monitor no longer directly inherits field-style tiling decisions,
   - the coherent diffraction path now routes through `monitors/radio_map/native_coherent.py` plus the monitor-neutral `trace/diffraction/suffix.py` module instead of calling the field-only accumulation adapter directly,
   - regression coverage now includes baseline-vs-native incoherent parity on a wall scene with both reflection and diffraction present, plus a public `Tracer.trace(...)` test that verifies explicit backend dispatch and runtime metadata.

Still pending:

1. Phase 8 final rollout closure is still pending, but the benchmark harness now exists through `python -m tests.support.bin.benchmark_radio_map_monitor --json --strict-gates` for axis-aligned center, quadrature multipath, multi-TX SINR, oriented-plane, matched baseline incoherent wall, explicit native incoherent wall parity, and native coherent wall workloads.
2. The explicit `cell_accumulation` backend remains correctness-first at the architecture level, but the `512 x 512` wall regression is no longer a blanket dense-map failure after the April 4 scheduler fix. The latest reviewed dense wall benchmark (`--repeats 1 --warmup 1 --strict-gates --include-large-wall-512`) showed:
   - `baseline_incoherent_wall_512_center`: about `90.46 ms`
   - `cell_accumulation_wall_512_center`: about `81.89 ms`
   while `radio_map_accumulation_backend="auto"` still keeps incoherent radio maps on the baseline path until the broader production kernel ownership plan is finished.
3. The large-wall diagnostics identified the real regression cause: the previous implementation reused field-style receiver tiling too aggressively for radio-map scalar-power diffraction replay, fragmenting a `512 x 512` wall run into `1024` tiny tile calls even when the tile plan only reduced the pair set to about `62.5%` of the dense cartesian workload. The new radio-map-specific scheduler guard forces those dense cases back onto cartesian replay and removes the multi-second overtiling path.
4. The next optimization phase must still push incoherent scalarization and final power accumulation further into the production CUDA kernels so the broader scaling story no longer depends on Python/Dr.Jit replay orchestration around the diffraction path.
5. The Sionna-alignment analysis recorded in `docs/dev/optimization/radio-map-monitor-sionna-alignment-analysis-2026-04-04.md` still raises an additional production priority:
   - an explicit radio-map receiver model is needed to separate Sionna-style matched-isotropic power accumulation from the current projected-polarization reducers.
6. The current native coherent diffraction path still has a known parity gap on a richer cell-centered multipath scene even though the simpler wall parity case passes. That defect is tracked in `docs/dev/bugs/known-bugs.md` and blocks treating the native coherent path as fully general-purpose.
7. The focused production follow-up for Sionna-style direct in-loop cell accumulation is tracked separately in `docs/dev/plans/radio-map-native-cell-accumulation-plan.md`. That document owns the remaining work to turn `cell_accumulation` from a correctness-first replay path into a true direct cell-accumulation solver.

## Authority And Constraints

This plan is constrained by the following active repository standards:

- `docs/dev/standards/cuda-kernel-development-guide.md`
- `docs/dev/standards/cuda-kernel-migration-workflow.md`
- `docs/dev/standards/grid-monitor-sampling-standard.md`
- `docs/dev/standards/test-and-acceptance-workflow.md`

The key consequences are:

1. `FieldMonitor` sampling semantics are frozen as boundary-point samples plus legacy span-over-`n` binning. `RadioMapMonitor` must therefore be a new explicit monitor kind with its own sampling contract. We must not silently change `FieldMonitor`.
2. Native CUDA is the production target. Dr.Jit reference implementations remain required for parity and regression checks, not as the intended dense-field endpoint.
3. AD-complete kernels must own `eval()`, `forward()`, and `backward()` at the C++ custom-op boundary when the runtime path is differentiable.
4. We must not introduce CPU staging, Python pointer bridges, or Torch transport in the kernel hot path.
5. Reflection and diffraction gradients must remain physically derived. No smoothing hacks, heuristic interpolation, or artificial gradient surrogates are allowed.

## Sionna RT 2.0.0 Reference Baseline

The local reference snapshot for this plan is:

- `sionna-rt-reference-2.0.0/`
- upstream tag: `v2.0.0`

The relevant Sionna RT 2.0.0 behaviors are:

1. `RadioMapSolver` computes cell-averaged path-gain style metrics over a measurement surface.
2. `PlanarRadioMap` is cell-centered and parameterized by `center`, `orientation`, `size`, and `cell_size`.
3. Metrics include at least path gain, RSS, SINR, transmitter association, and random position sampling.
4. The solver updates radio-map cells inline during path traversal instead of materializing a separate dense path-times-receiver product for specular contributions.
5. The reference implementation is not a usable drop-in runtime architecture for this repository because:
   - its default loop mode is symbolic and not the AD-complete path we require,
   - it is built around Sionna/Mitsuba internals instead of the current `witwin.channel` stack,
   - it only covers first-order diffraction,
   - it does not satisfy this repository's CUDA custom-op ownership standard.

## Current Repository Baseline

The repository already has several pieces that should be reused instead of replaced:

1. `FieldMonitor` / `trace_field_monitor()` already orchestrate LoS, reflection, and diffraction within the monitor-first `Tracer.trace()` flow.
2. Reflection already has:
   - EPC,
   - native `reflection_grid` CUDA custom ops,
   - receiver-tiling infrastructure,
   - axis-aligned plane support (`x`, `y`, `z`).
3. Diffraction already has:
   - unified state preparation,
   - native UTD accumulation,
   - native reflected-suffix accumulation,
   - receiver-tiling infrastructure,
   - solver metadata and audit reporting.
4. `Result` already supports monitor-first payloads and can be extended with a dedicated radio-map result object.

What is still missing for a full radio-map feature is:

1. a dedicated monitor kind and result contract,
2. a cell-centered sampling contract distinct from `FieldMonitor`,
3. metric reducers for path gain, RSS, SINR, and transmitter association,
4. dense-field accumulation paths optimized for radio-map semantics instead of coherent field snapshots,
5. rotated planar support as a later optional extension,
6. end-to-end acceptance coverage for radio-map semantics and gradients.

## Design Position

The implementation should not treat `RadioMapMonitor` as a renamed `FieldMonitor`.

The correct design is:

1. `FieldMonitor` remains the coherent field monitor with frozen legacy sampling semantics.
2. `RadioMapMonitor` becomes a new monitor type with explicit cell-centered measurement semantics.
3. The default radio-map metric is incoherent path gain style accumulation:
   - per sample point, accumulate per-family projected field contributions,
   - compute `sum |a_family|^2` for radio-map metrics,
   - keep coherent total-field diagnostics optional rather than making them the public metric definition.
4. An explicit opt-in coherent combine mode is allowed:
   - compute `|sum a_path|^2` per sample point before quadrature averaging,
   - keep this as a monitor-level switch instead of replacing the default incoherent contract.
4. Differentiability is preserved because each family contribution is still produced by differentiable reflection and diffraction kernels, and the metric reducer uses differentiable algebra (`abs^2`, weighted sums, scaling by transmit power, noise, and interference terms).
5. Dense runtime work must be organized around receiver tiles and reusable family descriptors, not Python-side Cartesian expansion. The explicit `cell_accumulation` parity backend is a step in that direction, but it is not yet the final production performance architecture.

## Target Public Surface

The planned public surface is:

1. `RadioMapMonitor(...)`
   - with explicit `accumulation_backend="auto"|"baseline"|"native_coherent"|"cell_accumulation"` backend selection
2. `Tracer.trace(..., monitor=RadioMapMonitor(...))`
3. `Result.monitor(name)` returning a radio-map-aware result payload
4. `RadioMapResult` convenience accessors for:
   - `path_gain`
   - `rss`
   - `sinr`
   - `tx_association()`
   - optional `sample_positions(...)`

Planned v1 monitor parameters:

- `name`
- `axis`
- `position`
- `bounds`
- `cell_size` or `grid_shape`
- `metric`
- `tx_power`
- `noise_power` override or scene-derived noise use
- `max_diffractions`
- `quadrature_mode`
- `samples_per_cell`

Deferred parity parameters:

- arbitrary rotated planar surfaces
- advanced position-sampling utilities

## Metric Contract

The radio-map metric contract should be frozen early:

1. Cell geometry:
   - radio-map cells are cell-centered, not boundary-point samples,
   - planar v1 uses axis-aligned monitor planes,
   - rotated planar support is optional and later,
   - mesh measurement surfaces are out of scope.
2. Path gain:
   - default metric is the cell average of incoherent received power contribution over the cell,
   - computed as a weighted sum over fixed sample points per cell,
   - optional coherent mode computes the weighted cell average of `|sum a_path|^2` over the same fixed sample points.
3. RSS:
   - `rss = tx_power * path_gain`
4. SINR:
   - `sinr_tx = rss_tx / (noise + sum_other_tx rss_other_tx)`
5. Differentiability scope:
   - gradients are supported with respect to TX position, geometry, and material parameters already supported by the underlying reflection and diffraction solvers,
   - no promise is made that cell-membership discontinuities become smooth,
   - fixed quadrature positions and fixed tile topology remain the differentiable contract.

## Architectural Decomposition

The implementation should be split into five layers.

### 1. Public Monitor Layer

New modules:

- `witwin/channel/monitors/radio_map/radio_map_monitor.py`
- `witwin/channel/monitors/radio_map/__init__.py`

Responsibilities:

- validate radio-map parameters,
- freeze the cell-centered monitor contract,
- convert user configuration into a runtime grid or measurement-surface descriptor.

### 2. Radio-Map Grid / Surface Layer

New modules:

- `witwin/channel/monitors/radio_map/grid.py`

Responsibilities:

- cell-centered planar coordinate generation,
- per-cell quadrature sample generation,
- tile metadata generation for radio-map cells,
- later rotated-plane descriptors if the planar v1 path is stable.

### 3. Monitor Orchestration Layer

New module:

- `witwin/channel/monitors/radio_map/trace_radio_map.py`

Responsibilities:

- call LoS, reflection, and diffraction solvers,
- choose metric reducers,
- manage transmitter aggregation for RSS/SINR,
- assemble `RadioMapResult` payloads and metadata.

### 4. Kernel Layer

Expected kernel families:

- `witwin/channel/kernels/radio_map_grid/*`
- `witwin/channel/kernels/reflection_family/*`
- `witwin/channel/kernels/utd_tile/*`
- `witwin/channel/kernels/suffix_tile/*`

Responsibilities:

- dense-field tiled accumulation for radio-map samples,
- per-family power accumulation without Python-side dense pair products,
- AD-complete custom-op ownership for differentiable runtime paths.

### 5. Result Layer

Changes:

- extend `witwin/channel/result.py`

Responsibilities:

- add `RadioMapResult`,
- keep `Result.monitor(name)` stable,
- expose metrics and metadata without forcing Torch into the core runtime path.

## Phase Plan

## Phase 0: Semantic Lock And Reference Audit

Objective:

- freeze the target semantics before implementation starts.

Tasks:

1. Audit the Sionna RT 2.0.0 planar and mesh radio-map feature surface.
2. Decide the exact v1 scope split:
   - axis-aligned planar only for first delivery,
   - rotated planar only if later data justifies it,
   - mesh surfaces excluded from this plan.
3. Freeze the metric equations, metadata contract, and non-goals.
4. Define parity fixtures that will be used later for validation.

Deliverables:

- this plan document,
- fixture inventory for planar path-gain/RSS/SINR checks,
- explicit statement that `RadioMapMonitor` is a new monitor kind.

Exit criteria:

- no remaining ambiguity about cell-centered semantics,
- no ambiguity about incoherent radio-map metrics versus coherent field outputs.

## Phase 1: Public API And Result Scaffolding

Objective:

- land the monitor/result surface without changing solver numerics yet.

Tasks:

1. Add `RadioMapMonitor`.
2. Add `RadioMapResult`.
3. Extend `Tracer.trace()` monitor resolution and dispatch.
4. Extend metadata schemas for radio-map monitor provenance.
5. Keep the existing `Scene + Tracer + Result` architecture unchanged.

Deliverables:

- monitor dataclass and validation helpers,
- result dataclass and serialization helpers,
- tracer dispatch integration.

Exit criteria:

- `Tracer.trace(..., monitor=RadioMapMonitor(...))` is routable,
- result objects can carry radio-map payloads,
- no solver hot path is yet rewritten.

## Phase 2: Cell-Centered Planar Baseline On Existing Solvers

Objective:

- create a correctness-first planar baseline before any new CUDA kernel migration.

Tasks:

1. Implement a cell-centered planar grid descriptor independent of `Field`.
2. Generate one fixed sample per cell center as the first baseline.
3. Reuse existing LoS, reflection, and diffraction solvers to evaluate those sample points.
4. Implement differentiable metric reducers:
   - path gain,
   - RSS,
   - placeholder SINR with explicit multi-TX aggregation path.
5. Preserve optional coherent diagnostic outputs for debugging and parity.

Rationale:

- this phase proves the API and metric contract with the current solver stack,
- it creates a regression baseline for later native kernel work,
- it avoids designing kernels against an unstable semantic target.

Known limitation of this phase:

- it is not yet the final performance architecture,
- one center sample per cell is not a full cell-average estimator.

Exit criteria:

- axis-aligned planar radio maps are numerically correct at cell centers,
- reflection and diffraction remain differentiable through the baseline path,
- regression fixtures exist before kernel migration begins.

## Phase 3: Reflection Radio-Map Native Accumulation

Objective:

- replace reflection-side dense radio-map accumulation with a CUDA-first tiled path.

Tasks:

1. Introduce a reflection-family descriptor specialized for radio-map evaluation.
2. Reuse or extend existing receiver-tile infrastructure for cell-centered radio-map samples.
3. Add a native reflection radio-map kernel that:
   - replays reflection families per tile,
   - accumulates per-sample projected fields,
   - emits incoherent power contributions and optional coherent diagnostics.
4. Keep AD-complete ownership in the C++ custom op.
5. Preserve EPC semantics for fresh AD-sensitive traces. Do not freeze discovery incorrectly.

Key design rule:

- reflection radio-map accumulation must not fall back to Python-side pair expansion for the intended dense-field path.

Exit criteria:

- planar reflection radio-map workloads run through a native tiled path,
- JVP and VJP parity exist against the baseline,
- no Torch or NumPy staging exists in the reflection hot path.

## Phase 4: Diffraction Radio-Map Native Accumulation

Objective:

- migrate direct and mixed diffraction radio-map accumulation to the CUDA-first tiled path.

Tasks:

1. Add a radio-map-aware UTD accumulation path that works on cell-centered sample points.
2. Preserve current diffraction family semantics:
   - direct diffraction,
   - reflection-prefix diffraction,
   - higher-order diffraction,
   - reflected suffix after the last diffraction.
3. Extend the native tiled UTD path to emit radio-map metric contributions without forcing a dense `(state, receiver)` Python materialization.
4. Extend suffix accumulation to the radio-map sample grid using existing tile planning as the baseline.
5. Keep diffraction AD native for the runtime path.

Key design rule:

- direct and mixed diffraction must stay differentiable and must not route large workloads through Dr.Jit replay as the long-term implementation.

Exit criteria:

- direct and mixed diffraction radio-map workloads run through native accumulation,
- gradient tests pass for geometry-sensitive and material-sensitive cases,
- memory use no longer scales through a Python-side dense pair product.

## Phase 5: Cell Integration And Quadrature Upgrade

Objective:

- move from cell-center sampling to an actual cell-average estimator.

Tasks:

1. Add fixed sampling modes for planar cells:
   - `center`,
   - possibly `stratified_fixed_n`.
2. Keep sample positions fixed per cell so the differentiability contract remains stable.
3. Share tile-local data across quadrature samples to avoid multiplying intermediate buffers linearly in Python.
4. Add radio-map kernels that accumulate directly into per-cell sample accumulators or fuse the quadrature loop inside the kernel when practical.

Key design rule:

- the first-quality cell-average estimator must still obey the CUDA standard: Dr.Jit arrays in, C++ owns pointers, CUDA owns the heavy loop.

Exit criteria:

- path gain is computed as a true weighted per-cell average,
- intermediate memory remains tile-local or sample-local inside kernels,
- no Python loop becomes the dominant dense-field cost.

## Phase 6: Multi-Transmitter Metrics And Utilities

Objective:

- complete the metric surface expected from a practical radio-map feature.

Tasks:

1. Support multi-transmitter aggregation in a single trace request or across `trace_many(...)`.
2. Finalize `RSS` and `SINR` reducers using scene or explicit noise power.
3. Add transmitter association maps.
4. Add optional position-sampling utilities on top of the computed radio map.
5. Extend result metadata to record the metric reducer inputs and assumptions.

Key design rule:

- transmitter aggregation logic may use Torch only at explicit result-adapter utilities if needed, but core metric assembly should remain in the Dr.Jit/CUDA path whenever it participates in gradients.

Exit criteria:

- practical radio-map metrics match the frozen contract,
- result metadata can explain every reported metric value.

## Phase 7: Rotated Planar Surfaces

Objective:

- move beyond axis-aligned planar monitors toward Sionna-style oriented planes.

Tasks:

1. Introduce an explicit rotated planar surface descriptor.
2. Add coordinate transforms between world space and surface-local sample space.
3. Extend reflection and diffraction kernels or add a measurement-surface boundary layer that maps local tile coordinates to world positions.
4. Preserve axis-aligned fast paths for the common case instead of replacing them with a slower generic path.

Key design rule:

- rotated planar support should be additive. The axis-aligned path must remain the optimized fast path.

Exit criteria:

- rotated planar radio maps work with the same metric contract,
- axis-aligned performance does not regress.

## Phase 8: Validation, Benchmarking, And Rollout Closure

Objective:

- close the implementation with the repository's standard acceptance and performance gates.

Tasks:

1. Add targeted pytest coverage first, then broader GPU and acceptance runs.
2. Add AD-sensitive tests for:
   - TX position,
   - scene geometry,
   - scene materials where supported.
3. Add benchmark matrices for:
   - planar axis-aligned radio maps,
   - reflection-heavy scenes,
   - diffraction-heavy scenes,
   - multi-TX SINR workloads,
   - rotated planar cases when that phase lands.
4. Run benchmarks at least twice and compare steady-state results.
5. Update `FEATURE_LIST.md` only when the feature is actually implemented.

Exit criteria:

- parity and regression tests are stable,
- benchmark results are attached to the rollout,
- the runtime default is justified by measured data, not by assumption.

## Recommended Landing Order

The recommended implementation order is:

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5
6. Phase 6
7. Phase 7
8. Phase 8

Reasons:

1. The semantic contract must stabilize before kernel migration.
2. Reflection and diffraction should migrate only after a correctness-first baseline exists.
3. Rotated planar support should not block the first axis-aligned release.

## Main Risks

1. Reusing `FieldMonitor` semantics by accident would violate the frozen sampling standard and cause silent behavior drift.
2. Computing radio-map metrics from coherent total field instead of per-family power would not be a real Sionna-style replacement.
3. Allowing Python-side dense pair expansion to survive in the intended dense-field runtime path would break the performance goal.
4. Letting native kernels be primal-only while AD falls back to Dr.Jit replay would violate the current kernel-completeness standard.
5. Rotated planar and mesh support can easily contaminate the axis-aligned fast path if not isolated behind explicit descriptors.

## Acceptance Summary

The feature should only be considered complete when all of the following are true:

1. `RadioMapMonitor` exists as a first-class monitor kind.
2. Path gain, RSS, and SINR are implemented with a frozen documented contract.
3. Reflection and diffraction contributions remain differentiable on the supported runtime path.
4. Dense radio-map execution is CUDA-first and AD-complete on the intended production path.
5. Axis-aligned planar radio maps are validated first; rotated planar support lands only behind its own tests and benchmarks.

## Out Of Scope For The First Delivery

1. Changing `FieldMonitor` semantics.
2. CPU fallback implementations added for convenience.
3. Torch-transport kernel paths.
4. Heuristic smoothing of diffraction or cell-boundary discontinuities.
5. Treating a correctness-only baseline as the final runtime architecture.
6. Mesh radio maps and arbitrary mesh measurement surfaces.
