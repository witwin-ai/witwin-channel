# Radio-Map Monte Carlo Mode Plan

Status: Draft
Category: Plan
Last reviewed: 2026-04-07

## Objective

Add an explicit Monte Carlo mode to `RadioMapMonitor` that reproduces the
structural behavior of the `sionna-rt-reference-2.0.0` radio-map solver:

1. radio-map accumulation is driven by traced ray or wedge samples,
2. the dominant runtime scales with sample count rather than grid cell count,
3. contributions are accumulated directly into radio-map cells in-loop,
4. hot paths use native CUDA kernels instead of Python-side or Dr.Jit-side
   pair replay wherever production performance matters.

This plan treats radiomap execution as two peer modes:

1. `deterministic`: the current cell-driven integration family,
2. `monte_carlo`: the new sample-driven Sionna-style family.

The new mode does not replace the existing deterministic semantics. The current
`quadrature_mode` and `samples_per_cell` contract remains part of the
deterministic path.

## Why A New Mode Is Required

The current `RadioMapMonitor` stack already contains useful native cell-scatter
building blocks, but it is not yet a Sionna-style Monte Carlo solver.

The main structural mismatch is that the current radiomap path is still
receiver-grid driven:

1. `witwin/channel/monitors/radio_map/grid.py` materializes sample positions
   per cell,
2. `witwin/channel/monitors/radio_map/trace.py` iterates over
   `grid.sample_sets`,
3. the current `cell_accumulation` path still performs reflection and
   diffraction replay over `(path, receiver)` or `(state, receiver)` products in
   `witwin/channel/monitors/radio_map/cell_accumulation.py`.

This means the current production cost still depends on receiver count even
when the final accumulation target is a dense radio map. That is exactly the
behavior the Sionna-style coverage solver avoids.

## Sionna 2.0.0 Behavior To Replicate

The local reference snapshot under
`sionna-rt-reference-2.0.0/src/sionna/rt/radio_map_solvers/` shows the target
semantics clearly:

1. `radio_map_solver.py` defines the radio map as a cell-average path-gain
   quantity and states explicitly that the solver uses Monte Carlo integration.
2. The solver launches ray samples from each transmitter and traces them through
   the scene instead of evaluating every receiver cell explicitly.
3. When a path hits the measurement surface, `PlanarRadioMap.add_paths(...)`
   maps the hit to a cell index and immediately calls `dr.scatter_reduce(...)`
   into the path-gain tensor.
4. `PlanarRadioMap.finalize()` applies the global scale factor
   `((wavelength / 4pi)^2) / cell_area`.
5. Diffraction follows the same structural model: sample wedge interactions,
   trace them, and scatter surviving contributions directly into the hit cell.

The essential property is not merely "use scatter." The essential property is
that the solver never expands a dense receiver product to evaluate the whole map.

## Competitive Expectation Versus Sionna

This plan should be explicit about what "using C++/CUDA kernels" can and cannot
buy us relative to Sionna.

### Practical Expectation

The repository can plausibly beat Sionna's radiomap solver on efficiency and
memory, but only if the final implementation is a true radiomap-specific Monte
Carlo solver rather than a native rewrite of the current replay structure.

This is an engineering inference from the current code audit and local
benchmarks, not a claim that the repository already outperforms Sionna today.

### What Is Not Enough

The following is not sufficient on its own:

1. moving Python code into C++ while keeping `(path, receiver)` replay,
2. moving Python code into C++ while keeping `(state, receiver)` replay,
3. keeping one heavily shared implementation that tries to serve
   `deterministic`, `monte_carlo`, coherent, incoherent, and multiple receiver
   models through the same hot loop.

That approach may reduce overhead, but it does not change the dominant
complexity driver, and it is not enough to reliably beat a solver that is
already sample-driven.

### Where A Win Is Realistic

The realistic winning configuration is narrower:

1. `sampling_mode="monte_carlo"` only,
2. default Sionna-aligned metric:
   `combine_mode="incoherent"` with `receiver_model="matched_isotropic"`,
3. direct in-loop cell accumulation for LoS, reflection, and diffraction,
4. no path export,
5. no pair replay,
6. compact device-resident state and per-transmitter cell buffers,
7. radiomap-specific native CUDA kernels with minimal branching and no
   deterministic-mode compatibility burden inside the hot path.

Under those conditions, the repository has two plausible advantages:

1. lower memory use because large pair-product intermediates can be eliminated,
2. lower runtime because the kernels can be specialized to the exact radiomap
   contract instead of serving a more generic path-solver workflow.

### Memory Versus Speed

The probability of beating Sionna is higher for memory than for raw speed.

Reason:

1. memory improves immediately when pair products and exported path payloads are
   removed,
2. speed only improves if the kernels, launch shape, occupancy, and scheduler
   are all aligned with the Monte Carlo accumulation model.

So the plan should target:

1. memory win as the first realistic competitive milestone,
2. speed parity as the next milestone,
3. speed win only after parity is established on the target workloads.

### Where A Win Is Unlikely

The repository should not assume it will beat Sionna on every radiomap-like
workload.

It is unlikely to win if the implementation tries to preserve too much generic
behavior in the first production path, especially:

1. coherent radiomap as a co-equal primary target,
2. projected-polarized receiver accumulation as the default target,
3. tight coupling between `deterministic/` and `monte_carlo/`,
4. arbitrary monitor geometry before the axis-aligned planar path is mature,
5. full autodiff support in the first performance closure if that forces extra
   state retention or kernel indirection.

The initial production target should therefore be narrower than "beat Sionna at
everything." It should be "beat or match Sionna on the default Monte Carlo
incoherent planar radiomap contract."

## Code Audit Summary

### Existing Components We Can Reuse

1. `witwin/channel/monitors/radio_map/monitor.py` already provides the public
   `RadioMapMonitor` entrypoint and radiomap-owned options such as
   `combine_mode`, `receiver_model`, and `accumulation_backend`.
2. `witwin/channel/trace/cache.py`,
   `witwin/channel/monitors/orchestration.py`, and
   `witwin/channel/trace/executors/radio_map.py` already give radiomap-specific
   execution intent instead of routing everything through a generic
   path-export monitor flow.
3. `witwin/channel/kernels/monitors/field/reflection_grid/` already contains a
   native reflection grid-accumulation path that is structurally ray-driven and
   cell-scattering.
4. `witwin/channel/kernels/monitors/common/suffix_grid/` already contains
   native segment-to-grid accumulation logic that is a strong fit for reflected
   suffix handling.
5. `witwin/channel/monitors/radio_map/native_grid.py` already provides an
   axis-aligned grid adapter with world-space indexing utilities.

### Structural Gaps That Must Be Closed

1. `witwin/channel/monitors/radio_map/trace.py` is still organized around
   sample sets generated from the receiver grid.
2. Reflection scalar-power accumulation in
   `witwin/channel/monitors/radio_map/cell_accumulation.py` still builds
   `(path, receiver)` cartesian chunks before scattering into outputs.
3. Diffraction scalar-power accumulation in the same module still builds
   `(state, receiver)` pair products and relies on pair replay helpers such as
   `utd_accumulate_scalar_power_pairs(...)`.
4. Current radiomap scheduler logic in
   `witwin/channel/monitors/radio_map/scheduler.py` is still solving a
   receiver-target replay problem rather than a sample-budget problem.
5. Current metadata marks the native incoherent path as direct cell
   accumulation, but in practice that label still covers replay-driven
   accumulation for major parts of the workload.

The key implementation decision is therefore to introduce a distinct Monte Carlo
execution path instead of trying to reinterpret the current fixed-grid path as
equivalent.

## Target Public Contract

The public API should expose Monte Carlo radiomap behavior explicitly instead of
overloading `quadrature_mode`.

### Proposed Monitor Parameters

Add an explicit integration or sampling mode:

1. `sampling_mode="deterministic" | "monte_carlo"`,
2. `samples_per_tx` for the primary ray budget,
3. `rr_depth`, `rr_prob`, and `stop_threshold` for Russian roulette and
   low-gain path termination,
4. `seed` or equivalent deterministic sampler control,
5. optionally `diffraction_samples_per_tx` only if diffraction needs a separate
   user-visible budget; otherwise keep a single top-level sample budget and
   derive wedge sampling internally.

### Mode Semantics

For `sampling_mode="monte_carlo"`:

1. the default metric should be incoherent path gain with
   `receiver_model="matched_isotropic"`,
2. the primary output should estimate cell-average power over the measurement
   surface,
3. runtime should be dominated by traced sample count and interaction depth,
   not by the number of grid cells,
4. the final normalization should match Sionna-style path-gain scaling,
5. coherent radiomap support should be treated as a separate follow-up unless a
   strong use case requires parity in the first rollout.

For `sampling_mode="deterministic"`:

1. keep the current deterministic cell-sampling contract unchanged,
2. keep `quadrature_mode` and `samples_per_cell` as deterministic-only controls,
3. retain current backends and metadata semantics,
4. do not silently redirect existing users onto the new Monte Carlo path.

The naming matters because `deterministic` is the actual conceptual counterpart
to `monte_carlo`. The API should reflect that directly instead of exposing an
implementation detail such as `fixed_quadrature` as the top-level mode name.

## Code Organization Recommendation

The repository structure should reflect the fact that `deterministic` and
`monte_carlo` are separate execution families, not two flags inside one large
shared implementation.

Recommended layout under `witwin/channel/monitors/radio_map/`:

1. `deterministic/` for the current grid-sample-set-driven path,
2. `monte_carlo/` for the new transmitter-driven path,
3. a thin shared top level for `monitor.py`, public option parsing,
   result-shape normalization, and truly common helpers only.

The goal is low coupling between the two modes:

1. each mode owns its own `trace`, scheduler, and backend wiring,
2. each mode can evolve without preserving internal compatibility with the
   other,
3. shared code should be limited to stable concepts such as monitor parameter
   normalization, cell indexing helpers, and common result metadata assembly.

In practice, this means the new Monte Carlo mode should not be implemented as
another branch inside the current deterministic execution modules. It should be
given its own package boundary.

## Target Architecture

### Phase Split

The Monte Carlo radiomap path should be a dedicated execution family with its
own scheduler and native accumulators:

1. sample generation and bounce progression are transmitter-driven,
2. measurement-surface hits are converted directly to cell indices,
3. native kernels accumulate scalar power into final cell buffers in-loop,
4. finalization applies only global normalization and optional per-transmitter
   reduction.

### Data Flow

The target high-level flow is:

1. allocate per-transmitter cell accumulators on device,
2. spawn Monte Carlo ray samples from each transmitter,
3. trace LoS and reflection paths using a radiomap-specific sample loop,
4. whenever a path segment hits the radiomap surface, scatter its weighted
   contribution directly into the cell buffer,
5. generate diffraction samples from wedge states and scatter their
   contributions through the same cell-indexing contract,
6. finalize with global scale factors and return a standard `Result` payload.

### Result Metadata

The result payload should make the Monte Carlo contract explicit:

1. `sampling_mode`,
2. `samples_per_tx`,
3. effective random seed,
4. Russian roulette settings,
5. traced sample counts and accepted-hit counts by family,
6. normalization mode and cell area,
7. backend names for reflection and diffraction accumulation.

This metadata matters because Monte Carlo coverage maps are stochastic
estimators, not deterministic quadrature outputs.

## Native CUDA Kernel Strategy

### Reflection

Reflection is the nearest-term win because the repository already has the right
shape in `kernels/monitors/field/reflection_grid/`.

Planned direction:

1. factor the reflection-grid accumulation core into a radiomap Monte Carlo
   kernel contract that accepts traced ray segments and writes scalar path gain
   directly into radio-map cells,
2. keep the kernel Dr.Jit-native at the boundary and native CUDA inside,
3. specialize for the default `matched_isotropic` power metric first,
4. avoid rebuilding `(path, receiver)` products anywhere in this path.

The reflection Monte Carlo implementation should not be a faster version of the
current pair replay. It should be a different execution shape.

### Diffraction

Diffraction is the main blocker and needs new native work.

Planned direction:

1. introduce a diffraction Monte Carlo path that samples wedge interactions or
   diffraction states directly from the transmitter-driven trace,
2. replace `(state, receiver)` replay with a kernel that maps each valid
   diffracted hit directly to a cell index and scatters weighted scalar power,
3. keep UTD coefficient evaluation inside native code for the production path,
4. use the existing `suffix_grid` infrastructure where reflected suffix
   traversal already matches the needed segment-to-cell accumulation pattern.

This is the part that changes the asymptotic behavior. Without it, a
"Monte Carlo mode" would still inherit the current receiver-count bottleneck.

### LoS

LoS support should share the same direct-hit accumulation contract:

1. trace the direct segment,
2. intersect against the measurement surface,
3. scatter the weighted contribution into the cell buffer,
4. reuse the same normalization path as reflection and diffraction.

LoS does not justify a separate replay-based implementation in Monte Carlo mode.

### Autodiff Strategy

The first rollout can be primal-first if necessary, but the implementation
should be structured so it can grow into full CustomOp ownership instead of
locking in a non-differentiable dead end.

Practical rule:

1. no Torch, NumPy, or DLPack transport inside hot paths,
2. device pointers remain owned by C++ or Dr.Jit-native buffers,
3. if Monte Carlo radiomap becomes part of differentiable optimization
   workflows, the accumulation kernels must migrate to the repository-standard
   native CustomOp pattern rather than stay as opaque forward-only helpers.

## Implementation Plan

### Phase 0: Public API And Contract Freeze

1. Add `sampling_mode` and Monte Carlo-specific budget parameters to
   `RadioMapMonitor`.
2. Define `deterministic` as the explicit peer mode of `monte_carlo`.
3. Validate that the current `quadrature_mode` contract remains untouched for
   deterministic users.
4. Extend result metadata and monitor validation so invalid parameter
   combinations fail early.
5. Document that Monte Carlo mode targets Sionna-style path-gain estimation,
   not deterministic per-cell quadrature.
6. Document the intended `radio_map/deterministic/` and
   `radio_map/monte_carlo/` package split before code migration begins.

Exit criterion:

The monitor API can express the new mode without changing existing behavior.

### Phase 1: Radiomap Monte Carlo Execution Path

1. Add a dedicated radiomap Monte Carlo executor path instead of routing through
   `grid.sample_sets`.
2. Introduce a sample-budget-oriented scheduler that reasons about rays,
   wedges, and bounce depth rather than receiver tiles.
3. Land the execution path inside `radio_map/monte_carlo/` rather than inside
   the deterministic modules.
4. Keep final outputs in the existing `Result` shape so callers do not need a
   second output format.

Exit criterion:

The runtime can execute a transmitter-driven radiomap solve without materialized
receiver sample sets.

### Phase 2: Native Reflection Monte Carlo Accumulator

1. Reuse or refactor the existing reflection-grid native kernel family so it
   writes radiomap scalar power directly.
2. Add radiomap-specific accumulation wrappers under
   `witwin/channel/monitors/radio_map/monte_carlo/` and native kernel ownership under
   `witwin/channel/kernels/monitors/`.
3. Remove reflection dependence on `_accumulate_reflection_chunk_scalar_power`
   for Monte Carlo mode.
4. Validate scaling on large maps while keeping `samples_per_tx` fixed.

Exit criterion:

Reflection runtime in Monte Carlo mode is insensitive to dense grid resolution
apart from minor indexing and final reshape overhead.

### Phase 3: Native Diffraction Monte Carlo Accumulator

1. Add a dedicated diffraction sample representation for Monte Carlo radiomap
   accumulation.
2. Replace `_accumulate_diffraction_pairs_scalar_power` in Monte Carlo mode with
   direct native wedge-sample or state-hit accumulation.
3. Reuse `suffix_grid` for reflected suffix traversal where possible instead of
   creating a second segment-grid implementation.
4. Keep all scalar-power weighting and wedge normalization in the native path.

Exit criterion:

Diffraction runtime in Monte Carlo mode no longer scales with `n_states * n_rx`.

### Phase 4: Multi-Transmitter Native Accumulation

1. Accumulate directly into `[num_tx, n_cells]` device buffers during tracing.
2. Avoid per-transmitter Python loops that serialize the whole solve when a
   shared native launch can handle the stack.
3. Keep `trace_many(...)` aggregation compatible with the standard
   `Scene + Tracer + Result` contract.

Exit criterion:

The Monte Carlo path does not require post-trace dense TX-stack materialization
to produce final maps.

### Phase 5: Validation, Benchmarks, And Rollout Gates

1. Add parity checks against the local `sionna-rt-reference-2.0.0` snapshot for
   representative planar scenes.
2. Add scaling benchmarks that vary grid resolution while keeping
   `samples_per_tx` fixed.
3. Add complementary benchmarks that vary sample count while keeping the grid
   fixed.
4. Record acceptance gates for LoS-only, reflection-only, and diffraction
   scenes.
5. Keep Monte Carlo mode opt-in until correctness and variance behavior are
   characterized.
6. Add explicit Sionna-comparison gates for both wall-clock time and peak memory
   on the default incoherent planar Monte Carlo contract.

Exit criterion:

The repository has evidence that the new mode reproduces the intended Sionna
structure, achieves the expected scaling behavior, and has a measurable path to
competitive speed and memory on the target workload.

## Validation Matrix

The minimum acceptance matrix should include:

1. LoS-only planar map with analytic expectations,
2. single-bounce reflection map on an axis-aligned wall,
3. wedge diffraction map with known shadow transition structure,
4. mixed reflection plus diffraction scene,
5. dense-grid scaling checks such as `128 x 128`, `256 x 256`, and `512 x 512`
   at fixed sample budgets,
6. sample-budget sweeps such as `1e5`, `1e6`, and `1e7` rays on one fixed grid,
7. stability checks across multiple seeds.

The critical performance acceptance test is:

At fixed `samples_per_tx`, increasing the radio-map resolution should not
recreate the current receiver-product growth pattern in the dominant solver
wall-clock path.

The competitive acceptance ordering should be:

1. structural parity with Sionna's sample-driven accumulation,
2. correctness parity on representative scenes,
3. peak-memory parity or better on the default Monte Carlo contract,
4. runtime parity on the default Monte Carlo contract,
5. runtime win only after the first four conditions hold.

## Non-Goals

This plan does not include:

1. introducing CPU fallback implementations,
2. adding Torch-native internal transport layers,
3. redefining the deterministic radiomap path,
4. claiming coherent Monte Carlo parity in the first rollout unless explicitly
   prioritized,
5. keeping pair-replay code alive as the long-term production backend for Monte
   Carlo radiomap accumulation.

## File-Level Impact Estimate

The implementation will likely touch at least:

1. `witwin/channel/monitors/radio_map/monitor.py`,
2. a new `witwin/channel/monitors/radio_map/deterministic/` package created by
   moving or wrapping the current deterministic path,
3. a new `witwin/channel/monitors/radio_map/monte_carlo/` package for the new
   transmitter-driven path,
4. `witwin/channel/monitors/radio_map/helpers.py`,
5. `witwin/channel/monitors/radio_map/native_grid.py`,
6. new or refactored native kernels under
   `witwin/channel/kernels/monitors/`,
7. diffraction-native code under `witwin/channel/kernels/trace/utd/` and
   related radiomap-facing wrappers,
8. validation and benchmark coverage under `tests/`.

## Recommended Implementation Order

1. Freeze the API contract and metadata first.
2. Land the transmitter-driven Monte Carlo executor skeleton.
3. Replace reflection with a true native ray-to-cell scalar-power accumulator.
4. Replace diffraction pair replay with native wedge-sample-to-cell
   accumulation.
5. Add multi-transmitter native accumulation.
6. Only then consider making Monte Carlo mode the preferred path for large
   dense radiomaps.

This order minimizes the risk of shipping a mode that looks like Sionna on the
surface but still scales like the current receiver-grid solver internally.

## References

1. Local snapshot:
   `sionna-rt-reference-2.0.0/src/sionna/rt/radio_map_solvers/radio_map_solver.py`
2. Local snapshot:
   `sionna-rt-reference-2.0.0/src/sionna/rt/radio_map_solvers/planar_radio_map.py`
3. Technical report:
   <https://nvlabs.github.io/sionna/rt/tech-report/S4.html>
