# RadioMap Architecture Refactor Plan

Status: Draft
Category: Plan
Last reviewed: 2026-04-09

## Objective

Refactor `witwin/channel/monitors/radio_map/` so the package follows
`docs/dev/standards/python-package-architecture-standard.md` without adding a
new layer of low-value wrappers.

The practical goals are:

1. remove oversized mixed-responsibility modules,
2. delete one-line wrapper helpers and duplicate normalization logic,
3. replace cross-module "bag of dicts" payloads with a small number of typed
   dataclasses where state actually has ownership,
4. keep the public `RadioMapMonitor` and `Result` contract stable,
5. keep deterministic and Monte Carlo execution families separate without
   building "manager / builder / executor" indirection that only renames
   existing functions.

This is an internal architecture cleanup plan. It is not a solver redesign and
it must not change numerical behavior unless a follow-up bug fix explicitly says
so.

## Why This Refactor Is Needed

The current package already has a useful top-level split between
`deterministic/` and `monte_carlo/`, but the code still carries several
architecture problems that violate the active package standard.

### Current problems

1. `monitor.py` owns too many responsibilities:
   - monitor dataclass definition,
   - parameter normalization,
   - surface-mode resolution,
   - mode-compatibility validation,
   - default-value policy.

2. `common.py` is a catch-all module rather than one cohesive concept:
   - backend selection,
   - gradient-sensitivity checks,
   - diagnostics allocation,
   - diagnostics accumulation,
   - metadata assembly,
   - result-payload assembly.

3. `deterministic/trace.py` and `monte_carlo/trace.py` are oversized
   orchestration files whose main functions own too many lifecycle stages.

4. Several helpers are low-value wrappers rather than real abstractions, for
   example small `_add_*`, `_scale_*`, or re-export style helpers that only
   rename a trivial operation.

5. The package still relies on loosely structured dictionaries for internal
   state exchange:
   - weighted diagnostics,
   - runtime backend metadata,
   - runtime reuse counters,
   - batch shape,
   - diffraction state pool summaries.

6. Normalization and backend naming logic is partially duplicated across
   modules.

7. Many call sites flatten ownership with long `from ..common import ...`
   imports instead of keeping module ownership visible.

### What This Refactor Should Not Do

This refactor must not "solve" those problems by adding more abstraction layers
that simply wrap the same logic.

Specifically, do not:

1. add `Manager`, `Builder`, `Facade`, or `Orchestrator` classes that only
   forward calls into existing functions,
2. add thin wrapper methods whose only job is renaming a function call,
3. add new `helpers.py` or `common.py` style sink modules,
4. create re-export shim packages whose only content is `from .x import y`,
5. move complexity without reducing it.

The preferred shape is:

1. thin public entry functions,
2. small cohesive modules,
3. a small number of typed state objects only where repeated state ownership is
   real,
4. direct function calls at the call site when a wrapper would add no
   invariant.

## Scope

This plan applies to:

- `witwin/channel/monitors/radio_map/monitor.py`
- `witwin/channel/monitors/radio_map/common.py`
- `witwin/channel/monitors/radio_map/grid.py`
- `witwin/channel/monitors/radio_map/result.py`
- `witwin/channel/monitors/radio_map/deterministic/`
- `witwin/channel/monitors/radio_map/monte_carlo/`

This plan does not include:

1. changing the public `Scene + Tracer + Result` architecture,
2. changing solver math,
3. introducing CPU fallback paths,
4. rewriting native CUDA kernels as part of the first pass,
5. changing user-visible `RadioMapMonitor` semantics unless required for a
   correctness fix in a separate change.

## Refactor Principles

### 1. Keep Public API Stable

The following public entrypoints remain stable:

- `RadioMapMonitor`
- radio-map tracing through the tracer
- `RadioMapResult`
- current result dictionary keys and metadata structure unless a compatibility
  note is explicitly documented

### 2. Prefer Module Ownership Over Wrapper Objects

If a concept can be made clear by placing logic in the right module, prefer
that over introducing a class.

Examples:

- backend resolution belongs in `backend.py`, not in `RadioMapBackendResolver`
- metadata assembly belongs in `metadata.py`, not in `RadioMapMetadataBuilder`
- diagnostics accumulation belongs in `diagnostics.py`, not in
  `DiagnosticsAccumulator` unless stateful reuse is genuinely needed

### 3. Use Typed Objects Only For Real Shared State

Dataclasses are allowed when they replace repeated dict payloads with stable
ownership, for example:

- `BackendResolution`
- `RuntimeBackends`
- `RuntimeReuse`
- `PathCounts`
- `MonteCarloBatchPlan`
- `DiffractionStatePoolSummary`

Do not introduce dataclasses that merely rename two or three local variables
used in one short function.

### 4. Delete Trivial Wrappers

Functions that only rename addition, scaling, or a direct return should be
inlined or removed unless they protect a real DrJit invariant.

### 5. Keep Deterministic And Monte Carlo Separated

The package already distinguishes `deterministic/` and `monte_carlo/`. Keep
that split. Do not merge them back into a single configurable mega-module.

## Target Package Layout

The target layout is:

```text
witwin/channel/monitors/radio_map/
  __init__.py
  monitor.py
  validation.py
  backend.py
  diagnostics.py
  metadata.py
  result_builder.py
  grid.py
  result.py
  types.py
  deterministic/
    trace.py
    samples.py
    scheduler.py
    native_coherent.py
    cell_accumulation.py
  monte_carlo/
    trace.py
    batch_plan.py
    specular.py
    diffraction.py
    fixed_cell_ad.py
```

### Intended ownership

- `monitor.py`: public dataclass and simple monitor-facing convenience methods
  only
- `validation.py`: normalization and compatibility checks for monitor
  parameters
- `backend.py`: accumulation-backend resolution and support checks
- `diagnostics.py`: diagnostics initialization, accumulation, and finalization
- `metadata.py`: metadata construction
- `result_builder.py`: final result dictionary assembly
- `types.py`: shared typed payloads used across modules
- `deterministic/trace.py`: deterministic trace entrypoint only
- `monte_carlo/trace.py`: Monte Carlo trace entrypoint only
- `monte_carlo/fixed_cell_ad.py`: the fixed-cell AD fallback path that is
  currently buried inside the main Monte Carlo trace file

## File-Level Change Plan

### A. `monitor.py`

#### Keep

- `RadioMapMonitor`
- `resolve_radio_map_monitor`
- simple monitor-local derived properties such as `tangential_axes`,
  `resolve_grid_shape`, and `resolve_cell_size` if they remain concise

#### Move out

- `_normalize_metric`
- `_normalize_combine_mode`
- `_normalize_receiver_model`
- `_normalize_accumulation_backend`
- `_normalize_shadow_boundary_mode`
- `_normalize_shadow_support_cutoff_db`
- `_normalize_positive_power`
- `_normalize_sampling_mode`
- `_normalize_positive_int`
- `_normalize_nonnegative_int`
- `_normalize_probability`
- `_normalize_nonnegative_threshold`
- `_normalize_seed`
- `_normalize_point2`
- `_normalize_point3`
- `_normalize_optional_point2`
- `_normalize_quadrature_mode`
- most of the branching currently inside `__post_init__`

#### End state

`__post_init__` should become a short coordination step:

1. normalize the monitor through `validation.normalize_monitor_fields(...)`,
2. assign the normalized values,
3. run one compatibility check for mutually dependent options.

It should not contain a long inline policy engine.

### B. `common.py`

`common.py` should be deleted after its content is redistributed.

#### Move to `backend.py`

- `_normalize_radio_map_accumulation_backend`
- `_radio_map_native_coherent_supported`
- `_radio_map_grad_sensitive_workload`
- `_radio_map_cell_accumulation_supported`
- `_radio_map_native_monte_carlo_supported`
- `_resolve_radio_map_accumulation_backend`
- helper functions used only by backend support checks

#### Move to `diagnostics.py`

- `_empty_radio_map_diagnostics`
- `_ensure_utd_shadow_boundary_diagnostics`
- `_accumulate_complex_by_rx`
- `_accumulate_power_by_rx`
- `_finalize_radio_map_component_totals`
- `_finalize_utd_shadow_boundary_surrogate_total`
- `_finalize_projected_isb_completion_total`
- `_finalize_matched_isb_completion_total`
- `_accumulate_sample_diagnostics`
- `_baseline_los_power`

#### Move to `metadata.py`

- `_resolve_noise_power`
- `_count_reflection_paths`
- `_count_nonzero_complex`
- `_radio_map_diffraction_state_layout`
- `_radio_map_diffraction_cache_key`
- `_build_radio_map_metadata`

#### Move to `result_builder.py`

- `_build_radio_map_result_payload`

#### Delete or inline

- `_add_complex`
- `_scale_complex`
- `_add_complex_vector`
- `_scale_complex_vector`
- `_add_float`
- `_scale_float`

These should survive only if a concrete DrJit type invariant exists and the
invariant is documented. Otherwise, inline them at the call sites.

#### Evaluate carefully

- `_scatter_float`
- `_gather_positions`
- `_raw_path_count`
- `_remap_raw_rx_index`
- `_zero_float`
- `_vector_power`

Keep these only if they still provide one canonical low-level operation used by
multiple modules. If a helper ends up used in only one file after the split,
inline it there.

### C. `deterministic/trace.py`

#### Keep

- a single thin deterministic trace entrypoint

#### Remove from this file

- metadata assembly details
- diagnostics implementation details
- backend support rules
- any code that belongs to sample-specific helper modules

#### End state

The file should read roughly as:

1. build grid,
2. resolve backend,
3. iterate sample sets,
4. call sample-specific helpers,
5. finalize diagnostics,
6. build metadata,
7. build result payload.

If a helper belongs to one of those concepts, the helper must live in the
module that owns that concept, not in a shared junk drawer.

### D. `monte_carlo/trace.py`

#### Keep

- `trace_monte_carlo_radio_map_monitor`

#### Move out

- fixed-cell AD fallback into `fixed_cell_ad.py`
- batch-shape and sample-budget logic into `batch_plan.py`
- Monte Carlo metadata assembly into `metadata.py` or a small Monte Carlo local
  helper module if it is truly mode-specific

#### End state

The Monte Carlo entrypoint should be a readable mode coordinator:

1. resolve backend,
2. choose native Monte Carlo or fixed-cell AD fallback,
3. prepare common scene / monitor / grid inputs,
4. call the chosen path,
5. finalize through shared metadata and result builders.

The file should no longer own every implementation detail for every Monte Carlo
sub-mode.

### E. `deterministic/__init__.py` and `monte_carlo/__init__.py`

Delete these thin re-export shims unless they are required by an actual import
contract. Callers should import from the canonical module directly.

If external imports already rely on these paths, keep them temporarily, but
mark them as compatibility-only and remove them in a dedicated cleanup after all
internal imports are updated.

### F. `types.py`

Add a small set of dataclasses only for cross-module state that is currently
passed as loosely typed dicts.

Initial candidate types:

```python
@dataclass(slots=True)
class BackendResolution:
    requested: str
    resolved: str
    cell_accumulation_mode: str


@dataclass(slots=True)
class RuntimeBackends:
    reflection: dict[str, object]
    diffraction: dict[str, object]
    suffix: dict[str, object]


@dataclass(slots=True)
class RuntimeReuse:
    cache_mode: str
    state_preparation_hits: int
    state_preparation_misses: int
    state_layout: str


@dataclass(slots=True)
class MonteCarloBatchPlan:
    sample_batch_size: int
    sample_batch_count: int
    target_batch_size: int
    target_batch_count: int


@dataclass(slots=True)
class DiffractionStatePoolSummary:
    total: int
    kept: int
```

This list is intentionally small. Add more only when repeated cross-module dict
payloads remain after the first split.

## Execution Phases

### Phase 0: Freeze The Refactor Boundaries

Tasks:

1. document this plan,
2. identify which imports depend on the thin `__init__.py` shims,
3. record the current behavior with targeted tests before moving code.

Exit criteria:

1. the refactor scope is written down,
2. no public API change is intended,
3. a baseline test list exists.

### Phase 1: Split `common.py`

Tasks:

1. create `backend.py`, `diagnostics.py`, `metadata.py`, and
   `result_builder.py`,
2. move functions without changing behavior,
3. replace flattened imports with module-qualified imports,
4. delete duplicate accumulation-backend normalization logic,
5. remove trivial wrappers that are no longer justified after the move.

Exit criteria:

1. `common.py` is gone or reduced to zero repository-owned logic pending final
   deletion,
2. deterministic and Monte Carlo trace modules import concept-owned modules
   instead of a giant shared helper list,
3. tests still pass with no behavior change.

### Phase 2: Simplify `monitor.py`

Tasks:

1. move normalization helpers into `validation.py`,
2. keep `RadioMapMonitor` as the public dataclass,
3. shrink `__post_init__`,
4. centralize all mode compatibility checks in one validation module,
5. remove duplicated normalization logic that also exists elsewhere.

Exit criteria:

1. `monitor.py` becomes primarily the public monitor definition,
2. `__post_init__` is short enough to read without scrolling through policy
   branches,
3. the monitor contract remains unchanged.

### Phase 3: Trim Trace Entrypoints

Tasks:

1. move fixed-cell AD code from `monte_carlo/trace.py` to
   `monte_carlo/fixed_cell_ad.py`,
2. move batch-planning logic to `monte_carlo/batch_plan.py`,
3. simplify deterministic and Monte Carlo trace entrypoints so they only
   coordinate lifecycle stages,
4. keep direct function calls rather than introducing forwarding classes.

Exit criteria:

1. each trace file is readable as an execution flow,
2. mode-specific subpaths are housed in mode-owned modules,
3. no thin wrapper classes were added.

### Phase 4: Replace Internal Dict Payloads Where It Matters

Tasks:

1. replace repeated cross-module dict payloads with a small number of
   dataclasses from `types.py`,
2. keep the public result payload in the existing dict-based result contract,
3. avoid over-typing short local scopes.

Exit criteria:

1. backend and runtime summary data has explicit field ownership,
2. internal call signatures are clearer,
3. no explosion of micro-dataclasses was introduced.

### Phase 5: Cleanup Pass

Tasks:

1. delete stale compatibility imports if they are no longer needed,
2. remove dead helpers,
3. ensure all remaining functions have explicit return types,
4. ensure every remaining function has a short purpose comment or one-line
   docstring,
5. make imports module-qualified where ownership matters.

Exit criteria:

1. there are no dead alias helpers,
2. there are no giant flattened helper imports from one shared file,
3. package ownership is easy to read at the call site.

## Validation Workflow

Follow `docs/dev/standards/test-and-acceptance-workflow.md` and prefer targeted
validation first.

Recommended command order after each phase:

```bash
conda activate witwin2
cd channel
python -m pytest tests/scene/test_radio_map_monitors.py
python -m pytest tests/main/test_radiomap_main.py --gpu
python -m pytest tests/main/test_radiomap_native_wall_main.py --gpu
python -m pytest tests/main/test_radiomap_gradients_three_cubes_main.py --gpu
python -m pytest tests/main/test_radiomap_monte_carlo_gradients_three_cubes_main.py --gpu
```

If a phase changes only monitor normalization or metadata shape handling, start
with the scene-level radio-map monitor tests before running the GPU-heavy
benchmarks.

Run benchmark and comparison scripts only after behavior-preserving phases are
stable:

```bash
conda activate witwin2
cd channel
python -m tests.support.bin.benchmark_radio_map_monitor --json --strict-gates
python -m tests.support.bin.compare_radiomap_sionna_three_cubes
```

## Acceptance Criteria

The refactor is complete when:

1. `common.py` no longer exists as a mixed-responsibility sink module,
2. `monitor.py` is primarily a public dataclass instead of a policy dump,
3. deterministic and Monte Carlo trace entrypoints are both substantially
   smaller and easier to navigate,
4. low-value wrappers were deleted instead of moved,
5. internal ownership is visible from imports and file names,
6. no public `RadioMapMonitor` / `RadioMapResult` contract changed without an
   explicit compatibility note,
7. targeted radio-map tests and the maintained GPU validations pass.

## Non-Goals

This plan is not intended to:

1. rewrite the radio-map solver architecture around new class hierarchies,
2. introduce a generic monitor framework under `radio_map/`,
3. rewrite DrJit math into alternate tensor transports,
4. merge deterministic and Monte Carlo into one large configurable engine,
5. use "clean architecture" indirection where a direct module split is enough.

## Recommended Change Order For Actual Implementation

The preferred implementation order is:

1. land this plan and the docs index update,
2. split `common.py`,
3. simplify `monitor.py`,
4. extract Monte Carlo fixed-cell AD and batch-plan logic,
5. trim the deterministic trace module,
6. replace only the highest-value internal dict payloads with dataclasses,
7. remove temporary compatibility imports and dead code.

This order keeps the first changes mechanical and low-risk, and it avoids
starting with the most entangled Monte Carlo trace code before shared ownership
boundaries are fixed.
