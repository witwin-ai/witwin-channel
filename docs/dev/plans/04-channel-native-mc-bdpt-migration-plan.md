# Channel Native Monte Carlo BDPT Migration Plan

**Goal:** Implement `witwin.channel_native.montecarlo.bdpt` as a complete native CUDA/OptiX bidirectional Monte Carlo solver for receiver-grid radiomaps and point receiver estimates, with Torch used only as the tensor storage/API carrier. The solver covers LoS, reflection, diffraction, multiple importance sampling, fixed-seed reproducibility, diagnostics, and parity against the original `witwin.channel` BDPT behavior.

**Non-negotiable boundary:** `witwin.channel_native.montecarlo.bdpt` must not call Python `raydn.Scene`, Python `raydn.autograd`, DrJit, Mitsuba, Sionna solver code, or the original `witwin.channel` solver during package import or solver hot paths. Original code is allowed only in parity tests and benchmark scripts.

**Position in roadmap:** BDPT starts after MC basic component maps and deterministic/path topology are stable. BDPT reuses scene compilation, RayDN native visibility/intersection, material runtime tensors, deterministic/path topology conventions, and MC basic component-map output contracts.

---

## Solver Contract

### Public API

```python
import witwin.channel_native as cn
from witwin.channel_native.montecarlo.bdpt import Config, solve

result = solve(
    scene,
    Config(
        samples=4096,
        seed=11,
        max_depth=3,
        max_diffraction_order=1,
        components={"los", "reflection", "diffraction"},
        mis="power_heuristic",
    ),
)
```

### Config Contract

`src/witwin/channel_native/montecarlo/bdpt/config.py` exposes:

- `samples: int = 4096`
  - Number of primary BDPT samples per transmitter for the public estimate.
- `seed: int = 0`
- `max_depth: int = 3`
  - Maximum total scattering depth for connected paths.
- `max_light_depth: int | None = None`
  - Defaults to `max_depth`.
- `max_sensor_depth: int | None = None`
  - Defaults to `max_depth`.
- `max_diffraction_order: int = 1`
  - Initial native migration supports `0` or `1`.
- `components: frozenset[str] = {"los", "reflection", "diffraction"}`
- `mis: str = "power_heuristic"`
  - Supported values: `"balance"`, `"power_heuristic"`, `"none"`.
- `power_heuristic_beta: float = 2.0`
- `receiver_strategy: str = "grid_area"`
  - Supported values: `"grid_area"`, `"point_sphere"`.
- `accumulation_strategy: str = "auto"`
  - Supported values: `"auto"`, `"atomic"`, `"staged"`, `"compact"`.
- `sample_streams: int = 1`
  - Number of independent RNG streams per transmitter.
- `diagnostics: bool = False`
- `export_paths: bool = False`
- `max_exported_paths: int | None = None`
- `ad_mode: str = "none"`
  - BDPT is primal-only for the first complete migration.
  - Non-`none` modes raise tested config errors.

### Result Contract

`src/witwin/channel_native/montecarlo/bdpt/result.py` exposes:

- `path_gain: torch.Tensor`
  - Shape `(tx, rx)` for point receivers.
  - Shape `(tx, cols, rows)` for the first `ReceiverGrid` whose declared
    shape is `(rows, cols)`, matching the maintained MC/RayDN grid-map layout.
- `component_power: dict[str, torch.Tensor]`
  - Keys: `los`, `reflection`, `diffraction`.
- `component_maps: dict[str, torch.Tensor] | None`
  - Present when a receiver grid exists.
- `variance: torch.Tensor | None`
  - Same public layout as `path_gain`, present when variance diagnostics are enabled.
- `path_samples: BDPTPathSamples | None`
  - Present when `export_paths=True`.
  - Includes sampled topology, contribution, pdf, MIS weight, component id, and validity.
- `metadata: dict[str, Any]`
- `diagnostics: dict[str, Any] | None`

### BDPT Semantics

- Light subpaths start at transmitters.
- Sensor subpaths start at receivers or receiver-grid samples.
- Connections are tested with RayDN native visibility.
- Reflection events use native RayDN intersection and material scattering terms.
- Diffraction events use selected edge/wedge state from Channel Native/RayDN native edge records.
- MIS weights account for the sampling strategy that generated each connected path.
- `path_gain` is an unbiased MC estimate under the configured sampling strategy.
- Fixed seed and unchanged topology produce stable counts and numerically stable estimates within floating-point accumulation tolerance.

---

## Acceptance Contract

1. `import witwin.channel_native.montecarlo.bdpt` does not import forbidden modules.
2. `bdpt.Config()` validates all numeric values, component names, MIS choices, receiver strategy, accumulation strategy, and `ad_mode`.
3. `bdpt.solve()` no longer behaves as a reserved package and requires CUDA at runtime.
4. Empty-space LoS estimate matches analytic reference within deterministic LoS tolerance when `components={"los"}`.
5. Single-reflector BDPT estimate converges to the maintained reference within:
   - relative tolerance `0.20` at `4096` samples
   - relative tolerance `0.10` at `16384` samples
6. Single-wedge diffraction estimate converges to the maintained reference within:
   - relative tolerance `0.35` at `4096` samples
   - relative tolerance `0.20` at `16384` samples
7. Fixed-seed tests verify stable metadata counts and output reproducibility within `1.0e-6` relative tolerance for maintained tiny scenes.
8. Reduced Munich BDPT parity compares original and native outputs with documented MC tolerance:
   - total path gain relative tolerance `0.35`
   - component map correlation above `0.85` for non-empty components
   - no systematic all-zero or all-NaN component map
   - current reduced synthetic gate is green in pytest; see "Current
     Implementation Status" for the latest measured values.
9. No production BDPT hot path calls Python RayDN wrappers, DrJit, Mitsuba, Sionna solver code, or original `witwin.channel`.
10. Metadata reports:
    - sample count
    - seed
    - stream count
    - MIS mode
    - path counts by strategy
    - valid contribution count
    - component status
    - native capability flags
    - launch count
    - accumulation strategy
    - variance status
    - AD status (`none` for primal BDPT)

---

## Architecture

### Data Flow

```text
Scene
  -> build_scene_from_structures()
  -> RayDN native scene handle
  -> BDPT launch state
  -> transmitter/light subpath generation
  -> receiver/sensor subpath generation
  -> _channel_native direct RayDN bridge for visibility/reflection/diffraction
  -> MIS contribution evaluation
  -> component-map accumulation
  -> Result(path_gain, component maps, variance, optional path samples, metadata)
```

### Reused Components

- `core.runtime.raydn.build_scene_from_structures()` for the RayDN native scene handle.
- BDPT-local grid-bound helpers for RayDN-native component-map layout.
- BDPT native launch-state and seeded direction kernels for seed policy and stream splitting.
- BDPT native material helpers for per-face material tensors from host scene structures.
- `core.kernels.ops` for Channel Native native helper calls.
- RayDN native:
  - exported `raydn_native_visibility_forward`
  - exported `raydn_native_reflection_accumulation_forward`
  - exported diffraction edge discovery and accumulation bridge entry points.

### New BDPT Components

- `src/witwin/channel_native/montecarlo/bdpt/config.py`
- `src/witwin/channel_native/montecarlo/bdpt/result.py`
- `src/witwin/channel_native/montecarlo/bdpt/solver.py`
- `src/witwin/channel_native/montecarlo/bdpt/metadata.py`
- `src/witwin/channel_native/montecarlo/bdpt/sampling.py`
- `src/witwin/channel_native/montecarlo/bdpt/subpaths.py`
- `src/witwin/channel_native/montecarlo/bdpt/connections.py`
- `src/witwin/channel_native/montecarlo/bdpt/mis.py`
- `native/channel_native/bdpt.cpp`
- `native/channel_native/kernels/bdpt_sampling.cu`
- `native/channel_native/kernels/bdpt_subpaths.cu`
- `native/channel_native/kernels/bdpt_connect.cu`
- `native/channel_native/kernels/bdpt_accum.cu`

---

## Internal Tensor Schema

### Subpath State

BDPT native kernels exchange subpaths as struct-of-arrays tensors:

- `origin: float32[N, 3]`
- `direction: float32[N, 3]`
- `throughput_real: float32[N]`
- `throughput_imag: float32[N]`
- `pdf_forward: float32[N]`
- `pdf_reverse: float32[N]`
- `depth: int32[N]`
- `component_mask: int32[N]`
- `primitive_id: int32[N]`
- `edge_id: int32[N]`
- `tx_id: int32[N]`
- `rx_id: int32[N]`
- `grid_linear_id: int32[N]`
- `valid: bool[N]`

### Connection Samples

Connections are emitted as:

- `tx_id: int32[M]`
- `rx_id: int32[M]`
- `grid_linear_id: int32[M]`
- `light_depth: int32[M]`
- `sensor_depth: int32[M]`
- `component_id: int32[M]`
- `path_length_m: float32[M]`
- `contribution: float32[M]`
- `mis_weight: float32[M]`
- `pdf: float32[M]`
- `valid: bool[M]`

This schema is used by accumulation, diagnostics, and optional path export.

---

## Implementation Stages

### Stage 1: Public API And Reserved Package Replacement

Create the BDPT package surface and validation tests.

Required files:

- `src/witwin/channel_native/montecarlo/bdpt/__init__.py`
- `src/witwin/channel_native/montecarlo/bdpt/config.py`
- `src/witwin/channel_native/montecarlo/bdpt/result.py`
- `src/witwin/channel_native/montecarlo/bdpt/solver.py`
- `src/witwin/channel_native/montecarlo/bdpt/metadata.py`
- `tests/montecarlo/bdpt/test_config.py`
- `tests/montecarlo/bdpt/test_import_contract.py`
- `tests/montecarlo/bdpt/test_solver_cuda_requirement.py`
- `tests/reserved_api/test_reserved_solver_imports.py`

Steps:

- Replace docstring-only package with `Config`, `Result`, `BDPTPathSamples`, and `solve`.
- Add config validation for sample counts, seed, depth, MIS, components, receiver strategy, accumulation strategy, path export limits, and AD mode.
- Add import-contract tests proving BDPT import does not import forbidden modules.
- Keep runtime CUDA requirement explicit.
- Update reserved API tests so BDPT is no longer only a docstring stub.

Acceptance command:

```powershell
conda run -n witwin2 python -m pytest tests/montecarlo/bdpt/test_config.py tests/montecarlo/bdpt/test_import_contract.py tests/montecarlo/bdpt/test_solver_cuda_requirement.py tests/reserved_api -q
```

### Stage 2: Native Launch State And RNG Streams

Build deterministic CUDA launch state for BDPT sampling.

Required files:

- `src/witwin/channel_native/montecarlo/bdpt/sampling.py`
- `src/witwin/channel_native/core/kernels/ops.py`
- `native/channel_native/bdpt.cpp`
- `native/channel_native/kernels/bdpt_sampling.cu`
- `native/channel_native/bindings.cpp`
- `tests/kernels/test_bdpt_ops_facade.py`
- `tests/montecarlo/bdpt/test_seed_stability.py`

Steps:

- Add `ops.bdpt_launch_state(...)` facade that validates CUDA tensors and scalar parameters.
- Add native kernel that maps `(tx, sample, stream)` to stable RNG state.
- Add stream splitting so light and sensor subpaths use independent random dimensions for the same seed.
- Test fixed seed stability and different seed divergence on tiny scenes.

Acceptance command:

```powershell
conda run -n witwin2 python -m pytest tests/kernels/test_bdpt_ops_facade.py tests/montecarlo/bdpt/test_seed_stability.py -q
```

### Stage 3: LoS-Only BDPT Baseline

Implement a LoS-only estimator that proves result layout, accumulation, metadata, and seed handling.

Required files:

- `src/witwin/channel_native/montecarlo/bdpt/solver.py`
- `src/witwin/channel_native/montecarlo/bdpt/connections.py`
- `src/witwin/channel_native/montecarlo/bdpt/mis.py`
- `native/channel_native/kernels/bdpt_connect.cu`
- `native/channel_native/kernels/bdpt_accum.cu`
- `tests/montecarlo/bdpt/test_los_empty_space.py`
- `tests/montecarlo/bdpt/test_component_layout.py`
- `tests/montecarlo/bdpt/test_metadata.py`

Steps:

- Build transmitter and receiver tensors using existing scene object conventions.
- Generate direct transmitter-to-receiver or transmitter-to-grid connections.
- Use RayDN visibility for LoS masking when structures exist.
- Accumulate LoS contribution into the public layout.
- Return native zero-filled storage for components that were not requested.
- Report launch, sample, valid contribution, and component metadata.

Acceptance command:

```powershell
conda run -n witwin2 python -m pytest tests/montecarlo/bdpt/test_los_empty_space.py tests/montecarlo/bdpt/test_component_layout.py tests/montecarlo/bdpt/test_metadata.py -q
```

### Stage 4: Reflection Subpaths

Generate and connect reflection subpaths using RayDN native geometry and material tensors.

Required files:

- `src/witwin/channel_native/montecarlo/bdpt/subpaths.py`
- `src/witwin/channel_native/montecarlo/bdpt/connections.py`
- `native/channel_native/kernels/bdpt_subpaths.cu`
- `native/channel_native/kernels/bdpt_connect.cu`
- `tests/montecarlo/bdpt/test_reflection_single_plane.py`
- `tests/montecarlo/bdpt/test_reflection_capability.py`

Steps:

- Generate light subpaths from transmitters with cosine/sphere sampling compatible with original BDPT semantics.
- Generate sensor subpaths from receiver grid cells or point receivers.
- Intersect subpath rays through RayDN native scene handle.
- Apply material scattering and update throughput/pdf fields.
- Connect compatible light and sensor vertices with visibility checks.
- Accumulate reflection contributions with selected MIS mode.
- Raise explicit errors whenever reflection is requested and native capability is unavailable.

Acceptance command:

```powershell
conda run -n witwin2 python -m pytest tests/montecarlo/bdpt/test_reflection_single_plane.py tests/montecarlo/bdpt/test_reflection_capability.py -q
```

### Stage 5: MIS Weighting

Implement and test multiple importance sampling independent of scene complexity.

Required files:

- `src/witwin/channel_native/montecarlo/bdpt/mis.py`
- `native/channel_native/kernels/bdpt_connect.cu`
- `tests/montecarlo/bdpt/test_mis_weights.py`
- `tests/kernels/test_bdpt_mis_kernel.py`

Steps:

- Implement balance heuristic.
- Implement power heuristic with configurable beta.
- Implement `mis="none"` as unit weight for diagnostic comparisons.
- Clamp invalid or zero pdf paths to zero contribution.
- Verify weight normalization on synthetic pdf tensors.
- Verify native MIS output against hand-computed constants within `1.0e-6`; production and tests must not keep a Torch-computed MIS alternate/reference path.

Acceptance command:

```powershell
conda run -n witwin2 python -m pytest tests/montecarlo/bdpt/test_mis_weights.py tests/kernels/test_bdpt_mis_kernel.py -q
```

### Stage 6: Diffraction Events

Add first-order diffraction support to BDPT paths.

Required files:

- `src/witwin/channel_native/montecarlo/bdpt/subpaths.py`
- `src/witwin/channel_native/montecarlo/bdpt/connections.py`
- `native/channel_native/kernels/bdpt_subpaths.cu`
- `native/channel_native/kernels/bdpt_connect.cu`
- `tests/montecarlo/bdpt/test_diffraction_single_wedge.py`
- `tests/montecarlo/bdpt/test_diffraction_capability.py`

Steps:

- Reuse Channel Native selected-edge policy and RayDN edge records.
- Generate diffraction candidate states from selected edges.
- Sample or connect paths through edge events with UTD contribution terms.
- Track `edge_id` and component id in subpath and connection tensors.
- Include diffraction pdf terms in MIS weight calculation.
- Raise explicit errors whenever diffraction is requested and native capability is unavailable.

Acceptance command:

```powershell
conda run -n witwin2 python -m pytest tests/montecarlo/bdpt/test_diffraction_single_wedge.py tests/montecarlo/bdpt/test_diffraction_capability.py -q
```

### Stage 7: Component Maps, Variance, And Path Export

Complete public outputs and diagnostics.

Required files:

- `src/witwin/channel_native/montecarlo/bdpt/result.py`
- `src/witwin/channel_native/montecarlo/bdpt/solver.py`
- `src/witwin/channel_native/montecarlo/bdpt/metadata.py`
- `native/channel_native/kernels/bdpt_accum.cu`
- `tests/montecarlo/bdpt/test_component_maps.py`
- `tests/montecarlo/bdpt/test_variance.py`
- `tests/montecarlo/bdpt/test_path_export.py`

Steps:

- Accumulate `los`, `reflection`, and `diffraction` component maps separately.
- Fuse total `path_gain` from component maps using native finalize kernels or BDPT-specific accumulation output.
- Add optional per-cell variance estimate using first and second moments.
- Add optional path sample export capped by `max_exported_paths`.
- Ensure path export order is stable for fixed seed.

Acceptance command:

```powershell
conda run -n witwin2 python -m pytest tests/montecarlo/bdpt/test_component_maps.py tests/montecarlo/bdpt/test_variance.py tests/montecarlo/bdpt/test_path_export.py -q
```

### Stage 8: Accumulation Strategy And Memory Guardrails

Make BDPT viable on dense grids and Munich-size scenes.

Required files:

- `src/witwin/channel_native/montecarlo/bdpt/solver.py`
- `src/witwin/channel_native/montecarlo/bdpt/metadata.py`
- `native/channel_native/kernels/bdpt_accum.cu`
- `tests/montecarlo/bdpt/test_accumulation_strategy.py`
- `tests/montecarlo/bdpt/test_memory_guardrails.py`
- `benchmarks/bench_bdpt_accumulation.py`

Steps:

- Implement `atomic` accumulation.
- Implement `staged` accumulation for high sample-per-cell regimes.
- Implement `compact` accumulation for sparse valid connections.
- Implement `auto` strategy selection based on samples, grid size, and estimated valid connection ratio.
- Add memory guardrails that fail early with an actionable error when requested export or accumulation buffers exceed the configured cap.
- Report selected strategy and estimated workspace bytes in metadata.

Acceptance command:

```powershell
conda run -n witwin2 python -m pytest tests/montecarlo/bdpt/test_accumulation_strategy.py tests/montecarlo/bdpt/test_memory_guardrails.py -q
```

### Stage 9: Reduced Munich BDPT Parity

Add parity and artifact generation against original BDPT behavior.

Required files:

- `tests/support/bin/benchmark_munich_bdpt_native_vs_original.py`
- `tests/montecarlo/bdpt/test_munich_bdpt_parity.py`
- `benchmarks/bench_bdpt_munich.py`

Reduced config:

- grid size `32`
- samples `4096`
- `max_depth=3`
- `max_diffraction_order=1`
- seed `11`
- frequency `2.4e9`
- tx `(8.5, 21.0, 27.0)`
- bounds `((-120, 120), (-120, 140))`
- plane z `1.5`
- MIS `power_heuristic`

Artifacts:

- original total BDPT radiomap
- native total BDPT radiomap
- native component maps
- native variance map
- native/original dB delta
- sample contribution histogram
- metadata JSON

Acceptance command:

```powershell
conda run -n witwin2 python -m pytest tests/montecarlo/bdpt/test_munich_bdpt_parity.py -q
```

### Stage 10: Performance Gates

Add maintained benchmark gates before declaring migration complete.

Required files:

- `benchmarks/bench_bdpt_basic.py`
- `benchmarks/bench_bdpt_munich.py`
- `tests/support/bin/benchmark_single_plane_bdpt_native_vs_original.py`
- `tests/montecarlo/bdpt/test_performance_gate.py`
- `docs/dev/perf/bdpt_native_baselines.md`

Gates:

- Empty-space LoS BDPT has no more than `1.25x` overhead versus MC basic LoS for the same grid and sample count.
- Single-plane open-mesh reflection BDPT is faster than original `witwin.channel` BDPT on the maintained hand-authored scene, with strict subprocess original/native gates and a maintained minimum speedup threshold.
- Munich reduced BDPT reports no all-zero component maps and completes within the documented local baseline budget.
- Native profiler metadata records launch counts and selected accumulation strategy.

Acceptance command:

```powershell
conda run -n witwin2 python -m pytest tests/montecarlo/bdpt/test_performance_gate.py -q
conda run -n witwin2 python benchmarks/bench_bdpt_basic.py --json
```

---

## Complete Migration Definition

The BDPT migration is complete when:

- `witwin.channel_native.montecarlo.bdpt` exposes `Config`, `Result`, `BDPTPathSamples`, and `solve`.
- LoS, reflection, and first-order diffraction work on CUDA.
- MIS modes `balance`, `power_heuristic`, and `none` are implemented and tested.
- Receiver grids and point receivers are supported.
- Component maps, total path gain, optional variance, metadata, and optional path sample export are implemented.
- Fixed-seed reproducibility tests pass.
- Reduced Munich BDPT parity artifacts are generated and tested.
- Performance gates pass on maintained local baselines.
- No solver hot path imports or calls forbidden dependencies.
- Unsupported AD modes fail at config validation with explicit errors.

---

## Current Implementation Status

As of the latest native iteration:

- BDPT import/config/solver surfaces are active, not reserved stubs.
- Runtime artifact-loader alternate search paths have been removed from Channel Native and RayDN extension loaders.
- `Scene.load_mitsuba()` now requires the native loader path; the Python Sionna/Mitsuba scene-construction branch has been removed.
- BDPT solver hot paths no longer call `Scene.compile()`, original `witwin.channel`, MC basic solver code, `torch.ops.raydn`, or `torch.classes.raydn`.
- RayDN visibility, ray/scene intersection, reflection accumulation, diffraction edge discovery, diffraction accumulation, reflection EPC path export, and scene create/destroy/edge-record extraction are reached through `_channel_native` direct native bridge entry points backed by exported `_raydn` C symbols.
- Channel Native runtime scene ownership now stores a native integer scene handle plus a `_channel_native` owner capsule; Python does not instantiate RayDN Torch custom classes for Channel Native scenes.
- The bridge resolves symbols from `_raydn.native_module_handle()` and does not call `LoadLibrary`/`dlopen`, so it does not implicitly load a second RayDN DSO or duplicate the RayDN scene registry.
- Deterministic scalar field, vector-field reduction, reflection field, multi-bounce reflection field, delay-to-length, phase-from-field, phase-from-length, real/imag-to-complex packing, topology padding, topology concat, face grouping, face sequence generation, reflection EPC input expansion, reflection compaction, and first-order diffraction RayDN export compaction are now `_channel_native` CUDA entry points for the migrated deterministic topology/field path.
- Torch remains the tensor storage and extension ABI carrier for these paths; migrated production code must call required native ops and raise when the native op is unavailable rather than using Python/Torch fallback computation.
- BDPT solver accumulation now uses native connection samples for LoS and reflected-light endpoint connections. Receiver-grid first-order diffraction is accumulated directly through RayDN native direct/Keller component maps instead of path-block samples.
- BDPT path export now comes from native connection samples and native connection-sample compaction. The old matrix/component-map path export facades (`bdpt_export_paths`, `bdpt_export_component_paths`, `bdpt_component_connection_samples`), legacy path-block sample/export entry points (`bdpt_sample_path_block`, `bdpt_connection_samples_from_path_block`), and derived `bdpt_variance_estimate` facade have been removed from public Python/native bindings and their native BDPT CUDA/C++ wrappers.
- Receiver-grid first-order diffraction path export is now native direct/Keller tape-backed through `_channel_native.bdpt_diffraction_connection_samples_from_tape`. Diffraction point receivers now use `_channel_native.bdpt_diffraction_point_connection_samples` to emit native point-target connection samples and two native RayDN visibility passes for source-edge and edge-receiver segments.
- Variance diagnostics use native first/second moment accumulation from connection samples (`bdpt_connection_variance`) rather than derived map buffers.
- Receiver-grid `path_gain`, component maps, and diagnostics `variance` now return the public grid layout `(tx, cols, rows)` for `ReceiverGrid.shape=(rows, cols)` through native CUDA map/finalize kernels; point receivers remain `(tx, rx)`.
- BDPT accumulation strategy selection is now passed into `_channel_native.bdpt_accumulate_connection_samples`. Explicit `atomic`, `staged`, and `compact` configurations dispatch to native CUDA accumulation variants and are tested for numerical agreement; metadata reports the concrete native kernel variant (`atomic_add`, `cell_reduce`, or `compact_atomic_add`).
- BDPT endpoint light subpath state now consumes the native launch-state `light_seed` tensor and generates per-sample unit directions in native CUDA instead of using a fixed placeholder direction. Torch remains only the tensor carrier for the seed and subpath-state buffers.
- `_raydn` now exports `raydn_native_intersect_forward`, `_channel_native.bdpt_intersect_forward` dispatches to it through the loaded RayDN module handle, and `_channel_native.bdpt_reflected_light_subpath_state` converts light subpaths plus RayDN hit geometry into depth-1 reflected light subpath state in native CUDA. Reflected-light subpaths apply native material validity/gain, preserve component masks with a reflection bit, and are connected to sensor endpoints through native endpoint-connection and RayDN visibility kernels.
- LoS estimates with structures now use BDPT endpoint connection samples plus RayDN-native visibility filtering rather than path-export helpers.
- Mixed-component LoS/reflection/diffraction connection samples now compute MIS weights per row in native CUDA from path pdf, MIS mode, beta, and topology-scoped competing strategy pdf sums. LoS, reflection, and diffraction topologies are not treated as mutual competitors; explicit competing-strategy behavior is covered at the native direct/Keller operator level. There is no Torch/reference fallback computation. BDPT LoS matrix-to-map conversion now uses the MC/RayDN `(tx, cols, rows)` native grid layout instead of the point-layout no-op path.
- Reflection order-1 receiver-grid estimates now use RayDN native reflection accumulation directly for radiomap solves. Reflection point receivers and reflection path export now use native reflected-light subpaths, native endpoint connection samples, and RayDN-native visibility filtering instead of the old path-block adapter.
- RayDN order-1 diffraction accumulation now applies the wavelength gain `(lambda / 4pi)^2` in both primal OptiX accumulation and AD accumulation kernels.
- Receiver-grid first-order diffraction now uses the original order-1 BDPTDiffractionMIS default direct/Keller sample split for `max_depth=1`: `direct=(samples+2)//3`, `keller=(samples+1)//3`, `suffix=0`. The solver calls `_channel_native.bdpt_diffraction_accumulation_forward` for component maps and `_channel_native.bdpt_diffraction_connection_samples_from_tape` for path export, and no longer imports or calls `diffraction_paths_order1` or `bdpt_sample_path_block`.
- BDPT `valid_contribution_count` now comes from `_channel_native.bdpt_count_valid_connection_samples` when connection samples exist, avoiding Torch-side tensor reductions for metadata.
- BDPT package helpers that performed Python LoS visibility loops have been removed from the package surface.
- Current smoke/regression verification covers LoS, reflection, diffraction, native scene construction, component maps, Munich reduced nonzero smoke, strict Munich native-vs-original parity, strict-loader/static dispatcher checks, reserved import checks, MIS-weighted accumulation, variance diagnostics, path export, single-reflector convergence, single-wedge point-receiver diffraction convergence, diffraction fixed-seed/direct-Keller split checks, and performance smoke.
- `tests/support/bin/benchmark_munich_bdpt_native_vs_original.py` now runs the original `witwin.channel` BDPT solver in a subprocess and records native/original timing, delta maps, component correlations, and artifacts instead of wrapping the native-only smoke benchmark. Its default synthetic-reduced mode avoids full Munich XML OptiX compile instability. The latest local strict steady-state run (`samples=16`, `grid_size=4`, `max_depth=1`, `warmup_runs=1`) measured native solve `1.901800002087839 ms`, original solve `68.98979999823496 ms`, native speedup `36.27605422362835x`, LoS correlation `1.0000000744796738`, reflection correlation `0.9999999400453701`, diffraction correlation `1.0`, diffraction relative sum error `9.127283806135762e-08`, and total relative sum error `9.127168010444401e-08`. Strict parity gates passed.
- `tests/support/bin/benchmark_single_plane_bdpt_native_vs_original.py` now runs original `witwin.channel` BDPT in a subprocess on a hand-authored open two-triangle plane scene and compares it with native Channel Native BDPT. The latest strict local smoke passed shape/nonzero/speed and relative-sum-error gates with native median `1.8860999989556149 ms`, original median `2.713899993977975 ms`, native speedup `1.4388950720962492x`, and relative sum error `2.69366456949439e-07`.

Known residual notes:

- Reflection point receivers and reflection path export no longer route through the RayDN reflection path-export adapter or BDPT path-block conversion. The current native subpath route is seeded and fully native, but it is stochastic rather than the previous deterministic image-source candidate export, so maintained export tests assert fixed-seed stability rather than seed-independent candidate identity.
- Stage 8 now has native strategy dispatch. `staged` and `compact` are maintained native variants; further dense-grid profiling and threshold tuning remain optimization follow-up work rather than a fallback or correctness dependency.
- Cold first-use native OptiX/RayDN pipeline setup can dominate the first solve. The maintained performance advantage gate is steady-state with warmup; cold-start behavior is reported separately.
- Original `witwin.channel` code remains allowed only in parity/benchmark support scripts.

---

## Explicitly Out Of Scope

- CPU solvers.
- Python RayDN wrapper dispatch.
- Runtime Mitsuba/Sionna solver calls.
- DrJit dependency.
- Topology-changing gradients.
- BDPT fixed-topology AD in the first complete primal migration.
- Multi-edge diffraction before first-order diffraction parity passes.
- Arbitrary non-axis-aligned receiver grids beyond the current MC basic grid rules.
- Public arbitrary tensor-only scene construction API beyond the internal native scene owner.

---

## Risk Register

1. **MIS mistakes can look like noisy MC error.**
   - Mitigation: isolate MIS kernel tests on synthetic pdfs and compare native output to hand-computed constants.

2. **RNG stream reuse can bias estimates.**
   - Mitigation: split light, sensor, connection, and diffraction random dimensions and test fixed-seed stability plus seed divergence.

3. **BDPT memory can explode with depth and path export.**
   - Mitigation: chunk subpath generation, cap exported paths, and add workspace estimates before allocating large buffers.

4. **Diffraction and reflection component attribution can overlap.**
   - Mitigation: carry explicit component masks in the subpath schema and test component sums against total output.

5. **Original BDPT parity may have high stochastic variance.**
   - Mitigation: use fixed reduced scenes, multiple sample levels, correlation checks, and artifact inspection rather than relying only on one scalar tolerance.

---

## Command Set

Use the `witwin2` environment:

```powershell
conda run -n witwin2 python -m pip install -e . --no-deps
conda run -n witwin2 python -m pytest tests/montecarlo/bdpt -q
conda run -n witwin2 python -m pytest tests/kernels/test_bdpt_ops_facade.py tests/kernels/test_bdpt_mis_kernel.py -q
conda run -n witwin2 python benchmarks/bench_bdpt_basic.py --json
conda run -n witwin2 python benchmarks/bench_bdpt_munich.py --json
```

Native iteration should use the same CMake and editable-install workflow as MC basic native development.
