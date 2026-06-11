# Channel Native Monte Carlo BDPT Migration Plan

**Goal:** Implement `witwin.channel_native.montecarlo.bdpt` as a complete native Torch/CUDA/OptiX bidirectional Monte Carlo solver for receiver-grid radiomaps and point receiver estimates, with LoS, reflection, diffraction, multiple importance sampling, fixed-seed reproducibility, diagnostics, and parity against the original `witwin.channel` BDPT behavior.

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
- `require_reflection: bool = False`
- `require_diffraction: bool = False`
- `export_paths: bool = False`
- `max_exported_paths: int | None = None`
- `ad_mode: str = "none"`
  - BDPT is primal-only for the first complete migration.
  - Unsupported modes raise tested config errors.

### Result Contract

`src/witwin/channel_native/montecarlo/bdpt/result.py` exposes:

- `path_gain: torch.Tensor`
  - Shape `(tx, rx)` for point receivers.
  - Shape `(tx, dim0, dim1)` for the first `ReceiverGrid`.
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
    - AD unsupported status

---

## Architecture

### Data Flow

```text
Scene
  -> Scene.compile()
  -> CompiledScene(GeometryStore, MaterialStore, AssignmentStore, RayDNScene)
  -> BDPT launch state
  -> transmitter/light subpath generation
  -> receiver/sensor subpath generation
  -> native connection and visibility
  -> MIS contribution evaluation
  -> component-map accumulation
  -> Result(path_gain, component maps, variance, optional path samples, metadata)
```

### Reused Components

- `Scene.compile()` for geometry/material/assignment stores and RayDN native scene handle.
- `montecarlo.basic.raydn_components.grid_spec()` for grid bounds and public layout.
- `montecarlo.basic.sampling.make_cuda_generator()` for seed policy, with BDPT-specific stream splitting.
- `core.material_runtime.face_material_tensors()` for material tensors.
- `core.kernels.ops` for Channel Native native helper calls.
- RayDN native:
  - `visibility_forward`
  - reflection/intersection kernels
  - diffraction accumulation and edge discovery primitives where compatible.

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
- Return zero maps for disabled or unavailable components.
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
- Raise explicit errors when `require_reflection=True` and native capability is unavailable.

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
- Verify native and Torch reference MIS match within `1.0e-6`.

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
- Raise explicit errors when `require_diffraction=True` and native capability is unavailable.

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
- `tests/montecarlo/bdpt/test_performance_gate.py`
- `docs/dev/perf/bdpt_native_baselines.md`

Gates:

- Empty-space LoS BDPT has no more than `1.25x` overhead versus MC basic LoS for the same grid and sample count.
- Single-plane reflection BDPT is no slower than `1.50x` original BDPT on the maintained reduced scene.
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

## Explicitly Out Of Scope

- CPU fallback solvers.
- Python RayDN wrapper fallback.
- Runtime Mitsuba/Sionna solver calls.
- DrJit dependency.
- Topology-changing gradients.
- BDPT fixed-topology AD in the first complete primal migration.
- Multi-edge diffraction before first-order diffraction parity passes.
- Arbitrary non-axis-aligned receiver grids beyond the current MC basic grid rules.
- Public tensor-only scene construction.

---

## Risk Register

1. **MIS mistakes can look like noisy MC error.**
   - Mitigation: isolate MIS kernel tests on synthetic pdfs and compare native output to Torch references.

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
