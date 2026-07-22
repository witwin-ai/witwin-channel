# Channel Native Deterministic Solver Migration Plan

**Goal:** Implement `witwin.channel_native.deterministic` as a complete native Torch/CUDA/OptiX deterministic RF solver for LoS, specular reflection, and first-order diffraction, using the existing Channel Native scene/runtime model and vendored RayDN native backend.

**Non-negotiable boundary:** `witwin.channel_native.deterministic` must not call Python `raydn.Scene`, Python `raydn.autograd`, DrJit, Mitsuba, Sionna solver code, or the original `witwin.channel` solver during package import or solver hot paths. The original solver may be used only in parity tests and benchmark scripts.

**Position in roadmap:** This plan starts after MC basic native parity and the path solver are stable. It consumes the path solver's topology export, but the deterministic solver owns deterministic field evaluation, coherent accumulation, grid output, metadata, diagnostics, and parity gates.

---

## Solver Contract

### Public API

The package replaces the reserved shell:

```python
import witwin.channel_native as cn
from witwin.channel_native.deterministic import Config, solve

result = solve(
    scene,
    Config(
        max_depth=2,
        max_diffraction_order=1,
        components={"los", "reflection", "diffraction"},
        coherent=True,
        export_paths=False,
    ),
)
```

### Config Contract

`src/witwin/channel_native/deterministic/config.py` exposes:

- `max_depth: int = 1`
  - `0` means LoS only.
  - `1` means LoS plus first-order reflection/diffraction.
  - `2+` enables deterministic multi-bounce reflection after the first release gate.
- `max_diffraction_order: int = 1`
  - Initial native migration supports `0` or `1`.
  - Values above `1` raise an explicit `RuntimeError` until the multi-edge diffraction phase lands.
- `components: frozenset[str] = {"los", "reflection", "diffraction"}`
- `coherent: bool = True`
  - `True` accumulates complex field before converting to power.
  - `False` accumulates per-path power incoherently for MC-style comparison.
- `return_field: bool = True`
  - When true, return complex aggregate and component fields.
  - When false, return zero-sized complex tensors while still returning power outputs.
- `export_paths: bool = False`
  - When true, result includes a path table with the same column semantics as `witwin.channel_native.path.Result`, plus deterministic field columns.
- `max_paths: int | None = None`
- `sort_key: str = "receiver_transmitter_depth_component"`
- `diagnostics: bool = False`
- `require_reflection: bool = False`
- `require_diffraction: bool = False`
- `ad_mode: str = "none"`
  - Initial deterministic migration supports primal only.
  - Fixed-topology VJP/JVP is a late phase and must be explicit before it is enabled.

### Result Contract

`src/witwin/channel_native/deterministic/result.py` exposes:

- `path_gain: torch.Tensor`
  - Shape `(tx, rx)` for point receivers.
  - Shape `(tx, dim0, dim1)` for the first `ReceiverGrid`.
  - Values are received power in linear watts under the configured coherent/incoherent accumulation contract.
- `field: torch.Tensor`
  - Complex64 tensor with the same layout as `path_gain`.
  - Zero when `coherent=False` and `return_field=False`.
- `component_power: dict[str, torch.Tensor]`
  - Keys: `los`, `reflection`, `diffraction`.
  - Values match the aggregate public layout.
- `component_fields: dict[str, torch.Tensor]`
  - Complex64 component fields for coherent debugging and parity.
- `paths: PathTable | None`
  - Present when `export_paths=True`.
  - Includes topology columns plus `field_real`, `field_imag`, `phase_rad`, `interaction_count`, and `valid`.
- `metadata: dict[str, Any]`
- `diagnostics: dict[str, Any] | None`

### Output Semantics

- LoS uses visibility-aware direct propagation.
- Reflection uses deterministic specular image/path export from RayDN native reflection support.
- Diffraction uses selected wedge edges from the Channel Native edge policy and RayDN native UTD diffraction primitives.
- `path_gain == abs(field) ** 2` when `coherent=True`.
- `path_gain == sum(component_power.values())` when `coherent=False`.
- Component maps use the existing public `(tx, dim0, dim1)` layout used by MC basic.
- Deterministic path order is stable for fixed scene topology and config.

---

## Acceptance Contract

1. `import witwin.channel_native.deterministic` does not import `drjit`, `mitsuba`, `sionna`, Python `raydn`, or original `witwin.channel`.
2. `solve(scene, Config(...))` raises a CUDA requirement error when CUDA is unavailable.
3. Reserved `NotImplementedError` behavior is removed and replaced by validated config, result dataclasses, and a native solver path.
4. Empty-space LoS matches the analytic Torch reference within:
   - relative tolerance `1.0e-5`
   - absolute tolerance `1.0e-8`
5. Single-plane reflection matches the maintained deterministic reference scene within:
   - path count exact match
   - path length relative tolerance `1.0e-5`
   - power relative tolerance `5.0e-4`
6. Single-wedge diffraction matches the maintained deterministic reference scene within:
   - selected edge id exact match
   - power relative tolerance `5.0e-3`
7. Reduced Munich deterministic parity emits JSON and PNG artifacts comparing original and native outputs:
   - LoS, reflection, diffraction, total
   - dB delta maps
   - path count histogram
8. Solver hot paths call native code only through `witwin.channel_native.core.kernels.ops`, `torch.ops.raydn.*`, `torch.classes.raydn.Scene`, or narrow `_channel_native` glue.
9. No production deterministic test uses a CPU fallback solver.
10. Metadata reports:
    - native capability flags
    - component status
    - path counts
    - coherent mode
    - launch counts
    - accumulation strategy
    - unsupported AD status until fixed-topology AD lands

---

## Architecture

### Data Flow

```text
Scene
  -> Scene.compile()
  -> CompiledScene(GeometryStore, MaterialStore, AssignmentStore, RayDNScene)
  -> deterministic.path_topology export
  -> deterministic.field evaluation
  -> deterministic.accumulation
  -> Result(path_gain, field, components, optional paths, metadata)
```

### Reused Components

- `Scene.compile()` supplies geometry/material/assignment stores and RayDN native scene handle.
- `path.solve()` and `path.raydn_export` supply existing LoS/reflection/diffraction path topology patterns.
- `montecarlo.basic.raydn_components.grid_spec()` supplies receiver-grid layout rules.
- `core.material_runtime.face_material_tensors()` supplies material tensors.
- `core.kernels.ops` remains the Python facade for Channel Native kernels.

### New Deterministic Components

- `deterministic/config.py`
  - Public configuration and validation.
- `deterministic/result.py`
  - Public result dataclasses and path table dataclass.
- `deterministic/solver.py`
  - Orchestration, capability checks, receiver layout, metadata.
- `deterministic/topology.py`
  - Deterministic topology export by component.
- `deterministic/field.py`
  - Native field evaluation wrappers.
- `deterministic/accumulation.py`
  - Coherent and incoherent accumulation into point/grid layouts.
- `deterministic/metadata.py`
  - Solver metadata builder.
- `native/channel_native/deterministic.cpp`
  - PyTorch extension bindings for deterministic field/accumulation glue.
- `native/channel_native/kernels/deterministic_field.cu`
  - LoS, reflection, diffraction field evaluation helpers.
- `native/channel_native/kernels/deterministic_accum.cu`
  - Stable accumulation and optional path-table compaction helpers.

---

## Implementation Stages

### Stage 1: Public API And Contract Tests

Replace the reserved deterministic shell with validated config/result classes while keeping the solver capability-gated.

Required files:

- `src/witwin/channel_native/deterministic/__init__.py`
- `src/witwin/channel_native/deterministic/config.py`
- `src/witwin/channel_native/deterministic/result.py`
- `src/witwin/channel_native/deterministic/solver.py`
- `tests/deterministic/test_config.py`
- `tests/deterministic/test_import_contract.py`
- `tests/deterministic/test_solver_cuda_requirement.py`
- `tests/reserved_api/test_reserved_solver_errors.py`

Steps:

- Add config validation for depth, diffraction order, components, coherent mode, `return_field`, `max_paths`, and `ad_mode`.
- Add `Result` and `PathTable` dataclasses.
- Update reserved API tests so deterministic is no longer expected to raise `NotImplementedError`.
- Add import-contract tests proving deterministic import does not import forbidden modules.
- Keep `solve()` raising CUDA requirement on non-CUDA hosts and capability-specific errors when native pieces are missing.

Acceptance command:

```powershell
conda run -n witwin2 python -m pytest tests/deterministic/test_config.py tests/deterministic/test_import_contract.py tests/deterministic/test_solver_cuda_requirement.py tests/reserved_api -q
```

### Stage 2: Receiver Layout And LoS Deterministic Solver

Implement LoS field and accumulation for point receivers and one receiver grid.

Required files:

- `src/witwin/channel_native/deterministic/solver.py`
- `src/witwin/channel_native/deterministic/field.py`
- `src/witwin/channel_native/deterministic/accumulation.py`
- `src/witwin/channel_native/core/kernels/ops.py`
- `native/channel_native/deterministic.cpp`
- `native/channel_native/kernels/deterministic_field.cu`
- `native/channel_native/kernels/deterministic_accum.cu`
- `tests/deterministic/test_los_empty_space.py`
- `tests/deterministic/test_component_layout.py`

Steps:

- Reuse `path_los_export()` for LoS topology.
- Add a native field kernel that computes complex free-space field from path length, transmitter power, and wavelength.
- Add visibility masking through RayDN native `visibility_forward` when structures exist.
- Add point receiver accumulation into `(tx, rx)`.
- Add receiver-grid accumulation into `(tx, dim0, dim1)` using existing grid coordinate ordering.
- Add tests for analytic empty-space path gain and component layout.

Acceptance command:

```powershell
conda run -n witwin2 python -m pytest tests/deterministic/test_los_empty_space.py tests/deterministic/test_component_layout.py -q
```

### Stage 3: Deterministic Path Topology Integration

Create a deterministic topology layer that normalizes LoS, reflection, and diffraction path records before field evaluation.

Required files:

- `src/witwin/channel_native/deterministic/topology.py`
- `src/witwin/channel_native/path/raydn_export.py`
- `src/witwin/channel_native/deterministic/solver.py`
- `tests/deterministic/test_topology_contract.py`
- `tests/deterministic/test_path_export_contract.py`

Steps:

- Define an internal topology batch with columns:
  - `valid`
  - `tx_id`
  - `rx_id`
  - `depth`
  - `component_id`
  - `primitive_id`
  - `edge_id`
  - `path_length_m`
  - `interaction_position`
  - `interaction_normal`
  - `material_id`
- Wrap LoS export into this topology schema.
- Wrap current first-order reflection export into this topology schema.
- Wrap current first-order diffraction export into this topology schema.
- Keep sorting identical to path solver sorting for stable parity.
- Apply `max_paths` after stable sorting.

Acceptance command:

```powershell
conda run -n witwin2 python -m pytest tests/deterministic/test_topology_contract.py tests/deterministic/test_path_export_contract.py -q
```

### Stage 4: Specular Reflection Field Evaluation

Evaluate deterministic reflection fields over exported reflection paths.

Required files:

- `src/witwin/channel_native/deterministic/field.py`
- `src/witwin/channel_native/core/kernels/ops.py`
- `native/channel_native/kernels/deterministic_field.cu`
- `native/channel_native/bindings.cpp`
- `tests/deterministic/test_reflection_single_plane.py`
- `tests/deterministic/test_component_power.py`

Steps:

- Convert material store values into per-path reflection coefficients through native kernels.
- Compute phase from wavelength and path length.
- Multiply free-space field by reflection coefficient and transmitter polarization gain.
- Accumulate reflection fields into component fields.
- Test a single known reflector scene against a maintained Torch reference under CUDA.

Acceptance command:

```powershell
conda run -n witwin2 python -m pytest tests/deterministic/test_reflection_single_plane.py tests/deterministic/test_component_power.py -q
```

### Stage 5: First-Order Diffraction Field Evaluation

Evaluate deterministic UTD diffraction fields over selected edges.

Required files:

- `src/witwin/channel_native/deterministic/topology.py`
- `src/witwin/channel_native/deterministic/field.py`
- `src/witwin/channel_native/core/kernels/ops.py`
- `native/channel_native/kernels/deterministic_field.cu`
- `tests/deterministic/test_diffraction_single_wedge.py`
- `tests/deterministic/test_diffraction_edge_policy.py`

Steps:

- Reuse Channel Native selected edge policy and RayDN edge records.
- Export first-order edge paths with stable `edge_id`.
- Add native diffraction field evaluation for wedge angle, source/receiver geometry, wavelength, and material terms.
- Accumulate diffraction component field and power.
- Validate edge selection and field value on a maintained single-wedge scene.

Acceptance command:

```powershell
conda run -n witwin2 python -m pytest tests/deterministic/test_diffraction_single_wedge.py tests/deterministic/test_diffraction_edge_policy.py -q
```

### Stage 6: Grid Radiomap And Diagnostics

Make deterministic dense radiomap output a first-class path, not a point-receiver afterthought.

Required files:

- `src/witwin/channel_native/deterministic/solver.py`
- `src/witwin/channel_native/deterministic/accumulation.py`
- `src/witwin/channel_native/deterministic/metadata.py`
- `tests/deterministic/test_grid_radiomap.py`
- `tests/deterministic/test_metadata.py`
- `benchmarks/bench_deterministic_radiomap.py`

Steps:

- Support one primary `ReceiverGrid` with public `(tx, dim0, dim1)` layout.
- Support mixed point receivers only when `export_paths=True`; aggregate dense output remains tied to the first grid for consistency with MC basic.
- Add diagnostics for path counts per component, visibility rejection, selected edge count, native launch count, and accumulation mode.
- Add benchmark output with JSON fields compatible with existing benchmark scripts.

Acceptance command:

```powershell
conda run -n witwin2 python -m pytest tests/deterministic/test_grid_radiomap.py tests/deterministic/test_metadata.py -q
```

### Stage 7: Multi-Bounce Reflection

Extend reflection support beyond first order using RayDN native path export and stable compaction.

Required files:

- `src/witwin/channel_native/deterministic/topology.py`
- `src/witwin/channel_native/deterministic/field.py`
- `native/channel_native/kernels/deterministic_field.cu`
- `tests/deterministic/test_reflection_multibounce.py`
- `benchmarks/bench_deterministic_multibounce.py`

Steps:

- Add reflection topology export for depths `2..max_depth`.
- Preserve stable order by receiver, transmitter, depth, component, primitive sequence.
- Carry one material id per reflection interaction.
- Evaluate product of per-bounce coefficients.
- Add memory guardrails for `max_paths`.

Acceptance command:

```powershell
conda run -n witwin2 python -m pytest tests/deterministic/test_reflection_multibounce.py -q
```

### Stage 8: Munich Parity

Add reduced Munich deterministic parity against the original solver.

Required files:

- `tests/support/bin/benchmark_munich_deterministic_native_vs_original.py`
- `tests/deterministic/test_munich_deterministic_parity.py`
- `benchmarks/bench_deterministic_munich.py`

Reduced config:

- grid size `32`
- `max_depth=2`
- `max_diffraction_order=1`
- frequency `2.4e9`
- tx `(8.5, 21.0, 27.0)`
- bounds `((-120, 120), (-120, 140))`
- plane z `1.5`
- components `{"los", "reflection", "diffraction"}`

Artifacts:

- original total deterministic radiomap
- native total deterministic radiomap
- native component maps
- native/original dB delta
- path count comparison
- metadata JSON

Acceptance command:

```powershell
conda run -n witwin2 python -m pytest tests/deterministic/test_munich_deterministic_parity.py -q
```

### Stage 9: Fixed-Topology AD

Add differentiability only after the primal deterministic solver is stable.

Required files:

- `src/witwin/channel_native/deterministic/field.py`
- `src/witwin/channel_native/deterministic/solver.py`
- `native/channel_native/kernels/deterministic_field.cu`
- `tests/deterministic/test_fixed_topology_ad.py`
- `tests/deterministic/test_ad_errors.py`

Scope:

- Supported differentiable inputs:
  - transmitter position
  - transmitter power
  - receiver point position
  - material scalar parameters already represented in `MaterialStore`
- Excluded gradients:
  - topology selection
  - visibility decisions
  - primitive id
  - edge id
  - path count
  - compaction order

Acceptance command:

```powershell
conda run -n witwin2 python -m pytest tests/deterministic/test_fixed_topology_ad.py tests/deterministic/test_ad_errors.py -q
```

---

## Complete Migration Definition

The deterministic solver migration is complete when:

- `witwin.channel_native.deterministic.solve()` no longer raises reserved-phase errors.
- LoS, reflection, and first-order diffraction work on CUDA for point receivers and a receiver grid.
- Multi-bounce reflection works through `max_depth`.
- Reduced Munich parity artifacts are generated and checked in CI-compatible tests.
- All native hot paths avoid Python RayDN wrappers, DrJit, Mitsuba, Sionna solver code, and original `witwin.channel`.
- Metadata exposes enough detail to reproduce capability, component, launch, and path-count decisions.
- Performance benchmarks show native deterministic execution is not slower than the original solver on maintained reduced scenes.
- Fixed-topology AD is either implemented for the scoped differentiable inputs above or explicitly rejected at config validation with tested errors.

---

## Explicitly Out Of Scope

- CPU fallback solvers.
- Python RayDN wrapper fallback.
- Runtime Mitsuba/Sionna solver calls.
- Topology-changing gradients.
- Multi-edge diffraction before first-order diffraction parity passes.
- Arbitrary receiver-grid orientations beyond the axis-aligned layout already accepted by MC basic.
- Public tensor-only scene construction.

---

## Risk Register

1. **Path topology and deterministic field semantics can drift.**
   - Mitigation: topology schema tests compare deterministic topology with path solver output before field evaluation.

2. **Coherent accumulation can expose phase convention mismatches.**
   - Mitigation: maintain both complex field parity and power parity tests; report phase convention in metadata.

3. **Diffraction edge policy can diverge between Channel Native and RayDN.**
   - Mitigation: reuse `Scene.diffraction_edge_count()` selection logic and add single-wedge edge-id tests.

4. **Dense radiomap memory can grow quickly.**
   - Mitigation: enforce `max_paths`, add chunked accumulation, and report memory planning metadata.

5. **AD can silently include topology decisions.**
   - Mitigation: keep AD disabled until explicit fixed-topology tests pass, and reject unsupported `ad_mode` values.

---

## Command Set

Use the `witwin2` environment:

```powershell
conda run -n witwin2 python -m pip install -e . --no-deps
conda run -n witwin2 python -m pytest tests/deterministic -q
conda run -n witwin2 python -m pytest tests/path tests/core tests/kernels -q
conda run -n witwin2 python benchmarks/bench_deterministic_radiomap.py --json
```

Native iteration should prefer the existing CMake build workflow used by MC basic and path solver work.
