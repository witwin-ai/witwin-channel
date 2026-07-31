# `witwin.channel.scene`

The scene lifecycle is the supporting boundary between a logical
`witwin.core` world and Channel-owned resident native resources. The four
solver entry points compile internally; direct propagation-consumer callers
use this module explicitly.

```python
from witwin.channel.scene import CompiledScene, clear_compile_cache, compile
```

## `compile`

```text
compile(
    scene_or_snapshot: witwin.core.Scene | witwin.core.SceneSnapshot,
    *,
    reference_frequency_hz,
) -> CompiledScene
```

Compiles a Core world at one reference frequency and returns an application-immutable
Channel-owned scene containing the native RayD scene/BVH facade, resident
material stores, endpoint state, and version provenance.

### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `scene_or_snapshot` | `Scene | SceneSnapshot` | Logical world or time-labelled snapshot owned by `witwin.core`. |
| `reference_frequency_hz` | `float | torch.Tensor` | Positive reference frequency in hertz. A differentiable scalar tensor is accepted where the selected route supports it. |

### Returns

`CompiledScene`, suitable for `consumer.evaluate`, `consumer.reevaluate`, and
`consumer.evaluate_time_varying`.

### Raises

- `TypeError` or `ValueError` for an invalid world or frequency contract.
- `RuntimeError` when CUDA, the packaged extension, RayD capability, ABI, or a
  supported GPU architecture is unavailable.
- A typed capability error when the selected native trace backend cannot
  support a required operation.

### Example

```python
from witwin.channel.scene import compile

compiled = compile(scene, reference_frequency_hz=3.5e9)
print(compiled.materials.frequency_hz)  # 3500000000.0
print(compiled.time_s)                  # None for Scene; snapshot label otherwise
```

The reference frequency is part of the compiled material/resource contract.
A later request-frequency mismatch fails. Channel does not implicitly
recompile or replay material physics on the host.

## `CompiledScene`

`CompiledScene` is the resource owner returned by `compile`. Applications must
treat its public state as immutable and normally pass it to the consumer and read only version/reporting
fields. Resource fields and resource-producing methods are documented for
type completeness, but their returned stores remain Channel-owned and must
not be mutated.

```text
CompiledScene(
    source: Scene | SceneSnapshot,
    structures: tuple[object, ...],
    geometry: GeometryStore,
    materials: MaterialStore,
    assignments: AssignmentStore,
    rayd: RayDSceneResource,
    reference_frequency_hz: float | torch.Tensor,
    reference_frequency_revision: int | None,
    topology_version: int,
    geometry_version: int,
    material_version: int,
    assignment_version: int,
    time_s: float | torch.Tensor | None = None,
    enumerated_penetration_scene_diagonal_m: float = 0.0,
    montecarlo_penetration_scene_diagonal_m: float = 0.0,
)
```

### Fields

| Field | Type | Description |
| --- | --- | --- |
| `source` | `Scene | SceneSnapshot` | Core world used for compilation and source revalidation. |
| `structures` | `tuple[object, ...]` | Stable compiled structure tuple. |
| `geometry` | `GeometryStore` | Channel-owned resident geometry store. |
| `materials` | `MaterialStore` | Channel-owned resident material store; `frequency_hz` records the compile frequency. |
| `assignments` | `AssignmentStore` | Channel-owned logical-to-native assignment store. |
| `rayd` | `RayDSceneResource` | Typed native RayD scene resource. Never reinterpret it as an integer handle. |
| `reference_frequency_hz` | `float | torch.Tensor` | Exact frequency declaration used to compile the scene. |
| `reference_frequency_revision` | `int | None` | Revision of a tensor frequency leaf when applicable. |
| `topology_version` | `int` | Compiled Core topology version. |
| `geometry_version` | `int` | Compiled Core geometry version. |
| `material_version` | `int` | Compiled Core material version. |
| `assignment_version` | `int` | Compiled Core assignment version. |
| `time_s` | `float | torch.Tensor | None` | Optional snapshot time; reporting metadata, never a compute gate. |
| `enumerated_penetration_scene_diagonal_m` | `float` | Compile-known scene diagonal for enumerated penetration policy. |
| `montecarlo_penetration_scene_diagonal_m` | `float` | Compile-known scene diagonal for Monte Carlo penetration policy. |

### `require_reference_frequency`

```text
require_reference_frequency(
    reference_frequency_hz: float | torch.Tensor,
) -> None
```

Validates that a request uses the compiled reference frequency and, for a
tensor leaf, the expected revision. Returns `None`; raises before native work
on mismatch. It never recompiles the scene.

### `fixed_reevaluation_tables`

```text
fixed_reevaluation_tables() -> dict[str, object]
```

Returns the CompiledScene-owned tables required by fixed-topology
reevaluation. Primal tables may be cached; differentiable leaves bypass a
stale autograd cache. The dictionary and contained tensors are read-only to
callers.

### `kirchhoff_resources`

```text
kirchhoff_resources() -> KirchhoffRuntimeResources
```

Lazily constructs and returns immutable scene-static Kirchhoff runtime
resources. Construction is publish-after-success; unrelated solves do not
allocate them.

### `phase_screen_resources`

```text
phase_screen_resources() -> PhaseScreenRuntimeResources
```

Lazily constructs immutable scene-static phase-screen resources. Endpoint,
frequency, and solve-dependent work remains outside this cache.

### `kirchhoff_tables`

```text
kirchhoff_tables() -> dict[int, KirchhoffTable]
```

Returns the material-ID keyed Kirchhoff table view owned by
`kirchhoff_resources()`.

### `rough_material_runtimes`

```text
rough_material_runtimes() -> dict[int, RoughMaterialRuntime]
```

Returns the material-ID keyed rough-material runtime view owned by the
compiled scene.

## `clear_compile_cache`

```text
clear_compile_cache() -> None
```

Clears the process-local compiled-scene cache. It does not mutate a
`witwin.core.Scene` and does not invalidate `CompiledScene` objects already
held by the caller.

```python
from witwin.channel.scene import clear_compile_cache

clear_compile_cache()
```

Use this primarily in tests, lifecycle management, or after deliberately
discarding cached worlds. Normal solving should reuse compiled resources.
