# `witwin.channel`

The package root exposes native field ABI types, the solver capability
manifest, and deployment diagnostics. It does not re-export solvers, logical
world types, or runtime implementation objects.

```python
import witwin.channel as channel
```

## `Complex3State`

```text
Complex3State(field: torch.Tensor, direction: torch.Tensor)
```

Canonical world-Cartesian complex field state.

| Field | Type and shape | Description |
| --- | --- | --- |
| `field` | `complex64`, `(N, 3)` | Complex electric field in world Cartesian coordinates. |
| `direction` | `float32`, `(N, 3)` | Propagation direction. |

Both tensors must be CUDA tensors with equal row count and device. Construction
normalizes storage to contiguous layout.

```python
state = channel.Complex3State(
    field=torch.tensor([[1 + 0j, 0 + 0j, 0 + 0j]], device="cuda"),
    direction=torch.tensor([[0.0, 0.0, 1.0]], device="cuda"),
)
print(state.field.shape)  # torch.Size([1, 3])
```

Raises `TypeError` for incorrect dtypes and `ValueError` for invalid shapes,
non-CUDA tensors, row mismatch, or device mismatch.

## `JonesState`

```text
JonesState(
    value: torch.Tensor,
    basis: torch.Tensor,
    direction: torch.Tensor,
)
```

Two-component complex field in an explicit transverse world basis.

| Field | Type and shape | Description |
| --- | --- | --- |
| `value` | `complex64`, `(N, 2)` | Two complex field components. |
| `basis` | `float32`, `(N, 2, 3)` | Two world-coordinate basis vectors per row. |
| `direction` | `float32`, `(N, 3)` | Propagation direction. |

```python
state = channel.JonesState(
    value=torch.tensor([[1 + 0j, 0 + 0j]], device="cuda"),
    basis=torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device="cuda"
    ),
    direction=torch.tensor([[0.0, 0.0, 1.0]], device="cuda"),
)
print(state.value.shape)  # torch.Size([1, 2])
```

All tensors must share rows and device and must be CUDA-resident.

## `capabilities`

```text
capabilities() -> dict[str, Any]
```

Returns the versioned solver-level semantic capability manifest. Use this
record for capability discovery rather than inferring support from private
modules.

```python
caps = channel.capabilities()
print(caps["components"])
print(caps["propagation_consumer"]["contract_version"])
```

The function takes no parameters and launches no propagation work.

## `build_info`

```text
build_info() -> dict[str, Any]
```

Returns the build identity reported directly by the loaded `_channel`
extension, including native ABI, CUDA architectures, build fingerprint, and
locked RayD identity.

```python
info = channel.build_info()
print(info["backend"])              # "channel"
print(info["channel_abi_version"])  # 1
print(info["build_fingerprint"])    # 64-character SHA-256 digest
```

The dictionary may gain reporting fields. Use `.get(...)` for optional
non-contractual metadata.

## `runtime_diagnostics`

```text
runtime_diagnostics() -> dict[str, Any]
```

Returns an import-safe report covering the package, PyTorch/CUDA runtime,
active device, native build, declared SM policy, and pipeline-cache status.
It is diagnostic output, not a fallback or capability override.

```python
diagnostics = channel.runtime_diagnostics()
print(diagnostics["deployment_abi"])
print(diagnostics["declared_sm_architectures"])
print(diagnostics.get("device"))
print(diagnostics["errors"])
```

## `pipeline_cache_key`

```text
pipeline_cache_key(
    *,
    geometry_version: int,
    material_version: int,
    assignment_version: int,
    frequency_hz: float,
    solver: str,
    build: dict[str, Any] | None = None,
) -> str
```

Produces a deterministic SHA-256 digest for the pipeline-cache ABI. The API
generates a key; it does not implement a pipeline cache.

| Parameter | Description |
| --- | --- |
| `geometry_version` | Core geometry-version integer. |
| `material_version` | Core material-version integer. |
| `assignment_version` | Core logical-assignment-version integer. |
| `frequency_hz` | Reference frequency in hertz. |
| `solver` | Solver identity, for example `"deterministic"`. |
| `build` | Optional build record; `None` is equivalent to an empty dictionary. |

### Returns

A 64-character lowercase hexadecimal digest.

```python
key = channel.pipeline_cache_key(
    geometry_version=12,
    material_version=4,
    assignment_version=7,
    frequency_hz=3.5e9,
    solver="deterministic",
    build=channel.build_info(),
)
print(len(key))  # 64
```

Changing any version, frequency, solver, or build field changes the digest.
A non-JSON-serializable `build` value raises `TypeError`.
