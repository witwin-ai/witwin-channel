# PathMonitor — Path-Level Structured Output Design

## 1. Motivation

The current tracer outputs **grid-accumulated field maps** via `FieldMonitor` — every receiver pixel gets the coherent sum of all propagation paths, but individual path attributes (delay, AoD, AoA, interaction sequence, per-path amplitude) are discarded after accumulation. This is sufficient for field visualization but insufficient for:

- Channel impulse response (CIR) / channel frequency response (CFR) generation
- MIMO channel matrix construction (H-matrix per path)
- Path-level gradient analysis and inverse problems
- Dataset export for ML channel models
- Comparison with measurement-based path extractors

Sionna RT's `Paths` object provides this per-path output. We need an equivalent that fits the `Scene + Tracer + Result` architecture.

---

## 2. Design Principles

1. **New monitor type, not a mode switch.** `PathMonitor` is a peer of `FieldMonitor`, not a flag on `Tracer.trace()`. Users declare intent at the scene level.
2. **Discrete receiver points, not a grid.** `PathMonitor` takes explicit `positions: Tensor[N, 3]` instead of bounds + grid_size.
3. **PyTorch-native output.** All per-path tensors are `torch.Tensor` on CUDA. DrJit is internal plumbing; the public surface is pure PyTorch.
4. **Padded + masked layout.** Variable path counts across receivers are handled with a fixed `max_num_paths` dimension and a boolean `valid` mask, matching the Sionna convention.
5. **Lazy geometry.** Core fields (`a`, `tau`, `theta_t`, `phi_t`, `theta_r`, `phi_r`, `valid`, `types`) are always populated. Interaction-level geometry (`vertices`, `objects`, `normals`) is populated only when `return_geometry=True` to avoid memory bloat.
6. **Differentiable.** `a` and `tau` remain in the AD graph when the scene has differentiable vertices.
7. **Composable with FieldMonitor.** A single `trace()` call can include both `FieldMonitor` and `PathMonitor` instances.

---

## 3. Public API

### 3.1 PathMonitor (scene-level declaration)

```python
from witwin.channel import PathMonitor

monitor = PathMonitor(
    "rx_array",
    positions=torch.tensor([       # [num_rx, 3] — explicit receiver positions
        [2.0, 0.0, 1.5],
        [4.0, 1.0, 1.5],
        [6.0, -1.0, 1.5],
    ]),
    ray_mode="3d",                 # "2d" | "3d", default "3d"
    max_num_paths=None,            # auto-determined if None; cap if int
)

scene = Scene(structures=[...], monitors=[plane_mon, monitor])
```

**Fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Unique monitor name |
| `positions` | `Tensor[N,3]` | required | Receiver world-space positions |
| `ray_mode` | `str` | `"3d"` | Reflection sampling mode |
| `max_num_paths` | `int \| None` | `None` | Hard cap on paths per receiver |
| `return_geometry` | `bool` | `False` | Populate per-interaction vertices/normals |
| `kind` | `str` | `"path"` | Monitor type discriminator (read-only) |

### 3.2 PathResult (output container)

```python
result = tracer.trace(tx_pos)
pr = result.paths("rx_array")   # -> PathResult

pr.a          # [num_rx, max_num_paths]  complex64 — channel coefficient
pr.tau        # [num_rx, max_num_paths]  float32   — propagation delay (seconds)
pr.theta_t    # [num_rx, max_num_paths]  float32   — zenith AoD (radians from +z)
pr.phi_t      # [num_rx, max_num_paths]  float32   — azimuth AoD (radians)
pr.theta_r    # [num_rx, max_num_paths]  float32   — zenith AoA (radians from +z)
pr.phi_r      # [num_rx, max_num_paths]  float32   — azimuth AoA (radians)
pr.valid      # [num_rx, max_num_paths]  bool      — padding mask
pr.types      # [num_rx, max_num_paths, max_depth]  uint8  — InteractionType per bounce
pr.num_paths  # [num_rx]  int32 — actual path count per receiver

# Optional geometry (when return_geometry=True)
pr.vertices   # [num_rx, max_num_paths, max_depth, 3]  float32 — interaction points
pr.normals    # [num_rx, max_num_paths, max_depth, 3]  float32 — surface normals
pr.objects    # [num_rx, max_num_paths, max_depth]      int32   — structure index
```

### 3.3 InteractionType

```python
class InteractionType:
    NONE       = 0   # unused depth slot (or LoS)
    REFLECTION = 1   # specular reflection
    DIFFRACTION = 2  # edge diffraction (UTD)
    # Reserved for future:
    # TRANSMISSION = 4
    # SCATTERING   = 8
```

### 3.4 CIR / CFR Methods on PathResult

```python
# Channel impulse response
a_cir, tau_cir = pr.cir(normalize_delays=True)
# a_cir: [num_rx, max_num_paths] complex64
# tau_cir: [num_rx, max_num_paths] float32 (normalized)

# Channel frequency response
H = pr.cfr(frequencies=torch.linspace(27.5e9, 28.5e9, 1024))
# H: [num_rx, num_frequencies] complex64

# Discrete taps
h = pr.taps(bandwidth=100e6, num_taps=64)
# h: [num_rx, num_taps] complex64
```

### 3.5 Integration with Result

```python
class Result:
    # Existing:
    def monitor(self, name) -> MonitorResult: ...
    @property
    def primary(self) -> MonitorResult: ...

    # New:
    def paths(self, name) -> PathResult: ...
    @property
    def path_monitors(self) -> Mapping[str, PathResult]: ...
```

`Result` holds both `MonitorResult` (from `FieldMonitor`) and `PathResult` (from `PathMonitor`) in separate dicts. The `primary` property still returns a `MonitorResult` (grid-based). `paths()` is the accessor for path-level data.

---

## 4. Data Model

### 4.1 PathResult (frozen dataclass)

```python
@dataclass(frozen=True)
class PathResult:
    """Per-path structured output for a set of discrete receivers."""

    name: str
    num_rx: int
    max_num_paths: int
    max_depth: int
    tx_pos: tuple[float, float, float]
    rx_positions: torch.Tensor          # [num_rx, 3]
    frequency: float
    wavelength: float

    # Core per-path tensors — always populated
    a: torch.Tensor                     # [num_rx, max_num_paths] complex64
    tau: torch.Tensor                   # [num_rx, max_num_paths] float32
    theta_t: torch.Tensor               # [num_rx, max_num_paths] float32
    phi_t: torch.Tensor                 # [num_rx, max_num_paths] float32
    theta_r: torch.Tensor               # [num_rx, max_num_paths] float32
    phi_r: torch.Tensor                 # [num_rx, max_num_paths] float32
    valid: torch.Tensor                 # [num_rx, max_num_paths] bool
    types: torch.Tensor                 # [num_rx, max_num_paths, max_depth] uint8
    num_paths: torch.Tensor             # [num_rx] int32

    # Optional interaction-level geometry
    vertices: torch.Tensor | None       # [num_rx, max_num_paths, max_depth, 3]
    normals: torch.Tensor | None        # [num_rx, max_num_paths, max_depth, 3]
    objects: torch.Tensor | None        # [num_rx, max_num_paths, max_depth] int32

    # Solver provenance
    metadata: Mapping[str, object]

    def cir(self, *, normalize_delays: bool = True) -> tuple[torch.Tensor, torch.Tensor]: ...
    def cfr(self, frequencies: torch.Tensor, *, normalize_delays: bool = True) -> torch.Tensor: ...
    def taps(self, bandwidth: float, num_taps: int, *, normalize_delays: bool = True) -> torch.Tensor: ...

    def filter_by_type(self, *interaction_types: int) -> "PathResult": ...
    def to_dict(self) -> dict[str, torch.Tensor]: ...
```

### 4.2 Shape Conventions

We deliberately keep it simpler than Sionna because we don't yet have antenna arrays:

```
Dimension ordering:  [num_rx, max_num_paths, ...]

num_rx         = number of discrete receiver positions
max_num_paths  = global maximum path count across all receivers (padded)
max_depth      = maximum interaction depth across all collected paths
```

When antenna arrays are added later, the shape extends to `[num_rx, num_rx_ant, max_num_paths, ...]`. The current design reserves this by keeping `num_rx` as the outermost dimension.

### 4.3 Padding Convention

- Slots with `valid[i, j] == False` have: `a = 0+0j`, `tau = -1`, angles = 0, `types = NONE`
- `num_paths[i]` gives the actual count so `valid[i, :num_paths[i]]` is all-True
- Paths are sorted by ascending `tau` within each receiver

---

## 5. Tracer Integration

### 5.1 Monitor Dispatch

`Tracer.trace()` already resolves monitors via `_resolve_trace_monitors()`. Extend this to accept `PathMonitor` alongside `FieldMonitor`:

```python
def trace(self, tx_pos, *, monitor=None, ...):
    tx_pos = self._coerce_tx_pos(tx_pos)
    plane_monitors, path_monitors = self._resolve_all_monitors(monitor=monitor)

    monitor_payloads = {}
    for pm in plane_monitors:
        monitor_payloads[pm.name] = self._trace_field_monitor(tx_pos, pm, ...)

    path_payloads = {}
    for pm in path_monitors:
        path_payloads[pm.name] = self._trace_path_monitor(tx_pos, pm, ...)

    return Result(
        scene=self.scene,
        monitors=monitor_payloads,
        path_monitors=path_payloads,
        primary_monitor_name=plane_monitors[0].name if plane_monitors else None,
    )
```

### 5.2 `_trace_path_monitor()` — New Core Method

This is the central new method. It runs the same three solvers (LoS, reflection, diffraction) but in **path-collection mode** instead of grid-accumulation mode.

```python
def _trace_path_monitor(
    self,
    tx_pos: bk.Point3f,
    monitor: PathMonitor,
    *,
    verbose: bool,
    return_timing: bool,
) -> PathResult:
```

**High-level flow:**

```
1. Convert monitor.positions to DrJit arrays (X, Y, Z per receiver)
2. Collect LoS paths         → list[RawPath]
3. Collect reflection paths  → list[RawPath]
4. Collect diffraction paths → list[RawPath]
5. Merge + pad + sort        → PathResult tensors
```

### 5.3 Path Collection from Each Solver

#### LoS Path Collection

For each receiver `rx_i`:
- Compute `d = ||tx - rx_i||`, check occlusion
- If visible: one path with `a = λ/(4πd) * exp(-jkd)` (with polarization), `tau = d/c`, `types = [NONE]*max_depth` (empty interaction chain = LoS)
- AoD/AoA from the TX→RX direction vector

This is straightforward — `compute_los_field` already computes `d_los` and the blocked mask. We refactor to expose per-rx path data rather than only the accumulated field.

**New function:**

```python
def collect_los_paths(
    scene, rx_positions, tx_pos, wavelength, k, tx_polarization, rx_polarization,
) -> dict:
    """Returns per-rx LoS path attributes as DrJit arrays."""
    # blocked mask, distance, complex amplitude, departure/arrival angles
```

#### Reflection Path Collection

The reflection solver already discovers unique image-source paths in `source_paths_per_bounce`. Currently these are replayed onto a grid. For `PathMonitor`, we replay them onto discrete receiver points instead.

**Key change:** Instead of `accumulate_reflection_paths_to_receivers` writing into a grid `Complex2f`, we write into per-(rx, path_slot) tensors.

For each bounce order `b` and each unique image-source path `p`:
- Replay the reflection chain to each `rx_i` → get exact hit points, total path length, per-bounce Fresnel coefficients
- `a` = product of Fresnel coefficients × `λ/(4πd_total)` × `exp(-jkd_total)` (with full Jones transport)
- `tau` = `d_total / c`
- `types[0..b-1]` = `REFLECTION`
- AoD from TX → first hit point direction
- AoA from last hit point → RX direction
- `vertices[0..b-1]` = hit points (if `return_geometry`)

**New function:**

```python
def collect_reflection_paths(
    scene, rx_positions, tx_pos, wavelength, k,
    n_rays, max_bounces, mode, materials, tx_pol, rx_pol,
    *, return_geometry=False,
) -> dict:
    """
    Run Monte Carlo reflection discovery + image-method replay to discrete receivers.
    Returns per-path attributes as flat arrays to be assembled later.
    """
```

This reuses the existing Monte Carlo path discovery (same `_compute_reflection_field_impl` inner loop) but replaces the grid replay with a discrete-receiver replay.

#### Diffraction Path Collection

The diffraction solver already maintains full `state_arrays` with per-edge-state geometry. Currently, `_edge_state_field_to_targets` accumulates `(state × rx_grid)` products into a grid. For `PathMonitor`, we instead evaluate the UTD field per `(state × rx_point)` pair and store each as a separate path.

For each diffraction state `s` and receiver `rx_i`:
- Compute `phi`, `s_dist`, `sin_beta` for the (state, rx) pair
- Evaluate UTD coefficient → `a`
- Total path length: `s_prime + s_dist` (plus any prefix reflection legs)
- `tau` from total path length
- `types` sequence from `state_arrays.path_edge_idx` and `prefix/intermediate/suffix_reflection_depth`
- AoD from source direction
- AoA from edge → rx direction

**New function:**

```python
def collect_diffraction_paths(
    state_arrays, rx_positions, tx_pos, scene,
    wavelength, k, tx_pol, rx_pol,
    *, return_geometry=False,
) -> dict:
    """
    Evaluate diffraction states at discrete receivers.
    Returns per-(rx, state) path attributes.
    """
```

### 5.4 Path Merging and Padding

After collecting raw paths from all three solvers, merge them per receiver:

```python
def merge_paths(
    los_paths: dict,
    reflection_paths: dict,
    diffraction_paths: dict,
    *,
    num_rx: int,
    max_num_paths: int | None,
) -> dict[str, torch.Tensor]:
    """
    Merge heterogeneous path lists into padded [num_rx, max_num_paths, ...] tensors.
    Sort by ascending tau within each receiver.
    Apply max_num_paths cap if specified (keep strongest paths by |a|).
    """
```

---

## 6. Implementation Phases

### Phase 1 — Data Model + LoS Paths

**Files to create:**
- `witwin/channel/monitors/` — add `PathMonitor` alongside `FieldMonitor`
- `witwin/channel/result.py` — add `PathResult` dataclass, extend `Result` with `paths()` accessor
- `witwin/channel/trace/path/__init__.py` — new package for path collection
- `witwin/channel/trace/path/los.py` — `collect_los_paths`
- `witwin/channel/trace/path/merge.py` — `merge_paths` + padding logic

**Files to modify:**
- `witwin/channel/trace/tracer.py` — add `_trace_path_monitor`, extend `trace()` dispatch
- `witwin/channel/scene/core.py` — accept `PathMonitor` in monitor validation
- `witwin/channel/__init__.py` — export `PathMonitor`, `PathResult`

**Deliverable:** `tracer.trace(tx, monitor=PathMonitor(...))` returns `PathResult` with LoS paths only. CIR/CFR methods work.

**Test:**
```python
def test_path_monitor_los_basic():
    scene = Scene(structures=[...])
    tracer = Tracer(28e9, scene)
    result = tracer.trace(tx, monitor=PathMonitor("rx", positions=rx_pts))
    pr = result.paths("rx")
    assert pr.a.shape == (num_rx, pr.max_num_paths)
    assert pr.tau.shape == (num_rx, pr.max_num_paths)
    # Verify LoS path delay matches distance/c
    d = torch.norm(rx_pts - tx, dim=1)
    expected_tau = d / C
    assert torch.allclose(pr.tau[:, 0], expected_tau, rtol=1e-5)
```

### Phase 2 — Reflection Paths

**Files to create:**
- `witwin/channel/trace/path/reflection.py` — `collect_reflection_paths`

**Files to modify:**
- `witwin/channel/trace/reflection/epc.py` — factor out discrete-receiver replay from grid replay
- `witwin/channel/trace/tracer.py` — wire reflection collection into `_trace_path_monitor`

**Deliverable:** `PathResult` contains both LoS and reflection paths. `types` array correctly encodes `[REFLECTION]*n` for n-bounce paths.

**Key implementation detail:** The Monte Carlo discovery pass is shared between `FieldMonitor` and `PathMonitor`. Only the accumulation/replay target changes. Factor the discovery pass into a shared function that returns `source_paths_per_bounce`, then dispatch to either grid-replay or discrete-replay.

### Phase 3 — Diffraction Paths

**Files to create:**
- `witwin/channel/trace/path/diffraction.py` — `collect_diffraction_paths`

**Files to modify:**
- `witwin/channel/trace/diffraction/geometry/fields.py` — factor UTD evaluation to support per-(state, rx) output
- `witwin/channel/trace/tracer.py` — wire diffraction collection into `_trace_path_monitor`

**Deliverable:** Full `PathResult` with LoS + reflection + all diffraction families. The `types` array encodes mixed sequences like `[REFLECTION, DIFFRACTION, NONE]`.

**Key implementation detail:** The existing `_edge_state_field_to_targets` does `state × grid` accumulation via chunked scatter. For discrete receivers, we instead do a Cartesian expansion `state × rx` and keep each product as a separate path entry. This is simpler (no DDA grid walking) but produces `n_states × n_rx` candidate paths that must be filtered by visibility and merged.

### Phase 4 — CIR / CFR / Taps Methods

**Files to modify:**
- `witwin/channel/result.py` — implement `PathResult.cir()`, `.cfr()`, `.taps()`

These are pure PyTorch operations on the already-collected `a` and `tau` tensors:

```python
def cfr(self, frequencies, *, normalize_delays=True):
    # H[rx, f] = sum_p a[rx, p] * exp(-j * 2pi * f * tau[rx, p])
    tau = self.tau.clone()
    if normalize_delays:
        tau_min = tau.where(self.valid, torch.inf).min(dim=1, keepdim=True).values
        tau = tau - tau_min
    # [num_rx, max_num_paths, 1] x [1, 1, num_freq] -> [num_rx, max_num_paths, num_freq]
    phase = -2 * math.pi * tau.unsqueeze(-1) * frequencies.unsqueeze(0).unsqueeze(0)
    H = (self.a.unsqueeze(-1) * torch.exp(1j * phase)).sum(dim=1)
    return H  # [num_rx, num_freq]
```

### Phase 5 — Geometry + Interaction Details

**Files to modify:**
- Path collection functions — populate `vertices`, `normals`, `objects` when `return_geometry=True`

**Deliverable:** Full interaction-level data for visualization and analysis.

---

## 7. Detailed File Map

```
witwin/channel/
├── monitors.py              # + PathMonitor class
├── result.py                # + PathResult, Result.paths()
├── __init__.py              # + exports
├── scene/
│   └── core.py              # accept PathMonitor in monitor list
└── trace/
    ├── tracer.py            # + _trace_path_monitor, dispatch
    └── path/        # NEW package
        ├── __init__.py      # re-exports
        ├── types.py         # InteractionType constants
        ├── los.py           # collect_los_paths
        ├── reflection.py    # collect_reflection_paths
        ├── diffraction.py   # collect_diffraction_paths
        ├── merge.py         # merge + pad + sort
        └── angles.py        # AoD/AoA computation helpers
```

---

## 8. Key Design Decisions

### 8.1 Why Not Modify FieldMonitor?

`FieldMonitor` produces **spatially sampled field maps** — the output is a 2D grid of complex amplitudes where all paths are coherently summed. This is fundamentally different from per-path decomposition. Mixing the two concerns would complicate both the monitor API and the accumulation backend.

### 8.2 Why DrJit-Native Output With Torch Helpers?

`PathMonitor` now aligns with `FieldMonitor`: the public `PathResult` surface is DrJit-native, so traced path amplitudes, delays, masks, interaction codes, and optional geometry stay on the same backend family as the rest of the tracer. Torch is still available where it adds value, but it is deferred to helper-style post-processing such as CIR/CFR/tap evaluation instead of becoming the primary result representation.

### 8.3 Why Padded Tensors, Not Ragged?

1. GPU kernels work best on regular shapes
2. Regular shapes still matter even when the final result is DrJit-native
3. The `valid` mask pattern is well-established (Sionna, PyTorch3D)
4. CIR/CFR operations reduce over the path dimension, where masking is trivial

### 8.4 Why Not a Flat PathsBuffer Like Sionna?

Sionna uses a flat internal `PathsBuffer` during tracing, then reshapes into the `[num_rx, ..., max_num_paths]` public tensor in a post-processing scatter. We can do the same internally, but the public surface should be the reshaped tensor form — it's simpler for users and compatible with standard PyTorch operations.

### 8.5 max_num_paths Cap

When `max_num_paths=None`, the dimension is set to the actual maximum across receivers. When specified, paths are pruned by `|a|` (keep the strongest). This prevents memory explosion in complex scenes where one receiver might see thousands of diffraction states.

### 8.6 Differentiability

`a` is differentiable w.r.t. scene vertex positions (through DrJit AD → PyTorch). `tau` is differentiable w.r.t. vertex positions (path length depends on vertex-controlled hit points). Angles are not differentiable (discrete direction).

### 8.7 Path Sorting

Paths are sorted by ascending `tau` within each receiver. This is the natural ordering for CIR construction and matches Sionna's convention. If users need amplitude-sorted paths, they can do `idx = pr.a.abs().argsort(dim=1, descending=True)`.

---

## 9. Interaction with Existing Features

### 9.1 Mixed FieldMonitor + PathMonitor

A `trace()` call can include both:

```python
scene = Scene(
    structures=[...],
    monitors=[
        FieldMonitor("field_map", position=1.5, bounds=...),
        PathMonitor("rx_array", positions=rx_pts),
    ],
)
result = tracer.trace(tx)
field_map = result.primary           # MonitorResult (grid)
paths     = result.paths("rx_array") # PathResult (per-path)
```

The tracer runs the shared Monte Carlo reflection discovery once and uses it for both monitors.

### 9.2 Diffraction Audit Compatibility

`return_diffraction_audit=True` still works for `FieldMonitor`. For `PathMonitor`, the `PathResult.metadata` dict includes equivalent solver provenance without the grid-specific audit fields.

### 9.3 Solver Mode Compatibility

`solver_mode="accuracy"` and `"fast_approximate"` apply the same guardrails to path collection as they do to grid accumulation. The budget controls (`diffraction_state_budget`, etc.) limit the number of diffraction states enumerated, which directly limits the number of diffraction paths collected.

---

## 10. Comparison with Sionna RT Paths

| Feature | Sionna RT | witwin PathMonitor |
|---|---|---|
| **Core fields** | `a, tau, theta_t/r, phi_t/r, valid` | Same |
| **Interaction types** | `SPECULAR, DIFFUSE, REFRACTION, DIFFRACTION` | `REFLECTION, DIFFRACTION` (extensible) |
| **Per-interaction** | `vertices, objects, primitives` + lazy normals | `vertices, normals, objects` (opt-in) |
| **Antenna arrays** | `[num_rx, num_rx_ant, num_tx, num_tx_ant, max_paths]` | `[num_rx, max_paths]` (antenna dim reserved) |
| **CIR/CFR** | `cir()`, `cfr()`, `taps()` | Same |
| **Doppler** | Per-path Doppler shift | Not yet (no dynamic scenes) |
| **Transmission** | Supported | Not yet |
| **Scattering** | Diffuse reflection | Not yet |
| **Output format** | DrJit/TF/JAX/NumPy/Torch | PyTorch native |
| **Differentiable** | Through DrJit symbolic loops | Through DrJit AD → PyTorch |

---

## 11. Future Extensions

These are NOT in scope for the initial implementation but the design accommodates them:

1. **Antenna arrays** — add `num_tx_ant`, `num_rx_ant` dimensions to `a`, apply phase shifts from AoD/AoA and array geometry. The `PathResult` shape convention extends naturally.

2. **Doppler** — add `doppler: Tensor[num_rx, max_num_paths]` field when dynamic scenes are supported. The `cir()` method gains a `num_time_steps` parameter.

3. **Transmission paths** — `InteractionType.TRANSMISSION = 4`. The path collector adds a transmission solver. `types` array encodes the interaction sequence.

4. **Diffuse scattering** — `InteractionType.SCATTERING = 8`. Similar extension pattern.

5. **Multi-TX** — extend to `[num_tx, num_rx, max_num_paths, ...]`. Current single-TX design is a special case with `num_tx=1` squeezed.

6. **Path identity tracking** — add `path_id: Tensor[num_rx, max_num_paths]` for persistent path tracking across scene updates in optimization loops.

---

## 12. Implementation Order and Estimated Scope

| Phase | What | New LOC (est.) | Modified LOC (est.) |
|---|---|---|---|
| P1 | Data model + LoS | ~350 | ~100 |
| P2 | Reflection paths | ~300 | ~150 |
| P3 | Diffraction paths | ~400 | ~200 |
| P4 | CIR/CFR/taps | ~120 | ~20 |
| P5 | Geometry details | ~150 | ~50 |
| **Total** | | **~1320** | **~520** |

Phase 1–2 are independent of Phase 3 and can be validated separately. Phase 4 depends only on the `PathResult` data model (Phase 1). Phase 5 is independent of Phase 4.

Recommended implementation sequence: **P1 → P4 → P2 → P3 → P5** (get the data model + CIR working first with LoS-only, then add complexity).
