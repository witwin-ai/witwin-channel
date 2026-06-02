# FieldMonitor 3D Extension Analysis

> Date: 2026-03-29
> Status: Analysis & Implementation Plan

---

## 1. Current Architecture Summary

### 1.1 What is Already 3D

The ray tracing pipeline **already operates in 3D space**:

- **Ray representation**: All rays are `Vector3f(dx, dy, dz)` with 3D origins (`Point3f`)
- **Ray generation**: `generate_sphere_directions()` in `raygen.py` produces Fibonacci-lattice spherical directions with full `(sin(phi)cos(theta), sin(phi)sin(theta), cos(phi))` components
- **Intersection**: Mitsuba `ray_intersect` returns 3D hit points, 3D normals
- **Image method**: `_reflect_point_across_plane()`, cumulative image source, mirror distance — all fully 3D
- **Reflection chain**: `prev_refl_p`, `prev_refl_n`, `prev_tx` are all `Point3f`/`Vector3f`, z components correctly tracked
- **Polarization transport**: `reflect_field_vector()` operates on 3D incident/normal directions

### 1.2 What is Hard-Coded 2D

The **receiver accumulation layer** is the bottleneck — everything downstream of "ray hits a surface and needs to contribute to a receiver" assumes a z-constant horizontal plane with a 2D grid:

| File | Location | 2D Assumption |
|------|----------|---------------|
| `monitors.py:142-146` | `to_field()` | `axis != 'z'` raises `NotImplementedError` |
| `dda.py:84` | `grid.pos_to_idx(cur_x, cur_y)` | Grid index uses only x, y |
| `dda.py:93` | `cell_pos = Point3f(cell_x, cell_y, rx_z)` | Receiver z-coordinate is a fixed scalar |
| `dda.py:248-249` | `dt_x`, `dt_y` from `ray_dir.x`, `ray_dir.y` | DDA step sizes ignore z component |
| `dda.py:190-195` | `move_x = t_max_x < t_max_y` | DDA stepping only in x/y, no z axis |
| `dda.py:254-255` | `step_x`, `step_y` | No `step_z` |
| `field.py:72-88` | Grid coordinate setup | Only x and y linspace, no z grid |
| `field.py:75` | `max_steps = 2 * (nx + ny)` | 2D traversal budget |
| `tracer.py:516-520` | Monitor validation | Only z-normal monitors accepted |
| `tracer.py:549` | `rx_z = monitor.position` | Receiver height is a single scalar |
| `los.py` | LoS distance | Uses `Point3f(X, Y, rx_z)` — z is scalar broadcast |

### 1.3 Current `ray_mode='3d'` Behavior

The `ray_mode='3d'` option already exists and works, but its semantics are:

1. Tx emits rays in all 3D directions (sphere)
2. Rays bounce off surfaces in 3D
3. **After each bounce, the reflected ray is projected onto the xy-plane for DDA grid traversal**
4. Each grid cell's receiver is placed at `z=rx_z` and distance is computed in 3D from the image source

This means 3D rays that hit walls/ceilings above or below the monitor plane **can** contribute field (the image method geometry is correct), but the DDA traversal efficiency degrades for rays with large z-components because `dt_x`/`dt_y` become very large.

---

## 2. Problem Statement

**Goal**: Enable FieldMonitor to work with 3D spherical ray emission (Tx as an isotropic point source) while sampling the received field on axis-aligned planes (x-normal, y-normal, z-normal).

**Scope constraint**: Only axis-aligned monitor planes are in scope. Arbitrary-normal planes are explicitly out of scope — the `axis` parameter remains an enum of `'x' | 'y' | 'z'`.

Two sub-goals with different difficulty:

1. **Easier**: Tx spherical emission + z-constant horizontal monitor (improve current `ray_mode='3d'`)
2. **Moderate**: Tx spherical emission + x-normal or y-normal monitor planes (extend to all three axis-aligned orientations)

---

## 3. Difficulty Assessment

### 3.1 Sub-goal A: Spherical Tx + Horizontal Monitor — Low Difficulty

**Current status**: Mostly working. `ray_mode='3d'` + `axis='z'` already runs.

**Issues to fix**:

| Issue | Severity | Description |
|-------|----------|-------------|
| Ray utilization | Medium | Spherical emission wastes rays pointing steeply up/down. For a horizontal monitor, most useful rays are near-horizontal. With N sphere rays, effective horizontal-plane density is ~sqrt(N) vs N for circle mode. |
| DDA robustness for steep rays | Low | Rays with `dir.z >> dir.x, dir.y` produce very large `dt_x`/`dt_y`, making DDA nearly stationary. Not a correctness bug (distance is still computed correctly), but a noise/efficiency issue. |
| Ray count scaling | Medium | To match 2D quality with 10K rays, 3D mode needs ~100K-1M rays. Memory and compute scale linearly. |

**Estimated effort**: 1-2 days (validation + optional importance sampling).

### 3.2 Sub-goal B: Axis-Aligned Non-Z Monitor Planes (x-normal, y-normal) — Medium Difficulty

Since we restrict to axis-aligned planes, there is no general coordinate transform needed — each axis case is a fixed permutation of (x, y, z). This significantly reduces complexity compared to arbitrary normals.

**Core changes required**:

| Component | Effort | Description |
|-----------|--------|-------------|
| **DDA axis permutation** | Medium | The 2D DDA walks in (x, y) and fixes z. For x-normal: walk in (y, z), fix x. For y-normal: walk in (x, z), fix y. This is a systematic axis relabeling, not a rewrite. Alternatively, replace DDA with ray-plane intersection + scatter. |
| **Field/Grid axis mapping** | Medium | `Field` grid coordinates are currently (x, y). Need to map `bounds[0]` and `bounds[1]` to the correct tangential axes per monitor axis. |
| **Monitor `to_field()`** | Low | Remove the `NotImplementedError`. Each axis case maps `bounds` to the two tangential axes: z→(x,y), x→(y,z), y→(x,z). |
| **LoS computation** | Low | Replace `Point3f(X, Y, rx_z)` with axis-appropriate 3D receiver positions. The distance calculation is already 3D. |
| **DDA receiver position** | Low | `cell_pos = Point3f(cell_x, cell_y, rx_z)` becomes axis-dependent: e.g., for x-normal → `Point3f(rx_x, cell_y, cell_z)`. |
| **Diffraction module audit** | Medium | Verify no hidden z-constant assumptions in `_edge_state_field_to_targets`. |
| **Result storage** | Low | `MonitorResult` needs to label its grid axes correctly per monitor orientation. |

**Estimated effort**: 1-2 weeks for a complete implementation.

---

## 4. Key Technical Challenges

### 4.1 DDA Adaptation Strategy

The current DDA (`dda.py`, ~334 lines) marches rays through a 2D grid cell-by-cell in (x, y), accumulating field contributions at each cell. For axis-aligned monitors, two approaches:

**Option A: Axis-Permuted DDA (Recommended for axis-aligned)**

Since we only support x/y/z normals, the DDA can be parameterized by a simple axis permutation:

| Monitor axis | DDA walks in | Fixed coordinate |
|-------------|-------------|-----------------|
| z (current) | (x, y) | z = position |
| x | (y, z) | x = position |
| y | (x, z) | y = position |

Implementation: extract the current (x, y, z) references into `(tang0, tang1, normal)` variables selected by the monitor axis. The DDA loop structure and stepping logic remain identical — only the axis mapping changes.

```python
# Current hard-coded:
cur_x = ray_origin.x;  cur_y = ray_origin.y
dt_x = cell_size_x / abs(ray_dir.x);  dt_y = cell_size_y / abs(ray_dir.y)
cell_pos = Point3f(cell_x, cell_y, rx_z)

# Axis-parameterized:
cur_0 = getattr(ray_origin, tang0_name)  # e.g., 'y' for x-normal
cur_1 = getattr(ray_origin, tang1_name)  # e.g., 'z' for x-normal
dt_0 = cell_size_0 / abs(getattr(ray_dir, tang0_name))
dt_1 = cell_size_1 / abs(getattr(ray_dir, tang1_name))
# Reconstruct cell_pos with fixed normal coordinate
```

Pros: Preserves the multi-cell accumulation property. Minimal code change (axis relabeling).
Cons: DrJit symbolic loops may not like dynamic attribute access — may need three explicit code paths or a helper that builds the correct arrays at setup time.

**Option B: Direct Ray-Plane Intersection + Scatter**

```
# For axis-aligned plane, intersection is trivial:
# z-normal: t = (position - ray_origin.z) / ray_dir.z
# x-normal: t = (position - ray_origin.x) / ray_dir.x
# y-normal: t = (position - ray_origin.y) / ray_dir.y
hit_point = ray_origin + t * ray_dir
grid_idx = quantize_to_grid(hit_tang0, hit_tang1)
scatter_reduce(field_buffer, grid_idx, contribution)
```

Pros: Much simpler; no DDA loop; axis-aligned intersection is a single division.
Cons: Each ray contributes to exactly one cell (no continuous accumulation).

**Recommendation**: Option A for z-normal (preserves current quality), Option B as a fallback or for non-z monitors where the DDA refactor effort isn't justified. Since axis-aligned intersection is trivially cheap, Option B is attractive for x/y monitors.

### 4.2 Ray Efficiency / Importance Sampling

With spherical emission onto a planar monitor, the ray utilization problem is significant:

- **2D circle**: 100% of rays propagate in the monitor plane. 10K rays → 10K useful rays.
- **3D sphere**: Only rays within a narrow elevation band near the monitor plane are useful. 10K rays → ~1K-3K useful rays depending on scene geometry.
- **3D sphere + vertical walls**: Rays with large z-components hit ceiling/floor, reflect back down, and can contribute — but via longer paths with more attenuation.

**Mitigation strategies**:
1. **Stratified sphere sampling**: Bias the Fibonacci lattice toward the monitor plane's elevation
2. **Cone sampling**: If the monitor subtends a known solid angle from Tx, sample only that cone
3. **Brute force**: Simply use 10x more rays (cheapest to implement, GPU handles it)

### 4.3 Oblique Ray–Plane Crossing

Rays are **not** constrained to be parallel to the monitor plane. A ray can cross the monitor plane at any angle. The DDA handles this correctly because it does not step along the ray itself — it steps along the ray's **projection** onto the monitor plane's tangential axes:

```
3D ray:     origin ─────→ obliquely crosses monitor plane ─────→ hits wall
                                      │ projection
DDA walk:   proj_origin ──→ cell₁ → cell₂ → cell₃ → ...
                              │        │        │
                          d₃d(img,rx₁) d₃d(img,rx₂) d₃d(img,rx₃)
```

- `dt_tang0` / `dt_tang1` are computed from the ray direction's **tangential components** — a steep ray (large normal component, small tangential) produces large dt values, meaning the DDA takes few steps. This is physically correct: a steep ray sweeps fewer grid cells.
- `blocker_dist` is the full 3D `si.t` from Mitsuba. The DDA check `t < blocker_dist` uses the projected `t`, which is always ≤ the 3D distance, so occluded cells are correctly excluded.
- At each cell, the field contribution uses the **full 3D** image-source-to-receiver distance, so the phase and FSPL are exact regardless of crossing angle.

This property holds identically for x/y/z-normal monitors — only the choice of which two axes are "tangential" changes.

### 4.4 Multi-Bounce Image Method with Axis-Aligned Planes

The image method math is already fully 3D. The only change is how `cell_pos` is constructed:

| Monitor axis | `cell_pos` construction |
|-------------|------------------------|
| z (current) | `Point3f(cell_x, cell_y, position)` |
| x | `Point3f(position, cell_y, cell_z)` |
| y | `Point3f(cell_x, position, cell_z)` |

The `d_mirror = norm(cell_pos - mirror)` distance computation works unchanged regardless of which coordinate is fixed. **No change needed** to the core image method geometry.

### 4.5 Diffraction: Edge Selection and Scaling

#### 2D vs 3D Edge Selection

The edge filter in `topology.py:filter_diffraction_edges()` has two modes:

| Mode | Filter | When to use |
|------|--------|-------------|
| `vertical_only` (default) | `abs(edge_vec.z) / edge_len > vertical_ratio` (0.7) | 2D: rays in xy-plane only interact with near-vertical edges |
| `all_edges` | `edge_len > SMALL_EPS` | 3D: rays in all directions can diffract around any edge |

**Physical reasoning**: UTD diffraction coefficient depends on `sin(beta)` where `beta` is the angle between the ray and the edge axis. For 2D horizontal rays hitting a horizontal edge, `sin(beta) ≈ 0` and the diffraction contribution vanishes. So `vertical_only` correctly prunes zero-contribution edges in 2D. In 3D, rays with z-components produce nonzero `sin(beta)` for horizontal and oblique edges, so all edges must be included.

#### Scaling Impact

Switching to `all_edges` significantly increases the diffraction workload:

- **Edge count**: A typical mesh has ~5-10x more total edges than vertical-only edges. Indoor scenes with floors, ceilings, and sloped surfaces are worst.
- **Diffraction state count**: Grows as `O(n_edges)` for direct diffraction, `O(n_edges × n_reflections)` for mixed RD/DR families. With 5x edges, state count grows 5-25x.
- **Computation**: Each state is evaluated against all receiver grid cells. Total work scales as `O(n_states × n_receivers)`.

**Mitigation**:
1. `diffraction_state_budget` in `TraceConfig` already caps state count — this becomes critical in 3D.
2. `sin(beta)` threshold: edges where max `sin(beta)` over all source directions is below a threshold (e.g., 0.05) can be pruned without significant field error.
3. Edge importance sampling: prioritize edges closest to the Tx or with largest wedge angles.

#### Edge Spatial Acceleration: RayDi Edge BVH for Higher-Order Diffraction

**Current state of the `wedge/` package**:

The `witwin.channel.wedge` package already has a clean layered architecture:

```
RayDi scene.edge_info() / edge_topology()     ← GPU topology extraction (via RayDSceneAdapter)
→ build_wedge_geometry()                   ← Compute wedge_n, order face normals, classify validity
        → select_wedges()                      ← Filter by vertical_ratio / wedge_n threshold
            → pack_wedges()                    ← Final data for diffraction solver
```

- **Edge extraction**: Already GPU-accelerated via RayDi adapter — `RayDSceneAdapter.edge_info()` calls `scene.edge_info()` directly.
- **Wedge classification**: `build.py` computes `wedge_n = exterior_angle / π` from the two adjacent face normals. This is channel physics that correctly lives outside RayDi (RayDi provides raw geometry; channel decides what counts as a diffraction wedge).
- **Selection**: `select.py` applies `vertical_only` vs `all_edges` mode and `min_wedge_n` threshold.

**What is NOT accelerated**: Higher-order diffraction state construction in `diffraction/builders/higher.py` still uses brute-force Cartesian expansion (`n_prev_states × n_edges`), filtered by visibility. This is the bottleneck for 3D `all_edges` mode.

**RayDi Edge BVH** provides spatial queries that can replace the Cartesian expansion:

```python
# BVH-accelerated queries (batched DrJit arrays, differentiable)
result = scene.nearest_edge(points)   # Point → NearestPointEdge
result = scene.nearest_edge(ray)      # Ray   → NearestRayEdge
```

**Integration plan — only for higher-order state construction**:

| Step | Current | With Edge BVH |
|---|---|---|
| Edge extraction + wedge classification | `wedge/` package via RayDi adapter (already done) | No change |
| 1st order states | One state per selected wedge (already done) | No change |
| Higher-order candidates | Cartesian `n_states × n_edges` → visibility filter | `nearest_edge(ray)` per state → only nearby edges, O(n log n) |
| Candidate validation | `ray_intersect` per pair | Same, but far fewer candidates |

**Note**: `nearest_edge` returns raw edge IDs. The result must be cross-referenced with the `WedgeSelection` to verify the edge is actually a valid wedge (i.e., `wedge_n > min_wedge_n` and not excluded by selection policy). RayDi's BVH indexes all mesh edges, not just diffraction wedges.

**Key benefits for 3D**:
1. **Scalability**: O(n log n) vs O(n²). 500 wedges → ~4500 BVH queries instead of 250K brute-force pairs.
2. **GPU-native**: LBVH build + traversal fully on GPU, DrJit-compatible.
3. **Differentiable**: Edge queries support AD — enables future gradient-based optimization through diffraction paths.
4. **Refit**: After vertex updates (optimization loops), BVH refits without full rebuild.

#### Receiver Positions

For x/y-normal monitors, receiver positions change from `Point3f(X, Y, rx_z)` to axis-appropriate 3D points. All UTD quantities (`phi`, `s_dist`, `sin_beta`) are computed from 3D vectors, so **no fundamental changes needed** to UTD math — only receiver position generation changes.

### 4.6 Polarization: Transport is 3D, Scalarization is 2D

#### What is already 3D (no changes needed)

The polarization transport chain is fully 3D:

- **Reflection** (`reflect_field_vector`): Decomposes into TE/TM via `s_hat = cross(normal, incident)`, applies Fresnel coefficients, reconstructs output field. All 3D vectors.
- **Diffraction** (`transport_diffraction_vector`): Uses `phi_in = cross(incoming, edge)`, `phi_out = -cross(outgoing, edge)` as basis. Fully 3D.
- **Basis construction** (`stable_perpendicular_basis`): Has singularity handling — when ray is near-parallel to z-axis, falls back to y-axis as preferred direction. Works for all ray directions.

#### What needs to change: receiver-side scalarization

The final step converts the 3D complex field vector `{x, y, z}` to a scalar by projecting onto the receiver polarization direction. Currently this is hard-coded to the xy-plane:

```python
def xy_jones(vec):
    return {"x": vec["x"], "y": vec["y"]}   # discards z

def scalarize_xy_jones(jones, polarization):
    px, py = xy_receiver_polarization(polarization)  # only x, y components
    return jones["x"] * px + jones["y"] * py
```

This assumes the receiver's sensitive polarization lies in the xy-plane — correct for z-normal monitors (horizontal plane), **wrong for x/y-normal monitors**:

| Monitor axis | Tangential field components | `xy_jones` extracts | Error |
|-------------|---------------------------|--------------------|----|
| z-normal | (x, y) | (x, y) | Correct |
| x-normal | (y, z) | (x, y) | Takes normal component x, drops tangential z |
| y-normal | (x, z) | (x, y) | Takes normal component y, drops tangential z |

#### Required fix

Generalize to axis-aware tangential Jones extraction:

```python
def tangential_jones(vec, axis='z'):
    """Extract the two tangential field components for the given monitor normal axis."""
    if axis == 'z': return {"u": vec["x"], "v": vec["y"]}
    if axis == 'x': return {"u": vec["y"], "v": vec["z"]}
    if axis == 'y': return {"u": vec["x"], "v": vec["z"]}
```

The receiver polarization direction also needs to be expressed in tangential coordinates. For axis-aligned monitors this is a fixed permutation — no rotation matrices needed.

**Effort**: Low (small, localized change). **Risk if missed**: High — x/y-normal monitors would produce completely wrong polarization response (mixing normal and tangential components).

### 4.7 Validation Difficulty

For axis-aligned planes, validation is significantly simpler than for arbitrary normals:
- z-normal: existing 2D benchmarks (knife-edge, canonical wedge) apply directly
- x-normal / y-normal: equivalent benchmarks rotated 90° — same physics, axis-permuted geometry
- Spherical spreading factor is already in the FSPL formula

**Key risk**: Axis permutation bugs where `bounds[0]` and `bounds[1]` map to the wrong tangential axes, producing stretched or transposed field maps. Easy to catch with LoS-only tests on known geometry.

---

## 5. Implementation Plan

### Phase 1: Validate and Harden `ray_mode='3d'` on Horizontal Monitor

**Scope**: No architecture changes. Fix edge cases and add validation for the existing `ray_mode='3d'` + `axis='z'` path.

**Tasks**:
1. Add test comparing `ray_mode='2d'` vs `ray_mode='3d'` on a canonical scene (e.g., single wall reflection). Verify field amplitude agrees within expected Monte Carlo variance.
2. Profile DDA behavior with high-z-component rays. Quantify how many rays produce degenerate `dt_x`/`dt_y` (> 1e6) and whether they cause numerical artifacts.
3. Add a `min_ray_contribution_threshold` to skip DDA for rays whose projected xy-component is below a threshold (saves compute without losing significant contributions).
4. Document ray count scaling recommendations (e.g., "for 3D mode, use 10x the ray count of 2D mode").

**Acceptance Criteria**:
- [ ] Test: `ray_mode='3d'` with 100K rays on single-wall scene matches `ray_mode='2d'` with 10K rays within 1 dB RMS error on the field magnitude map
- [ ] Test: No NaN or Inf in DDA output with 3D rays including near-vertical directions
- [ ] Benchmark: `ray_mode='3d'` with 100K rays runs in < 2x wall time of `ray_mode='2d'` with 10K rays

**Estimated effort**: 2-3 days

---

### Phase 2: Make Field Natively 3D

**Scope**: `Field` becomes a 3D-aware planar grid. It knows its axis and position, and natively produces `Point3f` receiver positions. 2D is just the `axis='z'` case — no separate code path.

**Design**:

`FieldMonitor` already carries `axis` and `position`. `to_field()` simply passes them through — no axis guard, no special cases:

```python
# FieldMonitor.to_field() — works for any axis, no NotImplementedError
def to_field(self, wavelength, *, default_resolution=None):
    return Field(
        bounds=self.bounds,
        size=self.resolve_grid_shape(wavelength, default_resolution=default_resolution),
        axis=self.axis,            # 'x', 'y', or 'z'
        position=self.position,    # fixed coordinate value on the normal axis
    )
```

`Field` gains `axis` and `position`. Its coordinates are always the two tangential axes. The key new property is `receivers` — the canonical way to get `Point3f` positions:

```python
class Field:
    def __init__(self, bounds, size, axis='z', position=0.0):
        self.bounds = bounds
        self.size = size
        self.axis = axis
        self.position = position
        ...

    @property
    def tangential_axes(self) -> tuple[str, str]:
        """The two world axes spanned by this grid."""
        if self.axis == 'x': return ('y', 'z')
        if self.axis == 'y': return ('x', 'z')
        return ('x', 'y')

    @property
    def receivers(self) -> bk.Point3f:
        """Flattened 3D receiver positions on the monitor plane."""
        coords = self.get_coordinates()
        t0, t1 = coords['T0'], coords['T1']  # tangential grid coords
        fixed = bk.Float(self.position)
        if self.axis == 'x': return bk.Point3f(fixed, t0, t1)
        if self.axis == 'y': return bk.Point3f(t0, fixed, t1)
        return bk.Point3f(t0, t1, fixed)
```

`pos_to_idx(t0, t1)` stays as two-float input — callers extract the correct tangential components. No world-space projection needed because the caller always knows which axes are tangential.

`get_coordinates()` renames `X`/`Y` to axis-neutral `T0`/`T1` (tangential axis 0 and 1) internally, with backward-compatible `X`/`Y` properties that map to `T0`/`T1` for `axis='z'`.

**Tasks**:
1. Add `axis` and `position` to `Field.__init__()`. Default `axis='z', position=0.0` preserves backward compatibility.
2. Add `tangential_axes` property (same logic as `FieldMonitor.tangential_axes`).
3. Add `receivers` property returning `Point3f` with the fixed coordinate filled in.
4. Rename internal coordinate arrays from `X`/`Y` to `T0`/`T1`. Keep `X`/`Y` as aliases for `axis='z'`.
5. Remove the `axis != 'z'` guard in `FieldMonitor.to_field()`. Pass `axis` and `position` through.
6. Update `MonitorResult` metadata to include `axis` and `plane_position`.

**Acceptance Criteria**:
- [ ] `FieldMonitor("m", axis='x', position=5.0).to_field(wavelength)` returns a Field with `axis='x'`, `position=5.0`
- [ ] `field.receivers` for x-normal → `Point3f(5.0, grid_t0, grid_t1)` with correct tangential coordinates
- [ ] `field.receivers` for z-normal → `Point3f(grid_t0, grid_t1, 0.0)` — identical to current `Point3f(X, Y, rx_z)` behavior
- [ ] `field.pos_to_idx(t0, t1)` works identically for all axes (axis-agnostic — just two tangential floats)
- [ ] No regressions on existing z-normal tests (backward-compatible defaults)

**Estimated effort**: 3-4 days

---

### Phase 3: Extend DDA / Add Scatter Backend for x/y-Normal Monitors

**Scope**: Enable the reflection accumulation to work on x-normal and y-normal monitors. Two implementation options (can choose per-axis or globally):

**Option A — Axis-Permuted DDA** (preserve DDA for all axes):

1. Refactor `run_dda_traversal()` and `_run_dda_symbolic()` to accept axis-mapping parameters instead of hard-coded x/y/z:
   - Accept `tang0`, `tang1`, `normal` selectors
   - At DDA setup, extract `ray_dir.{tang0}`, `ray_dir.{tang1}` for step computation
   - Construct `cell_pos` with the correct axis mapping
2. Since DrJit symbolic loops need concrete array references (not dynamic getattr), implement this via a setup function that extracts the correct arrays before entering the loop:
   ```python
   def _extract_dda_axes(ray_origin, ray_dir, axis):
       if axis == 'z':
           return ray_origin.x, ray_origin.y, ray_dir.x, ray_dir.y, ...
       elif axis == 'x':
           return ray_origin.y, ray_origin.z, ray_dir.y, ray_dir.z, ...
       elif axis == 'y':
           return ray_origin.x, ray_origin.z, ray_dir.x, ray_dir.z, ...
   ```

**Option B — Ray-Plane Scatter** (simpler, for x/y monitors):

1. Implement `intersect_and_scatter()`:
   ```python
   def intersect_and_scatter(
       ray_origin, ray_dir,     # reflected ray
       axis, position,          # monitor plane: 'x'|'y'|'z', scalar
       grid,                    # Field grid on tangential plane
       image_source,            # for distance computation
       weight, polarization,    # field contribution
       wavelength, k,
       result_buffers,          # scatter targets
       blocker_dist,            # max valid t
   ):
       # axis-aligned intersection: t = (position - origin.{axis}) / dir.{axis}
       # Project hit to tangential coords, quantize, scatter
   ```
2. Wire as the accumulation backend for x/y-normal monitors (keep DDA for z-normal).

**Acceptance Criteria**:
- [ ] Test: x-normal monitor with single-wall scene produces correct single-reflection pattern (compare with rotated z-normal reference)
- [ ] Test: y-normal monitor LoS + reflection matches expected field
- [ ] Test: z-normal monitor is unchanged (DDA path preserved)
- [ ] Test: Edge cases — ray parallel to monitor plane (`dir.{axis} ≈ 0`), ray pointing away (t < 0)
- [ ] Performance: x/y-normal with 100K 3D rays runs within 3x wall time of z-normal DDA with 10K 2D rays

**Estimated effort**: 5-7 days

---

### Phase 4: Integrate LoS + Diffraction with x/y/z Monitor Planes + RayDi Edge BVH

**Scope**: Extend LoS and diffraction solvers to work with x/y-normal monitors. Migrate edge discovery and higher-order diffraction state construction from brute-force to RayDi Edge BVH.

**Tasks**:

**4a — LoS + Tracer axis generalization**:
1. **LoS**: Refactor `compute_los_field()` to accept 3D receiver positions from `Field.receiver_positions_3d(axis, position)` instead of constructing `Point3f(X, Y, rx_z)` inline.
2. **Tracer**: Remove the `axis != 'z'` guard in `_trace_field_monitor()`. Pass `monitor.axis` and `monitor.position` through to all sub-solvers. Replace all hard-coded `rx_z` references with axis-generic equivalents.

**4b — RayDi Edge BVH for higher-order diffraction**:
3. Replace brute-force Cartesian expansion in `diffraction/builders/higher.py` (`n_prev_states × n_edges`) with `scene.nearest_edge(ray)` queries from each state's outgoing direction. Cross-reference BVH results against `WedgeSelection` to ensure only valid wedges are used.
4. The existing `wedge/` package (RayDi adapter → build → select → pack) remains unchanged for 1st-order state construction. Only the higher-order builder changes.

**4c — Diffraction receiver positions**:
6. Refactor `_edge_state_field_to_targets()` to accept 3D receiver positions. Audit UTD geometry computations for z-constant assumptions.

**Acceptance Criteria**:
- [ ] Test: LoS field on x-normal monitor matches analytical FSPL within 0.1 dB for unobstructed line-of-sight
- [ ] Test: Single-edge diffraction on y-normal monitor matches UTD reference within 2 dB in the shadow region
- [ ] Test: Full trace (LoS + reflection + diffraction) on x-normal and y-normal monitors produces continuous, physically plausible field patterns
- [ ] Test: Mixed monitor list `[FieldMonitor(axis='z', ...), FieldMonitor(axis='x', ...)]` in a single `trace()` call returns correct results for both
- [ ] Test: Rotated equivalence — a scene rotated 90° with monitor axis swapped produces the same field pattern (within numerical tolerance)
- [ ] Test: Higher-order states via BVH match brute-force Cartesian results on z-normal test scenes (no regression)
- [ ] Benchmark: Higher-order diffraction state construction with RayDi BVH is ≥5x faster than brute-force for scenes with >100 edges

**Estimated effort**: 6-8 days (expanded from 4-5 to include RayDi integration)

---

### Phase 5: Importance Sampling and Performance Optimization

**Scope**: Address ray efficiency for 3D emission. Optimize the scatter backend.

**Tasks**:
1. Implement hemisphere / cone sampling in `raygen.py`:
   - `generate_hemisphere_directions(n_rays, normal)`: sample the hemisphere facing the monitor
   - `generate_cone_directions(n_rays, axis, half_angle)`: sample a cone for directed emission
2. Add `FieldMonitor.suggested_ray_mode()` heuristic: if Tx is close to the monitor plane, recommend `'2d'`; if Tx is far above/below, recommend `'3d'` with hemisphere sampling.
3. Optimize `intersect_and_scatter()`:
   - Batch early-exit for rays pointing away from the monitor
   - Use `dr.scatter_reduce` with `active` mask to skip invalid rays before scatter
4. Memory optimization: for very large ray counts (>1M), chunk the scatter to avoid peak memory spikes.

**Acceptance Criteria**:
- [ ] Test: Hemisphere sampling with N/2 rays matches full-sphere with N rays in field accuracy (within 0.5 dB) — confirms no bias from the sampling change
- [ ] Benchmark: 3D trace with hemisphere sampling + 50K rays matches 2D trace with 10K rays in both accuracy and wall time (within 2x)
- [ ] Memory: 1M rays on a 500x500 grid does not exceed 4 GB GPU memory

**Estimated effort**: 3-4 days

---

## 6. Phase Summary and Dependencies

```
Phase 1: Validate 3D on horizontal monitor       [2-3 days]  ── standalone
    │
Phase 2: Generalize Field/Grid                    [3-4 days]  ── standalone
    │
    ├──→ Phase 3: Scatter accumulation backend     [5-7 days]  ── depends on Phase 2
    │        │
    │        └──→ Phase 4: LoS + Diffraction       [6-8 days]  ── depends on Phase 2 & 3
    │              + RayDi Edge BVH integration         │
    │                                                   │
    │                 └──→ Phase 5: Optimization    [3-4 days]  ── depends on Phase 3 & 4
    │
    └──→ (Phase 1 can run in parallel with Phase 2)

Total estimated effort: 19-26 days
```

Phase 1 and Phase 2 are independent and can run in parallel. Phase 3 is the critical path for reflection. Phase 4 is the critical path for diffraction (RayDi integration is the largest single work item). Phase 5 is optional for initial correctness but important for production use.

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Scatter backend has higher variance than DDA for z-normal monitors | High | Medium | Keep DDA as the default for z-normal; scatter is fallback/required only for non-z monitors |
| UTD diffraction has hidden 2D assumptions in field evaluation | Medium | High | Phase 4 includes a thorough audit; test against 3D analytical wedge solutions |
| Ray count explosion for 3D makes optimization loops too slow | Medium | High | Phase 5 importance sampling; also consider adaptive ray count |
| Axis permutation bugs (bounds[0]/[1] mapped to wrong tangential axis) | Medium | High | Validate with LoS-only test on simple geometry — incorrect mapping produces visibly stretched/transposed patterns |
| Diffraction edge count explosion with `all_edges` | High | High | RayDi Edge BVH replaces brute-force; `sin(beta)` threshold pruning; `diffraction_state_budget` cap as fallback |
| Performance regression on existing 2D workflows | Low | High | DDA path is preserved for z-normal monitors; 3D changes are additive |

---

## 8. Relationship to PathMonitor

The `PathMonitor` design (`path_monitor_design.md`) solves a different problem: per-path structured output (CIR, angles, interaction types) at discrete receiver points. The 3D FieldMonitor work here is about **spatial field maps on axis-aligned planes (x/y/z-normal)**.

However, the two share infrastructure:
- Both need 3D receiver position generation
- Both benefit from the scatter-based accumulation model (PathMonitor already plans to use it)
- Phase 2's Field/Grid generalization is useful for both

If both are in the roadmap, implement Phase 2 (Field/Grid generalization) first as shared infrastructure.

---

## 9. Appendix: Current Code References

| Component | File | Key Lines |
|-----------|------|-----------|
| FieldMonitor class | `witwin/channel/monitors.py` | 64-152 |
| Ray generation (circle + sphere) | `witwin/channel/raygen.py` | 10-53 |
| DDA symbolic loop | `witwin/channel/trace/reflection/dda.py` | 13-198 |
| DDA entry point | `witwin/channel/trace/reflection/dda.py` | 201-334 |
| Reflection field computation | `witwin/channel/trace/reflection/field.py` | 35-602 |
| Image method reflection | `witwin/channel/trace/reflection/geometry.py` | `_reflect_point_across_plane` |
| LoS computation | `witwin/channel/trace/los.py` | 40-72 |
| Tracer plane monitor entry | `witwin/channel/trace/tracer.py` | 507-755 |
| MonitorResult dataclass | `witwin/channel/result.py` | 8-181 |
| TraceConfig | `witwin/channel/config.py` | 174-195 |
