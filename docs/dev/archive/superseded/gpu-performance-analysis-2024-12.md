# GPU Performance Analysis Report

UTD Ray Tracing Simulator - GPU Parallelism and Data Transfer Analysis

**Last Updated**: 2024-12 (after optimizations)

---

## 1. Architecture Overview

```
trace/tracer.py::trace()
    |
    +-- compute_los_field()        [GPU Parallelism: 5/5]
    |
    +-- compute_reflection_field() [GPU Parallelism: 4/5]
    |
    +-- compute_diffraction_field()[GPU Parallelism: 5/5] (IMPROVED from 3/5)
```

### Data Flow (Optimized)

```
+-------------------------------------------------------------------------+
|                     Current Data Flow (Optimized)                       |
+-------------------------------------------------------------------------+
|                                                                         |
|  [Initialization - Once]                                                |
|  NumPy(CPU) --> PyTorch(CUDA) --> DrJit(GPU)    Triangle preload        |
|                                                                         |
|  [First trace() call]                                                   |
|  Field: torch.linspace(CUDA) --> mi.Float()      Zero-copy, cached       |
|  Edges: CPU project_to_2d() --> preload_edges() Cached after first call |
|                                                                         |
|  [Subsequent trace() calls]                                             |
|  Field: Cache hit                                 Zero transfer!         |
|  Edges: Cache hit                                Zero transfer!         |
|  Rays: torch.linspace(CUDA) --> mi.Float()      Zero-copy               |
|                                                                         |
|  [Results]                                                              |
|  DrJit(GPU) --> .torch()                        Zero-copy, stays on GPU |
|                                                                         |
+-------------------------------------------------------------------------+
```

---

## 2. Issue Resolution Status

### RESOLVED Issues

| Issue | Original Problem | Solution | Status |
|-------|------------------|----------|--------|
| **#1 Triangle Data** | Python list comprehension per trace() | PyTorch fancy indexing + preload at init (`trace/tracer.py:107-141`) | FIXED |
| **#3 Ray Directions** | NumPy CPU generation | torch.linspace on CUDA (`ray_generation.py`) | FIXED |
| **#4 Field Coordinates** | Recreated per trace() | Cached by (grid_size, range_x, range_y) (`trace/tracer.py:142-164`) | FIXED |
| **#5 Redundant .to('cuda')** | `.torch().to('cuda')` | `.torch()` directly returns GPU tensor (`utils.py`) | FIXED |
| **#6 Field Centers** | Missing device='cuda' | Explicit `device='cuda'` (`grid.py`) | FIXED |
| **P0 Batched Diffraction** | Serial edge loop | All (rx, edge) pairs in single kernel (historical monolithic diffraction implementation, lines 137-284) | FIXED |
| **Per-bounce dr.eval()** | Evaluated per bounce | Batched evaluation (`trace/reflection/field.py batched evaluation section`) | FIXED |
| **torch.meshgrid** | CPU round-trip | DrJit tile/repeat native (`tracer.py:234-245`) | FIXED |

### PARTIALLY FIXED Issues

| Issue | Status | Notes |
|-------|--------|-------|
| **#2 Edge Extraction** | 80% Fixed | `project_to_2d()` still CPU, but result is cached |
| **Reflection DDA** | Unchanged | Still serial `for _ in range(max_steps)` loop - algorithmic constraint |

---

## 3. Module-Level GPU Parallelism Analysis

### 3.1 LoS Computation (`trace/los.py`) - Rating: 5/5

| Operation | Parallelization | Complexity |
|-----------|-----------------|------------|
| Ray creation | Broadcast (single TX -> all RX) | O(N_rx) |
| Ray intersection | Mitsuba BVH batch query | O(N_rx * log(N_tri)) |
| Occlusion check | DrJit vectorized `dr.select` | O(N_rx) |
| Field calculation | DrJit vectorized | O(N_rx) |

**Strengths**: Fully GPU parallel, no serial bottlenecks

---

### 3.2 Reflection Computation (`trace/reflection/field.py`) - Rating: 4/5

**Parallel Structure**:
```
for bounce in range(max_bounces):        # Serial (max_bounces iterations)
    scene.ray_intersect(ray, active)     # GPU parallel (all rays)
    for _ in range(max_steps):           # Serial (2*(nx+ny) iterations)
        DDA_step()                        # GPU parallel (all rays advance together)
        dr.scatter_add()                  # GPU atomic operations
```

**Complexity**: O(max_bounces * max_steps * N_rays)

| Bottleneck | Location | Impact | Status |
|------------|----------|--------|--------|
| DDA loop | `for _ in range(max_steps)` | max_steps = 2*(nx+ny) | Algorithmic constraint |
| Triangle data preload | `trace/tracer.py:107-141` | Was CPU, now GPU preloaded | FIXED |
| `dr.eval` synchronization | Line 360 | Was per-bounce, now batched | FIXED |
| Field coordinates | `trace/tracer.py:142-164` | Was recreated, now cached | FIXED |

---

### 3.3 Diffraction Computation (`diffraction/`) - Rating: 5/5 (IMPROVED)

**Parallel Structure (Before)**:
```
for edge in diffraction_points:          # Serial! (N_edges iterations)
    compute_diffraction_from_edge()      # GPU parallel (all receivers)
```

**Parallel Structure (After)**:
```
# All (N_rx * N_edges) pairs computed in single kernel
_compute_diffraction_field_batched()     # GPU parallel (all pairs)
```

**Complexity**: O(N_edges * N_rx) fully parallel

| Improvement | Location | Impact |
|-------------|----------|--------|
| Batched computation | `_compute_diffraction_field_batched()` | N_edges speedup |
| Edge data preload | `preload_diffraction_edges()` | Zero per-trace overhead |
| Cached edge data | `trace/tracer.py:166-190` | Zero transfer on cache hit |

---

## 4. Current File Structure

```
new_simulator/
  trace/
    tracer.py         # Main Tracer class (orchestration)
    los.py            # LoS computation
    reflection.py     # Reflection with DDA
  diffraction/        # UTD diffraction (batched)
  material.py         # Fresnel/material response helpers
  diffraction/utd.py  # UTD diffraction formulas
  scene/              # Scene package: API, topology, projection, runtime
  grid.py             # Field class (moved from samples)
  ray_generation.py   # Ray direction generation (split from geometry)
  geometry.py         # Mesh utilities only
  utils.py            # DrJit-PyTorch conversion
  visualization.py    # Plotting
```

---

## 5. Remaining Optimization Opportunities

### 5.1 DDA Loop (Low priority - algorithmic constraint)

The DDA loop in `trace/reflection/field.py` still iterates `2*(nx+ny)` times:

```python
for _ in range(max_steps):  # 4000 iterations for 1000x1000 grid
    # GPU parallel within each iteration
    DDA_step()
```

**Possible approaches** (high difficulty):
- Sparse DDA: Skip empty regions
- Hierarchical grid: Multi-resolution traversal

### 5.2 Edge Extraction CPU Path (Low priority - cached)

`project_to_2d()` in `scene/projection.py` still runs on CPU, but:
- Only executed on first trace() with new calculation_height
- Result cached in `_edge_cache`
- Impact: ~10-50ms one-time cost

---

## 6. Performance Metrics

### Before Optimizations
```
Triangle data: CPU list comprehension per trace()
Ray directions: NumPy CPU
Field coords: Recreated per trace()
Diffraction: Serial edge loop (N_edges iterations)
dr.eval: Per-bounce synchronization
```

### After Optimizations
```
Triangle data: GPU preload at init (zero per-trace cost)
Ray directions: torch.linspace on CUDA (zero-copy)
Field coords: Cached (zero transfer on hit)
Diffraction: Single batched kernel (N_edges speedup)
dr.eval: Batched evaluation (single sync point)
Meshgrid: DrJit native (no CPU round-trip)
```

### Expected Speedup

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| Diffraction (8 edges) | 8 kernel launches | 1 kernel launch | ~8x |
| Triangle preload | Per trace() | Once at init | ~10x for repeated traces |
| Field coordinates | Per trace() | Cached | ~5x for repeated traces |
| Per-bounce eval | 2 syncs per bounce | 1 batched sync | ~2x |

---

## 7. Code Quality Improvements

### Renamed Functions
- `compute_diffraction_field_batched` -> `_compute_diffraction_field_batched` (internal)
- `compute_diffraction_from_edge` -> `_compute_diffraction_single_edge` (internal)

### Moved Modules
- `Field` class: legacy sample code -> `witwin/channel/monitors/field/field.py`
- Ray generation: `geometry.py` -> `ray_generation.py`

### Removed Dead Code
- `trace_reflection_cpu.py` (unused CPU fallback)

---

*Generated: 2024*
*Codebase: UTD Ray Tracing Simulator*
