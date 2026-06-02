# R-D Shadow Boundary Smoothing - Implementation Plan

## Goal
Extend the current two-stage reflection architecture to support proper R-D (Reflection-Diffraction) shadow boundary smoothing for multi-bounce reflections.

## User Choices
- **Detection Method:** Edge-Centric (Option B) - GPU friendly
- **Field Combination:** Direct Sum - UTD auto-smoothing
- **Bounce Support:** All bounces (same as max_reflections)

## Current Architecture Analysis

### Stage 1: Monte Carlo Ray Tracing (Lines 453-485)
```
TX --n_rays--> Ray-Mesh Intersection (Mitsuba GPU)
                      |
                      v
              For each hit ray:
              - prev_refl_p: reflection point
              - prev_refl_n: surface normal
              - prev_tx: source position (for image calc)
              - prev_prim_idx: triangle index
              - blocker_dist: distance to next surface
```

### Stage 2: Method of Images via DDA (Lines 486-596)
```
For each bounce b:
  For each active ray:
    1. Compute image_source = mirror(prev_tx, reflection_plane)
    2. DDA traverse grid cells along ray direction
    3. For each cell: accumulate field from image_source
    4. Update prev_* for next bounce
```

### Key Insight
The DDA loop already has `blocker_dist` which represents the **next occlusion** along the ray. This is partial secondary visibility info, but it's per-ray, not per-RX.

---

## Critical Insight: The Real Problem

After analyzing `_compute_rd_diffraction_field` (lines 141-278), I realized:

**The existing UTD code is correct!** It properly computes:
- Incident angle phi_prime from image source to edge
- Scattering angle phi from edge to RX
- UTD diffraction coefficient D (auto-handles lit/shadow transition)
- Proper spreading and phase

**The actual bug is EDGE SELECTION:**
- Current: Only selects edges of the **reflection triangle** (`get_diffraction_edges_for_triangle`)
- Correct: Should select **ALL edges that can cast shadows** from the image source

### Why Current Approach Fails
```
Image Source (S)
      |
      |  (reflection triangle edge - currently selected)
      |  /
      | /
      |/____________
      X  Reflection surface
         |
         |  <-- Other edges that SHOULD cast shadows
         |      but are NOT selected!
```

The shadow boundary is created by edges that **block the line from Image Source to RX**, not necessarily the edges of the reflection surface.

---

## Proposed Extension: Stage 2.5 - Shadow Boundary R-D

Insert a new sub-stage after DDA traversal but before bounce update:

```
Stage 2 (existing):
  DDA traversal -> accumulate reflection field

Stage 2.5 (NEW):
  Shadow Boundary Detection -> R-D Diffraction -> accumulate R-D field

Stage 2 continues:
  Update prev_* for next bounce
```

---

## Detailed Design

### Phase 1: Image Source Collection

After Stage 1 ray intersection, collect image sources per bounce:

```python
# Data structure to store image sources per bounce
image_sources_per_bounce = []  # List of (image_pos, amplitude, valid_mask)

for bounce in range(max_reflections):
    # After computing mirror source in DDA:
    # Collect unique image sources (cluster similar ones)
    image_sources_per_bounce.append({
        'positions': collected_image_positions,  # mi.Point3f
        'amplitudes': collected_amplitudes,       # mi.Float
        'source_triangles': triangle_indices,     # mi.UInt32
    })
```

### Phase 2: Edge-Centric R-D Computation (Revised)

**Key Realization:** We don't need explicit shadow boundary detection!

UTD's diffraction coefficient D automatically:
- Approaches 0 in the lit region (away from shadow boundary)
- Peaks at shadow boundary (Fresnel transition)
- Provides correct contribution in shadow region

**Therefore:** Just compute R-D for ALL potentially relevant edges, and UTD handles the rest.

```python
def compute_rd_for_all_edges(
    image_source,       # mi.Point3f - image source position
    source_amplitude,   # mi.Float - amplitude at image source
    X, Y, rx_z,         # Receiver grid
    all_edges,          # All diffraction edge geometry
    wavelength, k,
):
    """
    Compute R-D diffraction from image source through ALL edges.
    UTD automatically weights contributions based on geometry.
    """
    total_rd = Complex2f(0, 0)

    for edge_idx in range(n_edges):
        edge_data = get_edge_data(edge_idx)

        # Skip edges that can't contribute (geometric filter)
        if not edge_can_contribute(image_source, edge_data, rx_bounds):
            continue

        # Existing UTD computation (already correct!)
        rd_real, rd_imag = _compute_rd_diffraction_field(
            X, Y, rx_z,
            image_source, edge_data,
            wavelength, k, source_amplitude
        )

        total_rd += Complex2f(rd_real, rd_imag)

    return total_rd
```

### Phase 2.1: Edge Relevance Filter (Optimization)

To avoid O(n_edges * n_rx) for every edge, add a quick geometric filter:

```python
def edge_can_contribute(image_source, edge_data, rx_bounds):
    """
    Quick check if edge can possibly cast shadow into RX region.

    An edge contributes if:
    1. Edge is between image source and RX region (roughly)
    2. Edge's shadow cone overlaps with RX region
    """
    edge_pos = edge_data['pos']

    # Direction from image source to edge
    src_to_edge = edge_pos - image_source
    src_to_edge_dist = norm(src_to_edge)

    # Direction from image source to RX region center
    rx_center = Point3f(mean(rx_bounds_x), mean(rx_bounds_y), rx_z)
    src_to_rx = rx_center - image_source

    # If edge is behind source (relative to RX), skip
    if dot(src_to_edge, src_to_rx) < 0:
        return False

    # If edge is farther than RX region, it can still cast shadow
    # into partial RX region - include it
    return True
```

### Phase 3: R-D Diffraction Computation

For RX points near shadow boundaries, compute UTD diffraction:

```python
def compute_rd_shadow_diffraction(
    shadow_info,        # From Phase 2
    image_sources,      # Image source data
    edge_geometry,      # Diffraction edge data
    X, Y, rx_z,
    wavelength, k,
):
    """
    Compute R-D field for shadow boundary smoothing.

    Path: Image_Source -> Blocking_Edge -> RX

    Phase must be continuous with reflection path:
        phase_rd = -k * (d_tx_to_image + d_image_to_edge + d_edge_to_rx)
                 = -k * (d_reflection_path + d_image_to_edge + d_edge_to_rx)
    """

    # Only compute for RX near shadow boundary
    active = shadow_info['near_boundary']

    # Get blocking edge for each active RX
    edge_idx = shadow_info['blocking_edge_idx']
    edge_pos = gather(edge_geometry.pos, edge_idx)
    edge_dir = gather(edge_geometry.edge_dir, edge_idx)
    ...

    # UTD diffraction coefficient
    D = diffraction_coefficient_2d(phi, phi_prime, wedge_n, k, s, s_prime)

    # Spreading factor
    spreading = 1 / sqrt(s * s_prime * (s + s_prime))

    # Phase (continuous with reflection)
    phase = -k * (path_length_to_image + s_prime + s)

    return D * spreading * exp(i * phase) * source_amplitude
```

### Phase 4: Field Combination

Combine reflection and R-D fields for smooth transition:

```python
# In lit region (visible from image source):
#   field = reflection_field (full amplitude)
#
# In shadow region (blocked):
#   field = 0 (geometrically)
#
# At shadow boundary:
#   field = reflection_field * transition + rd_field
#
# The UTD diffraction naturally provides the correct transition!

a_ref_total = reflection_field + rd_diffraction_field
```

---

## Implementation Steps

### Step 1: Add Image Source Collection
**File:** `trace/reflection/field.py`
**Location:** Inside the solver bounce loop, after DDA traversal

Currently, image sources are computed per-ray inside the DDA loop body but not collected.
We need to collect representative image sources for R-D computation.

```python
# After the DDA while_loop, add:
if enable_rd_diffraction and bounce > 0:
    # Collect image sources from active rays
    # Strategy: Use mean position of rays that hit same triangle
    #           OR use clustering to find distinct image sources

    # Simple approach: weighted mean by amplitude
    active_mask = prev_ampl > EPS
    total_weight = dr.sum(dr.select(active_mask, prev_ampl, 0))

    if total_weight > EPS:
        mean_image_x = dr.sum(dr.select(active_mask, prev_tx.x * prev_ampl, 0)) / total_weight
        mean_image_y = dr.sum(dr.select(active_mask, prev_tx.y * prev_ampl, 0)) / total_weight
        mean_image_z = dr.sum(dr.select(active_mask, prev_tx.z * prev_ampl, 0)) / total_weight
        mean_image = mi.Point3f(mean_image_x, mean_image_y, mean_image_z)
        mean_ampl = total_weight / dr.sum(dr.select(active_mask, 1.0, 0))

        image_sources_this_bounce = {
            'position': mean_image,
            'amplitude': mean_ampl,
        }
```

### Step 2: Add Edge Relevance Filter
**File:** `trace/reflection/field.py` (new function)

```python
def _filter_relevant_edges(image_source, scene, rx_bounds):
    """
    Filter edges that could cast shadows from image source into RX region.

    Returns:
        List of edge indices that pass the geometric filter
    """
    relevant_edges = []
    edge_gpu = scene._diffraction_edge_gpu

    if edge_gpu is None:
        return []

    n_edges = edge_gpu['n_edges']
    rx_center = mi.Point3f(
        (rx_bounds[0][0] + rx_bounds[0][1]) / 2,
        (rx_bounds[1][0] + rx_bounds[1][1]) / 2,
        rx_bounds[2]  # rx_z
    )

    src_to_rx = rx_center - image_source
    src_to_rx_dist = dr.norm(src_to_rx)

    for i in range(n_edges):
        edge_pos = mi.Point3f(
            edge_gpu['pos'].x[i],
            edge_gpu['pos'].y[i],
            edge_gpu['pos'].z[i]
        )
        src_to_edge = edge_pos - image_source

        # Edge should be roughly between source and RX
        if scalar(dr.dot(src_to_edge, src_to_rx)) > 0:
            relevant_edges.append(i)

    return relevant_edges
```

### Step 3: Replace R-D Computation in Main Loop
**File:** `trace/reflection/field.py`
**Location:** Bounce-loop R-D traversal block (current R-D code)

Replace the current flawed R-D computation with corrected version:

```python
# REPLACE lines 555-596 with:
if enable_rd_diffraction and scene is not None:
    # Get RX grid bounds
    rx_bounds = ((x_min, x_max), (y_min, y_max), rx_z)

    # Get image source for this bounce (from Step 1)
    image_source = image_sources_this_bounce['position']
    source_ampl = image_sources_this_bounce['amplitude']

    # Filter relevant edges
    relevant_edges = _filter_relevant_edges(image_source, scene, rx_bounds)

    # Compute R-D for each relevant edge
    for edge_idx in relevant_edges:
        edge_data = scene.get_diffraction_edge_data(mi.Int32(edge_idx))

        rd_real, rd_imag = _compute_rd_diffraction_field(
            X_grid, Y_grid, rx_z,
            image_source, edge_data,
            wavelength, k, source_ampl
        )

        rd_results_real[bounce - 1] = rd_results_real[bounce - 1] + rd_real
        rd_results_imag[bounce - 1] = rd_results_imag[bounce - 1] + rd_imag
```

### Step 4: Ensure Proper Grid Coordinates
**File:** `trace/reflection/field.py`
**Location:** Around line 404-416

The current code computes X_grid, Y_grid only when `enable_rd_diffraction=True`.
Ensure this is always done when R-D is enabled:

```python
# Already exists but verify:
if enable_rd_diffraction and scene is not None:
    X_grid = grid.X if hasattr(grid, 'X') else None
    Y_grid = grid.Y if hasattr(grid, 'Y') else None
    if X_grid is None or Y_grid is None:
        # Compute flat grid coordinates
        idx_flat = dr.arange(mi.UInt32, n_rx)
        ix = idx_flat % nx
        iy = idx_flat // nx
        x_step = (x_max - x_min) / (nx - 1) if nx > 1 else 0
        y_step = (y_max - y_min) / (ny - 1) if ny > 1 else 0
        X_grid = mi.Float(x_min) + mi.Float(ix) * mi.Float(x_step)
        Y_grid = mi.Float(y_min) + mi.Float(iy) * mi.Float(y_step)
```

---

## Data Flow Diagram

```
                    Stage 1
                       |
         TX ----[n_rays]----> Hit Detection
                       |
                       v
    +------------------+------------------+
    |                  |                  |
    v                  v                  v
  Ray 0              Ray 1      ...     Ray N
    |                  |                  |
    +--------+---------+--------+---------+
             |
             v
         Reflection Points + Normals
             |
    +--------+--------+
    |                 |
    v                 v
  Stage 2          Stage 2.5 (NEW)
    |                 |
    v                 v
  DDA Grid      Shadow Boundary
  Traversal       Detection
    |                 |
    v                 v
  Reflection      R-D Diffraction
  Field (a_ref)   Field (a_dif_mixed)
    |                 |
    +--------+--------+
             |
             v
      a_total = a_ref + a_dif_mixed
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `witwin/channel/trace/reflection/field.py` | Main implementation: image source tracking, shadow detection, R-D computation |
| `witwin/channel/scene/` | May need helper for edge-based shadow plane computation |
| `witwin/channel/trace/tracer.py` | No changes needed (already has `enable_rd_diffraction` flag) |

---

## Verification Plan

### Test 1: Visual Comparison (Primary)
```python
# In sim.py or test script:
from witwin.channel import Box, Material, Scene, Structure, Tracer

# Setup scene with single cube
scene = Scene(
    structures=[
        Structure(
            geometry=Box(position=(0, 0, 1), size=(4, 4, 4), device="cuda"),
            material=Material(),
            name="cube",
        )
    ]
)

# TX position that creates clear shadow boundary
tx_pos = (-6, 0, 1)
monitor = FieldMonitor(
    "comparison_plane",
    axis="z",
    position=1.0,
    bounds=((-10, 10), (-10, 10)),
    grid_size=128,
)

# Run without R-D
tracer_no_rd = Tracer(frequency=2.4e9, scene=scene, enable_rd_diffraction=False)
result_no_rd = tracer_no_rd.trace(tx_pos, monitor=monitor)

# Run with R-D
tracer_with_rd = Tracer(frequency=2.4e9, scene=scene, enable_rd_diffraction=True)
result_with_rd = tracer_with_rd.trace(tx_pos, monitor=monitor)

# Plot comparison: a_ref field should show smoother shadow boundary with R-D
```

**Expected Result:**
- Without R-D: Sharp discontinuity at reflection shadow boundary
- With R-D: Smooth transition at shadow boundary

### Test 2: Cross-section Plot
```python
# Extract 1D cross-section through shadow boundary
y_slice = 0  # or another y value crossing the shadow
ref_power_no_rd = to_power_db(result_no_rd['a_ref'])[:, y_idx]
ref_power_with_rd = to_power_db(result_with_rd['a_ref'] + result_with_rd['a_dif_mixed'])[:, y_idx]

plt.plot(x_coords, ref_power_no_rd, label='Without R-D')
plt.plot(x_coords, ref_power_with_rd, label='With R-D')
plt.legend()
```

**Expected Result:**
- Without R-D: Step function at shadow boundary
- With R-D: Smooth S-curve transition

### Test 3: Phase Continuity
```python
# Check phase across shadow boundary
phase_no_rd = np.angle(result_no_rd['a_ref'].numpy())
phase_with_rd = np.angle((result_with_rd['a_ref'] + result_with_rd['a_dif_mixed']).numpy())

# Phase should not have sudden 180-degree jumps at boundary
phase_diff = np.diff(phase_with_rd, axis=0)
assert np.all(np.abs(phase_diff) < np.pi/2), "Phase discontinuity detected"
```

### Test 4: Multi-bounce
```python
# Use max_reflections=2, verify R-D applied to both bounces
tracer = Tracer(frequency=2.4e9, scene=scene,
                reflection_max_bounces=2,
                enable_rd_diffraction=True)
result = tracer.trace(tx_pos)

# Check that reflection-coupled diffraction is non-zero
assert dr.any(dr.abs(result['a_dif_mixed'].real) > 0), "R-D field should be non-zero"
```

---

## Summary: Changes Required

| File | Function/Location | Change |
|------|------------------|--------|
| `trace/reflection/field.py` | Inside the bounce loop after DDA traversal | Add image source collection |
| `trace/reflection/field.py` | New function | Add `_filter_relevant_edges()` |
| `trace/reflection/field.py` | Bounce-loop R-D traversal block | Replace with corrected R-D loop |
| `trace/reflection/field.py` | `_compute_rd_diffraction_field` | No change needed (already correct) |

---

## Potential Issues / Edge Cases

1. **Multiple Image Sources per Bounce**
   - Current: Using mean image source
   - Better: Cluster image sources, compute R-D for each cluster

2. **Edge Filtering Too Aggressive**
   - If geometric filter excludes valid edges, shadow boundaries won't smooth
   - May need to relax the filter condition

3. **Phase Sign Convention**
   - Ensure R-D phase matches reflection phase convention (exp(-ikd))
   - Current code uses correct convention

4. **Numerical Stability**
   - UTD coefficient can be large near shadow boundary
   - Fresnel transition function should handle this, but verify
