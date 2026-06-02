# Diffraction Gradient Analysis: RSB/ISB Discontinuities

## Overview

This document analyzes gradient vanishing issues in the UTD diffraction computation, focusing on RSB (Reflection Shadow Boundary) and ISB (Incident Shadow Boundary) discontinuities, and proposes an edge sampling solution based on Reynolds Transport Theorem.

---

## Part 1: Gradient Vanishing Points

### Key Files

- `witwin/channel/trace/diffraction/` - Main diffraction field computation
- `witwin/channel/trace/diffraction/utd.py` - UTD diffraction coefficient (Kouyoumjian-Pathak)
- `sim_grad_vis.py` - Gradient visualization tool

Note: the former monolithic diffraction implementation has since been split into
`witwin/channel/trace/diffraction/`. Historical line references below refer to that
earlier monolithic layout.

---

### 1. RSB/ISB Hard Validity Mask (CRITICAL)

**Location**: historical monolithic diffraction implementation, lines 115-126

```python
# Line 115-116
n_pi = wedge_n * dr.pi
valid = (phi_prime >= 0) & (phi_prime <= n_pi) & (phi >= 0) & (phi <= n_pi) & (s > 0.05)

# Line 126 - Hard discontinuity
a_pair = dr.select(valid, a_pair, mi.Complex2f(0, 0))
```

**Problem**:
- When `phi` crosses 0 or `n*pi` (RSB), field drops to zero instantly
- When `phi_prime` crosses 0 or `n*pi` (ISB), field drops to zero instantly
- `dr.select()` is a hard branch with no gradient across the boundary

**Physical meaning**:
- **ISB** (`phi_prime` boundary): Shadow region where incident ray cannot reach the edge
- **RSB** (`phi` boundary): Shadow region where scattered ray would hit the wedge face

---

### 2. Sign Function Discontinuity

**Location**: historical monolithic diffraction implementation, lines 100 and 110

```python
# Line 100 - Incident angle sign determination
phi_prime = phi_prime * (-dr.sign(-dr.dot(ki_proj, n0)))

# Line 110 - Scattering angle sign determination
phi = phi * (-dr.sign(dr.dot(ko_proj, n0)))
```

**Problem**:
- `dr.sign(x)` has undefined derivative at `x=0`
- Returns -1, 0, or +1 with no smooth transition
- When the projected direction is parallel to the face normal, gradient vanishes

---

### 3. Acos Clipping at Boundaries

**Location**: historical monolithic diffraction implementation, lines 99 and 109

```python
# Line 99
phi_prime = dr.pi - dr.safe_acos(dr.clip(-dr.dot(ki_proj, to_hat), -1.0, 1.0))

# Line 109
phi = dr.pi - dr.safe_acos(dr.clip(dr.dot(ko_proj, to_hat), -1.0, 1.0))
```

**Problem**:
- `acos(x)` has derivative `-1/sqrt(1-x^2)` which goes to infinity at `x = +/- 1`
- `dr.clip()` forces the value to stay in [-1, 1], creating flat regions with zero gradient at boundaries
- Grazing incidence angles (dot product near +/-1) have vanishing gradients

---

### 4. UTD Coefficient Cotangent Poles

**Location**: `diffraction/utd.py`

```python
d1 = cot((dr.pi + dif_phi) / two_n) * f_utd(kL * a1)
d2 = cot((dr.pi - dif_phi) / two_n) * f_utd(kL * a2)
d3 = cot((dr.pi + sum_phi) / two_n) * f_utd(kL * a3)
d4 = cot((dr.pi - sum_phi) / two_n) * f_utd(kL * a4)
```

**Problem**:
- `cot(x)` has poles at `x = 0, pi, 2pi, ...`
- At shadow boundaries, the argument approaches these poles
- The `f_utd()` function is designed to cancel these poles (UTD transition function)
- But numerically, the cancellation is imperfect near the boundary
- `cot()` in `diffraction/utd.py` uses `dr.select()` to handle inf/nan, blocking gradients

---

### 5. Intentional Coefficient Detachment

**Location**: historical monolithic diffraction implementation, lines 293-298

```python
# Line 293 - Reference phase uses detached distance
phase_ref = torch.exp(-1j * k_t * d_i.detach())
coeff = a_drjit / phase_ref

# Line 298 - Coefficient is detached, only phase has gradient
edge_complex = coeff.detach() * torch.exp(-1j * k_t * d_i)
```

**Problem**:
- The DrJit-computed UTD coefficient is intentionally detached (`.detach()`)
- Only the phase term `exp(-jkd)` retains gradients
- This means gradients only flow through path length changes, not through:
  - UTD coefficient magnitude changes
  - Spreading factor changes
  - Angle-dependent amplitude variations

**Current workaround rationale**: DrJit operations don't support PyTorch autograd, so only the phase is made differentiable. But this loses most of the gradient information.

---

### 6. Fresnel Integral Branching

**Location**: `diffraction/utd.py`

```python
# 24 dr.select() calls for polynomial coefficient selection
r_coef = dr.select(cond, mi.Float(a0), mi.Float(c0)) + \
         dr.select(cond, mi.Float(a1), mi.Float(c1)) * arg + ...
```

**Problem**:
- `cond = x < 4` creates a branch at `x = 4`
- Each `dr.select()` is a hard branch with no gradient flow across
- The Fresnel integral is computed with different polynomial coefficients on each side

---

### Summary Table

| Issue | Location | Type | Severity |
|-------|----------|------|----------|
| RSB/ISB hard mask | historical monolithic diffraction line 126 | Hard discontinuity | CRITICAL |
| Sign function | historical monolithic diffraction lines 100,110 | Undefined at 0 | HIGH |
| Acos clipping | historical monolithic diffraction lines 99,109 | Infinite derivative | MEDIUM |
| Cotangent poles | `diffraction/utd.py` | Pole cancellation | HIGH |
| Coefficient detach | historical monolithic diffraction lines 293-298 | Intentional cutoff | CRITICAL |
| Fresnel branching | `diffraction/utd.py` | Branch at x=4 | LOW |

---

### Gradient Flow Diagram

```
mesh_position (torch.Tensor, requires_grad=True)
       |
       v
  edge position (pos_torch) -----> UTD via DrJit (NO GRADIENT)
       |                                    |
       v                                    v
  distance d (PyTorch)              coefficient (DETACHED)
       |                                    |
       v                                    v
  phase = exp(-jkd)  <------ multiply ---- coeff.detach()
       |
       v
  field output (HAS GRADIENT, but only from phase)
```

---

## Part 2: Edge Sampling Solution for RSB/ISB Boundaries

### Theoretical Foundation

Based on:
- [Li et al. 2018 - Differentiable Monte Carlo Ray Tracing through Edge Sampling](https://dl.acm.org/doi/10.1145/3272127.3275109)
- [Reynolds Transport Theorem](https://en.wikipedia.org/wiki/Reynolds_transport_theorem)

---

### Reynolds Transport Theorem Applied to Diffraction

For a parameter-dependent integral over a domain with moving boundaries:

```
d/dtheta integral_Omega(theta) f(x,theta) dx = integral_Omega(theta) df/dtheta dx + integral_dOmega(theta) f(x,theta) * v_n dS
                                               \______________________________/   \_____________________________________/
                                                        Interior term                         Boundary term
                                                       (AD can handle)                       (AD misses this!)
```

Where:
- `Omega(theta)` = Valid diffraction region (where `valid = True`)
- `dOmega(theta)` = RSB/ISB boundary curves
- `v_n` = Normal velocity of boundary (how fast the boundary moves when theta changes)
- `f` = Diffraction field amplitude

---

### Why AD Fails at RSB/ISB

Current code:
```python
valid = (phi >= 0) & (phi <= n_pi) & (phi_prime >= 0) & (phi_prime <= n_pi)
a_pair = dr.select(valid, a_pair, 0)  # Hard discontinuity
```

AD only computes the **interior term** (derivative of `a_pair` where `valid=True`).

It completely misses the **boundary term** - when the mesh moves, the RSB/ISB boundary sweeps across receivers, causing a sudden field change from `a_pair` to `0`.

---

### Explicit Boundary Gradient Computation

The boundary gradient CAN be computed explicitly using edge sampling:

#### Step 1: Identify Boundary Curves

The RSB/ISB boundaries are curves in the (x, y) receiver plane where:
- **RSB**: `phi(x, y) = 0` or `phi(x, y) = n*pi`
- **ISB**: `phi_prime(x, y) = 0` or `phi_prime(x, y) = n*pi`

For a single diffraction edge, these are straight lines emanating from the edge point (the shadow boundary rays).

#### Step 2: Compute Field Jump

At the boundary, the field jumps from the diffraction value to zero:
```
[f] = f_inside - f_outside = a_dif - 0 = a_dif
```

#### Step 3: Compute Boundary Velocity

When mesh parameter theta changes, the boundary moves. The normal velocity is:

```
v_n = -grad_x(phi) / |grad_x(phi)| * d(boundary_position)/dtheta
```

For RSB (phi = 0 boundary):
- The boundary is where the scattered ray is tangent to the wedge face
- When the edge moves by dtheta, the shadow line rotates
- v_n = rate of rotation * distance from edge

#### Step 4: Boundary Integral

The boundary gradient contribution is:

```
d/dtheta [integral_Omega f dx]_boundary = integral_dOmega [f] * v_n * dL
```

Where `dL` is the arc length along the boundary curve.

---

### Practical Implementation for 2D Diffraction

For a 2D receiver grid with vertical edges:

```python
def compute_rsb_isb_boundary_gradient(X, Y, edge_pos, edge_dir, n0, tx_pos, wavelength, k, d_edge_pos):
    """
    Compute boundary gradient contribution for RSB/ISB.

    Args:
        X, Y: Receiver grid (N,)
        edge_pos: Diffraction edge position (3,)
        edge_dir: Edge direction (3,)
        n0: Face 0 normal (3,)
        tx_pos: Transmitter position (3,)
        wavelength, k: Wave parameters
        d_edge_pos: Perturbation direction for edge position (3,)

    Returns:
        boundary_grad: Gradient contribution from boundary motion (N,)
    """
    # 1. Find RSB/ISB boundary lines in receiver plane
    # RSB: ray from edge along n0 direction (face 0 tangent plane)
    # ISB: ray from edge along shadow of incident ray

    # 2. For each receiver, compute signed distance to boundary
    # (positive = valid region, negative = shadow region)

    # 3. Find receivers near the boundary (within some tolerance)

    # 4. Compute field value at boundary (one-sided limit from valid region)
    # This is the jump [f] = a_dif

    # 5. Compute boundary velocity
    # v_n = d(boundary_position)/d(edge_pos) * d_edge_pos
    # For RSB: boundary rotates around edge point
    # v_n = (distance_to_edge) * (angular_velocity)

    # 6. Accumulate: boundary_grad += [f] * v_n * (1/grid_spacing)

    return boundary_grad
```

---

### Geometric Analysis: RSB Boundary Velocity

For a vertical edge at position `(ex, ey)` with face normal `n0 = (nx, ny, 0)`:

**RSB boundary line**: Ray from edge in direction `t0 = cross(n0, edge_dir)`

When edge moves by `(dx, dy)`:
1. The RSB line origin moves by `(dx, dy)`
2. If n0 also changes (due to mesh deformation), the line rotates

**Boundary velocity at receiver (rx, ry)**:
```
r = distance from edge to receiver
theta = angle from edge to receiver

v_n = dx * cos(theta_RSB - theta) + dy * sin(theta_RSB - theta)
    + r * d(theta_RSB)/d(edge_params)
```

---

### Geometric Analysis: ISB Boundary Velocity

**ISB boundary line**: Shadow of incident ray (TX -> edge), extended past the edge

The ISB line direction is: `ki_proj = normalize(edge_pos - tx_pos)` projected to 2D

When edge moves by `(dx, dy)`:
1. The ISB line origin moves by `(dx, dy)`
2. The incident direction `ki` changes slightly (second-order effect if TX is far)

**Boundary velocity**:
```
v_n = dx * cos(theta_ISB - theta) + dy * sin(theta_ISB - theta)
```

---

### Recommended Implementation Approach

```python
def compute_diffraction_field_with_boundary_grad(X, Y, rx_z, tx_pos, edge_data, wavelength, k):
    """
    Compute diffraction field with proper boundary gradients.

    Returns:
        field: Complex field values (torch.Tensor)
        interior_grad: Gradient from interior (AD-based)
        boundary_grad: Gradient from RSB/ISB boundaries (explicit)
    """
    # 1. Compute field using existing code (with AD for interior)
    field_real, field_imag, per_edge = _compute_diffraction_field_batched_grad(...)

    # 2. For each edge, compute boundary gradient contribution
    boundary_grad = torch.zeros_like(field_real)

    for i, edge in enumerate(edge_data['valid_edges']):
        # Find RSB boundary for this edge
        rsb_grad = compute_rsb_boundary_gradient(X, Y, edge, tx_pos, ...)

        # Find ISB boundary for this edge
        isb_grad = compute_isb_boundary_gradient(X, Y, edge, tx_pos, ...)

        boundary_grad += rsb_grad + isb_grad

    # 3. Total gradient = interior + boundary
    # (interior comes from autograd, boundary is explicit)

    return field, boundary_grad
```

---

### Key Insight: Why This Works

The Reynolds Transport Theorem tells us that the total derivative has two parts:
1. **Interior**: How the integrand changes at fixed points (AD handles this)
2. **Boundary**: How the domain boundary moves, carrying field values with it

For RSB/ISB:
- The field jumps from `a_dif` to `0` at the boundary
- When the boundary moves (due to mesh motion), receivers cross from valid to invalid
- This crossing creates a Dirac delta in the derivative
- Edge sampling integrates this delta along the boundary curve

This is exactly what Li et al. 2018 does for visibility discontinuities in rendering!

---

## Part 3: Comparison of Approaches

| Component | Current Approach | Soft Masking | Edge Sampling |
|-----------|-----------------|--------------|---------------|
| Interior gradient | AD on phase only | AD on phase | AD on phase |
| RSB/ISB gradient | Zero (hard mask) | Approximate (sigmoid) | Exact (boundary integral) |
| Physical accuracy | Incomplete | Biased | Correct |
| Implementation | Simple | Simple | Requires boundary detection |
| Numerical stability | Good | Good | Needs care at boundaries |

---

## Part 4: Advantages of Edge Sampling

1. **Physically correct**: Captures the true gradient at discontinuities
2. **No soft approximations**: No need for sigmoid smoothing that introduces bias
3. **Efficient**: Only need to integrate along 1D curves, not 2D domain
4. **Composable**: Can combine with interior AD gradients
5. **Unbiased**: No temperature parameter to tune

---

## Part 5: Implementation Challenges

1. **Finding boundary curves**: Need to solve `phi(x,y) = 0` and `phi_prime(x,y) = 0`
2. **Numerical integration**: Need to discretize the boundary integral
3. **Multiple edges**: Each edge contributes its own RSB/ISB boundaries
4. **3D generalization**: Boundaries become surfaces, not curves
5. **Boundary intersection**: RSB/ISB from different edges may intersect

---

## References

- [Li et al. 2018 - Differentiable Monte Carlo Ray Tracing through Edge Sampling](https://dl.acm.org/doi/10.1145/3272127.3275109)
- [Redner - Open source differentiable renderer](https://github.com/BachiLi/redner)
- [Reynolds Transport Theorem - Wikipedia](https://en.wikipedia.org/wiki/Reynolds_transport_theorem)
- [Differentiable Transient Rendering](https://arxiv.org/pdf/2206.06193) - Applies RTT to path tracing
