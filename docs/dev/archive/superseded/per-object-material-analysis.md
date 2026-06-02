# Per-Object Material Analysis: Reflection & Diffraction

## 1. Current Status Summary

**Core issue: material parameters are "globally hard-coded" rather than "per-object configurable".**

The current architecture has a clear disconnect:

| Layer | Has material info? | Actually used? |
|-------|-------------------|----------------|
| `witwin.core.Material` | Yes (`eps_r`, `mu_r`, `sigma_e`) | Declaration only |
| `witwin.core.Structure` | Yes (`material: MaterialSpec`) | Scene assembly validation only |
| `Scene` | Holds all Structures | **Does not pass materials to tracer** |
| `Tracer` | Accepts global dict | **Uses single uniform material** |
| Reflection/Diffraction | Uses global dict | **All faces share the same material** |

## 2. How Material Parameters Enter Reflection

Two modes coexist (`tracer.py:43-46`):

- **Scalar mode** (default): `reflection_coef=0.7`, all surfaces multiply by -0.7
- **Fresnel mode** (opt-in): `reflection_material={"relative_permittivity": ..., "conductivity": ..., "gain": ...}`

Fresnel mode formulas (`reflection/materials.py:27-36`):

```
eta = eps_r - j * sigma / (omega * eps_0)
r_TE = (cos_theta - sqrt(eta - sin^2(theta))) / (cos_theta + sqrt(eta - sin^2(theta)))
r_TM = (eta * cos_theta - sqrt(eta - sin^2(theta))) / (eta * cos_theta + sqrt(eta - sin^2(theta)))
R_scalar = 0.5 * (r_TE + r_TM) * gain
```

**Problem: regardless of how many objects or materials are in the scene, every reflection bounce uses the same `(eps_r, sigma, gain)` tuple.**

## 3. How Material Parameters Enter Diffraction

The UTD diffraction coefficient (Kouyoumjian-Pathak) does not directly contain material parameters. Materials enter through **R0/Rn** (Fresnel reflection coefficients for the two wedge faces):

```
D(phi, phi', n) = factor * (d1 + d2 + R0*d3 + Rn*d4)
```

R0/Rn computation (`geometry.py:447-478`):
- **Without material**: R0 = Rn = -1 (PEC, perfect electric conductor)
- **With material**: uses the single global `diffraction_material` dict

**Same problem: the two wedge faces may belong to different objects/materials, but currently share one global dict.**

## 4. Root Cause of the Disconnect

`build_structure_meshes()` (`scene/runtime.py:148`) extracts **only geometry (vertices/faces)** and **discards material**:

```python
def build_structure_meshes(scene):
    for structure in scene.structures:
        vertices, faces = geometry.to_mesh(...)
        compiled_meshes.append((to_point3f(vertices), to_vector3u(faces)))
        # material is ignored here
```

After merging all Structures into a unified Mitsuba scene, ray tracing returns `si.prim_index` as a global triangle index, but **there is no mapping table to trace a triangle back to its originating Structure and Material**.

## 5. Differentiability Requirement

**Material coefficients (eps_r, sigma_e) must be differentiable.** This is a hard requirement for the project's optimization-facing workflows.

### Why differentiability matters

- **Inverse design**: optimizing material parameters to achieve a target channel response requires gradients of the field w.r.t. eps_r and sigma_e.
- **Sensitivity analysis**: understanding how channel quality changes with material properties (e.g., wall permittivity drift) requires differentiable material evaluation.
- **Joint optimization**: scene geometry (vertex positions) is already differentiable via DrJit AD. Material parameters must participate in the same AD graph so that joint geometry+material optimization is possible.

### Current differentiability status

| Component | Differentiable? | Notes |
|-----------|----------------|-------|
| Vertex positions | Yes | DrJit AD through Mitsuba scene params |
| Fresnel formula | Yes | All ops (`complex_sqrt`, `dr.rcp`, etc.) are DrJit-differentiable |
| Material values | **No** | Passed as Python `float` via dict, never enters DrJit AD graph |
| UTD transition function | Yes | Boersma polynomial + exp/sqrt are differentiable |
| R0/Rn in UTD | Partially | Formula is differentiable, but inputs are static floats |

### What must change for differentiable materials

1. **Per-triangle material arrays must be `bk.Float` (DrJit)**, not Python float lists.
   - `eps_r_per_triangle: bk.Float` — length = total triangle count
   - `sigma_per_triangle: bk.Float` — length = total triangle count
   - These arrays must have `dr.enable_grad()` when optimization is active.

2. **`dr.gather()` preserves gradients.** Gathering material values by `prim_index` is already a differentiable operation in DrJit, so no custom backward pass is needed.

3. **Fresnel formulas already use differentiable DrJit ops** (`complex_sqrt`, `dr.rcp`, `dr.dot`, etc.). Once the input `eta_r` and `sigma` are DrJit floats with gradients attached, the entire chain from material → Fresnel coefficient → field contribution is automatically differentiable.

4. **Slang shader path**: the fused Slang accumulator (`utd_accumulate_math.slang`) currently receives R0/Rn as pre-computed values. For differentiable materials, either:
   - Pre-compute R0/Rn in DrJit (differentiable), pass them into Slang as input buffers (forward-only in Slang), and rely on DrJit AD for the material gradient; or
   - Implement the Fresnel calculation inside Slang with `slang_autograd` support (more complex but avoids the two-pass overhead).

5. **PyTorch interop**: if material parameters come from `torch.Tensor` (e.g., a learnable material embedding), they must be converted to DrJit floats with gradient bridging. `witwin.core.Material` now accepts DrJit-native differentiable scalars directly, but tensor-backed public material parameters still need a dedicated bridge or material-table API.

## 6. Required Modifications for Per-Object Materials

### 6.1 Compile-time: build triangle-to-material mapping

In `build_structure_meshes()`, record which Structure each triangle belongs to:

```
triangle_to_structure: [0,0,0,...,1,1,1,...,2,2,...]
```

And compile the per-structure material parameters into per-triangle GPU arrays:

```python
eps_r_per_triangle: bk.Float     # length = total triangles
sigma_per_triangle: bk.Float     # length = total triangles
```

These arrays must be DrJit floats to support AD.

### 6.2 Reflection path: gather material per bounce

`field.py:298` already obtains `si.prim_index`. The modification path:

```
si.prim_index -> dr.gather(eps_r_per_triangle, prim_idx) -> per-hit eps_r
si.prim_index -> dr.gather(sigma_per_triangle, prim_idx) -> per-hit sigma
```

Replace the global `reflection_material["relative_permittivity"]` in `_bounce_reflection_weight()` with per-ray gathered values.

**Affected files**:
- `reflection/materials.py` — `_bounce_reflection_weight()` signature change
- `reflection/field.py` — reflection loop needs to gather material per bounce

### 6.3 Diffraction path: per-face material for wedge faces

A diffraction edge's two faces (face_0 and face_n) may belong to different Structures with different materials:

1. Edge data structure already has `adjacent_faces` (global triangle indices)
2. R0 = Fresnel(eps_r[face_0], sigma[face_0], cos_theta_0)
3. Rn = Fresnel(eps_r[face_n], sigma[face_n], cos_theta_n)

**Affected files**:
- `geometry.py:447-478` — `_edge_face_reflection_coefficients()` takes per-face material instead of global dict
- Mixed paths (R-D, D-R) suffix/prefix reflections need the same per-hit treatment

### 6.4 Slang accumulator

`utd_accumulate_math.slang` receives R0/Rn as inputs. For per-edge materials:
- Pre-compute per-edge R0/Rn in DrJit (using per-face material gather + Fresnel formula)
- Pass the resulting R0/Rn arrays into the Slang shader as input buffers
- Gradients w.r.t. material flow back through the DrJit-side Fresnel computation

This approach avoids modifying the Slang shader internals while maintaining full differentiability of the material parameters.

### 6.5 API design

**Recommended: automatic extraction from Structure.material, zero additional API**

User-side code remains unchanged:

```python
scene = Scene(structures=[
    Structure(geometry=wall_mesh,  material=Material(eps_r=4.0, sigma_e=0.01, name="concrete")),
    Structure(geometry=glass_mesh, material=Material(eps_r=2.5, name="glass")),
    Structure(geometry=metal_mesh, material=Material(eps_r=1000, sigma_e=1e7, name="metal")),
])
tracer = Tracer(frequency=3.5e9, scene=scene)
```

Tracer automatically reads materials from `scene.structures` at init and compiles them into per-triangle material tables. Existing `reflection_material`/`diffraction_material` parameters are retained as **global overrides** — when the user does not pass them, per-structure materials are used automatically.

For differentiable workflows:

```python
# Material parameters as torch tensors for joint optimization
eps_r = torch.tensor([4.0, 2.5, 1000.0], requires_grad=True)
# Scene compiles these into per-triangle DrJit arrays with gradient bridging
```

## 7. Key Technical Challenges

| Challenge | Details | Solution |
|-----------|---------|----------|
| **Per-ray material gather** | Each ray at each bounce needs different material | `dr.gather()` by `prim_index`, well-established pattern |
| **Heterogeneous wedge faces** | One edge's two faces may have different materials | Record per-face triangle indices at edge compile time, gather separately |
| **Mixed R-D-R paths** | Each segment in a reflection-diffraction chain needs its own material | Existing `chain_prim_history` records per-bounce `prim_index`, reuse directly |
| **GPU memory** | Per-triangle material arrays | Only 2 x N_triangles floats added, negligible overhead |
| **Differentiability** | Material params must carry gradients | DrJit `bk.Float` with `dr.enable_grad()`, `dr.gather()` preserves AD graph |
| **PyTorch bridge** | Torch tensor materials for learnable params | DrJit-PyTorch gradient bridging (existing pattern in vertex optimization) |
| **Slang shader** | R0/Rn in `utd_accumulate_math.slang` | Pre-compute in DrJit, pass as input buffers; shader remains unchanged |
| **Backward compatibility** | No material specified -> same result as before | When `Material()` defaults are used (eps_r=1, sigma=0), fall back to current behavior |

## 8. Modification Scope

```
                                    Priority    Change Size
scene/runtime.py                    P0          Medium — build triangle->material mapping at compile
scene/core.py                       P0          Small  — Scene exposes material table as DrJit arrays
trace/reflection/materials.py       P0          Medium — _bounce_reflection_weight accepts per-ray material
trace/reflection/field.py           P0          Medium — reflection loop gathers material per bounce
trace/diffraction/geometry.py       P0          Medium — _edge_face_reflection_coefficients per-face material
trace/diffraction/suffix.py         P1          Medium — suffix reflections use per-hit material
trace/tracer.py                     P0          Small  — read scene material table, pass to submodules
```

## 9. Validation Strategy

1. **Regression**: without `reflection_material`, results must match current behavior exactly.
2. **Single-material**: entire scene with `Material(eps_r=5.0)` must match `reflection_material={"relative_permittivity": 5.0}`.
3. **Dual-material metal/dielectric**: one face metal (eps_r large, sigma large) should approach PEC (R ~ -1), one face glass (eps_r=2.5) should show clear transmission loss.
4. **Heterogeneous wedge**: wedge with different materials on each face — UTD terms d3 and d4 should use different R0 vs Rn.
5. **Gradient verification**: differentiate field w.r.t. eps_r, verify gradient direction is physically correct (increasing permittivity should increase reflection magnitude).
6. **Torch round-trip**: set eps_r as `torch.tensor(..., requires_grad=True)`, run forward + backward, verify `eps_r.grad` is non-zero and finite.
