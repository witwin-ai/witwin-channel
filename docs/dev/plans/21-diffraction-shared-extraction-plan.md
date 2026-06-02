# Diffraction Shared-Code Extraction Plan

Status: in progress (Phases B1/B2/A3 complete; Phases A1/A2/A4 remaining)
Owners: channel team
Related: `20-material-diffraction-rebuild.md`, `40-diffraction-path-taxonomy.md`, `42-monte-carlo-radiomap-package-overview.md`

## Motivation

The deterministic and Monte Carlo solvers carry two near-identical copies of
the same diffraction scaffolding — segment visibility, mesh containment,
triangle/ray helpers, wedge-plane geometry, UTD primitives, edge-subset
gathering, and two-side face material lookup. These were grown independently
and have drifted only stylistically; the algorithms are the same.

Targets for this refactor:

- One implementation of segment visibility, mesh containment, and triangle
  helpers, owned by `Scene` (since `Scene` already holds the
  `_rayd_scene` and `_triangle_runtime` they read).
- One pure-math module for wedge-plane geometry and UTD pole helpers under
  `witwin.channel.core.diffraction_geometry`, with no scene/state dependency.
- One pure-math module for low-level UTD primitives (`cot`, Fresnel integral,
  transition function) and Jones-operator primitives under
  `witwin.channel.core.wave_math` and `witwin.channel.core.polarization`.
- One scene-level entry point for "gather edge fields by edge index" and
  for "resolve materials for the two faces adjacent to an edge".

Scope explicitly excludes:

- The diffraction state schemas (`State` / `SK_*` vs `DiffractionStates`).
  They are correctly divergent: the deterministic schema carries
  Jones/operator/lineage fields needed for chained orders, while the MC
  schema only carries first-hit geometry plus per-face material.
- UTD evaluation kernels (`utd_accumulate_forward` vs `UTD.eval_diff_contribution`).
  Execution models differ (CUDA native vs DrJit symbolic loop).
- Slope-derivative machinery in `UTDMath` (`beta_state`, `beta_groups*`,
  `slope2d/3d`, `coeff*_angle_derivative`). Deterministic-only; MC AD goes
  through support-override tape replay, not analytic derivatives.
- Candidate edge discovery for higher-order or inserted reflections; the
  algorithms are not the same shape.
- `polarization` vector/jones/jones-operator dicts. The `{x,y,z}`,
  `{u,v}`, `{m00,m01,m10,m11}` axis-keyed dicts are functor-over-dimensions
  patterns used across 30+ files including native CUDA/SLANG kernels.
  Refactoring is a separate, multi-PR effort.

## Phase Status

| Phase | Status | Net lines |
|-------|--------|-----------|
| **B1** — `cot` / `fresnel_integral` / `f_utd` shared in `wave_math` | done | −80 |
| **B2** — Jones primitives shared in `polarization` (`jones_operator_matmul`, `jones_operator_rotator`, `jones_operator_mask_zero`, `jones_operator_mask_detach`, `fresnel_diagonal_operator`, `normalize_real_with_fallback`, `sanitize_complex`) | done | −120 |
| **A3** — Wedge-plane geometry in `channel_utils.diffraction_geometry` (`project_to_wedge_plane`, `normalize_in_wedge_plane`, `edge_angles`, `wedge_geometry`, `incident_edge_geometry`, `distance_to_cot_pole`, `cotangent_pole_safe_mask`, `slope_derivative_safe_mask`, `wedge_exterior_mask`, `WedgeGeometry` dataclass) | done | −71 |
| **A1** — Scene-owned visibility (`segment_visible`, `segment_visible_batched`, `segment_visible_fused`) | pending | −139 est. |
| **A2** — Scene-owned triangle / ray helpers (`triangle_group_id`, `triangle_canonical_prim`, `triangle_contains_point`, `intersect_rays_raw_with_prim`, `intersect_rays_with_prim`, `point_inside_closed_mesh`, `triangle_surface_intersection`, `reflected_path_visible`) | pending | −270 est. |
| **A4** — Scene-owned edge subset + face material (`gather_edge_subset`, `edge_face_materials`) | pending | −48 est. |
| **Det dedup** — Drop `UTDMath.normalize_real/sanitize_coeff/matmul_op/rotate_op/detach_op/mask_op/sampled_face_diag`; drop wedge-plane / pole helpers from `GeometrySupport` | pending | −90 est. |

Subtotal **completed**: **−271 lines**. Subtotal **remaining**: **~−547 lines**.

## What's Already in Place

### `witwin/channel/core/wave_math.py`

```python
cot(x, eps=SMALL_EPS) -> Float
fresnel_integral(x) -> Complex2f          # Boersma series
f_utd(x) -> Complex2f                      # UTD transition function
fresnel_reflection(cos_theta, eta) -> (r_te, r_tm)
complex_relative_permittivity(eta_r, sigma, omega) -> Complex2f
material_angular_frequency(wavelength) -> Float
```

MC `diffraction_utd.py` and deterministic `diffraction_impl/math.py` and
`forward.py` already import from here. The MC-local `UTD.cot/fresnel_integral/f`
were deleted in Phase B1.

### `witwin/channel/core/polarization.py`

Already-existing dict-based primitives plus the Phase B2 additions:

```python
# Phase B2 additions:
normalize_real_with_fallback(vec, fallback) -> Vector3f
sanitize_complex(coeff) -> Complex2f
jones_operator_matmul(lhs, rhs) -> operator
jones_operator_rotator(k, s_current, s_target) -> operator
jones_operator_mask_zero(operator, mask) -> operator
jones_operator_mask_detach(operator, mask) -> operator
fresnel_diagonal_operator(*, eta_r, sigma, gain, use_fresnel,
                          cos_theta, wavelength) -> operator
```

MC `apply_jones_chain` now uses these. The MC-local `UTD.normalize_real_vector/
sanitize_complex/jones_matmul/jones_rotator/detach_mask_jones/
face_reflection_diagonal` were deleted in Phase B2.

### `witwin/channel/core/diffraction_geometry.py`

```python
@dataclass class WedgeGeometry
project_to_wedge_plane(vec, edge_dir) -> Vector3f
normalize_in_wedge_plane(vec, edge_dir) -> Vector3f
distance_to_cot_pole(arg) -> Float
edge_angles(source_pos, edge_pos, edge_dir, n0, target_pos)
   -> (phi, phi_prime, s, s_prime)
wedge_geometry(source_pos, edge_pos, edge_dir, n0, target_pos) -> WedgeGeometry
incident_edge_geometry(source_pos, edge_pos, edge_dir, n0) -> (phi_prime, s_prime)
cotangent_pole_safe_mask(phi, phi_prime, wedge_n, pole_guard) -> Bool
slope_derivative_safe_mask(phi, phi_prime, wedge_n, step) -> Bool
wedge_exterior_mask(direction_from_edge, edge_dir, n0, nn) -> Bool
```

MC `setup_edge_geometry` and `diffraction_coefficients` now use these. The
MC-local `DiffractionGeometry` class was deleted in Phase A3.

## Remaining Phases

### Phase A1 — Scene-owned visibility

Move `DiffractionScene.segment` / `batched` / `fused` / `reflected_path` (and
the deterministic `Geo.segment_visible` / `segment_visible_batched` /
`fused_visibility` / `reflected_path_support`) into `Scene` methods:

```python
class Scene:
    def segment_visible(self, start_pos, end_pos, *,
                        ignore_prim_idx=None,
                        ignore_surface_group_idx=None,
                        ignore_structure_idx=None,
                        max_ignored_hits=4) -> Bool: ...

    def segment_visible_batched(self, starts, ends) -> tuple[Bool, ...]: ...

    def segment_visible_fused(self, *, source_pos, diff_point,
                              diff_point_offset, target_pos,
                              target_valid) -> tuple[Bool, Bool, Bool, Bool]: ...
```

Preserve verbatim: the `use_raw_ignore_loop` branch keyed on `point_grad_enabled`
/ `scene_geometry_grad_enabled`, and the `dr.flag(dr.JitFlag.Recording)`
symbolic-recording detection.

### Phase A2 — Scene-owned triangle / ray helpers

Move into `Scene`:

```python
def triangle_group_id(self, prim_idx) -> Int32
def triangle_canonical_prim(self, prim_idx) -> Int32
def triangle_contains_point(self, p, prim_idx) -> (Bool, Int32)
def intersect_rays_raw_with_prim(self, ray_origin, ray_dir, active,
                                  *, tmax=None) -> (Bool, Float, UInt32)
def intersect_rays_with_prim(self, ray_origin, ray_dir,
                              active) -> (Bool, Float, Point3f,
                                          Vector3f, UInt32)
def point_inside_closed_mesh(self, point, *, robust=False, ray_dir=None,
                              active=None) -> Bool
def triangle_surface_intersection(self, image_source, target_pos,
                                   prim_idx) -> (Bool, Point3f,
                                                  Vector3f, Int32)
def reflected_path_visible(self, image_source, target_pos, prim_idx) -> Bool
```

These pass `tri_data` internally instead of taking it as a parameter.

Replace `DiffractionScene.X(...)` and `Geo.X(...)` call sites with
`scene.X(...)`. Delete the duplicated methods from both packages.

### Phase A4 — Scene-owned edge subset + face material

```python
def gather_edge_subset(self, edge_idx, *, valid_mask=None) -> dict
def edge_face_materials(self, face0_idx, face1_idx, *, valid_mask=None,
                        default_gain=1.0) -> tuple[FaceMaterial, FaceMaterial]
```

The four inline `dr.gather(...)` blocks in `deterministic/path/diffraction_impl/builders.py`
(`tx_first`, `prefix_first`, `higher`, `inserted`) and the MC
`DiffractionEdgeSampler.gather_edge_subset` collapse to one Scene method.

`FaceMaterial` is currently in `montecarlo/path/diffraction.py` — promote it to
`channel_utils.materials` so deterministic can use it too.

### Det dedup

Once the shared helpers exist, delete from
`deterministic/path/diffraction_impl/math.py`:

- `UTDMath.normalize_real` (use `polarization.normalize_real_with_fallback`)
- `UTDMath.sanitize_coeff` (use `polarization.sanitize_complex`)
- `UTDMath.matmul_op` / `rotate_op` / `detach_op` / `mask_op`
  (use `polarization.jones_operator_matmul/rotator/mask_detach/mask_zero`)
- `UTDMath.sampled_face_diag` (use `polarization.fresnel_diagonal_operator`)
- `GeometrySupport.project_wedge_plane` / `normalize_wedge_plane` / `angles`
  / `geometry` / `incident_geometry` / `cot_pole_distance` /
  `cotangent_safe_mask` / `slope_safe_mask` / `wedge_exterior_mask`
  (use `channel_utils.diffraction_geometry`)
- `GeometrySupport` visibility / triangle / containment helpers
  (use `Scene` methods from Phases A1, A2)
- `GeometrySupport.face_material_inputs` callers switch to
  `scene.edge_face_materials` (Phase A4)

## Risk and Validation

- **Numerical equivalence**: each phase must keep `tests/integration/` and
  `tests/montecarlo/` and `tests/deterministic/` passing. Phases B1, B2,
  A3 are byte-identical — no drift expected; the remaining phases preserve
  algorithms verbatim.
- **Gradient paths**: `segment_visible` has the `use_raw_ignore_loop` branch
  keyed on `point_grad_enabled` / `scene_geometry_grad_enabled`. The
  Scene-owned version must keep that branch; do not collapse it.
- **Symbolic-recording branch**: same code path checks
  `dr.flag(dr.JitFlag.Recording)` and skips the early-out under symbolic
  loops. Preserve verbatim.
- **Backward compatibility**: do not introduce a `DiffractionScene` mixin
  or a `EdgeQuery` adapter class. The Scene methods should be plain
  methods, called as `scene.segment_visible(...)`.

## Non-Goals (unchanged)

- Unifying `DiffractionStates` (MC) with `State` / `SK_*` (deterministic).
- Unifying UTD evaluation kernels.
- Unifying candidate-edge discovery (`bvh_pairs` / cartesian dedupe vs
  `DiffractionBuilderKernel.best_edge_indices`).
- Sharing `Tx` / `Rx` / `Wave` / `Material` runtime types between
  solvers (separate concern; out of scope here).
- Refactoring `polarization` axis-keyed dicts to dataclasses (cross-cuts
  CUDA/SLANG kernels; separate PR).
