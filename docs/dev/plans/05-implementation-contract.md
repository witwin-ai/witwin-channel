# Plan 05 Implementation Contract (binding conventions)

This document fixes the shared conventions for implementing
`05-physical-scattering-transmission-plan.md` (transmission + rough-surface
scattering across deterministic, MC basic, BDPT, and the path solver).
Every implementing change MUST follow these choices exactly. Where this file
and the plan narrative differ in detail, this file wins; where this file is
silent, the plan wins.

## 1. Component identity

Five public components: `los`, `reflection`, `diffraction`, `transmission`,
`scattering`.

- `_VALID_COMPONENTS` (all four solver `config.py` files, `path/config.py`)
  becomes the frozenset of the five names.
- `core/path_topology.py` integer `component_id` scheme is extended:
  `0=los, 1=reflection, 2=diffraction, 3=reflection->diffraction,
  4=diffraction->reflection, 5=transmission, 6=scattering`.
- `path/result_v2.py` `InteractionType` IntFlag stays as-is
  (`REFLECTION=1, DIFFRACTION=2, TRANSMISSION=4, SCATTERING=8`); adapters map
  `component_id 5 -> TRANSMISSION` per-event sequence, `6 -> SCATTERING`.
- BDPT `component_mask` bits: `1=los, 2=reflection, 4=diffraction,
  8=transmission, 16=scattering`.
- `capabilities.py` manifest: `"components": ["los", "reflection",
  "diffraction", "transmission", "scattering"]`.
- Component power reporting uses a mutually exclusive `path_class` so mixed
  paths are never double counted. Classification priority (highest wins):
  `scattering > diffraction > transmission > reflection > los`.

## 2. Electromagnetic conventions (operational)

- Time factor `e^{+j w t}`, propagation `e^{-j k r}` (matches
  `core/field_state.py::PHASE_CONVENTION`).
- Complex relative permittivity `eps = eps_r' - j*sigma_e/(w*eps0)`; complex
  `mu_r` allowed but v1 materials expose real `mu_r`.
- `k_m = k0*sqrt(eps_m*mu_m)`, passive branch: `Re(k_m) >= 0, Im(k_m) <= 0`.
  The complex sqrt used everywhere (oracle, torch, CUDA) must implement this
  branch explicitly, not rely on library defaults.
- `k_z,m = sqrt(k_m^2 - k_par^2)` with the same branch rule.
- Admittances: `Y_TE = k_z/(w*mu)`, `Y_TM = w*eps/k_z` (absolute eps/mu).
- Interface amplitudes: `r = (Y1-Y2)/(Y1+Y2)`, `t = 2*Y1/(Y1+Y2)`.
- Powers: `R=|r|^2`, `T = Re(Y2)/Re(Y1)*|t|^2`, `A = 1-R-T`. Lossless
  interfaces must satisfy `|A| < 1e-6` in float64, `< 1e-4` in float32; larger
  violations are errors, never silently clamped.
- Layer stack: transfer matrix per plan section 5.1 is the *oracle* form
  (complex128 CPU). Production (torch/CUDA float32) uses the numerically
  stable scattering-matrix (Redheffer) or scaled admittance recursion and must
  match the oracle to 2e-5 relative in the normal-condition domain.
- Local s/p basis: `s = normalize(n x d)` with deterministic fallback axis at
  normal incidence (reuse the existing `stable_perp_basis` /
  `orthogonal_transverse` behavior); `p = s x d`. TM sign convention follows
  from the boundary conditions above; never hand-flip signs to match a
  reference formula.

## 3. Material ABI v3

`MATERIAL_ABI_VERSION = 3`. `MaterialStore` keeps every existing v2 field
(so legacy kernels keep working) and adds:

- CSR layers (`M` materials, `L` total layers):
  `layer_offset int32[M]`, `layer_count int32[M]`,
  `layer_thickness_m f32[L]`, `layer_eps_r f32[L]`, `layer_sigma_e f32[L]`,
  `layer_mu_r f32[L]`.
- Roughness (front surface only in v1):
  `rough_sigma_h_m f32[M]`, `rough_corr_x_m f32[M]`, `rough_corr_y_m f32[M]`,
  `rough_axis_rad f32[M]`. `sigma_h == 0` means smooth.
- `geometry_mode_id int32[M]`: `0=thin_sheet`, `1=closed_volume` (v1 solvers
  reject 1 with a clear error).
- `scatter_model_id int32[M]`: `0=smooth/none`, `1=kirchhoff_ensemble`.
- New material model id: `PHYSICAL_SURFACE_MODEL_ID = 4`.
- Legacy scalar fields (`eps_r`, `sigma_e`, `mu_r`, `thickness_m`) for a
  `PhysicalSurface` are populated from layer 0; multilayer materials set
  metadata flag `legacy_scalar_approximation=True`.
- The store gains a `layers_valid`/roughness section in `cache_token` hashing.

Python API (in `core/materials.py`), per plan section 8.1:

```python
DebyeModel(eps_inf, delta_eps, tau_s, sigma_dc=0.0)
TabulatedPermittivity(frequency_hz, eps_real, eps_imag)  # linear interp
Layer(thickness_m, eps_r=..., sigma_e=0.0, mu_r=1.0, eps_model=None)
Roughness(rms_height_m, corr_length_x_m, corr_length_y_m,
          principal_axis_rad=0.0, correlation="gaussian")
PhysicalSurface(layers, geometry_mode="thin_sheet",
                roughness_front=None, roughness_back=None, name=...)
PhaseScreen(height, height_scale_m, height_offset_m=0.0, realization_id=0,
            mode="realization_coherent", correlation=None,
            quadrature_tolerance=1e-4)
SurfaceAssignment(material, phase_screen=None)
```

`Layer.complex_eps(frequency_hz)` evaluates the dispersion model (constant,
Debye, tabulated). `Dielectric` remains and compiles as a single-layer
`PhysicalSurface` equivalent; `Dielectric.gain` is deprecated for physics
(kept, documented as calibration-only). Missing roughness parameters mean
smooth surface; never guess roughness defaults.

## 4. thin_sheet transmission contract

New specular transmission event (`TRANSMIT_SPECULAR`).

- Jones operator `diag(t_TE_stack, t_TM_stack)` in the incident s/p basis;
  outgoing direction equals incident direction (parallel-plate exit).
- Exit point: `x_e = x_i - d_total*n_in + (sum_l d_l*tan(theta_l)) * u_par`
  where `n_in` is the mean-plane normal flipped toward the incident side and
  `u_par` is the normalized tangential component of the incident direction;
  `theta_l` from Snell with phase index `Re(k_l)/k0` per layer.
- `t_stack` is defined interface-to-interface and already contains the
  interior `k_z*d` phase and absorption. Because the exit point is displaced
  laterally by `dx_par`, the transverse phase `exp(-j*k_par*|dx_par|)` with
  `k_par = k0*sin(theta_i)` MUST be applied additionally.
- `geometric_length_m` (delay, localization) includes the physical jump
  `||x_e - x_i||`; the free-space carrier phase `e^{-j k0 L}` is accumulated
  over exterior segments ONLY (interior handled by `t_stack` + transverse
  term). `delay_s = geometric_length_m/c0` in v1 (narrowband; metadata notes
  `group_delay: "geometric"`).
- Decisive unit test: a vacuum layer (`eps_r=1, sigma_e=0`) thin_sheet wall
  must reproduce the no-wall complex LoS field to 1e-5 relative accuracy
  (amplitude AND phase), at normal and oblique incidence.
- Exit-validity: offset exit point is re-validated (ray epsilon + primitive
  ignore); if the exit projection leaves the surface group, the path is
  invalid and counted in diagnostics, never teleported.
- Hit offsets use `max(|position|*1e-6, scene_diagonal*1e-6, 1e-6 m)` scale
  logic instead of a single fixed epsilon wherever new offsets are introduced.

## 5. BDPT throughput contract

The Complex3 Jones field (`field_real/imag[N,3]`) is the single authoritative
amplitude carrier; connection contributions must use it exclusively (already
true for endpoint connections). The scalar `throughput_real/imag` becomes a
real-valued diagnostic amplitude proxy: at specular events multiply by the
AMPLITUDE `sqrt(R_eff)` (not the power `R_eff`), at transmission by
`sqrt(T_eff)`. It may only be used for event/RR probabilities, never in
contributions. Fix `kernels/bdpt_subpaths.cu` accordingly and document at the
struct definition.

Event/measure rules (plan section 7.1): specular reflection/transmission are
discrete (delta) events handled with discrete probabilities in MIS;
Kirchhoff scattering is a continuous solid-angle density with forward AND
reverse PDFs evaluated from the same table with swapped arguments.

## 6. Rough-surface scattering (v1 scope)

Only Kirchhoff: `ensemble_bsdf` (production, incoherent power) and
`realization_coherent` (phase-screen patch integral, reference/deterministic).
No SPM, no Beckmann microfacet, no displacement geometry. RayD geometry is
never modified; heights only enter complex phase.

- Gaussian correlation `C(x,y) = sigma_h^2 exp[-(x/lx)^2 - (y/ly)^2]`.
- Coherent specular attenuation `C_r = exp[-2*(k_z1*sigma_h)^2]` with
  `k_z1 = k0*cos(theta_i)`; coherent reflection Jones = `r_q_stack * C_r`.
- Diffuse budget `R_diff_q = max(0, R_bar_q - |r_q*C_r|^2)` where `R_bar_q`
  is the smooth-stack reflectance. Transmission budget stays the smooth-stack
  `T_bar_q` (no diffuse transmission in v1).
- Ensemble diffuse lobe uses the Beckmann series for Gaussian correlation:
  `I(q) = pi*lx*ly*exp(-g) * sum_{m>=1} g^m/(m!*m) *
  exp[-(qx^2*lx^2 + qy^2*ly^2)/(4m)]`, `g = q_n^2*sigma_h^2`, with
  `q = k_s - k_i` decomposed in the local (rough principal axes) frame.
  The polarized kernel multiplies `|r_q_stack|^2` factors; cross-pol arises
  only from s/p frame rotation in v1. Required properties (tested, not
  assumed): (a) `sigma_h -> 0` gives zero diffuse and full coherent Fresnel;
  (b) hemispherical integral of the lobe equals `R_diff` within declared
  tolerance (small renormalization allowed only inside the tolerance,
  otherwise hard error); (c) reciprocity `f_pq(wi,wo) = f_qp(wo,wi)`;
  (d) passivity per angle/pol `R+T+A <= 1 + 1e-4`.
- Tables precomputed at scene compile per rough material at scene frequency:
  axes `cos_theta_i in (0,1], N=32` (uniform in cos), `phi_i N=16` (only for
  anisotropic `lx != ly`, else collapsed to 1), outgoing grid
  `cos_theta_o N=32 x phi_o N=64`. Channels: TE and TM co-pol power kernels,
  `R_diff` per incident bin, marginal/conditional CDFs for sampling. Sampling
  uses the CDF tables; evaluation always uses the raw high-precision table.
- Runtime implementation is PyTorch-native GPU tensor code in
  `src/witwin/channel_native/scattering/` (tables, eval, sample, pdf_fwd,
  pdf_rev). CUDA-side `em/*.cuh` in v1 covers complex/medium/fresnel/
  layer_stack only; kirchhoff/event kernels stay torch until profiling says
  otherwise. This is a deliberate, documented deviation from plan section 9
  (PyTorch-native is a repo hard requirement; tables are gather+FMA).
- Phase screen (`realization_coherent`): heights sampled from a GPU texture
  via bilinear interpolation in UV; phasor `exp(-j*q_n*h(u,v))`; deterministic
  patch quadrature accumulates the complex field over patch samples. Height
  low-pass/footprint averaging must happen on the complex phasor, never on
  heights. Fixed `(scene_seed, surface_id, realization_id)` selects the
  realization; nearby rays share the same continuous height field.
- The two modes are never summed for the same surface in one result.
- Applicability guards: `realization_coherent` requires UV; ensemble requires
  `k0*l >= ~6` (tangent-plane) and moderate slope `sqrt(2)*sigma_h/l <= 0.5`;
  out-of-domain surfaces produce a per-surface error status
  (`phase_screen_geometry_limit_exceeded` / `kirchhoff_domain_exceeded`)
  instead of silently degrading.

## 7. UV plumbing

`Structure` gains optional `uv: float32[V,2]` and `face_uv: int32[F,3]`.
`core/runtime/raydn.py::build_scene_from_structures` forwards them (replacing
the current empty tensors) when present; structures without UV keep empties.
A helper generates planar UVs for rectangle/box test structures. Native side
already carries UV end-to-end; RayD hit UV is returned by the existing
intersect ops.

## 8. CPU oracle

`src/witwin/channel_native/physics/oracle.py` (+ siblings), numpy complex128,
CPU-only, no torch dependency in the math core:

- `complex_sqrt_passive`, `medium_params(eps_r, sigma_e, mu_r, f)`,
  `fresnel_interface(cos_theta_i, medium1, medium2)` -> `r_te,r_tm,t_te,t_tm,
  R,T,A per pol`,
- `layer_stack_rt(layers, cos_theta_i, f)` via transfer matrix (both pols),
- `refraction_direction`, `coherent_attenuation(sigma_h, k_z)`,
- `kirchhoff_diffuse_lobe(...)` (Beckmann series) and
  `kirchhoff_diffuse_lobe_quadrature(...)` (brute-force 2D correlation
  quadrature) which must agree,
- `phase_screen_patch_integral(height_fn, patch, k_i, k_s, f)` direct complex
  surface quadrature,
- hemisphere integration helpers for energy tests.

Golden tests live in `tests/physics/` and cover plan section 11.1 items 1-8
plus the Beckmann-series-vs-quadrature and vacuum-slab-equivalence checks.
Tests compute oracle values at runtime (no committed NPZ needed).

## 9. Native `em/` layer (v1 scope)

`native/channel_native/em/{complex.cuh, medium.cuh, fresnel.cuh,
layer_stack.cuh}`; built on `utd::Complex/Complex3/JonesOperator` types.
`layer_stack.cuh` implements the scaled admittance recursion returning
`r_te, r_tm, t_te, t_tm` for CSR layer input. `field_transport.cuh`
`slab_fresnel` is reimplemented on top of `em/` (single-layer fast path must
reproduce current behavior bit-comparably or within 1e-6). The three
duplicated Fresnel implementations (`deterministic_field.cu`,
`reflection.cu`, `bdpt_subpaths.cu`) are migrated to the shared header in the
same change that touches each kernel, without behavior change for smooth
single-layer materials. New native ops:

- `field_transmission_sequence(...)` mirroring `field_reflection_sequence`
  (Complex3, applies stack `t` per wall + contract section 4 phases),
- `bdpt_transmitted_light_subpath_state(...)` mirroring the reflected one,
- CSR layer tensors passed material-indexed (not per-face expanded).

`build_info()` gains `material_abi_version`.

## 10. Testing / build workflow

- Env: `C:/Users/Asixa/miniconda3/envs/witwin2/python.exe` (py311).
- The native pyd is discovered from `artifacts/cmake-*`; a junction
  `artifacts/cmake-dev -> build-sionna-dev` exists in the main checkout. In a
  worktree run `cmd /c mklink /J <wt>\artifacts\cmake-dev
  E:\Code\witwin-platform\channel_native\build-sionna-dev` before pytest.
- Native rebuilds: Ninja + vcvars64, build dir under `artifacts/cmake-*`,
  `-j 4` (higher parallelism OOMs cl.exe). See memory doc; only the
  integration session rebuilds native, worktree agents never do.
- Fast contract loop:
  `pytest -q tests/physics tests/test_capabilities.py
  tests/montecarlo/basic/test_basic_materials.py tests/core/test_runtime_stores.py
  tests/core/test_public_scene.py tests/montecarlo/bdpt/test_component_maps.py
  tests/montecarlo/bdpt/test_config.py tests/deterministic/test_config.py
  tests/deterministic/test_metadata.py`.
- Every stage keeps the full suite green before the next stage starts; GPU
  parity gates (Munich, three-way) run at integration milestones.
