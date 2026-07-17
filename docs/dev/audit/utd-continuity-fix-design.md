# Design: UTD continuity fixes (R1-R6)

Date: 2026-07-16. Companion to
`fullwave-deterministic-discontinuity-audit.md`. This document is the
math-level design for the fixes; implementation lands on RayD branch
`fix/utd-continuity` (from lock commit 6047089) and worktree branch
`codex/fullwave-ground-truth`.

## F1 (fixes R2 + R5) — polarization-consistent vector field with projected scalar export

Model: every path transports the transmitter polarization as a transverse
complex 3-vector; the exported coherent scalar is the projection onto the
receiver polarization. Projection is linear, so per-path projection followed
by the existing scalar accumulation equals projecting the coherent vector
sum — the native accumulator does not change.

- Incident wave from TX with polarization p̂ along direction k̂:
  `E⃗ = A(r)·e^{-jkr}·(p̂ − (p̂·k̂)k̂)` — the *unnormalized* transverse
  projection (short-dipole sin θ pattern). This matches what the UTD path
  already does implicitly (Jones projection of p̂ onto the edge-transverse
  basis), which is why the ISB step ratio is already ~1.0 in magnitude.
- LoS: replace the isotropic scalar with the vector above; exported scalar
  `= p̂_rx·E⃗`; `path_gain = |E⃗|²` (acquires sin²θ — update pinned tests).
  Amplitude convention `1/(2k d)` (= (λ/4π)/d) is shared with RayD's
  `direct_source_field` — keep it.
- Reflection (and thin-sheet transmission): feed `reflect_field_vector` the
  transverse projection of the *actual* TX polarization instead of
  `initial_transverse_polarization(incident)`. Precise defects in the
  current helper (deterministic_field.cu:169-178): (a) TX polarization is
  hard-coded (0,0,1) instead of read from the transmitter; (b) the
  transverse projection is **normalized to unit length**, destroying the
  sin θ pattern weight that the diffraction path carries — this alone
  breaks RSB cancellation whenever sin θ ≠ 1; (c) at the axial null it
  substitutes an arbitrary unit transverse vector instead of zero. Fix:
  `p̂_t = p̂ − (p̂·k̂)k̂` unnormalized, no fallback (zero at the null is
  correct physics). The Jones machinery in `reflect_field_vector` stays.
- Diffraction: vector field already correct; replace
  `deterministic_diffraction_vector_field_kernel`'s dominant-component
  collapse with `p̂_rx·E⃗` (pass rx polarization per path via rx_id lookup or
  a uniform vector for the single-grid case). Same for
  `collapse_field_vector` used by reflection.
- `path_gain` stays total power |E⃗|²; only the coherent scalar changes.
- AD note: the deterministic field kernels have native backward/jvp
  companions (plan 07); their math must be updated in lockstep, and
  tests/ad must pass.

Consequences: the exported deterministic field becomes a fixed physical
observable (ẑ·E⃗ for the benchmark), directly comparable with Maxwell Ez —
including the axial dipole null. Cross-component interference becomes
physically meaningful. All dominant-component phase jumps disappear.

## F2 (fixes R1) — remove the 5 cm gate

`utd_math.h`: `UTD_MIN_DISTANCE` 5.0e-2 → 1.0e-4 (pure numerical guard,
matches the ray bias scale). The UTD expressions are finite at small s;
sub-wavelength observation distances are outside the asymptotic regime but
a finite value beats an exact zero by ~60 dB in practice (proven dead
sliver). Applies to both `compute_pair_field_terms` and
`compute_pair_vector_at_angles` (same constant), so MC benefits too.

## F3 (fixes R3a) — replace the endpointContinuation branch with angle clamping

Delete `endpointContinuation = selectedStationary && !tgtExt` and the
`diff_coeff_*_endpoint_continued` / `endpoint_unpaired_direct_beta` /
`compute_op_terms_*_endpoint_continued` code paths (dead after this change;
repo rule: no legacy paths). Instead clamp the observation angle into the
wedge domain with nearest-boundary wrap:

```
if (phi > n·π):  phi = (phi − n·π < 2π − phi) ? n·π : 0
```

Rationale: `wedge_exterior_mask` tests against *infinite* face planes, but
faces are finite — targets slightly past an extended face plane are
legitimately illuminated by the edge (the ray clears the finite face) and
must see a coefficient continuous with the grazing value at φ = nπ.
Genuinely blocked directions are handled by segment occlusion, not by the
coefficient. `srcExt` stays (source-side illumination test).
Same treatment in the vector path (`compute_pair_vector_at_angles`).

## F4 (fixes R3b) — take diffraction visibility rays off the wedge planes

In `paths_optix.cu` the two segments TX→Q and Q→RX start/end at Q *on* the
edge (intersection of two faces); grazing rays along a face plane
hit/miss chaotically in the watertight intersector. Fix: offset the Q-side
endpoint by `kDfrRayBias · normalize(n̂0 + n̂1)` (exterior bisector of the
wedge) before the visibility trace, in addition to the existing
along-ray bias. Face normals are already available in the state
(`state_n0/state_n1`); load them before the visibility test.

## F5 (fixes R3c + R4) — unified Fresnel endpoint truncation, no clamping hump, no vertex double-count

Replace `finite_wedge_stationary_completion_factor` (branch: 1.0 inside,
`raw/boundary·exp(−u²)` outside) with the already-existing
`finite_wedge_truncation_factor_bounds(state, tgt, k, lineMin, lineMax,
stationaryAtOrigin=false)` — the normalized Fresnel integral over the edge
extent around the (paraxial) stationary parameter:

```
factor = (0.5+0.5j) · conj( F(σ·(t_max−t*)) − F(σ·(t_min−t*)) ),
σ = sqrt(k·κ/π),  κ = ρ'²/R'³ + ρ²/R³
```

Properties: → 1 for interior stationary points on long edges; → 1/2 when
the stationary point reaches an endpoint; → the oscillating Fresnel tail
(~1/u) beyond the endpoint. C¹ in receiver position, single formula, no
inside/outside branch. This is the same estimator the MC path already uses
(`finite_wedge_truncation_factor`), so it also unifies deterministic/MC
finite-edge behavior.

Why this fixes R4: on a vertex-generated shadow-boundary ray, each of the
two edges meeting the vertex now contributes ≈ ½ · (E_i/2) = E_i/4 (the
truncation halves the transition value), and their sum ≈ E_i/2 matches the
physical corner limit to leading order — the previous code contributed
1.0 · E_i/2 *per edge* (double count → fake nulls). A true vertex
diffraction coefficient is not required for continuity; the truncated-edge
(equivalent-current style) sum is standard and continuous. Residual
accuracy at vertex rays is quantified against the Maxwell reference in
validation.

Frame consistency (RESOLVED by code reading): `pair_state_at_stationary_point`
(utd_math.h:324) already sets `state.edgePos` to the **exact unclamped
Fermat point** (`first_order_diffraction_parameter`, the unfold-rotation
formula) and `edgeLineMin/Max = [−t*, L−t*]` — bounds relative to that
point. The coefficient (angles, s, s') is therefore already evaluated at
the analytic stationary point even when off-edge, which is exactly the
equivalent-current formulation the truncation factor expects. The
implementation is a one-line swap in both `compute_pair_field_terms`
(:1188) and the vector twin:

```cpp
ComplexT<T> finiteFactor = selectedStationary
    ? finite_wedge_truncation_factor_bounds(state, tgtPos, k,
          state.edgeLineMin, state.edgeLineMax, /*stationaryAtOrigin=*/true)
    : finite_wedge_truncation_factor(state, tgtPos, k);
```

`finite_wedge_stationary_completion_factor` becomes dead code — delete.
Note the raygen keeps testing visibility at the *clamped* point (the
physical ray path) — that stays.

### F5c (E1c, supersedes F5b): closed form + odd-part corner mend

E1b's quadrature FAILED its goal and is retired: for a *straight* edge the
wedge-plane projection makes every sample share the same φ, so the KP
integrand flips on the same extension plane for all samples at once — the
integral steps exactly like the closed form (verified: 4.9 dB step,
resolution-independent). Distributing the transition per-sample requires
off-cone (Michaeli) equivalent currents, not on-cone KP. The quadrature
also introduced scalar/vector export divergence (~8-10%), adaptive-panel
FD discontinuities (real AD-test failures), and 96x cost. Revert to the
E1 closed form (stationary re-anchoring + `finite_wedge_truncation_factor_bounds`
with `stationaryAtOrigin=true`; restore `pair_state_at_stationary_point`,
`finite_edge_diffraction_point`, and the 6-arg truncation overload verbatim
from `git show HEAD:` of utd_math.h).

Corner mend (new): in the four-term assembly, split each boundary-active
term into even/odd parts about its *nearest* GO boundary b\*
(incident group: b\* = ±π of β=φ−φ′; reflection group: b\* ∈ {π, (2n−1)π}
of β=φ+φ′) by evaluating the term additionally at the mirrored argument
β′ = 2b\* − β:

```
K_odd = (K(β) − K(2b*−β))/2 · w(δ),   δ = β − b*,
w(δ)  = exp(−(δ/δ_w)²),  δ_w = 4·sqrt(2π/kL)   (locality window)
K_used = K − (1 − γ(u))·K_odd
```

γ(u) is a smooth sigmoid of the stationary-exit Fresnel coordinate
u = σ·(t* − nearest edge end) (σ from the truncation helper; u<0 interior):
γ = 1/(1+exp(u/0.1)) — γ→1 deep interior, γ→0 beyond the corner.

**E1d refinement (2026-07-17, supersedes the E1c application of the
factors):** E1c multiplied the WHOLE coefficient by the complex truncation
factor T and scaled the odd part by γ — both distort the GO-compensation
step at interior boundaries. For the 0.2 m cube edges (~2 Fresnel units
long) T has a Gibbs-type overshoot (|T|≈1.27, arg≈15° at mid-edge), and
γ<0.95 within |u|<0.3 of an end; measured interior-ISB total-field jump
degraded 0.29 dB (pre-E1 baseline) → 2.28 (E1) → 3.57 (E1c). The GO
toggle is binary and full-strength wherever the stationary point lies on
the edge, so the step-carrying component must enter with factor exactly 1.
Correct structure, per boundary-active term:

```
odd_w  = ½·(t(β) − t(2b*−β))·w(δ)          (unchanged mirror machinery)
t_used = (t − odd_w)·T + odd_w·γ
```

Identities: disc(odd_w) ≡ disc(t) across the boundary and (t − odd_w) is
continuous — so γ=1 preserves the GO-compensating discontinuity EXACTLY
regardless of T (interior boundaries recover baseline-quality
continuity), while γ→0 still removes the extension-plane step. T now
multiplies only the smooth background (finite-edge endpoint interference,
where its Gibbs phase is physical) and the slope feeds. The outer
finiteFactor multiplication is removed on the stationary path (truncation
lives inside the assembly); the MC/non-stationary path keeps the outer
`finite_wedge_truncation_factor` exactly as before. The channel_native AD
wedge kernel (`field_wedge_ad_diffraction.cu`) replicates the assembly and
must thread the same (γ, T) — E1c missed this and broke forward parity
(drift 0.008, 11 tests/ad failures).

Known residual (documented, future work = true two-variable corner
transition / generalized Fresnel or Michaeli EEC): within ~1 Fresnel zone
of a corner ray the compensation undershoots smoothly (bounded ~E_i/2 at
the corner itself) and a smooth ridge runs along the corner cone;
everywhere else the mend is exact. The MC edge-sampling estimator has the
same shared-plane disease in its integrand (γ≡1 semantics preserved
there; documented, not fixed here).

### F5b (E1b, RETIRED — kept for the record): edge-integral (EEC) evaluation

E1 verification exposed a residual the truncation factor cannot fix: with
the stationary point beyond an edge end, the analytically-continued
coefficient still carries the infinite wedge's ISB/RSB **step across the
extension planes** (planes containing TX and the edge line, beyond the
ends), scaled by the continuous truncation tail. Measured: NW-vertical
edge, line y=0.684 (LoS lit on both sides), contribution flips
1.34e-3∠−89° → 0.73e-3∠+113° in one 0.5 mm step at the plane crossing
(t*=+0.2078, 7.8 mm past the end, constant). No continuous local factor
can cancel a finite step (Stokes phenomenon); the corner wave that mends
it carries an equal-and-opposite discontinuity by construction.

Fix: evaluate the deterministic (selectStationaryPoint=1) finite-edge
contribution as the truncated equivalent-edge-current line integral

```
E(rx) = Σ_i w_i · N(t_i) · PV(t_i),   t_i ∈ [t_min, t_max]
N(t)  = sqrt(k·κ(t)/(2π)) · e^{+jπ/4},  κ = ρ'²/s'³ + ρ²/s³
```

with PV(t) the existing single-point machinery (D_op at local angles,
incident 1/(2ks')e^{-jks'}, spreading √(s'/(s(s'+s))), phase e^{-jks},
unit finite factor). Stationary-phase limit reproduces the closed form
exactly (SPA constant checked numerically); the truncated integral's
endpoint contribution IS the corner wave, so the field is continuous
through endpoint, vertex, and extension-plane crossings by construction
— and no interior sample's local shadow plane is ever crossed when the
singular parameter sits outside the interval. Composite Gauss–Legendre
8×12 = 96 samples; per-sample s,s' > UTD_MIN_DISTANCE guard. The MC
branch (selectStationaryPoint=0) is untouched; this also makes the
deterministic solver the quadrature twin of the MC edge-sampling
estimator. Partial-edge occlusion stays a single binary gate at the
clamped Q (per-sample occlusion would break AD-companion lockstep — the
companions instantiate the same templates without OptiX); documented as
a future refinement.

## F6 (R6 + Tier 3) — secondary gates and regression guards

- TX edge prefilter: raise the sample count from 4 to a spacing-based count
  (≥ ceil(edge_length/λ) capped at 16), or drop the prefilter for scenes
  below a size threshold. Binary per-edge dropping remains but with far
  smaller blind spots.
- Continuity regression test (new, deterministic-only, no fullwave data
  needed): line scans across (a) an ISB, (b) an RSB, (c) a vertex ray,
  (d) an extended-face-plane crossing, (e) the near-edge sliver (receiver
  plane cutting the geometry). Assert max adjacent-sample jump of
  |E_total| < 3 dB at λ/10 sampling and no |E| < threshold dead cells.
- Keep the per-path arrays (edge id, Q, field vector) documented as the
  debug dump surface (already exported when `export_paths=True`).

## Cross-solver policy (Phase-B audit results, 2026-07-17)

- Path solver shares the RayD code paths → F2/F3/F4/F5 fix it too. It has
  no R2 (already projects onto rx polarization via
  `field_project_complex3` — reuse that convention for the deterministic
  export). Its order-1 diffraction hardcodes tx_pol=(0,0,1) in
  `native/channel_native/rayd/diffraction.cpp:167-169` and `:303-305`
  (per-tx variant) — thread the scene TX polarization through (this also
  fixes the deterministic solver, which uses the same bridge).
- MC-basic: R1 exists as a TRANSCRIBED copy in
  `native/channel_native/kernels/diffraction.cu:314-317` using
  `utd::UTD_MIN_DISTANCE` — picks up the F2 constant change automatically
  (verify). R3a/R3c structurally unreachable (selectStationaryPoint=0,
  truncation factor). R5: LoS isotropic (los.cu:78), reflection fabricated
  unit z-pol (reflection.cu:133-138), diffraction hardcoded z-pol
  (diffraction.cu:255-257) — thread true TX polarization the same way.
  Same-family extras (documented, lower priority): φ>exterior hard reject
  (diffraction.cu:341), reflection blocker occlusion bias/no-self-face
  (reflection.cu:177).
- BDPT: diffraction is a non-UTD heuristic
  (`bdpt_connect_common.cuh:226-243`, 1/r²·1/r² with clamped wedge-angle
  scale) — R1-R6 do not apply; physically correct BDPT diffraction is
  separate future work and is documented, not fixed, in this pass.
- AD lockstep: channel_native wedge AD kernels
  (`field_wedge_ad_diffraction.cu` replicates the endpointContinuation
  selection at ~:245; `field_wedge_ad_coupled.cu` already uses the
  truncation factor) must mirror every F2/F3/F5 change or primal/AD
  lockstep tests fail.

### F5e (G1, 2026-07-17): monotone even-part truncation

Post-F metrics against Maxwell exposed a systematic deep-shadow
over-brightness: median +3.8 dB, p90 +7.7 dB over the 5856 shadow cells.
Root cause (validated by a per-path dose-response experiment,
`artifacts/fullwave-fix/final/fable_ripple_test.py`): the complex
truncation factor T carries the finite-aperture Fresnel ripple
(|T| ≈ 1.29, arg ≈ +14° at mid-edge for the 0.2 m ≈ 2-Fresnel-unit cube
edges). That ripple is the PO/Kirchhoff corner-wave pair implied by sharp
truncation — and the full-wave reference contradicts its strength:

| even-part factor | shadow gap vs Maxwell (median / p10 / p90) |
| --- | --- |
| T (complex, rippled) | +3.54 / +1.66 / +7.96 dB |
| 1 (no truncation) | +4.45 / −0.22 / +10.93 dB |
| T_mono = clamp(1−|tail_lo|−|tail_hi|, 0, 1) | **−1.11 / −4.64 / +4.45 dB** |

Fix: the even (smooth background) part multiplies the monotone real
T_mono; the odd/γ machinery is unchanged; MC path unchanged. Residual
−1.1 dB median (slightly dark: the true corner waves carry some energy) —
an explicit vertex-diffraction term with correct (sub-PO) amplitude
remains the recorded refinement.

## Known collateral

- RayD `backends/drjit` consumes the same shared header with COMMITTED
  pre-generated PTX; a device-behavior change requires PTX regen (RayD
  release chore). Verify compile-compat per-TU (nvcc --ptx) at minimum.
- `dependencies/rayd.lock.json` pins the RayD commit; update it after the
  RayD side lands (integration.h untouched → only the commit field).
- Tests pinning LoS/reflection gains or diffraction values will shift
  (sin²θ pattern, truncation factor at edges): update expected values with
  justification, do not weaken assertions.
