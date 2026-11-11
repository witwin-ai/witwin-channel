# ADR-050: Textbook UTD coefficient, reciprocal face angles and the dipole source in diffraction

- **Status:** Accepted; accepted by the owner, 2026-11-15.
- **Date:** 2026-11-10.
- **Decision ID:** `m3-2-utd-coefficient-corrections`.
- **Scope:** every diffraction family that evaluates RayD's shared UTD coefficient: deterministic
  order-1 diffraction (cid 2), the Path solver's diffraction rows, the D leg of D-D (cid 7), Monte
  Carlo Sionna-style diffraction, and their native AD. Findings A-2, A-4 and the per-path part of C-6;
  RayD side recorded in RayD ADR-0047. The edge set is ADR-049.

## Context

The reference-accuracy suite (M0-3) compares Channel's diffraction with float64 Kouyoumjian-Pathak
(KP) UTD and Sommerfeld's exact half-plane solution. Before this record:

- A-2: a perfectly conducting 90 degree wedge was 2.2 % low at every angle and a half-plane 3-11 %
  off in the lit region, with deficits that changed with s and s' on 800 m edges.
- A-3 (shadow part): the half-plane shadow region was 23-106 % off and D_soft = D_hard at phi = 270.
- A-4: swapping transmitter and receiver changed the pair field by up to 139 % of the incoherent
  amplitude sum (Path solver, 4x4 city), while LoS + reflection was bit-identical under the swap.
- C-6 after the edge-set fix (ADR-049): merged edges still moved individual path gains by up to 1.7x.

Root causes, each reproduced on a host-compiled copy of RayD's `utd.h` (RayD
`tests/diffraction/test_utd_reference.py`) and in Channel:

1. Finite-edge aperture factors fed the linear Fresnel variable to a Fresnel integral over the phase,
   so the aperture tails fell as `1/sqrt(v)` instead of `1/v` (2.2 % loss on an 800 m edge, scaling as
   `1/sqrt(length)`): A-2.
2. The face s/p axes came from rounding noise when a ray left along a face normal: the half-plane
   phi = 270 case of A-3 and the wedge phi = 180 case.
3. Luebbers' face angles (face 0 at phi', face n at n pi - phi) are not reciprocal and change when the
   faces are relabelled; a re-triangulated mesh relabels them (ADR-049). On concrete faces relabelling
   changed a field by up to 14.7x and a source/observer swap by 9.1x: A-4 and the C-6 residual.
4. The direct source normalised the dipole's transverse projection, removing `sin(theta)`; the receiver
   side kept its projection, so the swap was not symmetric: A-4. ADR-045 had accepted this fix, but the
   RayD commit carrying it never reached RayD's history.
5. A boundary half-plane had no back-face material and was oriented by vertex order: ADR-049.

## Decision

Adopt RayD ADR-0047 (aperture factor phase argument, face axis fallback, reciprocal face cosines
`(|sin phi| + |sin phi'|) / 2` and `(|sin(n pi - phi)| + |sin(n pi - phi')|) / 2`, unnormalised dipole
projection) by re-locking to the RayD commit that carries it. Channel's own Monte Carlo diffraction
kernel (`native/channel/kernels/diffraction.cu`) uses the same `utd::reciprocal_face_cosines` for its
slab face operators, primal and Dual alike, so MC and deterministic diffraction keep one face-angle
rule. Coupled R-D/D-R legs keep their 3D incidence-angle rule; making them reciprocal is a separate
change.

## Evidence

`tests/accuracy` (strict xfails removed by this change), max error per group:

| Test | Before | After | Tolerance |
| --- | ---: | ---: | ---: |
| PEC half-plane vs Sommerfeld, soft/hard, phi = 150-320 | 3.4-11 % lit, 23-106 % shadow | 9.5e-4 | UTD-vs-exact + 1e-3 |
| 90 degree PEC wedge vs KP | 2.2 % | 9.8e-4 | 3.9e-3 |
| PEC screen winding | 1.36 | 3.5e-6 | 1e-4 |
| Reciprocity with diffraction (Path, 4x4 city) | 1.39 | 1.2e-3 | 1.7e-2 |
| Subdivision invariance, diffraction, s1 / s3 / s1 unwelded | 10.6 % / 27 % / 35 % | 5.2e-5 / 5.2e-5 / 1.8e-6 | 1e-3 |

C-6 acceptance (report section 10.6: total diffraction gain within 1e-4, same path count), 4x4 city,
32x32 grid, diffraction only, deterministic solver:

| Mesh | Rows before (0.5.0) | Rows after | Total gain before | Total gain after |
| --- | ---: | ---: | ---: | ---: |
| original | 11,151 | 12,377 | 8.21e-08 | 7.6638015e-08 |
| subdivided once, welded | 14,244 | 12,374 | 7.38e-08 | 7.6638016e-08 |
| subdivided three times, welded | 23,810 | 12,374 | 6.11e-08 | 7.6638016e-08 |
| subdivided once, unwelded | 14,245 | 12,378 | 6.72e-08 | 7.6638014e-08 |

The total gains agree to 2e-8 relative. The welded subdivision snaps vertices to a 1e-4 m grid, so
its geometry differs from the original by up to 5e-5 m; the 3 (unwelded: 1) extra or missing rows
are corner rows at building vertices with gains of 3e-24 to 4e-22, whose end-point visibility
flips with that snap. Snapping the original mesh the same way gives identical row counts (12,374)
and a largest per-row gain difference of 8.8e-4, again on corner rows (gains near 1e-16; edge rows
are near 1e-12).

The half-plane and wedge residual (about 9e-4) is the physical finite aperture of the 800 m edges:
each end removes `|F(inf) - F(v)| ~ 1/(pi v)` with `v ~ 600`.

## Consequences

- Every diffraction value moves; bit-level diffraction baselines are regenerated in the same commit
  (listed there).
- The full-wave-calibrated corner-mend constants (ADR-016/017 context) were fitted with the old
  aperture factor and are unchanged; a recalibration needs the full-wave reference.
- Corner rows (the stationary point lies past an edge end, the row is clamped to the end point) are
  now small, because the corrected aperture factor removes most of the smooth background there. Their
  value is the remainder of the even part of mirrored transition terms that are large near a GO
  boundary, so it is ill-conditioned: on the wedge fixture of
  `tests/scene/test_trace_backend.py::test_diffraction_paths_match_for_x_polarized_sources` one corner
  row (0.5 m past the end, 3.5 GHz) is 1.3e-6 with precise math and 2.6e-8 with fast math, and the
  OptiX (fast-math PTX) and CUDA tracers differ by 50x on it (the edge rows agree to 2e-5). The
  corner model (ADR-016/017, RayD's "corner-mend" weights) needs a proper corner coefficient; this
  record does not change it.
- The face angle rule is a heuristic, like Luebbers'. It agrees with Luebbers on both reflection
  shadow boundaries and with KP for perfect conductors at every angle.

## Owner acceptance

Accepted by the owner, 2026-11-15: the symmetric face-angle rule (mean of the incidence and
observation sines), the refit of the corner-weight constants (`C_BLEND`, the corner sigmoid width)
against the full-wave reference with the corrected finite-edge factor, and the resulting movement
of every diffraction value.

## Amendment (2026-11-15): edge-fixed face operator, coupled legs, corner rows

- **Corner-row parity, root cause.** The OptiX/CUDA disagreement on corner rows
  (`tests/scene/test_trace_backend.py`, x- and z-polarized) was not an ill-conditioned corner model.
  The face reflection operator aligned the outgoing face s axis to the incoming one by the sign of
  their dot product; on a corner row of the wedge fixture the two axes are exactly perpendicular, so
  rounding picked the sign of the whole face term. Precise math and `--use_fast_math` fell on opposite
  sides (1.4e-6 versus 2.6e-8 in a standalone kernel), and so did the OptiX PTX (fast math) and the
  CUDA tracer, which is why the CUDA value looked like the fast-math one. RayD now builds the operator
  in the edge-fixed bases (Luebbers' form, RayD ADR-0047 amendment); the corner row is 2.565e-8 for
  host, precise GPU and fast-math GPU alike.
- **Coupled legs.** Channel's coupled R-D, D-R and D-D field kernels (primal in `fields.cu`, duals in
  `coupled.cu`, in lockstep) evaluate their face operators with `utd::pair_face_cosines`, the same
  reciprocal rule, instead of the incident-direction cosine on both faces.
  `tests/accuracy/test_consistency.py::test_coupled_reflection_diffraction_is_reciprocal` checks the
  coupled rows alone under a transmitter/receiver swap.
- **Corner-weight refit.** The owner accepted refitting `C_BLEND` and the corner sigmoid width
  against the full-wave reference with the corrected finite-edge factor. The reference arrays
  (`artifacts/fullwave/*`, Tidy3D/Maxwell runs of the single and three cube scenes) are not on this
  machine, and regenerating them needs either the external Tidy3D service or multi-second full-wave
  GPU kernels, both outside the shared-GPU rules. The constants are therefore unchanged; the parity
  tests pass without a refit because their failure had the cause above.

