# ADR-072: The face term under oblique incidence

- **Status:** Accepted on the full-wave principle; the coordinator is confirming it with the owner. Kept as a
  separate commit on top of fw-d3 so it can be dropped alone (RayD commit `a21450c`; dropping it means reverting that commit on the RayD branch and re-locking; the later RayD commits do not depend on it).
- **Date:** 2026-11-11.
- **Decision ID:** `fw-d2-oblique-face-dyadic`.
- **Scope:** RayD `pair_face_operator`, `face_reflection_dyadic`, `reciprocal_face_cosines(..., sinBeta0, ...)`
  (`include/rayd/utd.h`, RayD ADR-0047 amendment "the face term under oblique incidence"). Every Channel
  diffraction caller goes through it (fw-d3 routed them all through the providers); no Channel code change.

## Context

Defect D2 of `docs/dev/fullwave-validation.md`: with 45 degree skew a concrete half-plane's reflection shadow
boundary kept a step of 26-28 % of the reflected field (0.2 % without skew, 0.2 % for PEC at any skew). The face
term evaluated its Fresnel coefficients at the edge-plane angle `(|sin phi| + |sin phi'|) / 2` and applied them
as Luebbers' diagonal `-diag(r_TM, r_TE)` on the edge-fixed (hard, soft) components. The geometric reflection it
must cancel at the boundary uses the 3D incidence cosine `sin(beta0) |sin phi'|` and the face's own TE/TM axes,
which under oblique incidence are rotated against the edge-fixed axes. A standalone harness of the shared header
(task scratch `harness/rsb.cu`) separated the two: the 3D angle alone makes the step larger (0.26 -> 0.55), the
rotated dyadic at the 3D angle removes it.

## Decision

The face coefficients are taken at `sin(beta0)` times the reciprocal mean sine, and the face operator is the mean
of the incident ray's specular reflection dyadic (in its edge-fixed basis and its mirror image's) and the
reciprocal image of the reversed observation ray's dyadic. It equals the geometric reflection on each reflection
shadow boundary, keeps the coefficient reciprocal and independent of the face labels (harness: reciprocity error
1.6e-7 median, 8e-6 p95 over 2,000 random pairs on a half-plane and a 90 degree wedge, the same as the
previous diagonal operator), and reduces to `-diag(r_TM, r_TE)` for a perfect conductor or normal incidence on
the edge.

## Evidence

RSB step left in the total field, fraction of the reflected field (`rsb_probe3.py`, 5 GHz, 10 m plate):

45 degree skew of the incidence on the edge (normal incidence: 0.002 for every face before and after, fw-d3 table):

| Face | Polarization | fw-d3 only | fw-d3 + fw-d2 |
| --- | --- | ---: | ---: |
| PEC | E along the edge / in the plane | 0.002 / 0.002 | 0.002 / 0.002 |
| concrete half-space | E along the edge / in the plane | 0.264 / 0.277 | 0.002 / 0.002 |
| 0.1 m concrete slab | E along the edge / in the plane | 0.338 / 0.239 | 0.002 / 0.002 |
| 6 mm glass | E along the edge / in the plane | 0.219 / 0.165 | 0.002 / 0.002 |
| three-layer wall (2 / 15 / 2 cm) | E along the edge / in the plane | 0.718 / 0.594 | 0.002 / 0.002 |

The same holds with the plate's winding reversed (normal into the illuminated side): 0.002-0.004.

Full-wave gate (RSB p95 jump excess, dB; target 1.5 unless the case has a tighter ceiling). "fw-d3 only" is the
build with fw-d1, fw-d3, fw-d4 and fw-d6; "+ fw-d2" adds this change:

| Metric | Before every fix | fw-d3 only | + fw-d2 | Final (all fixes) |
| --- | ---: | ---: | ---: | ---: |
| `cube_concrete_z` plane 1 RSB | 3.80 | 3.96 | 0.47 | 0.47 |
| `cube_concrete_x` plane 1 RSB | 2.13 | 2.00 | 0.05 | 0.05 |
| `cube_dielectric_z` plane 1 RSB | 3.11 | 4.56 | 0.19 | 0.19 |
| `canyon_z_3g5` plane 0 RSB | 8.61 | 5.94 | 5.55 | 5.55 |
| `canyon_x_3g5` plane 0 RSB | 4.52 | 1.22 | 1.14 | 1.14 |

The `cube_concrete` rows are the strict xfails this change flips. Every other gate metric moved by less than
the gate's resolution or within its ceiling.

Tests: RayD `test_dielectric_face_term_cancels_the_reflection_step_under_oblique_incidence` (fails before) and
the unchanged KP, reciprocity and face-label tests; the full-wave gate rows above.
