# ADR-0xx (fw-d6): A receiver exactly on a GO boundary

- **Status:** Accepted (full-wave principle set by the owner; coordinator task `fwfix`).
- **Date:** 2026-11-11.
- **Decision ID:** `fw-d6-boundary-sample`.
- **Scope:** RayD `cot_transition_product_value` and the term assembly in `include/rayd/utd.h` (RayD ADR-0047
  amendment "the coefficient exactly on a GO boundary"); every Channel caller of the shared UTD coefficient
  (order-1 export and wedge field, chains, double diffraction, Monte Carlo diffraction).

## Context

Defect D6 of `docs/dev/fullwave-validation.md`: a receiver exactly on a reflection shadow boundary was off by
half the reflected field on that sample. Channel-only probe (`d6.py` in the task scratch: a 10 m PEC or
0.1 m concrete half-plane, receivers 0, +-1e-6, +-1e-4 and +-2e-3 m from the boundary):

- Within +-1e-6 m the reflection was present (the in-face test includes the edge) while the diffraction took
  its shadow-side value, so the total was about zero instead of the lit value (|T| 3.4e-5 against 8.2e-4,
  PEC).
- The same happened at the incident boundary with the opposite sign: the LoS segment through the edge is
  blocked, and the diffraction took the lit side or a value near zero.

The RayD host harness located two faults: the one-sided limit was also given to the paired term whose
transition argument vanishes at the tie of its branch index while its cotangent stays finite (about zero on
a half-plane), so a band of about +-2e-6 rad around each boundary carried a spurious +-n sqrt(pi kL)(1+j);
and within 1e-6 rad the side came from float rounding.

## Decision

Only the term at its cotangent pole takes the one-sided limit. In the float band where the angles cannot
tell the side it takes the side the geometric tests choose for a point exactly on the boundary: the lit side
of a reflection boundary (the reflection is kept: inclusive in-face test, fw-d1) and the shadow side of an
incident boundary (the LoS segment through the edge is blocked). The total on that sample is then the lit
(respectively shadow) side's continuous value; the samples next to the boundary are unchanged except within
+-2e-6 rad, where the spurious limit is gone.

## Evidence

- RayD `tests/diffraction/test_utd_reference.py::test_a_receiver_exactly_on_a_boundary_takes_the_side_the_geometric_tests_choose`
  (fails on the previous header in all four cases).
- Channel `tests/deterministic/test_fullwave_boundaries.py::test_receiver_on_a_reflection_boundary_keeps_the_lit_total`
  and the probe table below.

Total |T| on a PEC half-plane, transmitter (-1, 0, 0.5), receivers at x = -1 near the reflection boundary
z = -0.5 (5 GHz; the lit side is below):

| Receiver offset from the boundary | -1e-4 m | -1e-6 m | 0 | +1e-6 m | +1e-4 m |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before | 8.15e-4 | 3.43e-5 | 3.68e-5 | 3.50e-5 | 8.23e-4 |
| After | 8.15e-4 | 8.25e-4 | 8.19e-4 | 8.25e-4 | 8.23e-4 |

The concrete slab and the oblique (y offset) variants behave the same (task scratch `d6_d3.txt`). The reflection
support covers the +1e-6 m sample (inside the fw-d1 slack) and the diffraction's float band covers it too, so it is
lit on both counts.

Under grazing incidence (a source within 1e-6 rad of a face plane) the incident and reflection boundaries of
that face coincide and the two terms carry one boundary; they keep one side between them, the side of the
observation angle `phi` (the cotangent's sign at `phi` on the term's branch; each term's own angle `phi -+ phi'`
can fall on either side). A double-diffraction leg between two edges of one face is grazing by construction, and the PEC beam
reference grid lies on the beam's end-face plane: with the incident and reflection terms split between the two
sides, those rows flipped sign (the D-D rows on the `y = 0.25` row doubled) and `beam_y` plane 0 shadow NMSE went
0.157 -> 0.243 (ceiling 0.165). Keeping the grazing pair on one side restores it (see the evidence table of
fw-d4 for the final numbers).

## Follow-ups found by the boundary gradient ledger

Channel's `tests/ad/test_boundary_gradient_ledger.py` moves a receiver across a dielectric wall's reflection
boundary in 1e-5 m steps. With the side rule above it still saw a jump of 0.79 of the peak field (normal facing
away from the transmitter) and interior residuals of 1.1-1.2 % against its 1 % tolerance, within about 1e-4 rad
of the boundary. Two float faults of the coefficient, both fixed in RayD (ADR-0047 follow-ups):

- **One small angle near the pole.** The cotangent and the transition argument were computed from the angle
  separately and lost the small angle to different roundings, and `cot_val`'s `|sin| < 1e-6` guard capped the
  cotangent; the product missed its finite limit by up to 8 % within 1e-4 rad. Within 1e-2 rad of a term's
  pole both now come from one `d`. Host test `test_the_coefficient_is_smooth_up_to_the_boundary` (8 failures
  before).
- **The grazing pair's side.** Once the near-pole values resolve `phi -+ phi'`, the grazing pair could split
  sides (the `beam_y` end-face row went back to 0.176 shadow NMSE); the pair now takes the side of `phi`. Host
  test `test_grazing_incidence_keeps_one_side_for_both_terms` (18 failures before).

Ledger after both (|missed| / peak at the boundary; interior residual / peak):

| Case | Before every fix | fw-d1 to fw-d6, before the follow-ups | After |
| --- | ---: | ---: | ---: |
| `rsb-rx-diffraction` (normal away from the transmitter) | 0.880 / 5.4e-3 | 0.787 / 1.09e-2 | 0.086 / 7.9e-3 |
| `rsb-rx-diffraction-flipped` | 0.570 / 3.3e-3 | 0 / 1.17e-2 | 0 / 8.7e-3 |
| `rsb-wall-diffraction` (the wall moves) | 2.393 / 8.3e-3 | 0 / (not recorded) | 0 / 6.8e-3 |

The flipped and wall cases are now in the ledger's continuous set (they were strict xfails). The 0.086 left on
the first is a single sample within micrometres of the boundary, where the reflection's in-face test and the
diffraction's float band decide their sides separately; it stays a strict xfail there. Full-wave `beam_y`
plane 0 / 1 shadow NMSE is 0.164 / 0.155 (ceilings 0.165 / 0.185).
