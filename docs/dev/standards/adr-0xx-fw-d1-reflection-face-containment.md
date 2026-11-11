# ADR-0xx (fw-d1): The reflection in-face test is a distance

- **Status:** Accepted (full-wave principle set by the owner; coordinator task `fwfix`).
- **Date:** 2026-11-11.
- **Decision ID:** `fw-d1-reflection-face-containment`.
- **Scope:** RayD `point_inside_triangle` (`src/reflection/reflection_algorithms.cuh`, RayD ADR-0048
  amendment), used by every reflection EPC validator and by the reflection legs of the interaction chains;
  Channel's path-solver candidate test `inside_triangle3` (`native/channel/kernels/paths.cu`).

## Context

Defect D1 of `docs/dev/fullwave-validation.md`. Both containment tests accepted barycentrics down to
`-1e-4`. A barycentric is a distance in units of the opposite edge's altitude, so the slack grew with the
face: a reflection point 0.38 mm past the edge of a 4 m screen and 3.96 mm past a 40 m one still counted
(walls and screens that stand for unbounded surfaces are 40 m blocks). The diffraction of that edge
switches its reflection term at the geometric shadow boundary, so on every sample in between the total
field carried nearly the full reflected amplitude as a step. On the two-screen case the RSB p95 jump
excess was 6.25 / 4.32 dB (y / z polarization); with 1.2 m screens it was 2.4 / 0.0 dB.

## Decision

The slack is a distance measured outside each edge line, `b_i * 2A >= -slack * |e_i|`, with
`slack = 8 * 2^-23 * max(|vertex|_inf, |point|_inf)`: a few float roundings of the coordinates involved.
A point on an edge shared by two coplanar triangles is still accepted by one of them; a point past a
boundary edge is rejected at every face size. The reflection validators (RayD discovery and probe-first
containment, field EPC) and the chain legs share the RayD rule; Channel's path solver uses the same rule
in its own exhaustive candidate test. The reflection support therefore ends at the face edge, where the
edge's diffraction switches its reflection term: one geometric boundary for both.

## Evidence

Full-wave gate, before -> after (the build that added this change and the chain exterior test of fw-d4;
the two screens have no chain whose legs that test removes, so the change is this one):

| Metric | Before | After | Target |
| --- | ---: | ---: | ---: |
| `double_edge_z` plane 1 RSB p95 jump excess | 4.32 dB | -0.03 dB | 1.50 |
| `double_edge_y` plane 1 RSB p95 jump excess | 6.25 dB | 2.84 dB | 1.50 |
| `double_edge_y` plane 1 transition RMS | 2.26 dB | 1.03 dB | 1.50 |
| `double_edge_z` plane 1 transition RMS / NMSE | 1.49 / 0.083 | 1.65 / 0.029 | 1.75 / 0.10 |

The `double_edge_y` RSB excess left after this change is not a containment effect (see the fw-d4 and
fw-d2 records for its final value).

Tests: `tests/deterministic/test_fullwave_boundaries.py::test_reflection_support_ends_at_the_edge_of_a_large_face`
(40 m plate; receivers whose reflection point is 0.25-1.5 mm past the edge get no reflection; fails before);
RayD `tests/reflection/test_torch_epc_face_containment.py` (both tracers).
