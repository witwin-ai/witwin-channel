# ADR-075: A double-diffraction leg along a sheet runs on the side the other edge sees

- **Status:** Accepted for review (coordinator task `chainperf`; requested physics fix, full-wave principle).
- **Date:** 2026-11-11.
- **Decision ID:** `fw-dd-sheet-side`.
- **Scope:** the coupled double-diffraction field kernel and its derivative companions
  (`coupled_dd_field_kernel` in `native/channel/kernels/fields.cu`, `coupled_dd_row_dual` in
  `native/channel/kernels/coupled.cu`), and RayD's UTD angle computation (`compute_edge_angles`,
  `PairInputs::sourceSide` / `targetSide`, `grazing_face_side` in `include/rayd/utd.h`; RayD ADR-0047
  amendment "the side of a sheet a grazing ray runs on").

## Context

The coincident-edge fix (fw-d5) turned the two ground-wall junctions of the `trihedral_z` corner into
270 degree wedges whose faces are the plates' outer sides. They then receive double-diffraction (D-D) legs that
run from a plate's free edge along the plate to the junction, and `trihedral_z` plane 0 shadow NMSE went from
0.161 to 0.462 (a strict xfail since). Dropping those rows gave 0.161 again.

The UTD pair measures the observation angle `phi` and the incidence angle `phi'` from face 0 and takes the side of
face 0 from the sign of the ray's component along the face-0 normal. A free plate edge is a half-plane: both faces
lie in the plate's plane, so a leg along the plate has no side of its own. Exactly on the plane the sign test gives
face 0, within float noise either face. The free edge therefore evaluated the leg on its `n0` side while the
junction received it on its exterior side, which is the plate's outer side. A Channel-only probe of plane 0
(`dd_side_probe.py` in the task scratch) counted 594,962 D-D rows, of which 211,951 graze a face at both ends;
92,055 run from a free edge to a junction face and 88,393 from a junction face to a free edge, and 61,293 /
62,843 of these were evaluated on the wrong side.

## Decision

A D-D leg that runs along one face of one of its edges runs on that face's side, and the other edge evaluates it
there:

1. The leg-1 target (`Q2`, seen from the first edge) takes the hint `grazing_face_side(Q1 - Q2, e2)`: the
   outward normal of the one face of the second edge that the leg runs along, zero when it runs along none or
   along a sheet (both faces hold it).
2. The leg-2 source (`Q1`, seen from the second edge) takes `grazing_face_side(Q2 - Q1, e1)`.
3. RayD applies a hint only to a ray within 1e-5 of the face-0 plane of the evaluating edge, which in practice is
   a sheet edge; any other ray and any zero hint keep the angle's own sign, so every other pair is bitwise
   unchanged.

The hints come from the frozen discovery geometry (`Q1`, `Q2`, edge axes and normals) and are identical in the
primal and in the JVP/VJP dual passes, which carry them as constants; the angles' derivatives are unchanged, so
the three stay in lockstep. A leg between the two free edges of one plate (both ends sheets) keeps its previous
evaluation: it has a wave on each side and one row cannot carry both.

## Evidence

Full-wave gate, before -> after (all other fixes in place, RayD `ac3f55a`):

| Metric | Before | After | Target / ceiling |
| --- | ---: | ---: | ---: |
| `trihedral_z` plane 0 shadow NMSE | 0.462 | **0.107** | 0.195 (strict xfail -> ceiling 0.130) |
| `trihedral_z` plane 0 shadow RMS | 0.338 dB | 0.297 dB | 0.60 |
| `trihedral_x` plane 1 corner NMSE | 0.633 | **0.587** | 0.60 (strict xfail -> ceiling 0.600) |
| `trihedral_x` plane 0 corner RMS | 0.794 dB | 0.711 dB | 1.05 |
| `trihedral_z` plane 1 corner NMSE | 0.023 | 0.034 | 0.035 |
| `trihedral_x` plane 1 shadow NMSE | 0.139 | 0.141 | 0.17 |
| `double_edge_y` plane 1 RSB | 2.82 dB | 2.82 dB | 1.5 (strict xfail) |
| `canyon_z` / `canyon_x` RSB and ISB | unchanged | unchanged | strict xfails |

Dropping the grazing rows instead gave 0.161 on `trihedral_z` plane 0, so the fixed rows now carry field the
full-wave reference has. The flipped entries take the headroom rule capped at their former targets (0.130 =
1.2 x 0.107 rounded up; 0.600, the target, is below 1.2 x 0.587). One metric worsens inside its ceiling:
`trihedral_z` plane 1 corner NMSE, 0.023 -> 0.034 (ceiling 0.035); its rows are the same legs at the plane near
the triple junction, where the junction wedges end and the remaining corner error of `trihedral_x` sits too.
No other case has a leg along a sheet: every other gate metric is unchanged.

Tests: `tests/deterministic/test_fullwave_boundaries.py::test_a_double_diffraction_leg_along_a_sheet_runs_on_the_junction_side`
(each grazing free-edge-to-junction row equals the row with the plate turned 1e-4 rad so that the leg runs on the
junction's side geometrically; fails before) and RayD's
`test_a_side_hint_picks_the_face_of_a_sheet_a_grazing_ray_runs_on`,
`test_grazing_face_side_names_the_one_face_a_ray_runs_along`.
