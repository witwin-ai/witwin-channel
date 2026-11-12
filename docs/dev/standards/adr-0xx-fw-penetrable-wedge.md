# ADR-0xx (fw-penetrable-wedge): The incident terms of a penetrable wedge compensate only the blocked field

- **Status:** Accepted (owner-approved item, coordinator task `chainperf2`; full-wave principle).
- **Date:** 2026-11-11.
- **Decision ID:** `fw-penetrable-wedge`.
- **Scope:** RayD's shared UTD pair (`compute_pair_vector_at_angles_faces`, `wedge_incident_transmission` in
  `include/rayd/utd.h`), the order-1 export's face providers on both tracers (`TransmittingFaceCoefficients`,
  `src/diffraction/paths.h`, `paths_optix.cu`) and the wedge field with its JVP/VJP (`src/diffraction/wedge.cu`);
  the optional `DiffractionFaceLayers::transmission_material` / `transmission_crossings` (RayD ADR-0047 amendment
  "the incident terms of a penetrable wedge"). Channel: the order-1 export call (`interactions.diffraction`), the
  differentiable wedge rows (`propagation.fields`, `kernels.fields.wedge_face_layers`), the per-face transmission
  row of the material bundle (`materials`), and the two face-layer bindings.

## Context

The deterministic GO field of a solve with transmission is LoS plus reflection plus the straight-through
transmission paths: past an obstacle the transmission paths carry `T E_i`, with `T` the product of the thin-sheet
wall operators of the faces the segment crosses (`diag(t_TE, t_TM)` of the face's layer stack in its s / p basis,
times the carrier correction `exp(+j k d cos(theta))`; RayD `transmission_sequence_kernel`). At the incident shadow
boundary (ISB) of a wedge edge the GO field therefore steps by `(I - T) E_i`, not by `E_i`. The UTD incident terms
(the two `cot((pi -+ (phi - phi')) / 2n)` terms) compensated the whole `E_i`, as for an opaque wedge, so the total
stepped by `T E_i` wherever the wedge transmits.

`canyon_x_3g5` showed it first (`chainperf` probe): the RSB edges that are also transmission-support edges carried
the whole RSB excess. A Channel-only probe of two edges (a 2 cm eps_r 4 sheet; a box of the same walls, 5 GHz,
receivers 0.1-0.2 mm either side of the ISB, `tests/deterministic/test_fullwave_boundaries.py`) gives the step
of the total over the LoS:

| Edge | Step before | Step after | Transmitted field behind it |
| --- | ---: | ---: | ---: |
| sheet | 0.880 | 0.011 | 0.886 |
| box, `max_depth` 2 (two walls crossed) | 0.687 | 0.003 | 0.686 |
| box, `max_depth` 1 (no transmission path) | 0.003 | 0.003 (bitwise) | 0 |

## Decision

**Which terms.** Only the incident terms are weighted. They compensate the step of the field that comes straight
from the source, and it is that field the transmission paths continue past the edge. The reflection terms
compensate the reflected field off face 0 or face n, which already reflects with the face's own slab or stack
coefficient (fw-d3). No GO field of the model continues a reflection past the edge (the transmission paths carry
no reflection), so the full reflected step remains and those terms stay as they are. The double-diffraction and
reflection-diffraction chains compensate fields that the model never transmits (a reflected or diffracted field
blocked by an object), so their coefficients are unchanged too.

**Which T.** `T` is the transmission paths' own wall product for the incident ray `d` continued through the
wedge at the stationary point. It is evaluated as the transmission sequence evaluates it: each crossed face's
layer-store row, TE / TM at `cos(theta) = |d . n|`, its s / p basis (`s` along `n x d`), and the carrier
correction. The geometry of the crossings is:

- A sheet (`n = 2`) is crossed once.
- A wedge with one lit face (`d . n0 < 0 < d . nn` or the reverse) is entered through the lit face and left
  through the other.
- When both or neither face is lit, the ISB is not in the exterior and nothing is weighted. At the transition the
  grazing wall's `T` is near zero, so the coefficient stays continuous.

The weighted coefficient is `-D_inc (I - T) E_i + D_0 R_0 E_i + D_n R_n E_i`. In the edge-fixed incident basis
`T` is the Jones operator `[(T u).u, (T v).u; (T u).v, (T v).v]`.

**When.** The weighting follows the GO field the solve actually has. It applies only when the solve carries the
transmission component (`"transmission"` in the components), the crossings fit its wall-crossing limit
(`max_depth`), and every crossed face's material is a thin sheet with a usable layer stack, exactly the rows the
transmission topology keeps. Otherwise the coefficient is the opaque one, bit for bit. A PEC face transmits nothing
(its slab underflows to zero), so every PEC case is unchanged.

**Derivatives.** The forward-mode twin evaluates the same wall product with the stack's dual recursion (angle and
frequency tangents). The wedge VJP is assembled from per-seed forward passes of the same function, so primal, JVP
and VJP stay in lockstep. The layer store is a constant of the wedge companions, as for the face stack (fw-d3);
the transmission component itself keeps its layer derivatives.

**Not in scope.** A boundary of the transmission paths away from any LoS edge (the second block of a two-block
crossing, a transmitted field passing another block's edge) needs a transmission-then-diffraction chain, a new
interaction class. Those boundaries stay uncompensated and their gate entries stay strict xfails (below).

## Evidence

Full-wave gate (`tests/accuracy/test_fullwave.py`), all 335 metrics diffed before / after. 80 values move, all in
the dielectric and concrete cases; no metric crosses its ceiling; no PEC case moves.

| Metric | Before | After | Ceiling / target |
| --- | ---: | ---: | ---: |
| `cube_dielectric_z` plane 0 ISB p95 excess | 4.45 dB | **-0.38 dB** | 5.10 -> 0.65 |
| `cube_concrete_z` plane 0 ISB | 0.19 dB | 0.09 dB | 1.20 -> 1.10 |
| `cube_concrete_x` plane 0 ISB | 0.46 dB | 0.24 dB | 1.50 -> 1.25 |
| `canyon_z_3g5` plane 1 ISB | 1.45 dB | 1.07 dB | 2.45 -> 2.10 |
| `canyon_x_3g5` plane 1 ISB | 1.89 dB | 1.43 dB | 1.95 |
| `canyon_z_3g5` plane 0 ISB | 2.46 dB | 2.25 dB | 1.50 (strict xfail) |
| `canyon_x_3g5` plane 0 ISB | 3.68 dB | 3.24 dB | 1.50 (strict xfail) |
| `canyon_x_3g5` plane 0 / 1 RSB | 3.27 / 2.76 dB | 2.98 / 2.32 dB | 1.50 (strict xfails) |
| `canyon_z_3g5` plane 0 / 1 RSB | 3.84 / 4.93 dB | 3.98 / 4.95 dB | 1.50 (strict xfails) |
| `cube_dielectric_z` plane 0 shadow NMSE | 1.229 | 1.326 | 1.385 |
| `cube_dielectric_z` plane 0 transition RMS | 3.69 dB | 3.50 dB | 4.05 |
| `cube_concrete_z` plane 0 shadow NMSE | 0.187 | 0.193 | 0.20 |
| `canyon_z_3g5` plane 0 shadow NMSE | 0.697 | 0.710 | 0.84 |

The ceilings this change earns are tightened under the headroom rule (jump excess + 1 dB, rounded up). The
`cube_dielectric_z` shadow NMSE rises inside its ceiling. There the weighting removes diffracted field the opaque
coefficient put into a region the thin-sheet transmission already fills; the region's RMS falls slightly.

**`canyon_x` RSB, re-diagnosed.** The probe split the 121 RSB edges that are also transmission-support edges:

| Edges (plane 0 / 1) | Before | After |
| --- | ---: | ---: |
| 62 that are also LoS edges (ISB) | 4.54 / 2.47 dB | 5.01 / 1.76 dB |
| 59 that are not | 14.6 / 13.9 dB | 14.6 / 13.6 dB |
| 471 RSB edges off transmission support | -0.72 / -0.96 dB | -0.61 / -1.00 dB |

The 59 set the p95: those are steps of a field transmitted through one block at another block's edge, the
out-of-scope class above. `canyon_x` RSB therefore stays a strict xfail with that diagnosis. The ISB share of the
excess is what this change removes: ISB 3.67 -> 3.24 dB (plane 0) and 1.89 -> 1.43 dB (plane 1).

**Tests.**
- `tests/deterministic/test_fullwave_boundaries.py::test_a_penetrable_sheet_edge_compensates_only_the_blocked_field`
  and `::test_a_penetrable_block_edge_compensates_only_the_blocked_field[1-2]`: the sheet and `max_depth` 2 cases
  fail before; the `max_depth` 1 case pins the crossing limit.
- `tests/ad/test_solver_diffraction_coupled_ad.py::test_weighted_wedge_*`: export and re-evaluated wedge parity,
  endpoint and frequency gradients against central differences, and JVP/VJP duality, on a dielectric wedge lit on
  one face with the transmission paths on. These are the deterministic and path solvers' AD rows.
- `tests/scene/test_trace_backend.py::test_weighted_diffraction_paths_match_across_backends`: OptiX and CUDA
  tracers agree.
- RayD `tests/diffraction/diffraction_wedge_test.cpp::test_face_transmission_weights_incident_terms`: the field
  changes; a crossing limit below the wedge's two walls, or a face the paths cannot cross, is bitwise unweighted;
  the JVP matches a central difference; JVP/VJP duality holds.

**Cost.** One stack evaluation per crossed wall and polarization per order-1 diffraction lane, only in solves with
transmission. Launches, syncs and copies are unchanged; the per-face transmission row is part of the cached material
bundle.

**Limits.** Monte Carlo diffraction (the stochastic radiomap) and BDPT keep the opaque coefficient; their face
providers do not describe transmission.
