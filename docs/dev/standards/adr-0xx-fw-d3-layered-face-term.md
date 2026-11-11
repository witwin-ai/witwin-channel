# ADR-0xx (fw-d3): The diffraction face term reflects like the face

- **Status:** Accepted on the full-wave principle; the coordinator is confirming it with the owner. Kept as a
  separate commit so it can be dropped (dropping it also drops fw-d2, which builds on it).
- **Date:** 2026-11-11.
- **Decision ID:** `fw-d3-layered-face-term`.
- **Scope:** RayD face-coefficient providers (`compute_pair_*_faces`, `DiffractionFaceLayers`, RayD ADR-0047
  amendment); Channel's order-1 export and wedge field (`interactions.diffraction`,
  `kernels.fields.wedge_face_layers`), the chain field (`coupled.cu`, whose `bounce_material` becomes
  `(N, W + 2)` to carry the two wedge faces' layer-store rows), double diffraction (`fields.cu`,
  `coupled.cu`) and Monte Carlo diffraction (`diffraction.cu`), all through
  `layered_reflection::WedgeFaceFresnel`.

## Context

Defect D3 of `docs/dev/fullwave-validation.md`. At a reflection shadow boundary the edge's face term must
cancel the reflected field's step exactly. Channel's specular reflection of a face uses its finite slab
(thickness from the material) or, for a multilayer material, its whole layer stack (ADR-051), while the order-1
diffraction evaluated the half-space Fresnel coefficient of one (eps, sigma) per face. The step left over,
as a fraction of the reflected field (normal incidence on the edge, `rsb_probe3.py` in the task scratch):

| Face | Before | After |
| --- | ---: | ---: |
| PEC | 0.002 | 0.002 |
| concrete half-space (30 m) | 0.002 | 0.002 |
| 0.1 m concrete slab | 0.05 | 0.002 |
| 6 mm glass | 0.38-0.42 | 0.002 |
| three-layer wall (2 / 15 / 2 cm) | 0.54-0.60 | 0.001-0.002 |

(The oblique-incidence rows are fw-d2's.)

## Decision

Every diffraction face term in Channel reflects with the coefficient Channel's specular reflection uses for that
face: `layered_reflection::face_fresnel` (slab, or stack when the material has two or more layers) in the
primal, `face_fresnel_dual` in the forward-mode passes. The order-1 export and the wedge field (and its JVP/VJP)
pass RayD the per-face slab thickness and layer-store row with the store. The chain field reads the wedge
faces' layer rows next to the reflection slots'. Double diffraction and Monte Carlo diffraction pass no layer
store and use the slab view of each face (as their reflections did before); a multilayer face there still
reflects as its first layer's slab, a limit recorded here. The face operators were formerly evaluated by the
callers and stored in the pair (`pair_face_cosines`, `slab_face_operator`); they are now evaluated inside the
pair at its stationary point from the provider, so the angle and, with fw-d2, the operator are the pair's own.

Derivatives: the face slab parameters keep their eps / sigma / gain tangents; the slab thickness and layer
stacks carry none (the wedge AD already refused multilayer material derivatives for chains).

## Evidence

Full-wave gate, before -> after this change (the build before it has fw-d1, fw-d4 and fw-d6):

| Metric | Before | After |
| --- | ---: | ---: |
| `wall_column_z` plane 0 RSB p95 jump excess | 1.66 dB | 0.70 dB |
| `wall_column_y` plane 0 RSB | 1.40 dB | 0.98 dB |
| `three_cube_z` plane 0 ISB | 2.21 dB | 1.90 dB (2.21 with the fw-d6 follow-ups; ceiling 2.90) |
| `cube_concrete_z` plane 1 RSB | 3.80 dB | 3.96 dB |
| `cube_dielectric_z` plane 1 RSB | 3.11 dB | 4.56 dB |

The gate's dielectric cubes are lit obliquely, where the face term is wrong until fw-d2 (the cube rows above
go to 0.47 and 0.19 dB with it); the 0.1 m walls of `wall_column` are lit near normal incidence and gain
directly. `beam_y` shadow NMSE rose 0.157 -> 0.243 in this build through grazing double-diffraction legs, a
fault of the boundary-side rule (fw-d6 follow-up), not of the face coefficient; with that fixed it is 0.164 (fw-d6 follow-ups).

Channel `tests/ad/test_boundary_gradient_ledger.py`: the dielectric wall's reflection boundary
(`rsb-wall-diffraction`, `rsb-rx-diffraction-flipped`) is now continuous (they were strict xfails); see fw-d6
for the remaining winding.

Tests: RayD `test_diffraction_face_layer_reflection`, `test_face_layers_forward_and_ad` (forward, JVP against
central difference, JVP/VJP duality); Channel full-wave gate rows below and the existing wedge forward-parity
and AD suites (`tests/ad`).
