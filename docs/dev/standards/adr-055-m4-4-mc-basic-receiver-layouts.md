# ADR-055: MC Basic point receivers and multiple or mixed receivers

- **Status:** Accepted.
- **Date:** 2026-11-10.
- **Decision ID:** `mc-basic-receiver-layouts`.
- **Scope:** receiver layouts of `witwin.channel.montecarlo.basic`; public
  `Config.receiver_radius_m`; result layout for anything but exactly one grid.
  Numerical change (new estimator for point receivers).

## Context

S-9 and S-10: MC Basic returned an "enabled" zero for every non-LoS component at a
point receiver and silently dropped every receiver after the first grid. The 0.6.0
refusals made both errors loud; this record implements the layouts.

## Decision

**Layouts.** A scene with exactly one `ReceiverGrid` keeps the grid result layout
(`component_maps[name]` of shape `[T, cols, rows]`). Any other layout (points only,
several grids, grids and points mixed) is split into receiver blocks (each grid, and
each run of consecutive points) and every component is returned as a `[T, R]`
matrix whose columns follow `scene.receivers` (a grid in row-major receiver order,
like Deterministic); `path_gain` has the same columns and `component_power` is the
total over all columns as before.

**Estimators per component.**

- LoS and transmission are evaluated per receiver position for every block (the
  analytic LoS gain with RayD visibility; the straight-penetration wall product).
- Scattering evaluates the existing Kirchhoff area estimator at every receiver
  position.
- Reflection traces the rays once per transmitter and deposits into every block.
  Grids keep the cell flux estimator. A point receiver is a capture sphere: a
  segment whose closest approach to the point lies within radius `rho` (and before
  the segment's blocker) deposits `|E|^2 dOmega (lambda / 4 pi)^2 / (pi rho^2)`.
  The expected number of rays through the sphere's cross-section is
  `pi rho^2 / (r^2 dOmega)`, so the estimator converges to the sphere average of the
  specular path gain; the bias is second order in `rho / r` for a smooth field.
  `rho` is `Config.receiver_radius_m` when set, else 1% of the straight transmitter
  distance (at least 1 mm). The AD companions use the same deposit visitor, so
  `ad_mode` works for point receivers too.
- Diffraction at point receivers uses a capture-sphere Keller-cone sampler. Every sampling
  lane draws its edge state, edge parameter and cone azimuth exactly as the grid tape row does
  (`mc_point_diffraction_tape`, one entry per (lane, receiver)); an entry is active when the
  cone ray passes within the receiver's capture radius beyond the edge, and two RayD segment
  visibility launches test source -> point just off the wedge (face-normal bisector) ->
  receiver. The surviving entries run through the same tape accumulate kernels and their AD
  companions with a point mode: the deposit plane is the plane through the receiver normal to
  the primal cone ray (frozen, like the grid binning), and the deposit divides by the sphere
  cross-section pi rho^2 instead of the cell area. Point entries are chunked to 2^22 per launch.
  MC Basic keeps zero enumerated-path dependency.

## Evidence

`tests/montecarlo/basic/test_basic_receiver_layouts.py`: point reflection vs.
Deterministic within the seed spread; two grids equal the two single-grid solves
column for column (same seed, same rays); a mixed grid-and-points layout has one
column per receiver and `path_gain` is the sum of the component columns; the
capture-sphere diffraction agrees with a 0.1 m grid cell around the same point within
4 standard errors plus 10% (both MC diffraction estimators sit ~1.6x above Deterministic on
this finite wedge, a pre-existing MC diffraction convention difference); point diffraction AD
forward equals the primal and carries a source gradient; `receiver_radius_m` validation.

## Consequences

The S-9/S-10 refusals in `montecarlo.basic` are lifted, including diffraction at points.
Point receivers are noisier than grids at the same sample count; the default radius
keeps the bias below about 1e-4 relative, users trade bias for variance with
`receiver_radius_m`.
