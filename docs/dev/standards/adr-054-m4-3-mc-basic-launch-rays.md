# ADR-054: MC Basic launch rays, seeded lattice, bounds fold and bounded final segment

- **Status:** Accepted.
- **Date:** 2026-11-10.
- **Decision ID:** `mc-basic-launch-rays`.
- **Scope:** the reflection estimator of `witwin.channel.montecarlo.basic` (and the
  first-hit wedge events MC diffraction discovers edges from). Numerical change.

## Context

Findings P-14 and C-3: the reflection launch directions were a double-precision
Fibonacci lattice regenerated per transmitter, identical for every `seed` (so seeds
gave no error estimate); every ray was launched over the full sphere although in
open scenes about two thirds never hit anything; and RayD traced `max_depth + 1`
unbounded closest hits, the last one only to learn how far the final reflected
segment may travel before it is blocked.

## Decision

**Directions.** One launch (`mc_reflection_rays`) writes the rays of every
transmitter in float32. Ray `i` of `N` starts from the lattice point
`z = 1 - (2i + 1) / N`, `phi = 2 pi frac(i * 2654435769 / 2^32)` (golden-ratio
azimuth in 32-bit fixed point, exact for any `N`). The lattice is rotated by a
uniformly random rotation (Shoemake quaternion) drawn from `splitmix64(seed,
transmitter index)`; each rotated point is then marginally uniform on the sphere, so
the estimator is unbiased for every seed and different seeds give independent
estimates.

**Bounds fold.** Let `B` be the axis-aligned bounds of all scene vertices
(`mc_scene_bounds`, cached per RayD scene). For each axis on which the transmitter
lies outside `B` (above the top, say), a direction pointing away from `B` on that
axis can never hit geometry, and a reflection deposit needs at least one hit. The
sign of that direction component is therefore folded towards `B`. The fold maps the
uniform sphere measure 2^k-to-1 onto the region that can reach `B`, so each ray
carries solid angle `4 pi / (2^k N)` (per-transmitter device scalar) and the
estimator stays unbiased: it integrates the same function over the same set, with
the zero part removed. The fold never applies to a transmitter inside `B`.

**Bounded final segment.** RayD now traces `max_depth` closest hits. For every ray
that completed all bounces, `mc_reflection_final_segment` replays the frame chain and
records the farthest distance at which any receiver block could take a deposit from
the final segment (grid plane crossing inside the grid, or a point's closest
approach within its capture radius). One more RayD trace of depth 1 runs only for
those rays, bounded by that distance and spawned `1e-5 * max(1, |hit|_inf)` along
the segment; `mc_reflection_append_final` appends its hit as the last trace column
(distance measured from the last hit). The accumulate kernels and their AD
companions read every segment blocker from that table exactly as before, so
occlusion is decided by the same rule ("deposit if the target lies before the next
hit"), only the final hit is searched within a bounded interval instead of to
infinity.

## Evidence

- `tests/montecarlo/basic/test_basic_receiver_layouts.py`: the same seed reproduces,
  a different seed changes the estimate; six seeds on the one-wall scene give a
  point-receiver reflection whose mean agrees with the deterministic image-method
  gain within four standard errors plus the sphere-averaging bias.
- The existing MC Basic statistical tests and the grid component-map tests
  (`tests/montecarlo/basic`) run against the new estimator; their tolerances are
  unchanged.
- Bit-level MC Basic reflection baselines move (new directions); see the commit.

## Consequences

`seed` now matters for reflection. Results differ from earlier releases by Monte
Carlo noise. A transmitter outside the scene bounds gets up to 8x more rays on the
geometry.
