# ADR-069: Interaction chains that keep every reflection path continuous

- **Status:** Accepted (full-wave principle set by the owner; coordinator task `fwfix`).
- **Date:** 2026-11-11.
- **Decision ID:** `fw-d4-chain-completeness`.
- **Scope:** the interaction-chain enumeration (`witwin.channel.interactions.coupled`,
  `chain_depth_limit`), the chain candidate kernel (`chain_candidate` in
  `native/channel/kernels/discovery.cu`), the path-table width (`propagation.enumerated`), the
  path-solver capability record (`max_reflections_in_coupled_path`).

## Context

Defect D4 of `docs/dev/fullwave-validation.md`: where another object blocks a reflected path, the coupled
component stepped by only 0.3-0.8 % of the reflected field, so the total stepped by the full |R|
(`wall_column` RSB excess 5.35 / 2.37 dB, canyon 2.8-8.6 dB). The Channel-only probe of the alley between
the column and the wall (`wc_probe.py`, `paths_at.py` in the task scratch) found two causes.

1. **A missing chain class.** The boundary is the edge of the column's shadow on the first leg of a
   wall-column-wall reflection (`R-R-R`, depth 3 = `max_depth`). The field that keeps it continuous is the
   column edge's diffraction of the incident field followed by the same three reflections, a `D-R-R-R`
   chain of depth 4. Chains were limited to `max_depth` interactions (`a + b <= max_depth - 1`
   reflections), so a reflection path of `max_depth` reflections never had its blockage compensated. In
   general a reflection path with `k` reflections needs chains with `k` reflections and one diffraction.
2. **Chains through closed solids.** The candidate test joins a transmitter-side and a receiver-side image
   on a shared edge, but nothing required the legs to leave the edge into free space. On the single PEC
   cube (convex, so no reflection-diffraction chain exists) 9,803 `D-R` rows reached the cube's back face
   from inside it (the diffraction point on a front-top edge, the reflection point on the opposite face):
   segment visibility cannot see them, because a segment inside a closed box crosses no triangle. Their
   field (up to 2e-3, against 5e-3 for the order-1 diffraction) lit the cube's shadow.

## Decision

1. Chains carry up to `max_depth` reflections and the diffraction (`a + b <= max_depth`, at most
   `CHAIN_MAX_DEPTH` = 8 interactions): every reflection path of the solve has its blockage chains.
   `chain_depth_limit(max_depth) = min(max_depth + 1, 8)` sets the enumeration depth and the path-table
   width; `max_reflections_in_coupled_path` becomes 5 (`max_reflection_depth`).
2. A chain candidate is kept only when both legs at the diffraction point leave into the wedge exterior
   (`utd::wedge_exterior_mask` of the directions to the two half images, which are the directions to the
   adjacent interaction points). The candidate kernel receives the edges' face normals.

The chain's own boundaries (a chain's reflection point leaving its face, or a chain leg blocked) are
second-order and stay uncompensated (they need two diffractions), as before.

## Cost

The depth-`max_depth + 1` classes add receiver-side image-tree halves of depth `max_depth`. Measured on the
gate scenes (GPU shared with other jobs; contended, single runs, wall time of one case solve):

| Case | Before | After |
| --- | ---: | ---: |
| `cube_z` (depth 2) | 1.8 s | 1.9 s |
| `wall_column_z` (depth 3) | 0.2 s | 0.6 s |
| `trihedral_z` (depth 3) | 0.6 s | 0.7 s |
| `canyon_z_3g5` (depth 3, 21 plane groups) | 3.1 s | 28 s |

The canyon cost is the node-edge beam test of the depth-3 receiver halves (`chain_node_edges`, 11.5 s of
the 28 s in a profile) and is linear in receivers. The receiver chunk was raised from 2^20 to 2^22 image-tree
nodes (peak memory 3.9 GiB on the canyon plane, unchanged). The launch-budget cases are depth 1 and do not
see the change.

Since then (task `chainperf`, rows unchanged bit for bit): the deepest receiver halves are tested as the
children of the level above inside the join (`chain_leaf_candidates`) instead of being materialized and
beam-tested against every edge, so receiver chunks are sized by the deepest materialized level again; the
image-tree and beam tests gain conservative one-sphere quick rejects. On the canyon plane (contended) the coupled
launches went 2,444 -> 256 and the warm solve 15-22 s -> 2.9-4.3 s; a solve with interaction chains over more
than 32,768 receivers accumulates in receiver windows of about 2M rows, so the canyon at 170k receivers peaks at
3.4 GiB (104k receivers had run out of 8 GiB).

## Evidence

Full-wave gate, before -> after. The exterior test and the chain depth were measured as separate builds:

| Metric | Before | Exterior test (with fw-d1, fw-d6) | + chain depth | Final (all fixes) |
| --- | ---: | ---: | ---: | ---: |
| `cube_z` plane 0 shadow RMS | 2.91 dB | 2.14 dB | 2.14 dB | 2.14 dB |
| `trihedral_x` plane 0 shadow NMSE | 1.654 | 0.908 | 0.908 | 0.908 |
| `wall_column_z` plane 0 RSB | 5.35 dB | 5.35 dB | 1.66 dB | 0.70 dB |
| `wall_column_y` plane 0 RSB | 2.37 dB | 2.15 dB | 1.40 dB | 0.98 dB |
| `canyon_x_3g5` plane 0 RSB | 4.52 dB | 4.39 dB | 1.21 dB | 1.14 dB |
| `canyon_z_3g5` plane 0 RSB | 8.61 dB | 8.61 dB | 5.64 dB | 5.55 dB |
| `beam_y` plane 0 shadow NMSE | 0.136 | 0.135 | 0.157 | 0.164 |

`wall_column_z`, `wall_column_y` and `canyon_x_3g5` plane 0 RSB are strict xfails this change flips (the last
step to their final values is fw-d3). The canyon RSB left (5.55 / 4.65 dB on `canyon_z`) is at second-order
chain boundaries (above). `beam_y` shadow NMSE stays under its 0.165 ceiling.

Tests: `tests/deterministic/test_fullwave_boundaries.py::test_chain_legs_leave_the_diffraction_edge_into_open_space`
(three-cube scene, no chain leg crosses a cube; fails before) and
`::test_chains_carry_as_many_reflections_as_the_reflection_paths` (wall and column at `max_depth` 2:
chains of depth 3 exist; fails before).
