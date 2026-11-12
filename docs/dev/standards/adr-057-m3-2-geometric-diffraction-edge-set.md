# ADR-057: The diffraction edge set is the set of geometric wedges

- **Status:** Accepted.
- **Date:** 2026-11-10.
- **Decision ID:** `m3-2-geometric-diffraction-edge-set`.
- **Scope:** the compile-time diffraction edge geometry (`diffraction_edge_geometry_kernel` in
  `native/channel/kernels/diffraction.cu`, `refine_edge_geometry` in `witwin.channel.scene.resources`)
  and the wedge re-evaluation inputs of diffraction AD (`propagation.fields`). Plan item M3-2; findings
  C-6 and A-3. The UTD coefficient corrections that complete C-6 are ADR-050.

## Context

C-6: on the 4x4 synthetic city (32x32 grid, diffraction only) the total diffraction gain was
8.21e-08 / 7.38e-08 / 6.11e-08 for the original mesh and its 1x / 3x welded 4-way subdivisions,
although the geometry is identical, and row counts grew from 11,151 to 23,810. Two separate causes:

1. **Extra rows: the edge set followed the triangulation.** Every triangle edge was a candidate.
   Coplanar interior edges were already dropped by the kernel, but a physical edge split into `m`
   collinear segments produced `m` states. The deterministic exporter evaluates each segment at the
   line's stationary point; when that point lies outside the segment the row is clamped to the
   segment end and kept, so every receiver got one row per segment.
2. **Changed per-path gains: the coefficient depended on the segment and on the face order.** The
   finite-edge aperture factor is a nonlinear function of the segment extent (a sum over segments is
   not the whole edge's factor), and the factor itself used the wrong Fresnel argument (ADR-050).
   Even after merging, the residual moved paths by up to 1.7x: the merged edge of a subdivided mesh
   took the other face as "face 0" (the kernel orders faces from the vertex order), and the
   finite-conductivity coefficient was not invariant under that relabelling (ADR-050, RayD ADR-0047).

A-3: a single screen's coefficient depended on its winding (ratio 0.25 to -3.6 for the reversed
normal). UTD measures phi and phi' from the in-face direction `n0 x edge_dir`; the kernel oriented
interior wedges by swapping normals, but a boundary (half-plane) edge kept the record's vertex order,
so for one of the two windings `n0 x edge_dir` pointed into the missing half-plane. A half-plane also
carried no material on its back face (`face1 = -1`), so the face-n reflection term was zero.

A concave (reflex) interior edge was treated as convex: its exterior angle was computed as
`2 pi - acos(-n0 . n1)`, which is at least pi for every edge.

## Decision

The diffraction edge set is the set of geometric wedges of the scene, built once per RayD scene and
cached with it:

1. **Coplanar interior edges** are not wedges (unchanged rule; plane tolerance 1.34e-5 m).
2. **Concave interior edges** are not wedges: with outward normals, each face of a convex wedge lies
   behind the other's plane; an edge whose faces both lie in front of each other's planes (by more
   than the plane tolerance) is dropped. Mixed winding (inconsistent normals) keeps the old rule.
3. **Orientation.** Every exported edge is oriented so that `n0 x edge_dir` points from the edge into
   face 0 (checked against face 0's opposite vertex). The kernel exports the faces in the order of its
   normals (`face0`/`face1` now follow `n0`/`n1`; before, they followed the record order even when the
   normals were swapped).
4. **Half-planes.** A boundary edge keeps `n1 = -n0` and exterior angle `2 pi`; its back face is the
   sheet itself (`face1 = face0`), so both reflection terms see the sheet's material. Code that needs
   "is this a half-plane" tests `face1 == face0` (or `face1 < 0` for raw records).
5. **Coincident boundary edges** of two shapes or of an unwelded mesh merge into one wedge (unchanged)
   and are dropped when coplanar or, new, concave.
6. **Collinear merge.** Selected segments merge into one edge when one's start point is the other's
   end point (positions quantised to 1e-4 m, so unwelded duplicates match), they are parallel, carry
   the same unordered pair of face normals (quantised to 1e-3) and join the same pair of shapes. Chains
   are followed by pointer jumping. The chain's first segment keeps its orientation and faces and takes
   the chain's whole extent; the other segments are deselected. Segments with different shapes on a
   side are not merged, so a material change along an edge stays a separate edge.
7. **AD inputs.** Mesh-vertex diffraction AD rebuilds the wedge from vertex positions. It now receives
   the first and last end vertex of the (merged) edge along `edge_dir` (`diffraction_edge_vertex_ids`,
   cached next to the geometry), the opposite vertices of the representative segment, and the
   half-plane flag `face1 == face0`. Interior chain vertices get no gradient: they are constrained to
   the line, so moving one off it would make a kink the wedge model cannot represent.

The edge geometry is compile-time topology: the merge runs once per RayD scene in Torch, behind the
existing cache in `cached_diffraction_edge_geometry`, and is not per-solve work. The warm-solve launch,
sync and copy budgets do not see it.

## Evidence

- `tests/propagation/geometry/test_wedge_edge_set.py`: the 2x2 city and its 1x / 2x welded and 1x
  unwelded subdivisions export the identical list of wedges (end points to 1 mm, exterior angles); both
  windings of a screen give `n0 x edge_dir` into the screen and `face1 == face0`; an L-shaped prism
  exports its five convex vertical edges and not the reflex one.
- `tests/accuracy/test_consistency.py::test_pec_screen_winding_invariance`: 1.36 relative before,
  3.5e-6 after (tolerance 1e-4); strict xfail removed.
- C-6 subdivision invariance (radiomap L1 difference / total, `tests/accuracy`), see ADR-050 for the
  combined result.

## Consequences

- Diffraction row counts drop on meshes whose physical edges are split, and per-row gains change
  wherever the stationary point of a split edge used to fall on a segment end.
- MC diffraction samples edges in proportion to their length; merged edges keep the total length, so
  the sampling density is unchanged, but the sampled edge set and its states change.
- Every scene, imported or built, uses the refined edge table. (The metadata-keyed exception that kept
  a `merge_shapes` Mitsuba import's edge table unrefined is removed; see ADR-073.)
- Boundary edges that rest on another surface (the open bottom of a box standing on a ground plane, a
  sheet ending on a wall) are not half-planes: at compile time two short visibility probes cross
  the edge line just past the edge, one on each side of the sheet, 1 mm off the sheet and 1 mm into
  and beyond it; if either is blocked the edge is deselected (`_drop_contact_edges`). Free sheet
  edges see nothing there and stay half-planes. `test_sheet_edge_resting_on_a_surface_is_not_a_half_plane`.
