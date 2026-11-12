# ADR-073: Wedges from every face that meets an edge, across structures

- **Status:** Accepted by the owner (relayed by the coordinator, 2026-11-11): build each edge's wedge from
  the union of all faces meeting at the edge across structures, with the true exterior angle and per-face
  materials; drop fully interior edges.
- **Date:** 2026-11-11.
- **Decision ID:** `fw-d5-coincident-edges`.
- **Scope:** the compile-time diffraction edge geometry (`diffraction_edge_geometry_kernel`
  in `native/channel/kernels/diffraction.cu`, `refine_edge_geometry` and `_drop_contact_edges` in
  `witwin.channel.scene.resources`). No per-solve work, no launch or sync change in a solve.

## Context

The full-wave suite (`docs/dev/fullwave-validation.md`, defect D5) found that edges where separate
structures touch are diffracted as if each structure stood alone. Two geometries in the reference set
show it, and both occur in every imported city (building meshes standing on a ground mesh):

| Case / region (RMS dB / NMSE) | As built | Geometry without the defect |
| --- | --- | --- |
| `trihedral_x` plane 0 shadow | 4.23 / 1.65 | 1.61 / 0.10 (one welded mesh) |
| `trihedral_x` plane 1 shadow | 3.67 / 0.61 | 1.84 / 0.14 (one welded mesh) |
| `trihedral_x` plane 1 corner | 4.33 / 1.34 | 2.15 / 0.49 (one welded mesh) |
| `canyon_z_3g5` plane 0 shadow | 4.23 / 1.56 | 2.92 / 0.85 (block bases hidden 5 cm below the ground) |
| `canyon_x_3g5` plane 0 shadow | NMSE 1.04 | NMSE 0.75 (bases hidden) |

(Numbers from the `fullwave2` report, before the D1-D4 and D6 fixes; the D1/D4 fixes since moved
`trihedral_x` plane 0 shadow to 3.04 dB / 0.91 and the canyon RSB metrics, see ADR-069 (fw-d4).)

Channel-only diagnosis of the edge set (`selected_diffraction_edges`, this worktree):

- **Canyon.** Every block is a closed box resting on the ground box. All 16 base edges are exported as
  convex wedges with exterior angle 270 degrees between a side face and the block's bottom face. The
  bottom face lies on the ground's top face, coplanar and facing it, so the real free space at a base is
  the 90 degree concave corner between the side face and the ground: the edge must not diffract at all.
  The exported wedge sends half of its diffracted field into the ground, and its incident and
  reflection shadow boundaries are the boundaries of fields that do not exist there. (The unexplained
  canyon LoS residual turned out not to depend on them; see the last section.)
- **Trihedral.** Three zero-thickness PEC plates meet pairwise. The two ground-wall junctions are
  dropped entirely: `_drop_contact_edges` treats a sheet edge resting on another surface as no edge.
  For a zero-thickness sheet that is wrong on the outside: below the ground plate and behind the wall
  the free space is a 270 degree sector bounded by the two sheets' outer sides, a convex wedge that
  diffracts into the shadow region the x-polarized case measures. The wall-wall junction is merged into
  one 270 degree wedge, but with one outward and one inward normal (double-sided sheets have no
  outward side), so its exterior test and face-n angles refer to the wrong side of the second sheet.

## Decision

Build each edge's wedge from the union of all faces that meet the edge line, across structures:

1. **Gather.** For each candidate edge segment (after the collinear merge), collect every face of any
   structure that contains a neighbourhood of the segment in its closure: faces sharing the edge (the
   current records), faces whose own boundary edge coincides with it (the current coincident-edge
   merge), and faces whose interior contains it (a block base on a ground plane). Containment is tested
   with the RayD in-face rule (metric slack, fw-d1) on the two segment ends and its midpoint, against the
   faces the scene BVH returns near the segment.
2. **Sectors.** In the plane normal to the edge, each gathered face is a ray from the edge (a sheet) or,
   for a face of a closed solid, the boundary of an occupied sector (the solid side is known from the
   outward normal); a face whose interior contains the edge occupies its back half-plane when it bounds a
   solid and is two rays otherwise. The union of occupied sectors and sheet rays splits the 2 pi around
   the edge into free sectors.
3. **Wedges.** Every free sector wider than pi is a wedge with exterior angle equal to its width,
   face 0 and face n being the two faces that bound it, each with its own material (the UTD coefficient
   already takes the two faces' materials separately). A sector of at most pi (concave or flat) does not
   diffract. A fully enclosed edge (no free sector) is dropped. An edge with several free sectors (a
   sheet ending on another sheet) yields one wedge per sector wider than pi.
4. **Partial overlap.** Where the gathered set changes along the edge (a block base on a ground that
   ends), the segment is split at the change; each piece gets its own sectors.
5. **Compile time only.** This replaces `_drop_contact_edges` and the coincident-edge branch of the
   merge; it runs once per RayD scene behind the existing edge-geometry cache. Reflection, transmission
   and the solve-time kernels are unchanged.

Expected effect, from the full-wave proxies above: `trihedral_x` shadow 4.23 -> about 1.6 dB (welded mesh),
canyon shadow 4.23 -> about 2.9 dB (hidden bases). Both proxies are geometric equivalents of the proposed
edge set (a welded mesh gives the same sectors for the plates; hidden bases remove the base wedges that
step 3 drops), not runs of the proposed code.

## Alternatives

- **Drop concave contact edges only** (the other option the full-wave report names): drop a wedge one
  of whose faces lies on another structure's face. That fixes the canyon bases but not the trihedral,
  whose sheet junctions need a wedge, not a deletion; it is a subset of step 3.
- **Ask users to weld meshes.** Imported cities do not, and the result would depend on the mesh layout.

## Implementation

`witwin.channel.scene.resources._wedges_from_every_face` replaces `_duplicate_boundary_pairs` and
`_drop_contact_edges` inside `refine_edge_geometry` (compile time, cached with the edge geometry):

- **Gather.** Around each selected edge's midpoint a polygon of 32 chords on a 1 mm circle in the plane
  normal to the edge is traced with RayD segment penetration (one launch for all edges). Every face through
  the edge line crosses it once per side it extends to; a crossing counts when the face's plane holds both
  edge end points. Each crossing is a ray `±(n_f x t)` from the edge.
- **Sectors.** A face of a closed (edge-connected, boundary-free) surface occupies the side opposite its
  normal; the edge's own two faces use the wedge's outward normals; sheets occupy nothing. Rays within
  1e-4 rad merge.
- **Wedges.** The free sector wider than pi becomes the wedge: face 0 and face n are its bounding faces
  (global face ids, so each keeps its own material), `n0`/`n1` their normals turned into the sector,
  the exterior angle its width; `n0 x dir` runs along face 0 as before. An edge without one is
  deselected (fully interior edges, concave corners, coplanar seams). Records that resolve to the same
  wedge (coincident edges of several structures) keep the first.
- **Unchanged.** An edge that no other face meets keeps its record bit for bit; scenes without touching
  structures are unaffected.
- **Every scene.** The imported-edge exception is removed (owner decision, 2026-11-12): a scene whose
  metadata carries `{"mitsuba": {"merge_shapes": True}}` used to keep RayD's raw edge table, counting the
  coincident edges of different meshes as separate wedges (Munich, 32 x 32 grid: 45,429 edges and 28,910
  depth-1 rows with the key, 26,909 and 19,162 without); every scene now diffracts from this wedge union,
  and no metadata selects another edge table.

Limits: the gather probes the midpoint only, so a coincident edge that overlaps another only in part is
not split (step 4 above is not implemented). The mesh-vertex AD path rebuilds a wedge's normals from the
record's own triangle and the opposite vertex of face n, which for a face of another structure is found by
vertex id; vertex AD through a cross-structure wedge is not covered by a test.

## Evidence

Tests: `tests/propagation/geometry/test_wedge_edge_set.py::test_block_base_on_the_ground_is_not_a_wedge`
and `::test_sheets_meeting_at_their_edges_form_one_wedge` (both windings; one 270 degree wedge, normals
into the free sector, one face of each plate); both fail before. The existing edge-set tests (subdivided
and unwelded city, concave L prism, sheet resting on a surface) pass unchanged.

Edge sets (`d5_edges.py` in the task scratch): the canyon loses its 16 block-base wedges (36 edges left: the
blocks' vertical and roof edges and the ground plate's four rims); the trihedral has its three junctions
as 270 degree wedges (two ground-wall, one wall-wall) with normals into the outer sector, so a source in
the inner corner fails the wedge's exterior test.

Full-wave gate, before -> after (all other fixes in place):

| Metric | Before | After | Target |
| --- | ---: | ---: | ---: |
| `trihedral_x` plane 0 shadow RMS / NMSE | 3.04 / 0.908 | 1.67 / 0.115 | 2.00 / 0.15 |
| `trihedral_x` plane 1 shadow RMS / NMSE | 2.77 / 0.336 | 1.86 / 0.139 | 2.10 / 0.17 |
| `trihedral_x` plane 1 corner RMS / NMSE | 4.33 / 1.344 | 2.15 / 0.633 | 2.45 / 0.60 |
| `canyon_z_3g5` plane 0 shadow RMS / NMSE | 3.87 / 1.246 | 2.65 / 0.697 | 3.25 / 1.00 |
| `canyon_x_3g5` plane 0 shadow NMSE | 0.962 | 0.855 | 0.90 |
| `canyon_z_3g5` plane 1 ISB | 2.86 dB | 1.45 dB | 1.50 |
| `canyon_z_3g5` plane 0 / 1 RSB | 5.55 / 4.65 dB | 3.84 / 4.93 dB | 1.50 |
| `trihedral_z` plane 0 shadow NMSE | 0.161 | **0.462** | 0.195 (ceiling) |
| `canyon_x_3g5` plane 0 / 1 RSB | 1.14 / 1.96 dB | **3.27 / 2.76 dB** | 2.15 (ceiling) / 1.50 |

Nine strict xfails flip; twelve ceilings tighten. Two metrics regress, both diagnosed (Channel-only probes):

- **`trihedral_z` plane 0 shadow NMSE.** The new ground-wall junction wedges receive double-diffraction
  legs that run on the plates' planes (from the plates' free rims to the junction). Dropping the D-D rows
  whose two points lie on one plate plane gives 0.161, the value before (halving them 0.276); on
  `trihedral_x` the same rows cost 0.008-0.011 NMSE. The coefficient of a D-D leg on a two-sided sheet does
  not know which side of the sheet the leg runs on (the first edge evaluates it on its `n0` side, the
  junction receives it on its exterior side), which is a double-diffraction defect this change exposes,
  not one it introduces. Recorded as a strict xfail (target 0.195, the previous ceiling).
- **`canyon_x_3g5` RSB.** The order-1 diffraction is unchanged; the chains through the 16 spurious base
  wedges had filled reflection and transmission steps at the block bases (the transmission component's
  support edges are not compensated by any diffraction). The plane 0 value had passed its ceiling (2.15,
  set when fw-d4 flipped it); it is again a strict xfail, at 3.27 dB against the 4.52 dB the integration
  branch records.

Both are owner decisions: the two gate entries moved from ceilings to strict xfails.

## Unexplained ISB residual (resolved as a measurement effect)

The "25 % of the LoS" figure came from a probe that also counted support edges on block faces, where one
sample lies inside a block. Excluding block interiors, the canyon LoS steps are compensated to 1.4 % (steps
along y) and 2.3 % (along x) of the LoS, and the ground-reflection steps that coincide with them to 0.0-0.5 %.
The ISB p95 excess left (2.25 / 2.86 dB on `canyon_z` planes 0 / 1, 3.44 dB on `canyon_x` plane 0) is at
samples where the total is 15-20 dB below the LoS (deep cancellation near the block corners), where a few
percent of the LoS is a large dB jump. The base wedges are not its cause: hiding the bases moves the
interior-inclusive residual median 0.22 -> 0.32, and extending the blocks' free ends below the ground plane
changes it by 0.005. It is not part of this decision.
