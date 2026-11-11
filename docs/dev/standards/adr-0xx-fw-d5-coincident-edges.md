# ADR-0xx (fw-d5): Wedges from every face that meets an edge, across structures

- **Status:** Proposed. Design only; no code change until the owner decides (the coordinator relays the
  decision). The recommendation is the coordinator's; the owner has not accepted it yet.
- **Date:** 2026-11-11.
- **Decision ID:** `fw-d5-coincident-edges`.
- **Scope (if accepted):** the compile-time diffraction edge geometry (`diffraction_edge_geometry_kernel`
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
`trihedral_x` plane 0 shadow to 3.04 dB / 0.91 and the canyon RSB metrics, see the fw-d4 record.)

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

## Proposed decision

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

## Unexplained ISB residual (resolved as a measurement effect)

The "25 % of the LoS" figure came from a probe that also counted support edges on block faces, where one
sample lies inside a block. Excluding block interiors, the canyon LoS steps are compensated to 1.4 % (steps
along y) and 2.3 % (along x) of the LoS, and the ground-reflection steps that coincide with them to 0.0-0.5 %.
The ISB p95 excess left (2.25 / 2.86 dB on `canyon_z` planes 0 / 1, 3.44 dB on `canyon_x` plane 0) is at
samples where the total is 15-20 dB below the LoS (deep cancellation near the block corners), where a few
percent of the LoS is a large dB jump. The base wedges are not its cause: hiding the bases moves the
interior-inclusive residual median 0.22 -> 0.32, and extending the blocks' free ends below the ground plane
changes it by 0.005. It is not part of this decision.
