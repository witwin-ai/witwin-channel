# ADR-064: Interaction chains: one diffraction anywhere in a reflection chain

- **Status:** Accepted.
- **Date:** 2026-11-11.
- **Decision ID:** `interaction-chains`.
- **Scope:** `coupled_paths` discovery and fields of Path, Deterministic and the BDPT enumerated
  oracle (`interactions/coupled.py`, `kernels/discovery.cu`, `kernels/coupled.cu`,
  `rayd/geometry.cpp`, the chain rows of `propagation/fields.py`). Plan item M3-6, finding P-11.
  Numerical change. Revises ADR-011 (coupled reflection-diffraction rows), ADR-012 (their
  stationary field) and ADR-046 (coupled D-D domain); see "Revisions" below.

## Context

`coupled_paths=True` added three row classes at `max_depth >= 2`: one reflection then one
diffraction (component 3), the reverse (4), and double diffraction (7). Discovery enumerated the
full product `T * R * (2 * G * E + E * (E - 1))` of transmitters, receivers, plane groups `G` and
edges `E`, built every candidate's indices and geometry with about fourteen Torch element-wise
operations, compacted with `torch.nonzero`, and refused any request above
`coupled_candidate_limit` (hard cap 1,000,000); the grid solver streamed receiver blocks under
that cap. On the 4x4 city with a 64x64 grid this is about 100 blocks and 64,269 kernel events,
910 ms (contended, see the evidence). A path with two reflections and a diffraction could not be
expressed at all. Double diffraction traced its three visibility legs independently (a candidate
blocked on its first leg still traced the other two) and paired every ordered edge pair.

## Decision

**One chain framework.** A single-diffraction row is a chain `a` reflections, one diffraction,
`b` reflections, `1 <= a + b <= max_depth - 1` (so `max_depth = 2` gives exactly the former
classes 3 and 4). As in Sionna's image method for mixed chains, the diffraction point is the
Keller point between the transmitter imaged through the `a` prefix planes and the receiver imaged
through the `b` suffix planes (reflecting the diffraction point through the suffix planes is the
same construction seen from the other side), and both halves are then back-traced with the
reflection validator:

- **Halves.** The transmitter half of depth `a >= 1` is a node of the transmitter's image tree
  (the reflection candidate tree of ADR-060), kept for an edge only when the edge may lie
  in the node's beam: some edge point beyond the node's last plane, and an edge piece (the edge
  split in eight, bounding spheres) meeting a cone from the node image around one of the last
  group's occupied macro cells. This is the image-tree child test with the child group replaced by
  the edge, so it is conservative. A half with `a = 0` is the transmitter itself with every edge
  (the validator's visibility segment decides). The receiver half is the same construction from
  the receiver (its tree uses the reversed suffix order), built per receiver chunk so a chunk's
  trees stay under 2^20 nodes.
- **Join.** For each class, the two halves are joined on their shared edge with an implicit index
  (per-edge products, CSR offsets, no materialized candidate table). One native launch per
  2^23 lanes computes the Keller point (RayD `first_order_diffraction_parameter`, which is
  symmetric in source and target), rejects a point outside the finite edge (interior tolerance
  1e-6 m) and back-traces each half from the diffraction point with the reflection receiver test
  (plane crossing within the segment, occupied cell of the group, M3-1 minimum segment). Survivors
  are compacted on the device (one count read per launch).
- **Validator.** `coupled_chain_geometry_forward` traces the transmitter half to the diffraction
  point with the RayD reflection EPC (or, for `a = 0`, one visibility segment from the diffraction
  point to the transmitter) and the receiver half the same way, only for rows whose first half is
  valid. The EPC uses the M3-1 path-validity rules, including the relative minimum segment length,
  which the coupled rows previously did not. `coupled_candidate_limit` keeps its name and range
  but now sizes the validator launches; it never caps or rejects a request.
- **Rows.** Component 3 is a chain whose diffraction is last, 4 one whose diffraction is first,
  and `8 + s` one whose diffraction is at interior slot `s`. The packed component sort key grows
  from 3 to 4 bits; the interior components share field row group 3, so the D2H row-group count
  copy is unchanged; all chain components accumulate into the coupled slot and the BDPT
  diffraction bucket.
- **Field.** One templated row body (`chain_row_body` in `coupled.cu`) serves the float primal and
  the dual JVP/VJP, so primal and derivatives cannot drift apart. The incident field reaches the
  edge through the prefix reflections as a spherical wave from the prefix image source (unfolded
  distance); the UTD pair (stationary-point re-anchoring, truncation, corner correction and the
  reciprocal face-angle rule unchanged) sees the prefix image as source and the suffix image of
  the receiver as target, so its spreading uses the unfolded suffix length; the diffracted field is
  reflected through the suffix planes and projected on the receiver. At `a + b = 1` the body
  performs the same operations in the same order as the former R-D/D-R kernels. Derivatives: the
  endpoints, eps_r / sigma_e / gain / thickness of every reflection slot and both wedge faces, and
  frequency. Under geometry AD the interaction points follow the endpoints through the image
  re-solve of the frozen winner inside the dual pass (planes and edge line fixed; primal values
  stay the discovered ones), which replaces the separate `coupled_rd_prepare` operation and its
  companions. Mesh-vertex gradients and multilayer-material derivatives stay refused.

**Double diffraction stays its own class** (component 7, ADR-046 solve), with three changes:
the first edge is restricted to the edges the transmitter sees and the second to the edges the
receiver sees (the four-sample axial visibility rule that selects order-1 diffraction states); the
three legs are traced in sequence, each only for candidates whose previous legs are visible; and
the solve is orientation-free: the outer legs are traced from the diffraction points to the
endpoints, the middle leg in both directions (visible when either trace is), and the reduced
bisection always runs on the edge with the larger id, so a candidate and its reverse publish the
same pair of points. A segment through a solid is blocked in both directions; one that grazes a
facade (common between the coplanar corners of aligned buildings) could be blocked by a hit at its
start point in one direction only, which made the former single forward trace drop such rows in
one direction and keep them in the other.

## Revisions

- ADR-011 / ADR-012: "one reflection and one diffraction" becomes "reflections with one
  diffraction anywhere, up to `max_depth` interactions"; the stationary-field construction is
  unchanged and generalised as above. The fixed-winner prepare operation is removed; its role is
  inside the chain dual. The 1,000,000 hard cap is removed (the constant remains the upper bound of
  the per-launch validator size).
- ADR-046: the certificate, bracket and leg domain are unchanged; the solve's edge order is now
  canonical (by edge id), the visibility legs are gated and orientation-free, and the candidate
  edges come from the endpoint-visible sets. The metadata geometry tags change to
  `native_interaction_chains` (Path) and `native_interaction_chains_plus_dd` (Deterministic), and
  the capability manifest reports `one_diffraction_anywhere_in_a_reflection_chain` with
  `max_reflections_in_coupled_path = 4` (public `max_depth` cap 5).

## Evidence

All GPU runs on GPU 1 under the slot locks, at most 8 GiB, contended (shared GPU); timings are
the minimum of five warm solves in one process.

**Agreement with the former enumerator** (Path, `max_depth=2`, `los/reflection/diffraction`):

| Scene | Former rows (RD/DR/DD) | Now | Common rows: max relative field difference |
|---|---|---|---|
| 2x2 city, 2 tx x 2 rx | 13/21/225 | 13/21/221 | 1.8e-4 (2.2e-7 on chain rows) |
| 4x4 city, 2 tx x 6 rx | 71/83/2072 | 71/83/2239 | 4.5e-4 (4.0e-7 on chain rows) |

- Every single-diffraction row the former enumerator found is found again with the same field to
  float32 rounding, and no new one appears. (An intermediate version that restricted the `a = 0`
  halves to the visible edges lost 5 single-diffraction rows of the second scene; the halves
  therefore take every edge.)
- Double diffraction differs in both directions. Lost: 24 / 248 rows (0.73 % / 0.48 % of the former
  D-D |a|) whose first edge none of the four axial samples sees from the transmitter, or second edge
  from the receiver: the restriction this record adopts. Gained: 20 / 415 rows (2.2 % / 14.8 % of
  the former D-D |a|) whose middle leg the former forward trace reported blocked while the reverse
  trace is clear (grazing legs, see Decision); 6 / 84 of them are the reverse of a row the former
  enumerator kept in the other edge order. The common D-D rows differ by up to 4.5e-4 because the
  bisection now runs on the edge with the larger id.
- Deterministic 4x4 city, 16x16 grid: the los, reflection and diffraction maps are unchanged
  (relative 3e-8); the coupled map moves by 3.0 % (summed magnitude), all from double diffraction
  (48,083 former rows, 47,636 now).

**Performance**, deterministic 4x4 city, 64x64 grid, one transmitter, `max_depth=2`:

| | Former | Now |
|---|---:|---:|
| Coupled solve | 910 ms | 66 ms |
| Same solve without coupled_paths | 8.7 ms | 8.1 ms |
| CUDA kernel events per solve | 64,269 | 1,846 |
| Longest kernel | 5.5 ms | 11 ms (node-edge beam test of 4,096 receiver nodes) |
| Peak allocated | 896 MiB | 870 MiB |
| Coupled rows (chains / D-D) | 86,381 / 776,580 | 86,378 / 767,497 |

At `max_depth=3` (chains with two reflections, which the former enumerator could not express) the
same solve takes 608 ms with 1.9 GiB peak and no kernel above 14 ms; the receiver-side image
trees dominate.

**Reciprocity:** `tests/accuracy/test_consistency.py::test_coupled_reflection_diffraction_is_reciprocal`
now includes double diffraction and `max_depth=3` chains and checks equal per-pair row counts in
both directions besides the coherent sums: relative error 1.8e-4 at both depths (tolerance
8.5e-3); the row sets of both directions are identical.

**AD:** endpoint, material and frequency gradients of the depth-2 rows against central differences
(existing tests), and of two-reflection chains (endpoint gradients against central differences,
material JVP against VJP and central differences).

## Consequences

- `coupled_paths` no longer refuses any scene size, and the grid solver no longer streams fixed
  receiver blocks.
- Rows are ordered by receiver chunk, chain class and edge, not by the former linear candidate
  index; consumers that relied on the former coupled row order must sort.
- Double diffraction depends on the sampled endpoint-edge visibility, like order-1 diffraction; a
  pair whose edges are partly visible but miss all four samples is not found.
