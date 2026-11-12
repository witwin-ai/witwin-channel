# ADR-059: One definition of a valid specular reflection path

- **Status:** Accepted.
- **Date:** 2026-11-10.
- **Decision ID:** `reflection-path-validity`.
- **Scope:** every specular reflection row of the enumerated engine (Path, Deterministic, the
  consumer, BDPT's enumerated oracle) and the specular legs of scatter chains. Numerical change.

## Context

Finding C-1: the two reflection candidate schemes of 0.5.0 (exhaustive plane-group sequences up to
100,000 sequences, otherwise rays traced from the transmitter) returned different rows for the same
depth, and which one ran depended on `max_depth`. On the 4x4 city with a 64 x 64 grid, depth 2 lost
80 rows and depth 3 gained 1,008 rows when `max_depth` went from 3 to 4. Every inspected difference
was a degenerate path that one scheme accepted and the other did not:

- ray discovery accepted the chain `[g, g, g]` (three reflections at the same ground point; the path
  length equals the single reflection) because nothing rejected a repeated plane group;
- the exhaustive scheme accepted depth-2 chains whose two reflection points coincide on the shared
  vertical edge of two facades, once in each order (`[102, 105]` and `[105, 102]`), each carrying
  about the gain of a real single reflection.

The EPC validator treated a segment shorter than an absolute 2e-5 m as trivially visible, so a
zero-length segment never failed, and nothing else looked at segment lengths.

## Decision

A candidate is a plane-group sequence `[g_1 .. g_d]` with a transmitter and a receiver. It is a valid
specular path exactly when the RayD reflection EPC (image method plus segment visibility) accepts it
under these rules, applied by the one validator whatever produced the candidate:

1. **Minimum segment length.** Every segment (transmitter to the first point, between consecutive
   points, last point to the receiver) is at least `1e-5` times the scene diagonal (the diagonal of
   the axis-aligned bounds of the compiled geometry, `enumerated_penetration_scene_diagonal_m`;
   2.3 mm on the 4x4 city). Shorter means two interactions, or an interaction and an endpoint,
   coincide: the path is the edge-degenerate or repeated-plane artefact above, or a receiver lying on
   the reflecting plane (where the reflected and direct paths are the same ray). RayD enforces it
   in the EPC kernel through the new `ReflectionEpcRequest::min_segment_length` (RayD ADR-0048).
2. **No repeated plane group.** Consecutive interactions are on different plane groups. Candidate
   generation never forms such a sequence (the image tree only appends a different group; traced
   chains with a repeat are dropped); a numerical repeat would also fail rule 1, because reflecting
   twice in one plane puts both points at the same place.
3. **Edge ownership.** A plane group is the closed union of its coplanar triangles. A point on the
   group boundary (within the EPC barycentric slack) belongs to it. Two different groups that share
   an edge can therefore both own a point on that edge; that is not a duplicate, because the two
   sequences are different physical paths (different planes, different images). The only way one
   path appeared twice was with coincident points, which rule 1 removes. Inside a group, the
   resolved triangle of a point on an internal edge is the one RayD's containment probe meets
   (RayD ADR-0048), which does not change the path.
4. **Sidedness.** Every surface reflects from both sides, for `surface` and `solid` fill modes alike,
   and the material decides only the coefficient. A path that reaches the inner side of a closed
   solid needs an endpoint inside it, and the visibility test is what rejects or keeps it; adding a
   one-sided rule would make validity depend on mesh winding, which imported meshes do not
   guarantee.
5. **Relative thresholds.** All validity and pruning tolerances are fractions of the scene diagonal
   (the probe of rule 3 is a fraction of the point's coordinate magnitude), never absolute metres.

## Consequences

- The C-1 scheme-equivalence tests of the accuracy suite pass (they were strict xfails). Together
  with the per-depth source choice of the candidate-generation record (M3-3 to M3-5), the rows of a
  depth no longer depend on `max_depth`.
- Rows removed: the depth-3 `[g, g, g]` rows, the coincident-edge depth-2/3 rows, and reflections
  whose receiver is closer than the threshold to the reflecting plane. Bit-level baselines that
  contain such rows move.
- Coupled R-D/D-R geometry still calls the EPC with `min_segment_length = 0`; whether the rule
  should cover it belongs to the coupled-path owner.

## Evidence

See the candidate-generation record for the measured row counts before and after.
