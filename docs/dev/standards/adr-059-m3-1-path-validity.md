# ADR-059: One definition of a valid specular reflection path

- **Status:** Accepted; rule 1 amended 2026-11-12 (endpoint segments, see the amendment).
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

1. **Minimum segment length between interactions.** Every segment between two consecutive
   interactions is at least `1e-5` times the scene diagonal (the diagonal of the axis-aligned bounds
   of the compiled geometry, `enumerated_penetration_scene_diagonal_m`; 2.3 mm on the 4x4 city).
   Shorter means two interactions coincide: the path is the edge-degenerate or repeated-plane
   artefact above. The segments from the transmitter and to the receiver are exempt, because an
   endpoint is not an interaction: a receiver on a reflecting plane, or within the threshold of it,
   keeps its reflection (see the amendment). Where one end of a specular chain is itself an
   interaction (the diffraction point of a reflection-diffraction chain, or the scattering vertex of
   a scatter-chain leg, which lies on its own surface), the segment to it is an interaction segment
   and the rule covers it. RayD enforces the rule in the EPC kernel through
   `ReflectionEpcRequest::min_segment_length` and `receiver_is_interaction` (RayD ADR-0048).
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
- Rows removed: the depth-3 `[g, g, g]` rows and the coincident-edge depth-2/3 rows. Bit-level
  baselines that contain such rows move. (Reflections whose receiver is closer than the threshold
  to the reflecting plane were also removed until the amendment restored them.)
- Coupled R-D/D-R geometry still calls the EPC with `min_segment_length = 0`; whether the rule
  should cover it belongs to the coupled-path owner.

## Evidence

See the candidate-generation record for the measured row counts before and after.

## Amendment: endpoint segments are exempt (2026-11-12)

**Context.** Rule 1 as first written covered every segment, including the transmitter and receiver
segments, and called a receiver on the reflecting plane a degenerate double count. It is not
(finding C-12). At an interface the total field is `(1 + Gamma) E_inc`, where the reflected ray is the
limit of the image-method path whose reflection point slides onto the receiver: exactly zero for a
PEC plane and the tangential field, the Fresnel transmitted value for a dielectric, and
`|1 + Gamma exp(-2jk delta)|` times the incident field at a distance `delta` in front of the plane. The
full-wave answer follows the image method. Radar measured the cost of the rule: a scatter site on a
PEC wall, or 0.05 mm in front of one, kept only its line-of-sight row on 0.6.0 and echoed at the
free-space level, 32 dB too strong at 0.05 mm and infinitely too strong on the wall, where 0.5.0
(which had no rule) matched the image method.

Fixed-topology replay disagreed with discovery in the other direction (finding C-16): `reevaluate`
re-solved frozen reflection rows with no minimum segment at all, so a coincident-interaction row
that discovery rejects stayed `row_valid=True` when replayed.

**Decision.**
- Rule 1 applies to segments between two interactions only. The transmitter and receiver segments
  (TX, RX, a consumer source or sink, a Radar scatter site) are exempt; RayD 0.10.3 applies the
  EPC minimum to segments `1 .. B - 1` (RayD ADR-0048 amendment). For an endpoint exactly on the
  plane the reflection point is the endpoint, the zero-length segment adds 0 to the path length and
  segment visibility treats it as unobstructed. The reflection field kernels (primal, VJP and JVP
  in lockstep) take the reflected direction as the arrival direction of a receiver segment too
  short to normalize (as before), and now take the departure from the first interaction mirrored in
  its plane as the launch direction of such a source segment; it fell back to `+z`, which zeroed the
  field of a source on the plane. The mirrored direction is held constant in the field derivatives,
  which a zero-length segment leaves undefined. No NaN arises either way.
- The candidate pruning applies the same split: the reflection receiver test no longer checks the
  transmitter segment, and checks the receiver segment only when the receiver is an interaction.
- Interaction ends keep the rule. The reflection halves of a reflection-diffraction chain end at
  the diffraction point, so the chain validator sets `receiver_is_interaction` and the chain
  pruning checks the segment to the diffraction point; their transmitter and receiver ends are
  exempt. A scatter-chain leg ends at a sample drawn on a scatter face, a scattering interaction on
  its own surface: its validator sets `receiver_is_interaction`, so the leg segment into the sample
  keeps the rule. This also keeps out a reflection off the sample's own plane, whose reflection
  point is the sample itself and which would count the surface's own response twice.
- Diffraction endpoint distances are unchanged. A chain half without reflections still requires
  the diffraction point to be at least the threshold away from the transmitter or receiver, as
  does the coupled chain finalization: the UTD coefficient is singular at zero distance from the
  edge, so there is no finite limit to keep.
- Fixed-topology replay applies rule 1 exactly as discovery does: `reflection_epc_paths` passes
  the same minimum segment (the same fraction of the same scene diagonal) to the EPC, so a replayed
  row is valid exactly when fresh discovery at the same endpoints keeps it, in both directions.
  The enumerated fixed-winner re-solve uses the same call.

**Consequences.**
- Rows restored: reflections whose transmitter or receiver lies on or within the threshold of a
  reflecting plane (and the corresponding transmitter-side or receiver-side reflection halves of
  reflection-diffraction chains). Rows that touch a scatter-chain sample or a diffraction point are
  unchanged.
- Coupled R-D/D-R rows built in `coupled.cu` keep their own distance tests, which concern
  diffraction points only.

**Evidence.** `tests/propagation/consumer/test_endpoint_on_reflecting_plane.py`: a receiver or a
source on a PEC plane, and 0.05 mm in front of one, keeps its reflection row, and the total field
equals the image method (zero on the plane; `|1 - exp(-2jk delta cos theta)|` at 0.05 mm); the two
legs of a monostatic site 0.05 mm from the wall multiply to the squared one-way factor.
`tests/propagation/consumer/test_replay_validity.py`: fresh discovery and fixed-topology replay
agree on a receiver moved between 1 cm, 0.05 mm and 0 from a wall, and replay rejects, as discovery
does, a corner chain whose two reflection points meet on the shared edge. The C-1 scheme-equivalence and brute-force completeness
tests of the accuracy suite pass unchanged.
