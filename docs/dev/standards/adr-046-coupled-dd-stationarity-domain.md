# ADR-046: Certify coupled D-D stationarity and the ordinary UTD leg domain

- **Status:** Proposed
- **Date:** 2026-07-27
- **Kind:** One Channel-owned numerical correction to coupled sequential
  double-diffraction discovery. It changes which cid-7 candidates are
  published and their two stationary interaction points. It changes no public
  API or native ABI, adds no Torch/CPU production physics or fallback, adds no
  kernel launch, host/device transfer, synchronization, allocation, visibility
  query, or reduction-order change.
- **Related:** ADR-012 (stationary coupled diffraction), ADR-013 (the original
  fixed-16 coupled D-D solve), ADR-016 (vertex and inter-edge closure is one
  inseparable physical unit), ADR-017 (the rejected/default-off boundary
  taper), ADR-025 (RayD ownership of the pure-wedge operator), and RayD
  ADR-0037 (the shared local-coordinate line-Fermat primitive)

## Context

The full-wave boundary audit isolated a numerical/topological defect before
the unresolved ISB, RSB, and vertex transition terms are considered. Coupled
D-D discovery alternated two unbounded-line stationary-point maps exactly 16
times and published the last pair without a convergence check. It then admitted
all three ray legs longer than `1e-6 m`, although RayD's ordinary UTD state has
the stricter domain `UTD_MIN_DISTANCE = 1e-4 m`.

Those are different failures. The first publishes a nonstationary field whose
phase and spreading change abruptly when the edge pair is born or dies. The
second assigns an ordinary sequential-wedge field to a shared-vertex or
near-zero middle leg for which that field is undefined. Neither is a missing
uniform boundary term, and smoothing either output would hide the cause.

The three-cube RSB audit makes the distinction observable. At the audited lower
sample, the old 16-map fixed-point residual was approximately `6.96e-4 m` for
edge pair `33 -> 30` and `1.85e-4 m` for `42 -> 30`. After 32 maps the former
was still `5.54e-5 m`; after 64 maps it was about `5.36e-7 m`. The `35 -> 30`
limit has a shared-vertex/zero-middle-leg geometry. Iteration count alone
therefore cannot decide physical eligibility.

If accepted, this ADR supersedes only ADR-013's coupled-D-D fixed-16
stationary-point iteration and its `1e-6 m` leg-domain check. It does not alter
the coupled field operator, visibility tracing, or finite-edge transition
model.

## Decision

### Reduce the convex two-edge problem to one dimension

For unit edge axes `d1`, `d2`, finite-edge origins `O1`, `O2`, source `T`, and
receiver `R`, define

\[
Q_1(u)=O_1+u d_1,\qquad Q_2(v)=O_2+v d_2
\]

and the optical path length

\[
f(u,v)=\lVert Q_1(u)-T\rVert+
       \lVert Q_2(v)-Q_1(u)\rVert+
       \lVert R-Q_2(v)\rVert .
\]

The sum of norms of affine functions is jointly convex. RayD's existing
`first_order_diffraction_parameter` supplies the unbounded-line minimizer
`u*(v)`. Partial minimization preserves convexity, so
`h(v) = f(u*(v), v)` is convex. At a regular point its envelope derivative is

\[
g(v)=d_2\mathbin{\cdot}\left(
\frac{Q_2-Q_1}{\lVert Q_2-Q_1\rVert}+
\frac{Q_2-R}{\lVert Q_2-R\rVert}\right),
\qquad Q_1=Q_1(u^*(v)).
\]

The native prepare kernel accepts an ordinary interior stationary candidate
only from a strict signed bracket `g(lo) < 0 < g(hi)`. Endpoint equality is not
silently nudged inward: it belongs to endpoint/vertex physics. Every reduced
state used by the bracket must be finite and must have incoming, middle, and
outgoing leg lengths greater than the numerical geometry epsilon. That is only
the nonsingularity domain of `h'`: applying RayD's larger
`UTD_MIN_DISTANCE` to intermediate bisection samples could discard a regular
root elsewhere in the bracket. The ordinary UTD domain is therefore checked on
the final output legs, where the field would actually be published.

Every reduced line solve and derivative is evaluated in coordinates relative
to edge 1: `tx-O1`, `O2-O1 + v*d2`, and `u*d1`. Absolute `Q1` and `Q2` are
reconstructed only for the output ABI. This preserves the direction of a
short-but-ordinary middle leg; constructing `Q=O+u*d` first and subtracting
two absolute FP32 points produced a measured `3.49e-5` false gradient on a
`126.6 micrometre` leg even though the signed root bracket was only
`4.37e-11 m` wide.

The bracket is narrowed by at most 32 FP32 bisections. If the midpoint rounds
to an endpoint, iteration stops and the final invariant below decides whether
the representable bracket is sufficient. An exact-zero midpoint is retained
only when its immediate representable neighbours restore strict negative and
positive signs; a flat zero interval is not assigned an arbitrary point.

Thirty-two is a fixed upper bound, not a claim of universal convergence for
arbitrary scene scale. The result is publishable only if the final interval is
strictly inside edge 2, both corresponding `u*(lo)` and `u*(hi)` values are
strictly inside edge 1, and the output pair passes an explicit scale-aware
root-location certificate for the analytic first-line solve and bracketed
reduced root.

### Certify the analytic first-line solve and bracketed reduced root

The output `v` is the midpoint of the final bracket. Its physical partner is a
fresh evaluation `u = u*(v)`, not an average of `u*(lo)` and `u*(hi)`. The
strict convex bracket proves existence. Its final widths directly bound the
represented interaction-point uncertainty:

\[
\Delta_2=hi-lo,\qquad
\Delta_1=\lVert Q_1(u^*(hi))-Q_1(u^*(lo))\rVert .
\]

Both must be finite and no larger than

\[
\tau_x=8\,\epsilon_{32}\,S_{local},
\]

with

\[
S_{local}=\max(L_1,L_2,r_{T1},r_{12},r_{2R}).
\]

Unlike a fixed-point replay, these are mathematical root-position bounds. The
edge-2 root is inside the strict signed bracket. Edge 1 has no iterative root:
RayD's local-coordinate weighted-axial formula is the analytic minimizer for
each represented `v`, and `Delta_1` bounds how its image moves over the edge-2
root bracket. The finite fixed arithmetic of that shared formula is covered by
the same `8 * epsilon32 * S_local` output-position budget and by RayD's direct
analytic, translation, scaling, axis-reversal, Dual, and finite-difference
contract tests. Channel does not attempt to certify the RayD owner by replaying
or numerically solving the same Fermat equation a second time.

The two physical tangential derivatives are

\[
g_1=d_1\mathbin{\cdot}\left(
\frac{Q_1-T}{r_{T1}}+\frac{Q_1-Q_2}{r_{12}}\right),\qquad
g_2=d_2\mathbin{\cdot}\left(
\frac{Q_2-Q_1}{r_{12}}+\frac{Q_2-R}{r_{2R}}\right),
\]

but their values at one rounded FP32 parameter are diagnostics, not absolute
acceptance residuals. A fixed gradient threshold or a second FP32 sign solve is
not condition-number invariant: on the audited `126.6 micrometre` middle leg
the local-coordinate edge-2 residual was `7.9e-8`, while the nearest
representable analytic edge-1 parameter had residual `9.25e-6` despite a
`4.37e-11 m` edge-2 root bracket. Three ordinary three-cube witnesses likewise
had fully resolved position intervals while a repeated edge-1 derivative probe
missed its sign by only nanometres. The analytic first-line owner, strict
edge-2 bracket, and two position widths certify the pair without hiding that
geometry behind an arbitrary gradient, replay, or world-coordinate tolerance.

A sequential replay `P2(P1(v)) - v` is explicitly not an acceptance gate. It
is another evaluation of a finite-precision line map, not a bound on the root.
The audited single-cube witness had a strict one-ULP bracket and was within
`1.8e-8 m` of the independent FP64 joint Fermat root, while the old RayD
rotation-based `P2` replay differed by `7.45e-7 m` and falsely device-asserted.
RayD ADR-0037 replaces that shared line primitive with its local-coordinate
weighted-axial closed form, but Channel's certificate remains the physical
edge-2 bracket and both position widths rather than depending on replay
equality.

No fixed iteration result is published merely because the loop ended. Once a
row has a strict, regular, interior physical bracket and all three output legs
are in the ordinary UTD domain, failure of the position invariant is an internal
numerical failure and device-asserts. It is never converted to an inactive
candidate, zero field, alternate backend, or detached result. A missing strict
bracket, a shared vertex, an intersecting-edge zero middle leg, a collinear
duplicate/flat interval, or an endpoint solution is instead a physical-domain
rejection for this ordinary sequential-D-D component.

### Keep three boundary tolerances under their physical owners

The existing `kGeometryEpsilon = 1e-6 m` remains only the finite-edge interior
and same-line geometry tolerance. `UTD_MIN_DISTANCE = 1e-4 m` remains RayD's
ordinary UTD field domain. Visibility continues to own its trace-origin bias.
No one of these constants is reused as another owner's boundary rule.

The existing geometry contract is FP32 absolute world space; Channel has no
world-origin rebasing ABI in this route. This decision certifies the geometry
that those FP32 tensors actually represent. It does not promise invariance
under arbitrarily large world translations or recover detail smaller than the
world-coordinate ULP. Crucially, it also does not hide lost precision by
inflating a stationarity tolerance with the absolute world coordinate: if an
input translation destroys the local signed bracket or Fermat certificate,
the native operation fails loudly. A future local-coordinate/rebased scene
contract would be a separate numerical and ABI decision, not an implied
property of this solver.

This correction does not introduce a taper or smoothing pass. ADR-017 remains
default-off. Vertex diffraction, endpoint-uniform diffraction, and inter-edge
coupling remain the ADR-016 physical closure that must be implemented and
validated together rather than approximated by admitting an ordinary D-D row
outside its domain.

## Cost and numerical order

The old fixed-16 alternating solve called a first-order line map 32 times per
candidate. The regular new path calls it 37 times: two initial bracket states,
at most 32 midpoint states, two final endpoint states, and one fresh midpoint
solution state. The isolated exact-zero check can add two neighbour states,
for a maximum of 39 calls. Thus the stationary primitive count rises by
`15.625%` on the regular full-iteration path and at most `21.875%` on the
exact-zero path. It also adds the direct gradient dot products and norms
required by the reduced derivative, plus the final position-width checks; it
no longer performs two sequential residual probe calls.

This is work inside the existing prepare kernel. The three visibility queries,
kernel-launch count, result shape, compact-cardinality boundary, memory traffic
outside the kernel, and host synchronization count are unchanged. Runtime and
end-to-end impact must be measured; operation counts are not substituted for a
performance result.

## Required evidence before acceptance

1. A direct-native CUDA regression contains: an ordinary construction that
   the old 64-map iteration still left materially unconverged; three-cube-like
   `33 -> 30` endpoint/near-zero, `35 -> 30` shared-vertex/zero-middle-leg, and
   `42 -> 30` slow-but-ordinary cases; independent FP64 stationary points for
   the published rows; the exact single-cube `rx=24362, edge 13 -> 9` false-
   assertion witness; diagnostic double-Fermat-gradient checks; and exact
   inactive classification for the singular rows.
2. Reversing source/receiver order, swapping the edge order, and reversing both
   edge-axis parameterizations preserves candidate classification, path length,
   and physical interaction points within a scale-aware FP32 tolerance.
3. Targeted native contract tests and the Channel CUDA tier pass in `witwin2`.
4. Single-cube and three-cube deterministic-vs-full-wave benchmarks are rerun
   with the full-wave references byte-for-byte unchanged and every boundary
   taper disabled. The audit reports DD pair births/deaths and localized
   ISB/RSB/shadow continuity metrics separately from global fit metrics.
5. Because this is a numerical change, applicable nightly evidence includes
   the benchmark deltas and prepare-kernel/end-to-end timing; acceptance does
   not follow from a visually smoother plot.

## Consequences

- Ordinary coupled D-D fields cannot be published without RayD's analytic
  first-line minimizer, a strict signed reduced-root bracket, and a bounded
  pair of interaction positions.
- Shared-vertex and sub-`UTD_MIN_DISTANCE` legs stop masquerading as ordinary
  sequential UTD propagation. Their removal can expose a missing vertex term;
  that visible gap is preferable to filling it with an out-of-domain field.
- A genuine physical candidate that cannot meet the scale-aware invariant in
  the fixed budget fails loudly and requires a new numerical decision rather
  than a larger hidden iteration count.
- The remaining ISB/RSB/vertex discontinuity is still a boundary-physics task.
  This ADR removes two confounders but makes no claim that it supplies the
  missing uniform field.
