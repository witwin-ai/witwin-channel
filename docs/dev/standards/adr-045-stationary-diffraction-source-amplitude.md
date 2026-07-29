# ADR-045: Correct stationary-diffraction source amplitude before boundary work

- **Status:** Proposed
- **Date:** 2026-07-27
- **Kind:** One RayD-owned numerical correction with no Channel ABI or public
  API change.  It changes direct-incident stationary diffraction fields and
  their native first-order derivatives.  It adds no production Torch/CPU
  physics, fallback, kernel launch, host/device transfer, synchronization,
  allocation, or reduction-order change.
- **Related:** ADR-012 (stationary coupled diffraction), ADR-016 (incomplete
  transition integral), ADR-017 (rejected/default-off boundary taper), ADR-025
  (RayD pure-wedge ownership), ADR-043 (published AD surface), RayD ADR-0035

## Context

The full-wave continuity audit found a source-state error that is independent
of the unresolved ISB/RSB/vertex closure problem.  RayD's shared stationary
UTD state used a unit transverse coordinate-basis helper to construct the
field radiated by the transmitter dipole.  Normalising the transverse
projection erased the short-dipole `sin(theta)` pattern; at the axial null the
basis helper selected an arbitrary nonzero fallback.

For a dipole moment `p`, propagation direction `r_hat`, and the package phase
convention, the direct incident state is

\[
\mathbf E_{\rm inc}(r)=
\left(I-\hat{\mathbf r}\hat{\mathbf r}^{T}\right)\mathbf p\,
\frac{e^{-jkr}}{2kr}.
\]

The norm of the vector projection, including its zero on axis, is part of the
field amplitude.  It is not a Jones-basis normalisation choice.

This correction cannot by itself close ISB, RSB, finite-edge endpoint, or
vertex transitions.  Counterfactual reweighting showed mixed changes in those
regions, so treating it as the complete continuity fix would be false.  It is
instead a correctness prerequisite that must be removed before attributing the
remaining residual.

## Decision

RayD's `direct_source_vector` uses the unnormalised transverse projection and
keeps its float/Dual implementations in the same template.  Channel does not
reconstruct or compensate this field in Torch, Python, or a second CUDA owner.

The affected Channel operation families are:

| Family | Expected effect |
|---|---|
| cid 2, direct D | Corrected source amplitude and native source/target/frequency AD |
| cid 7, D-D | Corrected direct incident state on the first D leg; the external second leg is unchanged |
| MC Sionna diffraction | Corrected direct incident Jones state and corresponding native AD |
| cid 3, R-D | Bitwise unchanged because the D leg consumes a frozen external incident field |
| cid 4, D-R | Bitwise unchanged because the D leg consumes a frozen external incident field |

The correction is evaluated with `isb_boundary_taper_width = 0`.  ADR-017's
heuristic taper remains default-off and is not used to accept this change.
The calibrated finite-edge corner surrogate remains unchanged until the
ADR-016-required `exact incomplete G + vertex + inter-edge coupling` unit has
passed its independent oracle and full-wave gates; deleting only one member of
that compensating unit is explicitly out of scope.

## Boundary-physics separation

After this prerequisite, physical boundary work follows four distinct
classifications rather than one overloaded Boolean `valid`:

1. **Model eligibility:** fixed scene/material support and nondegeneracy.
2. **Numeric domain:** ABI, finite inputs, and supported AD capability.
3. **Candidate cull:** performance-only and proven free of false negatives for
   required compensators.
4. **Physical boundary:** visibility, finite-face exit, finite-edge endpoint,
   and vertex ownership; every hard GO membership change requires its matched
   uniform physical term.

This ADR does not change any of those topology gates.  In particular, it does
not attempt to repair the currently inconsistent endpoint semantics in which
pure D clamps a stationary point to a finite-edge endpoint while R-D, D-R, and
D-D reject the same endpoint through a strict-interior test.  That change
requires its own accepted numerical ADR and the vertex/inter-edge evidence.

## Acceptance evidence

The proposal becomes Accepted only when all of the following are frozen:

1. **RayD analytic/native lockstep.** Axial zero, `sin(theta)` amplitudes,
   `cos(theta)` Dual tangent, native JVP versus fixed-winner central
   difference, and JVP/VJP contraction all pass in `witwin3`.
2. **Channel family matrix.** cid 2 and cid 7 change as declared; cid 3 and cid
   4 primal/JVP/VJP are exact unchanged sentinels; MC direct diffraction
   receives the expected amplitude-squared power change.
3. **Versioned full-wave benchmarks.** Single cube and `three_cube_320` are
   rerun from a fresh Channel build against the explicit RayD candidate.  The
   report includes complex field, envelope metrics, ISB, RSB, vertex/shadow
   regions, and convergence-aware complex jump scans.  Full-wave arrays are
   read-only and their hashes remain unchanged.
4. **No smoothing.** Every acceptance command records taper width zero.  No
   metric threshold is relaxed and no new empirical window is introduced.
5. **Build identity.** The evidence records Channel/RayD commit and dirty
   state, CUDA/driver/GPU, compiler, ABI header hash, and command line.  A dirty
   explicit RayD checkout is developer evidence only.

## Deployment boundary

The production `dependencies/rayd.lock.json` does not move while this ADR is
Proposed.  After acceptance, RayD must provide a clean, reachable commit and a
generated source-bundle manifest.  The lock then changes only the RayD commit
and source-manifest digest; the integration header hash, API version, binding
manifest, contract-coverage manifest, and public API snapshot remain unchanged.

## Consequences

- A confirmed source-physics bug is removed before fitting or attributing any
  boundary residual.
- Direct-D, the first D-D leg, and MC diffraction intentionally change.
- R-D and D-R supply exact sentinels against accidental contamination of the
  external-incident path.
- Remaining ISB/RSB/vertex jumps are not hidden: they become the input to the
  separately gated uniform-boundary research and implementation program.
