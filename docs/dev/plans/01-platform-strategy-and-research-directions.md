# Platform Strategy, Moats, and Exploratory Directions

Status: Active
Category: Plan
Last reviewed: 2026-05-22

This document is the strategy layer above `plans/00-research-feature-roadmap.md`. Plan 00 is the feature inventory ("what"); this document is the thesis ("why these, in what order, and what only witwin can do"). It also records exploratory brainstorm directions that are not yet committed plans.

---

## Part A — Strategic Framing

### Platform DNA

witwin is differentiable rendering (Tzu-Mao Li lineage) applied to RF, built by a wireless-systems research group (Xinyu Zhang, Dinesh Bharadia). Its identity is not "another ray tracer". It is a differentiable, GPU-native, full-wave-backed RF research instrument. Every strategic choice should reinforce that identity rather than chase feature parity with Sionna RT or Wireless InSite.

### The Moat Thesis

Two capabilities are structurally hard for competitors and should drive the roadmap:

1. **End-to-end differentiability.** Sionna has partial AD; InSite has none. witwin can make geometry, materials, antennas, RIS, and motion all differentiable through one solver. This is the precondition for inverse problems, calibration, and learning-based research.

2. **Full-wave physics grounding.** witwin owns a full-wave solver (`witwin/maxwell`, FDFD/FDTD). It can derive material response, rough-surface scattering, RIS unit-cell behavior, and antenna near-field from first-principles simulation instead of ITU empirical models, and keep autodiff through that derivation. This is structurally impossible in Sionna because its BSDF inheritance hijacks Mitsuba's dispatch and has no slot for a `(k_in, k_out, polarization)` T-matrix. See the T-matrix scattering plan (maxwell is the producer, channel the consumer).

Forward simulation performance (path discovery, diffraction, RayD OptiX migration) is now near mature. The platform's remaining value is not "compute more path families" but "be differentiable, be physically grounded, and be a research workflow".

### Big-Feature Layers

| Layer | Big feature | Why next-gen | witwin-specific angle | Status |
| --- | --- | --- | --- | --- |
| **L0 Foundation** | Differentiable correctness (reflection / geometry / diffraction FD-validated; reverse-mode everywhere) | Without it, "differentiable platform" is a promise, not a product | path solver already FD-validates as the reference | plan 29, in progress |
| **L1 Moat** | Full-wave-grounded physics: T-matrix materials/scattering, near-field full-wave ⊕ far-field RT multi-scale coupling, differentiable rough-surface scattering | RT tools use ITU empirical materials; witwin derives them from full-wave, differentiably | competitors cannot replicate without a full-wave backend | only the T-matrix plan exists; multi-scale coupling is near-empty |
| **L1 Moat** | Inverse-problem / calibration toolkit (joint geometry / material / pose / RIS inversion; CIR/CFR/RSS/ToF observation adapters; sparse-noisy-robust losses) | Research has shifted from forward sim to inferring scenes from measurements | the payoff of L0 + L1 physics | roadmap P0 #2, not started |
| **L2 Generality** | General mixed-interaction path graph (R / D / **T** / scattering, arbitrary chains) + transmission/penetration | Real urban/indoor channels are mixed chains | unified differentiable path-state | diffraction-specific builders today; no transmission |
| **L2 Generality** | Wideband / dispersive / THz materials and propagation | Research has moved past narrowband sub-6 | frequency-dependent materials fed directly from maxwell | scalar eps_r/sigma, narrowband |
| **L2 Generality** | RIS / metasurface as first-class differentiable objects | Large literature, strong optimization story | model RIS unit cells with maxwell near-field, embed in RT | none |
| **L2 Generality** | Dynamic scenes + Doppler + time-consistent path identity | Mobility / V2X / ISAC / robotics need temporal consistency, not snapshots | differentiable motion optimization | re-run static only |
| **L3 Ecosystem** | Benchmark & reproducibility suite + dataset / supervision export | Reproducible benchmarks build platform authority and feed ML research | path-level ground truth plus full-wave reference | scattered across test scripts |
| **L3 Ecosystem** | ISAC / sensing outputs (range-Doppler-angle, target attribution) | Joint communication-and-sensing is the hottest area | natural coupling with the `radar` subproject | none |
| **L3 Ecosystem** | Uncertainty quantification | Trustworthy digital twins need error bars | differentiable + MC supports it naturally | none |
| **L3 Ecosystem** | Antenna/array co-design, point-cloud / reconstructed scenes, multi-GPU city-scale | Real deployment and scale | — | array basics exist; point-cloud / multi-GPU none |

### The Three Bets

If forced to pick three, in order:

1. **L0 differentiable correctness.** Not optional; the precondition for everything below. Finish it first (plan 29).
2. **Full-wave-grounded physics (maxwell ⊕ channel multi-scale + T-matrix materials/scattering).** The only moat competitors cannot copy. Currently underweighted in plan 00 as a "future scattering feature"; it should be promoted to the top physics bet. A RT platform that derives materials, rough surfaces, and RIS responses from its own full-wave solver, differentiably, is genuinely next-generation rather than another ray tracer.
3. **Inverse-problem / calibration toolkit.** The way differentiability is monetized; turns the solver into a research instrument. Must follow 1 and 2 because it depends on trustworthy gradients and trustworthy physics.

One-line summary: forward simulation is near mature; the platform's next value is the three things that turn a solver into a research instrument — differentiability, full-wave physics, and a research workflow layer.

---

## Part B — Exploratory Directions (brainstorm 2026-05-22)

These are candidate directions raised for brainstorm, not committed plans. Each lists the opportunity, the witwin-specific angle, and the hard technical question that must be answered before it becomes a plan.

### 1. Scene Dynamics

Move from "re-run static snapshots" to a first-class time-dependent scene.

- **Capabilities:** rigid-body trajectories (vehicles, people, UAVs), articulated bodies (micro-Doppler), incremental BVH refit instead of full rebuild per frame, time-consistent path identity (birth/death of paths, event detection when an edge becomes visible), differentiable motion (optimize trajectories from channel observations).
- **witwin angle:** the monorepo's `witwin/genesis` module (physics/dynamics — scope to be confirmed) is the natural trajectory source; channel renders the RF; differentiability lets you invert motion from RF (tracking, localization). RayD already supports differentiable vertex updates without topology rebuild, which is the substrate for cheap per-frame refit.
- **Hard question:** how to maintain stable path correspondence across frames when topology changes (an edge appears/disappears) without re-enumerating from scratch, and how to define a differentiable event model at those transitions.
- **Depends on:** L0 geometry gradients (a moving object is a geometry gradient). Feeds Doppler (#3) and ISAC.

### 2. Real-Time / Interactive Radiomap

Sub-100ms radiomap updates when the transmitter moves or one object changes, for interactive design tools, RL environments, and teaching.

- **Approaches:** incremental recompute (only re-trace paths touched by the changed element), frame-to-frame path-topology caching with field-only refresh, progressive/adaptive sampling (coarse map instantly, refine in place), level-of-detail grids, and a learned surrogate / neural radiomap for instant preview backed by the exact differentiable solver for the final answer.
- **witwin angle:** the exact differentiable solver is the perfect teacher for a learned surrogate (multi-fidelity story); a fast forward eval also makes channel usable as the inner loop of RL / optimization. The `Tracer` already caches path-state across repeated traces, which is the seed of incremental update.
- **Hard question:** which fidelity mode owns interactivity vs differentiability vs exactness — pick per mode rather than forcing one path to be all three. Defining the cache-invalidation contract for "only this changed" is the crux.
- **Depends on:** path-state generalization; benefits from multi-fidelity / surrogate infra.

### 3. Scattering / Transmission / Doppler

Complete the interaction taxonomy (currently only reflection + diffraction).

- **Transmission (T):** Fresnel transmission + Snell refraction, path continuation through media, multi-layer walls. Unlocks outdoor-to-indoor (O2I) and through-wall sensing (a Bharadia-style area).
- **Diffuse scattering:** rough-surface scattering lobes. This is where the L1 full-wave moat shows up most concretely — derive a phase-accurate, full-angular, full-polarization scattering function (T-matrix / RF-BRDF) from maxwell instead of an ITU directive lobe, with differentiable roughness.
- **Doppler:** per-interaction Doppler from moving scatterers, not just moving endpoints; makes the CIR time-varying. Couples directly to dynamics (#1).
- **witwin angle:** T + scattering + Doppler together make the channel model physically complete and competitive with InSite, and scattering specifically is the moat feature.
- **Hard question:** tractability of a per-hit T-matrix lookup at 1e6+ rays, keeping autodiff through the T-matrix back into the full-wave solver, and the multi-scale mismatch (T at wavelength scale vs scene at km scale).
- **Depends on:** L1 full-wave coupling; Doppler depends on dynamics (#1).

### 4. Novel Scene Representations (SDF / Gaussian Splatting / Surfel)

The deepest fork, and the most aligned with the differentiable-rendering DNA. Scene is currently triangle meshes + extracted wedges.

- **SDF (signed distance fields):** smooth, differentiable geometry; ray-surface intersection by sphere tracing. Geometry gradients are clean with no triangle-boundary discontinuities, which could solve the hardest part of L0 (plan 29 Task 3) at the representation level. Neural SDF enables shape optimization from RF.
- **3D Gaussian Splatting:** scene as anisotropic Gaussians; strong for volumetric/soft scattering media and as a differentiable proxy for scenes reconstructed from images/lidar. Each Gaussian can carry a learned RF scattering response.
- **Surfels:** oriented surface elements with material; an explicit-surface, topology-free middle ground that maps naturally from point clouds for reconstructed digital twins.
- **witwin angle:** real-world digital twins start from point clouds / images, not CAD; SDF/GS/Surfel are the native outputs of reconstruction. Implicit/soft representations give clean geometry gradients (directly helping L0), and they connect channel to the neural-scene / inverse-rendering research community.
- **Hard question (the crux):** diffraction (UTD) fundamentally needs explicit edges/wedges, which meshes have and SDF/GS/Surfel do not. The likely answer is a hybrid: explicit edges for diffraction plus implicit/soft surfaces for reflection/scattering, or extracting edges from SDF zero-level-set curvature ridges. This tension must be resolved before committing.
- **Depends on:** nothing hard upstream, but it would change the scene-representation contract broadly, so it needs an architecture decision early.

### Cross-Cutting Connections

- **Geometry gradients (L0) ↔ novel reps (#4):** implicit/soft geometry may dissolve the discontinuity problem that plan 29 Task 3 is fighting on meshes. Worth a spike before brute-forcing the mesh boundary term.
- **Dynamics (#1) ↔ Doppler (#3) ↔ genesis:** one dynamics layer feeds both temporal path identity and per-interaction Doppler; `witwin/genesis` is the likely engine.
- **Scattering (#3) ↔ full-wave moat (L1):** diffuse scattering is the most concrete place the T-matrix advantage becomes visible.
- **Real-time (#2) ↔ surrogates ↔ inverse toolkit:** a fast forward path is also the inner loop for optimization and RL.
- **Reconstructed scenes (#4) ↔ inverse toolkit (L1):** novel reps are how real-world digital twins enter the platform.

### Suggested Sequencing Note

These four do not all sit at the same priority. Against Part A:

- #3 scattering is part of the L1 moat and should ride with the full-wave coupling work.
- #4 novel representations is high-leverage because it can help L0 geometry gradients and L1 reconstructed digital twins simultaneously; recommend a small architecture spike (how to keep diffraction edges) before committing.
- #1 dynamics and #3 Doppler are an L2 pairing best done together once L0 geometry gradients are trustworthy.
- #2 real-time is an L3 enabler (tools, RL, teaching) and a surrogate story; valuable but not a moat, so it should follow the moat work unless adoption/UX is an explicit near-term goal.

## Open Questions

- Is `witwin/genesis` the intended dynamics engine for scene dynamics, and what is its current scope?
- Does the full-wave coupling (L1) want a precomputed T-matrix cache per material/geometry, or an on-the-fly maxwell call for hero patches?
- For novel scene representations, is the near-term goal reconstructed digital twins (favoring Surfel/GS) or clean geometry gradients (favoring SDF)?
- Should real-time interactivity be pursued as a learned surrogate, an incremental exact solver, or both as explicit fidelity modes?
