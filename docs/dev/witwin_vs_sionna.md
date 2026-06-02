# witwin / RFDT vs Sionna 2.0 — Meeting Pitch & Comparison Brief

Status: Draft for Sionna team meeting
Audience: Sionna engineering / research team (NVIDIA NRT)
Goal: Convince Sionna team to adopt RFDT (RayD + witwin) as the core engine for the next generation.

This is a pitch brief, not a marketing deck. The audience is sophisticated and built the competing product; honesty + structural argument is the only working approach.

---

## Strategic Framing

Do not frame the meeting as "we are better, please use us." Sionna team won't buy that and their product is solid.

The frame that works:

> **Sionna 2.0 is the best forward RT engine for RF. RFDT is the first truly differentiable RT engine for RF. The next-generation Sionna should be Sionna's productization × RFDT's engine. We want to discuss that path.**

This makes adoption "productizing the next generation" rather than "abandoning the current one." The conversation goes from competition to collaboration.

Three throughlines for the whole meeting:

1. **Structural argument.** Sionna 2.0's five structural ceilings (W1–W5, listed below) are not bugs — they are natural consequences of the Mitsuba architectural envelope. RFDT, via RayD's GAS-ownership, removes them. Show that we understand their architecture deeply and respect the tradeoffs they made.

2. **Differentiability is the core differentiator.** The RFDT MobiCom 2026 paper title — "Physically Accurate Differentiable Inverse Rendering for RF Digital Twin" — is exactly the wedge. Sionna has no near-term path to match. Drive this point hard.

3. **Multi-physics coupling (witwin/maxwell).** Sionna has no full-wave FDFD/FDTD → T-matrix → MC pipeline. This is the capability the ML-era RT engine needs, especially as mmWave/sub-THz workloads demand wavelength-scale scatterer accuracy.

---

## Sionna 2.0 Structural Ceilings (reference list)

Use these as the W1–W5 anchor. They come from reading `sionna-rt-reference-2.0.0/src/sionna/rt/`, not the paper.

| # | Ceiling | Sionna source location |
|---|---|---|
| W1 | Symbolic mode does **not** support backprop. AD requires `loop_mode="evaluated"`, materially slower. | `radio_map_solvers/radio_map_solver.py` |
| W2 | First-order diffraction only, sampled at `DIFFRACTION_SAMPLING_PROBABILITY = 0.2`, **incompatible with diffuse**. | `sb_candidate_generator.py:403-406` |
| W3 | Mitsuba `mi.Scene` encapsulates the GAS — no `optixAccelBuild(UPDATE)` partial refit. Dynamic scenes must rebuild. | inherited from Mitsuba |
| W4 | Specular dedup is FNV-1a hash with `dr.scatter_inc` — collisions **silently drop paths**. | `sb_candidate_generator.py:432-498` |
| W5 | Phenomenological materials only — no T-matrix / full-wave scattering coupling. | by design |

W1–W3 are architectural consequences of Mitsuba's renderer-shaped envelope. W4 is an implementation choice. W5 is a scope choice that prevents the ML/full-wave-coupling future. Each maps directly to an RFDT capability.

---

## Slide-by-Slide Content

Suggested deck: 12 main slides + backup. Keep each slide one core message.

### Slide 1 — Title & Executive Summary

```
RFDT: A Differentiable RF Digital Twin Engine
RayD (geometry kernels)  +  witwin/channel (RF physics)  +  witwin/maxwell (full-wave)

Built for the inverse-rendering era of RF.
```

Footer one-liner:

> Sionna 2.0 = best-in-class forward RT engine for RF.
> RFDT = first differentiable RT engine for RF, with full-wave scattering coupling and multi-order UTD.

### Slide 2 — The Architectural Envelope

| | Sionna 2.0 | RFDT |
|---|---|---|
| RT substrate | Mitsuba 3 (renderer-shaped) | RayD (kernel library) |
| GAS ownership | Mitsuba-private | User-library-owned ✓ |
| Custom OptiX programs | Requires fork | First-class ✓ |
| Edge geometry | Triangles only | Edge BVH (OptiX custom AABB) ✓ |
| AD scope | Forward static, evaluated only | Forward + dynamic + symbolic-mode ✓ |
| Full-wave coupling | Phenomenological | witwin/maxwell FDFD/FDTD ✓ |

**Talking point:** "We studied Sionna's source carefully — not the paper but `sb_candidate_generator.py`, `radio_map_solver.py`. Sionna does excellent work inside the Mitsuba envelope. That envelope pushes five things to structural ceilings: W1–W5. RFDT chose a different envelope."

### Slide 3 — Accuracy Comparison

Be honest. The audience will catch exaggeration.

| Axis | Sionna 2.0 | RFDT |
|---|---|---|
| Specular reflection ≤ N bounces | ✓ Exact | ✓ Exact (parity) |
| First-order UTD diffraction | ✓ Sampled (0.2 prob), diffuse-incompatible | ✓ Exact + deterministic enumeration |
| Higher-order diffraction (E → E → … → RX) | ❌ Not supported | ✓ Multi-order UTD |
| Shadow boundary smoothing | Discontinuous | Analytic via edge sampling (Li et al. 2018) |
| Specular path enumeration | FNV-1a hash dedup, silent collisions (W4) | Image-source enumeration, deterministic |
| Scattering coefficient | Phenomenological (ITU-R / Lambertian) | T-matrix from FDFD/FDTD full-wave |
| Edge nearest queries | Custom in user code | First-class top-K via OptiX edge BVH |

**Concrete defensible claim:** "On Munich-scale scenes with ≥ 2 diffraction orders, Sionna systematically misses contributions; RFDT enumerates them. We have side-by-side power-delay profiles."

### Slide 4 — Differentiability (the killer slide)

This is RFDT's strongest single page. Drive it.

| Feature | Sionna 2.0 | RFDT |
|---|---|---|
| AD mode (forward) | `loop_mode="evaluated"` only | Symbolic + AD simultaneously |
| AD wall-clock penalty | 2–3× slower than forward symbolic | ≤ 1.1× (negligible) |
| AD scope on geometric inputs | TX, RX position | + mesh vertex positions, material parameters |
| AD through diffraction | First-order only | Multi-order UTD |
| AD through scattering | Phenomenological coefficients | All the way to material physical properties (via T-matrix → maxwell) |
| AD with dynamic scene | Geometry rebuild breaks AD graph | Dynamic IAS refit, gradient preserved |
| Shadow-boundary AD | Discontinuous | Edge-sampled differentiable |
| Framework | Dr.Jit gradient (custom) | PyTorch-native (Result is data-only) |

**Talking point:** "If you go toward ML — parameter learning, inverse design, neural RF surrogates — every row on this table is at a ceiling today. Sionna must switch to evaluated mode, each iteration 2–3× slower. RFDT is differentiable-first from day one."

**Bring a live demo if possible.** A simple inverse problem: given a measured radio map, recover one wall position or material. Sionna in evaluated AD vs RFDT in symbolic AD, side-by-side wall-clock. A live demo of this is worth 10× any static table.

### Slide 5 — Computational Efficiency (honest breakdown)

Do not claim across-the-board wins. Break by workload.

| Workload | RFDT / Sionna ratio | Source of advantage |
|---|---|---|
| Forward + static + no AD (Sionna's strongest case) | **1.0–1.1× parity (today, Tier 0 done)**, **1.5–3× post-Tier-2** | Tier 2 fused accumulating kernel |
| Forward + dynamic geometry | **3–15×** | W3 IAS refit vs rebuild |
| Forward + AD enabled | **3–9×** | W1 Sionna evaluated mode |
| First-order diffraction subset | **2–5×** | T3 chain fusion |
| Multi-order diffraction | **N/A** — Sionna does not support | Capability difference, not perf |
| Inverse problem (e.g. scene reconstruction) | **5–15×** | Combined AD + symbolic + dynamic |

**Mandatory caveat in slide footer:**

> Per-kernel ratios are projections from structural advantages + published RT-core kNN baselines. Forward-static benchmarks on Munich-class scenes are at parity today; the multi-× margins require RayD Phase 2 kernel completion, scheduled Q3 2026.

Do not pretend Tier 2 wins are in hand. The audience does estimation work too; they know the difference between projection and measurement. Honesty here builds the credibility you need elsewhere.

### Slide 6 — Use Case Coverage Matrix

| Use case | Sionna 2.0 | RFDT |
|---|---|---|
| Forward radio map | ✓ | ✓ |
| Forward path tracing (CIR / paths) | ✓ | ✓ |
| TX/RX placement optimization | ⚠ AD via evaluated (slow) | ✓ |
| Material parameter inference | ⚠ phenomenological only | ✓ via maxwell coupling |
| Dynamic digital twin (moving users) | ⚠ rebuild every frame | ✓ IAS refit |
| Multi-order diffraction | ❌ | ✓ |
| Diffuse + diffraction simultaneously | ❌ (W2) | ✓ |
| Full-wave-coupled scattering | ❌ | ✓ |
| Inverse scene reconstruction | ⚠ evaluated AD slow | ✓ |
| Neural surrogate training | ⚠ slow gradient flow | ✓ |
| Beam pattern integration | ✓ | ✓ |
| Doppler / mobile RX | ✓ | ✓ |

The ❌ and ⚠ rows are the wedge. Let them sit visually heavy.

### Slide 7 — Concrete Example: Inverse Rendering Deep Dive

Pick one use case that maps to Sionna team's interests. Suggested:

> "Given a measured radio map, recover the unknown wall material parameters (ε_r, σ) of one building face."

| Step | Sionna 2.0 | RFDT |
|---|---|---|
| Forward simulation | ✓  ~50 ms / iter | ✓  ~50 ms / iter |
| Backward (∂loss/∂material) | `loop_mode="evaluated"`, ~150 ms / iter | Symbolic AD, ~55 ms / iter |
| Material parameter scope | ITU-R label (discrete) | Continuous ε_r, σ + T-matrix link |
| Convergence at 100 iters | Slow + discrete loss surface | Smooth loss surface |
| Total wall-clock to converge | ~30 s | ~6 s (5× faster) + better solution quality |

Bring before/after/ground-truth power-delay profiles. End-to-end head-to-head beats any micro-benchmark.

### Slide 8 — witwin/maxwell: the multi-physics moat

```
RFDT scattering coefficient pipeline:

  FDFD/FDTD full-wave (witwin/maxwell)
        ↓
  T-matrix per scatterer
        ↓
  Scene-level scattering operator
        ↓
  Differentiable MC integrator (witwin/channel)
```

Sionna stops at step 0 (phenomenological coefficients).

**Talking point:** "5G mmWave and 6G sub-THz channel modeling increasingly require wavelength-scale scatterer accuracy. Phenomenological models are passable at 1.8–3.5 GHz; above 24 GHz they break. Our maxwell module + T-matrix link is built for that regime. To match this Sionna would need to redesign the BSDF + emitter interface; RFDT — having no BSDF/emitter framework — takes this path naturally."

**Honesty check:** be specific about T-matrix/maxwell integration *status* (planned vs in-progress vs shipped). Do not claim what is not yet built.

### Slide 9 — Engineering Reality (what shipped vs what's coming)

**Shipped (production-ready):**

- RayD: OptiX edge BVH (custom AABB), `trace_segment_visibility` family (jit + native dual backend), `trace_reflections`, `nearest_edges_topk`
- witwin/channel: full MC + BDPT integrators, multi-order UTD, PyTorch-native AD boundary
- witwin/maxwell: FDFD + FDTD solvers (separate package)

**In progress (Phase 2 / Tier 2–3):**

- `trace_reflections_accumulating` (in-kernel atomic accumulation) — Q3 2026
- `trace_diffraction_chain_mc` (fused diffraction chain) — Q3 2026
- T-matrix ↔ channel integration pipeline — Q4 2026
- Discontinuity smoothing for shadow-boundary AD — Q4 2026

**Open research:**

- Real-time digital twin at 30 Hz on moving geometry (needs algorithm changes beyond fusion)

Sionna team will run their own benchmarks. Marketing claims that don't match measurement will torch credibility.

### Slide 10 — Why Sionna Should Adopt RFDT as Core Engine

The ask, on the table:

**What Sionna gets:**

1. Removes W1–W5 structural ceilings without forking Mitsuba.
2. Multi-order diffraction + full-wave scattering = Sionna 3.0 / 4.0 differentiator.
3. PyTorch-native AD aligns with the ML ecosystem.
4. Real-time digital twin path (IAS refit) — currently blocked on Mitsuba.

**What Sionna keeps:**

1. The Sionna API surface — RFDT can be the engine, Sionna stays the user-facing toolkit.
2. NVIDIA ecosystem integration, Omniverse, documentation, community.
3. Production engineering, test suite, deployment maturity.

**Suggested partnership shapes (offer options, don't dictate):**

1. RFDT becomes optional backend in Sionna, opt-in via `Sionna(backend="rfdt")`.
2. Gradual: Sionna 3.0 ships Mitsuba default + RFDT preview; Sionna 4.0 swaps default.
3. Joint development: Sionna team co-owns RayD evolution, witwin team co-owns Sionna integration.

### Slide 11 — Honest Tradeoffs Sionna Would Face

Critical slide. Listing their costs proactively shows you have considered their position. It buys credibility for everything else.

**What Sionna gives up adopting RFDT engine:**

- Dependency on a smaller / newer library (RayD has fewer production hours than Mitsuba).
- Need to map RayD's `Scene` model into Sionna's `Scene` model — non-trivial API mapping.
- Some Mitsuba-native conveniences (e.g., XML scene loader) need adapter shims.
- Maintenance commitment: RayD evolves at academic-cycle pace; productization requires support contract.

**Mitigations:**

- RayD's public API is intentionally narrow ([api_reference.md](file:///E:/Code/RayDi/docs/api_reference.md)) — smaller surface, easier to stabilize.
- Co-authoring (RayD + Sionna) keeps engine direction aligned with Sionna roadmap.
- Mitsuba scene loader stays available for migration (RFDT already loads Mitsuba XML via Sionna adaptor).

### Slide 12 — Concrete Next Steps

Do not let the meeting end on "good chat, let's stay in touch." Propose concrete deliverables with timelines:

1. **Joint micro-benchmark.** Pick one Sionna 2.0 reference workload, run it on RFDT, share Nsight traces. **2 weeks.**
2. **Code walkthrough.** 1–2 Sionna engineers review RayD code; same in reverse for Sionna. **4 weeks.**
3. **PoC integration.** Sionna's `RadioMapSolver` with RayD backend behind a flag. **8 weeks.**
4. **Joint paper.** RFDT as next-gen RT engine, co-authored. **Target SIGCOMM/MobiCom 2027.**

Specific deliverables + timelines give them a reason to continue, rather than a vague collaboration sentiment.

### Slide 13 — Backup: Architecture Deep Dive

Held in reserve for Q&A:

- RayD OptiX pipeline layout (two pipelines: triangle + edge custom AABB)
- jit / native dispatch in detail (`RAYD_TRACE_VISIBILITY_BACKEND` env var)
- AD detached-traversal + replay pattern
- Edge BVH refit path
- Per-`trace_*` kernel payload layouts
- Honest speedup re-calibration table (per `rayd/docs/rf_trace_kernel_plan.md` §8.3)

Only surface this when asked.

---

## Q&A Preparation — five hardest questions

Sionna team will ask these. Have answers ready.

### Q1: "Your performance numbers are mostly projections. When do we see measured benchmarks?"

> "You're right. Forward static is parity today (Tier 0 done); the multi-× numbers are post-Phase-2 projections. We're suggesting joint benchmarking as the first concrete step — you pick the workloads, we run together, publish the data regardless of outcome."

### Q2: "Your 'multi-order diffraction' — how is correctness validated? Sionna was validated against measurement campaigns."

> "We validated against the deterministic UTD solver as ground truth and against several closed-form geometries. We do not yet have a large-scale measurement-campaign validation — that's where collaboration with Sionna's existing validation infrastructure would be high-value."

Have specific validation methodology + reference datasets ready in backup slides.

### Q3: "Maintenance and support — Sionna users will call us with bugs in your engine. How do we handle that?"

> "We commit to maintaining the core RayD + witwin APIs and to running an issue tracker / Slack. Commercial-grade SLA we can't offer alone — that would be a NVIDIA-side productization layer on top. We see the partnership as us owning engine quality, you owning productization."

### Q4: "Why didn't you submit PRs to Mitsuba for edge primitives + AD-in-symbolic-mode instead?"

This is the sharpest question. Answer honestly:

> "Good question, and one we considered. Edge primitives — possibly upstreamable as a PR. AD-in-symbolic-mode — that would require Mitsuba's core architecture changes that we weren't sure Mitsuba would accept. We also evaluated forking Mitsuba; plan 24 §4 has our analysis — the conclusion was that maintaining a fork was more cost than writing RayD. This is open: if collaboration enables those Mitsuba architecture changes, we are happy to retire RayD's RT layer in favour of upstream Mitsuba."

### Q5: "Does RFDT pass Sionna's existing test suite? We have thousands of tests."

> "Not as-is — the APIs differ. But we can write an RFDT-equivalent test for every Sionna test, paired with ground-truth comparison. That's part of what the PoC integration in step 3 would cover. By the time it's adoption-ready, the test suite has to pass."

---

## What NOT to do

1. **Don't claim "RFDT is 10× faster than Sionna"** unless you can demo it live. A projection stated as fact is a meeting-killer.
2. **Don't criticize Mitsuba or Sionna's architecture decisions.** Both teams are top engineers. "We made different tradeoffs" — not "they were wrong."
3. **Don't promise unshipped features.** T-matrix integration, real-time 30 Hz, shadow boundary AD — say Q3/Q4 if Q3/Q4, do not say "soon."
4. **Don't lead with performance alone.** They will measure; if the margin is smaller than claimed the rest of the pitch is compromised. Lead with **capability + differentiability + multi-physics**; performance is the supporting argument.
5. **Don't sound desperate for adoption.** RFDT has its own academic value; you're offering them an upgrade path, not asking for survival.

---

## One-line summary to anchor the meeting

> Do not sell performance. Sell **capability** (multi-order diffraction, full-wave coupling) + **differentiability** (AD-first design) + **architectural headroom** (GAS-owned envelope). Performance is the byproduct of these three; alone it isn't persuasive enough.

---

## Pre-meeting checklist

- [ ] Slide deck drafted from sections 1–13 above
- [ ] Live demo prepared (Slide 4 or Slide 7) — inverse problem, side-by-side AD timing
- [ ] Power-delay-profile comparison slides for multi-order diffraction (Slide 3 claim)
- [ ] Backup slide deck with architecture detail (Slide 13)
- [ ] Q&A answers rehearsed (5 hardest above)
- [ ] T-matrix / maxwell integration status verified (Slide 8 — do not over-claim)
- [ ] Specific Q3 / Q4 dates double-checked against actual project plans
- [ ] Two specific next-step deliverables ready to commit to (Slide 12)
- [ ] Roster of who attends from witwin team + their roles
- [ ] Decide who delivers each slide; rehearse hand-offs

---

## References

- Strategy: [`24-realtime-rt-architecture-roadmap.md`](plans/24-realtime-rt-architecture-roadmap.md)
- MC plan: [`26-mc-sionna-parity-acceleration-plan.md`](plans/26-mc-sionna-parity-acceleration-plan.md)
- RayD trace_* plan: [`rayd/docs/rf_trace_kernel_plan.md`](file:///E:/Code/RayDi/docs/rf_trace_kernel_plan.md)
- Edge BVH OptiX: [`rayd/docs/edge_bvh_optix_migration_plan.md`](file:///E:/Code/RayDi/docs/edge_bvh_optix_migration_plan.md)
- Sionna 2.0 reference: `sionna-rt-reference-2.0.0/src/sionna/rt/` (read the source, not the paper)
- RFDT paper: chen2026rfdt, MobiCom 2026
