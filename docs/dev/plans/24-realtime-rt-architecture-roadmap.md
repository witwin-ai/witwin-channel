# Real-time Ray Tracing Architecture Roadmap

Forward-looking analysis of where `witwin.channel` and `rayd` should evolve, and
how that compares to staying on (or moving back to) Mitsuba 3 for a real-time
RF digital-twin target.

This is a strategy document, not a concrete implementation plan. It distills
trade-offs that are otherwise scattered across the codebase and the RayD repo.

## Authorship and Scope

RayD and `witwin.channel` share a single author. RayD was purpose-built
to serve channel's RF channel-simulation requirements; it was never
designed as a generic Mitsuba alternative. This document treats them as
one connected design space — RayD's API surface and kernel roadmap are
driven by channel's needs, and channel's solver internals are written
assuming RayD is available to grow alongside them.

The Mitsuba 3 comparisons in §4 and §7 are not advocating a switch —
that decision has been made and is settled. They serve two purposes:

1. **Retrospective check** that the original build-vs-buy choice still
   pays off as the project moves toward real-time digital-twin targets.
2. **Forward filter** that separates which RayD advantages are
   structural (worth preserving as design constraints when adding new
   features) from those that are patch-like (could be matched on
   Mitsuba with enough effort, and therefore should not be load-bearing
   in the architectural argument).

Read the comparison sections in that spirit — they grade design
decisions, not vendor choices.

## 1. Current Architecture Snapshot

Channel runtime layering today:

- `Scene.ray_test` / `ray_intersect` / `nearest_edge` → `rayd.Scene`
- Multi-bounce reflection has two execution modes:
  - Python loop over `scene.ray_intersect()` (deterministic / MC path tracing).
  - `rayd.Scene.trace_reflections()` — a **fused OptiX raygen + closesthit
    kernel** inside RayD (`src/multipath/reflection_trace.cu`).
- UTD higher-order diffraction (`diffraction_impl/builders.py`) is a chunked
  Python BFS calling `nearest_edge` + `intersect` + Dr.Jit scatter/gather per
  expansion order.
- `segment_visible` with `ignore_prim` re-fires the same ray up to 8 times in
  Python to skip ignored hits.

Mitsuba is reduced to a one-shot scene loader (`Scene.load_mitsuba`); runtime
queries never touch it.

The fundamental cost structure is **not BVH traversal**; it is:

1. Per-step OptiX launch overhead from many small `scene.intersect()` calls.
2. Full `Intersection` struct (≈60–80 B) written to global memory on every hit,
   even when the caller only needs `t` or `global_prim_id`.
3. Python ↔ Dr.Jit IR ↔ PTX round-trips per BFS expansion level.

## 2. witwin.channel Improvement Directions

Ordered roughly by ratio of (expected speedup) / (engineering cost).

### 2.1 Symbolic loop fusion — no architectural change

Most channel hot paths still run in Dr.Jit *recorded* (symbolic) mode and emit
one large fused PTX kernel per Python frame. Two leverage points:

- `segment_visible` `ignore_prim` loop ([scene.py:761](../../witwin/channel/core/scene/scene.py#L761))
  currently caps at `max_ignored_hits = 4` and falls back to up to 8 ray
  re-fires. Worth measuring whether the inner step is already symbolically
  fused or breaks the trace on `dr.any(unresolved)`.
- `intersect_rays_with_prim` ([scene.py:642](../../witwin/channel/core/scene/scene.py#L642))
  re-gathers `v0, v1, v2` and re-projects to reconstruct hit position/normal.
  Nine extra global loads per hit that the OptiX closesthit *already had in
  registers* but discarded.

Lowest-hanging fix in this category: have RayD expose a richer
`Intersection` (or a thin variant) that carries the OptiX-reconstructed
position + geometric normal directly. Save the re-gather entirely.

### 2.2 Higher-order diffraction chain fusion

`bvh_pairs` ([builders.py:402](../../witwin/channel/deterministic/path/diffraction_impl/builders.py#L402))
launches 18 probe rays per previous state per order. Each order is one Python
iteration, one full state-array materialization, and one Dr.Jit kernel.

Target architecture:

- A RayD-side `trace_diffraction_chain(rays, edges, max_order)` kernel that
  walks one "TX → edge → edge → ... → RX" path in a single OptiX launch.
- Payload carries UTD field components + edge bookkeeping; final write is one
  `Complex2f` per (TX, RX) pair plus per-path lineage.
- UTD math (currently `kernels/utd/utd_math.h`, `utd_types.h`) moves into the
  closesthit body.

Expected gain: **3–10× on multi-order diffraction**, dominated by removing
state-array round-trips between orders.

Risk: gradient story. The current Dr.Jit-symbolic path participates in AD via
the discontinuity smoothing being designed in [`22-deterministic-discontinuity-plan.md`](22-deterministic-discontinuity-plan.md).
Fusing into a closesthit forces analytic gradients to be computed inside the
kernel; the smoothing scheme has to be representable in straight-line GPU code
without Dr.Jit-traced control flow.

### 2.3 Receiver-side reduction inside kernel

Today `RadioMapResult` is assembled by Python after each batch. For
real-time, the receiver grid wants `atomicAdd` into bins straight from the
closesthit, with one OptiX launch per frame instead of one per (TX × order).

This is mostly a witwin concern, not RayD: the bin layout, polarization
projection, and complex accumulation live in channel.

### 2.4 Persistent scene state

`Scene.sync()` (RayD `optix_sync_ms`) currently dominates dynamic-mesh
updates. For digital-twin scenarios where only a handful of transmitters or
walls move, the IAS-refit path is the right primitive — already supported by
`rayd_scene.update_mesh_vertices(...)` + `sync()`. The witwin side should
**not rebuild `_structure_meshes` on every endpoint move**; it currently does
([builder.py:46](../../witwin/channel/core/scene/builder.py#L46) gates this but
the surrounding rebuild logic is conservative).

## 3. RayD Improvement Directions

RayD is intentionally narrow. The right direction is not "expose a generic
shader API" (that drifts toward OptiX SDK / Falcor and erases the small-API
advantage). The right direction is "add more first-class fused kernels for RF
/ acoustic patterns."

### 3.1 More `trace_*` fused kernels

`trace_reflections` is a proof point. Natural siblings:

- `trace_segment_visibility(start, end, ignore_prims)` — replaces the
  Python re-fire loop in §2.1 with an `anyhit` that calls
  `optixIgnoreIntersection()` for ignored primitives. Single-launch
  guaranteed.
- `trace_diffraction_chain(rays, edge_ids, max_order)` — see §2.2.
- `trace_shadow_boundary_samples(...)` — once the discontinuity plan
  settles on an analytic form.

Each one stays geometry + topology only (no BSDF, no emitter, no
integrator). RayD remains a *kernel library*, not a renderer.

### 3.2 Thin / flexible payload variants

Right now `Scene.intersect()` always returns the full `Intersection` POD.
A `intersect_thin(ray)` that returns only `(t, global_prim_id)` would
remove ~50% of the GMEM writes for shadow-style queries (which are the
overwhelming majority of `ray_test` use sites). Channel can fold this into
existing `intersect_rays_raw_with_prim`.

### 3.3 Edge BVH refresh path

The 18-probe-per-state approximation in `bvh_pairs` is the main
recall-vs-cost knob. RayD's own
`docs/edge_bvh_lbvh_treelet_improvement_plan.md` already tracks this.
Channel does not depend on the exact algorithm, only on (a) recall
guarantees on dense scenes and (b) `set_edge_mask` semantics, so this is
an internal RayD concern.

## 4. Real-time Digital Twin: RayD vs. Mitsuba

Calibrate "real-time" first:

| Tier | Latency budget | Achievable today |
|---|---|---|
| Static inverse-rendering training | seconds / iter | Yes |
| Interactive TX placement | 100–500 ms | Marginal — bottleneck is Python launches |
| Live digital twin (moving users / TX) | 30–100 ms | Not with current architecture |
| Image-style real-time RT (1080p @ 60 Hz path tracing) | 16 ms | Not a useful target for RF |

The relevant question is the middle two tiers.

### Where Mitsuba would equal or beat RayD

- Triangle BVH throughput: identical (same OptiX backend).
- Forward Dr.Jit AD throughput: roughly identical.
- Dynamic mesh update: Mitsuba supports `params.update()` for vertex
  edits; RayD's `update_mesh_vertices` is cleaner but not faster.

### Where RayD is ahead — structural vs. patch-like

Three commonly-cited advantages of RayD over Mitsuba, graded by whether
Mitsuba could close the gap with a determined engineer or whether the gap
is forced by architecture.

#### Edge BVH + `nearest_edge` + `set_edge_mask` — **patch-like**

Mitsuba has no edge BVH, but nothing stops you from writing a Dr.Jit-native
LBVH on the side that consumes `mi.Mesh` vertices. Costs:

- Two acceleration structures, two refresh cycles on geometry change.
- Scene-global edge index table maintained by hand.

These are work, not perf ceilings. An attentive port could reach ~95% of
RayD's edge BVH throughput. **This is not a structural advantage.**

#### Dependency footprint — **patch-like for perf, structural for deployment**

For a digital-twin runtime that ships:

- Mitsuba pulls in BSDFs, integrators, emitters, samplers, films you do not
  use. They do not run on the hot path, but they affect container size,
  startup time, and Dr.Jit version pinning (Mitsuba 3 hard-pins Dr.Jit).
- These costs are real but do not constitute a "cannot reach real-time"
  ceiling.

#### Fused `trace_*` kernels sharing the same GAS — **structural**

This is the one structural delta, and it is the reason real-time digital
twin favors RayD.

`reflection_trace.cu` is an independent OptiX raygen + closesthit pipeline
that calls `optixTrace` directly against RayD's `OptixTraversableHandle`.
RayD's design premise is *the user library owns the GAS and can build
custom OptiX programs around it*.

Mitsuba 3's design premise is the opposite: `mi.Scene` treats the OptiX
GAS as a private implementation detail and exposes only high-level
queries (`scene.ray_intersect()`). **The GAS handle has no C-callable
interface for external kernels.**

For real-time the consequences are concrete:

| What you want to do | RayD path | Mitsuba path |
|---|---|---|
| One launch for full N-bounce reflection chain | `trace_reflections()` → `optixTrace` against the shared GAS | Use Mitsuba's integrator, or fork Mitsuba |
| Closesthit that accumulates UTD field into receiver bins | Write a custom CH against the shared GAS | Same |
| Custom payload semantics (non-BSDF physics) | Add a new `trace_*` kernel in RayD | Must fork |

The two workarounds without forking are both bad:

- **Parallel GAS on the side**: replicate RayD next to Mitsuba and pay
  doubled IAS/GAS refit on every geometry change. On Munich-scale scenes
  refit is in the milliseconds; doubling it eats ~10% of a 30 ms frame
  budget purely as tax for staying on Mitsuba.
- **Fork Mitsuba and expose `m_accel`**: not just the OptiX context — you
  inherit the pipeline manager, SBT layout, and program-group machinery.
  Maintenance cost is unacceptable for a long-running project.

The advantage is a direct consequence of an architectural choice, not of
how much code each project has written. Mitsuba's encapsulation is the
right call inside the "render-an-image" context (it lets the renderer
keep BSDF / integrator / sampler self-consistent). It is the wrong call
inside the "digital-twin runtime" context.

#### Why this only matters for real-time

- **Offline (inverse-rendering training, batch radiomap)**: extra Python ↔
  Dr.Jit round-trips per step are tolerable. Mitsuba's high-level query
  model is fine; GAS encapsulation is not in the critical path.
- **Real-time (30–100 ms frame budget)**: one frame ≈ one OptiX launch
  plus thin Python orchestration. "Custom OptiX program directly against
  the GAS" moves from nice-to-have to mandatory. Mitsuba's encapsulation
  becomes the actual ceiling.

### Verdict for real-time

For tier-2 (interactive) Mitsuba and RayD are roughly equivalent —
neither wins on the bottleneck (Python orchestration overhead).

For tier-3 (live digital twin) RayD has **one** real structural
advantage: the GAS is exposed and OptiX programs can be written against
it without forking. The path forward is "add a `trace_diffraction_chain`
(and a few siblings) inside RayD," which fits RayD's design and does not
fit Mitsuba's.

The other commonly-cited advantages (edge BVH, dependency footprint) are
real conveniences but a determined engineer could close them on Mitsuba.
They alone would not justify maintaining a separate ray tracer.

If real-time is the target, the answer is **not** "switch back to
Mitsuba." It is "extend RayD with one or two more fused trace kernels
and let channel orchestrate at a higher level."

## 5. OptiX Program vs. CUDA Kernel: Performance

The honest answer: **for ray-mesh intersection work, OptiX programs are
faster; for everything else, they are not — they are usually slower.**

### Why OptiX wins for traversal

- RT cores: on Turing+ GPUs, BVH traversal and triangle intersection
  execute on dedicated hardware. A hand-written CUDA traversal kernel
  cannot reach RT-core throughput. This is the single biggest reason
  to keep BVH queries inside `optixTrace`.
- Coherent ray scheduling: OptiX's SBT + traversal scheduler handles
  divergence (hit/miss split, primary/anyhit/closesthit fanout) better
  than a manually-written kernel that has to manage warp divergence.

### Why a plain CUDA kernel wins for non-traversal work

- Register pressure: OptiX closesthit/anyhit operate under tighter
  register limits than a free CUDA kernel. Heavy in-register state
  (UTD coefficient computation, multi-component complex math) can
  spill, and OptiX spills are expensive.
- Shared memory: OptiX programs cannot use shared memory in the
  cooperative-tiling sense; a plain CUDA kernel can.
- Block/grid topology: OptiX is launched as a 1D/2D/3D ray index space.
  Reductions across rays (e.g., atomic accumulation into a receiver
  grid) work but you do not control occupancy as finely.
- Tooling: profiling a closesthit is harder than profiling a
  free CUDA kernel (Nsight Compute support is improving but still
  thinner).

### The right split for channel

- **Inside OptiX (closesthit/anyhit/raygen):** ray casting, hit
  reconstruction, payload-carried per-ray state, single-step
  field-component math that fits in registers.
- **As a separate CUDA kernel after OptiX:** anything that needs
  block-level reductions, large LUT lookups, dense complex-matrix ops,
  or many global atomics into shared bins.

The current witwin kernels (`kernels/utd/utd_accumulate.cu`,
`utd_math.h`) are written as plain CUDA. That choice is correct for
their workload — they aggregate UTD contributions across many candidate
edges per state, which is reduction-heavy, not traversal-heavy. **They
should stay CUDA.** What should migrate into OptiX is the *ray casting
that currently happens between* these kernels, not the kernels
themselves.

Stated as a rule:

> OptiX owns "where did this ray go." Plain CUDA owns "what do I
> compute about it." Move the boundary so that one ray's worth of
> per-step physics fits in OptiX payload registers; everything heavier
> stays in CUDA, called *after* the OptiX launch finishes.

## 6. Staging Plan

Loose ordering, not a commitment:

1. **§2.1 quick fixes** — RayD exposes thin-payload `intersect`, channel
   drops the re-gather path. Low risk, measurable. Days, not weeks.
2. **§3.1 `trace_segment_visibility`** with anyhit-based ignore. Replaces
   the Python re-fire loop. Concrete and testable.
3. **§2.4 persistent scene** — eliminate full rebuilds on endpoint
   moves. Pure witwin work, no RayD change.
4. **§2.2 + §3.1 `trace_diffraction_chain`** — the big one. Decide once
   §1–§3 have left the diffraction BFS as the visible bottleneck.

At every step, re-evaluate against Mitsuba parity. If the answer keeps
being "Mitsuba could not do this without growing a parallel edge BVH
and a custom integrator," the current direction is justified. The day
that answer flips, it is time to reassess.

## 7. Fusion Candidate Inventory in witwin

Survey of every ray-query call site in `witwin/channel/montecarlo` and
`witwin/channel/deterministic`, graded by whether it benefits from a fused OptiX
kernel that accesses the GAS directly (`trace_reflections`-style), or
whether it is already at the right granularity.

### Deterministic candidates

#### D1 — `segment_visible` with `ignore_prim` re-fire loop

- Location: [scene.py:761](../../witwin/channel/core/scene/scene.py#L761)
- Current shape: 2–8 sequential `ray_test` / `intersect` calls in Python,
  advancing the origin past each ignored primitive.
- Fusion: `trace_segment_visibility(start, end, ignore_prim_ids)` —
  single OptiX launch, **anyhit** calls `optixIgnoreIntersection()` for
  matches; closesthit returns blocked/clear in one payload bit.
- Expected speedup: **3–8× on calls that need ignores.**
- Difficulty: low. Anyhit ignore is textbook OptiX.
- Strategic value: highest. Every other deterministic candidate calls
  into this. Land D1 first and most of D4/D5/M3/M4 inherit the win.

#### D2 — Higher-order diffraction BFS chain

- Location: [builders.py:402](../../witwin/channel/deterministic/path/diffraction_impl/builders.py#L402)
  (`bvh_pairs` + `higher`)
- Current shape: per-order Python iteration with 18-probe `nearest_edge`
  per state, per-pair `segment_visible`, UTD math, then state-array
  materialization between orders.
- Fusion: `trace_diffraction_chain(rays, edges, max_order)` — one OptiX
  launch walks one TX → edge₁ → edge₂ → … → RX path per ray; payload
  carries `Complex2f` field components and edge lineage.
- Expected speedup: **3–10×** on multi-order diffraction.
- Difficulty: high. UTD math has to fit in OptiX payload registers; the
  discontinuity-smoothing scheme (plan 22) must be representable as
  straight-line GPU code.
- Strategic value: the centrepiece. If real-time digital twin happens,
  D2 is the kernel that makes it possible.

#### D3 — Suffix reflection chain in forward.py

- Location: [forward.py:570-619](../../witwin/channel/deterministic/path/diffraction_impl/forward.py#L570)
- Current shape: B-bounce Python loop, two `intersect_rays_with_prim`
  per bounce (one for the bounce hit, one for the next-blocker probe),
  field/material updates, grid accumulation per segment.
- Fusion: `trace_diffraction_suffix_chain(...)` — a near-copy of
  RayD's `reflection_trace.cu` template with field-accumulation payload
  and per-segment atomic accumulation into the receiver grid.
- Expected speedup: **2–5×.**
- Difficulty: medium. Closest existing analogue, so the second-easiest
  `trace_*` kernel to write.
- Strategic value: high. Pairs naturally with D2 and reuses the same
  payload pattern.

#### D4 — EPC reflection-path validation chain

- Location: [epc.py:656–705](../../witwin/channel/deterministic/path/reflection_impl/epc.py#L656)
- Current shape: 4 sequential `segment_visible` calls per path
  candidate.
- Fusion: D1 covers each individual call. A further fusion that
  validates the whole TX→r₁→r₂→…→RX visibility in one launch is
  possible but complex.
- Expected speedup: 4× from D1 alone; another ~1.5× if the chain is
  fused.
- Strategic value: secondary; D1 already delivers most of the win.

#### D5 — Direct source-edge visibility

- Location: [builders.py:244](../../witwin/channel/deterministic/path/diffraction_impl/builders.py#L244),
  [postprocessing.py:269,317](../../witwin/channel/deterministic/path/diffraction_impl/postprocessing.py#L269)
- Current shape: `segment_visible` with `ignore_prim`.
- Fusion: fully subsumed by D1. No separate kernel needed.

#### D6 — LOS shadow test

- Location: [los.py:25](../../witwin/channel/deterministic/path/los.py#L25)
- Current shape: one `ray_test`.
- Fusion: already a single OptiX launch; nothing to gain unless §3.2
  (thin `Intersection` variant) lands and removes the unused payload
  writes.

### Monte Carlo candidates

#### M1 — MC reflection trace loop

- Location: [reflection.py:535](../../witwin/channel/montecarlo/path/reflection.py#L535)
- Current shape: `dr.syntax` symbolic loop over B bounces. Dr.Jit
  records the Python loop into a single fused PTX, but each
  `scene.ray_intersect` is still a separate OptiX launch with a full
  `Intersection` write to GMEM. Wedge events also write through
  `diff_state_store`.
- Fusion: extend RayD's `trace_reflections` to a
  `trace_reflections_accumulating` variant — closesthit accumulates
  field into receiver bins via `atomicAdd`, carries polarization vector
  in payload (~6 u32 slots, fits), and writes wedge events to a
  pre-allocated per-ray buffer.
- Expected speedup: **2–4×.** Wins come from (a) skipping the full
  `Intersection` GMEM write per bounce, (b) merging per-step Dr.Jit
  math into the OptiX kernel, (c) in-kernel atomic accumulation
  replacing the post-loop reduction.
- Difficulty: medium-high. Wedge collection is the awkward part —
  needs a warp-coherent per-ray buffer index.
- Strategic value: high. This is the MC analogue of D3 and shares
  payload/kernel patterns.

#### M2 — MC LOS visibility

- Location: [los.py:91](../../witwin/channel/montecarlo/path/los.py#L91)
- Current shape: one `segment_visible(tx_pos, grid.cell_centers)`.
- Fusion: already a single launch when there are no ignores. With
  ignores, D1 subsumes it.

#### M3 — Diffraction source/target with shadow-boundary offset

- Location: [diffraction.py:309–318](../../witwin/channel/montecarlo/path/diffraction.py#L309)
- Current shape: 4 `segment_visible` calls per diffraction state to
  evaluate the smoothed shadow boundary (a→b and a→b_offset for both
  source and target sides).
- Fusion: `trace_segment_pair_visible(a, b, b_offset)` — single launch
  with two traces per ray, payload carries 2 visibility bits. Partially
  done already as `VisibilityKernel.segment_pair_visible`
  ([bdpt_diffraction.py:672](../../witwin/channel/montecarlo/integrators/bdpt_diffraction.py#L672))
  but that path is a CUDA kernel reading raw triangle data, not an
  OptiX-backed query — and only activates when `VisibilityKernel.available()`.
- Expected speedup: **2×** (folds 2 launches into 1, per side).
- Difficulty: low.
- Strategic value: medium. Improves the existing fallback path.

#### M4 — BDPT diffraction edge→reflection→target chain

- Location: [bdpt_diffraction.py:1683-1694](../../witwin/channel/montecarlo/integrators/bdpt_diffraction.py#L1683)
- Current shape: 2 sequential `segment_visible` with `ignore_prim` and
  `max_ignored_hits=2`.
- Fusion: D1 covers both calls; no separate kernel needed.

#### M5 — Source-edge axial visibility sampling

- Location: [postprocessing.py:140–163](../../witwin/channel/montecarlo/path/postprocessing.py#L140)
- Current shape: Python loop over `SOURCE_VISIBILITY_SAMPLE_FRACTIONS`,
  one `segment_visible(source_pos, sample_point)` per fraction (~3–5
  samples per edge state).
- Fusion: `trace_axial_edge_visibility(source_pos, edge_pos, edge_dir,
  line_min, line_max, n_samples)` — single launch fires N rays along
  the edge axis per state, OR-reduces the visibility bits in payload.
- Expected speedup: **3–5×** (fraction count).
- Difficulty: medium. Requires a 1-to-N ray expansion inside one launch.
- Strategic value: medium.

### Priority summary

By (impact × tractability):

1. **D1** — `trace_segment_visibility` with anyhit ignore. Cheap to write,
   unlocks D4 / D5 / M3 / M4 with zero additional engineering.
2. **D3 / M1** — `trace_*_accumulating` reflection chains. Direct
   extensions of `trace_reflections.cu` template.
3. **D2** — `trace_diffraction_chain`. The big one. Tackle only after
   D1+D3 prove the pattern and after the plan-22 discontinuity story
   resolves.
4. **M3 / M5** — small fusions that fold per-state sampling loops into
   a single launch.

D4, D5, D6, M2, M4 are not separate work items — they are byproducts of
D1 and §3.2.

### Mapping the inventory back to §4

A check on whether this inventory actually demonstrates the structural
advantage claimed in §4, or whether it is just a generic fusion to-do
list that any tracer would face.

| Candidate | Technical requirement | Mitsuba can match without forking? |
|---|---|---|
| D1 | Custom `__anyhit__` calling `optixIgnoreIntersection` against the same GAS | No — GAS is private |
| D2 | Custom `__raygen__` + `__closesthit__` walking a multi-edge chain with field-accumulating payload | No |
| D3 | Custom `__closesthit__` with per-segment atomic accumulation | No |
| D4 / D5 | Themselves simple, but the win comes from D1 | No (inherits D1) |
| D6 / M2 | Already single-launch | Yes |
| M1 | Custom CH with `atomicAdd` into receiver bins + polarization payload | No |
| M3 | Custom raygen firing two rays per launch index | No |
| M4 | Subsumed by D1 | No (inherits D1) |
| M5 | Custom raygen firing N rays per state | No |

Every candidate that produces meaningful speedup (D1, D2, D3, M1, M3, M5)
requires writing custom OptiX programs that call `optixTrace` against a
GAS the user library owns. This is the property §4 calls the one
structural advantage. The inventory is not a generic fusion roadmap; it
is the concrete materialization of that structural property.

### Realisation note

The structural advantage opens the path; it does not walk it. Every
item above still needs RayD-side kernel work, and channel cannot add
these kernels itself — RayD deliberately does not expose a
user-extensible program-group API (§3).

Because RayD and channel are co-authored, "will these kernels be
written" is internal planning rather than a dependency on a separate
project. The §4 structural property therefore reduces in practice to:

- On the Mitsuba path, this work would require **forking before
  starting** — most of the engineering cost is upfront, before any
  channel-specific feature can land.
- On the current path, the kernel pattern is already proven by
  `reflection_trace.cu`; the candidates in §7 are **incremental
  extensions of an established template**.

The candidate inventory above is the concrete to-do list that the
structural argument in §4 buys us the right to attempt.

## 8. Open Questions

- How much of the diffraction BFS state actually fits in OptiX payload
  registers? Need a register-pressure budget before committing to §2.2
  / §3.1 fusion.
- Does the discontinuity-smoothing scheme in plan 22 survive the move
  into a closesthit body? If it requires Dr.Jit-traced control flow,
  fusion may be blocked.
- For tier-3 live updates, what is the dominant cost: BVH refit, mesh
  upload, or per-frame BFS expansion? A single profiling pass should
  answer this and reorder the staging plan.
