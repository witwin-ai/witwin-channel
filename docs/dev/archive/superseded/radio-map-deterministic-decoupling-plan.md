# Deterministic Radio-Map Decoupling Plan

Status: Draft
Category: Plan
Last reviewed: 2026-04-10

## Objective

Refactor the deterministic `RadioMapMonitor` path so it becomes genuinely
radio-map-owned instead of reusing `FieldMonitor`-style field evaluation and
field accumulation structure.

The redesign must optimize for:

1. fewer kernel launches in both forward and backward,
2. lower peak memory during dense radio-map workloads,
3. preserved numerical results,
4. preserved gradients with respect to TX position, geometry, and supported
   material parameters.

The design position is explicit:

1. a radio map is not a field monitor with a different reducer,
2. the final observable is a radio-map scalar quantity per cell:
   - incoherent mode: accumulated scalar power,
   - coherent mode: accumulated complex scalar amplitude and its derived power,
3. vector electric fields may exist only as short-lived internal intermediates
   when the receiver model requires them, but they must not be the primary
   runtime payload or the dominant shared execution boundary.

## Why This Refactor Is Needed

The current deterministic radio-map path still inherits too much of the field
monitor execution shape.

### Current structure problems

1. `monitors/radio_map/deterministic/coherent.py` still routes through generic
   field-style reflection and diffraction entrypoints.
2. `monitors/radio_map/deterministic/cell_accumulation.py` still consumes
   field-oriented gathered state and field-vector evaluation helpers before
   reducing them into radio-map observables.
3. several radiomap-specific kernels still live under
   `kernels/monitors/field/`, which keeps the ownership boundary
   field-centric instead of radiomap-centric.
4. public UTD forward/backward still rely on the Dr.Jit reference replay path
   on important deterministic routes, which preserves `gather`,
   `compress`, `scatter`, and `scatter_reduce` heavy graphs.
5. higher-order diffraction candidate generation still behaves more like a
   global field-path expansion than a local radio-map workload.

### Why this hurts performance

On the maintained `three cubes` dense benchmark
(`256 x 256`, coherent radiomap, diffraction enabled), the current deterministic
mainline still shows a large replay footprint:

- `KernelType.JIT = 2943`
- `KernelType.Reduce = 2256`
- `KernelType.Other = 1091`

The forced tiled diffraction experiment reduced peak pair footprint but blew up
launch counts:

- `KernelType.JIT = 37220`
- `KernelType.Reduce = 70340`
- `KernelType.Other = 4621`

This demonstrates that the dominant problem is not just tile selection. The
deeper issue is that deterministic radiomap still uses a field-style execution
graph with too many Python and Dr.Jit boundaries around pair work.

## Current Investigation Update

The active implementation status changed materially during the 2026-04-10
native UTD investigation.

### Confirmed fixes

1. Receiver-tiled native UTD is no longer treated as part of the supported main
   execution path. The tiled replay experiment remains archived only as
   historical reference because it increased launch counts and complicated
   correctness analysis without solving the mainline bottleneck.
2. The full-cartesian native UTD primal zero-output regression is now isolated
   and understood. The bug was not in diffraction math. It was an output
   ownership bug: native `..._into(...)` launchers mutated Python-owned Dr.Jit
   arrays through raw data pointers, and those pointer writes did not reliably
   become new Dr.Jit-visible values.
3. The current safe rule is explicit: new native UTD forward wrappers must use
   return-valued launchers or another Dr.Jit-visible output construction
   contract. Raw mutation of preallocated Dr.Jit arrays is not a valid design
   assumption for this path.
4. With that fix, the low-level full-cartesian native UTD forward and geometry
   JVP parity tests now pass again on the maintained `_build_utd_case()`
   coverage.

### Still unresolved

1. The zero-output bug fix did not by itself achieve the plan objective of
   lower launch counts on the radiomap contract that matters most here:
   `matched_isotropic`.
2. The visible repeatability regression on coherent `matched_isotropic`
   radiomap is now isolated to the diffraction-only native vector replay path,
   not the builder or state layout:
   - the issue reproduces with `shadow_boundary_mode="none"` and with
     `matched_isb_completion`,
   - `projected_polarized` coherent runs remain repeatable,
   - disabling only diffraction `native_vector_replay` restores
     `256 x 256` rerun differences to about `1e-11`.
3. The concrete source of that regression is the native
   `radiomap_accumulate_vector_power_pairs` kernel. It reduces many diffraction
   pairs into the same receiver bins using unordered `atomicAdd` accumulation,
   so dense matched-isotropic diffraction is numerically non-repeatable even
   when the underlying pair field evaluation is correct.
4. The current mainline mitigation is explicit: deterministic radiomap now
   keeps reflection on the native matched-isotropic vector accumulator, but
   forces diffraction back to `direct_state_vector_power` until a deterministic
   native reduction design exists.
5. Because of that mitigation, the main remaining performance problem is no
   longer "why is matched-isotropic unstable?" It is now "how do we recover the
   intended launch reduction without reintroducing non-repeatable atomic pair
   aggregation?"
6. Dense `256 x 256` matched-isotropic launch measurement still needs a
   lower-overhead instrumentation path. Naive `dr.JitFlag.KernelHistory`
   capture on the baseline coherent route can exhaust GPU memory before the
   comparison completes.
7. The earlier full-cartesian native UTD zero-output fix is orthogonal to this
   repeatability regression and must not be reverted as part of this follow-up.

### Immediate interpretation

The main blocker has moved.

It is no longer "native full-cartesian UTD writes zeros." It is now:

1. deterministic diffraction pair aggregation for matched-isotropic native
   replay,
2. remaining launch fragmentation around the radiomap accumulation contract
   after the temporary fallback to `direct_state_vector_power`,
3. lack of a dense-scene launch counter that does not distort or OOM the run.

### Root cause summary for the current matched-isotropic regression

The unstable striped forward maps were not caused by:

1. diffraction builder timing barriers,
2. path-export lineage metadata split,
3. reduced path-export support gathers,
4. suffix hot-state gather selection,
5. the previously fixed native UTD full-cartesian zero-output bug.

The regression is specific to the diffraction matched-isotropic native vector
pair accumulator:

1. deterministic radiomap coherent matched-isotropic requested
   `native_vector_replay=True` for diffraction,
2. that path used the native
   `radiomap_accumulate_vector_power_pairs` CUDA reduction,
3. the kernel used unordered `atomicAdd` accumulation into receiver-sized
   coherent, power, and vector buffers,
4. dense diffraction workloads therefore changed run-to-run with thread
   scheduling order even though support masks and pair generation were stable.

Current policy for the maintained branch:

1. keep the UTD zero-output ownership fix,
2. keep diffraction matched-isotropic on `direct_state_vector_power`,
3. treat a deterministic native reduction design as a separate follow-up item
   before re-enabling diffraction native vector replay.

## Benchmark Setup And Maintained Commands

The maintained dense-scene setup for this work is the `three cubes` scene and
its committed benchmark helpers. Do not replace these commands with ad hoc
notebooks or gradient scripts during kernel-count iteration.

### Maintained entrypoints

1. Forward kernel-count acceptance:
   - `tests/support/bin/profile_radiomap_forward_three_cubes.py`
2. Forward visual parity / component inspection:
   - `tests/main/plot_radiomap_forward_three_cubes.py`
3. Forward/reverse AD kernel-history inspection:
   - `tests/support/bin/profile_radiomap_backward_three_cubes.py`
4. Gradient regression figure after forward gates pass:
   - `tests/main/plot_radiomap_gradients_three_cubes.py`
5. Witwin-vs-Sionna scene comparison reference:
   - `tests/main/plot_radiomap_sionna_three_cubes.py`

### Fixed acceptance workload

Use the same environment and scenario for every no-diff launch-count check:

```bash
conda activate witwin2
cd channel
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.support.bin.profile_radiomap_forward_three_cubes --grid-size 256 --n-rays 384 --max-diffractions 0 --witwin-profile matched_isb_completion --output-json tests/output/radiomap_forward_profile_256_384_nodiff.json
```

Acceptance contract for Phase 1 no-diff work:

1. `KernelHistory total_count < 2000`,
2. `runtime_backends.no_diff_fast_path == true`,
3. `runtime_backends.no_diff_reflection_scheduler == "cartesian_chunked"`,
4. the no-history forward runtime does not regress against the maintained
   baseline,
5. forward plots still retain visible LoS and reflection structure instead of
   collapsing to completion-only output.

The maintained forward profile now reports separate
`scene_build_seconds`, `monitor_tracer_setup_seconds`, `forward_seconds`, and
`end_to_end_seconds` fields so cold-start scene compilation can be analyzed
separately from the steady-state no-diff trace.

Current checkpoint after the latest cold-start/no-diff pass:
- initial scene rebuild no longer pays the redundant RayD `update_mesh_vertices(...)`
  replay after `rayd.Scene.build()`
- reflection-prefix discovery now consumes the flat RayD reflection-chain payload
  directly instead of first splitting it into per-slot Dr.Jit tuples, which
  drops the maintained three-cube reflection-detail discovery microprofile from
  `17` kernels to `11`
- no-diff reflection EPC now ignores coplanar surface groups directly during
  visibility replay, so the earlier reflection ignore-loop plumbing is no longer
  the maintained repeated small-kernel hotspot
- the maintained `size=36 op=8 x9` repeated-kernel cluster is now understood and
  fixed: it came from lazy first-use materialization of `scene.tri_data_gpu["v0"]`
  / `["v1"]` / `["v2"]` inside the no-diff matched-ISB completion path, and the
  triangle coordinate caches are now eagerly evaluated during scene preload so
  the maintained no-diff end-to-end profile drops from `130` to `122` kernels
  with no remaining repeated small-kernel cluster
- the maintained backward inspection entrypoint is now
  `tests/support/bin/profile_radiomap_backward_three_cubes.py`, which records
  forward and backward kernel histories separately and keeps `Other` entries in
  the report so custom CUDA launches are counted instead of silently discarded

### Companion visual commands

Use these commands after kernel-count edits:

```bash
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.main.plot_radiomap_forward_three_cubes --grid-size 256 --n-rays 384 --witwin-profile matched_isb_completion --output-prefix tests/output/radiomap_forward_three_cubes_matched_isb_completion
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.main.plot_radiomap_gradients_three_cubes --grid-size 256 --n-rays 384
```

The forward plot is part of every no-diff optimization checkpoint. The
gradient figure is a regression gate after the forward kernel target is met,
not the inner-loop benchmarking command.

## Scope

This plan applies to the deterministic radio-map implementation:

- `witwin/channel/monitors/radio_map/deterministic/`
- `witwin/channel/kernels/monitors/field/radio_map_accumulate/`
- `witwin/channel/kernels/trace/utd/`
- deterministic radiomap-facing parts of
  `witwin/channel/trace/diffraction/`
- deterministic radiomap-facing reflection replay / EPC integration

This plan explicitly does not redesign:

1. the public `Scene + Tracer + Result` architecture,
2. the baseline parity implementation,
3. the Monte Carlo radio-map mode,
4. diffraction physics formulas,
5. visibility semantics,
6. wedge-angle conventions,
7. CPU fallback behavior.

## Non-Negotiable Invariants

The refactor must preserve the following.

### Numerical invariants

1. existing coherent and incoherent radio-map outputs remain within current
   parity tolerances,
2. path-family inclusion rules remain unchanged,
3. no heuristic smoothing, soft support masks, or ad hoc gradient fixes are
   introduced,
4. wedge support, visibility, reflected suffix, and ownership semantics remain
   physically derived.

### AD invariants

1. gradients with respect to TX position remain unchanged,
2. geometry gradients remain unchanged,
3. supported material gradients remain unchanged,
4. deterministic radiomap must not silently fall back to new Torch, NumPy, or
   CPU transport in hot paths.

### Architecture invariants

1. `RadioMapMonitor` stays a first-class monitor kind distinct from
   `FieldMonitor`,
2. deterministic and Monte Carlo remain separate execution families,
3. shared code is limited to low-level geometry, visibility, material lookup,
   and core diffraction / reflection physics only.

## Design Principles

### 1. Radio-map-owned execution boundary

Deterministic radiomap should own its kernel boundary and scheduler policy.

The intended ownership split is:

- shared:
  - scene compilation,
  - triangle and edge metadata,
  - visibility primitives,
  - reflection and diffraction physics formulas,
  - low-level custom-op plumbing,
- radiomap-owned:
  - receiver-window planning,
  - diffraction replay contracts,
  - reflection replay contracts for radiomap accumulation,
  - support filtering contracts,
  - final accumulation buffers,
  - deterministic radiomap backward contracts.

### 2. Scalar-observable-first runtime

The runtime should accumulate only the requested scalar observable:

1. incoherent mode accumulates scalar power,
2. coherent mode accumulates complex scalar amplitude and derives power from it,
3. if the receiver model requires vector field components, compute them only
   inside the fused kernel or tightly-coupled native replay loop and reduce them
   immediately before leaving that boundary.

The runtime must stop materializing monitor-facing vector electric field arrays
for dense deterministic radiomap execution.

### 3. Local candidate discovery over global expansion

The deterministic diffraction builder should stop behaving like a global
edge-expansion engine. It should instead discover candidate wedges from local
surface hits, following the structural idea already used by Monte Carlo:

1. use a hit triangle or local support surface,
2. fetch only that surface's edge candidates,
3. test those edges exactly,
4. keep all valid candidates,
5. never use a Monte Carlo-only "best edge" shortcut in deterministic mode.

### 4. Native forward and backward ownership

Deterministic radiomap should stop paying for public Dr.Jit replay graphs on
supported native paths.

The target state is:

1. forward uses radiomap-owned native kernels or a radiomap-owned custom-op
   boundary,
2. backward uses radiomap-owned fused native gradients or sparse-coefficient
   style replay,
3. public radiomap workloads no longer route through the generic
   `utd/drjit_impl.py` path on the supported dense deterministic path.

## Current Hotspots To Eliminate

### A. Field-style diffraction replay in deterministic radiomap

Current deterministic cell-accumulation replay still performs the following structure:

1. gather support state,
2. compress support lanes,
3. gather field state,
4. call generic field evaluation helpers,
5. scatter or scatter-reduce into dense receiver buffers,
6. densify or remap outputs for radiomap consumers.

This is the main source of remaining `JIT` and `Reduce` kernels.

### B. Generic UTD public forward/backward fallback

The current public UTD route used by deterministic radiomap still delegates
important workloads to the Dr.Jit reference implementation. This preserves:

1. repeated state gathers,
2. receiver gathers,
3. ownership gathers,
4. repeated `scatter_reduce` for direct and multi components,
5. replayed forward graphs inside backward.

### C. Higher-order diffraction state builder expansion

The current higher-order path still spends too much work on:

1. wide candidate expansion,
2. repeated state gathers,
3. inserted-reflection metadata gather,
4. point-source field evaluation on states that do not survive later filters,
5. packing hot runtime fields together with cold lineage or audit payload.

### D. Reflection EPC reference gradients

Reflection accumulation is already more native than diffraction, but EPC target
gradients still rely on reference gradient helpers on important paths. That
keeps backward heavier than it should be.

## Target Architecture

## Package layout

The deterministic radiomap path should keep monitor orchestration and kernel
ownership visibly separate from field-monitor code.

### Monitor-side modules

```text
witwin/channel/monitors/radio_map/deterministic/
  trace.py
  coherent.py
  incoherent.py
  scheduler.py
  state_layout.py
  wedge_discovery.py
  reflection_runtime.py
  diffraction_runtime.py
  diagnostics.py
  metadata.py
```

### Kernel-side modules

```text
witwin/channel/kernels/monitors/radio_map/deterministic/
  reflection_cells/
    native_impl.py
  diffraction_cells/
    native_impl.py
  wedge_discovery/
    native_impl.py
  shadow_completion/
    native_impl.py
```

The explicit intent is to stop placing deterministic radiomap kernels under
`kernels/monitors/field/`.

## Runtime boundary contract

### Incoherent mode

Inputs:

1. compact radiomap diffraction or reflection state arrays,
2. receiver window or tile descriptor,
3. monitor cell mapping metadata,
4. receiver-model metadata,
5. optional diagnostics flags.

Outputs:

1. scalar power buffer,
2. optional component power buffers,
3. optional diagnostics buffers,
4. optional compact AD tape or sparse coefficient buffers.

### Coherent mode

Inputs:

1. compact radiomap diffraction or reflection state arrays,
2. receiver window or tile descriptor,
3. monitor cell mapping metadata,
4. receiver-model metadata,
5. optional diagnostics flags.

Outputs:

1. complex scalar coherent buffer,
2. derived power buffer if requested,
3. optional component coherent buffers,
4. optional compact AD tape or sparse coefficient buffers.

No monitor-facing vector electric field buffer should cross this boundary on the
production deterministic path.

## Hot and cold state split

Deterministic radiomap must split state payloads into:

### Hot runtime state

Fields needed for support, visibility, and final accumulation only:

1. edge index and local edge geometry,
2. wedge normals and wedge parameter,
3. finite-edge bounds,
4. source or image-source position,
5. face material parameters required by the exact physics,
6. ownership code or family tag,
7. minimal reflection-prefix or suffix descriptors needed by replay.

### Cold metadata

Fields needed only for audits, debugging, lineage, or path export:

1. lineage identifiers,
2. history or replay provenance,
3. path export payloads not required by radiomap replay,
4. timing-only or audit-only metadata.

Cold fields must never be loaded into the main deterministic radiomap replay
kernels.

## Wedge Discovery Strategy

Monte Carlo already contains the correct structural idea:

1. discover a support surface hit,
2. fetch that surface's local edge candidates,
3. test silhouette and wedge-side conditions locally.

Deterministic radiomap should adopt that structure without inheriting Monte
Carlo's stochastic "choose one best edge" behavior.

## Order-1 direct diffraction

Target approach:

1. reuse direct hit or reflection-style support discovery from TX to scene
   surfaces,
2. for each discovered support triangle, fetch its local surface-edge candidate
   slots,
3. evaluate exact deterministic validity on all candidate edges:
   - source exterior region,
   - silhouette or boundary-edge condition,
   - finite-edge support,
   - exact visibility checks,
4. keep every valid candidate edge.

The deterministic path must not reduce this to the nearest or best edge.

## Reflection-prefix diffraction

For paths with one or more prefix reflections:

1. reuse the last reflected hit triangle from the reflection-family state,
2. enumerate only that triangle's local edge candidates,
3. test all candidates exactly,
4. materialize only the surviving local wedges into the next diffraction state
   pool.

This reuses reflection discovery as a structural probe instead of building
global diffraction candidate products.

## Higher-order diffraction

For higher-order diffraction, use the same local principle recursively.

### Preferred discovery rule

1. each diffraction or inserted-reflection prefix defines a virtual source or
   reflected-source state,
2. perform a local support-surface discovery from that prefix state,
3. use the discovered support triangle's edge candidates as the next wedge
   candidate set,
4. run exact deterministic filtering on all candidates,
5. keep every valid wedge and only then build the next-order state arrays.

### Practical consequence

Multi-diffraction wedge discovery becomes reflection-driven or support-surface
driven instead of edge-global. This is the deterministic analogue of the Monte
Carlo `first hit -> local edge candidates` idea.

### Rollout safety rule

During migration, keep a debug-only parity mode that compares:

1. local wedge discovery candidate results,
2. legacy global expansion candidate results.

The local path becomes the default only after parity is established on the
maintained regression scenes.

## Execution Model

## Reflection

Reflection family discovery may remain shared, but radiomap accumulation must be
radiomap-owned.

Required changes:

1. stop routing deterministic radiomap coherent reflection through generic
   field-oriented payload assembly,
2. keep radiomap reflection kernels under radiomap-owned kernel folders,
3. return only radiomap scalar observables and optional compact AD state.

## Diffraction

Diffraction replay must become the primary focus of this refactor.

Required changes:

1. stop routing deterministic radiomap through generic field evaluation helpers
   as the long-term path,
2. replace the current
   `support gather -> field gather -> field eval -> scatter back` structure with
   a fused radiomap replay contract,
3. compute support filtering, field evaluation, receiver-model reduction, and
   final accumulation inside one radiomap-owned kernel boundary whenever the
   workload is on the supported native path.

## Scheduler

The scheduler must remain radiomap-owned and must stop inheriting field monitor
assumptions.

Required changes:

1. receiver tiling or windowing decisions must be based on radiomap cost
   metrics, not field cost proxies,
2. avoid Python tile loops that multiply kernel launches,
3. prefer receiver-windowed cartesian replay or fully fused tiled native replay
   over Python-managed micro-tiles,
4. record planner diagnostics:
   - peak pair count,
   - estimated launches,
   - peak state count,
   - receiver-window density,
   - support survivor ratio.

## Forward And Backward Redesign

## Forward target

Forward deterministic radiomap should look like:

1. build compact hot state pool,
2. discover local wedge candidates from support surfaces,
3. compact surviving states,
4. launch radiomap-owned reflection or diffraction replay kernel per chunk or
   receiver window,
5. accumulate directly into final cell buffers,
6. export only scalar observables and compact metadata.

The forward path should not export per-pair vectors, dense receiver-vector
payloads, or field-monitor-style totals.

## Backward target

Backward deterministic radiomap should look like:

1. reuse a compact tape or sparse coefficient representation produced by the
   radiomap-owned forward path,
2. launch fused native VJP or JVP kernels for the supported radiomap path,
3. avoid rebuilding the full Dr.Jit pair graph from the public generic UTD
   route.

Two acceptable endpoint designs are:

1. a radiomap-owned custom-op with native `eval()`, `forward()`, and
   `backward()`,
2. a Monte Carlo-style sparse coefficient extraction plus native JVP/VJP
   launches.

The unacceptable endpoint is preserving the current pattern of replaying the
generic UTD forward Dr.Jit graph inside backward.

## Concrete Module-Level Changes

### 1. Remove field-monitor kernel ownership from radiomap accumulation

Move or replace:

- `kernels/monitors/field/radio_map_accumulate/*`

with:

- `kernels/monitors/radio_map/deterministic/*`

The new kernels should own radiomap semantics explicitly.

### 2. Stop using generic field-state gathers as the deterministic radiomap hot path

Replace the long-term use of:

1. field-evaluation state gather entrypoints,
2. generic field evaluation wrappers,
3. vector electric-field payload exports,
4. radiomap-side densification wrappers around field outputs.

with a dedicated compact radiomap state layout.

### 3. Split deterministic radiomap state builders

Add explicit builders for:

1. direct diffraction candidate states,
2. reflection-prefix diffraction candidate states,
3. higher-order local wedge discovery states,
4. shadow-completion support states.

Each builder should emit only the hot fields required by the next fused replay
stage.

### 4. Replace higher-order global candidate expansion

The higher-order builder should:

1. use support-surface-local edge candidate enumeration,
2. avoid gathering inserted-reflection fields twice,
3. defer point-source and normal-derivative evaluation until after local
   candidate pruning,
4. stop packing lineage or audit payload into replay state arrays.

### 5. Replace generic UTD public replay on the deterministic radiomap path

The supported deterministic radiomap route should stop depending on the generic
public UTD replay contract.

Instead:

1. radiomap-owned forward uses radiomap-owned native pair or cell kernels,
2. radiomap-owned backward uses radiomap-owned VJP or sparse-coefficient
   launches,
3. the generic public UTD path remains only as a parity reference or fallback.

### 6. Move EPC gradient cleanup into the same migration

Reflection EPC target gradients should move off reference gradient helpers on
the supported radiomap path so backward launch counts drop together with the
diffraction migration.

## Phased Implementation Plan

## Phase 0: Freeze Metrics, Baselines, And Counters

Tasks:

1. freeze the deterministic radiomap scalar-observable contract,
2. document current launch counts, reduce counts, and peak memory counters for
   the maintained scenes,
3. freeze regression scenes:
   - wall parity scene,
   - `three cubes`,
   - material-gradient scene,
   - geometry-gradient scene,
4. add stable instrumentation helpers for kernel history totals, reduce kernel
   totals, peak pair count, peak state count, and support survivor ratio,
5. add a dense-scene launch-measurement path for `matched_isotropic` that does
   not rely on memory-heavy full `KernelHistory` capture.

Exit criteria:

1. the migration has hard before-and-after counters,
2. parity scenes are fixed before the architecture changes.

## Phase 1: Ownership Split And New Kernel Namespaces

Tasks:

1. introduce the radiomap-owned kernel package namespace,
2. move radiomap accumulation wrappers out of `kernels/monitors/field/`,
3. add radiomap-local Python runtime adapters with no behavior change yet,
4. keep the old entrypoints only as temporary compatibility shims during the
   migration.

Exit criteria:

1. new work lands under radiomap-owned paths,
2. field monitor no longer owns radiomap kernel naming.

## Phase 2: Compact Hot State Layout

Tasks:

1. split deterministic radiomap replay state into hot and cold payloads,
2. make radiomap replay consume hot state only,
3. move lineage or audit metadata behind opt-in diagnostics paths,
4. delete remaining unconditional field-style state gathers on the hot path.

Exit criteria:

1. replay kernels no longer consume audit payload,
2. support and replay no longer request full field state by default.

## Phase 3: Local Wedge Discovery Migration

Tasks:

1. add deterministic local wedge discovery based on support-surface edge
   candidates,
2. direct diffraction uses direct support-surface discovery,
3. inserted-reflection and higher-order diffraction use reflection-hit or
   support-hit local candidate enumeration,
4. keep all valid candidates instead of a Monte Carlo best-edge reduction,
5. ship a debug parity gate against the legacy global expansion path.

Exit criteria:

1. higher-order builder no longer expands candidates globally on the mainline
   deterministic radiomap path,
2. parity holds on maintained scenes.

## Phase 4: Fused Radiomap Diffraction Forward Kernel

Tasks:

1. introduce a radiomap-owned diffraction replay kernel that performs:
   - support filtering,
   - exact field evaluation,
   - receiver-model reduction,
   - final cell accumulation,
2. eliminate monitor-facing pair-vector and dense receiver-vector payloads,
3. reduce Python-side scatter and densification logic to setup and result
   extraction only,
4. keep coherent and incoherent output modes on the same radiomap-owned replay
   contract.

Implementation constraint:

- do not use raw native mutation of preallocated Dr.Jit arrays as the output
  contract for new UTD replay kernels or wrappers. If a kernel produces new
  radiomap observables, it must return fresh Dr.Jit-visible arrays or use an
  equivalently explicit value-creation contract.

Exit criteria:

1. forward deterministic radiomap diffraction no longer routes through the
   generic field-evaluation hot path,
2. kernel counts on `three cubes` are materially below the current baseline.

## Phase 5: Fused Radiomap Diffraction Backward

Tasks:

1. add native VJP or sparse-coefficient backward for the radiomap-owned
   diffraction forward path,
2. remove deterministic radiomap dependence on generic UTD Dr.Jit replay inside
   backward,
3. ensure geometry, TX, and material gradients match current reference tests.

Exit criteria:

1. supported deterministic radiomap backward no longer replays the generic UTD
   forward graph,
2. gradient parity holds on maintained scenes.

## Phase 6: Reflection And EPC Cleanup

Tasks:

1. move deterministic radiomap reflection accumulation fully behind
   radiomap-owned reflection replay boundaries,
2. remove remaining field-oriented payload assembly from coherent radiomap
   reflection,
3. replace EPC reference-grad reliance on the supported radiomap path.

Exit criteria:

1. reflection no longer contributes avoidable field-style kernel overhead to
   deterministic radiomap,
2. reflection backward is materially lighter.

## Phase 7: Coherent Mode Scalarization Cleanup

Tasks:

1. make coherent deterministic radiomap accumulate complex scalar amplitude per
   cell directly,
2. stop exposing vector electric-field buffers as the main coherent payload,
3. keep any necessary polarization basis work inside the fused replay kernels,
4. preserve coherent diagnostics only as opt-in instrumentation.

Exit criteria:

1. coherent deterministic radiomap is structurally scalar-observable-first,
2. coherent mode no longer resembles field-monitor execution.

## Phase 8: Deletion And Closure

Tasks:

1. delete temporary compatibility shims,
2. delete deterministic radiomap dependencies on field-monitor-specific kernel
   modules,
3. update plan docs and performance reports,
4. archive superseded design notes when the new architecture is authoritative.

Exit criteria:

1. deterministic radiomap is visibly decoupled from field monitor execution,
2. only low-level shared physics and geometry remain shared.

## Validation And Acceptance

Follow the repository validation workflow and keep benchmarks serial.

Recommended validation order after each phase:

```bash
conda activate witwin2
cd channel
python -m pytest tests/scene/test_radio_map_monitors.py
python -m pytest tests/backend/test_native_kernel_consistency.py
python -m pytest tests/mixed/test_mixed_path_regression_scenes.py
python -m pytest tests/main/test_radiomap_gradients_three_cubes_main.py --gpu
python -m pytest tests/main/test_radiomap_monte_carlo_gradients_three_cubes_main.py --gpu
```

For performance and launch-count acceptance, keep the maintained dense scene:

1. `three cubes`, `256 x 256`,
2. coherent and incoherent deterministic modes,
3. representative geometry and material gradient traces.

Default launch-count acceptance contract:

1. treat `matched_isotropic` deterministic radiomap as the primary acceptance
   contract for diffraction launch reduction,
2. treat projected-polarized coherent measurements as secondary diagnostics,
   not the main success metric for this plan.

## Required acceptance outcomes

### Correctness

1. output parity remains within current tolerances,
2. gradient parity remains within current tolerances,
3. no maintained scene regresses in path-family counts unexpectedly,
4. pair-ratio reduction is not counted as success unless measured kernel-launch
   totals also go down on the maintained matched-isotropic scene.

### Kernel structure

1. deterministic radiomap forward no longer routes through the generic
   field-evaluation hot path on supported workloads,
2. deterministic radiomap backward no longer routes through generic UTD Dr.Jit
   replay on supported workloads,
3. forced Python tile loops are not used as the primary solution for dense
   launch-count reduction.

### Memory

1. peak memory is bounded by compact hot state plus cell buffers, not by
   field-monitor-style pair payload expansion,
2. higher-order builder no longer materializes large cold metadata payloads on
   the hot path.

## Success Criteria

This refactor is complete when all of the following are true.

1. deterministic radiomap kernels live under radiomap-owned kernel namespaces,
   not field-monitor namespaces.
2. deterministic radiomap accumulation is scalar-observable-first.
3. local wedge discovery replaces higher-order global candidate expansion on the
   supported deterministic path.
4. diffraction forward and backward no longer depend on the generic public UTD
   Dr.Jit replay path on supported dense workloads.
5. coherent deterministic radiomap no longer resembles field-monitor vector
   field execution.
6. the maintained dense scenes show materially lower kernel-launch counts
   without changing results or gradients.

## Immediate Recommended Implementation Order

The highest-value order is:

1. Phase 0 completion for dense matched-isotropic launch instrumentation,
2. audit and replace remaining invalid native `..._into(...)` output contracts,
3. Phase 1: ownership split,
4. Phase 2: hot and cold state split,
5. Phase 3: local wedge discovery,
6. Phase 4: fused radiomap diffraction forward,
7. Phase 5: fused radiomap diffraction backward,
8. Phase 6: reflection and EPC cleanup,
9. Phase 7: coherent scalarization cleanup.

This order attacks the current launch-count bottlenecks first instead of
spending more time on outer Python cleanup with limited remaining payoff.
