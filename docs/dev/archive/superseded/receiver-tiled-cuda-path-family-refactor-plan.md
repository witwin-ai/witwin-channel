# Receiver-Tiled CUDA Path-Family Refactor Plan

## 1. Scope

This document defines a CUDA-first refactor plan for the current field-monitor bottlenecks in
reflection and diffraction tracing. The target problem is the failure regime where receiver-grid
resolution and surviving path or ray count grow together, especially workloads such as:

- `512 x 512` receiver grids
- `40000` reflection or diffraction rays
- mixed reflection and diffraction families on the same monitor plane

The plan is intentionally native-first:

- the hot path must execute in C++/CUDA kernels,
- Dr.Jit remains an orchestration and AD boundary layer,
- new large intermediate products must not be materialized on the Python side,
- new designs should favor tile-local and family-local execution instead of dense global Cartesian
  expansion.

This is a development-plan document. It does not itself change runtime behavior.

## 2. Problem Statement

The current cell-state work reduced state width and path-payload duplication, but it did not remove
the main dense-field cliff. The remaining bottleneck is not "each receiver stores all states". The
remaining bottleneck is the many-to-many accumulation operator itself.

### 2.1 What Is Not the Main Problem

The current implementation already stores propagation state globally. Receiver outputs are mostly:

- final complex field accumulators,
- optional sparse path references for path export,
- no large persistent per-receiver state table in the dense field path.

This means a redesign that only replaces "receiver-owned states" with "receiver references global
states" will not remove the dominant cost for dense field monitors.

### 2.2 What Actually Blows Up

The current dense-field failure comes from two operators:

1. UTD accumulation

- current shape: effectively `visible (state, receiver)` pairs
- current execution: explicit pair-index materialization plus visibility filtering, then CUDA
  accumulation
- current symptom: very large temporary pair buffers and heavy global scatter

2. Reflected suffix accumulation

- current shape: effectively `ray or segment x traversed receiver cells`
- current execution: ray-by-ray DDA across the full receiver grid
- current symptom: large traversal counts, heavy atomic pressure, and poor scaling when grid size
  and ray count rise together

### 2.3 Observed Failure Mode

Current benchmark evidence is consistent:

- single-axis increases already push `UTD accumulation` and `suffix accumulation` much harder than
  `state preparation`
- the interaction case `512 x 512 / 40000 / 1 motif` remains the first OOM boundary
- historical reports place most of the forward time in:
  - `diffraction_utd_accumulation`
  - `diffraction_suffix`
- cell-state compaction improved memory, but did not remove the receiver fan-out cliff

## 3. Current Code Reality

The current hot spots are structurally:

- `witwin/channel/kernels/utd/native_impl.py`
  - Python still explicitly builds `state_idx` and `rx_idx`
  - visibility is applied before the native kernel
- `witwin/channel/kernels/utd/utd_accumulate.cu`
  - one thread effectively owns one `(state, rx)` pair
  - results are scattered into receiver outputs by atomics
- `witwin/channel/monitors/field/grid_diffraction.py`
  - suffix tracing expands `n_total_rays = n_states * rays_per_state`
- `witwin/channel/kernels/suffix_grid/native_impl.py`
  - rays are still chunked on the Python side
  - the kernel still walks per-ray DDA through the receiver plane

Reflection already contains the seeds of a better organization:

- `witwin/channel/trace/reflection/paths.py`
  - collects unique reflection chains
  - groups by chain and image-source-like descriptors
- `witwin/channel/trace/reflection/epc.py`
  - already separates path descriptors from EPC math
- `witwin/channel/trace/reflection/api.py`
  - already has an "EPC" policy distinction

The main issue is that the most expensive receiver-side work is still organized around dense
receiver fan-out rather than tile reuse.

## 4. Lessons Worth Borrowing from `mmWave-Simulator`

The most valuable ideas in `E:\Code\mmWave-Simulator` are not library-specific. They are execution
shape ideas that transfer well to this codebase.

### 4.1 Discover Path Families First, Reuse on Neighbor Receiver Tiles

In `optixPathTracer.cu`, stage 1 discovers reflection path families and stores compact path
descriptors into `tmp_results`, then stage 2 replays those paths against receiver-grid tasks.

The essential pattern is:

- do not re-discover the same specular family independently for every receiver,
- promote the reusable object to a path family,
- replay that family over a receiver region that can share it.

This is directly applicable to our reflection monitor, and partially applicable to diffraction if
we reinterpret the reusable object as an edge-family or wedge-family descriptor rather than a
specular chain.

### 4.2 Use Image Transform as the Compact Representation for Specular Chains

`mmWave-Simulator` stores:

- `image_transform`
- `hit_cnt`

instead of storing a full hit-point sequence as the primary reusable payload.

This is the right compact descriptor for specular reflection families in our code as well. It is
strictly better than using hit-point history as the hot storage for replay, because:

- it is smaller,
- it is stable under reuse across nearby receivers,
- it can regenerate EPC geometry later.

This idea should be adopted for specular reflection families. It should not be forced onto
diffraction families, because diffraction does not naturally reduce to a pure image-transform chain.

### 4.3 Separate Discovery and EPC

`mmWave-Simulator` has a very clear stage split:

- stage 1: discover candidate path families
- stage 2: EPC and accumulate

This aligns strongly with our existing reflection exact-replay structure. We should push the same
separation deeper into native kernels:

- discovery generates compact family descriptors and coarse receiver-tile coverage
- EPC works only on those family-tile tasks

This is the single most important architectural transfer.

## 5. Design Position

The right refactor is not:

- "store per-receiver references to global states"

The right refactor is:

- "promote reusable propagation objects to compact path families"
- "assign those families to receiver tiles"
- "run EPC and accumulation only for `(family, tile)` tasks"
- "keep the heavy math inside CUDA kernels"

This preserves correctness while attacking both memory and time.

## 6. Target End State

The target architecture has four layers:

1. Family discovery

- discover reusable propagation families once
- keep descriptors compact

2. Tile assignment

- identify which receiver tiles may need each family
- perform coarse culling before EPC

3. EPC

- run exact geometry and field evaluation per `(family, tile)` task
- keep replay in native CUDA kernels

4. Tile-local accumulation

- accumulate in tile-local memory first
- flush to global monitor outputs only once per tile buffer

The dominant dense-field path should no longer materialize:

- full `state x receiver` tables
- full per-ray full-grid traversal products
- large Python-side pair arrays for visibility and fan-out

## 7. Proposed Data Model

### 7.1 Receiver Tile

Define a first-class receiver tile object for dense field monitors:

- `tile_id`
- plane axis and plane position
- `tile_i0`, `tile_i1`
- tile dimensions
- contiguous receiver coordinate buffer
- tile AABB in world space

Recommended tile sizes:

- reflection EPC: `16 x 16` or `32 x 16`
- diffraction UTD: `16 x 16`
- suffix traversal: `16 x 16`

The exact choice should be benchmarked, but the architecture must assume tiles are the unit of
reuse.

### 7.2 SpecularPathFamily

For reflection, the compact reusable object should be:

- `family_id`
- `chain_depth`
- `image_transform`
- canonical primitive chain
- per-bounce plane identity or primitive identity
- per-bounce material slot or primitive-to-material indirection
- optional representative hit points for debugging only

Important rule:

- hit points are cold metadata
- image transform and primitive chain are the hot descriptor

### 7.3 DiffractionEdgeFamily

For diffraction, the reusable object should not be "a receiver-specific path". It should be an
edge-family descriptor:

- `family_id`
- `edge_id`
- prefix family identity
- ownership code
- wedge parameters
- compact incident field state
- incident basis and Jones payload
- optional inserted-reflection descriptor
- coarse support bounds on the receiver plane

This object represents "one diffraction family that can be evaluated over many receivers", not "one
receiver-specific state result".

### 7.4 TileTask

The execution unit becomes:

- `SpecularTileTask = (specular_family_id, tile_id)`
- `DiffractionTileTask = (diffraction_family_id, tile_id)`
- `SuffixTileTask = (segment_family_id, tile_id)`

The queue of tile tasks is what drives the EPC kernels.

## 8. Reflection Refactor

Reflection is the cleanest place to apply the `mmWave-Simulator` ideas almost directly.

### 8.1 Discovery

Replace reflection discovery outputs that currently emphasize replay-ready geometry with a more
compact family cache:

- use the current chain discovery machinery to identify unique primitive chains
- promote `image_transform + primitive chain` to the canonical hot descriptor
- keep hit points and full plane-point sequences as cold replay metadata or regenerate them later

This should reuse and then simplify:

- `witwin/channel/trace/reflection/paths.py`
- `witwin/channel/trace/reflection/epc.py`

### 8.2 Tile Coverage

For each specular family:

- compute a coarse support mask over receiver tiles
- use the transformed image-source geometry to get a fast tile relevance test
- do not immediately explode to per-receiver work

The tile coverage test can be conservative. It only needs to avoid false negatives.

### 8.3 EPC Kernel

Introduce a CUDA kernel family for exact reflection EPC per tile:

- inputs:
  - receiver tile coordinates
  - specular family descriptor
  - scene primitive and material buffers
- outputs:
  - tile-local field contributions
  - optional tile-local replay metadata in debug or audit mode

Key properties:

- the kernel recomputes hit points and field transport from the compact descriptor
- no Python-side per-receiver replay loop
- no global hit-point table required for hot execution

### 8.4 Why This Helps

This reuses discovery across neighboring receivers and removes the need to repeatedly replay the
same chain from scratch for each receiver in a global fan-out structure.

## 9. Diffraction UTD Refactor

Diffraction needs a different reuse object. The direct transplant is not `image_transform`; the
direct transplant is the `family + tile replay` pattern.

### 9.1 Discovery

Current diffraction state building should be split into:

- source-side family construction
- receiver-side exact evaluation

The compact diffraction family should capture:

- edge identity and adjacency
- source-side field state
- prefix reflection lineage or family id
- ownership class
- compact support hints for the receiver plane

### 9.2 Coarse Tile Culling

Before EPC, each diffraction family should be assigned to a conservative receiver-tile
footprint using:

- edge position and orientation
- source position or compact prefix image-source descriptor
- monitor plane
- optional power or support bounds

This stage must be native and cheap. It should generate `DiffractionTileTask` rather than
`(state, receiver)` pairs.

### 9.3 Exact UTD Tile Kernel

Replace the current dense pair path with a fused tile kernel:

- one block or cooperative group handles one `(state block, receiver tile)` region
- receiver coordinates are loaded once per tile
- state descriptors are loaded once per state block
- visibility, geometric validity, ownership, UTD coefficient evaluation, and accumulation all
  happen inside the kernel

Crucial rule:

- do not materialize `pair_idx`, `state_idx`, or `rx_idx` arrays on the Python side

Instead:

- kernel-internal loops enumerate `local_state x local_receiver`
- invalid pairs are rejected in registers or shared memory
- the only global outputs are tile accumulators

### 9.4 Visibility

The current code keeps visibility outside the CUDA kernel because it depends on BVH traversal. The
refactor should move to one of two native patterns:

1. OptiX-backed visibility inside the tile replay path
2. a native coarse visibility pass that reduces candidate tiles, followed by exact segment
   visibility inside the kernel launch path

The preferred long-term direction is the first one. The core issue is not whether OptiX or custom
BVH is used. The core issue is that visibility must no longer require Python-side Cartesian
materialization.

## 10. Reflected Suffix Refactor

Suffix is currently organized as per-ray traversal. That is correct physically, but it is the wrong
reuse unit for dense field monitors.

### 10.1 Promote Segment Families

After the specular or diffraction prefix is fixed, the reflected suffix should be reorganized into
segment-family work:

- a segment family is a traced ray segment or bounce segment packet that shares:
  - origin family
  - segment direction family
  - field payload family
  - common or nearby support on the receiver plane

The point is not to force all rays to be identical. The point is to bin coherent rays before
receiver accumulation.

### 10.2 Tile Binning Before DDA

Instead of letting every ray walk the full receiver grid independently:

- compute a conservative tile AABB for each segment family
- emit `SuffixTileTask` only for touched tiles
- run DDA locally inside the tile

This removes the worst form of full-grid fan-out.

### 10.3 Tile-Local Accumulation

The suffix kernel should:

- maintain tile-local complex accumulators in shared memory when feasible
- resolve per-cell contributions inside the tile
- flush one compact tile buffer to global memory

This lowers:

- global atomics
- repeated grid-coordinate loads
- full-grid traversal overhead for local segments

### 10.4 Packetization

Where rays are coherent, a second-level optimization is possible:

- process small ray packets per tile
- share repeated plane-coordinate and blocker computations

This should be added only after the tile-task reorganization is correct.

## 11. AD Strategy

The requirement is clear: the hot path should stay native, not fall back to Dr.Jit gather or
subset operations when the workload is large.

### 11.1 Principle

For each new native kernel family, ship three native entry points:

- primal
- JVP
- backward

Dr.Jit should only:

- own the `CustomOp` boundary,
- pass tensors and gradients into the native kernel,
- receive native outputs.

### 11.2 Discovery vs Replay

Discovery is usually not the place where full gradients are required.

Recommended policy:

- discovery and coarse tile assignment can be non-differentiable unless an explicit research need
  says otherwise
- EPC and final field accumulation must support AD for supported differentiable workloads

This is already close to how reflection EPC is conceptually treated today. The refactor
should make this explicit and native.

### 11.3 No New Dr.Jit Fallbacks

Do not add new large-workload fallback paths that:

- rebuild pair indices in Dr.Jit,
- scatter with Python-side reductions,
- or replay full path families on the Python side.

If a native AD path is not ready for a new kernel, the feature should stay behind a development
flag until primal and AD are both acceptable.

## 12. Proposed Kernel Inventory

### 12.1 Reflection

- `reflection_family_discovery_forward`
- `reflection_family_tile_assign`
- `reflection_family_replay_forward`
- `reflection_family_replay_jvp`
- `reflection_family_replay_backward`

### 12.2 Diffraction UTD

- `utd_family_build_forward`
- `utd_family_tile_assign`
- `utd_tile_accumulate_forward`
- `utd_tile_accumulate_jvp`
- `utd_tile_accumulate_backward`

### 12.3 Suffix

- `suffix_segment_tile_assign`
- `suffix_tile_accumulate_forward`
- `suffix_tile_accumulate_jvp`
- `suffix_tile_accumulate_backward`

### 12.4 Shared Infrastructure

- `receiver_tile_build`
- `tile_task_prefix_sum`
- `tile_task_compaction`
- `tile_output_flush`

## 13. Integration Map to the Current Repository

### 13.1 Python Becomes Orchestration, Not Expansion

The Python layer should be reduced to:

- build monitor-plane tile metadata
- choose native backend and execution mode
- launch discovery kernels
- launch tile assignment kernels
- launch EPC kernels
- package results

The Python layer should stop doing:

- explicit large Cartesian pair creation
- large receiver-side gather loops
- large replay math on packed arrays

### 13.2 Modules Likely to Change

Primary touch points:

- `witwin/channel/trace/reflection/paths.py`
- `witwin/channel/trace/reflection/epc.py`
- `witwin/channel/trace/reflection/api.py`
- `witwin/channel/monitors/field/grid_diffraction.py`
- `witwin/channel/kernels/utd/native_impl.py`
- `witwin/channel/kernels/utd/*.cu`
- `witwin/channel/kernels/suffix_grid/native_impl.py`
- `witwin/channel/kernels/suffix_grid/*.cu`

New native modules are likely warranted:

- `witwin/channel/kernels/reflection_family/*`
- `witwin/channel/kernels/utd_tile/*`
- `witwin/channel/kernels/suffix_tile/*`
- `witwin/channel/kernels/receiver_tile/*`

### 13.3 Reflection as the First Landing Zone

Reflection should be migrated first because:

- `mmWave-Simulator` gives a strong prior for the architecture
- the compact specular descriptor is naturally `image_transform`
- the current repository already has reflection path-family semantics

Once reflection family replay is proven, diffraction should adopt the same `family + tile` pattern
with an edge-family descriptor.

## 14. Implementation Phases

## Phase A: Receiver Tile Infrastructure

### Objective

Create a shared receiver-tile representation and native task queue infrastructure.

### Deliverables

- native tile builder from monitor-plane metadata
- tile coordinate and AABB buffers
- generic tile-task compaction helpers

### Acceptance

- reflection, diffraction, and suffix paths can consume the same receiver-tile descriptor
- no user-visible behavior change

## Phase B: Reflection Family Cache and Tile Replay

### Objective

Convert reflection EPC to a native `family + tile` architecture.

### Deliverables

- specular family descriptor using image transform
- tile assignment kernel
- EPC kernel per `(family, tile)`
- removal of Python-side reflection EPC loops for dense field tracing

### Acceptance

- numerical parity with the existing reflection monitor on benchmark scenes
- improved scaling on large receiver grids
- no large Python-side replay expansion

## Phase C: UTD Tile Accumulation

### Objective

Replace explicit `(state, receiver)` pair expansion with fused tile accumulation, but do it in two
landings so native visibility does not block the first usable tiled path.

### Phase C1: Tiled UTD Accumulation on Existing Diffraction State

#### Objective

Land a first dense-field tiled UTD path without waiting for a full visibility-stack rewrite.

#### Deliverables

- reuse the current compact diffraction state as the first hot EPC descriptor
- add native receiver-tile assignment from diffraction-family or state support hints
- add a fused tile UTD kernel that internally enumerates `local_state x local_receiver`
- remove Python-side dense `pair_idx`, `state_idx`, and `rx_idx` creation for dense field mode
- keep EPC and accumulation inside CUDA, even if visibility still uses an intermediate
  native boundary in the first landing

#### Acceptance

- parity with the current UTD accumulation for supported workloads
- no Python-side dense Cartesian pair materialization on the primary dense-field path
- materially lower temporary-memory pressure on `512 x 512 / 40000 / 1`

### Phase C2: Native Visibility Migration for UTD Replay

#### Objective

Remove the remaining visibility split so tiled UTD replay is fully native end to end.

#### Deliverables

- move exact segment visibility into the native tile replay launch path
- support either OptiX-backed visibility or an equivalent native BVH path behind the same kernel
  interface
- fuse visibility rejection with geometric validity, ownership, coefficient evaluation, and tile
  accumulation
- remove Python-side visibility filtering and pair compaction from the primary dense-field mode

#### Acceptance

- parity with the Phase C1 tiled path and with the current reference UTD accumulation
- no Python-side visibility filtering on the primary dense-field path
- improved forward scaling beyond Phase C1 on visibility-heavy scenes

## Phase D: Suffix Tile Accumulation

### Objective

Replace per-ray full-grid traversal with tile-local work, but keep the first landing close to the
current exact segment representation.

### Phase D1: Exact Segment to Tile Binning

#### Objective

Stop sending every reflected segment through full-grid traversal before introducing higher-level
segment families or packet abstractions.

#### Deliverables

- keep the current exact segment representation as the first landing unit of work
- compute a conservative receiver-tile AABB for each active segment
- emit tile tasks only for touched tiles instead of walking untouched grid regions
- keep the binning path native and compatible with the existing suffix kernel inputs

#### Acceptance

- parity with existing suffix results on controlled scenes
- no per-segment full-grid traversal on the primary dense-field suffix path
- peak tile-task counts remain bounded on the benchmark suite

### Phase D2: Tile-Local DDA Accumulation

#### Objective

Move DDA traversal and accumulation into touched tiles and flush compact tile outputs instead of
scattering from full-grid ray walks.

#### Deliverables

- tile-local DDA traversal kernel for touched tiles only
- tile-local complex accumulators with one compact flush per tile buffer
- lower global atomic pressure and lower repeated coordinate loads
- compatibility with the current exact suffix field math and blocker semantics

#### Acceptance

- parity with existing suffix results on controlled scenes
- materially lower accumulation time and lower atomic pressure on interaction workloads
- materially lower full-grid working-set pressure relative to the current suffix path

### Phase D3: Optional Segment Coherence and Packetization Follow-Up

#### Objective

Add a second optimization layer only after D1 and D2 are numerically correct and benchmarked.

#### Deliverables

- optional segment-family binning for coherent reflected segments
- optional small ray-packet processing per tile
- shared blocker or coordinate reuse only where coherence is demonstrated by benchmarks

#### Acceptance

- no regression in numerical parity from D2
- measurable speedup on coherence-friendly scenes
- packetization stays optional and is not required for the first tiled suffix landing

## Phase E: Unified Native Replay and AD Rollout

### Objective

Ensure the full dense-field pipeline is native for primal and AD.

### Deliverables

- native JVP and backward kernels for all new replay and accumulation operators
- execution policy that selects native family-tile path by default for dense field tracing

### Acceptance

- no large-workload Dr.Jit fallback on the primary dense-field path
- main gradient benchmarks run through native kernels

## 15. Performance Expectations

If implemented correctly, the main gains should be:

### 15.1 Memory

- remove Python-side `O(states x receivers)` temporary pair arrays
- lower suffix full-grid working-set pressure
- replace receiver-global fan-out with bounded tile-local working sets

### 15.2 Time

- better reuse of specular families across neighboring receivers
- fewer redundant replays of identical or near-identical families
- lower global atomic pressure through tile-local accumulation
- improved cache behavior from tile-local coordinate reuse

### 15.3 What This Will Not Magically Fix

Even after this refactor, exact dense-field evaluation still scales with the number of physically
relevant contributions. This design removes wasteful expansion and improves reuse. It does not turn
exact dense-field tracing into a sublinear solver.

If future workloads exceed even the tiled exact path, the next step would be approximate methods
such as low-rank tile bases or multipole-like expansions. Those are outside the scope of this
document.

## 16. Validation Plan

### 16.1 Correctness

Required parity checks:

- reflection EPC vs existing EPC on small canonical scenes
- diffraction UTD tile kernel vs existing UTD accumulation on wedge scenes
- suffix tile kernel vs existing suffix accumulation on controlled reflection scenes
- end-to-end `Scene + Tracer + Result` parity on dense field monitors

### 16.2 AD

Required AD checks:

- JVP parity for reflection EPC
- JVP parity for diffraction UTD tile accumulation
- backward parity for suffix accumulation
- end-to-end optimization smoke tests on existing GPU gradient workflows

### 16.3 Performance

Required benchmark cases:

- baseline `256 x 256 / 10000 / 1`
- high resolution `512 x 512 / 10000 / 1`
- high rays `256 x 256 / 40000 / 1`
- interaction cliff `512 x 512 / 40000 / 1`
- triangle-heavy stress
- reflection-heavy monitor case
- diffraction-heavy wedge validation case

Metrics:

- forward time by stage
- backward time by stage
- peak allocator memory
- peak tile-task counts
- peak family counts
- peak tile-local scratch size

## 17. Risks

### 17.1 Visibility Integration

Moving visibility fully native is the hardest technical piece. If this remains split across Python
and CUDA, the architecture will not realize its full benefit.

### 17.2 Descriptor Quality

Specular families naturally fit image transforms. Diffraction families do not. The diffraction
family descriptor must stay compact without losing the data needed for exact UTD replay.

### 17.3 Tile False Positives

Coarse tile assignment can over-admit work. Conservative false positives are acceptable, but they
must stay bounded or the refactor will only change the shape of the same explosion.

### 17.4 AD Maintenance Cost

A native-first design means every major kernel family needs primal, JVP, and backward support. This
raises implementation cost but is necessary to avoid large-workload fallback.

## 18. Recommended Execution Order

The highest-value order is:

1. receiver-tile infrastructure
2. reflection family replay with image transforms
3. Phase C1: tiled UTD accumulation on the existing diffraction-state descriptor
4. Phase C2: native visibility migration for tiled UTD replay
5. Phase D1: exact segment to tile binning for suffix work
6. Phase D2: tile-local DDA accumulation
7. Phase D3: optional suffix packetization where benchmarks justify it
8. full native AD rollout and cleanup

This order follows the strongest architectural prior from `mmWave-Simulator`, lands the most
transferable idea first, and creates reusable tile infrastructure before attacking the harder
diffraction kernels.

## 19. Final Recommendation

The ideas from `mmWave-Simulator` are worth adopting, with one important adaptation:

- use `image_transform` as the hot compact descriptor for specular reflection families
- use the same `discovery -> tile assignment -> EPC` architecture for diffraction, but do
  not try to force diffraction into a specular image-transform model

The most important refactor is not "store fewer bytes per state". The most important refactor is:

- stop organizing dense-field work as receiver-global fan-out
- organize it as reusable path families over receiver tiles
- keep EPC and accumulation in native CUDA kernels

That is the path most likely to move the real OOM boundary on `512 x 512 x 40000` workloads.
