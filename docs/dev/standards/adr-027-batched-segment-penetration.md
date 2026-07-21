# ADR-027: Batched segment-penetration geometry

- **Status:** Accepted (2026-07-20); RayD/Channel foundation and enumerated
  activation implemented, Monte Carlo activation pending
- **Date:** 2026-07-20
- **Kind:** Native geometry, launch/fusion, fixed-capacity, automatic
  differentiation, and cross-repository ownership decision.
- **Related:** [Plan 13](../plans/13-direct-rayd-integration-and-rf-runtime-ownership-plan.md),
  ADR-007 (propagation data ownership), ADR-009 (native fusion ownership),
  ADR-020 (Monte Carlo transmission polarization), ADR-023 (direct typed RayD
  integration), ADR-024 (shared RF/transmission ownership), and ADR-029
  (device-resident capacity results).

## Context

Channel currently discovers a straight transmission chain by repeatedly calling
RayD closest-hit intersection from Python. There are two live marches with
intentionally different endpoint and restart rules:

- enumerated Path/Deterministic discovery repeatedly compacts active endpoint
  pairs and launches one closest-hit query for each depth; and
- the Monte Carlo Basic transmission map repeats the same pattern per
  transmitter and additionally evaluates the ADR-020 incident-polarized wall
  product.

Both paths use Torch expressions, host Boolean tests, `nonzero`, row gathers,
and a Python depth loop around a RayD-owned geometry primitive. The repeated
host decisions and launches are a production hot-path violation and prevent a
single solver-neutral geometry owner. They also hide an important semantic
difference: the enumerated path traces the complete endpoint distance, whereas
the Monte Carlo path stops before the target and uses a different restart norm
and endpoint comparison.

ADR-024 intentionally deferred this work because replacing the repeated calls
with one batched traversal changes the launch and fusion boundary. This ADR
accepts that boundary change. It does not move material eligibility, material
encoding, layer-stack evaluation, or the Monte Carlo estimator into the
geometry API.

## Decision

### 1. Complete operation family and stable typed API

RayD becomes the sole numerical owner of solver-neutral batched straight-segment
penetration geometry. The stable typed boundary remains
`rayd/torch/integration.h`; its identity and target names do not change. Numeric
API versions may advance independently.

The public C++ contract uses the following stable names:

- `SegmentPenetrationPolicy` with exactly the values
  `EnumeratedFullDistance` and `MonteCarloTargetInset`;
- `SegmentPenetrationRequest`, `SegmentPenetrationResult`, and
  `SegmentPenetrationTapeResult`;
- `SegmentPenetrationBackwardRequest` and
  `SegmentPenetrationBackwardResult`;
- `SegmentPenetrationJvpRequest` and `SegmentPenetrationJvpResult`; and
- `segment_penetration_forward`, `segment_penetration_forward_tape`,
  `segment_penetration_backward`, and `segment_penetration_jvp`.

No symbol, type, file, target, identity, facade, or schema introduced by this
work may carry a generation suffix or a temporary name. There is no alternate
dispatcher, dynamic lookup, raw pointer handle, forwarding header, or RayD
Python extension.

`SegmentPenetrationRequest` contains the typed `SceneResource`, contiguous CUDA
`float32` origins and targets of shape `[N, 3]`, an optional contiguous CUDA
`bool[N]` input-active mask, an explicit host-known `input_active_any`, a
host-known non-negative `hit_capacity` `D`, the explicit policy, the frozen
scene-diagonal value used by the selected policy, the solve-owned contiguous
CUDA `int32[1]` capacity failure state, and a caller-assigned non-zero
single-bit `failure_bit`. Origins, targets, the active mask, the scene resource,
all output tensors, and the failure state must share one CUDA device. RayD
validates rank, dtype, contiguity, shape, device, CUDA/SM, Torch ABI, numeric API
version, build identity, and `failure_bit` before partial computation. RayD
never clears the caller-owned failure state.

`input_active_any` is structural request metadata, not a value recovered from a
CUDA tensor. The caller may set it false only when orchestration already knows
that every row is inactive. In that case RayD submits no OptiX traversal and
uses a same-stream CUDA status operation to verify that the optional device
mask contains no true row; a contradiction atomically ORs `failure_bit` and
keeps the complete result inert. This validation may not copy the mask or a
reduction to the host, synchronize the stream, or perform a host Boolean test.
When a mask is absent and `N > 0`, `input_active_any` must be true. A caller
whose arbitrary device mask has no host-known all-inactive guarantee must set
`input_active_any=true`; manufacturing that guarantee through a device-to-host
read is forbidden.

### 2. The two geometry policies are not interchangeable

The implementation must select one policy explicitly; inference from the
caller, solver, tensor shape, or a default is forbidden. The following table is
the frozen production baseline. `s` is the input origin, `q` the target,
`delta = q - s`, `L = ||delta||2`, `p` the current hit point, `t` the current
closest-hit distance, and `S` the frozen scene diagonal.

| Rule | `EnumeratedFullDistance` | `MonteCarloTargetInset` |
|---|---|---|
| Current owner | `propagation.enumerated.transmission._transmission_topology` | `montecarlo.events.transmission.straight_transmission_chains` |
| Direction | native deterministic normalization `delta / max(sqrt((x*x + y*y) + z*z), 1e-9)` | `delta / clamp_min(L, 1e-6)` |
| Degenerate start | row is done when `L <= 1e-9` | initial row is active only when the target-inset remaining distance is `> 0` |
| Scene diagonal baseline | `||max(records.vertices) - min(records.vertices)||2`, reduced from the RayD edge-record vertex table | union the per-structure vertex minima/maxima, then take the bounding-box diagonal L2 norm |
| Initial remaining distance | full `L` | `clamp_min(L - epsilon_inf(q), 0)` |
| Intersection flags | `7` | `7` |
| Accepted hit | primitive id is non-negative, `t` is finite, and `t < remaining` | primitive id is non-negative and `0 < t <= remaining` |
| Hit normal exported to the current caller | geometric normal, normalized with the native `1e-9` deterministic normalization | RayD shading normal, with no additional caller normalization |
| Restart epsilon | `epsilon_2(p) = max(||p||2 * 1e-6, S * 1e-6, 1e-6 m)` | `epsilon_inf(p) = max(||p||inf * 1e-6, S * 1e-6, 1e-6 m)` |
| Restart point | `p + direction * epsilon_2(p)` | `p + direction * epsilon_inf(p)` |
| Next remaining distance | tracked as `clamp_min(L - traveled, 0)`, with `traveled += t + epsilon_2(p)` | `remaining -= t + epsilon_inf(p)` |
| Clear segment | no accepted hit before the full endpoint | no accepted hit before the inset endpoint |

The expression order, constants, strict/inclusive comparisons, finite check,
normal source, and normal normalization in this table are numerical contract,
not implementation suggestions. Signed zero, NaN, infinity, coplanar sheets,
receiver-on-surface cases, and exact endpoint hits must follow the selected
column. A shared helper may be introduced only where it preserves both columns
bit-for-bit; the policies must not be collapsed into one approximate rule.
The scene diagonal is scene-static and must be cached before the solve; neither
policy may perform a solve-time scalar extraction. Replacing either baseline
construction with a shared RayD scene bound is allowed only after its float32
value is proven bitwise identical for the policy's supported scenes.

The current Monte Carlo loop stops marching a row after Channel determines that
the material is invalid or the accumulated transmittance is zero. The new RayD
geometry operation is deliberately solver-neutral and may finish the ordered
geometry trace before Channel applies those conditions. Channel must reproduce
the same final blocked/transmittance semantics from the resident hit block; the
extra geometry work must not turn an ineligible hit into a usable event or
change the ADR-020 product order.

### 3. Fixed-capacity hit and tape result

The result capacity is exactly `[N, D]`; it is never allocated from a
device-selected hit count. The forward result contains contiguous CUDA tensors:

- `valid: bool[N, D]` in original segment order and increasing hit order;
- `num_hits: int32[N]`;
- `reached_target: bool[N]`;
- `overflow: bool[N]` as a diagnostic discrete output;
- `distance: float32[N]` and `direction: float32[N, 3]`;
- `t: float32[N, D]`, `position: float32[N, D, 3]`,
  `normal: float32[N, D, 3]`, and
  `geometric_normal: float32[N, D, 3]`; and
- `global_primitive_id: int32[N, D]`.

The tape form additionally returns fixed `[N, D]` primitive and barycentric
tape plus the restart state required by the native VJP/JVP. Tape fields are
opaque to Channel numerical code: Channel may retain and pass them to the typed
companions but may not reconstruct intersection derivatives in Torch or CUDA.
Unused slots are canonical inert rows: `valid=false`, identifiers `-1`, `t=-1`,
and all floating vectors and tape values positive zero. `num_hits` equals the
device sum of `valid` for a successful row. Input-inactive rows have zero hits,
`reached_target=false`, and inert storage. A non-input-inactive
`EnumeratedFullDistance` row with `L <= 1e-9`, and a non-input-inactive
`MonteCarloTargetInset` row whose initial inset remaining distance is zero, has
zero hits and `reached_target=true`. This distinction is discrete and is
preserved by tape/VJP/JVP.

RayD does not return material ids. Channel maps the stable global primitive ids
through its own face/material and geometry-mode resources, checks every valid
slot before any gather, and owns thin-sheet eligibility and component-5
topology packing.

### 4. One batched traversal and the `D + 1` probe

For `N > 0` with `input_active_any=true`, each forward entry issues exactly one
explicit batched OptiX traversal launch. The caller must have a structural
guarantee that at least one row is active; it may not infer the flag by reading
the device mask. Each raygen lane performs at most `D + 1` ordered closest-hit
probes internally. The final probe is mandatory: it distinguishes a clear tail
after exactly `D` accepted hits from an over-capacity segment. There is no
per-depth host loop, host-shaped active compaction, per-transmitter traversal
call, or device count read.

`N == 0` and a structurally all-inactive request declared with
`input_active_any=false` perform no traversal launch and return correctly shaped
inert tensors, subject to the asynchronous device-mask consistency check above.
`D == 0` is valid; an active clear segment succeeds with zero hits, while its
first accepted hit is the overflow probe.

Initialization and transaction-finalization CUDA work is separately visible in
the launch ledger and must have a depth-independent count. It may not be hidden
as another traversal or used to perform RF/material computation. In particular,
the accepted contract means one OptiX batch traversal, not one host call that
internally submits `D + 1` OptiX launches.

### 5. Device fail-loud transaction; no partial geometry

An accepted `D + 1` hit is capacity overflow, never truncation and never an
ordinary clear/blocked result. The RayD operation atomically ORs its assigned
bit into the caller's solve-owned contiguous CUDA `int32[1]` failure state. It
may preserve `overflow` for direct diagnostics, but before the terminal failure
is observable it must make the complete hit/tape result inert: all `valid`
false, all counts zero, all `reached_target` false, identifiers `-1`, `t=-1`,
and all floating payload positive zero. No non-overflow lane from the same batch
may remain usable.

Intermediate geometry operations do not trap. The Channel solve/result boundary
owns one terminal asynchronous native failure operation on the same ordered
CUDA stream after all dependent outputs are inert. The operation must not copy
overflow/count to the host, call `cudaStreamSynchronize`, extract a scalar, or
introduce an implicit synchronization to raise earlier. The failure becomes
loud at the next normal CUDA synchronization boundary. A solver must not return
a usable partial result.

This failure contract composes with ADR-029 `CapacityFailureState`; it does not
create a second transaction flag or a segment-local host exception path.

### 6. Forward tape, VJP/backward, and JVP are one family

The forward-tape, VJP/backward, and JVP entries are part of the same RayD
numerical family and activate atomically. The tape freezes primitive identity,
barycentrics, valid slots, and all policy decisions from forward. Counts,
validity, primitive ids, overflow, policy comparisons, and restart/endpoint
branch decisions are discrete and have no tangent or cotangent.

The companions accept optional tangents/cotangents for the continuous origin,
target, scene-vertex, distance, direction, hit-distance, hit-position, and
normal quantities supported by the typed result. They:

- never retrace, reselect a primitive, or differentiate a `D + 1` decision;
- use the caller's current CUDA stream and a bounded, recorded launch count;
- return exact positive zero for input-inactive and invalid slots;
- preserve current fixed-winner semantics for endpoint and scene-geometry AD;
- keep crossing positions detached where the straight transmission field
  contract currently defines their contribution as zero; and
- preserve the existing face-normal derivative path for valid frozen
  primitives without making material ids differentiable.

Channel autograd may dispatch these companions and combine their results with
Channel-owned native estimator derivatives. It may not rebuild normalization,
restart, intersection, wall-product, or derivative math with Torch expressions,
finite differences, CPU code, or a fallback. Finite differences remain tests
only.

### 7. Ownership after activation

RayD owns:

- the two policy implementations in `SegmentPenetrationPolicy`;
- scene/AS/OptiX ordered closest-hit traversal;
- fixed-capacity hit/tape construction and `D + 1` overflow detection;
- the complete segment-geometry forward-tape/VJP/JVP family; and
- current-stream validation, launch, and device error propagation at its typed
  boundary.

Channel owns:

- solver configuration and the explicit policy choice;
- endpoint pair/grid construction, face-to-material encoding, geometry mode,
  thin-sheet eligibility, winner/topology/component-5 packing, and metadata;
- the ADR-020 Monte Carlo incident TE/TM projection and ordered wall-product
  estimator, including layer-stack inputs, material/frequency AD, blocked-row
  semantics, and component-map accumulation;
- Path/Deterministic field evaluation and result contracts; and
- the solve-owned capacity failure transaction and terminal asynchronous
  failure boundary.

The MC wall product must execute in a Channel native CUDA operation after the
switch. Moving that estimator into RayD, fusing it with traversal, changing its
TE/TM basis or product order, or changing layer-stack launch/reduction behavior
requires a separate accepted ADR. RayD receives no material-policy callback and
does not include Channel-private headers.

## Migration and deletion sequence

Every item below is an independently reviewable commit. A dormant producer
precedes its consumer, and deletion occurs in the same commit that makes the old
path unreachable.

1. **Channel documentation freeze:** accept this ADR, update Plan 13 and the
   architecture summaries. No runtime change.
2. **Dormant RayD geometry family:** add the stable typed API, both policy
   implementations, fixed hit/tape results, failure-state integration, and
   direct forward/tape/VJP/JVP tests. No Channel lock or production caller
   changes.
3. **Dormant Channel contracts:** add the single owning Python/native facades,
   binding/coverage manifests, failure-state wiring, and the Channel-native MC
   material/TE-TM/product operation. The live solvers still use the old path.
4. **Enumerated atomic switch (completed 2026-07-21):** pin the pushed RayD commit, switch Path and
   Deterministic to `EnumeratedFullDistance`, preserve pair-major order, and
   delete the repeated closest-hit discovery loop and now-dead transmission
   active-row/query helpers. No compatibility facade remains. The ADR-008 BDPT
   oracle continues to call the same public enumerated engine and therefore
   consumes this route without a solver-specific dependency. The engine owns
   one solve capacity transaction, keeps actual candidate/guardrail counts as
   device sidecars, and passes the transaction outward when Path or
   Deterministic still has scattering, accumulation, result, PathTable, or
   array packing work. Those solvers enqueue the single runtime observer after
   final assembly; the default ADR-008 oracle observes after field evaluation
   because it has no later owner. The switch deliberately retains the
   downstream legacy canonical valid-row compaction until ADR-029 activation;
   that device-selected-shape boundary remains a release/performance blocker
   and is not counted as completion of the public capacity-result contract.
5. **Monte Carlo atomic switch:** switch MC Basic to one flattened
   `MonteCarloTargetInset` batch, route its resident hit block through the
   Channel native estimator, and delete the Torch depth loop, host Boolean
   breaks, row compaction/index-update physics, and Torch TE/TM/product
   expressions. Keep unrelated scattering users of shared epsilon utilities.
6. **Evidence and closure:** record exactness, AD, launch/timeline, memory,
   performance, packaging, and clean-checkout evidence in both repositories;
   update the RayD lock/fingerprint, owner inventory, duplication ledger,
   launch/resource ledger, migration note, FEATURE_LIST, and authoritative
   documentation.

The switch commits must remove dead `TransmissionClosestHitQuery`,
`query_transmission_closest_hit`, `iter_transmission_active_rows`, and related
facades only after a zero-caller scan. Shared utilities with live scattering or
BDPT callers are retained under their actual owner and are not deleted merely
because the transmission caller disappeared.

## Frozen baseline and performance gates

Before implementation, capture the current base commit, locked RayD commit,
integration-header hash, GPU/driver/toolchain/build fingerprint, and independent
process measurements for both policies. The baseline matrix must include clear,
one-wall, exactly-`D`, `D + 1`, invalid-material, zero-transmittance,
receiver-on-surface, coplanar, zero-length, NaN/infinity negative-contract,
multi-transmitter/receiver, non-default-stream, and AD cases.

For valid non-overflow rows, freeze hit order, valid/count bits, primitive ids,
positions, selected policy normals, topology/result hashes, wall counts,
transmittance, and VJP/JVP outputs. The migration does not authorize a tolerance
increase, changed estimator order, or changed valid-row result.

Performance evidence uses at least two independent processes, one warmup and
seven steady repetitions per process. It records per-stage and end-to-end
median/p95, OptiX/CUDA launch counts, host/device copies, synchronizations,
stream waits, peak temporary/tape bytes, registers, occupancy, local-memory
spills, and ray divergence. Acceptance requires:

- exactly one traversal launch per non-empty forward batch and zero
  per-depth/per-transmitter traversal launches;
- zero production host Boolean/count reads and zero new synchronization/copy;
- at least 10% median improvement in each targeted penetration-discovery stage
  and at least 5% end-to-end median improvement in its target solver case;
- no more than 5% median or 10% p95 regression in non-target cases;
- identical valid-row and final-result hashes; and
- fixed-capacity scratch/tape and peak temporary memory within an explicitly
  recorded budget derived from `N`, `D`, and the named tape fields.

A borderline timing result expands to five independent processes and requires
the paired 95% bootstrap lower bound for improvement to remain above zero. A
memory or launch increase cannot be traded for a timing win unless a separate
accepted decision changes that budget.

## Acceptance

- direct RayD contract tests cover both policies, `N=0`, all inactive, `D=0`,
  clear, one hit, exactly `D`, `D + 1`, mixed overflow/non-overflow lanes,
  degenerate endpoints, exact endpoint hits, non-finite hits, and non-default
  streams;
- `input_active_any=false` proves zero traversal launches, validates the device
  mask on the same stream, and makes a contradictory true mask fail through the
  shared transaction without a host read; active degenerate/full-distance and
  zero-inset rows prove `reached_target=true` while input-inactive rows remain
  false;
- policy tests lock every table row above, including L2 versus L-infinity
  restart, strict versus inclusive endpoint comparisons, target inset, normal
  source, and normalization;
- overflow makes the entire hit/tape, Channel topology, MC wall product, and
  solver result inert before the asynchronous failure is observed;
- static and timeline audits find no `.item()`, host Boolean compaction,
  device-to-host count/overflow copy, `cudaStreamSynchronize`, Python depth
  march, Torch production geometry, or fallback on the replacement path;
- forward/tape/VJP/JVP direct tests cover optional leaves, zero/inactive
  derivatives, fixed-winner duality, and test-only finite-difference oracles;
- Channel exact integration tests cover Path, Deterministic, and MC Basic,
  including ADR-020 polarized oblique incidence and material/frequency/endpoint
  AD;
- every new `_channel_native` symbol has one Python owner, manifest coverage,
  a direct contract test, an end-to-end caller, and a no-fallback test;
- deleted helpers and old call paths have zero production references and no
  compatibility aliases; and
- Channel `quick`, full `cuda`, relevant `nightly`, clean multi-architecture
  wheel/fingerprint checks, and final `release`, plus RayD direct/CTest suites,
  pass from locked clean checkouts.

## Stop conditions

Stop before activation if the implementation needs a host-visible hit count,
dynamic result shape, more than one traversal launch per batch, an inferred
policy, changed policy constants/comparisons, silent truncation, a partial result
on overflow, unstable hit order, invalid-slot reads, Torch/CPU/fallback physics,
finite-difference production AD, a second numerical owner, a compatibility shim,
a temporary generation name, an unfrozen memory increase, changed valid-row
hashes, or performance outside the accepted gates.

## Consequences

Penetration discovery becomes a single solver-neutral RayD geometry family with
an explicit policy instead of repeated Python-controlled intersections. The
fixed `[N, D]` contract removes device-selected host shape decisions and makes
overflow transactional and loud. Channel retains the policy that belongs to
its domain: which hits are eligible transmission events and how the MC
polarized wall product contributes to a result. The cost is a larger resident
hit/tape block, a mandatory `D + 1` probe, and new cross-repository AD and
failure-state evidence; those costs are visible and bounded by the acceptance
gates above.
