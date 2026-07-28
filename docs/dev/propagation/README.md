# Propagation owner contract

## Ownership

This package owns the row-aligned topology machinery, the shared continuous
geometry helpers, RF fields, and the concept-agnostic enumerated engine that
Path and Deterministic solvers share. `EvaluatedPaths` is the internal
composition boundary; solver accumulation and public result types remain
solver-owned.

What this package no longer owns is any single RF interaction. Per-concept
topology discovery, path geometry, and enumerated orchestration were never
three owners: they were one concept split across three stage packages. They now
live one module per concept in `witwin.channel.interactions` — `los`,
`reflection`, `diffraction`, `transmission`, `scattering`, `coupled` — and
`propagation.topology.discovery` was deleted rather than left as an empty
namespace. `propagation.enumerated` is the only production caller of
those concept modules; it drives them and owns none of their physics. The stage
packages here keep exactly the pieces more than one concept shares: row export
and concatenation, endpoint/visibility/edge-state/reevaluate geometry, the
penetration and row contracts, and field evaluation. Under ADR-024 and the completed Phase
6B pin/switch, RayD is the numerical source owner of the complete-row
transmission primal/backward/JVP family, while row contracts, field facades, AD
dispatch, topology, and result assembly remain owned here.

The Phase 6A shared RF dependency closure is active: retained Channel field,
coupled-diffraction, BDPT, and scattering kernels consume RayD public RF
device headers directly. They do not retain or reconstruct a Channel-private
copy of complex, Fresnel, layer-stack, Jones, or field-transport AD math. The
Phase 6B move did not split the complete-row fusion or move the BDPT
transmitted-state family, which remains a complete Channel numerical owner.

ADR-027 assigns the complete RayD straight-segment penetration family behind
the stable typed boundary. The four Channel ABI/facade entries cover primal,
forward tape, VJP/backward, and JVP and require an explicit
`EnumeratedFullDistance` or `MonteCarloTargetInset` policy. Fixed `[N,D]`
storage, the mandatory `D+1` probe, and the exact solve-owned
`CapacityFailureState` are live for enumerated propagation. Path,
Deterministic, and the ADR-008 BDPT oracle all enter the same enumerated engine,
which submits one pair-major `EnumeratedFullDistance` batch, consumes the
compile-cached enumerated scene diagonal, and retains device actual counts in a
sidecar. The former per-depth active-row/closest-hit march is deleted. Monte
Carlo Basic completed its Phase M atomic switch and now consumes the same
`MonteCarloTargetInset` penetration family through its solver-owned
wall-product estimator; the previous scalar/per-transmitter route is deleted.
The canonical selector still compacts valid candidate rows after this
fixed-capacity producer. ADR-032 accepts this explicit, audited compact boundary
as the authoritative `O(K)` production route. Complete no-D2H and public
capacity results are no longer solver goals; E2E latency, peak memory, steady
throughput, capacity/active ratio, exactness, and device headroom are the gates.

`kernels.topology.enumerated_transmission_topology_pack` is the
live component-5 capacity producer for the enumerated route. It consumes the
same failure-state storage as penetration, preserves pair-major rows, checks
validity before every hit/primitive/material read, and makes the entire
topology/count result inert on overflow or contract failure. Actual cardinality
remains in CUDA Boolean validity and contiguous CUDA `int32` sidecars; there is
no host compaction or count read. The engine creates one solve transaction and
returns it as a typed sidecar when an outer solver still has scattering,
accumulation, or public packing work. Path and Deterministic sanitize after
scattering and enqueue the unique runtime observer only after their final
result/array/PathTable assembly. The default ADR-008 oracle has no later owner,
so the engine sanitizes after fields and observes there. This producer-local
statement does not create another count owner or waive the compact boundary's
copy/synchronization budget.

ADR-025 freezes diffraction ownership by complete operation. After the atomic
Phase 8A pin/switch/delete, RayD is the sole numerical owner of the pure-wedge
fixed-winner primal/backward/JVP family; this package retains the stable ABI,
typed field/autograd facades, and solver orchestration only. MC UTD
fixed-tape and coupled R-D/D-D families stay complete Channel numerical owners
and may use RayD public device primitives without adding a UTD sub-launch. Pure
wedge keeps its exporter-locked fast-math contract; retained MC and coupled
operations stay precise. Phase 8B separately owns the sample-tape semantic
rename and native transmitter-visible state planning cleanup. Under ADR-028,
`diffraction_tx_visible_state_plan` preserves all twelve state tensors as exact
aliases at capacity `N` and carries validity only as a CUDA Boolean mask from
the typed RayD axial-edge visibility primitive; it never obtains a host count
or performs Torch compaction.

ADR-029's downstream capacity closure is superseded for production by ADR-032.
The live compact-cardinality owner may copy only audited integer count metadata,
explicitly synchronize the current stream, allocate exact `O(K)` rows, and pack
them on device in stable order. For the frozen depth-3 Munich reflection case it
may issue at most six 4-byte copies, 24 bytes total. This is structural boundary
work, not CPU numerical selection or fallback. Compatibility/generation-
suffixed facades, partial results, and silent truncation remain forbidden.

`propagation.models.CapacityPathLayout` is the dormant typed contract for this
experimental layout. It carries host `pair_count` and historical
`path_capacity_per_pair` metadata plus
the same runtime-owned `CapacityFailureState`, CUDA-resident row validity,
per-pair counts, and typed local overflow state. Construction is metadata-only
and zero-copy; native producers own the numerical relationship among those
device tensors. It is not a solver result or a live solver boundary; no
ADR-029 activation is pending.

`propagation.topology.kernels.deterministic_capacity_finalize` is the dormant
native index producer for the final all-component candidate list. It stable-
groups rows by the frozen receiver-major key `rx_id * num_tx + tx_id`, retains
candidate order within each pair without truncation, and returns CUDA `int64`
source indices plus `CapacityPathLayout`. Invalid candidates are poison-safe.
Any per-pair overflow leaves every public index, validity bit, and count inert
and atomically records its owned bit without trapping in the intermediate.

`propagation.topology.kernels.enumerated_canonical_capacity_select` is the
dormant discrete selector that precedes final pair packing. It omits invalid
poison lanes and exactly reproduces the live receiver/transmitter/depth/
component/sequence stable order, canonical event/object deduplication,
shortest-path winner, and global/per-pair `max_paths` policy. Winners occupy a
compact valid prefix at the host-known candidate capacity, rather than early
pair slots, so later deterministic accumulation retains live valid-row
ordinals. Selection is frozen and has no AD companion; a later native gather
owns continuous primal/JVP/VJP. The selector shares the solve failure state,
publishes no partial rows after a contract failure, and returns device
`num_paths` for every pair. It neither consumes the historical
`path_capacity_per_pair` nor publishes a local overflow: that capacity is
enforced only by later experimental result export/packing. This lets the
selector feed
non-export enumeration and the ADR-008 BDPT oracle without requiring public
result storage, and it remains caller-free under ADR-032.

`propagation.enumerated.canonical_capacity.evaluated_paths_canonical_capacity_gather`
is the dormant continuous owner immediately after canonical selection. It
sanitizes the selector into a new `CanonicalPathSelection`, validates the
compact prefix, source-index uniqueness, source validity, endpoint IDs, and
device counts before reading candidate payload, and gathers all topology,
geometry, and field rows at the original candidate capacity. Its native VJP
uses source-unique scatter and its JVP uses valid-first gather for all eleven
continuous evaluated-path fields. Failure leaves the new selection, counts,
rows, and derivatives inert. It does not perform public pair-major padding,
consume the historical `path_capacity_per_pair`, or have a live solver caller.

`propagation.enumerated.capacity.evaluated_paths_capacity_pack` is the dormant
complete-row producer layered on that same no-trap finalizer helper. One native
initialization pass makes every topology, geometry, field, and layout slot
canonical-inert; the stable finalizer atomically records failure state without
publishing an intermediate error, and a valid-first CUDA gather copies only
successful rows. Thus overflow, an upstream failure, or a bad valid ID cannot
expose partially gathered RF fields. Its native backward uses source-unique
scatter and its JVP uses valid-first gather for all continuous geometry and
real/complex field tensors; absent cotangents/tangents and invalid rows are
native zeros. Topology is non-differentiable, row identity is shared throughout
the packed `EvaluatedPaths`, and no Torch numerical gather or live solver caller
is introduced; no atomic activation is pending.

`propagation.topology.kernels.coupled_candidate_capacity_block` is the dormant
discrete producer for coupled R-D, D-R, and D-D candidate axes. Its internal
capacity is the complete host-known theoretical candidate count
`tx * rx * (2 * groups * edges + edges * (edges - 1))`, bounded by the existing
coupled candidate guardrail; public path capacity and path-selection policy do
not participate. The producer freezes the historical 65,536-row R-D chunk
order (`R-D` then `D-R` inside each chunk), places D-D after every R-D/D-R
chunk, and keeps the ordered off-diagonal edge-pair sequence. Capacity overflow
atomically records the coupled bit and makes every discrete output inert with a
zero device count; it never traps independently. This producer has no AD
surface and no live caller;
mask-aware composed geometry and field companions must exist before activation.

`propagation.topology.kernels.reflection` owns the dormant post-RayD EPC
reflection capacity producer. One operation covers the existing order-1 and
multibounce consumer schemas, stable-selects `visible[N]` rows in their frozen
input order into explicit host capacity `Q`, and carries CUDA `valid`,
`candidate_count`, and overflow state. At activation `Q` comes from the
host-known theoretical EPC batch row count `N` (or an equivalent explicit
upper bound), never the device-selected count or the historical experimental
`path_capacity_per_pair`. Invalid rows are tested before any resolved face,
hit, receiver, or material payload is read. Overflow leaves the complete
candidate block inert and records the reflection bit without an intermediate
trap. The producer is dormant; the existing compact operations are the
production owner under ADR-032 and no ADR-029 switch is pending. ADR-031's
per-pair `Qr` remains a Proposed caller-free experiment and is not public solver
policy.

## Public entry points

`propagation.consumer` is the one public surface in this package: the stable,
solver-neutral contract that packages outside Channel use to obtain propagation
paths. See `docs/dev/consumer/README.md` for its vocabulary, capability record,
validation split, and result shape. It is a façade only — it owns no physics,
adds no second compaction, and never imports a solver.

Everything else here is internal. The internal package export surface is
`PathTopology`, `PathGeometry`, `PathFields`, and `EvaluatedPaths`, and
`propagation.rows` holds the typed row contracts behind them.
`propagation.penetration` holds the typed segment-penetration contracts that
the topology and geometry stages both consume.
`propagation.enumerated` is one module and owns the shared engine, its two
typed config protocols, and its capacity sanitizers; the per-concept stages it
drives live in
`witwin.channel.interactions`, and solver-specific result conversion stays
outside. Path and Deterministic keep using the internal contracts directly
rather than routing through the consumer façade.

## Dependency rules

Topology cannot depend on scene, continuous fields, geometry, or solver policy.
Geometry cannot choose winners; geometry kernel modules depend on runtime, not
scene. Fields cannot discover topology. Enumerated propagation serves Path and
Deterministic; Monte Carlo sampling, MIS, solver results, and deterministic
accumulation remain outside this package. Field facades may call the typed RayD
transmission and pure-wedge entries but cannot reconstruct or fall back from
their native math.

The locked RayD numeric API requires explicit contiguous CUDA Boolean
validity for the migrated transmission and scattering families and explicit
active state for order-1 diffraction export. Facades must forward an existing
canonical device mask; they may not synthesize an implicit all-valid path or
make optional-mask compatibility part of the production contract.

Raw native tuples may exist only inside domain `kernels` modules. A kernel
façade must validate and convert them to a named internal contract before any
solver or propagation pipeline can observe the result.

## Numerical and AD contract

Contract construction is metadata-only and zero-copy. Row order, row identity,
tensor object/storage aliasing, stride, dtype, device, and `requires_grad` must
remain exact. Topology IDs and winners are discrete; geometry and fields use SI
units and the package complex-phase convention.

### AD contract

Fixed-topology reevaluation cannot change winner selection or conceal a detach
boundary. Continuous endpoint, vertex, material, frequency, and field leaves
use explicit native backward/JVP companions. Unsupported tangents and
higher-order derivatives fail before returning a partial result.

`geometry/reevaluate.py::reflection_epc_paths` is the single fixed-winner
reflection re-solve, and it publishes RayD's per-row validity rather than
deciding what to do with it. The enumerated fixed-winner path requires every
row to reproduce and raises otherwise, because it re-solves a winner it just
discovered under the same scene tensors. Reevaluation at NEW endpoint positions
is a different question, and under ADR-037 the consumer publishes the mask per
row: a stationary point that leaves its facet is a complete answer, not a
failure. One implementation, two policies - do not fork it.

## Forbidden fallback

Propagation code must fail loudly when a required native capability or contract
is missing. It cannot recompute geometry on CPU/Torch, load a global extension,
return a zero result, or silently switch algorithms.

## Maintenance

New internal exports or owner moves require a migration note and contract
tests. The completed kernel migration ledger is archived at
`docs/dev/audit/phase12-ops-migration-ledger.json`; dependency changes must pass
the import graph. A new root or solver public export also updates
`ci/public-api-snapshot.json`.
