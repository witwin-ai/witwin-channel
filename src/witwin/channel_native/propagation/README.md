# Propagation owner contract

## Ownership

This package owns row-aligned topology, continuous geometry, RF fields, and
the enumerated propagation stages shared by Path and Deterministic solvers.
`EvaluatedPaths` is the internal composition boundary; solver accumulation and
public result types remain solver-owned. Under ADR-024 and the completed Phase
6B pin/switch, RayD is the numerical source owner of the complete-row
transmission primal/backward/JVP family, while row contracts, field facades, AD
dispatch, topology, and result assembly remain owned here.

The Phase 6A shared RF dependency closure is active: retained Channel field,
coupled-diffraction, BDPT, and scattering kernels consume RayD public RF
device headers directly. They do not retain or reconstruct a Channel-private
copy of complex, Fresnel, layer-stack, Jones, or field-transport AD math. The
Phase 6B move did not split the complete-row fusion or move the BDPT
transmitted-state family, which remains a complete Channel numerical owner.

ADR-025 freezes diffraction ownership by complete operation. After the atomic
Phase 8A pin/switch/delete, RayD is the sole numerical owner of the pure-wedge
fixed-winner primal/backward/JVP family; this package retains the stable ABI,
typed field/autograd facades, and solver orchestration only. MC Sionna
fixed-tape and coupled R-D/D-D families stay complete Channel numerical owners
and may use RayD public device primitives without adding a UTD sub-launch. Pure
wedge keeps its exporter-locked fast-math contract; retained MC and coupled
operations stay precise. Phase 8B separately owns the sample-tape semantic
rename and native transmitter-visible state planning cleanup. Under ADR-028,
`diffraction_tx_visible_state_plan` preserves all twelve state tensors as exact
aliases at capacity `N` and carries validity only as a CUDA Boolean mask from
the typed RayD axial-edge visibility primitive; it never obtains a host count
or performs Torch compaction.

ADR-029 defines the downstream closure. A device-stable capacity state block
with explicit `diffraction_state_capacity=M` drives RayD exporter planning;
actual counts remain CUDA `int32` tensors. Propagation contracts keep
host-known capacity rows plus CUDA `valid`, and every invalid row is inert
before topology, field, accumulation, backward, or JVP reads it. Dynamic ATen
row shapes, count D2H copies, host compaction, and compatibility/generation-
suffixed facades are forbidden.

`propagation.models.CapacityPathLayout` is the dormant typed contract for this
layout. It carries host `pair_count` and `path_capacity_per_pair` metadata plus
the same runtime-owned `CapacityFailureState`, CUDA-resident row validity,
per-pair counts, and typed local overflow state. Construction is metadata-only
and zero-copy; native producers own the numerical relationship among those
device tensors. It is not a solver result or a live solver boundary until the
ADR-029 atomic activation.

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
publishes no partial rows on pair overflow or bad valid endpoint ids, and has
no live caller before the atomic capacity switch.

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
is introduced before the atomic activation.

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
upper bound), never the device-selected count or public
`path_capacity_per_pair`. Invalid rows are tested before any resolved face,
hit, receiver, or material payload is read. Overflow leaves the complete
candidate block inert and records the reflection bit without an intermediate
trap. The producer is
dormant; the existing compact operations remain the live owner until the
atomic ADR-029 switch.

## Public entry points

There are no root public API exports. The internal package export surface is
`PathTopology`, `PathGeometry`, `PathFields`, and `EvaluatedPaths`.
`propagation.enumerated` owns shared component stages; solver-specific result
conversion stays outside.

## Dependency rules

Topology cannot depend on scene, continuous fields, geometry, or solver policy.
Geometry cannot choose winners; geometry kernel modules depend on runtime, not
scene. Fields cannot discover topology. Enumerated propagation serves Path and
Deterministic; Monte Carlo sampling, MIS, solver results, and deterministic
accumulation remain outside this package. Field facades may call the typed RayD
transmission and pure-wedge entries but cannot reconstruct or fall back from
their native math.

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
