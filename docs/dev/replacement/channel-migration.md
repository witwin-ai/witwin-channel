# Channel migration and runtime-dependency boundary

## Stage-I Phase 2: Core world owner switch

Channel now consumes the `witwin.core==0.4.0` world contract directly.
`witwin.channel.Scene`, `Structure`, `PhysicalMaterial`, `ReceiverGrid`, and
the other retained logical root names are the exact Core objects, not adapters
or subclasses. The former Channel logical implementations, loader ownership,
pickle rewrites, and `witwin.channel.core.*` compatibility facades are removed.

Construct endpoints with Core `AntennaState` / `ReceiverGrid`, place them in
`Scene(endpoints=...)`, and pass a Core `Scene` or `SceneSnapshot` to one of the
four solvers. Every solver now requires
`reference_frequency_hz=...`. The sole lower-level boundary is:

```python
from witwin.channel.scene import compile

compiled = compile(core_scene, reference_frequency_hz=3.5e9)
```

Channel owns the bounded registry and four-domain resource invalidation. Stable
Core IDs are retained as `int64` maps while native runtime rows remain dense
`int32`. A request/reference-frequency mismatch fails before native work.
RayD `Auto` may choose its full-result pure-CUDA tracer when OptiX is
unavailable; this remains a native GPU implementation choice, not a CPU/Torch
fallback.

## Stage-I Phase 3: propagation consumer contract

External consumers must import the stable versioned module:

```python
from witwin.channel.propagation.consumer import (
    CONTRACT_VERSION,
    EndpointBatch,
    PropagationRequest,
    evaluate,
    reevaluate,
)
```

Version 1 replaces any direct dependency on internal `EvaluatedPaths`,
`propagation.enumerated`, solver results, or native extension helpers. It
publishes actual compact rows and native pair segmentation. There is no
capacity-shaped compatibility result and no public `path_capacity_per_pair`,
`diffraction_state_capacity`, or `Qr`.

The version-1 component set is `los`, `reflection`, `transmission`, and
`diffraction`. Scattering remains outside this consumer boundary: its current
enumerated representation is incoherent power-domain output and does not meet
the coherent transport or canonical pair-major row contracts. A consumer
request containing `scattering` fails during preflight before compute.

Breaking consumer schema or semantics increment `CONTRACT_VERSION` and require
an atomic consumer update. Channel does not preserve both versions or add a
fallback adapter. Radar adoption is a later Stage-II change; Phase 3 does not
change Radar production source or dependencies.

The supported Stage-I release row is CPython 3.11 and Torch 2.10.0. `_channel`
uses the versioned LibTorch/Python extension ABI and is not a LibTorch Stable
ABI artifact. Release wheels are built for Windows x64 and real
`manylinux_2_28_x86_64`, contain native SM87 SASS as part of the full release
architecture set, and retain compute_120 PTX.

## Current decision

`witwin.channel` is the native entrypoint for the capabilities it
advertises. Repository-owned production Python must not import DrJit, Mitsuba,
Sionna, a RayD Python dispatcher, or `witwin.channel`. The old Channel may remain in tests
and benchmarks as an offline correctness oracle; it must never be a production
fallback. The independent radar implementation is a separate product and is
not evidence for either Channel implementation.

The audited sibling roots (`core`, `genesis`, `maxwell`, `radar`, and `studio`)
contain no production Channel imports. This does not prove that external users,
deployed jobs, plugins, or private repositories have migrated.

The platform `core` package's `channel` and `all` extras must route to
`witwin-channel>=0.4,<0.5` before the replacement becomes default-on. The
audited sibling checkout does not yet contain that consumer cutover, so it
remains an explicit external rollout prerequisite rather than accepted
evidence. Application-level canary/default-on state also requires confirmation
from each consumer owner.

### Plan 13 Phase 3: direct typed RayD integration

Channel now source-links RayD commit
`adf0ea2d1481f7548c5ef30c31b4adbaf831f963` into its single `_channel`
extension and calls `rayd::torch` through
`backends/torch/include/rayd/torch/integration_v2.h` (SHA-256
`d133b054e009fc5e9bf719df71cb91a3a0079382acdcbf3c04224d59cd3f7928`).
The copied C signatures, getter indirection, `bridge.h`/`common.cpp`, raw integer
scene handles, and compatibility re-exports are removed. Scene ownership is a
typed RAII `RayDSceneResource`, and edge tables are exposed as
`RayDEdgeRecords` without reconstructing RayD geometry in Python.

Sixteen strict internal ABI names moved to their canonical identity: scene
create/edge records, intersect JVP/VJP, reflection trace primal/tape/JVP/VJP,
reflection EPC primal/JVP/VJP, face-normal JVP/VJP, and diffraction order-1
export now use `rayd_*`; the two Channel-composed geometry operations use
`coupled_rd_geometry_forward` and `coupled_dd_geometry_forward`. The exact
mapping and current owner evidence live in
`docs/dev/audit/phase13-migration-delta.json` and
`docs/dev/audit/phase13-current-native-owner-inventory.json`.

### Plan 13 Phase 4: generic geometry and dead-bridge cleanup

Generic intersection and visibility now use `rayd_intersect_forward` and
`rayd_visibility_forward`. MC edge discovery is owned by
`montecarlo.basic` as `mc_diffraction_discover_edges{,_counted}` and dispatches
the existing Channel CUDA implementation without a RayD bridge alias.

A four-axis reachability audit deleted nine uncallable legacy bindings,
including the crude BDPT diffraction connection exporters and the dead
reflection/path wrappers. The binding count is therefore 202. The RayD fused
diffraction sample-tape producer was historically exposed as
`bdpt_diffraction_accumulation_forward` before its Phase 8B semantic rename; the
Channel `mc_sionna_diffraction_tape_accumulate` primal/JVP/VJP consumer family
is unchanged. Exact deletion and body-hash evidence lives in
`docs/dev/audit/phase13-phase4-dead-binding-reachability.json` and
`docs/dev/audit/phase13-migration-delta.json`.

### Plan 13 Phase 6A: shared RF and resident layer-stack ownership

Channel now pins the pushed RayD candidate
`4cb400acbfcc2da7fda4110d1298d311816905f1`; the locked
`backends/torch/include/rayd/torch/integration_v2.h` SHA-256 is
`c8e162c55a0e5abe789e4f1b19cd6ab00ee4ef59d70244cfc55d58166aeb646b`.
RayD is the unique numerical source owner of the ADR-024 shared RF
complex/medium/Fresnel/layer-stack/Jones primal/dual closure and the complete
`em_layer_stack_eval/backward/jvp` family. Channel still owns all three stable
`_channel` names, the materials facade, Material ABI v3/CSR encoding,
validation, caches and resources.

All Channel consumers include the versioned RayD public headers. The
former Channel-private numerical headers and `kernels/em_debug.cu` are removed
without forwarding aliases or a runtime fallback. Of the 129 frozen helper
records, 112 now have RayD as their active unique source owner, 10 remain
Channel boundary-only tensor/launch adapters, and 7 scattering-table helpers
remain Channel-owned pending Phase 10A activation under accepted ADR-026. The
live binding count remains 202; the
current owner split is RayD 20, layered Channel/RayD 2, and Channel 180.

### Plan 13 Phase 6B: complete-row transmission ownership

Channel now pins the pushed RayD candidate
`3988f0934fec7b521ee5190b0defc0883c84b9e6`; the integration v2 header
SHA-256 is
`6cb18f682e08cb0bb0853507e3b4b82a68e681bb1dad89dc8c36518705f74989`
and its identity is
`rayd.torch.integration.v2.20260719.rf-transmission-sequence`.

The complete `field_transmission_sequence/backward/jvp` family now dispatches
through the source-linked typed `rayd::torch` API. Channel retains the three
stable `_channel` names, field-row schemas, Python/autograd facades and
solver orchestration. RayD is the unique numerical source owner of the primal,
backward and JVP kernels; the former Channel AD translation unit and the
transmission primal section of `field_transport.cu` are removed without a
forwarding shim or fallback.

The move preserves one launch per active primal/JVP/backward entry, current
stream affinity, precise math, CSR traversal, call-local recomputation and the
existing shared-layer atomic accumulation order. The fused
`bdpt_transmitted_light_subpath_state/backward/jvp` family remains a complete
Channel owner and consumes the same RayD shared RF headers. The live binding
count remains 202; the owner split is now RayD 23, layered Channel/RayD 2, and
Channel 177. The frozen 129-helper partition remains 112/10/7 because
Phase 6B moved an operation family, not a new helper source closure.

### Plan 13 Phase 7: diffraction operation-family decision

ADR-025 is accepted, but Phase 7 does not move production code. It freezes
nine diffraction operation families by their full primal/JVP/VJP, fusion,
tape, and compile contracts. The pure-wedge fixed-winner
`field_diffraction_wedge/backward/jvp` family is approved for an atomic Phase
8A move to RayD; until that pin/switch/delete commit, Channel remains its sole
numerical implementation. MC Sionna fixed-tape accumulation and coupled R-D/
D-D fields remain complete Channel owners and stay on precise math. The pure
wedge family alone retains exporter-locked `--use_fast_math`.

Phase 8B renamed the misleading live
`bdpt_diffraction_accumulation_forward` sample-tape producer to
`rayd_diffraction_sample_tape_forward` without an alias or output trimming.
The five dead BDPT diffraction bindings deleted in Phase 4 remain deleted under
the four-axis reachability rule. The live transmitter visibility prefilter will
become a complete Channel planning/selection operation while preserving
the ordered fractions `(0.02, 1/3, 2/3, 0.98)`, any-visible rule, and stable row
selection. The accepted contracts and stop conditions are recorded in
`docs/dev/audit/phase13-diffraction-family-matrix.json` and
`docs/dev/audit/phase13-diffraction-legacy-audit.json`.

### Plan 13 Phase 8A: pure-wedge diffraction ownership

Channel now pins pushed RayD commit
`11e72526cdddf669678975c8921a9d44c6504e20`. The locked integration v2 header
SHA-256 is
`7a2b68f459e7e981a23735271eff2844fe0483d119cf514d59d2032d11be5aef`,
with identity
`rayd.torch.integration.v2.20260719.rf-transmission-sequence.pure-wedge-diffraction`.

The complete `field_diffraction_wedge/backward/jvp` family now dispatches
through the source-linked typed `rayd::torch` API. Channel retains all three
stable `_channel` names, field/autograd facades, row contracts and
solver orchestration. RayD is the unique numerical source owner; the former
Channel pure-wedge CUDA translation unit is deleted without a forwarding shim,
fallback, or second compiled owner.

The move preserves the exact `wedge_row_eval<T>` evaluation order, optional
five-tensor winner-vertex bundle, fixed-winner AD, result schema, caller current
CUDA stream, one launch per active entry and zero-row no-launch. Fast math is
source-local to RayD's pure-wedge TU; MC Sionna, coupled R-D/D-D, transmission
and the other retained Channel families remain precise. The live binding count
remains 202; the owner split is RayD 26, layered Channel/RayD 2, and Channel
Native 174. Evidence and deletion hashes live in
`docs/dev/audit/phase13-diffraction-phase8a-evidence.json` and
`docs/dev/audit/phase13-migration-delta.json`.

### Plan 13 Phase 9: generic scattering runtime ownership decision

ADR-026 is accepted, but Phase 9 moves no source and changes no production
owner. It freezes six complete families containing 17 Channel-facing runtime
contracts for ordered Phase 10A/10B transfer to RayD, together with the seven
shared table-interpolation helpers. Channel remains the production numerical
owner until each dormant RayD candidate is pinned, switched, validated, and
the corresponding local implementation is deleted atomically.

RayD will consume only caller-owned resident tensors. Channel retains
Kirchhoff and phase-screen construction/cache/seed lifecycle,
`scattering_event_probabilities`, topology/C1-C2 packing, RNG/MIS/event policy,
solver accumulation, and results. Table primal/sample/PDF remains on default
CUDA flags; table AD, ensemble, patch, and chain TUs retain `--fmad=false`.
Chain-ensemble geometry stays JVP-only with loud reverse rejection, while
chain-realization geometry retains its implemented VJP/JVP support. See
`docs/dev/standards/adr-026-rayd-generic-scattering-runtime-ownership.md`.

### Plan 13 Phase 10A: table and single-bounce scattering activation

Channel now pins pushed RayD commit
`4577e744adfe8665f7817e3aff5e8e533ec896e7`. The typed scattering header and
integration-v2 header SHA-256 values are respectively
`66d75a20be16057f03cdfb79e3b9dcc85cacec79b555cd73b019259aa510262a`
and `9f95ad9e8e3b790d00f8e762a3e6a09252d46afb65bfc3aba7c42325836cb1fb`;
the RayD shared scattering-table header SHA-256 is
`38ea9be424640301a88a97bccca9ab4bc599191ecfb0b259881ef6a300c96e38`.

The complete table-evaluation AD, table-sampling, single-bounce ensemble, and
patch-integral families now dispatch through typed `rayd::torch` requests and
results. Channel keeps all eleven `_channel` names, Python facades,
resident-resource lifecycle, and solver policy. The five former dedicated
Channel numerical TUs and private table helper header are deleted; the retained
`scattering.cu` contains only `scattering_event_probabilities`. Remaining chain
consumers include RayD's public table header directly, while
`kirchhoff_table_ad.cu` gains no unused dependency.

The live binding count remains 202. The numerical-owner split is RayD 37,
layered Channel/RayD 2, and Channel 163. Table primal/sample/PDF and the
retained event-policy TU remain on their default CUDA flags; table AD, ensemble,
and patch owners retain `--fmad=false` in RayD. Launch count, current stream,
reduction order, atomics, resident tensors, and public API remain unchanged.
At the Phase 10A cut, Channel still owned the six fused chain contracts pending Phase 10B.
Detailed Phase 10A activation, codegen/resource, deletion, and direct-test evidence is recorded in
`docs/dev/audit/phase13-scattering-phase10a-evidence.json`.

### Plan 13 Phase 10B: fused scattering-chain activation

Channel now pins pushed RayD commit
`768b96e42a95f70c32d55f98a72000085317e288`. The typed scattering,
integration-v2, and shared scattering-table header SHA-256 values are
respectively
`ac95c418860d109aeaa96623131592e4df8887992e5fc25ecab71b4ddbf1f55b`,
`0608bfbaf022379bc03442f9baa777ec05cfe3f6ab9b964e2385ec12a7b6c654`,
and `38ea9be424640301a88a97bccca9ab4bc599191ecfb0b259881ef6a300c96e38`.

The complete ensemble-chain and realization-chain primal/backward/JVP families
now dispatch through typed `rayd::torch` requests/results. Channel keeps all six
`_channel` names and typed Python/autograd facades, while the four local
chain CUDA TUs are deleted. `scattering_event_probabilities`, table/phase-screen
lifecycle, topology/C1-C2 packing, RNG/MIS/event policy, solver accumulation,
and results remain Channel owners. The geometry AD truth is unchanged:
ensemble is JVP-only with loud VJP rejection; realization supports VJP/JVP.

The live binding count remains 202. The numerical-owner split is RayD 43,
layered Channel/RayD 2, and Channel 157. All four RayD chain TUs retain
source-local `--fmad=false`; launch count, current stream, reduction order,
atomics, resident tensors, and public API are unchanged. Detailed activation,
codegen/resource, deletion, and direct-test evidence is recorded in
`docs/dev/audit/phase13-scattering-phase10b-evidence.json`.

The move-only cut reduced exact-token duplication from 11.913070% to
11.170566%, removed 12 stale chain regions, and classified three typed-adapter
packing regions. The frozen 10.211512% budget was not relaxed and remains a
Phase 11 nightly/release acceptance blocker; no unrelated deduplication was
mixed into this owner move.

### Plan 13 Phase 11: stable typed-integration naming

Stable naming was activated at pushed RayD commit
`3869c2ab76bb06498dc95e3cf634fdf117529906`. The stable typed boundary is
`backends/torch/include/rayd/torch/integration.h`, with SHA-256
`e88626c4486b99a88737d39dc3ec3d277a5b554b9bd664ba9c384577cd141c86`
and identity `rayd.torch.integration`. The numeric integration API version
remains 2; it is not encoded as a work-in-progress filename, target, or
capability-accumulating identity.

All live Channel C++ adapters, lock/build identity, wheel validation, boundary
tests, feature documentation, and current owner/migration records use the
stable name. No forwarding header, identity alias, dual include, runtime
fallback, binding-symbol change, numerical-owner change, kernel launch change,
or public Python API change is introduced. Earlier Phase 3/6/8/10 commit, path,
identity, and hash records remain explicitly historical activation evidence.

The current lock subsequently advanced to pushed RayD
`102470daf44b649030df3b2554d9ace5c1eea482` for the accepted Phase 8B typed
axial-edge visibility operation. The same stable header now has SHA-256
`65ae4e8e35cf6067cb320a770a1945e2685feab6af44a2233d4db0cfe6b1f435`;
the identity and independently validated numeric API version are unchanged.
Phase 11A/11B also closed the frozen duplication budget at 10.056413%, retired
the legacy RayD entries, and aligned live manifests, workflows, owner records,
and migration governance. Final clean-checkout nightly/release and final
wheel/fingerprint evidence remain explicitly pending in
`docs/dev/audit/phase13-phase11-release-acceptance.json`.

### Plan 13 Phase 6C Phase P: dormant penetration foundation

Channel now locks pushed RayD
`474c122aa3cd6b6d098675e076a73e6f485bd6be`, stable integration-header SHA-256
`57f83ea460e376166fd5ee22a8243a7c1576a290e1de99c0cbe8e86e93392e14`, identity
`rayd.torch.integration`, and independent numeric API version 6. Four new
internal `_channel` entries expose RayD's complete fixed-capacity
straight-segment primal/tape/VJP/JVP family. One Channel-owned
`enumerated_transmission_topology_pack` entry converts valid hit slots into
pair-major component-5 capacity rows. The live binding universe therefore
moves from 229 to 234.

All five entries are dormant. Path, Deterministic, and Monte Carlo Basic retain
their existing live penetration routes until the two accepted atomic
switch/delete commits. No public export, solver result shape, compatibility
alias, alternate backend, or generation-suffixed name is introduced. The
public export count stays 37 and the public API snapshot is unchanged.

The new family shares the solve-owned CUDA `CapacityFailureState`, owns bit
`1 << 7`, and publishes only completely inert hit/tape/topology outputs after
overflow or contract failure. Actual cardinality remains CUDA validity plus
contiguous CUDA `int32` counts. The migration adds no host count/Boolean read,
dynamic result shape, per-depth traversal, intermediate trap, or partial
result. Two policy-specific scene diagonals are frozen during scene
compilation rather than recomputed in a solver hot path.

RayD API 6 also makes validity explicit at every migrated RF boundary:
transmission, all 17 generic scattering runtime contracts, and order-1
diffraction export receive a required contiguous CUDA Boolean mask. Channel
forwards canonical masks and retains no implicit-all-valid or optional-mask
compatibility path. Exact symbol ownership, launch/resource formulas, and the
remaining activation work are recorded in
`docs/dev/audit/phase13-adr027-penetration-foundation.json`.

### Plan 13 Phase 6C Phase E: enumerated penetration activation

Path, Deterministic, and the ADR-008 BDPT discrete-path oracle now share the
single `evaluate_enumerated_paths` transmission route. The engine creates one
solve-owned `CapacityFailureState`; its transmission stage flattens endpoint
pairs in transmitter-major, receiver-minor order and submits exactly one
`EnumeratedFullDistance` RayD batch using the compile-cached enumerated scene
diagonal. Primal mode dispatches the native forward entry, while JVP/VJP modes
dispatch the native tape and derivative companions through Channel autograd.

The Channel topology pack receives the exact same failure-state object/storage
and preserves one capacity row per endpoint pair. Actual candidate and
guardrail counts remain CUDA `int32[1]` sidecars; metadata reports only the
host-known candidate capacity and does not hide a device-to-host count read.
The engine returns the typed transaction as a sidecar when Path or
Deterministic still has work. Path sanitizes after scattering and observes only
after base-result plus synthetic/explicit array packing; Deterministic
sanitizes after scattering and observes only after accumulation, optional
PathTable export, and `Result` construction. The default ADR-008 oracle has no
later owner and therefore sanitizes and observes after its field result. Every
route enqueues the unique runtime terminal observer exactly once. Overflow
therefore makes penetration, topology, fields, diffraction-vector sidecar, and
the returned solve result inert before the asynchronous failure becomes loud;
no intermediate traps or returns a partial result.

This activation replaces penetration discovery only. The canonical selector
still compacts valid rows from the fixed-capacity block and therefore retains a
device-selected result-shape boundary. ADR-032 accepts that explicit, audited
compact boundary because it preserves `O(K)` storage and wins the measured
E2E/memory/throughput comparison. Complete solver no-D2H and public-capacity
results are no longer migration goals.
Path additionally performs a post-sanitizer valid-row compaction before its
legacy result converter so failed `-1` identifiers cannot be gathered. This is
a stable safety boundary, not a second result contract; it is not replaced by
the superseded ADR-029 Phase D capacity pack.

The former `TransmissionClosestHitQuery`,
`query_transmission_closest_hit`, `iter_transmission_active_rows`, their source
modules, Python depth loop, Torch restart/normalization math, and compatibility
exports are deleted in the same atomic switch.

### Plan 13 Phase 6C Phase M: Monte Carlo Basic penetration activation

Monte Carlo Basic now flattens all transmitter/receiver endpoint pairs in
transmitter-major, receiver-minor order and submits one explicit
`MonteCarloTargetInset` RayD fixed-capacity batch. The resulting resident hit
block is consumed directly by the Channel-owned
`mc_transmission_wall_product` primal/VJP/JVP family; there is no per-transmitter
traversal, Python depth march, host Boolean break/compaction, Torch incidence
basis, or Torch wall-product computation on the live production route.

RayD penetration and the Channel estimator receive the exact same solve-owned
`CapacityFailureState`. A `D + 1` hit or downstream resident-contract failure
makes every geometry, estimator, component-map, and final-result output inert.
The sole terminal observer is enqueued once at the final MC Basic result
boundary, after sanitization and assembly, so no partial result is returned.
The existing RayD four-entry penetration family and Channel three-entry
wall-product family changed from dormant to live governance state. One new
three-symbol internal primal/backward/JVP family rooted at
`mc_capacity_failure_component_maps_sanitize` gives Basic a solver-local
five-map sanitizer without importing enumerated internals; binding count moves
from 238 to 241. Public exports, Config, and Result fields remain unchanged.
Basic metadata intentionally replaces `valid_contribution_count` with
`contribution_capacity`: the former promised an actual nonzero count and would
require a forbidden device count read, while the latter is exactly the
host-known component capacity. There is no compatibility alias and no silent
reinterpretation of an actual-count field as capacity.

Phase M does not activate ADR-029. The enumerated canonical selector and Path
post-sanitizer compact boundaries remain authoritative under ADR-032. Final
Phase 12 acceptance uses compact E2E, peak-memory, throughput, exactness,
headroom, and explicit copy/sync budgets rather than complete solver no-D2H.

## API surface changes

### ADR-013 coupled double diffraction (D->D)

`coupled_paths=True` now enables the uniform order-2 compensator family
{R->D, D->R, D->D} instead of the R->D / D->R pair alone. There is no new
`Config` field and no new public toggle: a partial family is non-uniform by
measurement, so the double-diffraction term (component id 7) shares the
existing `coupled_paths` gate, per-block candidate budget, and coupled
accumulator slot. Exported `coupled_paths=True` path tables now include cid 7
rows (kept distinct from cid 3/4 for audits), and the aggregated `coupled`
component map sums cids 3, 4, and 7. Coupled-off solves stay byte-identical.

The semantic capability manifest (`capabilities()`) gains
`coupled_double_diffraction: True` on every solver block that declares
reflection-diffraction coupling support (`path`, `deterministic`,
`montecarlo_bdpt`); `montecarlo_basic`, which does not support coupling, does
not expose the key. This is an intentional additive surface change under the
ADR-003 process; the curated `public-api-snapshot.json` function/class
contracts are unchanged because no public callable signature or `Config` field
changed.

## Rollout states

1. Inventory: route every real call to a supported Native capability or record
   an explicit unsupported product decision.
2. Shadow: execute old and Native implementations independently and record the
   comparison artifact below. Shadow failures must not trigger a production
   fallback.
3. Canary: make Native authoritative for a small declared cohort and retain the
   same correctness and operational evidence.
4. Default-on: the owning consumer makes Native authoritative. This repository
   exposes the Native entrypoint but cannot verify an application router or its
   rollout percentage; consumer-owner confirmation is required.
5. Delete: remove the old runtime integration only after every blocker below is
   closed.

## Shadow evidence artifact

Each maintained scenario must store versioned JSON under the release evidence
location chosen by CI. It must include: schema version, timestamp, release,
scenario/config/seed, Native and oracle commit/build identities, GPU/driver/
CUDA/OptiX/PyTorch metadata, cold and steady timing, peak memory, correctness
metrics and thresholds, pass/fail, and whether either side errored. Raw NPZ/JSON
outputs must be linked by content digest. No maintained Phase 10 shadow artifact
has been recorded in this repository yet; this is a required evidence contract,
not a fabricated run result.

The reduced three-way attempt on 2026-07-11 is recorded in
`path-threeway-shadow-attempt-2026-07-11.json`. Native timing and peak-memory
measurement completed, but both offline oracle processes failed during LLVM/
reference initialization, so the artifact is explicitly failed and does not
close the maintained shadow gate.

## Deletion blockers

Deletion remains blocked until all of the following are true:

- external consumers and private/deployed workloads have a signed inventory;
- maintained shadow and canary artifacts pass for the supported matrix;
- every consumer owner confirms Native is default-on with no production fallback;
- two consecutive release cycles complete without a fallback request;
- all P0/P1 items are closed or explicitly excluded by product decision;
- maintained correctness, performance, memory, cold-start and deployment gates pass;
- wheel smoke and the required GPU/SM matrix have runtime evidence;
- pipeline cache is either implemented and validated or explicitly removed from
  the release requirement.

As of 2026-07-11 the external audit, shadow/canary evidence, owner default-on
confirmation, two-release observation, wheel/SM evidence, and pipeline-cache
gate are not complete. Phase 10 therefore establishes the migration contract
and production dependency boundary but does not authorize deletion.

## Enforcement

Run the local contract with:

```powershell
python ci/check_production_dependencies.py
```

Sibling repositories can be audited without modifying them:

```powershell
python ci/check_production_dependencies.py --consumer-roots ..\core ..\genesis ..\maxwell ..\radar ..\studio
```

Consumer mode rejects the old Channel import only. This intentionally does not
classify Radar's independent DrJit/RayD tracer as a Channel runtime fallback.

## Public API additions (backward compatible)

### ADR-019: `montecarlo.bdpt` coherent combine (2026-07-18)

`witwin.channel.montecarlo.bdpt.Config` gains one field:

- `coherent: bool = False`

This is a purely additive, opt-in switch. The default (`False`) preserves the
existing power-domain incoherent accumulation BIT-IDENTICALLY, so no existing
caller, benchmark, or preset changes behaviour. Existing positional/keyword
construction of the config is unaffected (the field appends after `components`
with a default).

When set to `True`, BDPT sums the enumerated delta/UTD family (`los`,
`reflection`, `diffraction`, plus the `coupled_paths` compensator) coherently
per (tx, rx, component) and finalizes `|sum|^2`, tracking the deterministic
per-component coherent power. Coherent is refused for `transmission`/`scattering`
components (stochastic samplers, no coherent field) and for `ad_mode != 'none'`.
Result metadata records the active combine domain under `metadata["combine_domain"]`
(`"power"` or `"coherent"`). See
`docs/dev/standards/adr-019-bdpt-coherent-combine.md`.

The public-api snapshot updates only the `montecarlo.bdpt.Config`
`contract_sha256` (export count unchanged). No native ABI symbol is added; the
`bdpt_accumulate_connection_samples` binding gains defaulted `combine_domain` /
`coeff_real` / `coeff_imag` arguments (binding count unchanged at 193).

### ADR-021: multi-bounce coherent scattering (2026-07-18)

Three public `Config` classes gain purely additive, opt-in fields. Every default
preserves the existing behaviour BIT-IDENTICALLY, so no existing caller,
benchmark, or preset changes.

- `witwin.channel.deterministic.Config` gains four fields:
  `scattering_coherent: bool = False`, `scattering_chain_max_depth: int = 0`,
  `scattering_chain_samples_per_m2: float = 2.0`,
  `scattering_chain_max_rows: int = 256`.
- `witwin.channel.path.Config` gains three fields:
  `scattering_chain_max_depth: int = 0`,
  `scattering_chain_samples_per_m2: float = 2.0`,
  `scattering_chain_max_rows: int = 256`.
- `witwin.channel.montecarlo.bdpt.Config` gains one field:
  `max_scattering_order: int = 1`.

`scattering_chain_max_depth = 0` disables chain discovery (no allocation, launch,
or RNG); `scattering_coherent = False` keeps the incoherent power scattering
slot; `max_scattering_order = 1` keeps BDPT's terminal single-scatter behaviour.
See `docs/dev/standards/adr-021-multibounce-coherent-scattering.md` and
`docs/dev/plans/10a-scattering-v2-native-interfaces.md`.

The public-api snapshot updates three `contract_sha256` values only
(`path.Config`, `deterministic.Config`, `montecarlo.bdpt.Config`); the public
export count is unchanged at 37. Six new native ABI symbols are added (the ADR-021
chain forwards `scattering_chain_ensemble_eval` / `scattering_chain_realization_eval`
plus their `_backward`/`_jvp` companions), moving the binding count 193 -> 199
(`EXPECTED_NATIVE_BINDING_COUNT`, `EXPECTED_BINDING_COUNT`, and the phase-10
binding-ownership audit `expected_count`). ADR-021's D3 coherent combine adds NO
new primal symbol: it rides a defaulted `scattering_combine_domain` argument on
the existing `deterministic_accumulate_flat` op (and its `_backward`/`_jvp`),
mirroring ADR-019's `combine_domain`.

### ADR-032: compact cardinality stable recovery (2026-07-21)

ADR-029's device-capacity result design is superseded for production. The
measured Munich candidate changed reflection storage from actual `K=2,629`
rows to theoretical `N=4,552,704` rows, increased peak allocated CUDA memory
from 1,071,493,120 to 17,854,649,344 bytes, changed median solve latency from
66.467 to 1,009.587 ms, and reduced throughput from 15.374 to 0.793 solve/s.
Its reserved peak exceeded physical device memory. ADR-031's minimum-sufficient
`Qr=20` candidate still used 11,671,543,808 bytes and 226.429 ms. These are
failed production experiments, not migration benefits.

The stable production boundary returns to the `e7d82d2` `O(K)` compact route.
Its owning cardinality boundary is the sole accepted exception to the general
no-hot-path-D2H rule: it may copy the required integer count metadata, explicitly
synchronize the current stream, allocate exact compact storage, and perform
structural device packing. This is not CPU physics, numerical selection, or a
fallback. For the frozen depth-3 Munich solve the complete reflection budget is
at most six 4-byte D2H copies, 24 bytes total; the measured copy plus immediate
synchronization cost was approximately 0.152 ms.

`e7d82d2` is only the compact numerical/caller base. That historical commit
still contains construction-only `path_capacity_per_pair` and
`diffraction_state_capacity` fields that are silent no-ops on its live route.
Recovery R2 removes them with the snapshot; they must not be mistaken for an
already-clean historical public schema.

Public Path/PathTable shapes again represent actual compact rows and
`max_num_paths` represents the maximum actual count, not provisioned capacity.
The intermediate ADR-029 fields `path_capacity_per_pair` and
`diffraction_state_capacity`, capacity-shaped public `valid`/`num_paths`, and
the ADR-031 `reflection_candidate_capacity_per_pair` (`Qr`) field are not part
of the recovered formal public API or production planning contract. They are
removed with the public API snapshot; no ignored compatibility field, alias,
generation-suffixed name, or dual dispatch is retained.

ADR-029 capacity producers/selector/gather/packers, ADR-031 raw reflection
capacity code, and ADR-030 SourceLane/pair-reducer code may remain as strictly
internal caller-free experiments with their direct tests. They must not appear
in capabilities, defaults, formal Config, solver E2E dispatch, or fallback.
The existing `CapacityFailureState`, inert-output sanitizers, and unique
terminal observer remain authoritative for accepted fixed-capacity operations;
stable recovery does not permit silent truncation or a partial result.

Final migration acceptance requires Munich median latency no greater than
76 ms, steady throughput at least 13.8 solve/s, peak allocated memory no greater
than 1.25 GiB, at least 1 GiB physical device headroom, successful reflection
capacity/active ratio exactly 1, no more than six count copies/24 bytes, and
bitwise equality of all 24 logical PathTable fields with the compact baseline.
It also requires quick/cuda/nightly/release, wheel/fingerprint, public snapshot,
manifest, and no-partial-result evidence. Whole-map atomic nondeterminism must
be reported honestly rather than relabelled as exact.

No GPU tiled EPC, incremental canonical merge, SourceLane activation, exporter
AD family, CUDA Graph, or other large feature is included in this recovery.
Such work requires a future separately accepted ADR and evidence that improves
E2E latency, peak memory, throughput, capacity utilization, exactness, and
concurrency together.

### 2026-07-22: validated RayD package source discovery

RayD `402262d3b0c07dffb9d51d1852abb97ab2280f2f` changes packaging only: the
stable integration header remains API 6 with SHA-256
`57f83ea460e376166fd5ee22a8243a7c1576a290e1de99c0cbe8e86e93392e14`, and no
geometry/RF source, owner, launch, numerical order, solver API, or result schema
changes. The `rayd-torch 0.6.0` wheel now carries a passive relocatable source
bundle plus the full manifest SHA
`9c284b7861d6f25be2f103855a7c8842fc167792633b881ad9b4a0112e1c0800`.

Channel lock schema 2 resolves a valid explicit `RAYD_SOURCE_DIR` first. With no
explicit path, it asks only the selected Python interpreter for the unique
`rayd-torch` distribution and validates its fixed metadata, RECORD ownership,
commit/repository/API/header and every manifest file before source-linking.
Invalid explicit paths, missing/duplicate packages and any metadata/path/source
drift fail without fallback. No `CONDA_PREFIX`, site-packages, or global CMake
search is used, and `rayd.torch` is not imported.

`build_info` and the complete fingerprint add `rayd_source_kind` and
`rayd_source_manifest_sha256`; they do not record a machine-specific absolute
path. Local all-architecture recompilation is intentionally deferred to the
updated CUDA/nightly/release GitHub Actions. Historical Munich/ADR-032 evidence
remains bound to RayD `474c122` and is not rewritten by this packaging change.
The isolated package-resolution and current-platform SM120 wheel/PE smoke record
is `docs/dev/audit/phase13-rayd-package-source-discovery-acceptance.json`.
The package-source SM120 configure resolved successfully, but its native build
did not produce a wheel within the bounded local run and is not claimed as
accepted; that publication build remains a GitHub Actions gate.

### 2026-07-23: RayD 0.7.0 Stage-I dependency candidate

The Stage-I candidate lock now identifies the formal lightweight tag `v0.7.0`
at `49c58c4cb8212f6babb920cc88fb937509826cc5` and
`rayd-torch==0.7.0`. A clean tag archive reproduces integration API 6,
identity `rayd.torch.integration`, header SHA-256
`57f83ea460e376166fd5ee22a8243a7c1576a290e1de99c0cbe8e86e93392e14`,
and source-manifest SHA-256
`e2eb1a7577f906b3ab52e6345b039837228771c8f1582c9f821d0f2bb07d41b4`.

ADR-035 accepts this immutable dependency baseline while recording that it is
not a packaging-only delta. RayD owns `TraceBackend::Auto`: OptiX is preferred,
and RayD may select its full-result pure-CUDA native implementation when OptiX
is unavailable. This selection is not a Torch, CPU, Dr.Jit, retry,
reduced-result, or second-owner fallback. An operation unsupported by the
selected backend must fail its typed capability validation before that
operation launches or exposes output.

Phase 0A accepts the final dependency, header, source-manifest, workflow-pin,
product-identity, and compact-owner static baseline. It does not inherit
OptiX evidence for pure CUDA or claim complete numerical certification. The
Phase 2 and Phase 3 large-module checkpoints retain the separate capability,
numerical, AD, launch/resource, performance, wheel, and fingerprint gates for
both RayD-owned native trace paths.
Historical Plan 13 P-to-E and E-to-M comparative reports remain archived and
are not reactivated by this dependency review.

## Stage-I module and public-API standardization

This pass is structural only: no physics, numerical order, launch
configuration, reduction order, or result schema changed.

### Package root no longer re-exports the Core world model

`witwin.channel` exports only what Channel owns:

```python
from witwin.channel import (
    Complex3State, JonesState, build_info, capabilities,
    pipeline_cache_key, runtime_diagnostics,
)
```

`Scene`, `SceneSnapshot`, `Structure`, `PhysicalMaterial`, `MaterialLayer`,
`PhaseScreen`, `SurfaceRoughness`, `AntennaPattern`, `AntennaState`, and
`ReceiverGrid` are removed from the Channel root. Import them from
`witwin.core`, which is their owner. Each world type now has exactly one import
path.

```python
# before
from witwin.channel import Scene, Structure, PhysicalMaterial
# after
from witwin.core import Scene, Structure, PhysicalMaterial
```

Core also drops its `Material` alias; `PhysicalMaterial` is the only public
name. `Structure` and `MaterialAssignment` moved to `witwin.core.structure`,
though both stay exported from `witwin.core`.

### `witwin.channel.core` is dissolved

The package collided with `witwin.core` and was not a domain owner. Every
module moved to its real owner:

| Before | After |
|---|---|
| `core.kernels.extension` | deleted; `build_info` comes from `deployment` |
| `core.kernels.metadata` | `runtime.kernel_metadata` |
| `core.memory_budget` | `runtime.memory_budget` |
| `core.edge_policy` | `scene.edge_policy` |
| `core.edge_selection` | `scene.edge_selection` |
| `core.ad_geometry` | `scene.ad_geometry` |
| `core.antenna` | `scene.antenna` |
| `core.receiver_geometry` | `scene.receiver_geometry` |
| `core.diffraction_geometry` | `propagation.geometry.edge_state` |
| `core.components` | `components` |
| `core.field_state` | `field_state` |
| `core.tensor_math` | `tensor_math` |

`core.kernels.extension` described itself as a compatibility facade and was the
package root's only route to `build_info`. Deleting it repays the
`boundary-001` import-graph debt; `runtime` is now the sole owner of extension
loading and `deployment` the sole public reporting facade.

### `witwin.channel.physics` is dissolved

`physics.conventions` became the package-root `witwin.channel.constants`, which
also owns the phase convention that solver metadata and the consumer contract
quote. `physics.oracle` was a compatibility facade that re-exported private
names and rewrote `__module__` to impersonate itself; it is deleted. The NumPy
CPU reference oracle moved to `tests/reference/em_oracle.py`, where CLAUDE.md
requires reference implementations to live, and no longer ships in the wheel.

Note: Core's `VACUUM_PERMITTIVITY` and Channel's derived `EPS0` still differ in
the ninth significant digit. That is a numerical question and needs its own
ADR; this pass did not touch either value.

### Propagation consumer contract

Still contract version 1; the changes are removals of things that were never
implemented plus additions that are purely descriptive.

- `capabilities()` is exported. Query supported components, responses, topology
  modes, and AD modes before building a request.
- `PropagationComponent`, `PropagationResponse`, `PropagationTopologyMode`, and
  `PropagationAdMode` are `Literal` aliases, with `COMPONENTS`, `RESPONSES`,
  `TOPOLOGY_MODES`, `AD_MODES`, and `MAX_DEPTH` exported alongside them.
- `PropagationRequest` and `FixedTopologyRequest` validate their structure at
  construction instead of only inside `evaluate` / `reevaluate`.
- `PropagationGeometry.interaction_position_m` and `.interaction_normal` are
  removed. They were column 0 of `interaction_positions_m` and
  `interaction_normals`; slice those instead.
- `frequency_offsets_hz` and `PropagationConvention.frequency_offset_law` are
  removed. They were declared but always rejected. Frequency offsets will
  arrive with a `CONTRACT_VERSION` bump.
- `PropagationCapabilities` drops `supports_frequency_offsets` and gains
  `components_for`, `ad_modes_for`, and `ad_modes_for_component`.

### Capability manifests

`witwin.channel.capabilities()` stays the solver-level manifest and keeps
`scattering` in its component list. It now embeds the consumer contract record
under `propagation_consumer`, generated from
`witwin.channel.propagation.consumer.capabilities()` rather than restated, so
the two cannot drift.
