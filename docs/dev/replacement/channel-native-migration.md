# Channel Native migration and runtime-dependency boundary

## Current decision

`witwin.channel_native` is the native entrypoint for the capabilities it
advertises. Repository-owned production Python must not import DrJit, Mitsuba,
Sionna, a RayD Python dispatcher, or `witwin.channel`. The old Channel may remain in tests
and benchmarks as an offline correctness oracle; it must never be a production
fallback. The independent radar implementation is a separate product and is
not evidence for either Channel implementation.

The audited sibling roots (`core`, `genesis`, `maxwell`, `radar`, and `studio`)
contain no production Channel imports. This does not prove that external users,
deployed jobs, plugins, or private repositories have migrated.

The platform `core` package's `channel` and `all` extras now route to
`witwin-channel-native>=0.1,<0.2` (companion commit `9ee6655`) instead of the old
`witwin-channel` distribution. This establishes the repository-owned default
installation route; application-level canary/default-on state still requires
confirmation from each consumer owner.

### Plan 13 Phase 3: direct typed RayD integration

Channel Native now source-links RayD commit
`adf0ea2d1481f7548c5ef30c31b4adbaf831f963` into its single `_channel_native`
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

Channel Native now pins the pushed RayD candidate
`4cb400acbfcc2da7fda4110d1298d311816905f1`; the locked
`backends/torch/include/rayd/torch/integration_v2.h` SHA-256 is
`c8e162c55a0e5abe789e4f1b19cd6ab00ee4ef59d70244cfc55d58166aeb646b`.
RayD is the unique numerical source owner of the ADR-024 shared RF
complex/medium/Fresnel/layer-stack/Jones primal/dual closure and the complete
`em_layer_stack_eval/backward/jvp` family. Channel still owns all three stable
`_channel_native` names, the materials facade, Material ABI v3/CSR encoding,
validation, caches and resources.

All Channel native consumers include the versioned RayD public headers. The
former Channel-private numerical headers and `kernels/em_debug.cu` are removed
without forwarding aliases or a runtime fallback. Of the 129 frozen helper
records, 112 now have RayD as their active unique source owner, 10 remain
Channel boundary-only tensor/launch adapters, and 7 scattering-table helpers
remain Channel-owned pending Phase 10A activation under accepted ADR-026. The
live binding count remains 202; the
current owner split is RayD 20, layered Channel/RayD 2, and Channel Native 180.

### Plan 13 Phase 6B: complete-row transmission ownership

Channel Native now pins the pushed RayD candidate
`3988f0934fec7b521ee5190b0defc0883c84b9e6`; the integration v2 header
SHA-256 is
`6cb18f682e08cb0bb0853507e3b4b82a68e681bb1dad89dc8c36518705f74989`
and its identity is
`rayd.torch.integration.v2.20260719.rf-transmission-sequence`.

The complete `field_transmission_sequence/backward/jvp` family now dispatches
through the source-linked typed `rayd::torch` API. Channel retains the three
stable `_channel_native` names, field-row schemas, Python/autograd facades and
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
Channel Native 177. The frozen 129-helper partition remains 112/10/7 because
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
become a complete Channel native planning/selection operation while preserving
the ordered fractions `(0.02, 1/3, 2/3, 0.98)`, any-visible rule, and stable row
selection. The accepted contracts and stop conditions are recorded in
`docs/dev/audit/phase13-diffraction-family-matrix.json` and
`docs/dev/audit/phase13-diffraction-legacy-audit.json`.

### Plan 13 Phase 8A: pure-wedge diffraction ownership

Channel Native now pins pushed RayD commit
`11e72526cdddf669678975c8921a9d44c6504e20`. The locked integration v2 header
SHA-256 is
`7a2b68f459e7e981a23735271eff2844fe0483d119cf514d59d2032d11be5aef`,
with identity
`rayd.torch.integration.v2.20260719.rf-transmission-sequence.pure-wedge-diffraction`.

The complete `field_diffraction_wedge/backward/jvp` family now dispatches
through the source-linked typed `rayd::torch` API. Channel retains all three
stable `_channel_native` names, field/autograd facades, row contracts and
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

Channel Native now pins pushed RayD commit
`4577e744adfe8665f7817e3aff5e8e533ec896e7`. The typed scattering header and
integration-v2 header SHA-256 values are respectively
`66d75a20be16057f03cdfb79e3b9dcc85cacec79b555cd73b019259aa510262a`
and `9f95ad9e8e3b790d00f8e762a3e6a09252d46afb65bfc3aba7c42325836cb1fb`;
the RayD shared scattering-table header SHA-256 is
`38ea9be424640301a88a97bccca9ab4bc599191ecfb0b259881ef6a300c96e38`.

The complete table-evaluation AD, table-sampling, single-bounce ensemble, and
patch-integral families now dispatch through typed `rayd::torch` requests and
results. Channel keeps all eleven `_channel_native` names, Python facades,
resident-resource lifecycle, and solver policy. The five former dedicated
Channel numerical TUs and private table helper header are deleted; the retained
`scattering.cu` contains only `scattering_event_probabilities`. Remaining chain
consumers include RayD's public table header directly, while
`kirchhoff_table_ad.cu` gains no unused dependency.

The live binding count remains 202. The numerical-owner split is RayD 37,
layered Channel/RayD 2, and Channel Native 163. Table primal/sample/PDF and the
retained event-policy TU remain on their default CUDA flags; table AD, ensemble,
and patch owners retain `--fmad=false` in RayD. Launch count, current stream,
reduction order, atomics, resident tensors, and public API remain unchanged.
At the Phase 10A cut, Channel still owned the six fused chain contracts pending Phase 10B.
Detailed Phase 10A activation, codegen/resource, deletion, and direct-test evidence is recorded in
`docs/dev/audit/phase13-scattering-phase10a-evidence.json`.

### Plan 13 Phase 10B: fused scattering-chain activation

Channel Native now pins pushed RayD commit
`768b96e42a95f70c32d55f98a72000085317e288`. The typed scattering,
integration-v2, and shared scattering-table header SHA-256 values are
respectively
`ac95c418860d109aeaa96623131592e4df8887992e5fc25ecab71b4ddbf1f55b`,
`0608bfbaf022379bc03442f9baa777ec05cfe3f6ab9b964e2385ec12a7b6c654`,
and `38ea9be424640301a88a97bccca9ab4bc599191ecfb0b259881ef6a300c96e38`.

The complete ensemble-chain and realization-chain primal/backward/JVP families
now dispatch through typed `rayd::torch` requests/results. Channel keeps all six
`_channel_native` names and typed Python/autograd facades, while the four local
chain CUDA TUs are deleted. `scattering_event_probabilities`, table/phase-screen
lifecycle, topology/C1-C2 packing, RNG/MIS/event policy, solver accumulation,
and results remain Channel owners. The geometry AD truth is unchanged:
ensemble is JVP-only with loud VJP rejection; realization supports VJP/JVP.

The live binding count remains 202. The numerical-owner split is RayD 43,
layered Channel/RayD 2, and Channel Native 157. All four RayD chain TUs retain
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

`witwin.channel_native.montecarlo.bdpt.Config` gains one field:

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

- `witwin.channel_native.deterministic.Config` gains four fields:
  `scattering_coherent: bool = False`, `scattering_chain_max_depth: int = 0`,
  `scattering_chain_samples_per_m2: float = 2.0`,
  `scattering_chain_max_rows: int = 256`.
- `witwin.channel_native.path.Config` gains three fields:
  `scattering_chain_max_depth: int = 0`,
  `scattering_chain_samples_per_m2: float = 2.0`,
  `scattering_chain_max_rows: int = 256`.
- `witwin.channel_native.montecarlo.bdpt.Config` gains one field:
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

### ADR-029: explicit device-resident capacities (2026-07-20)

`witwin.channel_native.path.Config` and
`witwin.channel_native.deterministic.Config` each gain two stable fields:

- `path_capacity_per_pair: int | None = None`
- `diffraction_state_capacity: int | None = None`

This first activation step adds construction-time contracts only. `None`
remains constructible so the producer/consumer switch can be staged, but the
completed ADR-029 solver path will reject it before enumeration whenever the
corresponding capacity is required. A non-`None` value must have exact Python
type `int` and be non-negative; floats, NaN, and Boolean values are rejected,
while zero explicitly represents empty capacity. When
`max_paths_scope="per_pair"`, `max_paths` cannot exceed
`path_capacity_per_pair`. Deterministic's global `max_paths` comparison is
intentionally deferred until solve knows the endpoint-pair count.

These capacities are storage and launch-planning bounds, not implicit path
selection or truncation policy. No compatibility alias, generation-suffixed
name, solver dispatch, result-shape change, or native ABI change is introduced
by this configuration-only step. The public export count remains 37; only the
two Config `contract_sha256` values change.

The dormant internal `propagation.models.CapacityPathLayout` contract records
host pair/capacity metadata and exact CUDA `valid`, `num_paths`, and `overflow`
tensor metadata without reading or recomputing device values. It is exported
only from the internal `propagation.models` package; it is not a root public
export, solver result, or active solver boundary.
