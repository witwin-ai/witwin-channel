# ADR-024: Shared RF primitives and transmission runtime ownership

- **Status:** Accepted (2026-07-19)
- **Date:** 2026-07-19
- **Kind:** Move-only native numerical ownership and source-header boundary.
  This ADR does not authorize a physics, numerical-order, fusion, launch,
  synchronization, RNG, result-schema, or public Python API change.
- **Related:** [Plan 13](../plans/13-direct-rayd-integration-and-rf-runtime-ownership-plan.md),
  ADR-004 (numerical duplication), ADR-009 (native fusion ownership), ADR-020
  (transmission polarization), ADR-022 (fixed-topology AD), and ADR-023
  (direct typed RayD integration).

## Context

Channel owns the material, row, solver, and result contracts, while the
generic complex/medium/Fresnel/layer-stack/Jones device math is currently
stored in Channel-private headers. The resident layer-stack and complete-row
transmission operation families are likewise Channel numerical
implementations even though they are solver-neutral and must be shared by
Channel kernels after the direct RayD integration.

Moving only the host entries would create a reverse RayD-to-Channel private
include or duplicate device physics. Moving individual primal functions would
split their backward/JVP mirrors and make evaluation order ambiguous. The
accepted boundary therefore moves complete numerical families and their full
device-header dependency closure while retaining Channel's domain contracts.

## Decision

### 1. Public source layout and single owner

RayD becomes the unique source owner, after the Phase 6 Channel pin/switch, of:

- `shared/include/rayd/shared/rf/complex.cuh`;
- `shared/include/rayd/shared/rf/medium.cuh`;
- `shared/include/rayd/shared/rf/fresnel.cuh`;
- `shared/include/rayd/shared/rf/layer_stack.cuh`;
- `shared/include/rayd/shared/rf/field_transport.cuh`; and
- `backends/torch/include/rayd/torch/rf/field_transport_ad.cuh` for the AD
  closure that uses Torch/c10 complex types.

Channel kernels that continue to own reflection, coupled diffraction, BDPT,
or scattering policy include these RayD public headers. RayD never includes a
Channel-private header. Channel deletes the corresponding private numerical
headers in the same commit that pins and calls the already-pushed RayD owner.
There is no forwarding header, namespace alias, copied helper, or compatibility
shim.

The accepted helper decision is recorded in
`docs/dev/audit/phase13-shared-rf-helper-ownership-decision.json`. Of the 129
frozen helpers, 112 transfer under this ADR, 10 remain Channel boundary
adapters, and 7 scattering-table helpers transfer only after ADR-026.
`fold_output_cotangents` and `write_output_tangents` are numerical output-chain
AD helpers, not validation glue, and are therefore included in the RayD-owned
AD header. This resolves the Phase 0 header-level `split` decision without
duplicating their bodies.

### 2. Typed host integration

RayD exposes the transmission API through
`backends/torch/include/rayd/torch/rf/transmission.h`, included by
`rayd/torch/integration.h`. It uses named request/result structs containing
`at::Tensor` and `std::optional<at::Tensor>` values. It does not expose
`pybind11::object`, untyped sequences, dictionaries, raw native tuples, dummy
tensors, or a second dispatcher/extension.

All entries validate dtype, rank, contiguity, shape, device, CUDA availability,
supported SM, Torch ABI, and integration identity before partial computation.
They allocate and launch on the caller's active CUDA stream and translate CUDA
and C++ failures loudly. There is no CPU, Torch-expression, finite-difference,
legacy, or reduced-algorithm fallback.

### 3. Complete operation families

The following six `_channel_native` ABI names and Python facades remain stable,
but their numerical implementation owner transfers as complete families:

1. `em_layer_stack_eval`;
2. `em_layer_stack_backward`;
3. `em_layer_stack_jvp`;
4. `field_transmission_sequence`;
5. `field_transmission_sequence_backward`; and
6. `field_transmission_sequence_jvp`.

The layer-stack primal preserves its twelve ordered outputs: complex TE/TM
reflection and transmission components plus `R` and `T` powers. Backward keeps
the five defined gradient tensors and existing need-gating semantics. JVP
keeps the same twelve tangent outputs and optional-leaf behavior.

The transmission primal preserves its seven ordered outputs: field vector,
coefficient, path field, path gain, path length, delay, and direction.
Backward preserves optional output cotangents and optional gradients, including
the defined contract that interaction-position gradients are absent. JVP
preserves its six differentiable tangent outputs and does not add direction.
Fixed CSR topology, material IDs, valid masks, power, and polarization remain
fixed inputs. Existing empty-row, depth, `None`, shape, stride, dtype, device,
and error contracts remain exact.

### 4. Fusion, AD, and numerical contract

- Primal, backward/VJP, and JVP remain one launch each. Zero rows do not launch.
- Complete-row transmission remains fused; no per-layer intermediate tensor or
  persistent tape is introduced.
- Backward and JVP recompute the layer chain in their existing order.
- Shared-layer gradient atomics remain at the same sites, in the same traversal
  and request-gating order.
- Block size, grid formula, output layout, dtype, stride, device, storage,
  evaluation order, and floating-point expression order remain unchanged.
- These families use precise math. They do not inherit pure-wedge fast math or
  scattering's `--fmad=false` compilation policy.
- No launch, synchronization, memcpy, stream wait, resident tape, peak memory,
  register, occupancy, or reduction-order regression is accepted.

Any desired numerical or fusion change requires a separate ADR and evidence;
it cannot be hidden in this ownership move.

### 5. Channel-retained ownership

Channel continues to own:

- material models, Material ABI/CSR encoding, cache, validation, resources, and
  Python/native facades;
- topology pairs and winners, thin-sheet eligibility, and component-5 packing;
- field row contracts, autograd dispatch, result assembly, and solver metadata;
- the complete `bdpt_transmitted_light_subpath_state` primal/backward/JVP
  family and its 19-field state/PDF/event schema;
- BDPT MIS and RNG policy, MC Basic estimator policy, accumulations, and solver
  results.

The ten Channel-retained helper records are tensor loads/conversions, launch
arithmetic, allocation/zeroing, optional-gradient validation, and pointer
extraction. They may orchestrate native execution but do not reproduce RF
physics. If any later edit makes one numerical, it must move to its numerical
owner or receive an explicit duplication decision.

### 6. Dormant candidate and atomic switch

Each RayD implementation lands first as a dormant candidate. Before Channel
pins that pushed commit it is not a production owner and `_channel_native`
must not compile or call it. The corresponding Channel switch commit atomically:

1. pins the pushed RayD commit and integration-header SHA;
2. compiles/calls the typed RayD entry;
3. redirects all public-device-header consumers;
4. deletes the Channel numerical source; and
5. updates owner, binding, coverage, duplication, launch, and evidence records.

Rollback is a lock-pin rollback to the previous complete owner. Runtime flags,
capability fallbacks, two compiled owners, and temporary shims are forbidden.

### 7. Deferred fusion work

Batched penetration tracing and MC event-glue fusion alter launch/fusion
boundaries. They are outside this move-only decision and are now governed by
accepted ADR-027 with exact-equivalence and profiler evidence; they did not
block this migration.

## Consequences

RayD becomes the reusable source owner of solver-neutral RF device math and the
two selected Torch operation families. Channel retains its RF domain and
solver contracts without a reverse private dependency. The move requires two
RayD dormant commits and two Channel pin/switch commits, with exact/codegen/AD,
stream, resource, four-solver, no-fallback, and packaging evidence at each
switch.
