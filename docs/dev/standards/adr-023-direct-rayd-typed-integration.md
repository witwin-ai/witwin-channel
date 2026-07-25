# ADR-023: Direct RayD typed integration and RayDN retirement

- **Status:** Accepted (2026-07-19); typed switch, legacy retirement, and stable
  integration naming implemented; validated package-source discovery amendment
  accepted 2026-07-22; RayD native trace backend selection amendment accepted
  by ADR-035 on 2026-07-23; new-lock multi-architecture publication delegated
  to the Stage-I release checkpoint
- **Date:** 2026-07-19
- **Kind:** Native integration boundary and lifecycle ownership. This ADR does
  not change physics, numerical order, fusion, launch configuration, solver
  behavior, or the public Python API.
- **Related:** [Plan 13](../plans/13-direct-rayd-integration-and-rf-runtime-ownership-plan.md),
  ADR-001 (Python/native dispatch), ADR-003
  (public/internal API), ADR-004 (numerical duplication), ADR-006 (developer
  override), ADR-007 (propagation ownership), ADR-009 (native fusion
  ownership), ADR-012/013 (coupled diffraction), and ADR-020/021/022 (current
  transmission, scattering, and AD contracts), ADR-032 (compact cardinality),
  and ADR-035 (RayD native trace backend selection).

## Context

Channel already builds RayD in the same CMake graph and links RayD's
native Torch target into `_channel`. The production path nevertheless
crosses a historical compatibility boundary made of raw `int64_t` scene
handles, duplicated `extern "C"` array signatures, function-pointer getters,
capacity-managed outputs, and `RayDN/raydn`-named facades. This makes one native
backend appear to be two, obscures ownership and lifetime, and gives typed Torch
tensors less protection than they have at the Python boundary.

The desired direct integration is not a Python import of `rayd.torch`, a second
extension, a second dispatcher, or a second scene registry. It is a source-level
C++ API used inside the one `_channel` extension. RayD remains the owner
of generic scene, acceleration structures, native tracing, intersection,
visibility, and generic path-geometry primitives. Channel remains the owner of
RF contracts, solver policy, topology, reductions, and Channel-fused
operations.

The owner distinction matters for naming. Coupled reflection-diffraction and
double-diffraction geometry is a Channel-composed operation that calls RayD
EPC/visibility primitives. It is not a generic RayD operation merely because a
RayD primitive participates in it.

## Decision

### 1. One typed source-level integration API

RayD provides a versioned C++ integration API in `namespace rayd::torch`.
Its numeric API version is 6; the source header and identity use stable,
capability-neutral names.
The public integration header exposes typed resource, config, operation, and
result contracts. Its version and header identity participate in Channel's
locked RayD dependency and complete build fingerprint.

The API uses:

- `at::Tensor` for required tensor inputs and outputs;
- `std::optional<at::Tensor>` for semantically optional tensors, without dummy
  tensors or sentinel pointers;
- typed configuration values instead of unstructured positional arrays;
- named result structs such as `IntersectResult`, `VisibilityResult`,
  `ReflectionTraceResult`, `ReflectionEpcResult`, and
  `DiffractionPathResult`, rather than raw tuples or caller-owned output arrays;
- symmetric typed contracts for any operation family that has
  primal/JVP/VJP/backward entries.

Channel's C++ Torch bindings include `rayd/torch/integration.h` and call these functions
directly. There is no copied integration signature, function-pointer getter,
raw output-capacity protocol, dynamic symbol lookup, or Channel compatibility
adapter between the binding and this API. Python domain kernel facades and the
`_channel` pybind registry remain necessary typed validation, dispatch,
error-translation, and result-assembly boundaries; they do not reimplement
geometry or physics.

New typed entries reuse the existing RayD implementation and kernel launches.
They must not copy a kernel or create a second numerical implementation. During
the bounded migration window, old and new RayD entries are two interfaces to
the same implementation, not two production owners.

### 2. RAII scene resource and typed holder

RayD owns a RAII `SceneResource` (or equivalently named public resource type)
that owns the scene, mesh resources, acceleration structures, OptiX state, and
their destruction. Channel exposes it to Python as `RayDSceneResource` through
a typed pybind holder.

The lifecycle contract is:

- a scene is destroyed exactly once when the final owning holder is released;
- operations borrow a validated live resource and cannot reconstruct one from
  an integer or pointer value;
- no bare C++ pointer is encoded in `int64_t`, and no Python-visible handle id,
  dummy handle, global Channel registry, or cross-DSO duplicate registry is
  permitted;
- creation failure unwinds all already-created native resources; operation
  failure does not invalidate an otherwise-live scene; destruction is safe
  after exceptions and must not throw across the Python boundary;
- multiple simultaneous scenes, repeated create/destroy, holder copies/moves,
  and teardown in non-creation order preserve independent ownership;
- use after final release cannot become a stale-handle lookup or UAF; Python
  object lifetime and the typed holder keep every in-flight call's resource
  alive.

The Channel switch is blocked until direct RayD tests cover normal teardown,
construction failure, exception teardown, multiple scenes, stress/repetition,
and leak/UAF detection appropriate to the build.

### 3. Tensor, device, stream, ABI, and error contracts

Every typed entry validates the complete contract at the RayD host boundary before
launch or empty-result return:

- required rank, shape, scalar type, layout/contiguity policy, and optional
  tensor presence;
- CUDA residency and the operation's same-device relationships;
- a live scene on the matching CUDA device;
- integration API version, libtorch/CUDA ABI/build identity, supported SM, and
  every capability required by the operation.

An entry launches on the caller's active CUDA stream for the validated device.
It must not silently switch to the default stream, add device-wide
synchronization, add a host wait, or move data between host/device or devices.
Any necessary CUDA device guard is scoped to the call and restores caller
state. Outputs and temporary storage remain resident on that device and obey
the existing stream lifetime rules.

Empty inputs have an explicit per-operation typed result: correctly shaped
empty tensors on the input device with the established dtype, stride/alias,
and gradient semantics. Empty handling must match the old entry exactly, must
not manufacture a successful empty result for an invalid contract, and must
not hide a missing capability, ABI mismatch, stale resource, or device error.

Invalid shape/dtype/device/resource/ABI state and native CUDA/OptiX failures fail
loudly before partial computation is reported as success. RayD raises a typed
C++/c10 exception with operation and contract context; `_channel`
translates it once at the pybind boundary. Channel must not catch it to return
zeros, empty success, detached outputs, a reduced algorithm, or any Torch/CPU
fallback. No error-code tensor or partially filled output array is a valid typed
result.

### 4. Single extension and build/package boundary

`_channel` remains the only production Python extension and the only
Torch-facing dispatcher. RayD is source-linked as a locked native target in the
same build graph. Normal builds and wheels must not:

- build, package, import, or dynamically load a RayD Python module;
- ship a second RayD Torch extension or an undeclared RayD DSO;
- scan a conda prefix, site-packages tree, global CMake registry, or other
  uncontrolled location for RayD;
- create an independent scene registry or load a second copy of the runtime.

An explicit `RAYD_SOURCE_DIR` remains the highest-priority build input and is
validated as a Git checkout. When it is absent, Channel may locate only the
selected Python interpreter's unique `rayd-torch` distribution and read its
passive `rayd/torch/_source/rayd-source.json` resource without importing
`rayd.torch`. Before source-linking, Channel validates the lock-pinned
distribution/version, repository, commit, stable API/identity/header,
distribution RECORD ownership, and every file in the complete source manifest.
Missing, duplicate, dirty, escaped, or mutated package sources fail loudly.
This is build-source discovery, not a second runtime backend or dispatcher.

The packaged RayD commit, integration-header hash/version, compiler/CUDA/Torch
ABI, supported SM set, and relevant build flags are part of Channel's complete
fingerprint. Source provenance and the lock-pinned full-source manifest digest
also participate, while machine-specific absolute paths do not. A developer
override remains subject to ADR-006 and must validate that full fingerprint;
it is not a fallback.

### 5. Ownership-aware names and zero Channel shims

After Channel switches to the typed boundary, live Channel production source,
build rules, CI
manifests, tests, and current operational documentation use no historical
`RayDN/raydn`
identity and expose no alias, re-export, feature-flag dual path, capability
fallback, or compatibility shim for it.

Names follow operation ownership:

- generic RayD-owned primitives use `rayd_*` and typed RayD resource names;
- generic intersection/visibility operations must not retain a solver-specific
  `bdpt_*` prefix;
- Channel-owned composed coupled RD/DD geometry uses neutral owner names such
  as `coupled_*`; it must **not** receive a blanket `rayd_*` prefix merely
  because it calls a RayD EPC, intersection, or visibility primitive;
- Monte Carlo edge discovery/sample-tape operations use their Channel domain
  names and do not masquerade as RayD or generic BDPT primitives.

The same ownership rule applies to Python facade names, native symbols, result
types, tests, manifests, feature lists, and migration notes. Immutable audits
and accepted historical decision or migration records may retain historical
terms only through explicit archive-scoped allowances; they do not authorize a
live compatibility name.

### 6. Realized legacy retirement and stable naming

Plan 13 Phase 11 completed the repository/consumer reachability audit and
removed RayD's legacy `extern "C"` integration entries. Channel neither compiles
nor calls those entries and exposes no renamed wrapper, compatibility alias, or
dual path. The audit covered declarations, definitions, link references,
tests, examples, packages, downstream build files, and current documentation.

The live typed boundary now uses `rayd/torch/integration.h` and identity
`rayd.torch.integration`. Numeric API version 6 remains a separately validated
contract value; it is not encoded in a WIP filename, target, or identity. The
final Channel lock advances through later accepted RayD additions without
reintroducing the retired surface.

### 7. RayD-owned native trace selection

ADR-035 amends the original OptiX-only assumption without changing the typed
boundary or creating another owner. RayD 0.7.0 `TraceBackend::Auto` may select
RayD-owned OptiX or RayD-owned pure-CUDA tracing during scene construction.
OptiX remains the preferred performance path. Missing OptiX alone is no longer
a Channel capability failure, but an operation unsupported by the selected
RayD backend must fail typed capability validation before that operation
launches or exposes output. Channel never retries a failed operation on the
other backend and never substitutes Torch, CPU, Dr.Jit, reduced physics, or a
partial result.

Phase 0A freezes only dependency/static compatibility. Both native trace paths
retain separate numerical, AD, capability, launch/resource, and performance
evidence obligations at the Phase 2 and Phase 3 large-module checkpoints.

## Required migration and acceptance evidence

The interface migration is behavior-preserving. Before Channel can delete its
historical bridge and raw-handle plumbing, the legacy and typed entries must pass
exact lockstep on representative and boundary cases for scene operations,
intersection, visibility, reflection trace/EPC, diffraction export, and every
other moved generic geometry entry. The comparison freezes:

- values and exactness required by the existing baseline;
- row identity/order, tensor shape, dtype, device, stride, storage aliasing,
  and gradient state;
- exception class/category and failure timing at the public boundary;
- launch geometry/count, active stream, synchronization, host/device copies,
  temporary/tape lifetime, peak memory, compile flags, and numerical order.

RayD direct tests cover valid, empty, batched, non-contiguous (accepted or
rejected exactly per contract), shape, dtype, device mismatch, current-stream,
error propagation, multi-scene, construction/destruction, and exception
lifecycle cases. Channel runs targeted end-to-end coverage for Path,
Deterministic, Monte Carlo Basic, and BDPT plus the required `quick` and `cuda`
tiers. Missing native symbols/capabilities, ABI mismatches, and unsupported SMs
have negative fail-loud tests.

The Channel switch updates the locked, already-pushed RayD commit and header
hash, build fingerprint, native binding and contract-coverage manifests,
current-owner inventory and migration delta, duplication and launch/resource
ledgers, no-fallback tests, direct contracts, public API snapshot if affected,
feature list, and migration note. The immutable Phase-9 owner inventory is not
edited.

Packaging acceptance requires a clean locked checkout and wheel inspection
showing exactly one production extension (`_channel`), the declared
metadata, no RayD Python extension, and no undeclared runtime DSO. Compiler and
resource evidence must explain any output difference caused by the target/header
move. An unexplained numerical, codegen, resource, stream, lifecycle, or package
change stops the migration; tests, tolerances, budgets, manifests, and
allowlists are not weakened to accept it.

## Consequences

- RayD scene and generic geometry ownership becomes explicit at a typed,
  lifetime-safe C++ boundary without adding a production backend.
- Channel keeps its domain facades and composed/fused solver operations while
  deleting identity-only bridge code and raw-handle plumbing.
- The typed API addition and Channel switch can be reviewed and bisected separately:
  RayD first lands an exact dormant interface, then Channel pins that merged
  commit and switches atomically.
- Rollback is a lock-file/build-fingerprint change to the previous accepted
  RayD commit, never a runtime fallback or dual dispatch.
- This ADR does not authorize the transmission, diffraction-physics, or
  scattering owner moves. Those require ADR-024, ADR-025, and ADR-026
  respectively.
