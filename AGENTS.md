# Channel Native Architecture Guardrails

ADR-033 accepts the breaking replacement product identity `witwin.channel`,
the `witwin-channel` distribution, and the single `_channel` extension. The
checkout directory may retain its current name, but installed/runtime/build
identifiers must not retain the predecessor suffix or add a compatibility
alias. During the bounded migration, follow ADR-033 whenever an older name in
this file conflicts with that accepted target.

These repository instructions apply to every file under `channel_native/` and
take precedence over the monorepo-level agent guide. Keep `AGENTS.md` and
`CLAUDE.md` identical. Architecture changes must update both files in the same
commit.

## Non-negotiable compute policy

`witwin.channel_native` has exactly one production compute backend: the
compiled native CUDA/RayD extension.

- Every production hot path must execute in a native CUDA kernel. This includes
  geometry evaluation, RF/material/scattering physics, sampling, path
  evaluation, reductions, and production JVP/VJP/backward math.
- Torch is the tensor-facing API and orchestration layer, not a production
  numerical backend. A Torch CUDA expression is still a Torch implementation
  and is forbidden for hot-path physics or geometry.
- CPU production computation and CPU fallback are forbidden. Missing CUDA,
  RayD capability, a native symbol, a supported SM, or an ABI-compatible
  extension must fail loudly before partial computation.
- Never recover from a native failure by using Torch operations, NumPy, Python
  loops, CPU code, finite differences, legacy RayD/DrJit dispatch, a reduced
  algorithm, zero tensors, empty success results, or detached gradients.
- Production AD must call registered native forward/JVP/VJP/backward companions.
  Torch autograd may dispatch those companions but may not reconstruct the
  numerical operation. Finite-difference production derivatives are forbidden.
- Do not recompute geometry already owned by RayD in Python or Torch.
- Device data must remain resident through the compute pipeline except at the
  sole ADR-032-accepted new compact-cardinality allocation boundary. That owner
  may copy
  only audited integer count metadata to the host and explicitly synchronize
  the caller's current stream to allocate exact `O(K)` output. It may not run
  CPU/Torch physics or numerical selection, hide the transfer behind allocation
  or Boolean indexing, or become a fallback. Other hot-path `.cpu()`, `.numpy()`,
  `.tolist()`, scalar extraction, host iteration, implicit synchronization, or
  avoidable host/device copies remain forbidden. This is not a claim that the
  whole solve has only one D2H/synchronization: pre-existing observed boundaries
  remain measurable optimization debt and require named owners plus E2E,
  memory, throughput, and exactness evidence before any change.
- Under ADR-032, production Path and Deterministic result shapes and
  `max_num_paths` represent actual compact rows, not provisioned storage.
  `path_capacity_per_pair`, `diffraction_state_capacity`, capacity-shaped
  public Path/PathTable results, and ADR-031 `Qr` are not production public API
  or solver requirements. The measured depth-3 Munich reflection boundary may
  issue at most six 4-byte count D2H copies, 24 bytes total, and must report the
  copy/synchronization time.
- Accepted genuinely fixed-capacity operations retain one runtime
  `CapacityFailureState`: a contiguous CUDA `int32[1]` bitmask initialized
  asynchronously on the caller's current stream. Every participant receives
  and retains that same typed object/storage, atomically ORs its owned failure
  bit, publishes only inert outputs after failure, and never traps or returns a
  partial result. Compact output must likewise be all-or-nothing: exact complete
  `K` rows in stable order or no usable result. Capacity is never a silent
  truncation policy.
- `capacity_failure_terminal_check` is the unique runtime-owned terminal
  observer. It consumes that typed state once after all result sanitizers,
  launches on the caller's current CUDA stream, preserves the bitmask, and
  device-fails only when a bit is set. It must never read the state on the host,
  synchronize, allocate a result, sanitize payload, or gain an intermediate or
  duplicate owner. A transaction that uses it installs exactly one call after
  all result sanitizers; dormant experiments may not add a production caller.

Python and Torch may perform non-numerical boundary work: API validation,
typed-contract construction, dispatch, orchestration, row selection, structural
packing, metadata, and result assembly. CPU/Torch reference implementations are
allowed only under `tests/` as independent oracles; production packages must not
import or dispatch to them. Offline/compile-time table construction and the
plan-sanctioned Monte Carlo event glue are not fallback backends and must not
grow into production hot-path physics.

There is no compatibility exception to this policy. If a requested feature
cannot obey it, stop and require an explicit architecture decision and accepted
ADR before implementation. Do not hide an exception behind a configuration
flag, capability probe, import fallback, or compatibility shim.

## Architecture and ownership

Organize code by RF domain capability, with a single owner for each operation:

- `runtime`: packaged extension loading, ABI/build identity, symbol validation,
  Torch compatibility isolation, native buffers, and AD dispatch contracts.
- `scene`: scene lifecycle, compilation, immutable native resources, and RayD
  handles.
- `materials`: material contracts and native material evaluation facades.
- `scattering`: scattering models, resident tables, phase screens, and their
  native kernel facades.
- `propagation.topology`: discrete path rows, IDs, winners, and interaction
  sequences.
- `propagation.geometry`: continuous positions, lengths, delays, directions,
  and normals.
- `propagation.fields`: RF field evaluation and native derivative companions.
- `propagation.enumerated`: shared deterministic path evaluation for Path and
  Deterministic solvers.
- `path`, `deterministic`, `montecarlo.basic`, and `montecarlo.bdpt`: thin
  solver-owned configuration, orchestration, accumulation, result, and metadata
  layers. Solvers must never import another solver.

The only enumerated/Monte Carlo exception is ADR-008: `montecarlo.bdpt.pipeline`
may call the public `evaluate_enumerated_paths` entry read-only as an opaque
discrete-path oracle. It must not import `propagation.enumerated.*` internals,
mutate the result, or add BDPT policy to the enumerated engine.
`montecarlo.basic` has no enumerated dependency.

Internal enumerated propagation uses the typed, zero-copy contracts
`PathTopology`, `PathGeometry`, `PathFields`, and `EvaluatedPaths`. Preserve row
identity, row order, tensor object/storage aliasing, stride, dtype, device, and
gradient state. Do not reintroduce `TopologyBatch`, `core.path_topology`,
`core.kernels.ops`, raw native tuples outside domain kernel facades, or
compatibility re-export layers.

The package root and the four solver entry points are the stable public API.
Internal modules are not compatibility promises. Do not preserve deleted APIs
or add backward-compatibility shims unless a new accepted design explicitly
requires them.

## Native boundary and fusion

- `_channel_native` is the only production Python extension. It source-links
  the locked RayD target and calls the versioned `rayd::torch` typed C++ API
  directly; do not build/import a RayD Python extension, add a second
  dispatcher/registry, or route through copied `extern "C"` signatures,
  function-pointer getters, compatibility shims, or dynamic symbol lookup.
- RayD build-source discovery is explicit and fail-loud. A non-empty
  `RAYD_SOURCE_DIR` is authoritative and retains Git commit/remote/dirty/ABI
  validation; an invalid explicit path never falls back. Without it, CMake may
  use only the selected Python interpreter's unique `rayd-torch` distribution,
  locate its passive `rayd/torch/_source/rayd-source.json`, and validate the
  lock-pinned commit, repository, API/identity/header, RECORD ownership, and
  complete per-file source-manifest digest before `add_subdirectory`. Never
  import `rayd.torch`, scan `CONDA_PREFIX`/site-packages/CMake registries, trust
  self-reported package identity without the full manifest, or record an
  absolute source path in the build fingerprint.
- The stable public typed boundary is `rayd/torch/integration.h` with identity
  `rayd.torch.integration`. Validate its numeric API version independently; do
  not encode version or capability growth in a WIP filename, target, identity,
  forwarding header, or compatibility alias.
- RayD scene ownership crosses the boundary as an RAII `SceneResource` held by
  a typed `RayDSceneResource` holder. Never encode a native pointer as an
  integer handle or add dummy/stale-handle plumbing. Typed operations use
  `at::Tensor`, `std::optional<at::Tensor>`, named result structs, the caller's
  active CUDA stream, and fail-loud device/ABI/exception contracts.
- Generic RayD-owned primitives use `rayd_*` names. Channel-owned composed
  coupled RD/DD geometry uses neutral `coupled_*` owner names even when it
  invokes RayD primitives; do not blanket-rename composed operations to
  `rayd_*` or retain historical `RayDN/raydn` aliases.
- Python domain `kernels/` packages are thin facades: validate contracts,
  request a required symbol through `runtime`, dispatch the native operation,
  and convert its result to a named typed contract.
- C++ Torch bridges validate tensor/shape/dtype/device/ABI state and launch
  kernels. They must not contain a second host implementation of RF physics.
- Every native ABI symbol has one Python owner and must appear in
  `ci/native-binding-manifest.json` with direct contract coverage. A live symbol
  must have at least one production end-to-end caller. A retained dormant
  ADR-029/030 experiment instead requires an explicit dormant owner kind,
  direct execution coverage, and a caller-zero static gate; it must not appear
  in production E2E coverage, capabilities, defaults, or the public API.
- Native ownership follows ABI operation, fusion/launch contract, tape lifetime,
  device primitive, and numerical order—not the Python directory layout.
- Under ADR-025 and the completed Phase 8A atomic pin/switch/delete, RayD is
  the sole numerical owner of pure-wedge diffraction primal/backward/JVP;
  Channel retains only its `_channel_native` ABI and typed field/autograd
  facades. MC Sionna and coupled RD/DD diffraction stay complete Channel
  owners; do not extract a UTD sub-launch or spread the pure-wedge fast-math
  flag into their precise-math translation units.
- Under ADR-026 and the completed Phase 10A/10B atomic pin/switch/delete, RayD
  is the sole numerical owner of all 17 generic resident scattering runtime
  contracts and the seven shared scattering-table helpers. Channel retains
  their `_channel_native` ABI and typed facades. RayD only consumes
  caller-owned resident tensors;
  Channel retains table/phase-screen lifecycle, `scattering_event_probabilities`,
  topology/packing, RNG/MIS/event policy, accumulation, and results. Preserve
  per-TU flags and the as-built chain AD split: ensemble geometry is JVP-only,
  while realization geometry supports VJP and JVP.
- Under ADR-027, RayD is the sole numerical owner of solver-neutral batched
  straight-segment penetration geometry and its forward-tape/VJP/JVP family.
  Callers must explicitly choose `EnumeratedFullDistance` (full endpoint,
  strict hit test, L2 restart) or `MonteCarloTargetInset` (target inset,
  inclusive hit test, L-infinity restart). Results use fixed `[N, D]` resident
  hit/tape storage and one batched traversal with a mandatory `D + 1` probe.
  A zero-traversal all-inactive request requires explicit host-known
  `input_active_any=false` and a same-stream device-mask consistency check; do
  not read the mask to the host. Active degenerate/full-distance and zero-inset
  rows complete with `reached_target=true`, while input-inactive rows remain
  false.
  Overflow joins the solve-owned device failure transaction, makes the entire
  result inert, and fails asynchronously without a host count read. Channel
  retains material/geometry-mode encoding, thin-sheet eligibility, topology,
  and the MC incident-TE/TM wall-product estimator. Do not reintroduce a Python
  depth march, Torch geometry/estimator physics, or a per-transmitter trace.
- The live Channel-owned `mc_transmission_wall_product` primal/VJP/JVP
  family consumes fixed `[pair, hit_capacity]` penetration storage and the
  exact solve failure state. It checks canonical validity before every payload,
  has no AD depth cap or hidden contiguous copy, multiplies walls in ascending
  slot order, and reduces shared layer/frequency VJPs with fixed owners in
  ascending pair/slot order. The completed ADR-027 Phase M atomic switch makes
  it the Monte Carlo Basic `MonteCarloTargetInset` production estimator; the
  prior scalar/per-transmitter route is deleted and must not be restored.
- The superseded ADR-029 deterministic PathTable capacity export remains a
  caller-free experiment. It consumes the
  exact shared `CapacityFailureState` from its layout and preserves pair-major
  `P*C` rows. Native primal checks failure/overflow/valid before payload or ID
  reads; derivative companions consume only canonical output validity. Phase
  export remains non-differentiable, while the existing eleven continuous
  evaluated-path inputs retain native VJP/JVP.
- Under ADR-030, RayD's typed `SourceLane` layout and Channel's
  `deterministic_diffraction_pair_reduce` primal/VJP/JVP remain dormant and
  caller-free. Their experimental row is
  `((tx * rx_count + rx) * diffraction_state_capacity) + state`. One
  warp owns one endpoint pair; lanes may load consecutive states in parallel,
  but lane 0 must add all six float32 components in ascending state order with
  frozen non-contracted power evaluation. Floating-point atomics/tree
  reductions, pair splitting across chunks, and a second full lane-field
  workspace are forbidden in direct tests. The reducer publishes only inert
  output after failure. It cannot repair ADR-029's `O(P*M)` amplification and
  must not replace or delete the working compact production route without a new
  accepted ADR and complete E2E/resource evidence.
- Phase-screen mode resolution and scene-static realization resources are
  CompiledScene-owned and lazy. The first phase-screen consumer atomically
  caches immutable resident heights, structure/material ids, face ranges, UV
  tensors, scale, and RMS-slope state; endpoint/frequency/config-dependent
  subdivision and visibility remain solve-plan work. Unrelated compile and
  non-scattering solves must not allocate or validate these resources. Here
  "atomically" means publish-after-success cache replacement, not a
  multi-thread synchronization guarantee. This Plan-13 resource construction
  only caches the same scene-static UV/area/scale/slope work formerly performed
  by each consumer, preserving its exception and numerical order; it is not a
  Torch production-physics backend or fallback. Moving that retained static
  construction across the native boundary requires a separate accepted ADR.
  Its cache key may record `id(RayDSceneResource)` solely as Python wrapper
  identity while CompiledScene owns that wrapper; this is never a native
  pointer, scene handle, or ABI argument.
- `deterministic_reflection_candidate_capacity_block` is a dormant ADR-029
  post-EPC reflection producer for both order-1 and multibounce rows. Its
  internal `candidate_capacity` comes from the host-known theoretical EPC batch
  row count (or an equivalent explicit upper bound), never from a CUDA-selected
  count or public `path_capacity_per_pair`. It preserves visible input order,
  checks validity before reading any EPC/material payload, and fails
  asynchronously with a completely inert block on overflow. The existing live
  compact operations are production authoritative under ADR-032; no capacity
  switch is pending.
- `enumerated_canonical_capacity_select` is a caller-free ADR-029 experimental
  discrete selector. It reproduces the live stable topology order, canonical
  event/object deduplication, shortest-path winner, and global/per-pair
  `max_paths` policy into a candidate-capacity compact prefix. It has no public
  `path_capacity_per_pair` input or overflow output; pair-capacity enforcement
  belongs only to later result export/packing. The selector shares the solve
  failure state, has no AD companion, must remain before scattering append,
  and must not be reached by production or replaced by early pair-slot padding
  before deterministic accumulation.
- `evaluated_paths_canonical_capacity_gather` is a caller-free ADR-029
  continuous gather immediately after canonical selection. It produces a new
  sanitized compact-prefix `CanonicalEvaluatedPaths` at candidate capacity,
  keeps selector validity/counts device-resident, and validates compact-prefix,
  source-unique, endpoint-pair, and count contracts before any payload read.
  Its native VJP/JVP cover all eleven continuous evaluated-path fields; invalid
  or failed rows and derivatives are exact inert values. It is distinct from
  the experimental public pair-major capacity pack and remains dormant.
- `path_result_capacity_pack` is the caller-free Path-owned ADR-029 terminal
  storage packer. It consumes pair-major `CapacityEvaluatedPaths`, inherits the
  exact shared `CapacityFailureState`, and produces the public base tensor
  layout at the configured capacity without Ragged conversion, Boolean
  compaction, a device count read, or an intermediate trap. Shared failure or
  upstream overflow makes every output and device count inert. Its native
  backward and JVP companions differentiate only fixed-valid continuous rows;
  invalid rows and failed results contribute exact zero.
- Do not split a fused native operation merely to mirror Python modules. A
  refactor must not add kernel launches, synchronizations, materialized
  intermediates, persistent tape, host/device transfers, or reduction-order
  changes.
- Keep primal/JVP/VJP numerical duplicates in lockstep where the duplication
  ledger requires it. Deduplicate only in a separate numerical change with
  exactness, evaluation-order, compiler-output, and performance evidence.
- The packaged extension is the default and only normal load source. A developer
  override must be explicit and validate the complete build fingerprint. Never
  search for or silently load a global/stale extension.

## Change discipline

Before editing, identify the canonical domain owner and read its README plus the
relevant ADRs. Keep changes surgical and preserve existing semantics unless the
change is explicitly numerical.

For architecture moves:

- Move complete owners; do not opportunistically change physics, numerical
  order, launch configuration, synchronization, random-number consumption,
  result schema, metadata, or AD support.
- Preserve exact outputs where the frozen baseline requires exactness and keep
  performance/resource gates within their recorded budgets.
- New dependencies must pass `ci/check_import_graph.py`. Do not create, widen,
  relocate, or inherit allowlisted architecture debt.
- Public API changes require an intentional `ci/public-api-snapshot.json`
  update and migration note.
- Native binding changes require the binding manifest, contract-coverage
  manifest, owner inventory, negative no-fallback tests, direct contract tests,
  and end-to-end coverage to change together.
- A numerical or fusion-boundary change requires its own ADR and acceptance
  evidence; do not mix it into an architecture-cleanup commit.

Reject a review if it introduces production Torch/CPU computation, a fallback,
duplicate physics, solver-to-solver imports, raw extension access outside the
owning facade, an unowned ABI symbol, or any hot-path transfer/synchronization
outside the frozen ADR-032 compact owner and budget.

## Validation

Use the `witwin2` conda environment for every Python command, build, test, and
script. Do not use `witwin3` for this project.

Run the smallest relevant checks while developing, then the tier appropriate to
the change from the repository root:

```bash
conda run -n witwin2 python ci/run_ci_tier.py quick
conda run -n witwin2 python ci/run_ci_tier.py cuda
conda run -n witwin2 python ci/run_ci_tier.py nightly
conda run -n witwin2 python ci/run_ci_tier.py release
```

- Documentation-only changes: inspect links and run relevant static/quick gates.
- Python architecture or dependency changes: run `quick` plus targeted tests.
- Native, CUDA, AD, solver, or no-fallback changes: run `cuda` plus targeted
  lockstep/contract tests.
- Numerical, performance, packaging, or release-boundary changes: run the
  applicable `nightly`/`release` evidence in addition to lower tiers.

Never weaken a test, tolerance, manifest, allowlist, or maintenance budget just
to make a change pass.

## Authoritative records

These instructions summarize the accepted architecture. Detailed contracts and
acceptance evidence live in:

- `docs/dev/plans/08-channel-native-modular-architecture-hardening-plan.md`
- `docs/dev/standards/adr-001-python-native-dispatch.md`
- `docs/dev/standards/adr-003-public-internal-api.md`
- `docs/dev/standards/adr-004-numerical-duplication.md`
- `docs/dev/standards/adr-006-extension-developer-override.md`
- `docs/dev/standards/adr-007-propagation-data-ownership.md`
- `docs/dev/standards/adr-008-enumerated-propagation.md`
- `docs/dev/standards/adr-009-native-fusion-ownership.md`
- `docs/dev/standards/adr-010-native-scattering-kernels.md`
- `docs/dev/standards/adr-020-mc-transmission-polarization-unification.md`
- `docs/dev/standards/adr-021-multibounce-coherent-scattering.md`
- `docs/dev/standards/adr-022-bdpt-fixed-topology-ad.md`
- `docs/dev/standards/adr-023-direct-rayd-typed-integration.md`
- `docs/dev/standards/adr-033-channel-replacement-product-identity.md`
- `docs/dev/standards/adr-024-shared-rf-transmission-ownership.md`
- `docs/dev/standards/adr-025-diffraction-operation-family-ownership.md`
- `docs/dev/standards/adr-026-rayd-generic-scattering-runtime-ownership.md`
- `docs/dev/standards/adr-027-batched-segment-penetration.md`
- `docs/dev/standards/adr-028-device-resident-diffraction-state-selection.md`
- `docs/dev/standards/adr-029-device-resident-capacity-results.md`
- `docs/dev/standards/adr-030-deterministic-diffraction-pair-reduction.md`
- `docs/dev/standards/adr-031-per-pair-raw-reflection-epc-capacity.md`
- `docs/dev/standards/adr-032-controlled-compact-cardinality-boundary.md`

ADR-029 is superseded for production, ADR-031 is Proposed, and ADR-030 live
activation is deferred. They are retained for experimental contract audit;
ADR-032 is the authoritative production cardinality decision.

When detailed behavior is unclear, consult the accepted ADR and the owning
domain README. If an ADR and current implementation disagree, do not guess or
add a fallback; surface the mismatch and resolve it explicitly.
