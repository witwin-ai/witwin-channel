# Channel Architecture Guardrails

ADR-033 accepts the breaking replacement product identity `witwin.channel`,
the `witwin-channel` distribution, and the single `_channel` extension. The
checkout directory may retain its current name, but installed/runtime/build
identifiers must not retain the predecessor suffix or add a compatibility
alias. During the bounded migration, follow ADR-033 whenever an older name in
this file conflicts with that accepted target.

These repository instructions apply to every file under the checkout directory
`channel/` and
take precedence over the monorepo-level agent guide. Keep `AGENTS.md` and
`CLAUDE.md` identical. Architecture changes must update both files in the same
commit.

## Non-negotiable compute policy

`witwin.channel` has exactly one production compute backend: the
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
- Under ADR-035, RayD `TraceBackend::Auto` may select RayD-owned OptiX or
  RayD-owned pure-CUDA tracing during scene construction. OptiX is the
  preferred performance path; missing OptiX alone is not a missing Channel
  capability. This native implementation choice is not a Torch/CPU/Dr.Jit
  fallback, a second owner, a retry policy, or permission for reduced results.
  An operation unsupported by the selected RayD backend must fail its typed
  capability validation before that operation launches numerical work or
  exposes output.
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

- Under ADR-034, `witwin.core` is the sole owner of logical `Scene`,
  `SceneSnapshot`, `Structure`, stable world IDs, physical-material
  specifications, logical assignments, antenna state, dynamics, and the four
  topology/geometry/material/assignment version domains. Core world contracts
  never own frequency, native resources, stores, caches, compiled records,
  propagation results, solver tapes, or Radar RCS data.
- Channel is the sole owner of
  `scene.compile(..., reference_frequency_hz=...)`, `CompiledScene`, the
  Channel cache registry, RayD scene/BVH facade, GPU stores, material ABI and
  resident resources, propagation compute, compact assembly, and failure
  observation. A request/reference-frequency mismatch fails before native
  compute. Do not add an implicit recompile or host material replay.
- The Phase-2 Scene owner switch is intentionally breaking. Root logical world
  exports must resolve directly to `witwin.core`; delete the old Channel
  logical implementations and compatibility facades atomically with the four
  solver caller switch.
- The Phase-3 consumer reuses the ADR-032 native compact cardinality
  observation and publishes actual `K` rows with native-produced stable pair
  segmentation. It must not add a second count D2H/synchronization or use
  Torch/Python compaction. A public Jones result is a complete source-basis to
  sink-basis `2 x 2` transport operator, not a renamed two-component field.

- `runtime`: packaged extension loading, ABI/build identity, symbol validation,
  Torch compatibility isolation, native buffers, kernel metadata, memory
  budgets, AD dispatch contracts, and the capacity failure state, execution
  counts, and host-count guard that every capacity-shaped contract shares. It
  also owns `require_tensor`, the single dtype/shape/rank/device/CUDA/
  contiguity check that the row, capacity, and consumer contracts each apply to
  a declared tensor field; it sits here rather than beside the row contracts
  because `runtime` is below `propagation` and may never import it. It
  is one module, `runtime.py`, and every one of those names is imported from
  `witwin.channel.runtime` directly; there are no runtime submodules and no
  second spelling of an import. Its owner document is
  `docs/dev/runtime/README.md`, because a module has no directory to hold a
  README. No owner document lives inside the package tree at all: every domain,
  module or package alike, documents itself at `docs/dev/<domain>/README.md`,
  which is where the `materials`, `scattering`, `scene`, `propagation`,
  `propagation.consumer`, and `montecarlo` owner documents live as well. The
  packaged RayD identity lock and build-fingerprint sidecar sit beside the
  extension in `witwin/channel/`, not in a `runtime/` data directory.
- `scene`: scene lifecycle, compilation, immutable native resources, RayD
  handles, endpoint/antenna/receiver geometry, endpoint polarization tensors,
  diffraction edge policy and selection, the scene-leaf AD geometry seam, and
  the compile-time construction of the Kirchhoff table and phase-screen
  runtime those resources are made of. `scene.resources` is the single owner
  of that construction; it keeps the offline float64 NumPy build behind its own
  banner so the sanctioned CPU-compute island stays auditable, and nothing
  below that banner may grow into per-solve physics.
- `kernels`: the single home of every native facade, one package with one
  module per domain - `fields`, `geometry`, `topology`, `montecarlo`,
  `scattering`, `materials`, and `deterministic`. It sits above the RF domains,
  so a facade may import `runtime` and the shared row contracts but never a
  solver and never a domain that imports it back.
- `materials`: material ABI, per-face encoding, and offline layer-stack
  evaluation. Its native facade is `kernels.materials`.
- `scattering` is not a module. It is a subject with three owners, split on
  when the work runs, and the split is the point: `scene.resources` owns the
  compile-time Kirchhoff table and phase-screen construction, `kernels
  .scattering` owns the native facades, and `interactions.scattering` owns
  per-solve path evaluation. The root `scattering.py` that used to read like
  "the scattering owner" held only the compile-time half and was merged into
  `scene.resources`, which already cached it. Do not recreate it.
- `interactions`: one module per RF interaction concept - `los`, `reflection`,
  `diffraction`, `transmission`, `scattering`, `coupled`. A concept module owns
  its topology discovery, its path geometry, and its enumerated orchestration
  together, because those three were never an ownership boundary: they were one
  concept split across three stage packages. `interactions.transmission` and
  `interactions.scattering` also own the specular-transmission and Kirchhoff
  scattering event helpers both Monte Carlo solvers share; those helpers had a
  third, enumerated consumer, so they were never a Monte Carlo concept and
  `montecarlo.events` no longer exists. A concept module owns no native math -
  it calls the `kernels` facades - publishes no barrel surface, and must never
  import a solver. Cross-concept imports inside the package are allowed and
  must stay acyclic.
- `propagation.topology`: the stage-shared discrete row machinery - export,
  concatenation, and the IDs, winners, and interaction sequences it packs.
- `propagation.geometry`: the stage-shared continuous geometry helpers -
  endpoints, visibility, edge state, silhouette clearance, and reevaluation.
- `propagation.fields`: RF field evaluation and native derivative companions.
- `propagation.enumerated`: the concept-agnostic enumerated engine, its typed
  config protocols, and its capacity sanitizers - the shared deterministic path
  evaluation for the Path and Deterministic solvers. The per-concept discovery
  it drives lives in `interactions`. It is one module, `enumerated.py`. It
  declares two structural config views, and they are deliberately two objects:
  `TopologyConfig` is the larger view the enumerated scattering stages read and
  `interactions.scattering` imports by name, while `EnumeratedPathConfig` is the
  four-field view the engine itself reads. A Protocol is exactly its field set,
  so merging them would silently widen one of the two contracts.
- `propagation.rows`: the typed internal row contracts those stages exchange.
  One path table is four zero-copy views keyed on one opaque row-identity
  token, and they live in one module rather than one per stage because
  `propagation.topology` constructs all four together while the import
  graph forbids topology from reaching geometry or fields.
- `propagation.penetration`: the typed ADR-027 segment-penetration results.
  They sit beside the row contracts for the same reason: the component-5
  topology packer reads a `SegmentPenetrationResult`, so they cannot live
  under `propagation.geometry`.
- `propagation.consumer`: the stable solver-neutral public propagation
  contract, its vocabulary, and its capability record. It owns no physics, adds
  no second compaction, and must never import a solver. Under ADR-037 a frozen
  topology that carries interactions must be partitioned once by
  `prepare_fixed_topology` before reevaluation, and a reflection row that stops
  existing at new endpoint positions is published through the `row_valid` mask
  as a complete answer, never as a failure that voids the batch. The composed
  source-to-sink Jones operator is structural packing over already-owned native
  transport and basis operators; it must not restate a direction, a transverse
  basis, or a projection in Torch. Under ADR-039 the published scalar and
  complex3 transport carry the declared source amplitude `sqrt(powers_w)` and
  come from native excited outputs; the Jones operator stays excitation-free
  because it is a basis map. Never apply the amplitude in Torch. Under ADR-040
  a discovered topology carries the four `witwin.core` version domains it was
  discovered against, `prepare_fixed_topology` forwards that token verbatim,
  and `reevaluate` refuses a frozen replay against a moved world by name before
  any native work. A moved `topology_version`, `material_version`, or
  `assignment_version` is always fatal; a moved `geometry_version` is fatal
  unless the request declares `world_motion="fixed_winner_replay"`. The check
  is four host integer comparisons and must never grow a device read, a
  synchronization, or an `O(scene)` host walk in `_preflight_reevaluate`.
  `CompiledScene.time_s` is reporting metadata and is never a gate. Replay
  stays subtractive: a row can die through `row_valid`, a row is never born,
  and that limitation is documented rather than hidden. Under ADR-041 a
  request may declare `slot_count`, which states that the frozen rows and both
  endpoint batches are that many block-diagonal slots stacked slot-major and
  makes `pair_count` linear rather than quadratic in the slot count; a whole
  frame is then one launch per bucket, one four-byte validation copy, and one
  synchronization. `replicate_over_slots` is index arithmetic and bucket
  re-partitioning only, forwards `provenance` verbatim, adds no bucket, and
  requires the per-slot endpoint counts because an endpoint that publishes no
  row is invisible to a topology. `evaluate_time_varying` is the time axis over
  one such replay and nothing else: `[T, K]` views, no physics, no second
  compaction, no scene compilation, and `times_s` is a label that is never
  differenced into a rate. `slot_count > 1` requires a `PreparedFixedTopology`;
  the raw route's pairing law lives inside a native gather and refuses slot
  batching by name rather than growing a second Python owner of it. Under
  ADR-042 a fixed-topology request may declare `frequency_offsets_hz`, a host
  tuple of propagation-frequency offsets, and receive the same frozen rows
  evaluated at each absolute frequency as an additive `[K, F]` payload paired
  with the grid it was evaluated on. It is a host declaration, never a tensor
  and never differentiable, and it is a propagation-frequency grid only: it
  must never accept a subcarrier count, an FFT size, or a bandwidth. The row
  gather runs once above the column loop and stays the sole owner of the one
  validation copy and the one synchronization, whatever `F` is; the launch
  count is what grows, as `(1 + F) * buckets * launches_per_bucket`, and the
  ADR states that out loud. `[K, F]` assembly is `torch.stack` structural
  packing over native column outputs - no offset-dependent phase, magnitude, or
  basis may be applied in Torch, and `sqrt(powers_w)` stays with the native
  owner. `row_valid` stays `[K]` and geometry is published once from the
  reference column; a `0.0` offset must reproduce the reference coefficient
  bitwise. ADR-038 liveness is decided once, above the column loop, from the
  inputs every column shares, and the same explicit flag reaches every column.
  Dispersive scenes, rough or phase-screen scenes, and grids finer than
  `native_frequency_resolution_hz` are three independent fail-loud refusals
  before any native work, each individually reachable. Discovery has no
  frequency grid, frequency never becomes a fifth world version domain, and
  neither `replicate_over_slots` nor `evaluate_time_varying` gains a frequency
  variant. The wideband surface and the frozen-topology preparation helpers are
  their own sections of `consumer.py`, ahead of and beside the vocabulary
  section, which stays the single place a reader looks up a consumer type.
  Under
  ADR-043 the AD capability matrix is published, not inferred:
  `component_ad_modes` narrows `diffraction` to the primal,
  `component_material_leaves` and `differentiable_geometry_outputs` name the
  per-component and per-route derivative surface,
  `direction_differentiable_components` is `{los, reflection}` and
  `field_direction` liveness is ONE decision for the whole result rather than a
  per-row one, `primal_only_ad_inputs` is refused before any native work on
  every response and every route, `supports_higher_order_ad` is False and every
  second-order composition fails before a partial second-order result, and
  `PropagationDiagnostics` publishes the AD ledger with the reverse-only tape
  gate. RayD owns the transmission, wedge, and coupled direction seam; those
  cells stay declared non-differentiable with a named deferral and must never be
  reconstructed in Torch. The consumer diffraction primal defect is a recorded
  gap with a pinned regression test, not something to fix as a side effect.
- `path`, `deterministic`, `montecarlo.basic`, and `montecarlo.bdpt`: thin
  solver-owned configuration, orchestration, accumulation, result, and metadata
  layers. Solvers must never import another solver. A solver owner is named by
  its import path - `witwin.channel.path`, `witwin.channel.deterministic`,
  `witwin.channel.montecarlo.basic`, `witwin.channel.montecarlo.bdpt` - and
  that path is the contract. Whether an owner is one module or a package of
  private submodules is internal layout: collapsing one moves definition sites
  only, every public name keeps its import path, and the deleted submodules
  were never public API, so no alias or re-export replaces them. Every owner
  keeps its owner document outside the package under
  `docs/dev/<domain>/README.md`, which is why `path` and `deterministic`
  document themselves in `docs/dev/path/README.md` and
  `docs/dev/deterministic/README.md`.

Three package-root modules hold cross-domain values that the public root and
several domains all need, and that therefore cannot live under `runtime`,
`propagation`, or the `kernels` package without tripping the public-init
boundary:

- `constants`: electromagnetic constants and the package-wide phase convention.
  It is the single owner of the phasor, time-dependence, and narrowband
  frequency-offset strings that solver metadata and the consumer contract
  quote, including the quantified error law that states what the narrowband
  approximation costs.
- `abi`: the `Complex3State` / `JonesState` native field ABI contracts, and
  nothing else. It was called `field_state` and also held the scene-derived
  transmitter/receiver polarization tensors; those are endpoint geometry, not
  an ABI contract, so they moved to `scene.endpoints` and the root module was
  renamed for what remains. It depends on `torch` alone.
- `components`: cross-domain component identity.

There is no `tensor_math` module. It was a two-function grab-bag named after
neither of them: `require_tensor` is a contract validator and now lives with the
contracts' shared runtime in `runtime.py`, and `normalize_vec3` is plain vector
maths and now lives in `witwin.core.math`, exported from `witwin.core` beside
the quaternion helpers. Do not recreate it.

`capabilities` reports solver-level capability and embeds the consumer contract
record under `propagation_consumer` rather than restating it. `deployment` owns
package-level build and runtime reporting, including the public `build_info`.

There is no `core` or `physics` package. `core` was a grab-bag that collided
with `witwin.core` and has been dissolved into the owners above. `physics` held
a NumPy CPU reference oracle inside the shipped wheel; that oracle now lives in
`tests/reference/em_oracle.py`, where CLAUDE.md requires it. Do not recreate
either namespace.

The package root exports only what Channel owns: `build_info`, `capabilities`,
`pipeline_cache_key`, `runtime_diagnostics`, `Complex3State`, and `JonesState`.
The logical world model is owned by `witwin.core` and must be imported from
there. Channel does not re-export it, so each world type has exactly one import
path.

The only enumerated/Monte Carlo exception is ADR-008: the `montecarlo.bdpt`
pipeline may call the public `evaluate_enumerated_paths` entry read-only as an
opaque discrete-path oracle. It must not import `propagation.enumerated.*`
internals, mutate the result, or add BDPT policy to the enumerated engine.
`montecarlo.basic` has no enumerated dependency.

Internal enumerated propagation uses the typed, zero-copy contracts
`PathTopology`, `PathGeometry`, `PathFields`, and `EvaluatedPaths`. Preserve row
identity, row order, tensor object/storage aliasing, stride, dtype, device, and
gradient state. Do not reintroduce `TopologyBatch`, `core.path_topology`,
`core.kernels.ops`, raw native tuples outside the `kernels` facades, or
compatibility re-export layers.

The package root and the four solver entry points are the stable public API.
Internal modules are not compatibility promises. Do not preserve deleted APIs
or add backward-compatibility shims unless a new accepted design explicitly
requires them.

## Native boundary and fusion

- `_channel` is the only production Python extension. It source-links
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
- `witwin.channel.kernels` is the single home of every native facade. There are
  no per-domain `kernels/` packages: the one package holds one module per
  domain - `fields`, `geometry`, `topology`, `montecarlo`, `scattering`,
  `materials`, and `deterministic` - and each module is the single owner of its
  facades. Every facade stays thin: validate contracts, request a required
  symbol through `runtime`, dispatch the native operation, and convert its
  result to a named typed contract. A facade owns no physics, and importing one
  must not become a way to reach a domain package.
- C++ Torch bridges validate tensor/shape/dtype/device/ABI state and launch
  kernels. They must not contain a second host implementation of RF physics.
- Every supported native ABI symbol has one Python owner and must appear in
  `ci/native-binding-manifest.json` with direct contract coverage. A live symbol
  must have at least one production end-to-end caller. There is no caller-free
  native binding and no dormant-symbol allowlist: `ci/check_contract_coverage.py`
  keeps its dormant branch armed so a future caller-free binding needs a named,
  recorded decision, but the allowlist itself must stay empty.
- Native ownership follows ABI operation, fusion/launch contract, tape lifetime,
  device primitive, and numerical order—not the Python directory layout.
- Under ADR-025 and the completed Phase 8A atomic pin/switch/delete, RayD is
  the sole numerical owner of pure-wedge diffraction primal/backward/JVP;
  Channel retains only its `_channel` ABI and typed field/autograd
  facades. MC UTD and coupled RD/DD diffraction stay complete Channel
  owners; do not extract a UTD sub-launch or spread the pure-wedge fast-math
  flag into their precise-math translation units.
- Under ADR-026 and the completed Phase 10A/10B atomic pin/switch/delete, RayD
  is the sole numerical owner of all 17 generic resident scattering runtime
  contracts and the seven shared scattering-table helpers. Channel retains
  their `_channel` ABI and typed facades. RayD only consumes
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
- ADR-029, ADR-030, and ADR-031 are Removed. Their caller-free Python modules,
  native translation units, and 19 ABI symbols were deleted in the Phase-11
  cutover; only `evaluated_paths_capacity_pack_backward`/`_jvp` survive, because
  the live enumerated capacity failure sanitizer owns those AD companions. Do
  not reintroduce a capacity-shaped route: ADR-032's compact `O(K)` boundary is
  the only production cardinality contract.
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
- Every production module must stay reachable from the package root or one of
  the four solver entry points; `ci/check_orphan_modules.py` rejects the rest.
  A deleted owner does not come back as an unreachable file, and a module kept
  alive only by its own tests is an orphan.
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

- `docs/dev/plans/08-channel-modular-architecture-hardening-plan.md`
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
- `docs/dev/standards/adr-034-stage-i-world-and-propagation-boundary.md`
- `docs/dev/standards/adr-035-rayd-native-trace-backend-selection.md`
- `docs/dev/standards/adr-036-channel-public-surface-and-module-ownership.md`
- `docs/dev/standards/adr-037-fixed-topology-reflection-and-composed-jones.md`
- `docs/dev/standards/adr-038-wrapper-level-forward-ad-liveness.md`
- `docs/dev/standards/adr-039-consumer-source-amplitude.md`
- `docs/dev/standards/adr-040-world-provenance-and-fixed-topology-staleness.md`
- `docs/dev/standards/adr-041-slot-batched-reevaluation-and-time-varying-cir.md`
- `docs/dev/standards/adr-042-wideband-frequency-offsets.md`
- `docs/dev/standards/adr-043-propagation-ad-capability-matrix.md`

ADR-029, ADR-030, and ADR-031 are Removed (Phase-11 cutover). They are
historical records rather than implementation or release requirements; ADR-032
is the authoritative production cardinality decision.

When detailed behavior is unclear, consult the accepted ADR and the owning
domain README. If an ADR and current implementation disagree, do not guess or
add a fallback; surface the mismatch and resolve it explicitly.
## Source headers and shared math

Every tracked Python, C++, CUDA, and native header file starts with exactly two
plain-language comment lines: `Copyright Xingyu Chen.` followed by one concise
sentence saying what the file does. Keep that sentence under 100 characters.
Every source comment and docstring uses plain language to describe current
behavior, a current constraint, or the reason for nearby code. Do not put ADR
numbers, numbered plans, phases, waves, migration history, audit labels, or
changelog prose anywhere in source comments or docstrings. A Python module
docstring, when present, is one sentence no longer than 120 characters and does
not repeat an architecture narrative. Put durable design detail in the owner
documentation and local algorithm detail beside the code it explains. Frozen
identifiers and test data may retain required historical spellings, but prose
may not use them as explanations. Living documentation references canonical
files and symbol names instead of brittle source line numbers; refresh any
generated line-sensitive evidence in the same change.

`native/channel/kernels/math.cuh` is the single Channel owner of ordinary native
`Vec3`, `Complex`, and `Complex3` values and their basic load, store, arithmetic,
dot, cross, length, normalization, complex division, square-root, and power
operations. A translation unit must not redeclare those simple types or repeat
those helpers. Numerically distinct policies such as an epsilon floor, fallback
vector, zero fallback, explicit round-to-nearest order, or `rsqrt` floor stay as
separately named functions in that shared header; a cleanup must not erase the
difference or reorder arithmetic. Algorithm-specific derivative or tape state,
such as `DualVec3`, remains with its algorithm when it is not a general-purpose
math type.

Generic Python vector and quaternion math comes from `witwin.core.math`, as
exported by `witwin.core`; Channel must not create `tensor_math.py` or another
Python owner. Domain-specific offline material math stays with its domain owner.
Independent test-reference math remains under `tests/` and must never become a
production backend or import target.