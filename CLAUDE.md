# Channel Native Architecture Guardrails

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
- Device data must remain resident through the compute pipeline. Do not add
  hot-path `.cpu()`, `.numpy()`, `.tolist()`, scalar extraction, host iteration,
  implicit synchronization, or avoidable host/device copies.

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
  `ci/native-binding-manifest.json` with direct contract coverage and at least
  one end-to-end caller.
- Native ownership follows ABI operation, fusion/launch contract, tape lifetime,
  device primitive, and numerical order—not the Python directory layout.
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
owning facade, an unowned ABI symbol, or a new hot-path transfer/synchronization.

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
- `docs/dev/standards/adr-024-shared-rf-transmission-ownership.md`

When detailed behavior is unclear, consult the accepted ADR and the owning
domain README. If an ADR and current implementation disagree, do not guess or
add a fallback; surface the mismatch and resolve it explicitly.
