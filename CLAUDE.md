# Channel contributor rules

These rules apply to the whole repository and override any monorepo-level guide. `AGENTS.md` and
`CLAUDE.md` must stay identical (`ci/check_repository_hygiene.py`). Rationale and history live in
the ADRs (indexed at the end) and `docs/dev/archive.md`.

## 1. Compute policy (non-negotiable)

`witwin.channel` has exactly one production compute backend: the compiled `_channel` CUDA
extension, which source-links the locked RayD.

- Every production hot path runs in native CUDA: geometry, RF/material/scattering physics,
  sampling, path evaluation, reductions, and production JVP/VJP/backward math.
- Torch is the tensor API and orchestration layer. A Torch CUDA expression is still a Torch
  implementation and is forbidden for hot-path physics or geometry. Do not recompute geometry
  that RayD owns.
- No CPU computation and no fallback. Missing CUDA, RayD capability, native symbol, supported SM
  or ABI-compatible extension fails loudly before any partial computation. Never recover from a
  native failure with Torch, NumPy, Python loops, CPU code, finite differences, a reduced
  algorithm, zero tensors, empty results or detached gradients.
- RayD `TraceBackend::Auto` choosing OptiX or pure CUDA (ADR-035) is not a fallback; an operation
  the selected backend cannot run fails its capability check before launching.
- Production AD calls registered native forward/JVP/VJP/backward companions. Torch autograd may
  dispatch them but never reconstructs the operation; finite-difference production derivatives
  are forbidden.
- Python may do non-numerical boundary work: validation, typed contracts, dispatch, row
  selection, structural packing, metadata, result assembly. CPU/Torch reference implementations
  live only under `tests/` as independent oracles; production code never imports them.
  Offline compile-time table construction in `scene.resources` (behind its own banner) is not a
  fallback and must not grow into per-solve physics.
- No exceptions without an accepted ADR; never hide one behind a flag, probe or shim.

## 2. Launch, sync and copy budgets (same rank as "no fallbacks")

Work the GPU does not need, and host round trips the solve does not need, are bugs.

- Every solve has a budget of kernel launches, host synchronizations, host-to-device and
  device-to-host copies and bytes, stated per scene size. Exceeding it fails review exactly like
  a CPU fallback. The budget tests under `tests/` enforce it; budgets only tighten, and only with
  measured evidence. Never raise a budget to make a change pass.
- Hot-path `.cpu()`, `.numpy()`, `.tolist()`, `.item()`, scalar extraction, host iteration,
  Boolean-mask indexing that hides a sync, and avoidable copies are forbidden.
- The only sanctioned count copy on the compact path is the compact-cardinality owner
  (ADR-032): it copies audited integer counts and synchronizes the caller's current stream once
  to allocate exact `O(K)` output in stable row order. Capacity overflow is always reported,
  never a silent truncation.
- A refactor must not add launches, synchronizations, materialized intermediates, persistent
  tape, host/device transfers or reduction-order changes. Do not split a fused native operation
  to mirror Python modules.
- `ci/native-inventory.json` lists every kernel, launch site and host-sync call per native file;
  a new launch or sync site is a visible one-line diff in review.

## 3. Every legal input leaves the CUDA context usable

- Any input that passes validation either completes or raises a Python exception, and the CUDA
  context stays usable afterwards: no device trap, device assert, illegal address or hang.
- Validate on the host, before the first launch, everything a kernel assumes: index ranges,
  finite coordinates, sizes, depths. Data-dependent failure is recorded in the solve's shared
  `CapacityFailureState`, makes outputs inert, and reaches the host once per solve (inside the
  compact count copy, or one read at finalization), where it raises `CapacityError` naming the
  failure bits; it never traps. A row that legitimately exceeds a requested bound (more wall
  crossings than `max_depth`, an uncertified coupled D-D root) is dropped and counted, not failed.
- No `trap`, `__trap`, `CUDA_KERNEL_ASSERT`, `assert` or `torch._assert_async` in a kernel path
  that validated input can reach.
- Test a new failure mode in a fresh subprocess, then check the CUDA context still works.

## 4. Ownership map

One owner per operation, organized by RF capability. A solver owner is named by its import path,
and that path is the contract; whether it is one module or a package is internal layout.

- `witwin.core` (external): logical `Scene`, `SceneSnapshot`, structures, world IDs, material
  specs, assignments, antennas, dynamics and the four version domains (ADR-034). Channel never
  re-exports it.
- `runtime` (one module, `runtime.py`): extension loading, ABI/build identity, symbol
  validation, native buffers, memory budgets, AD dispatch contracts, `require_tensor`, and the
  capacity failure state. It sits below `propagation` and never imports it.
- `scene`: compilation (`scene.compile(..., reference_frequency_hz=...)`), `CompiledScene`, the
  cache registry, RayD handles, endpoints and polarization tensors, diffraction edge policy, and
  compile-time Kirchhoff/phase-screen construction (`scene.resources`).
- `kernels`: the single home of every native facade, one module per domain (`fields`, `geometry`,
  `topology`, `montecarlo`, `scattering`, `materials`, `deterministic`). A facade validates,
  requests a symbol through `runtime`, dispatches, and converts to a typed result. It owns no
  physics and never imports a solver.
- `materials`: material ABI, per-face encoding, offline layer-stack evaluation.
- `interactions`: one module per RF interaction (`los`, `reflection`, `diffraction`,
  `transmission`, `scattering`, `coupled`), owning its discovery, geometry and orchestration. It
  calls `kernels`, publishes no barrel surface, never imports a solver, and stays acyclic.
- `propagation`: `topology` (discrete row machinery), `geometry` (continuous helpers), `fields`
  (field evaluation and derivative companions), `enumerated` (the shared Path/Deterministic
  engine), `rows` and `penetration` (typed internal contracts), and `consumer` (the stable
  solver-neutral public contract; ADR-037 to ADR-043). The consumer owns no physics, adds no
  second compaction, and never imports a solver.
- `path`, `deterministic`, `montecarlo.basic`, `montecarlo.bdpt`: thin solver configuration,
  orchestration, accumulation and results. A solver never imports another solver. BDPT may call
  the public `evaluate_enumerated_paths` read-only as an opaque oracle (ADR-008).
- Root modules: `constants`, `abi` (`Complex3State`, `JonesState`), `components`,
  `capabilities`, `deployment`.
- Public API: the package root (`build_info`, `capabilities`, `pipeline_cache_key`,
  `runtime_diagnostics`, `Complex3State`, `JonesState`), the four solver modules, and
  `propagation.consumer`, exactly as frozen in `ci/public-api-snapshot.json`. Everything else is
  internal: no compatibility shims, aliases or re-exports for moved internals.
- Owner documents live at `docs/dev/<domain>/README.md`.

## 5. Native boundary

- `_channel` is the only production Python extension. It calls RayD's versioned `rayd::torch`
  typed C++ API directly (ADR-023, ADR-047); no RayD Python extension, second dispatcher,
  copied `extern "C"` signatures or dynamic symbol lookup.
- RayD source discovery is explicit and fail-loud: `RAYD_SOURCE_DIR` or the validated
  `rayd-torch` source bundle, both checked against `dependencies/rayd.lock.json` (commit,
  integration-header digests, API version). Re-lock with the lock tool; never edit digests by hand.
- RayD scene ownership crosses the boundary as an RAII `SceneResource` in a typed holder; never
  an integer handle. Typed operations use the caller's current CUDA stream.
- Every pybind symbol is dispatched from exactly one Python module and has a production caller.
  `ci/check_native_bindings.py` derives both from the AST; tests do not count as callers.
- C++ bindings validate and dispatch; they contain no kernel launches and no host physics.
- Native ownership follows ABI operation, fusion/launch contract, tape lifetime and numerical
  order, not the Python layout. Per-file compile flags are part of the numerical contract
  (`--fmad=false` translation units; no `--use_fast_math` in Channel).
- The packaged extension is the only normal load source; a developer override is explicit and
  validates the full build fingerprint (ADR-006).

## 6. Source style

- Every tracked Python, C++, CUDA and header file starts with `Copyright Xingyu Chen.` and one
  plain sentence (under 100 characters) saying what it does. A Python module docstring, if any,
  is one sentence of at most 120 characters. `ci/check_source_headers.py` enforces this.
- Comments and docstrings describe current behaviour or a nearby reason. No ADR numbers, plans,
  phases, waves, migration history or audit labels in source comments, docstrings or
  user-visible error strings. Design detail goes in owner docs and ADRs.
- Python signatures use compact 100-column parameter packing (`python tools/compact_signatures.py`);
  Ruff is the linter, not the formatter. Native code uses the root `.clang-format`; format only
  what you touch.
- Ordinary native `Vec3`/`Complex`/`Complex3` math has one owner, `native/channel/kernels/math.cuh`;
  Python vector math comes from `witwin.core`. Numerically distinct variants stay separately
  named. `ci/check_shared_math.py` enforces this.
- Native file names are short owner nouns; a header stays separate only with several consumers
  or a shared device contract.
- Primal/JVP/VJP arithmetic kept in lockstep is not cleanup debt; merging it is a separate
  numerical change with exactness evidence (ADR-004). Total duplication stays under the ceiling
  in `ci/check_duplication.py`, which may only be lowered.

## 7. Change discipline

A contributor edits code. The checks that guard structure are generated from the source:

| Check | What it guards | When it changes |
|---|---|---|
| `ci/check_native_bindings.py` | binding signatures, one owner module, production caller | run `--write` after adding, removing or changing a binding |
| `ci/check_native_inventory.py` | kernels, launches, host syncs per file; TU layout | run `--write` after changing native sources |
| `ci/check_import_graph.py`, `tests/test_import_cycles.py` | layer rules, no new import cycle | never widen `ci/import_graph_allowlist.json` |
| `ci/check_orphan_modules.py` | every module reachable from the public entry points | - |
| `ci/check_single_definition.py`, `ci/check_shared_math.py` | one definition per protected concept | - |
| `ci/public-api-snapshot.json` | the public API | intentional update plus a migration note |
| `ci/check_duplication.py`, `ci/check_maintenance_budgets.py` | tighten-only budgets | lower only |

- Behaviour tests and the public API snapshot are never weakened: no deleted assertions, looser
  tolerances or skipped cases to make a change pass. A test that only pins implementation text,
  line numbers, hashes or counts may be replaced by an equivalent generated check; do not add new
  ones. Never commit hand-maintained ledgers of lines, hashes or counts.
- Every behaviour change gets a test that fails before and passes after; GPU tests skip cleanly
  without CUDA.
- Read the owner's README and relevant ADRs first. An architecture move moves complete owners and
  changes no physics, numerical order, launches, syncs, RNG use, result schema or AD support.
- A numerical or fusion-boundary change needs its own ADR with acceptance evidence and its own
  commit; regenerate bit-level baselines it legitimately moves and say so in the commit.
- If an ADR and the implementation disagree, surface the mismatch; never guess or add a fallback.

Reject production Torch/CPU computation, fallbacks, duplicate physics, solver-to-solver imports,
raw extension access outside a facade, unowned or uncalled bindings, and out-of-budget launches,
syncs or copies.

## 8. Validation

Use the `solvers` conda environment (Python 3.11, Torch 2.10 + CUDA 12.8). Build the extension
with `scripts/dev_build.sh` and load it with `scripts/dev_env.sh`, which sets the developer
override variables (`WITWIN_CHANNEL_DEVELOPER_OVERRIDE`, `WITWIN_CHANNEL_EXTENSION_PATH`,
`WITWIN_CHANNEL_EXPECTED_FINGERPRINT`); read their headers. Pytest never picks a build on its
own. Put the pinned Ruff and mypy on `PATH` for the quick tier. Run from the repository root:

```bash
python ci/run_ci_tier.py quick     # static checks, generated checks, CPU-only tests; no extension
python ci/run_ci_tier.py cuda      # + full unit/contract suite, solver smoke, no-fallback, AD core
python ci/run_ci_tier.py nightly   # + coverage, Munich parity, full AD, statistics, wheel smoke
python ci/run_ci_tier.py release   # + performance, memory, scaling and wheel release gates
```

Docs only: `quick`. Python architecture: `quick` plus targeted tests. Native, CUDA, AD, solver
or no-fallback changes: `cuda` plus targeted contract tests. Numerical, performance, packaging or
release-boundary changes: also the applicable `nightly`/`release` evidence.

## 9. Design records

Plans are in `docs/dev/plans/` (the modular architecture is plan `08-*`); the style guide is
`docs/dev/standards/source-code-style.md`. ADRs are `docs/dev/standards/adr-NNN-*.md`:

- Boundaries and API: 001 dispatch, 002 RayD lock, 003 public/internal API, 006 developer
  override, 033 product identity, 034 world/propagation boundary, 036 public surface, 047 RayD
  0.8 source boundary.
- Ownership: 007 propagation data, 008 enumerated oracle, 009 native fusion, 023 direct RayD
  integration, 024 shared RF/transmission, 025 diffraction family, 026 RayD scattering runtime,
  027 segment penetration, 044 native translation units.
- Device residency: 028 diffraction state selection, 032 compact cardinality (029, 030, 031 are
  Removed and superseded by 032), 035 trace backend selection.
- Physics and AD: 004 numerical duplication, 010 scattering kernels, 011/012/013 coupled and
  double diffraction, 014/015 scattering AD, 016 transition integral, 017 ISB taper, 018/019 BDPT
  diffraction and coherent combine, 020 MC transmission polarization, 021 multibounce
  scattering, 022 BDPT AD, 045 stationary diffraction amplitude, 046 coupled D-D domain.
- Consumer: 037 fixed topology and Jones, 038 forward-AD liveness, 039 source amplitude, 040
  world provenance, 041 slot batching and time-varying CIR, 042 wideband offsets, 043 AD
  capability matrix.
- Process: 005 Git history.
